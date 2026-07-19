/**
 * 增强的 MCP 客户端实现
 *
 * 支持多种传输方式的 MCP 客户端，用于教学和实际应用。当前仓库没有引入
 * `@modelcontextprotocol/sdk`，因此这里实现一个轻量 JSON-RPC MCP 客户端：
 * - Memory: 直接调用内存对象上的 MCP 风格方法
 * - Stdio: 启动本地进程，通过 stdin/stdout 发送 JSON-RPC
 * - HTTP: 通过 POST 请求发送 JSON-RPC
 * - SSE: 通过 Event Stream 接收响应，通过 POST 发送请求
 */

import { spawn, type ChildProcessWithoutNullStreams } from "child_process";

type MCPTransportType = "memory" | "stdio" | "http" | "sse";
type JsonRpcId = number | string;

export type MCPServerSource =
    | string
    | string[]
    | MCPTransportConfig
    | MemoryMCPServer;

export interface MCPTransportConfig {
    transport?: MCPTransportType;
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    cwd?: string;
    url?: string;
    headers?: Record<string, string>;
    auth?: string;
    timeoutMs?: number;
    [key: string]: unknown;
}

export interface MCPClientOptions {
    serverArgs?: string[];
    transportType?: MCPTransportType;
    env?: Record<string, string>;
    transportOptions?: MCPTransportConfig;
}

export interface MCPToolInfo {
    name: string;
    description: string;
    input_schema: Record<string, unknown>;
}

export interface MCPResourceInfo {
    uri: string;
    name: string;
    description: string;
    mime_type?: string | null;
}

export interface MCPPromptInfo {
    name: string;
    description: string;
    arguments: unknown[];
}

export interface MCPPromptMessage {
    role: string;
    content: unknown;
}

export interface MCPTransportInfo {
    status: "not_connected" | "connected" | "unknown";
    transport_type?: string;
    transport_info?: string;
}

interface JsonRpcRequest {
    jsonrpc: "2.0";
    id: JsonRpcId;
    method: string;
    params?: Record<string, unknown>;
}

interface JsonRpcNotification {
    jsonrpc: "2.0";
    method: string;
    params?: Record<string, unknown>;
}

interface JsonRpcResponse {
    jsonrpc?: "2.0";
    id?: JsonRpcId;
    result?: unknown;
    error?: {
        code?: number;
        message?: string;
        data?: unknown;
    };
}

interface MCPTransport {
    readonly type: MCPTransportType;
    connect(): Promise<void>;
    close(): Promise<void>;
    request(method: string, params?: Record<string, unknown>): Promise<unknown>;
    notify?(method: string, params?: Record<string, unknown>): Promise<void>;
    info(): string;
}

export interface MemoryMCPServer {
    name?: string;
    request?: (method: string, params?: Record<string, unknown>) => unknown | Promise<unknown>;
    listTools?: () => unknown | Promise<unknown>;
    list_tools?: () => unknown | Promise<unknown>;
    callTool?: (name: string, args: Record<string, unknown>) => unknown | Promise<unknown>;
    call_tool?: (name: string, args: Record<string, unknown>) => unknown | Promise<unknown>;
    listResources?: () => unknown | Promise<unknown>;
    list_resources?: () => unknown | Promise<unknown>;
    readResource?: (uri: string) => unknown | Promise<unknown>;
    read_resource?: (uri: string) => unknown | Promise<unknown>;
    listPrompts?: () => unknown | Promise<unknown>;
    list_prompts?: () => unknown | Promise<unknown>;
    getPrompt?: (name: string, args: Record<string, string>) => unknown | Promise<unknown>;
    get_prompt?: (name: string, args: Record<string, string>) => unknown | Promise<unknown>;
    ping?: () => unknown | Promise<unknown>;
}

function isHttpUrl(value: string): boolean {
    return value.startsWith("http://") || value.startsWith("https://");
}

function isPythonScript(value: string): boolean {
    return value.endsWith(".py");
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === "object" ? value as Record<string, unknown> : {};
}

function arrayFromResult(result: unknown, key: string): unknown[] {
    if (Array.isArray(result)) return result;
    const record = asRecord(result);
    const value = record[key];
    return Array.isArray(value) ? value : [];
}

function contentValue(content: unknown): unknown {
    const record = asRecord(content);
    if ("text" in record) return record.text;
    if ("data" in record) return record.data;
    if ("blob" in record) return record.blob;
    return content;
}

function authHeaders(auth?: string): Record<string, string> {
    return auth ? { Authorization: auth } : {};
}

function mergeEnv(base: NodeJS.ProcessEnv, extra?: Record<string, string>): NodeJS.ProcessEnv {
    return { ...base, ...(extra ?? {}) };
}

function parseJsonMessage(raw: string): JsonRpcResponse | JsonRpcRequest | JsonRpcNotification | null {
    const trimmed = raw.trim();
    if (!trimmed) return null;
    try {
        return JSON.parse(trimmed) as JsonRpcResponse | JsonRpcRequest | JsonRpcNotification;
    } catch {
        return null;
    }
}

function extractContentLengthMessage(buffer: string): { message: string | null; rest: string } {
    const headerEnd = buffer.indexOf("\r\n\r\n");
    if (headerEnd < 0) return { message: null, rest: buffer };

    const headerText = buffer.slice(0, headerEnd);
    const match = headerText.match(/content-length:\s*(\d+)/i);
    if (!match) return { message: null, rest: buffer };

    const length = Number(match[1]);
    const bodyStart = headerEnd + 4;
    const bodyEnd = bodyStart + length;
    if (buffer.length < bodyEnd) return { message: null, rest: buffer };

    return {
        message: buffer.slice(bodyStart, bodyEnd),
        rest: buffer.slice(bodyEnd),
    };
}

async function responseToJson(response: Response): Promise<unknown> {
    const contentType = response.headers.get("content-type") ?? "";
    if (contentType.includes("text/event-stream")) {
        return readFirstSseJson(await response.text());
    }
    return response.json();
}

function readFirstSseJson(text: string): unknown {
    const dataLines: string[] = [];
    for (const line of text.split(/\r?\n/)) {
        if (line.startsWith("data:")) {
            dataLines.push(line.slice("data:".length).trim());
        }
    }
    const data = dataLines.join("\n").trim();
    if (!data) return null;
    return JSON.parse(data);
}

function normalizeJsonRpcResponse(response: unknown): unknown {
    const record = asRecord(response);
    if (record.error) {
        const error = asRecord(record.error);
        throw new Error(String(error.message ?? "MCP JSON-RPC error"));
    }
    return "result" in record ? record.result : response;
}

class MemoryTransport implements MCPTransport {
    readonly type = "memory" as const;

    constructor(private readonly server: MemoryMCPServer) {}

    async connect(): Promise<void> {}

    async close(): Promise<void> {}

    async request(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
        if (this.server.request) {
            return this.server.request(method, params);
        }

        switch (method) {
            case "initialize":
                return {
                    protocolVersion: "2024-11-05",
                    serverInfo: { name: this.server.name ?? "MemoryMCPServer", version: "memory" },
                    capabilities: {},
                };
            case "ping":
                return {};
            case "tools/list":
                return this.callAny(["listTools", "list_tools"], []);
            case "tools/call":
                return this.callAny(["callTool", "call_tool"], [
                    String(params.name ?? ""),
                    asRecord(params.arguments),
                ]);
            case "resources/list":
                return this.callAny(["listResources", "list_resources"], []);
            case "resources/read":
                return this.callAny(["readResource", "read_resource"], [String(params.uri ?? "")]);
            case "prompts/list":
                return this.callAny(["listPrompts", "list_prompts"], []);
            case "prompts/get":
                return this.callAny(["getPrompt", "get_prompt"], [
                    String(params.name ?? ""),
                    asRecord(params.arguments) as Record<string, string>,
                ]);
            default:
                throw new Error(`Memory server does not implement method: ${method}`);
        }
    }

    info(): string {
        return `MemoryTransport(${this.server.name ?? "anonymous"})`;
    }

    private async callAny(methodNames: string[], args: unknown[]): Promise<unknown> {
        const server = this.server as Record<string, unknown>;
        for (const methodName of methodNames) {
            const candidate = server[methodName];
            if (typeof candidate === "function") {
                return candidate.apply(this.server, args);
            }
        }
        throw new Error(`Memory server missing method: ${methodNames.join(" or ")}`);
    }
}

class StdioTransport implements MCPTransport {
    readonly type = "stdio" as const;

    private process: ChildProcessWithoutNullStreams | null = null;
    private nextId = 1;
    private stdoutBuffer = "";
    private readonly pending = new Map<JsonRpcId, {
        resolve: (value: unknown) => void;
        reject: (error: Error) => void;
        timer: NodeJS.Timeout;
    }>();

    constructor(
        private readonly command: string,
        private readonly args: string[] = [],
        private readonly options: {
            env?: Record<string, string>;
            cwd?: string;
            timeoutMs?: number;
        } = {},
    ) {}

    async connect(): Promise<void> {
        if (this.process) return;

        this.process = spawn(this.command, this.args, {
            cwd: this.options.cwd,
            env: mergeEnv(process.env, this.options.env),
            stdio: ["pipe", "pipe", "pipe"],
        });

        this.process.stdout.on("data", (chunk: Buffer) => {
            this.handleStdout(chunk.toString("utf-8"));
        });

        this.process.stderr.on("data", (chunk: Buffer) => {
            const text = chunk.toString("utf-8").trim();
            if (text) console.warn(`[MCP stdio stderr] ${text}`);
        });

        this.process.on("exit", (code, signal) => {
            const error = new Error(`MCP stdio process exited: code=${code}, signal=${signal}`);
            for (const pending of this.pending.values()) {
                clearTimeout(pending.timer);
                pending.reject(error);
            }
            this.pending.clear();
            this.process = null;
        });
    }

    async close(): Promise<void> {
        if (!this.process) return;
        this.process.kill();
        this.process = null;
    }

    async request(method: string, params?: Record<string, unknown>): Promise<unknown> {
        if (!this.process) throw new Error("Stdio transport is not connected");

        const id = this.nextId;
        this.nextId += 1;

        const request: JsonRpcRequest = { jsonrpc: "2.0", id, method };
        if (params) request.params = params;

        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error(`MCP request timed out: ${method}`));
            }, this.options.timeoutMs ?? 30_000);

            this.pending.set(id, { resolve, reject, timer });
            this.process?.stdin.write(`${JSON.stringify(request)}\n`);
        });
    }

    async notify(method: string, params?: Record<string, unknown>): Promise<void> {
        if (!this.process) throw new Error("Stdio transport is not connected");
        const notification: JsonRpcNotification = { jsonrpc: "2.0", method };
        if (params) notification.params = params;
        this.process.stdin.write(`${JSON.stringify(notification)}\n`);
    }

    info(): string {
        return `StdioTransport(${[this.command, ...this.args].join(" ")})`;
    }

    private handleStdout(chunk: string): void {
        this.stdoutBuffer += chunk;

        while (this.stdoutBuffer.length > 0) {
            const framed = extractContentLengthMessage(this.stdoutBuffer);
            if (framed.message) {
                this.stdoutBuffer = framed.rest;
                this.handleMessage(framed.message);
                continue;
            }

            const newlineIndex = this.stdoutBuffer.indexOf("\n");
            if (newlineIndex < 0) break;

            const line = this.stdoutBuffer.slice(0, newlineIndex);
            this.stdoutBuffer = this.stdoutBuffer.slice(newlineIndex + 1);
            this.handleMessage(line);
        }
    }

    private handleMessage(raw: string): void {
        const message = parseJsonMessage(raw);
        if (!message || !("id" in message)) return;

        const response = message as JsonRpcResponse;
        if (response.id === undefined) return;

        const pending = this.pending.get(response.id);
        if (!pending) return;

        clearTimeout(pending.timer);
        this.pending.delete(response.id);

        if (response.error) {
            pending.reject(new Error(response.error.message ?? "MCP JSON-RPC error"));
        } else {
            pending.resolve(response.result);
        }
    }
}

class HttpTransport implements MCPTransport {
    readonly type = "http" as const;
    private nextId = 1;

    constructor(
        private readonly url: string,
        private readonly options: {
            headers?: Record<string, string>;
            auth?: string;
            timeoutMs?: number;
        } = {},
    ) {}

    async connect(): Promise<void> {}

    async close(): Promise<void> {}

    async request(method: string, params?: Record<string, unknown>): Promise<unknown> {
        const id = this.nextId;
        this.nextId += 1;

        const request: JsonRpcRequest = { jsonrpc: "2.0", id, method };
        if (params) request.params = params;

        const response = await fetchWithTimeout(this.url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json, text/event-stream",
                ...authHeaders(this.options.auth),
                ...(this.options.headers ?? {}),
            },
            body: JSON.stringify(request),
        }, this.options.timeoutMs);

        if (!response.ok) {
            throw new Error(`HTTP MCP request failed: ${response.status} ${response.statusText}`);
        }

        return normalizeJsonRpcResponse(await responseToJson(response));
    }

    async notify(method: string, params?: Record<string, unknown>): Promise<void> {
        const notification: JsonRpcNotification = { jsonrpc: "2.0", method };
        if (params) notification.params = params;

        await fetchWithTimeout(this.url, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
                ...authHeaders(this.options.auth),
                ...(this.options.headers ?? {}),
            },
            body: JSON.stringify(notification),
        }, this.options.timeoutMs);
    }

    info(): string {
        return `HttpTransport(${this.url})`;
    }
}

class SseTransport implements MCPTransport {
    readonly type = "sse" as const;

    private nextId = 1;
    private postUrl: string;
    private abortController: AbortController | null = null;
    private readonly pending = new Map<JsonRpcId, {
        resolve: (value: unknown) => void;
        reject: (error: Error) => void;
        timer: NodeJS.Timeout;
    }>();

    constructor(
        private readonly url: string,
        private readonly options: {
            headers?: Record<string, string>;
            auth?: string;
            timeoutMs?: number;
        } = {},
    ) {
        this.postUrl = url;
    }

    async connect(): Promise<void> {
        this.abortController = new AbortController();

        fetch(this.url, {
            method: "GET",
            headers: {
                Accept: "text/event-stream",
                ...authHeaders(this.options.auth),
                ...(this.options.headers ?? {}),
            },
            signal: this.abortController.signal,
        })
            .then((response) => {
                if (!response.ok || !response.body) {
                    throw new Error(`SSE connection failed: ${response.status} ${response.statusText}`);
                }
                return this.readEventStream(response.body);
            })
            .catch((error) => {
                if (this.abortController?.signal.aborted) return;
                for (const pending of this.pending.values()) {
                    clearTimeout(pending.timer);
                    pending.reject(error instanceof Error ? error : new Error(String(error)));
                }
                this.pending.clear();
            });
    }

    async close(): Promise<void> {
        this.abortController?.abort();
        this.abortController = null;
        for (const pending of this.pending.values()) {
            clearTimeout(pending.timer);
            pending.reject(new Error("SSE transport closed"));
        }
        this.pending.clear();
    }

    async request(method: string, params?: Record<string, unknown>): Promise<unknown> {
        const id = this.nextId;
        this.nextId += 1;

        const request: JsonRpcRequest = { jsonrpc: "2.0", id, method };
        if (params) request.params = params;

        return new Promise((resolve, reject) => {
            const timer = setTimeout(() => {
                this.pending.delete(id);
                reject(new Error(`MCP request timed out: ${method}`));
            }, this.options.timeoutMs ?? 30_000);

            this.pending.set(id, { resolve, reject, timer });

            fetchWithTimeout(this.postUrl, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                    Accept: "application/json, text/event-stream",
                    ...authHeaders(this.options.auth),
                    ...(this.options.headers ?? {}),
                },
                body: JSON.stringify(request),
            }, this.options.timeoutMs)
                .then(async (response) => {
                    if (!response.ok) {
                        throw new Error(`SSE MCP request failed: ${response.status} ${response.statusText}`);
                    }

                    const contentType = response.headers.get("content-type") ?? "";
                    if (!contentType.includes("text/event-stream")) {
                        clearTimeout(timer);
                        this.pending.delete(id);
                        resolve(normalizeJsonRpcResponse(await responseToJson(response)));
                    }
                })
                .catch((error) => {
                    clearTimeout(timer);
                    this.pending.delete(id);
                    reject(error instanceof Error ? error : new Error(String(error)));
                });
        });
    }

    async notify(method: string, params?: Record<string, unknown>): Promise<void> {
        const notification: JsonRpcNotification = { jsonrpc: "2.0", method };
        if (params) notification.params = params;

        await fetchWithTimeout(this.postUrl, {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
                Accept: "application/json",
                ...authHeaders(this.options.auth),
                ...(this.options.headers ?? {}),
            },
            body: JSON.stringify(notification),
        }, this.options.timeoutMs);
    }

    info(): string {
        return `SseTransport(${this.url})`;
    }

    private async readEventStream(body: ReadableStream<Uint8Array>): Promise<void> {
        const reader = body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            let separatorIndex = buffer.search(/\r?\n\r?\n/);

            while (separatorIndex >= 0) {
                const eventText = buffer.slice(0, separatorIndex);
                buffer = buffer.slice(separatorIndex + (buffer[separatorIndex] === "\r" ? 4 : 2));
                this.handleSseEvent(eventText);
                separatorIndex = buffer.search(/\r?\n\r?\n/);
            }
        }
    }

    private handleSseEvent(eventText: string): void {
        let eventName = "message";
        const dataLines: string[] = [];

        for (const line of eventText.split(/\r?\n/)) {
            if (line.startsWith("event:")) eventName = line.slice("event:".length).trim();
            if (line.startsWith("data:")) dataLines.push(line.slice("data:".length).trim());
        }

        const data = dataLines.join("\n").trim();
        if (!data) return;

        if (eventName === "endpoint") {
            this.postUrl = new URL(data, this.url).toString();
            return;
        }

        const message = parseJsonMessage(data);
        if (!message || !("id" in message)) return;
        const response = message as JsonRpcResponse;
        if (response.id === undefined) return;

        const pending = this.pending.get(response.id);
        if (!pending) return;

        clearTimeout(pending.timer);
        this.pending.delete(response.id);

        if (response.error) {
            pending.reject(new Error(response.error.message ?? "MCP JSON-RPC error"));
        } else {
            pending.resolve(response.result);
        }
    }
}

async function fetchWithTimeout(
    input: RequestInfo | URL,
    init: RequestInit,
    timeoutMs = 30_000,
): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
        return await fetch(input, { ...init, signal: controller.signal });
    } finally {
        clearTimeout(timer);
    }
}

/**
 * MCP 客户端，支持 Memory / Stdio / HTTP / SSE 多种传输方式。
 */
export class MCPClient {
    private readonly serverArgs: string[];
    private readonly transportType?: MCPTransportType;
    private readonly env: Record<string, string>;
    private readonly transportOptions: MCPTransportConfig;
    private readonly transport: MCPTransport;
    private connected = false;

    constructor(
        serverSource: MCPServerSource,
        serverArgs?: string[],
        transportType?: MCPTransportType,
        env?: Record<string, string>,
        transportOptions?: MCPTransportConfig,
    );
    constructor(serverSource: MCPServerSource, options?: MCPClientOptions);
    constructor(
        serverSource: MCPServerSource,
        serverArgsOrOptions: string[] | MCPClientOptions = [],
        transportType?: MCPTransportType,
        env: Record<string, string> = {},
        transportOptions: MCPTransportConfig = {},
    ) {
        const options = Array.isArray(serverArgsOrOptions)
            ? { serverArgs: serverArgsOrOptions, transportType, env, transportOptions }
            : serverArgsOrOptions;

        this.serverArgs = options.serverArgs ?? [];
        this.transportType = options.transportType;
        this.env = options.env ?? {};
        this.transportOptions = options.transportOptions ?? {};
        this.transport = this.prepareServerSource(serverSource);
    }

    async connect(): Promise<this> {
        console.log("🔗 连接到 MCP 服务器...");
        await this.transport.connect();

        try {
            await this.transport.request("initialize", {
                protocolVersion: "2024-11-05",
                capabilities: {},
                clientInfo: {
                    name: "hello-agents-mcp-client",
                    version: "1.0.0",
                },
            });
            await this.transport.notify?.("notifications/initialized");
        } catch (error) {
            // 内存教学对象可能不实现 initialize；除 stdio/http/sse 外不强制失败。
            if (this.transport.type !== "memory") {
                throw error;
            }
        }

        this.connected = true;
        console.log("✅ 连接成功！");
        return this;
    }

    async disconnect(): Promise<void> {
        await this.transport.close();
        this.connected = false;
        console.log("🔌 连接已断开");
    }

    async withConnection<T>(callback: (client: this) => Promise<T>): Promise<T> {
        await this.connect();
        try {
            return await callback(this);
        } finally {
            await this.disconnect();
        }
    }

    async __aenter__(): Promise<this> {
        return this.connect();
    }

    async __aexit__(_excType?: unknown, _excVal?: unknown, _excTb?: unknown): Promise<void> {
        await this.disconnect();
    }

    async listTools(): Promise<MCPToolInfo[]> {
        this.assertConnected();
        const result = await this.transport.request("tools/list");
        const tools = arrayFromResult(result, "tools");

        return tools.map((tool) => {
            const record = asRecord(tool);
            return {
                name: String(record.name ?? ""),
                description: String(record.description ?? ""),
                input_schema: asRecord(record.inputSchema ?? record.input_schema),
            };
        });
    }

    async callTool(toolName: string, arguments_: Record<string, unknown> = {}): Promise<unknown> {
        this.assertConnected();
        const result = await this.transport.request("tools/call", {
            name: toolName,
            arguments: arguments_,
        });

        const record = asRecord(result);
        const content = record.content;
        if (Array.isArray(content) && content.length > 0) {
            if (content.length === 1) return contentValue(content[0]);
            return content.map(contentValue);
        }

        return "content" in record ? null : result;
    }

    async listResources(): Promise<MCPResourceInfo[]> {
        this.assertConnected();
        const result = await this.transport.request("resources/list");
        const resources = arrayFromResult(result, "resources");

        return resources.map((resource) => {
            const record = asRecord(resource);
            return {
                uri: String(record.uri ?? ""),
                name: String(record.name ?? ""),
                description: String(record.description ?? ""),
                mime_type: (record.mimeType as string | undefined)
                    ?? (record.mime_type as string | undefined)
                    ?? null,
            };
        });
    }

    async readResource(uri: string): Promise<unknown> {
        this.assertConnected();
        const result = await this.transport.request("resources/read", { uri });

        const record = asRecord(result);
        const contents = record.contents;
        if (Array.isArray(contents) && contents.length > 0) {
            if (contents.length === 1) return contentValue(contents[0]);
            return contents.map(contentValue);
        }

        return "contents" in record ? null : result;
    }

    async listPrompts(): Promise<MCPPromptInfo[]> {
        this.assertConnected();
        const result = await this.transport.request("prompts/list");
        const prompts = arrayFromResult(result, "prompts");

        return prompts.map((prompt) => {
            const record = asRecord(prompt);
            return {
                name: String(record.name ?? ""),
                description: String(record.description ?? ""),
                arguments: Array.isArray(record.arguments) ? record.arguments : [],
            };
        });
    }

    async getPrompt(promptName: string, arguments_: Record<string, string> = {}): Promise<MCPPromptMessage[]> {
        this.assertConnected();
        const result = await this.transport.request("prompts/get", {
            name: promptName,
            arguments: arguments_,
        });

        const messages = arrayFromResult(result, "messages");
        return messages.map((message) => {
            const record = asRecord(message);
            return {
                role: String(record.role ?? ""),
                content: contentValue(record.content),
            };
        });
    }

    async ping(): Promise<boolean> {
        this.assertConnected();
        try {
            await this.transport.request("ping");
            return true;
        } catch {
            return false;
        }
    }

    getTransportInfo(): MCPTransportInfo {
        if (!this.connected) return { status: "not_connected" };
        return {
            status: "connected",
            transport_type: this.transport.constructor.name,
            transport_info: this.transport.info(),
        };
    }

    // Python 风格别名，便于课程材料一一对应。
    async list_tools(): Promise<MCPToolInfo[]> {
        return this.listTools();
    }

    async call_tool(toolName: string, arguments_: Record<string, unknown> = {}): Promise<unknown> {
        return this.callTool(toolName, arguments_);
    }

    async list_resources(): Promise<MCPResourceInfo[]> {
        return this.listResources();
    }

    async read_resource(uri: string): Promise<unknown> {
        return this.readResource(uri);
    }

    async list_prompts(): Promise<MCPPromptInfo[]> {
        return this.listPrompts();
    }

    async get_prompt(promptName: string, arguments_: Record<string, string> = {}): Promise<MCPPromptMessage[]> {
        return this.getPrompt(promptName, arguments_);
    }

    get_transport_info(): MCPTransportInfo {
        return this.getTransportInfo();
    }

    private prepareServerSource(serverSource: MCPServerSource): MCPTransport {
        if (typeof serverSource === "object" && !Array.isArray(serverSource) && !("transport" in serverSource)) {
            console.log(`🧠 使用内存传输: ${(serverSource as MemoryMCPServer).name ?? "MemoryMCPServer"}`);
            return new MemoryTransport(serverSource as MemoryMCPServer);
        }

        if (typeof serverSource === "object" && !Array.isArray(serverSource) && "transport" in serverSource) {
            console.log(`⚙️ 使用配置传输: ${String(serverSource.transport ?? "stdio")}`);
            return this.createTransportFromConfig(serverSource as MCPTransportConfig);
        }

        if (typeof serverSource === "string" && isHttpUrl(serverSource)) {
            const selectedTransport = this.transportType ?? "http";
            console.log(`🌐 使用 ${selectedTransport.toUpperCase()} 传输: ${serverSource}`);
            return selectedTransport === "sse"
                ? new SseTransport(serverSource, this.transportOptions)
                : new HttpTransport(serverSource, this.transportOptions);
        }

        if (typeof serverSource === "string" && isPythonScript(serverSource)) {
            const command = process.env.PYTHON ?? "python";
            console.log(`🐍 使用 Stdio 传输 (Python): ${serverSource}`);
            return new StdioTransport(command, [serverSource, ...this.serverArgs], {
                env: this.env,
                timeoutMs: this.transportOptions.timeoutMs as number | undefined,
            });
        }

        if (Array.isArray(serverSource) && serverSource.length >= 1) {
            console.log(`📝 使用 Stdio 传输 (命令): ${serverSource.join(" ")}`);
            return new StdioTransport(serverSource[0], [...serverSource.slice(1), ...this.serverArgs], {
                env: this.env,
                timeoutMs: this.transportOptions.timeoutMs as number | undefined,
            });
        }

        if (typeof serverSource === "string") {
            console.log(`🔍 自动推断 Stdio 传输: ${serverSource}`);
            return new StdioTransport(serverSource, this.serverArgs, {
                env: this.env,
                timeoutMs: this.transportOptions.timeoutMs as number | undefined,
            });
        }

        throw new Error("Unsupported MCP server source");
    }

    private createTransportFromConfig(config: MCPTransportConfig): MCPTransport {
        const selectedTransport = config.transport ?? "stdio";
        const timeoutMs = config.timeoutMs ?? this.transportOptions.timeoutMs as number | undefined;

        if (selectedTransport === "stdio") {
            const args = config.args ?? [];

            if (args[0] && isPythonScript(args[0])) {
                return new StdioTransport(process.env.PYTHON ?? "python", [args[0], ...args.slice(1), ...this.serverArgs], {
                    env: config.env ?? this.env,
                    cwd: config.cwd,
                    timeoutMs,
                });
            }

            return new StdioTransport(config.command ?? "python", [...args, ...this.serverArgs], {
                env: config.env ?? this.env,
                cwd: config.cwd,
                timeoutMs,
            });
        }

        if (selectedTransport === "sse") {
            if (!config.url) throw new Error("SSE transport requires config.url");
            return new SseTransport(config.url, {
                headers: config.headers,
                auth: config.auth,
                timeoutMs,
            });
        }

        if (selectedTransport === "http") {
            if (!config.url) throw new Error("HTTP transport requires config.url");
            return new HttpTransport(config.url, {
                headers: config.headers,
                auth: config.auth,
                timeoutMs,
            });
        }

        if (selectedTransport === "memory") {
            const server = config.server as MemoryMCPServer | undefined;
            if (!server) throw new Error("Memory transport requires config.server");
            return new MemoryTransport(server);
        }

        throw new Error(`Unsupported transport type: ${String(selectedTransport)}`);
    }

    private assertConnected(): void {
        if (!this.connected) {
            throw new Error("Client not connected. Use connect()/disconnect() or withConnection().");
        }
    }
}
