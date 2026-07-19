/**
 * 基于轻量 JSON-RPC 的 MCP 服务器实现
 *
 * 原课程 Python 版本依赖 fastmcp。当前 TypeScript 项目没有引入 MCP SDK，
 * 因此这里实现一个教学用的最小 MCP 服务器封装，支持：
 * - 内存调用：可直接传给本目录的 MCPClient
 * - stdio：本地进程 JSON-RPC
 * - http：HTTP POST JSON-RPC
 * - sse：GET 保持 SSE 连接，POST 发送 JSON-RPC
 */

import { createServer, type IncomingMessage, type Server, type ServerResponse } from "http";
import { createInterface } from "readline";
import { pathToFileURL } from "url";

type MCPTransport = "stdio" | "http" | "sse";
type JsonRpcId = string | number | null;
type MCPCallable = (...args: unknown[]) => unknown | Promise<unknown>;

export interface MCPServerInfo {
    name: string;
    description: string;
    protocol: "MCP";
}

export interface MCPToolDefinition {
    name: string;
    description: string;
    inputSchema: Record<string, unknown>;
}

export interface MCPResourceDefinition {
    uri: string;
    name: string;
    description: string;
    mimeType?: string;
}

export interface MCPPromptDefinition {
    name: string;
    description: string;
    arguments: Array<Record<string, unknown>>;
}

export interface MCPRunOptions {
    host?: string;
    port?: number;
}

interface JsonRpcRequest {
    jsonrpc?: "2.0";
    id?: JsonRpcId;
    method?: string;
    params?: Record<string, unknown>;
}

interface RegisteredTool {
    func: MCPCallable;
    name: string;
    description: string;
    parameterNames: string[];
    inputSchema: Record<string, unknown>;
}

interface RegisteredResource {
    func: MCPCallable;
    uri: string;
    name: string;
    description: string;
}

interface RegisteredPrompt {
    func: MCPCallable;
    name: string;
    description: string;
    parameterNames: string[];
}

function getFunctionName(func: MCPCallable, fallback: string): string {
    return func.name && func.name !== "anonymous" ? func.name : fallback;
}

function getDescription(func: MCPCallable, fallback = ""): string {
    return fallback || `${getFunctionName(func, "handler")} handler`;
}

function stripComments(source: string): string {
    return source
        .replace(/\/\*[\s\S]*?\*\//g, "")
        .replace(/\/\/.*$/gm, "");
}

function getParameterNames(func: MCPCallable): string[] {
    const source = stripComments(func.toString());
    const parenMatch = source.match(/^[^(]*\(([^)]*)\)/);
    const arrowMatch = source.match(/^\s*([^=()\s,]+)\s*=>/);
    const paramsText = parenMatch?.[1] ?? arrowMatch?.[1] ?? "";

    return paramsText
        .split(",")
        .map((param) => param.trim())
        .filter(Boolean)
        .map((param) => param.split("=")[0].trim())
        .map((param) => param.replace(/^\.\.\./, ""))
        .filter((param) => /^[A-Za-z_$][\w$]*$/.test(param));
}

function createInputSchema(parameterNames: string[]): Record<string, unknown> {
    const properties: Record<string, Record<string, string>> = {};
    for (const name of parameterNames) {
        properties[name] = {
            type: "string",
            description: `${name} parameter`,
        };
    }

    return {
        type: "object",
        properties,
        required: parameterNames,
    };
}

function createPromptArguments(parameterNames: string[]): Array<Record<string, unknown>> {
    return parameterNames.map((name) => ({
        name,
        description: `${name} parameter`,
        required: true,
    }));
}

function asRecord(value: unknown): Record<string, unknown> {
    return value && typeof value === "object" && !Array.isArray(value)
        ? value as Record<string, unknown>
        : {};
}

function callWithArguments(func: MCPCallable, parameterNames: string[], args: Record<string, unknown>): unknown | Promise<unknown> {
    if (parameterNames.length === 0) {
        return func(args);
    }

    return func(...parameterNames.map((name) => args[name]));
}

function normalizeToolResult(result: unknown): Record<string, unknown> {
    const record = asRecord(result);
    if (Array.isArray(record.content)) return record;

    if (typeof result === "string") {
        return { content: [{ type: "text", text: result }] };
    }

    return { content: [{ type: "json", data: result }] };
}

function normalizeResourceResult(uri: string, result: unknown): Record<string, unknown> {
    const record = asRecord(result);
    if (Array.isArray(record.contents)) return record;

    if (typeof result === "string") {
        return { contents: [{ uri, text: result }] };
    }

    return { contents: [{ uri, text: JSON.stringify(result, null, 2) }] };
}

function normalizePromptResult(result: unknown): Record<string, unknown> {
    const record = asRecord(result);
    if (Array.isArray(record.messages)) return record;

    if (Array.isArray(result)) {
        return { messages: result };
    }

    return {
        messages: [{
            role: "user",
            content: { type: "text", text: String(result ?? "") },
        }],
    };
}

function jsonRpcSuccess(id: JsonRpcId, result: unknown): Record<string, unknown> {
    return { jsonrpc: "2.0", id, result };
}

function jsonRpcError(id: JsonRpcId, code: number, message: string, data?: unknown): Record<string, unknown> {
    return {
        jsonrpc: "2.0",
        id,
        error: data === undefined ? { code, message } : { code, message, data },
    };
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

async function readRequestBody(request: IncomingMessage): Promise<string> {
    const chunks: Buffer[] = [];
    for await (const chunk of request) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
    }
    return Buffer.concat(chunks).toString("utf-8");
}

function sendJson(response: ServerResponse, statusCode: number, payload: unknown): void {
    response.writeHead(statusCode, { "content-type": "application/json; charset=utf-8" });
    response.end(JSON.stringify(payload));
}

/**
 * MCP 服务器。
 *
 * 该实现覆盖课程示例所需的工具、资源、提示词注册和 JSON-RPC 调用。
 */
export class MCPServer {
    readonly name: string;
    readonly description: string;

    private readonly tools = new Map<string, RegisteredTool>();
    private readonly resources = new Map<string, RegisteredResource>();
    private readonly prompts = new Map<string, RegisteredPrompt>();

    constructor(name: string, description?: string) {
        this.name = name;
        this.description = description ?? `${name} MCP Server`;
    }

    addTool(func: MCPCallable, name?: string, description?: string): void {
        const resolvedName = name ?? getFunctionName(func, `tool_${this.tools.size + 1}`);
        const parameterNames = getParameterNames(func);

        this.tools.set(resolvedName, {
            func,
            name: resolvedName,
            description: getDescription(func, description),
            parameterNames,
            inputSchema: createInputSchema(parameterNames),
        });
    }

    addResource(func: MCPCallable, uri?: string, name?: string, description?: string): void {
        const resolvedName = name ?? getFunctionName(func, `resource_${this.resources.size + 1}`);
        const resolvedUri = uri ?? `resource://${resolvedName}`;

        this.resources.set(resolvedUri, {
            func,
            uri: resolvedUri,
            name: resolvedName,
            description: getDescription(func, description),
        });
    }

    addPrompt(func: MCPCallable, name?: string, description?: string): void {
        const resolvedName = name ?? getFunctionName(func, `prompt_${this.prompts.size + 1}`);
        const parameterNames = getParameterNames(func);

        this.prompts.set(resolvedName, {
            func,
            name: resolvedName,
            description: getDescription(func, description),
            parameterNames,
        });
    }

    run(transport: MCPTransport = "stdio", options: MCPRunOptions = {}): Server | void {
        if (transport === "stdio") {
            this.runStdio();
            return;
        }

        if (transport === "http" || transport === "sse") {
            return this.runHttp(transport, options);
        }

        throw new Error(`Unsupported transport: ${String(transport)}`);
    }

    getInfo(): MCPServerInfo {
        return {
            name: this.name,
            description: this.description,
            protocol: "MCP",
        };
    }

    async request(method: string, params: Record<string, unknown> = {}): Promise<unknown> {
        switch (method) {
            case "initialize":
                return {
                    protocolVersion: "2024-11-05",
                    capabilities: {
                        tools: {},
                        resources: {},
                        prompts: {},
                    },
                    serverInfo: {
                        name: this.name,
                        version: "1.0.0",
                    },
                };
            case "ping":
                return {};
            case "tools/list":
                return { tools: this.listTools() };
            case "tools/call":
                return this.callTool(String(params.name ?? ""), asRecord(params.arguments));
            case "resources/list":
                return { resources: this.listResources() };
            case "resources/read":
                return this.readResource(String(params.uri ?? ""));
            case "prompts/list":
                return { prompts: this.listPrompts() };
            case "prompts/get":
                return this.getPrompt(String(params.name ?? ""), asRecord(params.arguments));
            default:
                throw new Error(`Method not found: ${method}`);
        }
    }

    listTools(): MCPToolDefinition[] {
        return [...this.tools.values()].map((tool) => ({
            name: tool.name,
            description: tool.description,
            inputSchema: tool.inputSchema,
        }));
    }

    async callTool(name: string, args: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
        const tool = this.tools.get(name);
        if (!tool) throw new Error(`Tool not found: ${name}`);

        const result = await callWithArguments(tool.func, tool.parameterNames, args);
        return normalizeToolResult(result);
    }

    listResources(): MCPResourceDefinition[] {
        return [...this.resources.values()].map((resource) => ({
            uri: resource.uri,
            name: resource.name,
            description: resource.description,
        }));
    }

    async readResource(uri: string): Promise<Record<string, unknown>> {
        const resource = this.resources.get(uri);
        if (!resource) throw new Error(`Resource not found: ${uri}`);

        const result = resource.func.length > 0
            ? await resource.func(uri)
            : await resource.func();
        return normalizeResourceResult(uri, result);
    }

    listPrompts(): MCPPromptDefinition[] {
        return [...this.prompts.values()].map((prompt) => ({
            name: prompt.name,
            description: prompt.description,
            arguments: createPromptArguments(prompt.parameterNames),
        }));
    }

    async getPrompt(name: string, args: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
        const prompt = this.prompts.get(name);
        if (!prompt) throw new Error(`Prompt not found: ${name}`);

        const result = await callWithArguments(prompt.func, prompt.parameterNames, args);
        return normalizePromptResult(result);
    }

    async ping(): Promise<Record<string, unknown>> {
        return {};
    }

    // Python 风格别名，便于课程材料一一对应。
    add_tool(func: MCPCallable, name?: string, description?: string): void {
        this.addTool(func, name, description);
    }

    add_resource(func: MCPCallable, uri?: string, name?: string, description?: string): void {
        this.addResource(func, uri, name, description);
    }

    add_prompt(func: MCPCallable, name?: string, description?: string): void {
        this.addPrompt(func, name, description);
    }

    get_info(): MCPServerInfo {
        return this.getInfo();
    }

    list_tools(): MCPToolDefinition[] {
        return this.listTools();
    }

    call_tool(name: string, args: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
        return this.callTool(name, args);
    }

    list_resources(): MCPResourceDefinition[] {
        return this.listResources();
    }

    read_resource(uri: string): Promise<Record<string, unknown>> {
        return this.readResource(uri);
    }

    list_prompts(): MCPPromptDefinition[] {
        return this.listPrompts();
    }

    get_prompt(name: string, args: Record<string, unknown> = {}): Promise<Record<string, unknown>> {
        return this.getPrompt(name, args);
    }

    private runStdio(): void {
        const reader = createInterface({
            input: process.stdin,
            terminal: false,
        });

        reader.on("line", async (line) => {
            const trimmed = line.trim();
            if (!trimmed) return;

            let request: JsonRpcRequest;
            try {
                request = JSON.parse(trimmed) as JsonRpcRequest;
            } catch (error) {
                process.stdout.write(`${JSON.stringify(jsonRpcError(null, -32700, `Parse error: ${errorMessage(error)}`))}\n`);
                return;
            }

            const response = await this.handleJsonRpcRequest(request);
            if (response) {
                process.stdout.write(`${JSON.stringify(response)}\n`);
            }
        });
    }

    private runHttp(transport: "http" | "sse", options: MCPRunOptions): Server {
        const host = options.host ?? "127.0.0.1";
        const port = options.port ?? 8000;

        const server = createServer(async (request, response) => {
            if (transport === "sse" && request.method === "GET") {
                this.handleSseConnection(response);
                return;
            }

            if (request.method !== "POST") {
                sendJson(response, 405, { error: { message: "Method Not Allowed" } });
                return;
            }

            try {
                const body = await readRequestBody(request);
                const jsonRpcRequest = JSON.parse(body) as JsonRpcRequest;
                const payload = await this.handleJsonRpcRequest(jsonRpcRequest);
                if (!payload) {
                    response.writeHead(204);
                    response.end();
                    return;
                }
                sendJson(response, 200, payload);
            } catch (error) {
                sendJson(response, 400, jsonRpcError(null, -32600, errorMessage(error)));
            }
        });

        server.listen(port, host, () => {
            console.log(`🚀 ${this.name} listening on ${host}:${port} (${transport})`);
        });

        return server;
    }

    private handleSseConnection(response: ServerResponse): void {
        response.writeHead(200, {
            "content-type": "text/event-stream; charset=utf-8",
            "cache-control": "no-cache",
            connection: "keep-alive",
        });
        response.write(`event: endpoint\ndata: /\n\n`);

        const heartbeat = setInterval(() => {
            response.write(`: ping ${Date.now()}\n\n`);
        }, 15_000);

        response.on("close", () => {
            clearInterval(heartbeat);
        });
    }

    private async handleJsonRpcRequest(request: JsonRpcRequest): Promise<Record<string, unknown> | null> {
        const id = request.id ?? null;
        const method = request.method;

        if (!method) {
            return jsonRpcError(id, -32600, "Invalid Request: missing method");
        }

        try {
            const result = await this.request(method, request.params ?? {});
            if (request.id === undefined) return null;
            return jsonRpcSuccess(id, result);
        } catch (error) {
            if (request.id === undefined) return null;
            const message = errorMessage(error);
            const code = message.startsWith("Method not found") ? -32601 : -32000;
            return jsonRpcError(id, code, message);
        }
    }
}

/**
 * MCP 服务器构建器，提供链式 API。
 */
export class MCPServerBuilder {
    private readonly server: MCPServer;

    constructor(name: string, description?: string) {
        this.server = new MCPServer(name, description);
    }

    withTool(func: MCPCallable, name?: string, description?: string): this {
        this.server.addTool(func, name, description);
        return this;
    }

    withResource(func: MCPCallable, uri?: string, name?: string, description?: string): this {
        this.server.addResource(func, uri, name, description);
        return this;
    }

    withPrompt(func: MCPCallable, name?: string, description?: string): this {
        this.server.addPrompt(func, name, description);
        return this;
    }

    build(): MCPServer {
        return this.server;
    }

    run(transport: MCPTransport = "stdio", options: MCPRunOptions = {}): Server | void {
        return this.server.run(transport, options);
    }

    // Python 风格别名。
    with_tool(func: MCPCallable, name?: string, description?: string): this {
        return this.withTool(func, name, description);
    }

    with_resource(func: MCPCallable, uri?: string, name?: string, description?: string): this {
        return this.withResource(func, uri, name, description);
    }

    with_prompt(func: MCPCallable, name?: string, description?: string): this {
        return this.withPrompt(func, name, description);
    }
}

/**
 * 创建一个示例 MCP 服务器。
 */
export function createExampleServer(): MCPServer {
    const server = new MCPServer(
        "example-server",
        "A simple example MCP server with calculator and greeting tools",
    );

    function calculator(expression: string): string {
        try {
            const allowedPattern = /^[0-9+\-*/().\s]+$/;
            if (!allowedPattern.test(expression)) {
                return "Error: Invalid characters in expression";
            }

            const result = Function(`"use strict"; return (${expression});`)();
            return `Result: ${String(result)}`;
        } catch (error) {
            return `Error: ${errorMessage(error)}`;
        }
    }

    server.addTool(calculator, "calculator", "Calculate a mathematical expression");

    function greet(name: string): string {
        return `Hello, ${name}! Welcome to the MCP server example.`;
    }

    server.addTool(greet, "greet", "Generate a friendly greeting");

    return server;
}

export const create_example_server = createExampleServer;

function isMainModule(): boolean {
    return Boolean(
        process.argv[1]
        && import.meta.url === pathToFileURL(process.argv[1]).href,
    );
}

if (isMainModule()) {
    const server = createExampleServer();
    console.error(`🚀 Starting ${server.name}...`);
    console.error(`📝 ${server.description}`);
    console.error("🔌 Protocol: MCP");
    console.error("📡 Transport: stdio");
    server.run();
}
