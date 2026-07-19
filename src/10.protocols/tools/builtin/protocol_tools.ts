/**
 * 协议工具集合
 *
 * 当前文件只实现 MCP 协议相关工具：
 * - MCPTool: 连接并调用 MCP 服务器
 * - MCPWrappedTool: 将 MCP 服务器中的单个工具包装为 HelloAgents Tool
 */

import { Tool, ToolParameter } from "../../../7.hello-agents/tools/base";
import { MCPClient, type MCPServerSource, type MCPToolInfo, type MemoryMCPServer } from "../../mcp/client";
import { MCPServer } from "../../mcp/server";

export const MCP_SERVER_ENV_MAP: Record<string, string[]> = {
    "server-github": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
    "server-slack": ["SLACK_BOT_TOKEN", "SLACK_TEAM_ID"],
    "server-google-drive": ["GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "GOOGLE_REFRESH_TOKEN"],
    "server-postgres": ["POSTGRES_CONNECTION_STRING"],
    "server-sqlite": [],
    "server-filesystem": [],
};

export interface MCPToolOptions {
    name?: string;
    description?: string;
    serverCommand?: string[];
    serverArgs?: string[];
    server?: MemoryMCPServer;
    autoExpand?: boolean;
    env?: Record<string, string>;
    envKeys?: string[];
}

interface MCPToolParams {
    action?: string;
    tool_name?: string;
    toolName?: string;
    arguments?: Record<string, unknown>;
    uri?: string;
    prompt_name?: string;
    promptName?: string;
    prompt_arguments?: Record<string, string>;
    promptArguments?: Record<string, string>;
}

function prepareEnv(
    env: Record<string, string> | undefined,
    envKeys: string[] | undefined,
    serverCommand: string[] | undefined,
): Record<string, string> {
    const resultEnv: Record<string, string> = {};

    if (serverCommand) {
        const serverName = serverCommand
            .map((part) => part.includes("server-") ? part.split("/").at(-1) : undefined)
            .find(Boolean);

        if (serverName && MCP_SERVER_ENV_MAP[serverName]) {
            for (const key of MCP_SERVER_ENV_MAP[serverName]) {
                const value = process.env[key];
                if (value) {
                    resultEnv[key] = value;
                    console.log(`🔑 自动加载环境变量: ${key}`);
                }
            }
        }
    }

    if (envKeys) {
        for (const key of envKeys) {
            const value = process.env[key];
            if (value) {
                resultEnv[key] = value;
                console.log(`🔑 从envKeys加载环境变量: ${key}`);
            } else {
                console.warn(`⚠️ 警告: 环境变量 ${key} 未设置`);
            }
        }
    }

    if (env) {
        Object.assign(resultEnv, env);
        for (const key of Object.keys(env)) {
            console.log(`🔑 使用直接传递的环境变量: ${key}`);
        }
    }

    return resultEnv;
}

function createBuiltinServer(): MCPServer {
    const server = new MCPServer("HelloAgents-BuiltinServer");

    function add(a: number, b: number): number {
        return Number(a) + Number(b);
    }

    function subtract(a: number, b: number): number {
        return Number(a) - Number(b);
    }

    function multiply(a: number, b: number): number {
        return Number(a) * Number(b);
    }

    function divide(a: number, b: number): number | string {
        const divisor = Number(b);
        if (divisor === 0) return "Error: 除数不能为零";
        return Number(a) / divisor;
    }

    function greet(name: string = "World"): string {
        return `Hello, ${name}! 欢迎使用 HelloAgents MCP 工具！`;
    }

    function get_system_info(): Record<string, unknown> {
        return {
            platform: process.platform,
            node_version: process.version,
            server_name: "HelloAgents-BuiltinServer",
            tools_count: 6,
        };
    }

    server.addTool(add, "add", "加法计算器");
    server.addTool(subtract, "subtract", "减法计算器");
    server.addTool(multiply, "multiply", "乘法计算器");
    server.addTool(divide, "divide", "除法计算器");
    server.addTool(greet, "greet", "友好问候");
    server.addTool(get_system_info, "get_system_info", "获取系统信息");

    return server;
}

function discoverMemoryTools(serverSource: MCPServerSource): MCPToolInfo[] {
    const server = serverSource as MemoryMCPServer;
    const listTools = server.listTools ?? server.list_tools;
    if (!listTools) return [];

    try {
        const result = listTools.call(server);
        if (result instanceof Promise) return [];

        const tools = Array.isArray(result)
            ? result
            : Array.isArray((result as { tools?: unknown[] }).tools)
                ? (result as { tools: unknown[] }).tools
                : [];

        return tools.map((tool) => {
            const record = tool && typeof tool === "object" ? tool as Record<string, unknown> : {};
            return {
                name: String(record.name ?? ""),
                description: String(record.description ?? ""),
                input_schema: (record.inputSchema ?? record.input_schema ?? {}) as Record<string, unknown>,
            };
        }).filter((tool) => tool.name);
    } catch {
        return [];
    }
}

function generateDescription(availableTools: MCPToolInfo[], autoExpand: boolean): string {
    if (availableTools.length === 0) {
        return "连接到 MCP 服务器，调用工具、读取资源和获取提示词。支持内置服务器和外部服务器。";
    }

    if (autoExpand) {
        return `MCP工具服务器，包含${availableTools.length}个工具。这些工具会自动展开为独立的工具供Agent使用。`;
    }

    const lines = [`MCP工具服务器，提供${availableTools.length}个工具：`];
    for (const tool of availableTools) {
        const shortDescription = tool.description.split(".")[0] || "无描述";
        lines.push(`  • ${tool.name}: ${shortDescription}`);
    }
    lines.push("");
    lines.push("调用格式：返回JSON格式的参数");
    lines.push('{"action": "call_tool", "tool_name": "工具名", "arguments": {...}}');

    const firstTool = availableTools[0];
    if (firstTool) {
        lines.push("");
        lines.push(`示例：{"action": "call_tool", "tool_name": "${firstTool.name}", "arguments": {...}}`);
    }

    return lines.join("\n");
}

function stringifyResult(value: unknown): string {
    if (typeof value === "string") return value;
    return JSON.stringify(value, null, 2);
}

function schemaToParameters(schema: Record<string, unknown>): ToolParameter[] {
    const properties = schema.properties && typeof schema.properties === "object"
        ? schema.properties as Record<string, Record<string, unknown>>
        : {};
    const required = Array.isArray(schema.required)
        ? new Set(schema.required.map(String))
        : new Set<string>();

    return Object.entries(properties).map(([name, property]) => new ToolParameter({
        name,
        type: String(property.type ?? "string"),
        description: String(property.description ?? `${name} parameter`),
        required: required.has(name),
        default: property.default ?? null,
    }));
}

/**
 * MCP (Model Context Protocol) 工具。
 *
 * 连接到 MCP 服务器并调用其提供的工具、资源和提示词。如果没有提供外部
 * server，会自动创建内置演示服务器。
 */
export class MCPTool extends Tool {
    private readonly serverCommand?: string[];
    private readonly serverArgs: string[];
    private readonly server?: MemoryMCPServer;
    private readonly autoExpand: boolean;
    private readonly prefix: string;
    private readonly env: Record<string, string>;
    private availableTools: MCPToolInfo[];

    constructor(options?: MCPToolOptions);
    constructor(
        name?: string,
        description?: string,
        serverCommand?: string[],
        serverArgs?: string[],
        server?: MemoryMCPServer,
        autoExpand?: boolean,
        env?: Record<string, string>,
        envKeys?: string[],
    );
    constructor(
        optionsOrName: MCPToolOptions | string = "mcp",
        description?: string,
        serverCommand?: string[],
        serverArgs: string[] = [],
        server?: MemoryMCPServer,
        autoExpand = true,
        env?: Record<string, string>,
        envKeys?: string[],
    ) {
        const options = typeof optionsOrName === "string"
            ? { name: optionsOrName, description, serverCommand, serverArgs, server, autoExpand, env, envKeys }
            : optionsOrName;

        const resolvedName = options.name ?? "mcp";
        const resolvedAutoExpand = options.autoExpand ?? true;
        const resolvedEnv = prepareEnv(options.env, options.envKeys, options.serverCommand);
        const resolvedServer = options.server ?? (
            options.serverCommand ? undefined : createBuiltinServer()
        );
        const initialTools = resolvedServer ? discoverMemoryTools(resolvedServer) : [];
        const resolvedDescription = options.description ?? generateDescription(initialTools, resolvedAutoExpand);

        super(resolvedName, resolvedDescription);

        this.serverCommand = options.serverCommand;
        this.serverArgs = options.serverArgs ?? [];
        this.server = resolvedServer;
        this.autoExpand = resolvedAutoExpand;
        this.prefix = this.autoExpand ? `${resolvedName}_` : "";
        this.env = resolvedEnv;
        this.availableTools = initialTools;
    }

    async run(parameters: Record<string, unknown>): Promise<string> {
        const params = parameters as MCPToolParams;
        let action = String(params.action ?? "").toLowerCase();

        if (!action && (params.tool_name || params.toolName)) {
            action = "call_tool";
        }

        if (!action) {
            return "错误：必须指定 action 参数或 tool_name 参数";
        }

        try {
            return await this.withClient(async (client) => {
                if (action === "list_tools") {
                    const tools = await client.list_tools();
                    this.availableTools = tools;
                    if (tools.length === 0) return "没有找到可用的工具";

                    let result = `找到 ${tools.length} 个工具:\n`;
                    for (const tool of tools) {
                        result += `- ${tool.name}: ${tool.description}\n`;
                    }
                    return result;
                }

                if (action === "call_tool") {
                    const toolName = params.tool_name ?? params.toolName;
                    if (!toolName) return "错误：必须指定 tool_name 参数";

                    const result = await client.call_tool(toolName, params.arguments ?? {});
                    return `工具 '${toolName}' 执行结果:\n${stringifyResult(result)}`;
                }

                if (action === "list_resources") {
                    const resources = await client.list_resources();
                    if (resources.length === 0) return "没有找到可用的资源";

                    let result = `找到 ${resources.length} 个资源:\n`;
                    for (const resource of resources) {
                        result += `- ${resource.uri}: ${resource.name}\n`;
                    }
                    return result;
                }

                if (action === "read_resource") {
                    if (!params.uri) return "错误：必须指定 uri 参数";
                    const content = await client.read_resource(params.uri);
                    return `资源 '${params.uri}' 内容:\n${stringifyResult(content)}`;
                }

                if (action === "list_prompts") {
                    const prompts = await client.list_prompts();
                    if (prompts.length === 0) return "没有找到可用的提示词";

                    let result = `找到 ${prompts.length} 个提示词:\n`;
                    for (const prompt of prompts) {
                        result += `- ${prompt.name}: ${prompt.description}\n`;
                    }
                    return result;
                }

                if (action === "get_prompt") {
                    const promptName = params.prompt_name ?? params.promptName;
                    if (!promptName) return "错误：必须指定 prompt_name 参数";

                    const messages = await client.get_prompt(
                        promptName,
                        params.prompt_arguments ?? params.promptArguments ?? {},
                    );

                    let result = `提示词 '${promptName}':\n`;
                    for (const message of messages) {
                        result += `[${message.role}] ${stringifyResult(message.content)}\n`;
                    }
                    return result;
                }

                return `错误：不支持的操作 '${action}'`;
            });
        } catch (error) {
            return `MCP 操作失败: ${error instanceof Error ? error.message : String(error)}`;
        }
    }

    getParameters(): ToolParameter[] {
        return [
            new ToolParameter({
                name: "action",
                type: "string",
                description: "操作类型: list_tools, call_tool, list_resources, read_resource, list_prompts, get_prompt",
                required: false,
            }),
            new ToolParameter({
                name: "tool_name",
                type: "string",
                description: "工具名称（call_tool 操作需要）",
                required: false,
            }),
            new ToolParameter({
                name: "arguments",
                type: "object",
                description: "工具参数（call_tool 操作需要）",
                required: false,
            }),
            new ToolParameter({
                name: "uri",
                type: "string",
                description: "资源 URI（read_resource 操作需要）",
                required: false,
            }),
            new ToolParameter({
                name: "prompt_name",
                type: "string",
                description: "提示词名称（get_prompt 操作需要）",
                required: false,
            }),
            new ToolParameter({
                name: "prompt_arguments",
                type: "object",
                description: "提示词参数（get_prompt 操作可选）",
                required: false,
            }),
        ];
    }

    getExpandedTools(): Tool[] | undefined {
        if (!this.autoExpand) return undefined;

        const tools = this.availableTools.map((toolInfo) => new MCPWrappedTool(this, toolInfo, this.prefix));
        return tools.length > 0 ? tools : undefined;
    }

    async discoverTools(): Promise<MCPToolInfo[]> {
        return this.withClient(async (client) => {
            const tools = await client.list_tools();
            this.availableTools = tools;
            return tools;
        });
    }

    async invokeTool(toolName: string, args: Record<string, unknown> = {}): Promise<unknown> {
        return this.withClient((client) => client.call_tool(toolName, args));
    }

    private getClientSource(): MCPServerSource {
        if (this.server) return this.server;
        if (this.serverCommand) return this.serverCommand;
        return createBuiltinServer();
    }

    private async withClient<T>(callback: (client: MCPClient) => Promise<T>): Promise<T> {
        const client = new MCPClient(this.getClientSource(), {
            serverArgs: this.serverArgs,
            env: this.env,
        });

        await client.connect();
        try {
            return await callback(client);
        } finally {
            await client.disconnect();
        }
    }
}

/**
 * 将 MCP 服务器中的单个工具包装成 HelloAgents Tool。
 */
export class MCPWrappedTool extends Tool {
    private readonly mcpTool: MCPTool;
    private readonly toolInfo: MCPToolInfo;

    constructor(mcpTool: MCPTool, toolInfo: MCPToolInfo, prefix = "") {
        super(
            `${prefix}${toolInfo.name}`,
            toolInfo.description || `MCP tool: ${toolInfo.name}`,
        );

        this.mcpTool = mcpTool;
        this.toolInfo = toolInfo;
    }

    async run(parameters: Record<string, unknown>): Promise<string> {
        try {
            const result = await this.mcpTool.invokeTool(this.toolInfo.name, parameters);
            return stringifyResult(result);
        } catch (error) {
            return `MCP 工具 '${this.toolInfo.name}' 执行失败: ${error instanceof Error ? error.message : String(error)}`;
        }
    }

    getParameters(): ToolParameter[] {
        return schemaToParameters(this.toolInfo.input_schema);
    }
}
