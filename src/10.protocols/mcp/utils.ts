/**
 * MCP 协议工具函数
 *
 * 提供上下文管理、消息解析等辅助功能。这些函数主要用于处理 MCP 协议的
 * 数据结构，并保持与课程 Python 示例一致的返回形状。
 */

export type MCPMessage = Record<string, unknown>;
export type MCPToolDefinition = Record<string, unknown>;
export type MCPResourceDefinition = Record<string, unknown>;
export type MCPMetadata = Record<string, unknown>;

export interface MCPContext {
    messages: MCPMessage[];
    tools: MCPToolDefinition[];
    resources: MCPResourceDefinition[];
    metadata: MCPMetadata;
}

export interface MCPErrorResponse {
    error: {
        message: string;
        code: string;
        details?: Record<string, unknown>;
    };
}

export interface MCPSuccessResponse<T = unknown> {
    success: true;
    data: T;
    metadata?: MCPMetadata;
}

export interface CreateContextOptions {
    messages?: MCPMessage[];
    tools?: MCPToolDefinition[];
    resources?: MCPResourceDefinition[];
    metadata?: MCPMetadata;
}

/**
 * 创建 MCP 上下文对象。
 */
export function createContext(
    messages: MCPMessage[] = [],
    tools: MCPToolDefinition[] = [],
    resources: MCPResourceDefinition[] = [],
    metadata: MCPMetadata = {},
): MCPContext {
    return {
        messages,
        tools,
        resources,
        metadata,
    };
}

/**
 * 用对象参数创建 MCP 上下文，适合 TS 调用方按字段传参。
 */
export function createContextFromOptions(options: CreateContextOptions = {}): MCPContext {
    return createContext(
        options.messages ?? [],
        options.tools ?? [],
        options.resources ?? [],
        options.metadata ?? {},
    );
}

/**
 * 解析 MCP 上下文。
 *
 * 接受 JSON 字符串或普通对象；会补齐 `messages/tools/resources/metadata`
 * 字段，并校验前三个字段必须是数组、metadata 必须是对象。
 */
export function parseContext(context: string | Partial<MCPContext> | Record<string, unknown>): MCPContext {
    let parsed: unknown = context;

    if (typeof context === "string") {
        try {
            parsed = JSON.parse(context);
        } catch (error) {
            throw new Error(`Invalid JSON context: ${errorMessage(error)}`);
        }
    }

    if (!isPlainObject(parsed)) {
        throw new Error("Context must be a dictionary or JSON string");
    }

    const record = parsed as Record<string, unknown>;
    const messages = record.messages ?? [];
    const tools = record.tools ?? [];
    const resources = record.resources ?? [];
    const metadata = record.metadata ?? {};

    if (!Array.isArray(messages)) {
        throw new Error("Context field 'messages' must be an array");
    }
    if (!Array.isArray(tools)) {
        throw new Error("Context field 'tools' must be an array");
    }
    if (!Array.isArray(resources)) {
        throw new Error("Context field 'resources' must be an array");
    }
    if (!isPlainObject(metadata)) {
        throw new Error("Context field 'metadata' must be an object");
    }

    return {
        messages: messages.map(asRecord),
        tools: tools.map(asRecord),
        resources: resources.map(asRecord),
        metadata: metadata as MCPMetadata,
    };
}

/**
 * 创建错误响应。
 */
export function createErrorResponse(
    errorMessage: string,
    errorCode: string = "UNKNOWN_ERROR",
    details?: Record<string, unknown>,
): MCPErrorResponse {
    const response: MCPErrorResponse = {
        error: {
            message: errorMessage,
            code: errorCode,
        },
    };

    if (details) {
        response.error.details = details;
    }

    return response;
}

/**
 * 创建成功响应。
 */
export function createSuccessResponse<T = unknown>(
    data: T,
    metadata?: MCPMetadata,
): MCPSuccessResponse<T> {
    const response: MCPSuccessResponse<T> = {
        success: true,
        data,
    };

    if (metadata) {
        response.metadata = metadata;
    }

    return response;
}

// Python 风格别名，便于课程材料一一对应。
export const create_context = createContext;
export const parse_context = parseContext;
export const create_error_response = createErrorResponse;
export const create_success_response = createSuccessResponse;

function isPlainObject(value: unknown): value is Record<string, unknown> {
    return Boolean(
        value
        && typeof value === "object"
        && !Array.isArray(value),
    );
}

function asRecord(value: unknown): Record<string, unknown> {
    return isPlainObject(value) ? value : {};
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
