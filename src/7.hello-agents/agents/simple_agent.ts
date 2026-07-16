import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

import { Agent } from "../core/agent";
import type { Config } from "../core/config";
import type { HelloAgentsLLM, InvokeOptions } from "../core/llm";
import { Message } from "../core/message";
import type { Tool } from "../tools/base";
import { ToolRegistry } from "../tools/registry";

/** 工具参数的最小类型定义，供 Agent 根据声明自动转换参数值。 */
export interface ToolParameter {
  name: string;
  type: string;
}

/** 从模型文本协议中解析出的一次工具调用。 */
interface ParsedToolCall {
  toolName: string;
  parameters: string;
  original: string;
}

/** 单次运行参数；除迭代上限外，其余字段会原样传递给 LLM。 */
export interface SimpleAgentRunOptions extends InvokeOptions {
  maxToolIterations?: number;
}

/**
 * 简单对话 Agent，支持基于文本标记的可选工具调用。
 *
 * 模型需要输出 `[TOOL_CALL:工具名:参数]`。Agent 解析并执行工具后，
 * 会把结果作为新消息发回模型，直到得到普通回答或达到最大迭代次数。
 */
export class SimpleAgent extends Agent {
  /** 当前使用的工具注册表；首次调用 addTool 时可按需创建。 */
  protected toolRegistry?: ToolRegistry;

  /** 是否启用工具调用；只有存在工具注册表时，构造参数才会生效。 */
  protected enableToolCalling: boolean;

  constructor(
    name: string,
    llm: HelloAgentsLLM,
    systemPrompt?: string,
    config?: Config,
    toolRegistry?: ToolRegistry,
    enableToolCalling = true,
  ) {
    super(name, llm, systemPrompt, config);
    this.toolRegistry = toolRegistry;
    this.enableToolCalling = enableToolCalling && toolRegistry !== undefined;
  }

  /** 在基础系统提示词后追加工具清单、调用格式和参数书写规则。 */
  protected getEnhancedSystemPrompt(): string {
    const basePrompt = this.systemPrompt ?? "你是一个有用的AI助手。";

    if (!this.enableToolCalling || !this.toolRegistry) return basePrompt;

    const toolsDescription = this.toolRegistry.getToolsDescription();
    if (!toolsDescription || toolsDescription === "暂无可用工具") return basePrompt;

    // 使用数组集中维护提示词片段，便于阅读并避免大量字符串累加操作。
    const toolsSection = [
      "",
      "## 可用工具",
      "你可以使用以下工具来帮助回答问题：",
      toolsDescription,
      "",
      "## 工具调用格式",
      "当需要使用工具时，请使用以下格式：",
      "`[TOOL_CALL:{tool_name}:{parameters}]`",
      "",
      "### 参数格式说明",
      "1. **多个参数**：使用 `key=value` 格式，用逗号分隔",
      "   示例：`[TOOL_CALL:calculator_multiply:a=12,b=8]`",
      "   示例：`[TOOL_CALL:filesystem_read_file:path=README.md]`",
      "",
      "2. **单个参数**：直接使用 `key=value`",
      "   示例：`[TOOL_CALL:search:query=Python编程]`",
      "",
      "3. **简单查询**：可以直接传入文本",
      "   示例：`[TOOL_CALL:search:Python编程]`",
      "",
      "### 重要提示",
      "- 参数名必须与工具定义的参数名完全匹配",
      "- 数字参数直接写数字，不需要引号：`a=12` 而不是 `a=\"12\"`",
      "- 文件路径等字符串参数直接写：`path=README.md`",
      "- 工具调用结果会自动插入到对话中，然后你可以基于结果继续回答",
    ].join("\n");

    return `${basePrompt}\n${toolsSection}`;
  }

  /** 提取一段模型回复中出现的全部工具调用标记。 */
  protected parseToolCalls(text: string): ParsedToolCall[] {
    const pattern = /\[TOOL_CALL:([^:]+):([^\]]+)\]/g;
    const toolCalls: ParsedToolCall[] = [];

    for (const match of text.matchAll(pattern)) {
      const [, toolName, parameters] = match;
      toolCalls.push({
        toolName: toolName.trim(),
        parameters: parameters.trim(),
        original: match[0],
      });
    }

    return toolCalls;
  }

  /** 解析参数、执行指定工具，并把成功或失败统一格式化为模型可读文本。 */
  protected async executeToolCall(
    toolName: string,
    parameters: string,
  ): Promise<string> {
    if (!this.toolRegistry) return "❌ 错误：未配置工具注册表";

    try {
      const tool = this.toolRegistry.getTool(toolName);
      if (!tool) return `❌ 错误：未找到工具 '${toolName}'`;

      const parameterDictionary = this.parseToolParameters(toolName, parameters);
      // Promise.resolve 让同步工具和异步工具都可以使用相同的执行路径。
      const result = await Promise.resolve(tool.run(parameterDictionary));
      return `🔧 工具 ${toolName} 执行结果：\n${result}`;
    } catch (error) {
      return `❌ 工具调用失败：${this.getErrorMessage(error)}`;
    }
  }

  /**
   * 兼容三种参数形式：JSON 对象、逗号分隔的 key=value，以及纯文本。
   * JSON 解析失败时不会直接报错，而会继续尝试较宽松的文本格式。
   */
  protected parseToolParameters(
    toolName: string,
    parameters: string,
  ): Record<string, unknown> {
    const trimmedParameters = parameters.trim();

    if (trimmedParameters.startsWith("{")) {
      try {
        const parsed: unknown = JSON.parse(trimmedParameters);
        if (this.isParameterDictionary(parsed)) {
          return this.convertParameterTypes(toolName, parsed);
        }
      } catch {
        // JSON 格式无效时继续按 key=value 或简单文本解析。
      }
    }

    if (parameters.includes("=")) {
      const parameterDictionary: Record<string, unknown> = {};

      // 这里刻意沿用原实现的简单逗号分隔协议，不处理带引号的嵌套逗号。
      for (const pair of parameters.split(",")) {
        const separatorIndex = pair.indexOf("=");
        if (separatorIndex < 0) continue;

        const key = pair.slice(0, separatorIndex).trim();
        const value = pair.slice(separatorIndex + 1).trim();
        parameterDictionary[key] = value;
      }

      const convertedParameters = this.convertParameterTypes(
        toolName,
        parameterDictionary,
      );
      return "action" in convertedParameters
        ? convertedParameters
        : this.inferAction(toolName, convertedParameters);
    }

    return this.inferSimpleParameters(toolName, parameters);
  }

  /** 根据工具参数声明，把模型生成的字符串转换为数字或布尔值。 */
  protected convertParameterTypes(
    toolName: string,
    parameters: Record<string, unknown>,
  ): Record<string, unknown> {
    const tool = this.toolRegistry?.getTool(toolName);
    if (!tool) return parameters;

    let parameterDefinitions: ToolParameter[];
    try {
      parameterDefinitions = tool.getParameters();
    } catch {
      // 工具无法提供参数元数据时，仍允许它接收未经转换的原始值。
      return parameters;
    }

    const parameterTypes = new Map(
      parameterDefinitions.map((parameter) => [parameter.name, parameter.type]),
    );
    const convertedParameters: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(parameters)) {
      const parameterType = parameterTypes.get(key);
      convertedParameters[key] = this.convertParameterValue(value, parameterType);
    }

    return convertedParameters;
  }

  /** 转换单个参数；转换失败时保留原值，交由具体工具做最终校验。 */
  private convertParameterValue(value: unknown, parameterType?: string): unknown {
    if (parameterType === "number" && typeof value === "string") {
      const converted = Number(value);
      return Number.isNaN(converted) ? value : converted;
    }

    if (parameterType === "integer" && typeof value === "string") {
      const converted = Number(value);
      return Number.isInteger(converted) ? converted : value;
    }

    if (parameterType === "boolean") {
      if (typeof value === "string") {
        return ["true", "1", "yes"].includes(value.toLowerCase());
      }
      return Boolean(value);
    }

    return value;
  }

  /** 为 memory 和 rag 的简写参数补齐 action 及规范字段名。 */
  protected inferAction(
    toolName: string,
    parameters: Record<string, unknown>,
  ): Record<string, unknown> {
    if (toolName === "memory") {
      if ("recall" in parameters) {
        parameters.action = "search";
        parameters.query = parameters.recall;
        delete parameters.recall;
      } else if ("store" in parameters) {
        parameters.action = "add";
        parameters.content = parameters.store;
        delete parameters.store;
      } else if ("query" in parameters) {
        parameters.action = "search";
      } else if ("content" in parameters) {
        parameters.action = "add";
      }
    } else if (toolName === "rag") {
      if ("search" in parameters) {
        parameters.action = "search";
        parameters.query = parameters.search;
        delete parameters.search;
      } else if ("query" in parameters) {
        parameters.action = "search";
      } else if ("text" in parameters) {
        parameters.action = "add_text";
      }
    }

    return parameters;
  }

  /** 纯文本参数默认用于检索；其他工具则收到名为 input 的通用参数。 */
  private inferSimpleParameters(
    toolName: string,
    parameters: string,
  ): Record<string, unknown> {
    if (toolName === "rag" || toolName === "memory") {
      return { action: "search", query: parameters };
    }
    return { input: parameters };
  }

  /**
   * 执行一次完整对话。
   *
   * 工具模式下每轮先检查模型是否请求工具；请求存在时执行全部工具并把结果
   * 追加到临时上下文。只有本次用户输入与最终回答会写入持久历史。
   */
  async run(
    inputText: string,
    options: SimpleAgentRunOptions = {},
  ): Promise<string> {
    const { maxToolIterations = 3, ...invokeOptions } = options;
    const messages = this.buildMessages(inputText, true);

    if (!this.enableToolCalling) {
      const response = await this.llm.invoke(messages, invokeOptions);
      this.recordConversation(inputText, response);
      return response;
    }

    let currentIteration = 0;
    let finalResponse = "";

    while (currentIteration < maxToolIterations) {
      const response = await this.llm.invoke(messages, invokeOptions);
      // 解析模型回复中的工具调用标记
      const toolCalls = this.parseToolCalls(response);

      // 不需要调用工具则退出
      if (toolCalls.length === 0) {
        finalResponse = response;
        break;
      }

      // 保留包含调用标记的原始助手消息，让模型能看到此前作出的工具选择。
      messages.push({ role: "assistant", content: response });

      // 执行工具调用，并收集结果
      const toolResults: string[] = [];
      for (const toolCall of toolCalls) {
        toolResults.push(
          await this.executeToolCall(toolCall.toolName, toolCall.parameters),
        );
      }

      // 将工具调用结果加入到历史信息中
      messages.push({
        role: "user",
        content: `工具执行结果：\n${toolResults.join(
          "\n\n",
        )}\n\n请基于这些结果给出完整的回答。`,
      });
      currentIteration += 1;
    }

    // 达到上限后再请求一次最终回答，但不再解析其中可能出现的新工具调用。
    if (currentIteration >= maxToolIterations && !finalResponse) {
      finalResponse = await this.llm.invoke(messages, invokeOptions);
    }

    this.recordConversation(inputText, finalResponse);
    return finalResponse;
  }

  /** 动态添加工具；若构造时未注入注册表，会自动创建一个本地注册表。 */
  addTool(tool: Tool, autoExpand = true): void {
    if (!this.toolRegistry) {
      this.toolRegistry = new ToolRegistry();
      this.enableToolCalling = true;
    }
    this.toolRegistry.registerTool(tool, autoExpand);
  }

  /** 按名称移除工具，并返回本次操作是否真的移除了工具。 */
  removeTool(toolName: string): boolean {
    return this.toolRegistry?.unregisterTool(toolName) ?? false;
  }

  /** 返回当前注册表中全部可用工具的名称。 */
  listTools(): string[] {
    return this.toolRegistry?.listTools() ?? [];
  }

  /** 判断当前 Agent 是否已启用工具调用并持有工具注册表。 */
  hasTools(): boolean {
    return this.enableToolCalling && this.toolRegistry !== undefined;
  }

  /**
   * 流式执行一次对话并逐段产出文本。
   *
   * 与原实现一致，流式模式不解析工具标记，并且只使用原始系统提示词。
   * 生成器被完整消费后，本次完整回复才会写入历史。
   */
  async *streamRun(
    inputText: string,
    options: InvokeOptions = {},
  ): AsyncGenerator<string> {
    const messages = this.buildMessages(inputText, false);
    let fullResponse = "";

    for await (const chunk of this.llm.streamInvoke(messages, options)) {
      fullResponse += chunk;
      yield chunk;
    }

    this.recordConversation(inputText, fullResponse);
  }

  /** 组装系统提示词、历史记录和本次用户输入。 */
  private buildMessages(
    inputText: string,
    enhancedSystemPrompt: boolean,
  ): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [];
    const systemPrompt = enhancedSystemPrompt
      ? this.getEnhancedSystemPrompt()
      : this.systemPrompt;

    if (systemPrompt) messages.push({ role: "system", content: systemPrompt });

    for (const message of this.history) {
      // SimpleAgent 自身只会记录 user/assistant；类型断言兼容基类允许的宽角色集合。
      messages.push({
        role: message.role,
        content: message.content,
      } as ChatCompletionMessageParam);
    }
    messages.push({ role: "user", content: inputText });

    return messages;
  }

  /** 将一次完整交互写入历史，确保普通和工具模式使用相同的记录规则。 */
  private recordConversation(inputText: string, response: string): void {
    this.addMessage(new Message(inputText, "user"));
    this.addMessage(new Message(response, "assistant"));
  }

  /** 判断 JSON 解析结果是不是可用的参数对象，并排除数组与 null。 */
  private isParameterDictionary(value: unknown): value is Record<string, unknown> {
    return typeof value === "object" && value !== null && !Array.isArray(value);
  }

  /** 将 JavaScript 的 unknown 异常转换为稳定、可展示的字符串。 */
  private getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
