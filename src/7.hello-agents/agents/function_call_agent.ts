import type {
  ChatCompletionAssistantMessageParam,
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
} from "openai/resources/chat/completions";

import { Agent } from "../core/agent";
import type { Config } from "../core/config";
import type { HelloAgentsLLM, InvokeOptions } from "../core/llm";
import { Message } from "../core/message";
import type { Tool, ToolParameter, ToolParameterType } from "../tools/base";
import { ToolRegistry } from "../tools/registry";

/** 单次运行参数；Agent 参数不会透传给模型接口。 */
export interface FunctionCallRunOptions extends InvokeOptions {
  maxToolIterations?: number;
  toolChoice?: ChatCompletionToolChoiceOption;
}

/** 将工具自定义参数类型限制为 JSON Schema 支持的基础类型。 */
function mapParameterType(parameterType: string): ToolParameterType {
  const normalized = (parameterType || "").toLowerCase();
  if (
    normalized === "string" ||
    normalized === "number" ||
    normalized === "integer" ||
    normalized === "boolean" ||
    normalized === "array" ||
    normalized === "object"
  ) {
    return normalized;
  }
  return "string";
}

/** 基于 OpenAI 原生 Function Calling 协议调用本地工具的 Agent。 */
export class FunctionCallAgent extends Agent {
  protected toolRegistry?: ToolRegistry;
  protected enableToolCalling: boolean;
  readonly defaultToolChoice: ChatCompletionToolChoiceOption;
  readonly maxToolIterations: number;

  constructor(
    name: string,
    llm: HelloAgentsLLM,
    systemPrompt?: string,
    config?: Config,
    toolRegistry?: ToolRegistry,
    enableToolCalling = true,
    defaultToolChoice: ChatCompletionToolChoiceOption = "auto",
    maxToolIterations = 3,
  ) {
    super(name, llm, systemPrompt, config);
    this.toolRegistry = toolRegistry;
    this.enableToolCalling = enableToolCalling && toolRegistry !== undefined;
    this.defaultToolChoice = defaultToolChoice;
    this.maxToolIterations = maxToolIterations;
  }

  /** 构建包含可用工具描述的系统提示词。 */
  protected getSystemPrompt(): string {
    const basePrompt =
      this.systemPrompt ?? "你是一个可靠的AI助理，能够在需要时调用工具完成任务。";

    if (!this.enableToolCalling || !this.toolRegistry) return basePrompt;

    const toolsDescription = this.toolRegistry.getToolsDescription();
    if (!toolsDescription || toolsDescription === "暂无可用工具") return basePrompt;

    return [
      basePrompt,
      "",
      "## 可用工具",
      "当你判断需要外部信息或执行动作时，可以直接通过函数调用使用以下工具：",
      toolsDescription,
      "",
      "请主动决定是否调用工具，合理利用多次调用来获得完备答案。",
    ].join("\n");
  }

  /** 将完整 Tool 与轻量函数统一转换为 OpenAI 工具 schema。 */
  protected buildToolSchemas(): ChatCompletionTool[] {
    if (!this.enableToolCalling || !this.toolRegistry) return [];

    const schemas = this.toolRegistry.getAllTools().map((tool) =>
      FunctionCallAgent.buildToolSchema(tool),
    );

    for (const definition of this.toolRegistry.getAllFunctionDefinitions()) {
      schemas.push({
        type: "function",
        function: {
          name: definition.name,
          description: definition.description,
          parameters: {
            type: "object",
            properties: {
              input: {
                type: "string",
                description: "输入文本",
              },
            },
            required: ["input"],
          },
        },
      });
    }

    return schemas;
  }

  /** 执行原生工具调用循环，直至模型直接回答或达到迭代上限。 */
  async run(
    inputText: string,
    options: FunctionCallRunOptions = {},
  ): Promise<string> {
    const {
      maxToolIterations = this.maxToolIterations,
      toolChoice = this.defaultToolChoice,
      ...invokeOptions
    } = options;
    const messages = this.buildMessages(inputText);
    const toolSchemas = this.buildToolSchemas();

    if (toolSchemas.length === 0) {
      const responseText = await this.llm.invoke(messages, invokeOptions);
      this.recordConversation(inputText, responseText);
      return responseText;
    }

    let currentIteration = 0;
    let finalResponse = "";

    while (currentIteration < maxToolIterations) {
      const response = await this.llm.invokeWithTools(
        messages,
        toolSchemas,
        toolChoice,
        invokeOptions,
      );
      const assistantMessage = response.choices[0]?.message;
      if (!assistantMessage) throw new Error("LLM响应中没有可用的候选消息");

      const content = FunctionCallAgent.extractMessageContent(
        assistantMessage.content,
      );
      const toolCalls = (assistantMessage.tool_calls ?? []).filter(
        FunctionCallAgent.isFunctionToolCall,
      );

      if (toolCalls.length > 0) {
        const assistantPayload: ChatCompletionAssistantMessageParam = {
          role: "assistant",
          content,
          tool_calls: toolCalls,
        };
        messages.push(assistantPayload);

        for (const toolCall of toolCalls) {
          const toolName = toolCall.function.name;
          const parsedArguments = FunctionCallAgent.parseFunctionCallArguments(
            toolCall.function.arguments,
          );
          const result = await this.executeToolCall(toolName, parsedArguments);
          messages.push({
            role: "tool",
            tool_call_id: toolCall.id,
            content: result,
          });
        }

        currentIteration += 1;
        continue;
      }

      finalResponse = content;
      messages.push({ role: "assistant", content: finalResponse });
      break;
    }

    if (currentIteration >= maxToolIterations && !finalResponse) {
      const finalChoice = await this.llm.invokeWithTools(
        messages,
        toolSchemas,
        "none",
        invokeOptions,
      );
      finalResponse = FunctionCallAgent.extractMessageContent(
        finalChoice.choices[0]?.message.content,
      );
    }

    this.recordConversation(inputText, finalResponse);
    return finalResponse;
  }

  /** 按需创建注册表并注册工具；可展开行为由 ToolRegistry 统一处理。 */
  addTool(tool: Tool): void {
    if (!this.toolRegistry) {
      this.toolRegistry = new ToolRegistry();
      this.enableToolCalling = true;
    }
    this.toolRegistry.registerTool(tool);
  }

  /** 移除指定工具，并返回是否确实移除。 */
  removeTool(toolName: string): boolean {
    return this.toolRegistry?.unregisterTool(toolName) ?? false;
  }

  /** 列出当前注册的完整工具和轻量函数工具。 */
  listTools(): string[] {
    return this.toolRegistry?.listTools() ?? [];
  }

  /** 判断原生工具调用是否已启用。 */
  hasTools(): boolean {
    return this.enableToolCalling && this.toolRegistry !== undefined;
  }

  /** 流式调用暂未实现，与原版本一样回退为一次性结果。 */
  async *streamRun(
    inputText: string,
    options: FunctionCallRunOptions = {},
  ): AsyncGenerator<string> {
    yield await this.run(inputText, options);
  }

  private buildMessages(inputText: string): ChatCompletionMessageParam[] {
    const messages: ChatCompletionMessageParam[] = [
      { role: "system", content: this.getSystemPrompt() },
    ];

    for (const message of this.history) {
      messages.push({
        role: message.role,
        content: message.content,
      } as ChatCompletionMessageParam);
    }
    messages.push({ role: "user", content: inputText });
    return messages;
  }

  private static buildToolSchema(tool: Tool): ChatCompletionTool {
    const properties: Record<string, Record<string, unknown>> = {};
    const required: string[] = [];

    let parameters: ToolParameter[];
    try {
      parameters = tool.getParameters();
    } catch {
      parameters = [];
    }

    for (const parameter of parameters) {
      const type = mapParameterType(parameter.type);
      const property: Record<string, unknown> = {
        type,
        description: parameter.description || "",
      };
      if (type === "array") property.items = { type: "string" };
      if (parameter.default !== null && parameter.default !== undefined) {
        property.default = parameter.default;
      }
      properties[parameter.name] = property;
      if (parameter.required) required.push(parameter.name);
    }

    return {
      type: "function",
      function: {
        name: tool.name,
        description: tool.description || "",
        parameters: {
          type: "object",
          properties,
          ...(required.length > 0 ? { required } : {}),
        },
      },
    };
  }

  private static extractMessageContent(rawContent: unknown): string {
    if (rawContent === null || rawContent === undefined) return "";
    if (typeof rawContent === "string") return rawContent;
    if (Array.isArray(rawContent)) {
      return rawContent
        .map((item) => {
          if (typeof item !== "object" || item === null || !("text" in item)) {
            return "";
          }
          return typeof item.text === "string" ? item.text : "";
        })
        .join("");
    }
    return String(rawContent);
  }

  private static parseFunctionCallArguments(
    argumentsText?: string,
  ): Record<string, unknown> {
    if (!argumentsText) return {};

    try {
      const parsed: unknown = JSON.parse(argumentsText);
      return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
        ? (parsed as Record<string, unknown>)
        : {};
    } catch {
      return {};
    }
  }

  private convertParameterTypes(
    toolName: string,
    parameters: Record<string, unknown>,
  ): Record<string, unknown> {
    const tool = this.toolRegistry?.getTool(toolName);
    if (!tool) return parameters;

    let parameterDefinitions;
    try {
      parameterDefinitions = tool.getParameters();
    } catch {
      return parameters;
    }

    const typeMapping = new Map(
      parameterDefinitions.map((parameter) => [parameter.name, parameter.type]),
    );
    const converted: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(parameters)) {
      const parameterType = typeMapping.get(key)?.toLowerCase();

      if (parameterType === "number" || parameterType === "float") {
        const numericValue =
          typeof value === "string" && value.trim() === "" ? Number.NaN : Number(value);
        converted[key] = Number.isNaN(numericValue) ? value : numericValue;
      } else if (parameterType === "integer" || parameterType === "int") {
        const numericValue =
          typeof value === "string" && value.trim() === "" ? Number.NaN : Number(value);
        converted[key] = Number.isInteger(numericValue) ? numericValue : value;
      } else if (parameterType === "boolean" || parameterType === "bool") {
        if (typeof value === "boolean") converted[key] = value;
        else if (typeof value === "string") {
          converted[key] = ["true", "1", "yes"].includes(value.toLowerCase());
        } else converted[key] = Boolean(value);
      } else {
        converted[key] = value;
      }
    }

    return converted;
  }

  private async executeToolCall(
    toolName: string,
    argumentsDictionary: Record<string, unknown>,
  ): Promise<string> {
    if (!this.toolRegistry) return "❌ 错误：未配置工具注册表";

    const tool = this.toolRegistry.getTool(toolName);
    if (tool) {
      try {
        return await tool.run(
          this.convertParameterTypes(toolName, argumentsDictionary),
        );
      } catch (error) {
        return `❌ 工具调用失败：${FunctionCallAgent.getErrorMessage(error)}`;
      }
    }

    const func = this.toolRegistry.getFunction(toolName);
    if (func) {
      try {
        return func(String(argumentsDictionary.input ?? ""));
      } catch (error) {
        return `❌ 工具调用失败：${FunctionCallAgent.getErrorMessage(error)}`;
      }
    }

    return `❌ 错误：未找到工具 '${toolName}'`;
  }

  private static isFunctionToolCall(
    toolCall: { type: string },
  ): toolCall is ChatCompletionMessageFunctionToolCall {
    return toolCall.type === "function";
  }

  private recordConversation(inputText: string, response: string): void {
    this.addMessage(new Message(inputText, "user"));
    this.addMessage(new Message(response, "assistant"));
  }

  private static getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
