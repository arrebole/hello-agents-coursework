import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

import type { Config } from "../core/config";
import type { HelloAgentsLLM, InvokeOptions } from "../core/llm";
import { Message } from "../core/message";
import { ToolRegistry } from "../tools/registry";
import { SimpleAgent } from "./simple_agent";

/** 从模型文本协议中解析出的一次工具调用。 */
export interface ToolAwareParsedCall {
  toolName: string;
  parameters: string;
  original: string;
}

/** 工具执行完成后发送给监听器的信息。 */
export interface ToolCallInfo {
  agentName: string;
  toolName: string;
  rawParameters: string;
  parsedParameters: Record<string, unknown>;
  result: string;
}

export type ToolCallListener = (callInfo: ToolCallInfo) => void;

/** ToolAwareSimpleAgent 流式运行参数。 */
export interface ToolAwareStreamOptions extends InvokeOptions {
  maxToolIterations?: number;
}

/**
 * 记录工具调用的 SimpleAgent。
 *
 * 除了在每次工具调用后通知监听器之外，本类还支持嵌套数组参数、字符串中的
 * 方括号以及流式响应中的工具调用标记。
 */
export class ToolAwareSimpleAgent extends SimpleAgent {
  private readonly toolCallListener?: ToolCallListener;

  constructor(
    name: string,
    llm: HelloAgentsLLM,
    systemPrompt?: string,
    config?: Config,
    toolRegistry?: ToolRegistry,
    enableToolCalling = true,
    toolCallListener?: ToolCallListener,
  ) {
    super(name, llm, systemPrompt, config, toolRegistry, enableToolCalling);
    this.toolCallListener = toolCallListener;
  }

  /** 执行工具并通知监听器；监听器自身失败不会影响工具调用结果。 */
  protected override async executeToolCall(
    toolName: string,
    parameters: string,
  ): Promise<string> {
    if (!this.toolRegistry) return "❌ 错误：未配置工具注册表";

    let parsedParameters: Record<string, unknown> = {};
    let formattedResult: string;

    try {
      const tool = this.toolRegistry.getTool(toolName);
      if (!tool) return `❌ 错误：未找到工具 '${toolName}'`;

      parsedParameters = ToolAwareSimpleAgent.sanitizeParameters(
        this.parseToolParameters(toolName, parameters),
      );
      const result = await Promise.resolve(tool.run(parsedParameters));
      formattedResult = `🔧 工具 ${toolName} 执行结果：\n${result}`;
    } catch (error) {
      parsedParameters = {};
      formattedResult = `❌ 工具调用失败：${ToolAwareSimpleAgent.getErrorMessage(
        error,
      )}`;
    }

    if (this.toolCallListener) {
      try {
        this.toolCallListener({
          agentName: this.name,
          toolName,
          rawParameters: parameters,
          parsedParameters,
          result: formattedResult,
        });
      } catch (error) {
        console.error("Tool call listener failed", error);
      }
    }

    return formattedResult;
  }

  /** 解析工具调用，并允许参数中包含嵌套方括号或带引号的 `]`。 */
  protected override parseToolCalls(text: string): ToolAwareParsedCall[] {
    const marker = "[TOOL_CALL:";
    const calls: ToolAwareParsedCall[] = [];
    let start = 0;

    while (start < text.length) {
      const begin = text.indexOf(marker, start);
      if (begin < 0) break;

      const toolStart = begin + marker.length;
      const colon = text.indexOf(":", toolStart);
      if (colon < 0) break;

      const end = ToolAwareSimpleAgent.findToolCallEnd(text, begin);
      if (end < 0) break;

      calls.push({
        toolName: text.slice(toolStart, colon).trim(),
        parameters: text.slice(colon + 1, end).trim(),
        original: text.slice(begin, end + 1),
      });
      start = end + 1;
    }

    return calls;
  }

  /**
   * 按顶层逗号拆分 key=value，避免数组、对象或字符串内部的逗号截断参数。
   */
  protected override parseToolParameters(
    toolName: string,
    parameters: string,
  ): Record<string, unknown> {
    if (!parameters.includes("=") || parameters.trim().startsWith("{")) {
      return super.parseToolParameters(toolName, parameters);
    }

    const parameterDictionary: Record<string, unknown> = {};
    for (const pair of ToolAwareSimpleAgent.splitTopLevel(parameters)) {
      const separatorIndex = pair.indexOf("=");
      if (separatorIndex < 0) continue;

      const key = pair.slice(0, separatorIndex).trim();
      if (!key) continue;
      parameterDictionary[key] = pair.slice(separatorIndex + 1).trim();
    }

    const convertedParameters = this.convertParameterTypes(
      toolName,
      parameterDictionary,
    );
    return "action" in convertedParameters
      ? convertedParameters
      : this.inferAction(toolName, convertedParameters);
  }

  /** 在引号和各类括号之外拆分逗号分隔项。 */
  private static splitTopLevel(value: string): string[] {
    const items: string[] = [];
    let start = 0;
    const closingBrackets: string[] = [];
    let quote: "\"" | "'" | undefined;
    let escaped = false;
    const matchingBracket: Record<string, string> = {
      "[": "]",
      "{": "}",
      "(": ")",
    };

    for (let index = 0; index < value.length; index += 1) {
      const character = value[index];

      if (quote) {
        if (escaped) {
          escaped = false;
        } else if (character === "\\") {
          escaped = true;
        } else if (character === quote) {
          quote = undefined;
        }
        continue;
      }

      if (character === "\"" || character === "'") {
        quote = character;
      } else if (matchingBracket[character]) {
        closingBrackets.push(matchingBracket[character]);
      } else if (character === closingBrackets.at(-1)) {
        closingBrackets.pop();
      } else if (character === "," && closingBrackets.length === 0) {
        items.push(value.slice(start, index));
        start = index + 1;
      }
    }

    items.push(value.slice(start));
    return items;
  }

  /** 返回工具调用闭合方括号的位置，找不到时返回 -1。 */
  private static findToolCallEnd(text: string, startIndex: number): number {
    const marker = "[TOOL_CALL:";
    const toolStart = startIndex + marker.length;
    const colon = text.indexOf(":", toolStart);
    if (colon < 0) return -1;

    let depth = 0;
    let quote: "\"" | "'" | undefined;
    let escaped = false;

    for (let position = colon + 1; position < text.length; position += 1) {
      const character = text[position];

      if (quote) {
        if (escaped) {
          escaped = false;
        } else if (character === "\\") {
          escaped = true;
        } else if (character === quote) {
          quote = undefined;
        }
        continue;
      }

      if (character === "\"" || character === "'") {
        quote = character;
      } else if (character === "[") {
        depth += 1;
      } else if (character === "]") {
        if (depth === 0) return position;
        depth -= 1;
      }
    }

    return -1;
  }

  /** 按需为现有 Agent 挂载注册表并开启工具调用。 */
  static attachRegistry(
    agent: ToolAwareSimpleAgent,
    registry?: ToolRegistry,
  ): void {
    if (!registry) return;
    agent.toolRegistry = registry;
    agent.enableToolCalling = true;
  }

  /** 清理模型生成的参数值，并对少量约定字段执行类型归一化。 */
  private static sanitizeParameters(
    parameters: Record<string, unknown>,
  ): Record<string, unknown> {
    const sanitized: Record<string, unknown> = {};

    for (const [key, value] of Object.entries(parameters)) {
      if (typeof value !== "string") {
        sanitized[key] = value;
        continue;
      }

      const normalized = ToolAwareSimpleAgent.normalizeString(value);

      if (key === "task_id") {
        const taskId = Number(normalized);
        if (Number.isInteger(taskId)) {
          sanitized[key] = taskId;
          continue;
        }
      }

      if (key === "tags") {
        const parsedTags = ToolAwareSimpleAgent.coerceSequence(normalized);
        if (parsedTags) {
          sanitized[key] = parsedTags;
          continue;
        }

        if (normalized) {
          sanitized[key] = normalized
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean);
          continue;
        }
      }

      sanitized[key] = normalized;
    }

    return sanitized;
  }

  /** 移除模型偶发产生的多余引号，并补齐未闭合的方括号或圆括号。 */
  private static normalizeString(value: string): string {
    let trimmed = value.trim();

    if (
      trimmed &&
      (trimmed[0] === "\"" || trimmed[0] === "'") &&
      ToolAwareSimpleAgent.countCharacter(trimmed, trimmed[0]) === 1
    ) {
      trimmed = trimmed.slice(1);
    }
    if (
      trimmed &&
      (trimmed.at(-1) === "\"" || trimmed.at(-1) === "'") &&
      ToolAwareSimpleAgent.countCharacter(trimmed, trimmed.at(-1)!) === 1
    ) {
      trimmed = trimmed.slice(0, -1);
    }
    if (
      trimmed.length >= 2 &&
      (trimmed[0] === "\"" || trimmed[0] === "'") &&
      trimmed.at(-1) === trimmed[0]
    ) {
      trimmed = trimmed.slice(1, -1);
    }

    if (trimmed.startsWith("[") && !trimmed.endsWith("]")) {
      trimmed += "]";
    } else if (trimmed.startsWith("(") && !trimmed.endsWith(")")) {
      trimmed += ")";
    }

    return trimmed.trim();
  }

  /**
   * 流式生成回答，隐藏工具调用标记，并在每轮工具执行后继续请求模型。
   */
  override async *streamRun(
    inputText: string,
    options: ToolAwareStreamOptions = {},
  ): AsyncGenerator<string> {
    const { maxToolIterations = 3, ...invokeOptions } = options;
    const messages: ChatCompletionMessageParam[] = [
      { role: "system", content: this.getEnhancedSystemPrompt() },
    ];

    for (const message of this.history) {
      messages.push({
        role: message.role,
        content: message.content,
      } as ChatCompletionMessageParam);
    }
    messages.push({ role: "user", content: inputText });

    const finalSegments: string[] = [];
    let finalResponseText = "";
    let currentIteration = 0;
    const marker = "[TOOL_CALL:";

    while (currentIteration < maxToolIterations) {
      let residual = "";
      const segmentsThisRound: string[] = [];
      const toolCallTexts: string[] = [];

      const processResidual = (finalPass = false): string[] => {
        const readySegments: string[] = [];

        while (true) {
          const start = residual.indexOf(marker);
          if (start < 0) {
            const safeLength = finalPass
              ? residual.length
              : Math.max(0, residual.length - (marker.length - 1));
            if (safeLength > 0) {
              readySegments.push(residual.slice(0, safeLength));
              residual = residual.slice(safeLength);
            }
            break;
          }

          if (start > 0) {
            readySegments.push(residual.slice(0, start));
            residual = residual.slice(start);
            continue;
          }

          const end = ToolAwareSimpleAgent.findToolCallEnd(residual, 0);
          if (end < 0) break;

          toolCallTexts.push(residual.slice(0, end + 1));
          residual = residual.slice(end + 1);
        }

        return readySegments;
      };

      for await (const chunk of this.llm.streamInvoke(messages, invokeOptions)) {
        if (!chunk) continue;
        residual += chunk;

        for (const segment of processResidual()) {
          if (!segment) continue;
          segmentsThisRound.push(segment);
          finalSegments.push(segment);
          yield segment;
        }
      }

      for (const segment of processResidual(true)) {
        if (!segment) continue;
        segmentsThisRound.push(segment);
        finalSegments.push(segment);
        yield segment;
      }

      const cleanResponse = segmentsThisRound.join("");
      const toolCalls = toolCallTexts.flatMap((callText) =>
        this.parseToolCalls(callText),
      );

      if (toolCalls.length > 0) {
        messages.push({ role: "assistant", content: cleanResponse });

        const toolResults: string[] = [];
        for (const call of toolCalls) {
          toolResults.push(
            await this.executeToolCall(call.toolName, call.parameters),
          );
        }

        messages.push({
          role: "user",
          content: `工具执行结果：\n${toolResults.join(
            "\n\n",
          )}\n\n请基于这些结果给出完整的回答。`,
        });
        currentIteration += 1;
        continue;
      }

      finalResponseText = cleanResponse;
      break;
    }

    if (currentIteration >= maxToolIterations && !finalResponseText) {
      const fallbackResponse = await this.llm.invoke(messages, invokeOptions);
      finalSegments.push(fallbackResponse);
      finalResponseText = fallbackResponse;
      yield fallbackResponse;
    }

    const storedResponse = finalResponseText || finalSegments.join("");
    this.addMessage(new Message(inputText, "user"));
    this.addMessage(new Message(storedResponse, "assistant"));
  }

  /** 尝试把 JSON 或 Python 风格的列表文本转换为数组。 */
  private static coerceSequence(value: string): unknown[] | undefined {
    if (!value) return undefined;

    const candidates = [value];
    if (value.startsWith("[") && !value.endsWith("]")) candidates.push(`${value}]`);

    for (const candidate of candidates) {
      try {
        const parsed: unknown = JSON.parse(candidate);
        if (Array.isArray(parsed)) return parsed;
      } catch {
        // 再尝试常见的 Python 单引号列表格式。
      }

      const pythonList = ToolAwareSimpleAgent.parsePythonStringList(candidate);
      if (pythonList) return pythonList;
    }

    return undefined;
  }

  /** 安全解析只包含字符串、数字、布尔值和 null/None 的 Python 风格列表。 */
  private static parsePythonStringList(value: string): unknown[] | undefined {
    if (!value.startsWith("[") || !value.endsWith("]")) return undefined;
    const source = value.slice(1, -1).trim();
    if (!source) return [];

    const items: string[] = [];
    let current = "";
    let quote: "\"" | "'" | undefined;
    let escaped = false;

    for (const character of source) {
      if (quote) {
        current += character;
        if (escaped) {
          escaped = false;
        } else if (character === "\\") {
          escaped = true;
        } else if (character === quote) {
          quote = undefined;
        }
      } else if (character === "\"" || character === "'") {
        quote = character;
        current += character;
      } else if (character === ",") {
        items.push(current.trim());
        current = "";
      } else {
        current += character;
      }
    }

    if (quote) return undefined;
    items.push(current.trim());

    const parsed: unknown[] = [];
    for (const item of items) {
      if (!item) return undefined;

      if (
        item.length >= 2 &&
        (item[0] === "\"" || item[0] === "'") &&
        item.at(-1) === item[0]
      ) {
        parsed.push(ToolAwareSimpleAgent.unescapeQuotedString(item.slice(1, -1)));
      } else if (item === "True" || item === "true") {
        parsed.push(true);
      } else if (item === "False" || item === "false") {
        parsed.push(false);
      } else if (item === "None" || item === "null") {
        parsed.push(null);
      } else if (item.trim() !== "" && Number.isFinite(Number(item))) {
        parsed.push(Number(item));
      } else {
        return undefined;
      }
    }

    return parsed;
  }

  private static unescapeQuotedString(value: string): string {
    return value.replace(/\\([\\"'])/g, "$1");
  }

  private static countCharacter(value: string, character: string): number {
    return [...value].filter((item) => item === character).length;
  }

  private static getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
