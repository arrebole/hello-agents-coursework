import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

import { Agent } from "../core/agent";
import type { Config } from "../core/config";
import type { HelloAgentsLLM, InvokeOptions } from "../core/llm";
import { Message } from "../core/message";
import type { Tool } from "../tools/base";
import { ToolRegistry } from "../tools/registry";

/** 默认 ReAct 提示词模板。 */
export const DEFAULT_REACT_PROMPT = `你是一个具备推理和行动能力的AI助手。你可以通过思考分析问题，然后调用合适的工具来获取信息，最终给出准确的答案。

## 可用工具
{tools}

## 工作流程
请严格按照以下格式进行回应，每次只能执行一个步骤：

Thought: 分析问题，确定需要什么信息，制定研究策略。
Action: 选择合适的工具获取信息，格式为：
- \`{{tool_name}}[{{tool_input}}]\`：调用工具获取信息。
- \`Finish[研究结论]\`：当你有足够信息得出结论时。

## 重要提醒
1. 每次回应必须包含Thought和Action两部分
2. 工具调用的格式必须严格遵循：工具名[参数]
3. 只有当你确信有足够信息回答问题时，才使用Finish
4. 如果工具返回的信息不够，继续使用其他工具或相同工具的不同参数

## 当前任务
**Question:** {question}

## 执行历史
{history}

现在开始你的推理和行动：`;

/** 单次 ReAct 运行时传给 LLM 的扩展参数。 */
export type ReActRunOptions = InvokeOptions;

/**
 * 结合推理与行动的 Agent。
 *
 * 模型每轮输出一个 Thought 和一个 Action。普通 Action 会调用工具并把观察结果
 * 加入本次执行历史，Finish Action 则结束循环并记录最终对话。
 */
export class ReActAgent extends Agent {
  private readonly toolRegistry: ToolRegistry;
  private readonly maxSteps: number;
  private readonly promptTemplate: string;
  private currentHistory: string[] = [];

  constructor(
    name: string,
    llm: HelloAgentsLLM,
    toolRegistry?: ToolRegistry,
    systemPrompt?: string,
    config?: Config,
    maxSteps = 5,
    customPrompt?: string,
  ) {
    super(name, llm, systemPrompt, config);
    this.toolRegistry = toolRegistry ?? new ToolRegistry();
    this.maxSteps = maxSteps;
    this.promptTemplate = customPrompt || DEFAULT_REACT_PROMPT;
  }

  /** 注册工具；可展开工具由 ToolRegistry 自动注册为多个独立工具。 */
  addTool(tool: Tool, autoExpand = true): void {
    this.toolRegistry.registerTool(tool, autoExpand);
  }

  /** 执行 ReAct 循环，直到模型返回 Finish 或达到最大步数。 */
  async run(
    inputText: string,
    options: ReActRunOptions = {},
  ): Promise<string> {
    this.currentHistory = [];

    console.log(`\n🤖 ${this.name} 开始处理问题: ${inputText}`);

    for (let currentStep = 1; currentStep <= this.maxSteps; currentStep += 1) {
      console.log(`\n--- 第 ${currentStep} 步 ---`);

      const prompt = ReActAgent.renderPrompt(this.promptTemplate, {
        tools: this.toolRegistry.getToolsDescription(),
        question: inputText,
        history: this.currentHistory.join("\n"),
      });
      const messages: ChatCompletionMessageParam[] = [
        { role: "user", content: prompt },
      ];
      const responseText = await this.llm.invoke(messages, options);

      if (!responseText) {
        console.log("❌ 错误：LLM未能返回有效响应。");
        break;
      }

      const [thought, action] = ReActAgent.parseOutput(responseText);
      if (thought) console.log(`🤔 思考: ${thought}`);

      if (!action) {
        console.log("⚠️ 警告：未能解析出有效的Action，流程终止。");
        break;
      }

      if (action.startsWith("Finish")) {
        const finalAnswer = ReActAgent.parseActionInput(action);
        console.log(`🎉 最终答案: ${finalAnswer}`);
        this.recordConversation(inputText, finalAnswer);
        return finalAnswer;
      }

      const [toolName, toolInput] = ReActAgent.parseAction(action);
      if (!toolName || toolInput === undefined) {
        this.currentHistory.push("Observation: 无效的Action格式，请检查。");
        continue;
      }

      console.log(`🎬 行动: ${toolName}[${toolInput}]`);
      const observation = await this.toolRegistry.executeTool(toolName, toolInput);
      console.log(`👀 观察: ${observation}`);

      this.currentHistory.push(`Action: ${action}`);
      this.currentHistory.push(`Observation: ${observation}`);
    }

    console.log("⏰ 已达到最大步数，流程终止。");
    const finalAnswer = "抱歉，我无法在限定步数内完成这个任务。";
    this.recordConversation(inputText, finalAnswer);
    return finalAnswer;
  }

  /** 从模型回复中提取 Thought 和 Action，二者都允许包含多行文本。 */
  private static parseOutput(text: string): [string | undefined, string | undefined] {
    const thoughtMatch = text.match(/Thought:\s*(.*?)(?=\r?\nAction:|$)/s);
    const actionMatch = text.match(/Action:\s*(.*?)\s*$/s);
    return [thoughtMatch?.[1].trim(), actionMatch?.[1].trim()];
  }

  /** 解析 `toolName[input]`，并保留输入中的换行和方括号。 */
  private static parseAction(
    actionText: string,
  ): [string | undefined, string | undefined] {
    const match = actionText.match(/^(\w+)\[([\s\S]*)\]$/);
    return match ? [match[1], match[2]] : [undefined, undefined];
  }

  /** 提取 Finish 或其他 Action 方括号中的内容。 */
  private static parseActionInput(actionText: string): string {
    return actionText.match(/^\w+\[([\s\S]*)\]$/)?.[1] ?? "";
  }

  /**
   * 替换 Python 风格命名占位符，并把双花括号还原为字面量花括号。
   */
  private static renderPrompt(
    template: string,
    values: Record<"tools" | "question" | "history", string>,
  ): string {
    return template.replace(
      /\{\{|\}\}|\{(tools|question|history)\}/g,
      (match, key: keyof typeof values | undefined) => {
        if (match === "{{") return "{";
        if (match === "}}") return "}";
        return key ? values[key] : match;
      },
    );
  }

  /** 统一记录成功回答和步数上限回退结果。 */
  private recordConversation(inputText: string, response: string): void {
    this.addMessage(new Message(inputText, "user"));
    this.addMessage(new Message(response, "assistant"));
  }
}
