import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

import { Agent } from "../core/agent";
import type { Config } from "../core/config";
import type { HelloAgentsLLM, InvokeOptions } from "../core/llm";
import { Message } from "../core/message";

/** Reflection Agent 三个阶段使用的提示词。 */
export interface ReflectionPrompts {
  initial: string;
  reflect: string;
  refine: string;
}

/** 默认提示词模板。 */
export const DEFAULT_PROMPTS: Readonly<ReflectionPrompts> = {
  initial: `
请根据以下要求完成任务：

任务: {task}

请提供一个完整、准确的回答。
`,
  reflect: `
请仔细审查以下回答，并找出可能的问题或改进空间：

# 原始任务:
{task}

# 当前回答:
{content}

请分析这个回答的质量，指出不足之处，并提出具体的改进建议。
如果回答已经很好，请回答"无需改进"。
`,
  refine: `
请根据反馈意见改进你的回答：

# 原始任务:
{task}

# 上一轮回答:
{last_attempt}

# 反馈意见:
{feedback}

请提供一个改进后的回答。
`,
};

/** 单次 Reflection Agent 运行时透传给 LLM 的参数。 */
export type ReflectionRunOptions = InvokeOptions;

/** 一条执行或反思记录。 */
export interface MemoryRecord {
  type: string;
  content: string;
}

/** 存储当前任务执行与反思轨迹的短期记忆。 */
export class Memory {
  private readonly records: MemoryRecord[] = [];

  /** 向记忆末尾追加一条记录。 */
  addRecord(recordType: string, content: string): void {
    this.records.push({ type: recordType, content });
    console.log(`📝 记忆已更新，新增一条 '${recordType}' 记录。`);
  }

  /** 将执行和反思记录格式化为连续的文本轨迹。 */
  getTrajectory(): string {
    const trajectory: string[] = [];

    for (const record of this.records) {
      if (record.type === "execution") {
        trajectory.push(`--- 上一轮尝试 (代码) ---\n${record.content}`);
      } else if (record.type === "reflection") {
        trajectory.push(`--- 评审员反馈 ---\n${record.content}`);
      }
    }

    return trajectory.join("\n\n");
  }

  /** 返回最近一次执行结果；尚未执行时返回空字符串。 */
  getLastExecution(): string {
    for (let index = this.records.length - 1; index >= 0; index -= 1) {
      if (this.records[index].type === "execution") {
        return this.records[index].content;
      }
    }
    return "";
  }
}

/**
 * 通过“执行、反思、优化”循环迭代改进回答的 Agent。
 *
 * 适合代码生成、文档写作和分析报告等需要多轮优化的任务。
 */
export class ReflectionAgent extends Agent {
  readonly maxIterations: number;
  readonly prompts: Readonly<ReflectionPrompts>;
  memory: Memory;

  constructor(
    name: string,
    llm: HelloAgentsLLM,
    systemPrompt?: string,
    config?: Config,
    maxIterations = 3,
    customPrompts?: ReflectionPrompts,
  ) {
    super(name, llm, systemPrompt, config);
    this.maxIterations = maxIterations;
    this.memory = new Memory();
    this.prompts = customPrompts || DEFAULT_PROMPTS;
  }

  /** 执行初始回答，并通过反思与优化循环得到最终结果。 */
  async run(
    inputText: string,
    options: ReflectionRunOptions = {},
  ): Promise<string> {
    console.log(`\n🤖 ${this.name} 开始处理任务: ${inputText}`);

    // 每次运行只保留当前任务的反思轨迹；Agent 会话历史仍持续累积。
    this.memory = new Memory();

    console.log("\n--- 正在进行初始尝试 ---");
    const initialPrompt = renderPrompt(this.prompts.initial, {
      task: inputText,
    });
    const initialResult = await this.getLLMResponse(initialPrompt, options);
    this.memory.addRecord("execution", initialResult);

    for (let index = 0; index < this.maxIterations; index += 1) {
      console.log(`\n--- 第 ${index + 1}/${this.maxIterations} 轮迭代 ---`);

      console.log("\n-> 正在进行反思...");
      const lastResult = this.memory.getLastExecution();
      const reflectPrompt = renderPrompt(this.prompts.reflect, {
        task: inputText,
        content: lastResult,
      });
      const feedback = await this.getLLMResponse(reflectPrompt, options);
      this.memory.addRecord("reflection", feedback);

      if (
        feedback.includes("无需改进") ||
        feedback.toLowerCase().includes("no need for improvement")
      ) {
        console.log("\n✅ 反思认为结果已无需改进，任务完成。");
        break;
      }

      console.log("\n-> 正在进行优化...");
      const refinePrompt = renderPrompt(this.prompts.refine, {
        task: inputText,
        last_attempt: lastResult,
        feedback,
      });
      const refinedResult = await this.getLLMResponse(refinePrompt, options);
      this.memory.addRecord("execution", refinedResult);
    }

    const finalResult = this.memory.getLastExecution();
    console.log(`\n--- 任务完成 ---\n最终结果:\n${finalResult}`);
    this.recordConversation(inputText, finalResult);
    return finalResult;
  }

  /** 调用 LLM 并把缺失响应规范化为空字符串。 */
  private async getLLMResponse(
    prompt: string,
    options: ReflectionRunOptions,
  ): Promise<string> {
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: prompt },
    ];
    return (await this.llm.invoke(messages, options)) || "";
  }

  private recordConversation(inputText: string, response: string): void {
    this.addMessage(new Message(inputText, "user"));
    this.addMessage(new Message(response, "assistant"));
  }
}

/** 替换提示词中的命名占位符；未知占位符保持原样。 */
function renderPrompt(
  template: string,
  values: Record<string, string>,
): string {
  return template.replace(/\{([a-z_]+)\}/gi, (match, key: string) =>
    Object.prototype.hasOwnProperty.call(values, key) ? values[key] : match,
  );
}
