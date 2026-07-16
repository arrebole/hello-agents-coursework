import type { ChatCompletionMessageParam } from "openai/resources/chat/completions";

import { Agent } from "../core/agent";
import type { Config } from "../core/config";
import type { HelloAgentsLLM, InvokeOptions } from "../core/llm";
import { Message } from "../core/message";

/** 默认规划器提示词模板。 */
export const DEFAULT_PLANNER_PROMPT = `
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划:
\`\`\`python
["步骤1", "步骤2", "步骤3", ...]
\`\`\`
`;

/** 默认执行器提示词模板。 */
export const DEFAULT_EXECUTOR_PROMPT = `
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决"当前步骤"，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对"当前步骤"的回答:
`;

/** 单次规划或执行时透传给 LLM 的参数。 */
export type PlanAndSolveRunOptions = InvokeOptions;

/** 可分别覆盖规划器与执行器的提示词。 */
export interface PlanAndSolvePrompts {
  planner?: string;
  executor?: string;
}

/** 负责把复杂问题拆分为有序步骤。 */
export class Planner {
  private readonly llmClient: HelloAgentsLLM;
  private readonly promptTemplate: string;

  constructor(llmClient: HelloAgentsLLM, promptTemplate?: string) {
    this.llmClient = llmClient;
    this.promptTemplate = promptTemplate || DEFAULT_PLANNER_PROMPT;
  }

  /** 请求模型生成计划，并从代码块中解析字符串数组。 */
  async plan(
    question: string,
    options: PlanAndSolveRunOptions = {},
  ): Promise<string[]> {
    const prompt = renderPrompt(this.promptTemplate, { question });
    const messages: ChatCompletionMessageParam[] = [
      { role: "user", content: prompt },
    ];

    console.log("--- 正在生成计划 ---");
    const responseText = await this.llmClient.invoke(messages, options);
    console.log(`✅ 计划已生成:\n${responseText}`);

    try {
      const planSource = extractPlanSource(responseText);
      return parseStringList(planSource);
    } catch (error) {
      console.log(`❌ 解析计划时出错: ${getErrorMessage(error)}`);
      console.log(`原始响应: ${responseText}`);
      return [];
    }
  }
}

/** 负责按顺序执行规划器生成的步骤。 */
export class Executor {
  private readonly llmClient: HelloAgentsLLM;
  private readonly promptTemplate: string;

  constructor(llmClient: HelloAgentsLLM, promptTemplate?: string) {
    this.llmClient = llmClient;
    this.promptTemplate = promptTemplate || DEFAULT_EXECUTOR_PROMPT;
  }

  /** 逐步执行计划，并将最后一步的结果作为最终答案。 */
  async execute(
    question: string,
    plan: string[],
    options: PlanAndSolveRunOptions = {},
  ): Promise<string> {
    let history = "";
    let finalAnswer = "";

    console.log("\n--- 正在执行计划 ---");
    for (const [index, step] of plan.entries()) {
      const stepNumber = index + 1;
      console.log(`\n-> 正在执行步骤 ${stepNumber}/${plan.length}: ${step}`);

      const prompt = renderPrompt(this.promptTemplate, {
        question,
        plan: JSON.stringify(plan),
        history: history || "无",
        current_step: step,
      });
      const messages: ChatCompletionMessageParam[] = [
        { role: "user", content: prompt },
      ];

      const responseText = await this.llmClient.invoke(messages, options);
      history += `步骤 ${stepNumber}: ${step}\n结果: ${responseText}\n\n`;
      finalAnswer = responseText;
      console.log(`✅ 步骤 ${stepNumber} 已完成，结果: ${finalAnswer}`);
    }

    return finalAnswer;
  }
}

/**
 * 先规划、再逐步执行的 Agent，适合多步骤推理、数学问题与复杂分析任务。
 */
export class PlanAndSolveAgent extends Agent {
  readonly planner: Planner;
  readonly executor: Executor;

  constructor(
    name: string,
    llm: HelloAgentsLLM,
    systemPrompt?: string,
    config?: Config,
    customPrompts?: PlanAndSolvePrompts,
  ) {
    super(name, llm, systemPrompt, config);
    this.planner = new Planner(this.llm, customPrompts?.planner);
    this.executor = new Executor(this.llm, customPrompts?.executor);
  }

  /** 生成并执行计划，同时把本次问答写入会话历史。 */
  async run(
    inputText: string,
    options: PlanAndSolveRunOptions = {},
  ): Promise<string> {
    console.log(`\n🤖 ${this.name} 开始处理问题: ${inputText}`);

    const plan = await this.planner.plan(inputText, options);
    if (plan.length === 0) {
      const finalAnswer = "无法生成有效的行动计划，任务终止。";
      console.log(`\n--- 任务终止 ---\n${finalAnswer}`);
      this.recordConversation(inputText, finalAnswer);
      return finalAnswer;
    }

    const finalAnswer = await this.executor.execute(inputText, plan, options);
    console.log(`\n--- 任务完成 ---\n最终答案: ${finalAnswer}`);
    this.recordConversation(inputText, finalAnswer);
    return finalAnswer;
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

/** 提取带语言标记的代码块，也兼容模型直接返回数组。 */
function extractPlanSource(responseText: string): string {
  const fencedPlan = responseText.match(
    /```(?:python|json)?\s*([\s\S]*?)\s*```/i,
  );
  const planSource = (fencedPlan?.[1] ?? responseText).trim();

  if (!planSource.startsWith("[") || !planSource.endsWith("]")) {
    throw new SyntaxError("未找到有效的计划列表");
  }
  return planSource;
}

/**
 * 安全解析 JSON 或 Python 风格的字符串列表，不使用 eval 执行模型输出。
 */
function parseStringList(source: string): string[] {
  try {
    const parsed: unknown = JSON.parse(source);
    if (Array.isArray(parsed) && parsed.every((item) => typeof item === "string")) {
      return parsed;
    }
    throw new TypeError("计划必须是字符串数组");
  } catch (error) {
    if (!(error instanceof SyntaxError)) throw error;
  }

  const items: string[] = [];
  let index = 1;

  const skipWhitespace = (): void => {
    while (/\s/.test(source[index] ?? "")) index += 1;
  };

  skipWhitespace();
  if (source[index] === "]") return items;

  while (index < source.length - 1) {
    skipWhitespace();
    const quote = source[index];
    if (quote !== "'" && quote !== '"') {
      throw new SyntaxError("计划中的每个步骤都必须是字符串");
    }
    index += 1;

    let item = "";
    let closed = false;
    while (index < source.length) {
      const character = source[index];
      index += 1;

      if (character === quote) {
        closed = true;
        break;
      }
      if (character !== "\\") {
        item += character;
        continue;
      }

      if (index >= source.length) throw new SyntaxError("字符串转义不完整");
      const escaped = source[index];
      index += 1;
      const escapeCharacters: Record<string, string> = {
        "0": "\0",
        b: "\b",
        f: "\f",
        n: "\n",
        r: "\r",
        t: "\t",
        v: "\v",
      };
      item += escapeCharacters[escaped] ?? escaped;
    }

    if (!closed) throw new SyntaxError("计划步骤缺少结束引号");
    items.push(item);
    skipWhitespace();

    if (source[index] === "]") return items;
    if (source[index] !== ",") throw new SyntaxError("计划步骤之间缺少逗号");
    index += 1;
    skipWhitespace();

    // Python 列表允许最后一个元素后保留逗号。
    if (source[index] === "]") return items;
  }

  throw new SyntaxError("计划列表格式无效");
}

function getErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
