import { ChatCompletionMessageParam } from "openai/resources";
import { LLMClient } from "./llm_client";

const EXECUTOR_PROMPT_TEMPLATE = `
你是一位顶级的AI执行专家。你的任务是严格按照给定的计划，一步步地解决问题。
你将收到原始问题、完整的计划、以及到目前为止已经完成的步骤和结果。
请你专注于解决“当前步骤”，并仅输出该步骤的最终答案，不要输出任何额外的解释或对话。

# 原始问题:
{question}

# 完整计划:
{plan}

# 历史步骤与结果:
{history}

# 当前步骤:
{current_step}

请仅输出针对“当前步骤”的回答:
`


export class Executor {
    private llmClient: LLMClient;

    constructor(llmClient: LLMClient) {
        this.llmClient = llmClient;
    }

    /**
     * 根据计划，逐步执行并解决问题。
     */
    async execute(question: string, plan: string[]): Promise<string> {
        // 用于存储历史步骤和结果的字符串
        let history = "";
        let responseText = "";

        console.log("\n--- 正在执行计划 ---");

        for (let i = 0; i < plan.length; i++) {
            const step = plan[i];
            const stepNumber = i + 1;

            console.log(`\n-> 正在执行步骤 ${stepNumber}/${plan.length}: ${step}`);

            const prompt = EXECUTOR_PROMPT_TEMPLATE
                .replace(/\{question\}/g, question)
                .replace(/\{plan\}/g, JSON.stringify(plan))
                .replace(/\{history\}/g, history || "无") // 如果是第一步，则历史为空
                .replace(/\{current_step\}/g, step);

            const messages: ChatCompletionMessageParam[] = [
                { role: "user", content: prompt }
            ];

            responseText = (await this.llmClient.think(messages)) || "";

            // 更新历史记录，为下一步做准备
            history += `步骤 ${stepNumber}: ${step}\n结果: ${responseText}\n\n`;

            console.log(`✅ 步骤 ${stepNumber} 已完成，结果: ${responseText}`);
        }

        // 循环结束后，最后一步的响应就是最终答案
        const finalAnswer = responseText;
        return finalAnswer;
    }
}