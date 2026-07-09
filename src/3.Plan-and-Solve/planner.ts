import { ChatCompletionMessageParam } from "openai/resources";
import { LLMClient } from "./llm_client";

const PLANNER_PROMPT_TEMPLATE: string = `
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个json数组，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,\`\`\`json与\`\`\`作为前后缀是必要的:
\`\`\`json
["步骤1", "步骤2", "步骤3", ...]
\`\`\`

`;

export class Planner {
    private llmClient: LLMClient;

    constructor(llmClient: LLMClient) {
        this.llmClient = llmClient;
    }

    /**
     * 根据用户问题生成一个行动计划。
     */
    async plan(question: string): Promise<string[]> {
        const prompt = PLANNER_PROMPT_TEMPLATE.replace(/\{question\}/g, question);

        // 为了生成计划，我们构建一个简单的消息列表
        const messages: ChatCompletionMessageParam[] = [
            { role: "user", content: prompt }
        ];

        console.log("--- 正在生成计划 ---");
        // 使用流式输出来获取完整的计划
        const responseText = (await this.llmClient.think(messages)) || "";

        console.log(`✅ 计划已生成:\n${responseText}`);

        // 解析LLM输出的列表字符串
        try {
            // 找到```python和```之间的内容
            const planMatch = responseText.match(/```json\s*([\s\S]*?)\s*```/);
            if (!planMatch) {
                console.log("❌ 未找到Python代码块");
                return [];
            }

            const planStr = planMatch[1].trim();
            // 使用安全的方式解析字符串为数组
            const plan = this.parseStringList(planStr);
            return Array.isArray(plan) ? plan : [];
        } catch (error) {
            console.log(`❌ 解析计划时出错: ${error}`);
            console.log(`原始响应: ${responseText}`);
            return [];
        }
    }

    /**
     * 安全地将字符串解析为字符串数组
     * 支持格式: ['item1', 'item2'] 或 ["item1", "item2"]
     */
    private parseStringList(str: string): string[] {
        // 移除两端空白
        str = str.trim();

        // 检查是否是数组格式
        if (!str.startsWith('[') || !str.endsWith(']')) {
            throw new Error('不是有效的数组格式');
        }

        // 移除方括号并分割
        const content = str.slice(1, -1).trim();
        if (content === '') {
            return [];
        }

        // 解析字符串列表（支持单引号和双引号）
        const items: string[] = [];
        let current = '';
        let inQuotes = false;
        let quoteChar = '';
        let escapeNext = false;

        for (let i = 0; i < content.length; i++) {
            const char = content[i];

            // 处理转义字符
            if (escapeNext) {
                current += char;
                escapeNext = false;
                continue;
            }

            if (char === '\\') {
                escapeNext = true;
                continue;
            }

            if (!inQuotes && (char === "'" || char === '"')) {
                inQuotes = true;
                quoteChar = char;
                continue;
            }

            if (inQuotes && char === quoteChar) {
                inQuotes = false;
                items.push(current);
                current = '';
                // 跳过逗号和空格
                while (i + 1 < content.length && (content[i + 1] === ',' || content[i + 1] === ' ')) {
                    i++;
                }
                continue;
            }

            if (inQuotes) {
                current += char;
            }
        }

        return items;
    }
}
