import { ChatCompletionMessageParam } from "openai/resources"
import { LLMClient } from "./llm_client"
import { ToolExecutor } from "./tool_executor"


function genPrompt(tools: string, question: string, history: string) {
    return `
        请注意，你是一个有能力调用外部工具的智能助手。

        可用工具如下:
        ${tools}

        请严格按照以下格式进行回应:

        Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
        Action: 你决定采取的行动，必须是以下格式之一:
        - \`{{tool_name}}[{{tool_input}}]\`: 调用一个可用工具。
        - Finish[最终答案]: 当你认为已经获得最终答案时。
        - 当你收集到足够的信息，能够回答用户的最终问题时，你必须在Action:字段后使用 Finish[最终答案] 来输出最终答案。

        现在，请开始解决以下问题:
        Question: ${question}
        History: ${history}
    `
}

// 智能体
export class ReActAgent {
    private llmClient: LLMClient
    private toolExecutor: ToolExecutor
    private maxSteps: number
    private history: string[]

    public constructor(
        toolExecutor: ToolExecutor,
        maxSteps: number,
    ) {
        this.llmClient = new LLMClient(
            "deepseek-v4-pro",
            process.env['DEEPSEEK_API_KEY']!,
            process.env['DEEPSEEK_API_URL']!,
        )
        this.toolExecutor = toolExecutor
        this.maxSteps = maxSteps
        this.history = []
    }

    private parseOutput(text: string): [string | null, string | null] {
        // Thought: 匹配到 Action: 或文本末尾
        const thoughtMatch = text.match(/Thought:\s*(.*?)(?=\nAction:|$)/s);
        // Action: 匹配到文本末尾
        const actionMatch = text.match(/Action:\s*(.*?)$/s);

        const thought = thoughtMatch ? thoughtMatch[1].trim() : null;
        const action = actionMatch ? actionMatch[1].trim() : null;

        return [thought, action];
    }

    private parseAction(actionText: string): [string | null, string | null] {
        // 解析Action字符串，提取工具名称和输入
        const match = actionText.match(/^(\w+)\[(.*)\]/s);
        if (match) {
            return [match[1], match[2]];
        }
        return [null, null];
    }

    // 运行ReAct智能体来回答一个问题。
    public async run(question: string) {
        // 每次运行时重置历史记录
        this.history = []
        let current_step = 0

        while (current_step < this.maxSteps) {
            current_step++
            console.log(`--- 第 ${current_step} 步 ---`)

            // 格式化提示词
            const toolsDesc = this.toolExecutor.getAvailableTools();
            const historyStr = this.history.join("\n");

            const prompt = genPrompt(
                toolsDesc,
                question,
                historyStr,
            )
            console.log("\n", prompt, "\n");

            // 2. 调用LLM进行思考
            const messages: ChatCompletionMessageParam[] = [
                { "role": "user", "content": prompt }
            ]
            const responseText = await this.llmClient.think(messages)
            console.log("-- LLM 完整返回 --\n", responseText)
 
            if (!responseText) {
                console.log("错误:LLM未能返回有效响应。")
                break
            }

            // 解析LLM的输出
            const [thought, action] = this.parseOutput(responseText);
            if (!!thought) {
                console.log("\n")
                console.log(`匹配到思考: ${thought}`)
            }

            if (!action) {
                console.log("警告:未能解析出有效的Action，流程终止。")
                break
            }

            // 4. 执行Action
            if (action.startsWith("Finish")) {
                // # 如果是Finish指令，提取最终答案并结束
                const finalAnswer = action.match(/^Finish\[(.*)\]/s)?.[1] ?? null;
                console.log("\n")
                console.log(`🎉 最终答案: ${finalAnswer}`)
                return finalAnswer
            }

            const [toolName, toolInput] = this.parseAction(action)
            if (!toolName || !toolInput) {
                // # ... 处理无效Action格式 ...
                continue
            }
            console.log(`\n🎬 行动: ${toolName}[${toolInput}]`)

            let observation = '';
            const toolFunction = this.toolExecutor.getTool(toolName)
            if (!toolFunction) {
                observation = `错误:未找到名为 '${toolName}' 的工具。`
            } else {
                observation = await toolFunction(toolInput)
            }

            console.log(`👀 观察: ${observation}`)

            this.history.push(`Action: ${action}`)
            this.history.push(`Observation: ${observation}`)
        }

        console.log("已达到最大步数，流程终止。")
        return null
    }
}