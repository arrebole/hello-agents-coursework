import { LLMClient } from "./llm_client";
import { PlanAndSolveAgent } from "./plan_and_solve_agent";


async function main() {
    const llmClient = new LLMClient(
        "deepseek-v4-pro",
        process.env['DEEPSEEK_API_KEY']!,
        process.env['DEEPSEEK_API_URL']!,
    )
    const agent = new PlanAndSolveAgent(llmClient)
    await agent.run("问题: 一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？")
}

main();