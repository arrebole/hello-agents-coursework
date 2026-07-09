import { LLMClient } from "./llm_client";
import { ReflectionAgent } from "./reflection_agent";


async function main() {
    const llmClient = new LLMClient(
        "deepseek-v4-pro",
        process.env['DEEPSEEK_API_KEY']!,
        process.env['DEEPSEEK_API_URL']!,
    )
    const agent = new ReflectionAgent(llmClient)
    await agent.run("任务： 编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")
}

main();