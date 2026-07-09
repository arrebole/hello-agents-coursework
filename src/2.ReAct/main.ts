import { ReActAgent } from "./agent"
import { search } from "./search"
import { ToolExecutor } from "./tool_executor"


async function main() {
    const toolExecutor = new ToolExecutor()

    // # 2. 注册我们的实战搜索工具
    const search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    // # 3. 打印可用的工具
    console.log("\n--- 可用的工具 ---")
    console.log(toolExecutor.getAvailableTools())

    // // # 4. 智能体的Action调用，这次我们问一个实时性的问题
    // console.log("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    // const tool_name = "Search"
    // const tool_input = "英伟达最新的GPU型号是什么"

    // const tool_function = toolExecutor.getTool(tool_name)
    // if (tool_function) {
    //     const observation = await tool_function(tool_input)
    //     console.log("--- 观察 (Observation) ---")
    //     console.log(observation)
    // } else {
    //     console.log(`错误:未找到名为 '${tool_name}' 的工具。`)
    // }
    
    const agent = new ReActAgent(toolExecutor, 10)
    await agent.run("英伟达最新的GPU型号是什么")
}

main();