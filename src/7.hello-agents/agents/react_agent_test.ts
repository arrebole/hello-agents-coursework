import { HelloAgentsLLM } from "../core/llm";
import { WebSearchTool } from "../tools/builtin/web_search";
import { ToolRegistry } from "../tools/registry";
import { ReActAgent } from "./react_agent";

async function main(): Promise<void> {
  // Bun 会在启动时自动加载项目根目录下的 .env 文件。
  const llm = new HelloAgentsLLM();

  // 测试：带工具的 Agent
  console.log("=== 测试：ReAct Agent ===");
  const toolRegistry = new ToolRegistry();
  const webSearch = new WebSearchTool();
  toolRegistry.registerTool(webSearch);

  const agent = new ReActAgent(
    "增强助手",
    llm,
    toolRegistry,
    "你是一个智能助手，可以使用工具来帮助用户。",
  );

  const response = await agent.run("今天杭州的天气怎么样？");
  console.log(`工具增强响应: ${response}\n`);
}

main();
