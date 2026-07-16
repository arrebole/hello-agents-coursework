import { HelloAgentsLLM } from "../core/llm";
import { CalculatorTool } from "../tools/builtin/calculator";
import { ToolRegistry } from "../tools/registry";
import { SimpleAgent } from "./simple_agent";

async function main(): Promise<void> {
  // Bun 会在启动时自动加载项目根目录下的 .env 文件。
  const llm = new HelloAgentsLLM();

  // 测试1：基础对话 Agent（无工具）
  console.log("=== 测试1：基础对话 ===");
  const basicAgent = new SimpleAgent(
    "基础助手",
    llm,
    "你是一个友好的AI助手，请用简洁明了的方式回答问题。",
  );

  const response1 = await basicAgent.run("你好，请介绍一下自己");
  console.log(`基础对话响应: ${response1}\n`);

  // 测试2：带工具的 Agent
  console.log("=== 测试2：工具增强对话 ===");
  const toolRegistry = new ToolRegistry();
  const calculator = new CalculatorTool();
  toolRegistry.registerTool(calculator);

  const enhancedAgent = new SimpleAgent(
    "增强助手",
    llm,
    "你是一个智能助手，可以使用工具来帮助用户。",
    undefined,
    toolRegistry,
    true,
  );

  const response2 = await enhancedAgent.run("请帮我计算 15 * 8 + 32");
  console.log(`工具增强响应: ${response2}\n`);

  // 测试3：流式响应
  console.log("=== 测试3：流式响应 ===");
  process.stdout.write("流式响应: ");
  for await (const _chunk of basicAgent.streamRun("请解释什么是人工智能")) {
    // 内容已由 LLM 在 streamRun 中实时打印。
  }

  // 测试4：动态添加工具
  console.log("\n=== 测试4：动态工具管理 ===");
  console.log(`添加工具前: ${basicAgent.hasTools()}`);
  basicAgent.addTool(calculator);
  console.log(`添加工具后: ${basicAgent.hasTools()}`);
  console.log(`可用工具: ${basicAgent.listTools().join(", ")}`);

  // 查看对话历史
  console.log(`\n对话历史: ${basicAgent.getHistory().length} 条消息`);
}

main();
