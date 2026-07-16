import { HelloAgentsLLM } from "../core/llm";
import { CalculatorTool } from "../tools/builtin/calculator";
import { FunctionCallAgent } from "../agents/function_call_agent";

async function main(): Promise<void> {
  // Bun 会在启动时自动加载项目根目录下的 .env 文件。
  const llm = new HelloAgentsLLM({ httpDebug: true });
  const agent = new FunctionCallAgent(
    "函数调用助手",
    llm,
    "你是一个智能助手，可以在需要时调用工具。",
  );

  agent.addTool(new CalculatorTool());

  const response = await agent.run("请帮我计算 15 * 8 + 32");
  console.log(`函数调用响应: ${response}`);

  // 查看对话历史
  console.log(`\n对话历史: ${agent.getHistory().length} 条消息`);
  console.log(agent.getHistory());
}

main();
