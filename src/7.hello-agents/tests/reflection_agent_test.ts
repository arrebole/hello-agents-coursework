import { HelloAgentsLLM } from "../core/llm";
import { CalculatorTool } from "../tools/builtin/calculator";
import { ToolRegistry } from "../tools/registry";
import { ReflectionAgent } from "../agents/reflection_agent";

async function main(): Promise<void> {
  // Bun 会在启动时自动加载项目根目录下的 .env 文件。
  const llm = new HelloAgentsLLM({ httpDebug: true });

  // 测试：带工具的 Agent
  console.log("=== 测试：工具增强对话 ===");
  const toolRegistry = new ToolRegistry();
  const calculator = new CalculatorTool();
  toolRegistry.registerTool(calculator);

  const agent = new ReflectionAgent(
    "AI增强助手",
    llm,
  );

  const response = await agent.run("任务： 编写一个Python函数，计算斐波那契数列");
  console.log(`工具增强响应: ${response}\n`);
}

main();
