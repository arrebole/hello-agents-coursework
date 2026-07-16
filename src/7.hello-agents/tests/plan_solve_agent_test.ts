import { HelloAgentsLLM } from "../core/llm";
import { CalculatorTool } from "../tools/builtin/calculator";
import { ToolRegistry } from "../tools/registry";
import { PlanAndSolveAgent } from "../agents/plan_solve_agent";

async function main(): Promise<void> {
  // Bun 会在启动时自动加载项目根目录下的 .env 文件。
  const llm = new HelloAgentsLLM({ httpDebug: true });

  // 测试：带工具的 Agent
  console.log("=== 测试2：工具增强对话 ===");
  const toolRegistry = new ToolRegistry();
  const calculator = new CalculatorTool();
  toolRegistry.registerTool(calculator);

  const enhancedAgent = new PlanAndSolveAgent(
    "AI助手",
    llm,
  );

  const response = await enhancedAgent.run("问题: 一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？");
  console.log(`工具增强响应: ${response}\n`);
}

main();
