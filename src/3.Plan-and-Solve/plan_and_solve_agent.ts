import { Executor } from "./executor";
import { LLMClient } from "./llm_client";
import { Planner } from "./planner";


export class PlanAndSolveAgent {
  private llmClient: LLMClient;
  private planner: Planner;
  private executor: Executor;

  /**
   * 初始化智能体，同时创建规划器和执行器实例。
   */
  constructor(llmClient: LLMClient) {
    this.llmClient = llmClient;
    this.planner = new Planner(this.llmClient);
    this.executor = new Executor(this.llmClient);
  }

  /**
   * 运行智能体的完整流程:先规划，后执行。
   */
  async run(question: string): Promise<void> {
    console.log(`\n--- 开始处理问题 ---\n问题: ${question}`);
    
    // 1. 调用规划器生成计划
    const plan = await this.planner.plan(question);
    
    // 检查计划是否成功生成
    if (!plan || plan.length === 0) {
      console.log("\n--- 任务终止 --- \n无法生成有效的行动计划。");
      return;
    }

    // 2. 调用执行器执行计划
    const finalAnswer = await this.executor.execute(question, plan);
    
    console.log(`\n--- 任务完成 ---\n最终答案: ${finalAnswer}`);
  }
}