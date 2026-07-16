/** 异步工具执行器，为工具注册表提供并发调度、批处理和统一错误格式。 */

import type { ToolRegistry } from "./registry";

export interface ToolTask {
  toolName?: string;
  inputData?: string;
}

export interface ToolExecutionResult {
  taskId: number;
  toolName: string;
  inputData: string;
  result: string;
  status: "success" | "error";
}

interface NormalizedTask {
  taskId: number;
  toolName: string;
  inputData: string;
}

/**
 * 控制并发数量的异步执行器。
 *
 * 这里通过槽位队列限制
 * 同时启动的任务数，并在下一个事件循环 tick 调用注册表，避免调用方当前栈被阻塞。
 */
export class AsyncToolExecutor {
  private readonly waiters: Array<() => void> = [];
  private readonly registry: ToolRegistry;
  private readonly maxWorkers: number;
  private activeTasks = 0;
  private closing = false;
  private closePromise?: Promise<void>;
  private resolveClose?: () => void;

  constructor(
    registry: ToolRegistry,
    maxWorkers = 4,
  ) {
    if (!Number.isInteger(maxWorkers) || maxWorkers < 1) {
      throw new RangeError("maxWorkers 必须是大于 0 的整数");
    }
    this.registry = registry;
    this.maxWorkers = maxWorkers;
  }

  /** 异步执行单个工具；失败会转换为字符串，保持旧接口不会抛出异常的行为。 */
  async executeToolAsync(toolName: string, inputData: string): Promise<string> {
    if (this.closing) return `❌ 工具 '${toolName}' 异步执行失败: 执行器已关闭`;
    await this.acquireSlot();

    try {
      return await new Promise<string>((resolve) => {
        setTimeout(() => {
          try {
            resolve(this.registry.executeTool(toolName, inputData));
          } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            resolve(`❌ 工具 '${toolName}' 异步执行失败: ${message}`);
          }
        }, 0);
      });
    } finally {
      this.releaseSlot();
    }
  }

  /**
   * 并行执行多个工具并按输入顺序返回结果。
   * 缺少工具名的任务与 Python 版本一样被跳过，而不会产生一条错误结果。
   */
  async executeToolsParallel(tasks: ToolTask[]): Promise<ToolExecutionResult[]> {
    console.log(`🚀 开始并行执行 ${tasks.length} 个工具任务`);
    const normalizedTasks: NormalizedTask[] = [];

    tasks.forEach((task, taskId) => {
      const toolName = task.toolName
      if (!toolName) return;
      const inputData = task.inputData ?? "";
      console.log(`📝 创建任务 ${taskId + 1}: ${toolName}`);
      normalizedTasks.push({ taskId, toolName, inputData });
    });

    // 先创建全部 Promise，槽位队列负责执行上限；Promise.all 保证输出顺序稳定。
    const results = await Promise.all(
      normalizedTasks.map(async (task): Promise<ToolExecutionResult> => {
        try {
          const result = await this.executeToolAsync(task.toolName, task.inputData);
          console.log(`✅ 任务 ${task.taskId + 1} 完成: ${task.toolName}`);
          return { ...task, result, status: "success" };
        } catch (error) {
          const message = error instanceof Error ? error.message : String(error);
          console.log(`❌ 任务 ${task.taskId + 1} 失败: ${task.toolName} - ${message}`);
          return { ...task, result: message, status: "error" };
        }
      }),
    );

    const successful = results.filter((result) => result.status === "success").length;
    console.log(`🎉 并行执行完成，成功: ${successful}/${results.length}`);
    return results;
  }

  /** 把一组输入转换为同名工具任务，再复用并行执行逻辑。 */
  executeToolsBatch(toolName: string, inputList: string[]): Promise<ToolExecutionResult[]> {
    return this.executeToolsParallel(
      inputList.map((inputData) => ({ toolName, inputData })),
    );
  }

  /**
   * 停止接收新任务，并等待已经获得执行资格的任务结束。
   * 多次调用 close 会共享同一个 Promise，不会重复关闭或遗漏等待。
   */
  close(): Promise<void> {
    if (this.closePromise) return this.closePromise;
    this.closing = true;
    this.closePromise = new Promise<void>((resolve) => {
      this.resolveClose = resolve;
      if (this.activeTasks === 0) this.finishClose();
    });
    return this.closePromise;
  }

  private async acquireSlot(): Promise<void> {
    if (this.activeTasks < this.maxWorkers) {
      this.activeTasks += 1;
      return;
    }

    // 被唤醒时，释放方已经把原任务占用的槽位直接转交给了当前任务。
    await new Promise<void>((resolve) => this.waiters.push(resolve));
  }

  private releaseSlot(): void {
    const nextWaiter = this.waiters.shift();
    if (nextWaiter) {
      // 保持 activeTasks 不变可消除 close 在等待者恢复前观察到 0 的竞态窗口。
      nextWaiter();
      return;
    }

    this.activeTasks -= 1;
    if (this.closing && this.activeTasks === 0) this.finishClose();
  }

  private finishClose(): void {
    if (!this.resolveClose) return;
    const resolve = this.resolveClose;
    this.resolveClose = undefined;
    console.log("🔒 异步工具执行器已关闭");
    resolve();
  }
}

/** 自动管理执行器生命周期的并行便捷入口。 */
export async function runParallelTools(
  registry: ToolRegistry,
  tasks: ToolTask[],
  maxWorkers = 4,
): Promise<ToolExecutionResult[]> {
  const executor = new AsyncToolExecutor(registry, maxWorkers);
  try {
    return await executor.executeToolsParallel(tasks);
  } finally {
    await executor.close();
  }
}

/** 自动管理执行器生命周期的批量便捷入口。 */
export async function runBatchTool(
  registry: ToolRegistry,
  toolName: string,
  inputList: string[],
  maxWorkers = 4,
): Promise<ToolExecutionResult[]> {
  const executor = new AsyncToolExecutor(registry, maxWorkers);
  try {
    return await executor.executeToolsBatch(toolName, inputList);
  } finally {
    await executor.close();
  }
}

/**
 * 同步环境的兼容入口。
 * 异步工具无法在此入口等待完成，会返回明确错误；maxWorkers 仅用于保持调用签名一致。
 */
export function runParallelToolsSync(
  registry: ToolRegistry,
  tasks: ToolTask[],
  _maxWorkers = 4,
): ToolExecutionResult[] {
  console.log(`🚀 开始同步执行 ${tasks.length} 个工具任务`);
  const results: ToolExecutionResult[] = [];

  tasks.forEach((task, taskId) => {
    const toolName = task.toolName
    if (!toolName) return;
    const inputData = task.inputData ?? "";
    try {
      const result = registry.executeTool(toolName, inputData);
      if (result instanceof Promise) {
        // 防止已经启动的异步工具产生未处理的 rejection。
        void result.catch(() => undefined);
        results.push({
          taskId,
          toolName,
          inputData,
          result: `异步工具 '${toolName}' 不能通过同步入口执行`,
          status: "error",
        });
      } else {
        results.push({ taskId, toolName, inputData, result, status: "success" });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      results.push({ taskId, toolName, inputData, result: message, status: "error" });
    }
  });

  return results;
}

/** 同步批量执行同一个工具。 */
export function runBatchToolSync(
  registry: ToolRegistry,
  toolName: string,
  inputList: string[],
  maxWorkers = 4,
): ToolExecutionResult[] {
  return runParallelToolsSync(
    registry,
    inputList.map((inputData) => ({ toolName, inputData })),
    maxWorkers,
  );
}
