import type { Tool } from "./base";

/** 可以直接注册到工具表中的普通函数。 */
export type ToolFunction = (inputText: string) => string;

/** 函数工具除实现外还需要保存描述，以便后续生成提示词。 */
interface RegisteredFunction {
  description: string;
  func: ToolFunction;
}

/** 轻量函数工具可公开读取的声明信息，不暴露其内部存储对象。 */
export interface RegisteredFunctionDefinition {
  name: string;
  description: string;
}

/**
 * HelloAgents 工具注册表。
 *
 * 注册表支持两条互不干扰的注册路径：Tool 对象适合具备参数描述、
 * 可展开能力的完整工具；函数工具适合只接收一个字符串的简单场景。
 * 两类工具分别存放，查找优先级和枚举顺序。
 */
export class ToolRegistry {
  /** 完整 Tool 对象，Map 同时提供按名称查找和稳定的插入顺序。 */
  private readonly tools = new Map<string, Tool>();

  /** 轻量函数工具；与 tools 分开可确保同名时始终优先执行 Tool 对象。 */
  private readonly functions = new Map<string, RegisteredFunction>();

  /**
   * 注册一个 Tool 对象。
   *
   * 默认情况下，可展开工具不会以父工具自身的名称注册，而是将每个
   * 展开后的子工具独立写入注册表。这一点对 Function Calling 很重要，
   * 因为模型看到的是粒度更小、参数更明确的动作。
   */
  registerTool(tool: Tool, autoExpand = true): void {
    const expandedTools =
      autoExpand && tool.expandable ? tool.getExpandedTools() : undefined;

    if (expandedTools?.length) {
      for (const subTool of expandedTools) {
        if (this.tools.has(subTool.name)) {
          console.log(`⚠️ 警告：工具 '${subTool.name}' 已存在，将被覆盖。`);
        }
        this.tools.set(subTool.name, subTool);
      }

      console.log(
        `✅ 工具 '${tool.name}' 已展开为 ${expandedTools.length} 个独立工具`,
      );
      return;
    }

    if (this.tools.has(tool.name)) {
      console.log(`⚠️ 警告：工具 '${tool.name}' 已存在，将被覆盖。`);
    }

    this.tools.set(tool.name, tool);
    console.log(`✅ 工具 '${tool.name}' 已注册。`);
  }

  /** 注册一个接收字符串并返回字符串的轻量函数工具。 */
  registerFunction(
    name: string,
    description: string,
    func: ToolFunction,
  ): void {
    if (this.functions.has(name)) {
      console.log(`⚠️ 警告：工具 '${name}' 已存在，将被覆盖。`);
    }

    this.functions.set(name, { description, func });
    console.log(`✅ 工具 '${name}' 已注册。`);
  }

  /**
   * 注销指定名称的工具，并返回是否确实删除了内容。
   *
   * 当 Tool 对象与函数工具同名时，沿用执行时的优先级，先删除 Tool；
   * 再次调用才会删除同名函数。返回 boolean 可直接兼容 SimpleAgent。
   */
  unregisterTool(name: string): boolean {
    if (this.tools.delete(name)) {
      console.log(`🗑️ 工具 '${name}' 已注销。`);
      return true;
    }

    if (this.functions.delete(name)) {
      console.log(`🗑️ 工具 '${name}' 已注销。`);
      return true;
    }

    console.log(`⚠️ 工具 '${name}' 不存在。`);
    return false;
  }

  /** 获取完整 Tool 对象；函数工具不会由此方法返回。 */
  getTool(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  /** 获取轻量函数实现；未注册时返回 undefined。 */
  getFunction(name: string): ToolFunction | undefined {
    return this.functions.get(name)?.func;
  }

  /**
   * 执行指定工具。
   *
   * 完整 Tool 的 run 接收字典，因此这里把文本包装为 input 参数；
   * 函数工具则直接获得原始文本。工具内部的异常会转换成业务错误文本，
   * 避免单次工具失败中断 Agent 或整条工具链。
   */
  executeTool(name: string, inputText: string): string | Promise<string> {
    const tool = this.tools.get(name);
    if (tool) {
      try {
        const result = tool.run({ input: inputText });
        return result instanceof Promise
          ? result.catch((error) => ToolRegistry.formatExecutionError(name, error))
          : result;
      } catch (error) {
        return ToolRegistry.formatExecutionError(name, error);
      }
    }

    const registeredFunction = this.functions.get(name);
    if (registeredFunction) {
      try {
        return registeredFunction.func(inputText);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return `错误：执行工具 '${name}' 时发生异常: ${message}`;
      }
    }

    return `错误：未找到名为 '${name}' 的工具。`;
  }

  /**
   * 生成可直接拼入系统提示词的工具清单。
   * Tool 对象排在前面，函数工具排在后面。
   */
  getToolsDescription(): string {
    const descriptions: string[] = [];

    for (const tool of this.tools.values()) {
      descriptions.push(`- ${tool.name}: ${tool.description}`);
    }

    for (const [name, info] of this.functions) {
      descriptions.push(`- ${name}: ${info.description}`);
    }

    return descriptions.length > 0 ? descriptions.join("\n") : "暂无可用工具";
  }

  /** 返回全部已注册名称；若两类注册项同名，会保留重复项。 */
  listTools(): string[] {
    return [...this.tools.keys(), ...this.functions.keys()];
  }

  /** 返回所有完整 Tool 对象的数组快照，不包含轻量函数工具。 */
  getAllTools(): Tool[] {
    return [...this.tools.values()];
  }

  /** 返回轻量函数工具的名称与描述快照，供 Function Calling 生成 schema。 */
  getAllFunctionDefinitions(): RegisteredFunctionDefinition[] {
    return [...this.functions].map(([name, info]) => ({
      name,
      description: info.description,
    }));
  }

  /** 清空两类注册项。 */
  clear(): void {
    this.tools.clear();
    this.functions.clear();
    console.log("🧹 所有工具已清空。");
  }

  private static formatExecutionError(name: string, error: unknown): string {
    const message = error instanceof Error ? error.message : String(error);
    return `错误：执行工具 '${name}' 时发生异常: ${message}`;
  }
}

/** 供简单应用直接复用的进程级注册表实例。 */
export const globalRegistry = new ToolRegistry();
