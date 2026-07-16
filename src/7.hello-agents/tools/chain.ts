import { ToolRegistry } from "./registry";

/** 工具链中的一个顺序执行步骤。 */
export interface ToolChainStep {
  toolName: string;
  inputTemplate: string;
  outputKey: string;
}

/** 单个工具链的只读摘要，适合展示或序列化。 */
export interface ToolChainInfo {
  name: string;
  description: string;
  steps: number;
  stepDetails: ToolChainStep[];
}

/** 模板变量缺失时使用的内部异常，调用方会把它转换为稳定的中文错误。 */
class MissingTemplateVariableError extends Error {
  readonly variableName: string;

  constructor(variableName: string) {
    super(`缺少模板变量: ${variableName}`);
    this.name = "MissingTemplateVariableError";
    this.variableName = variableName;
  }
}

/**
 * 使用上下文替换形如 `{input}` 的模板变量。
 *
 * 双大括号分别表示字面量 `{` 和 `}`，与 Python str.format 的常见用法一致。
 * 工具链目前只承诺支持简单变量名，不解析属性访问、下标或格式化规格。
 */
function renderTemplate(
  template: string,
  context: Record<string, unknown>,
): string {
  const leftBraceToken = "\u0000HELLO_AGENTS_LEFT_BRACE\u0000";
  const rightBraceToken = "\u0000HELLO_AGENTS_RIGHT_BRACE\u0000";

  const escapedTemplate = template
    .replaceAll("{{", leftBraceToken)
    .replaceAll("}}", rightBraceToken);

  const renderedTemplate = escapedTemplate.replace(
    /\{([^{}]+)\}/g,
    (_placeholder, variableName: string) => {
      if (!Object.prototype.hasOwnProperty.call(context, variableName)) {
        throw new MissingTemplateVariableError(variableName);
      }

      return String(context[variableName]);
    },
  );

  return renderedTemplate
    .replaceAll(leftBraceToken, "{")
    .replaceAll(rightBraceToken, "}");
}

/**
 * 支持多个工具顺序执行的工具链。
 *
 * 每一步的结果会写回共享上下文，后续步骤可以通过模板变量引用它；
 * execute 最终返回最后一步结果。若任一步无法继续，则立即返回错误文本。
 */
export class ToolChain {
  readonly name: string;
  readonly description: string;
  readonly steps: ToolChainStep[] = [];

  constructor(name: string, description: string) {
    this.name = name;
    this.description = description;
  }

  /** 添加一个工具执行步骤，并为未指定的结果键生成稳定默认值。 */
  addStep(toolName: string, inputTemplate: string, outputKey?: string): void {
    this.steps.push({
      toolName,
      inputTemplate,
      outputKey: outputKey || `step_${this.steps.length}_result`,
    });
    console.log(`✅ 工具链 '${this.name}' 添加步骤: ${toolName}`);
  }

  /**
   * 从头到尾执行工具链。
   *
   * 传入的 context 会被原地补充 input 和各步骤结果，这与 Python 版本
   * 的可观察行为一致，也方便调用方在执行结束后检查所有中间结果。
   */
  async execute(
    registry: ToolRegistry,
    inputData: string,
    context: Record<string, unknown> = {},
  ): Promise<string> {
    if (this.steps.length === 0) {
      return "❌ 工具链为空，无法执行";
    }

    console.log(`🚀 开始执行工具链: ${this.name}`);
    context.input = inputData;

    let finalResult = inputData;

    for (const [index, step] of this.steps.entries()) {
      console.log(
        `📝 执行步骤 ${index + 1}/${this.steps.length}: ${step.toolName}`,
      );

      let actualInput: string;
      try {
        actualInput = renderTemplate(step.inputTemplate, context);
      } catch (error) {
        if (error instanceof MissingTemplateVariableError) {
          return `❌ 模板变量替换失败: '${error.variableName}'`;
        }
        const message = error instanceof Error ? error.message : String(error);
        return `❌ 模板变量替换失败: ${message}`;
      }

      try {
        const result = await registry.executeTool(step.toolName, actualInput);
        context[step.outputKey] = result;
        finalResult = result;
        console.log(`✅ 步骤 ${index + 1} 完成`);
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        return `❌ 工具 '${step.toolName}' 执行失败: ${message}`;
      }
    }

    console.log(`🎉 工具链 '${this.name}' 执行完成`);
    return finalResult;
  }
}

/** 保存、查询并执行多条命名工具链。 */
export class ToolChainManager {
  private readonly registry: ToolRegistry;
  private readonly chains = new Map<string, ToolChain>();

  constructor(registry: ToolRegistry) {
    this.registry = registry;
  }

  /** 同名链会覆盖旧链，与 Python 字典赋值语义一致。 */
  registerChain(chain: ToolChain): void {
    this.chains.set(chain.name, chain);
    console.log(`✅ 工具链 '${chain.name}' 已注册`);
  }

  /** 按名称执行工具链；不存在时返回业务错误而不是抛出异常。 */
  executeChain(
    chainName: string,
    inputData: string,
    context?: Record<string, unknown>,
  ): Promise<string> {
    const chain = this.chains.get(chainName);
    if (!chain) {
      return Promise.resolve(`❌ 工具链 '${chainName}' 不存在`);
    }

    return chain.execute(this.registry, inputData, context);
  }

  /** 按注册顺序列出全部工具链名称。 */
  listChains(): string[] {
    return [...this.chains.keys()];
  }

  /** 返回工具链信息的快照，避免调用方通过结果对象改写内部步骤。 */
  getChainInfo(chainName: string): ToolChainInfo | undefined {
    const chain = this.chains.get(chainName);
    if (!chain) return undefined;

    return {
      name: chain.name,
      description: chain.description,
      steps: chain.steps.length,
      stepDetails: chain.steps.map((step) => ({ ...step })),
    };
  }
}

/** 创建“搜索后计算”的示例链。 */
export function createResearchChain(): ToolChain {
  const chain = new ToolChain("research_and_calculate", "搜索信息并进行相关计算");

  // 第一步保留原始输入供搜索工具使用，结果可由后续步骤引用。
  chain.addStep("web_search", "{input}", "search_result");

  // 当前示例沿用 Python 版的固定表达式，展示不同工具的串联执行。
  chain.addStep("my_calculator", "2 + 2", "calc_result");

  return chain;
}

/** 创建只包含一个计算步骤的最小工具链示例。 */
export function createSimpleChain(): ToolChain {
  const chain = new ToolChain("simple_demo", "简单的工具链演示");
  chain.addStep("my_calculator", "{input}", "result");
  return chain;
}
