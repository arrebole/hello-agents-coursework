/**
 * 创建 Config 时允许传入的局部配置。
 *
 * 所有字段均为可选项；未提供的值会由 Config 构造函数补齐默认值。
 */
export interface ConfigOptions {
  defaultModel?: string;
  defaultProvider?: string;
  temperature?: number;
  maxTokens?: number;
  debug?: boolean;
  logLevel?: string;
  maxHistoryLength?: number;
}

/**
 * Config 序列化后的普通对象结构。
 *
 * 除 maxTokens 本身允许缺省外，其余字段在 Config 实例化时都已完成默认值解析，
 * 因此这里使用必填字段表达“可直接消费的完整配置”。
 */
export interface ConfigDictionary {
  defaultModel: string;
  defaultProvider: string;
  temperature: number;
  maxTokens?: number;
  debug: boolean;
  logLevel: string;
  maxHistoryLength: number;
}

/**
 * HelloAgents 的运行时配置快照。
 *
 * 配置在构造时一次性解析完成，属性通过 readonly 在 TypeScript 类型层禁止重新赋值，
 * 适合作为同一个配置快照在多个组件之间共享。
 */
export class Config {
  readonly defaultModel: string;
  readonly defaultProvider: string;
  readonly temperature: number;
  readonly maxTokens?: number;
  readonly debug: boolean;
  readonly logLevel: string;
  readonly maxHistoryLength: number;

  constructor(options: ConfigOptions = {}) {
    // 使用空值合并而非逻辑或，使 temperature=0、debug=false 等有效的假值能够被保留。
    this.defaultModel = options.defaultModel ?? "gpt-3.5-turbo";
    this.defaultProvider = options.defaultProvider ?? "openai";
    this.temperature = options.temperature ?? 0.7;
    this.maxTokens = options.maxTokens;
    this.debug = options.debug ?? false;
    this.logLevel = options.logLevel ?? "INFO";
    this.maxHistoryLength = options.maxHistoryLength ?? 100;
  }

  /**
   * 从进程环境变量创建配置。
   *
   * 这里只读取运行时常用的调试、日志与生成参数；模型、提供商和历史长度仍沿用
   * 构造函数默认值。DEBUG 仅在忽略大小写后严格等于 "true" 时启用。
   * 无效的数值不会静默回退，而是抛出 TypeError，避免带着错误配置继续运行。
   */
  static fromEnv(): Config {
    const maxTokens = process.env.MAX_TOKENS;

    return new Config({
      debug: (process.env.DEBUG ?? "false").toLowerCase() === "true",
      logLevel: process.env.LOG_LEVEL ?? "INFO",
      temperature: Config.parseNumber(process.env.TEMPERATURE ?? "0.7", "TEMPERATURE"),
      maxTokens: maxTokens ? Config.parseInteger(maxTokens, "MAX_TOKENS") : undefined,
    });
  }

  /**
   * 转换为不含类行为的普通对象，便于记录日志、序列化或传递给只接收数据的调用方。
   * 返回的是新对象，修改它不会反向影响当前 Config 实例。
   */
  toDict(): ConfigDictionary {
    return {
      defaultModel: this.defaultModel,
      defaultProvider: this.defaultProvider,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      debug: this.debug,
      logLevel: this.logLevel,
      maxHistoryLength: this.maxHistoryLength,
    };
  }

  /**
   * 按 JavaScript Number 语义解析环境变量，同时显式拒绝空字符串与 NaN。
   * variableName 仅用于生成能直接定位错误来源的异常信息。
   */
  private static parseNumber(value: string, variableName: string): number {
    if (value.trim() === "") {
      throw new TypeError(`${variableName} must be a number`);
    }

    const parsedValue = Number(value);
    if (Number.isNaN(parsedValue)) {
      throw new TypeError(`${variableName} must be a number`);
    }
    return parsedValue;
  }

  /** 在通用数字校验的基础上进一步要求结果为整数。 */
  private static parseInteger(value: string, variableName: string): number {
    const parsedValue = Config.parseNumber(value, variableName);
    if (!Number.isInteger(parsedValue)) {
      throw new TypeError(`${variableName} must be an integer`);
    }
    return parsedValue;
  }
}
