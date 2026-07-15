import { Config } from "./config";
import type { HelloAgentsLLM } from "./llm";
import type { Message } from "./message";

/**
 * 单次运行的扩展参数。
 *
 * 基类不会解析这些参数，具体支持哪些键及其含义由各 Agent 子类约定。
 */
export type AgentRunOptions = Record<string, unknown>;

/**
 * HelloAgents 中所有 Agent 的抽象基类。
 *
 * 该类只负责保存 Agent 的身份、模型客户端、系统提示词、运行配置与会话历史；
 * 如何组织提示词、调用模型以及处理返回结果，均由子类的 `run` 方法实现。
 */
export abstract class Agent {
  /** Agent 的可读名称，主要用于标识实例。 */
  readonly name: string;

  /** 由外部注入的统一模型客户端，子类通过它完成实际的模型调用。 */
  readonly llm: HelloAgentsLLM;

  /** 可选的系统提示词；基类只保存它，不会自动写入会话历史。 */
  readonly systemPrompt?: string;

  /** Agent 运行配置；未显式传入时，每个实例都会获得一份新的默认配置。 */
  readonly config: Config;

  /**
   * 由基类维护的内部会话历史。
   *
   * 使用 `protected` 允许子类读取和编排上下文，同时避免调用方直接替换数组。
   * 基类不会根据 `maxHistoryLength` 自动裁剪历史，相关策略应由具体 Agent 决定。
   */
  protected readonly history: Message[] = [];

  constructor(
    name: string,
    llm: HelloAgentsLLM,
    systemPrompt?: string,
    config: Config = new Config(),
  ) {
    this.name = name;
    this.llm = llm;
    this.systemPrompt = systemPrompt;
    this.config = config;
  }

  /**
   * 使用给定输入执行一次 Agent。
   *
   * 子类需要自行决定是否把输入与输出写入历史，以及如何使用运行扩展参数。
   */
  abstract run(inputText: string, options?: AgentRunOptions): Promise<string>;

  /**
   * 将一条消息追加到会话历史末尾。
   *
   * 此处只负责追加，不做去重、角色校验或长度裁剪。
   */
  addMessage(message: Message): void {
    this.history.push(message);
  }

  /** 清空当前实例的全部会话历史，并保留原数组引用供子类继续使用。 */
  clearHistory(): void {
    this.history.length = 0;
  }

  /**
   * 返回会话历史数组的浅拷贝。
   *
   * 调用方增删返回数组不会影响内部历史，但其中的 `Message` 对象仍是同一实例。
   */
  getHistory(): Message[] {
    return [...this.history];
  }

  /** 返回便于日志与调试展示的 Agent 摘要，不包含提示词或会话内容。 */
  toString(): string {
    return `Agent(name=${this.name}, provider=${this.llm.provider})`;
  }
}
