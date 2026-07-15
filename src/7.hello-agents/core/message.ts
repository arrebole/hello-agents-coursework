/**
 * 消息在对话中的来源或用途。
 *
 * `system` 用于承载系统级指令，`user` 与 `assistant` 表示对话双方，`tool` 表示工具执行结果。
 * 这里仅记录角色分类；工具调用 ID、函数参数等协议字段不属于这个最小消息模型。
 */
export type MessageRole = "user" | "assistant" | "system" | "tool";

/** 创建消息时可附加的本地上下文，这些字段不会被写入 `toDict()` 的结果。 */
export interface MessageOptions {
  /** 消息时间；省略时由构造函数记录当前时间。 */
  timestamp?: Date;

  /**
   * 供上层 Agent、追踪或业务逻辑保存的任意扩展信息。
   * `Message` 会直接保留传入对象的引用，不会复制、冻结或解释其中的键。
   */
  metadata?: Record<string, unknown>;
}

/**
 * 消息面向传输或序列化的最小投影，只包含模型对话共有的角色与文本内容。
 *
 * 特定模型协议可能要求额外字段，例如工具消息的调用 ID；调用方需要在协议适配层补齐。
 */
export interface MessageDictionary {
  role: MessageRole;
  content: string;
}

/**
 * HelloAgents 内部使用的单条对话消息。
 *
 * 消息正文和角色与本地时间、元数据保存在同一对象中，便于 Agent 维护历史记录；
 * 类本身不校验角色与正文是否匹配，也不执行内容清洗或协议转换。
 */
export class Message {
  /** 原始文本内容；构造时不会被裁剪、转义或规范化。 */
  readonly content: string;

  /** 消息角色，决定上层在组装对话时如何解释这段内容。 */
  readonly role: MessageRole;

  /** 本地记录时间，不会自动参与模型请求。 */
  readonly timestamp: Date;

  /** 本地扩展信息；`readonly` 只禁止替换属性引用，并不会冻结对象内容。 */
  readonly metadata: Record<string, unknown>;

  /**
   * 创建一条消息。
   *
   * `timestamp` 和 `metadata` 均按引用保存；如果调用方之后修改传入的 `Date` 或元数据对象，
   * 当前消息中观察到的值也会随之变化。
   */
  constructor(content: string, role: MessageRole, options: MessageOptions = {}) {
    this.content = content;
    this.role = role;
    this.timestamp = options.timestamp ?? new Date();
    this.metadata = options.metadata ?? {};
  }

  /**
   * 返回新的最小字典对象，供上层继续转换为具体模型协议。
   * 时间戳和元数据会被有意排除，因此该转换不能用于完整还原 `Message`。
   */
  toDict(): MessageDictionary {
    return {
      role: this.role,
      content: this.content,
    };
  }

  /**
   * 生成适合日志与调试的角色前缀文本。
   * 正文保持原样，不会转义换行或隐藏敏感信息，调用方应根据输出场景自行处理。
   */
  toString(): string {
    return `[${this.role}] ${this.content}`;
  }
}
