export type MessageRole = "user" | "assistant" | "system" | "tool";

export interface MessageOptions {
  timestamp?: Date;
  metadata?: Record<string, unknown>;
}

export interface MessageDictionary {
  role: MessageRole;
  content: string;
}

export class Message {
  readonly content: string;
  readonly role: MessageRole;
  readonly timestamp: Date;
  readonly metadata: Record<string, unknown>;

  constructor(content: string, role: MessageRole, options: MessageOptions = {}) {
    this.content = content;
    this.role = role;
    this.timestamp = options.timestamp ?? new Date();
    this.metadata = options.metadata ?? {};
  }

  toDict(): MessageDictionary {
    return {
      role: this.role,
      content: this.content,
    };
  }

  toString(): string {
    return `[${this.role}] ${this.content}`;
  }
}
