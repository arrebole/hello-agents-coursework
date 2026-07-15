import { Config } from "./config";
import type { HelloAgentsLLM } from "./llm";
import type { Message } from "./message";

export type AgentRunOptions = Record<string, unknown>;

/** Base class for HelloAgents agents. */
export abstract class Agent {
  readonly name: string;
  readonly llm: HelloAgentsLLM;
  readonly systemPrompt?: string;
  readonly config: Config;

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

  /** Run the agent with the supplied input. */
  abstract run(inputText: string, options?: AgentRunOptions): Promise<string>;

  /** Add a message to the conversation history. */
  addMessage(message: Message): void {
    this.history.push(message);
  }

  /** Remove all messages from the conversation history. */
  clearHistory(): void {
    this.history.length = 0;
  }

  /** Return a shallow copy of the conversation history. */
  getHistory(): Message[] {
    return [...this.history];
  }

  toString(): string {
    return `Agent(name=${this.name}, provider=${this.llm.provider})`;
  }
}
