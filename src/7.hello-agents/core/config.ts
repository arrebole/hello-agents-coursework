export interface ConfigOptions {
  defaultModel?: string;
  defaultProvider?: string;
  temperature?: number;
  maxTokens?: number;
  debug?: boolean;
  logLevel?: string;
  maxHistoryLength?: number;
}

export interface ConfigDictionary {
  defaultModel: string;
  defaultProvider: string;
  temperature: number;
  maxTokens?: number;
  debug: boolean;
  logLevel: string;
  maxHistoryLength: number;
}

/** HelloAgents runtime configuration. */
export class Config {
  readonly defaultModel: string;
  readonly defaultProvider: string;
  readonly temperature: number;
  readonly maxTokens?: number;
  readonly debug: boolean;
  readonly logLevel: string;
  readonly maxHistoryLength: number;

  constructor(options: ConfigOptions = {}) {
    this.defaultModel = options.defaultModel ?? "gpt-3.5-turbo";
    this.defaultProvider = options.defaultProvider ?? "openai";
    this.temperature = options.temperature ?? 0.7;
    this.maxTokens = options.maxTokens;
    this.debug = options.debug ?? false;
    this.logLevel = options.logLevel ?? "INFO";
    this.maxHistoryLength = options.maxHistoryLength ?? 100;
  }

  /** Create configuration from process environment variables. */
  static fromEnv(): Config {
    const maxTokens = process.env.MAX_TOKENS;

    return new Config({
      debug: (process.env.DEBUG ?? "false").toLowerCase() === "true",
      logLevel: process.env.LOG_LEVEL ?? "INFO",
      temperature: Config.parseNumber(process.env.TEMPERATURE ?? "0.7", "TEMPERATURE"),
      maxTokens: maxTokens ? Config.parseInteger(maxTokens, "MAX_TOKENS") : undefined,
    });
  }

  /** Convert the configuration to a plain object. */
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

  private static parseInteger(value: string, variableName: string): number {
    const parsedValue = Config.parseNumber(value, variableName);
    if (!Number.isInteger(parsedValue)) {
      throw new TypeError(`${variableName} must be an integer`);
    }
    return parsedValue;
  }
}
