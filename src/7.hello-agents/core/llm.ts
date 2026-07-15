import OpenAI from "openai";
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";

import { HelloAgentsException } from "./exceptions";

export type SupportedProvider =
  | "openai"
  | "deepseek"
  | "qwen"
  | "modelscope"
  | "kimi"
  | "zhipu"
  | "ollama"
  | "vllm"
  | "local"
  | "auto"
  | "custom";

export interface HelloAgentsLLMOptions {
  model?: string;
  apiKey?: string;
  baseURL?: string;
  provider?: SupportedProvider;
  temperature?: number;
  maxTokens?: number;
  timeout?: number;
  [key: string]: unknown;
}

export interface InvokeOptions {
  temperature?: number;
  maxTokens?: number;
  [key: string]: unknown;
}

/** A unified client for services compatible with the OpenAI API. */
export class HelloAgentsLLM {
  readonly model: string;
  readonly temperature: number;
  readonly maxTokens?: number;
  readonly timeout: number;
  readonly kwargs: Record<string, unknown>;
  readonly provider: SupportedProvider;
  readonly apiKey: string;
  readonly baseURL: string;

  private readonly client: OpenAI;

  constructor(options: HelloAgentsLLMOptions = {}) {
    const {
      model,
      apiKey,
      baseURL,
      provider,
      temperature = 0.7,
      maxTokens,
      timeout,
      ...kwargs
    } = options;

    this.temperature = temperature;
    this.maxTokens = maxTokens;
    this.timeout = timeout ?? this.readIntegerEnvironmentVariable("LLM_TIMEOUT", 60);
    this.kwargs = kwargs;

    const requestedProvider = provider?.toLowerCase() as SupportedProvider | undefined;
    this.provider = requestedProvider ?? this.autoDetectProvider(apiKey, baseURL);

    let resolvedApiKey: string | undefined;
    let resolvedBaseURL: string | undefined;
    if (this.provider === "custom") {
      resolvedApiKey = apiKey ?? process.env.LLM_API_KEY;
      resolvedBaseURL = baseURL ?? process.env.LLM_BASE_URL;
    } else {
      [resolvedApiKey, resolvedBaseURL] = this.resolveCredentials(apiKey, baseURL);
    }

    this.model = model ?? process.env.LLM_MODEL_ID ?? this.getDefaultModel();

    if (!resolvedApiKey || !resolvedBaseURL) {
      throw new HelloAgentsException(
        "API密钥和服务地址必须被提供或在.env文件中定义。",
      );
    }

    this.apiKey = resolvedApiKey;
    this.baseURL = resolvedBaseURL;
    this.client = new OpenAI({
      apiKey: this.apiKey,
      baseURL: this.baseURL,
      timeout: this.timeout * 1_000,
    });
  }

  /** Stream the model response one text fragment at a time. */
  async *think(
    messages: ChatCompletionMessageParam[],
    temperature?: number,
  ): AsyncGenerator<string> {
    console.log(`正在调用 ${this.model} 模型...`);

    try {
      const request: ChatCompletionCreateParamsStreaming = {
        model: this.model,
        messages,
        temperature: temperature ?? this.temperature,
        max_tokens: this.maxTokens,
        stream: true,
      };
      const response = await this.client.chat.completions.create(request);

      console.log("大语言模型响应成功:");
      for await (const chunk of response) {
        const content = chunk.choices[0]?.delta.content ?? "";
        if (content) {
          process.stdout.write(content);
          yield content;
        }
      }
      process.stdout.write("\n");
    } catch (error) {
      const message = this.getErrorMessage(error);
      console.error(`调用LLM API时发生错误: ${message}`);
      throw new HelloAgentsException(`LLM调用失败: ${message}`);
    }
  }

  /** Make a non-streaming request and return the complete response text. */
  async invoke(
    messages: ChatCompletionMessageParam[],
    options: InvokeOptions = {},
  ): Promise<string> {
    const {
      temperature = this.temperature,
      maxTokens = this.maxTokens,
      ...additionalOptions
    } = options;

    try {
      const request = {
        ...additionalOptions,
        model: this.model,
        messages,
        temperature,
        max_tokens: maxTokens,
        stream: false,
      } as ChatCompletionCreateParamsNonStreaming;
      const response = await this.client.chat.completions.create(request);

      return response.choices[0]?.message.content ?? "";
    } catch (error) {
      throw new HelloAgentsException(`LLM调用失败: ${this.getErrorMessage(error)}`);
    }
  }

  /** Backward-compatible streaming alias for think(). */
  streamInvoke(
    messages: ChatCompletionMessageParam[],
    options: InvokeOptions = {},
  ): AsyncGenerator<string> {
    return this.think(messages, options.temperature);
  }

  private autoDetectProvider(
    apiKey?: string,
    baseURL?: string,
  ): SupportedProvider {
    if (process.env.OPENAI_API_KEY) return "openai";
    if (process.env.DEEPSEEK_API_KEY) return "deepseek";
    if (process.env.DASHSCOPE_API_KEY) return "qwen";
    if (process.env.MODELSCOPE_API_KEY) return "modelscope";
    if (process.env.KIMI_API_KEY || process.env.MOONSHOT_API_KEY) return "kimi";
    if (process.env.ZHIPU_API_KEY || process.env.GLM_API_KEY) return "zhipu";
    if (process.env.OLLAMA_API_KEY || process.env.OLLAMA_HOST) return "ollama";
    if (process.env.VLLM_API_KEY || process.env.VLLM_HOST) return "vllm";

    const actualApiKey = apiKey ?? process.env.LLM_API_KEY;
    if (actualApiKey) {
      const lowerApiKey = actualApiKey.toLowerCase();
      if (actualApiKey.startsWith("ms-")) return "modelscope";
      if (lowerApiKey === "ollama") return "ollama";
      if (lowerApiKey === "vllm") return "vllm";
      if (lowerApiKey === "local") return "local";
      if (
        !(actualApiKey.startsWith("sk-") && actualApiKey.length > 50) &&
        (actualApiKey.endsWith(".") || actualApiKey.slice(-20).includes("."))
      ) {
        return "zhipu";
      }
    }

    const actualBaseURL = baseURL ?? process.env.LLM_BASE_URL;
    if (!actualBaseURL) return "auto";

    const lowerBaseURL = actualBaseURL.toLowerCase();
    if (lowerBaseURL.includes("api.openai.com")) return "openai";
    if (lowerBaseURL.includes("api.deepseek.com")) return "deepseek";
    if (lowerBaseURL.includes("dashscope.aliyuncs.com")) return "qwen";
    if (lowerBaseURL.includes("api-inference.modelscope.cn")) return "modelscope";
    if (lowerBaseURL.includes("api.moonshot.cn")) return "kimi";
    if (lowerBaseURL.includes("open.bigmodel.cn")) return "zhipu";

    if (lowerBaseURL.includes("localhost") || lowerBaseURL.includes("127.0.0.1")) {
      if (lowerBaseURL.includes(":11434") || lowerBaseURL.includes("ollama")) {
        return "ollama";
      }
      if (lowerBaseURL.includes(":8000") && lowerBaseURL.includes("vllm")) {
        return "vllm";
      }
      if (lowerBaseURL.includes(":8080") || lowerBaseURL.includes(":7860")) {
        return "local";
      }
      if (actualApiKey?.toLowerCase() === "ollama") return "ollama";
      if (actualApiKey?.toLowerCase() === "vllm") return "vllm";
      return "local";
    }

    if ([":8080", ":7860", ":5000"].some((port) => lowerBaseURL.includes(port))) {
      return "local";
    }

    return "auto";
  }

  private resolveCredentials(
    apiKey?: string,
    baseURL?: string,
  ): [string | undefined, string | undefined] {
    switch (this.provider) {
      case "openai":
        return [
          apiKey ?? process.env.OPENAI_API_KEY ?? process.env.LLM_API_KEY,
          baseURL ?? process.env.LLM_BASE_URL ?? "https://api.openai.com/v1",
        ];
      case "deepseek":
        return [
          apiKey ?? process.env.DEEPSEEK_API_KEY ?? process.env.LLM_API_KEY,
          baseURL ?? process.env.LLM_BASE_URL ?? "https://api.deepseek.com",
        ];
      case "qwen":
        return [
          apiKey ?? process.env.DASHSCOPE_API_KEY ?? process.env.LLM_API_KEY,
          baseURL ??
            process.env.LLM_BASE_URL ??
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ];
      case "modelscope":
        return [
          apiKey ?? process.env.MODELSCOPE_API_KEY ?? process.env.LLM_API_KEY,
          baseURL ?? process.env.LLM_BASE_URL ?? "https://api-inference.modelscope.cn/v1/",
        ];
      case "kimi":
        return [
          apiKey ??
            process.env.KIMI_API_KEY ??
            process.env.MOONSHOT_API_KEY ??
            process.env.LLM_API_KEY,
          baseURL ?? process.env.LLM_BASE_URL ?? "https://api.moonshot.cn/v1",
        ];
      case "zhipu":
        return [
          apiKey ??
            process.env.ZHIPU_API_KEY ??
            process.env.GLM_API_KEY ??
            process.env.LLM_API_KEY,
          baseURL ?? process.env.LLM_BASE_URL ?? "https://open.bigmodel.cn/api/paas/v4",
        ];
      case "ollama":
        return [
          apiKey ?? process.env.OLLAMA_API_KEY ?? process.env.LLM_API_KEY ?? "ollama",
          baseURL ??
            process.env.OLLAMA_HOST ??
            process.env.LLM_BASE_URL ??
            "http://localhost:11434/v1",
        ];
      case "vllm":
        return [
          apiKey ?? process.env.VLLM_API_KEY ?? process.env.LLM_API_KEY ?? "vllm",
          baseURL ??
            process.env.VLLM_HOST ??
            process.env.LLM_BASE_URL ??
            "http://localhost:8000/v1",
        ];
      case "local":
        return [
          apiKey ?? process.env.LLM_API_KEY ?? "local",
          baseURL ?? process.env.LLM_BASE_URL ?? "http://localhost:8000/v1",
        ];
      case "custom":
      case "auto":
        return [apiKey ?? process.env.LLM_API_KEY, baseURL ?? process.env.LLM_BASE_URL];
    }
  }

  private getDefaultModel(): string {
    switch (this.provider) {
      case "openai":
        return "gpt-3.5-turbo";
      case "deepseek":
        return "deepseek-chat";
      case "qwen":
        return "qwen-plus";
      case "modelscope":
        return "Qwen/Qwen2.5-72B-Instruct";
      case "kimi":
        return "moonshot-v1-8k";
      case "zhipu":
        return "glm-4";
      case "ollama":
        return "llama3.2";
      case "vllm":
        return "meta-llama/Llama-2-7b-chat-hf";
      case "local":
        return "local-model";
      case "custom":
        return "gpt-3.5-turbo";
      case "auto":
        return this.inferDefaultModelFromBaseURL(process.env.LLM_BASE_URL ?? "");
    }
  }

  private inferDefaultModelFromBaseURL(baseURL: string): string {
    const lowerBaseURL = baseURL.toLowerCase();
    if (lowerBaseURL.includes("modelscope")) return "Qwen/Qwen2.5-72B-Instruct";
    if (lowerBaseURL.includes("deepseek")) return "deepseek-chat";
    if (lowerBaseURL.includes("dashscope")) return "qwen-plus";
    if (lowerBaseURL.includes("moonshot")) return "moonshot-v1-8k";
    if (lowerBaseURL.includes("bigmodel")) return "glm-4";
    if (lowerBaseURL.includes("ollama") || lowerBaseURL.includes(":11434")) {
      return "llama3.2";
    }
    if (lowerBaseURL.includes(":8000") || lowerBaseURL.includes("vllm")) {
      return "meta-llama/Llama-2-7b-chat-hf";
    }
    if (lowerBaseURL.includes("localhost") || lowerBaseURL.includes("127.0.0.1")) {
      return "local-model";
    }
    return "gpt-3.5-turbo";
  }

  private readIntegerEnvironmentVariable(name: string, fallback: number): number {
    const value = process.env[name];
    if (value === undefined) return fallback;

    const parsedValue = Number.parseInt(value, 10);
    return Number.isNaN(parsedValue) ? fallback : parsedValue;
  }

  private getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
