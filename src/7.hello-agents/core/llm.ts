import OpenAI from "openai";
import type {
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionMessageParam,
} from "openai/resources/chat/completions";

import { HelloAgentsException } from "./exceptions";

/**
 * 内置支持的服务类型。
 *
 * 所有服务最终都通过 OpenAI 兼容协议访问；该类型主要用于选择环境变量、
 * 默认服务地址和默认模型，而不是切换不同的 SDK 实现。
 */
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
  /** 请求使用的模型标识；未提供时按环境变量和服务类型推导。 */
  model?: string;
  /** 显式 API 密钥，优先级高于对应的环境变量。 */
  apiKey?: string;
  /** OpenAI 兼容接口的根地址，优先级高于环境变量和内置地址。 */
  baseURL?: string;
  /** 服务类型；省略时会根据环境变量、密钥和地址进行启发式识别。 */
  provider?: SupportedProvider;
  /** 实例级默认采样温度，可在单次调用时覆盖。 */
  temperature?: number;
  /** 实例级默认最大生成 token 数，可在非流式调用时覆盖。 */
  maxTokens?: number;
  /** SDK 请求超时，单位为秒。 */
  timeout?: number;
  /** 保留未声明的扩展配置，供上层代码读取。 */
  [key: string]: unknown;
}

/** 单次非流式调用可以覆盖的生成参数及 OpenAI 兼容扩展字段。 */
export interface InvokeOptions {
  temperature?: number;
  maxTokens?: number;
  [key: string]: unknown;
}

/**
 * 面向多种 OpenAI 兼容服务的统一大语言模型客户端。
 *
 * 构造阶段负责确定服务类型、凭据、接口地址与默认模型；调用阶段则把统一的
 * 消息结构交给 OpenAI SDK，并将底层错误收敛为 `HelloAgentsException`。
 */
export class HelloAgentsLLM {
  /** 最终发送给服务端的模型标识。 */
  readonly model: string;
  /** 未被单次调用覆盖时使用的默认采样温度。 */
  readonly temperature: number;
  /** 未被单次调用覆盖时使用的默认生成上限。 */
  readonly maxTokens?: number;
  /** 以秒保存的请求超时；传给 OpenAI SDK 时会换算为毫秒。 */
  readonly timeout: number;
  /** 构造参数中未被本类消费的扩展配置；当前不会自动附加到请求。 */
  readonly kwargs: Record<string, unknown>;
  /** 解析后的服务类型，决定凭据、地址和模型的默认值。 */
  readonly provider: SupportedProvider;
  /** 按配置优先级解析后实际交给 SDK 的 API 密钥。 */
  readonly apiKey: string;
  /** 按配置优先级解析后实际交给 SDK 的 OpenAI 兼容接口地址。 */
  readonly baseURL: string;

  /** 底层 SDK 仅在类内使用，避免调用方绕过统一配置与错误边界。 */
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
    // 对外统一使用秒；`LLM_TIMEOUT` 无法解析为整数时回退到 60 秒。
    this.timeout = timeout ?? this.readIntegerEnvironmentVariable("LLM_TIMEOUT", 60);
    this.kwargs = kwargs;

    // 显式 provider 优先；省略时才进入按环境、密钥和地址识别的流程。
    const requestedProvider = provider?.toLowerCase() as SupportedProvider | undefined;
    this.provider = requestedProvider ?? this.autoDetectProvider(apiKey, baseURL);

    let resolvedApiKey: string | undefined;
    let resolvedBaseURL: string | undefined;
    if (this.provider === "custom") {
      // custom 不套用任何厂商默认值，调用方必须提供两项通用连接配置。
      resolvedApiKey = apiKey ?? process.env.LLM_API_KEY;
      resolvedBaseURL = baseURL ?? process.env.LLM_BASE_URL;
    } else {
      [resolvedApiKey, resolvedBaseURL] = this.resolveCredentials(apiKey, baseURL);
    }

    // 模型配置优先级：构造参数 > 通用环境变量 > 当前 provider 的默认模型。
    this.model = model ?? process.env.LLM_MODEL_ID ?? this.getDefaultModel();

    // 在创建 SDK 客户端前集中校验，避免把缺失配置延迟成难定位的网络错误。
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

  /**
   * 发起流式请求，并按服务端返回顺序逐段产出文本。
   *
   * 每个有效片段既会写入标准输出，也会通过异步生成器交给调用方；空增量
   * （例如只携带角色或结束原因的 chunk）会被忽略。底层异常会在记录日志后
   * 统一转换为 `HelloAgentsException`，因此调用方无需依赖具体 SDK 的错误类型。
   */
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
        // OpenAI 兼容流可能返回没有 choices 或没有文本增量的控制类 chunk。
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

  /**
   * 发起非流式请求，并返回第一条候选结果的完整文本。
   *
   * `options` 中的扩展字段会直接传给兼容接口，但本类管理的 model、messages、
   * temperature、max_tokens 和 stream 会在展开后写入，因此不能被扩展字段覆盖。
   * 服务端没有返回第一条文本候选时使用空字符串，保持返回类型始终为 string。
   */
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

  /**
   * 为旧调用方式保留的流式别名。
   *
   * 当前委托接口只把 `temperature` 传给 `think`；`maxTokens` 和其他扩展字段
   * 不会进入流式请求，这是为了保持该兼容入口的既有行为。
   */
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
    // 第一阶段优先识别厂商专用环境变量；检查顺序同时也是冲突时的优先级。
    if (process.env.OPENAI_API_KEY) return "openai";
    if (process.env.DEEPSEEK_API_KEY) return "deepseek";
    if (process.env.DASHSCOPE_API_KEY) return "qwen";
    if (process.env.MODELSCOPE_API_KEY) return "modelscope";
    if (process.env.KIMI_API_KEY || process.env.MOONSHOT_API_KEY) return "kimi";
    if (process.env.ZHIPU_API_KEY || process.env.GLM_API_KEY) return "zhipu";
    if (process.env.OLLAMA_API_KEY || process.env.OLLAMA_HOST) return "ollama";
    if (process.env.VLLM_API_KEY || process.env.VLLM_HOST) return "vllm";

    // 第二阶段使用密钥形态识别部分服务和本地部署约定。
    const actualApiKey = apiKey ?? process.env.LLM_API_KEY;
    if (actualApiKey) {
      const lowerApiKey = actualApiKey.toLowerCase();
      if (actualApiKey.startsWith("ms-")) return "modelscope";
      if (lowerApiKey === "ollama") return "ollama";
      if (lowerApiKey === "vllm") return "vllm";
      if (lowerApiKey === "local") return "local";
      // 智谱密钥通常包含点号；同时排除常见的长 `sk-` 厂商密钥形态。
      if (
        !(actualApiKey.startsWith("sk-") && actualApiKey.length > 50) &&
        (actualApiKey.endsWith(".") || actualApiKey.slice(-20).includes("."))
      ) {
        return "zhipu";
      }
    }

    const actualBaseURL = baseURL ?? process.env.LLM_BASE_URL;
    if (!actualBaseURL) return "auto";

    // 第三阶段根据已知域名和本地端口推断服务；未知远程地址保持 auto。
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
    // 各分支均遵循：显式参数 > 厂商环境变量 > 通用环境变量 > 内置默认值。
    // 云服务只为地址提供默认值；本地服务同时提供 SDK 所需的占位密钥。
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
        // 无法归属到已知服务时不猜测地址，要求使用通用连接配置。
        return [apiKey ?? process.env.LLM_API_KEY, baseURL ?? process.env.LLM_BASE_URL];
    }
  }

  /** 返回各服务的教学示例默认模型，调用方可通过参数或环境变量覆盖。 */
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
        // auto 模式只能依据通用环境变量继续推断；无法识别时使用通用默认值。
        return this.inferDefaultModelFromBaseURL(process.env.LLM_BASE_URL ?? "");
    }
  }

  /** 根据通用接口地址的域名或端口，为 auto 模式选择一个默认模型。 */
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

  /** 读取十进制整数环境变量；缺失或无法解析时返回给定回退值。 */
  private readIntegerEnvironmentVariable(name: string, fallback: number): number {
    const value = process.env[name];
    if (value === undefined) return fallback;

    const parsedValue = Number.parseInt(value, 10);
    return Number.isNaN(parsedValue) ? fallback : parsedValue;
  }

  /** 将 unknown 异常安全地归一化为可用于日志和领域异常的文本。 */
  private getErrorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
  }
}
