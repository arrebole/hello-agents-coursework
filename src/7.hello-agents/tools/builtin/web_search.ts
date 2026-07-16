import { tavily } from "@tavily/core";

import { Tool, ToolParameter } from "../base";

/** 使用 Tavily 查询互联网的标准工具。 */
export class WebSearchTool extends Tool {
  private readonly apiKey?: string;

  constructor(apiKey = process.env.TAVILY_API_KEY) {
    super(
      "web_search",
      "使用 Tavily 搜索互联网，获取与查询关键词相关的最新信息。",
    );
    this.apiKey = apiKey;
  }

  async run(parameters: Record<string, unknown>): Promise<string> {
    // input 兼容 ReAct 和 ToolRegistry 的纯文本调用协议。
    const query = String(parameters.query || parameters.input || "").trim();
    if (!query) return "错误：搜索关键词不能为空。";
    if (!this.apiKey) return "错误：未配置 TAVILY_API_KEY 环境变量。";

    const client = tavily({ apiKey: this.apiKey });

    try {
      const response = await client.search(query, {
        searchDepth: "basic",
        includeAnswer: true,
      });

      if (response.answer) return response.answer;

      const formattedResults = response.results.map(
        (result) => `- ${result.title}: ${result.content}`,
      );
      if (formattedResults.length === 0) {
        return "抱歉，Tavily 没有找到相关结果。";
      }

      return `根据搜索，为您找到以下信息:\n${formattedResults.join("\n")}`;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      return `错误：执行 Tavily 搜索时出现问题 - ${message}`;
    }
  }

  getParameters(): ToolParameter[] {
    return [
      new ToolParameter({
        name: "query",
        type: "string",
        description: "要搜索的关键词或问题",
        required: true,
      }),
    ];
  }
}

/** 无需显式创建工具实例的便捷搜索入口。 */
export async function webSearch(query: string): Promise<string> {
  return new WebSearchTool().run({ query });
}
