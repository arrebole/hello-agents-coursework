import { tavily } from "@tavily/core";

// web 查询工具
export async function search(query: string): Promise<string> {
  const api_key = process.env.TAVILY_API_KEY
  if (!api_key) {
    return "错误:未配置TAVILY_API_KEY环境变量。"
  }

  const tvly = tavily({ apiKey: api_key });

  try {
    const response = await tvly.search(query, {
      searchDepth: "basic",
      includeAnswer: true,
    });

    if (response.answer) {
      return response.answer;
    }

    const formatted_results = []
    for (const result of response.results) {
      formatted_results.push(`- ${result.title}: ${result.content}`)
    }

    if (formatted_results.length <= 0) {
      return "抱歉 TAVILY 没有找到相关的结果。"
    }

    return "根据搜索，为您找到以下信息:\n" + formatted_results.join("\n");
  } catch (e) {
    return `错误:执行Tavily搜索时出现问题 - ${e}`
  }
}