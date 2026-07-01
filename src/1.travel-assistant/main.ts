import OpenAI from 'openai';
import { tavily } from "@tavily/core";
import { ChatCompletionMessageParam } from 'openai/resources/index.mjs';

const AGENT_SYSTEM_PROMPT = `
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- get_weather(city: str): 查询指定城市的实时天气。
- get_attraction(city: str, weather: str): 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
`


// 根据城市获取当前天气
// 通过调用 wttr.in 获取结果
async function get_weather(city: string): Promise<string> {
  const url = `https://wttr.in/${encodeURIComponent(city)}?format=j1`;

  try {
    const response = await fetch(url);
    if (!response.ok) {
      return `错误:天气服务返回状态码 ${response.status}`;
    }

    const data: any = await response.json();
    const currentCondition = data.current_condition?.[0];
    const weatherDesc = currentCondition?.weatherDesc?.[0]?.value;
    const tempC = currentCondition?.temp_C;

    if (!currentCondition || !weatherDesc || tempC === undefined) {
      return "错误:天气数据格式异常";
    }

    return `${city}当前天气: ${weatherDesc}，气温${tempC}摄氏度`;
  } catch (e) {
    return `错误:查询天气时遇到网络问题 - ${e}`;
  }
}

// 通过城市和天气获取可以玩的景点
// 通过调用 www.tavily.com 的 ai 服务实现
async function get_attraction(city: string, weather: string): Promise<string> {
  const api_key = process.env['TAVILY_API_KEY']
  if (!api_key) {
    return "错误:未配置TAVILY_API_KEY环境变量。"
  }

  const tvly = tavily({ apiKey: api_key });

  // 构造一个精确的查询
  const query = `${city}' 在'${weather}'天气下最值得去的旅游景点推荐及理由`

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
      return "抱歉，没有找到相关的旅游景点推荐。"
    }

    return "根据搜索，为您找到以下信息:\n" + formatted_results.join("\n");
  } catch (e) {
    return `错误:执行Tavily搜索时出现问题 - ${e}`
  }
}

// 定义大模型能够调用的工具函数
// ReAct 解析出的参数为按名称索引的对象，故工具统一接收 kwargs 并按名取值
type AvailableTools = Record<string, (kwargs: Record<string, string>) => Promise<string>>;
const available_tools: AvailableTools = {
  "get_weather": ({ city }) => get_weather(city),
  "get_attraction": ({ city, weather }) => get_attraction(city, weather),
}

// 调用大模型
class OpenAICompatibleClient {
  private model: string
  private client: OpenAI

  constructor(model: string, api_key: string, base_url: string) {
    this.model = model;
    this.client = new OpenAI({
      baseURL: base_url,
      apiKey: api_key,
    });
  }

  public async generate(prompt: string, system_prompt: string): Promise<string> {
    try {
      const messages: ChatCompletionMessageParam[] = [
        { 'role': 'system', 'content': system_prompt },
        { 'role': 'user', 'content': prompt }
      ];

      const response = await this.client.chat.completions.create({
        model: this.model,
        messages: messages,
        stream: false,
      })

      return response.choices[0].message.content!
    } catch (e) {
      return `错误:调用语言模型服务时出错。${e}`
    }
  }
}

async function main() {

  const config = {
    MODEL : "deepseek-v4-pro",
    BASE_URL : process.env['DEEPSEEK_API_URL']!,
    API_KEY : process.env['DEEPSEEK_API_KEY']!,
  }
  const llm = new OpenAICompatibleClient(config.MODEL, config.API_KEY, config.BASE_URL)

  const user_prompt = "你好，请帮我查询一下今天北京的天气，然后根据天气推荐一个合适的旅游景点。"
  const prompt_history = [`用户请求: ${user_prompt}`]
  console.log(`用户输入: ${user_prompt}\n`)

  // --- 3. 运行主循环 ---
  for (let i = 0; i < 5; i++) {
    console.log(`--- 循环 ${i+1} ---\n`)

    // 构建 Prompt, 将历史信息都一次打包
    const full_prompt = prompt_history.join("\n");

    // 调用LLM进行思考
    let llm_output = await llm.generate(full_prompt, AGENT_SYSTEM_PROMPT)

    // 模型可能会输出多余的Thought-Action，需要截断
    const match = llm_output.match(/(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|$)/s);
    if (match) {
      const truncated = match[1].trim();
      if (truncated !== llm_output.trim()) {
        llm_output = truncated;
        console.log("已截断多余的 Thought-Action 对");
      }
    }
    console.log(`模型输出:\n${llm_output}\n`);
    prompt_history.push(llm_output);

    // 判断模型是否按标准输出
    const actionMatch = llm_output.match(/Action: (.*)/s);
    if (!actionMatch) {
      const observation = "错误: 未能解析到 Action 字段。请确保你的回复严格遵循 'Thought: ... Action: ...' 的格式。";
      const observationStr = `Observation: ${observation}`;

      console.log(`${observationStr}\n${"=".repeat(40)}`);
      prompt_history.push(observationStr);
      continue;
    }

    const actionStr = actionMatch[1].trim();

    // 判断该回复是否是最终答案，如果是最终答案则退出。
    if (actionStr.startsWith("Finish")) {
      const finalAnswer = actionStr.match(/Finish\[(.*)\]/)?.[1];
      console.log(`任务完成，最终答案: ${finalAnswer}`);
      break;
    }

    // 判断调用工具
    const toolName = actionStr.match(/(\w+)\(/)?.[1];
    const argsStr = actionStr.match(/\((.*)\)/)?.[1] ?? "";
    const kwargs = Object.fromEntries(
      [...argsStr.matchAll(/(\w+)="([^"]*)"/g)].map((match) => [
        match[1],
        match[2],
      ])
    );

    let observation = "";
    if (toolName && toolName in available_tools) {
      observation = await available_tools[toolName](kwargs);
    } else {
      observation = `错误:未定义的工具 '${toolName}'`;
    }

    // 记录观察结果
    const observationStr = `Observation: ${observation}`;
    console.log(`${observationStr}\n${"=".repeat(40)}`);
    prompt_history.push(observationStr);
  }
}

main()