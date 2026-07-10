import { AIMessage, BaseMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { Annotation, END, MemorySaver, START, StateGraph } from "@langchain/langgraph";
import { ChatOpenAI } from "@langchain/openai";
import { tavily, TavilySearchResponse } from "@tavily/core";

// 初始化模型
const llm = new ChatOpenAI({
    model: "deepseek-v4-pro",
    apiKey: process.env['DEEPSEEK_API_KEY']!,
    configuration: {
        baseURL: process.env['DEEPSEEK_API_URL'],
    },
    temperature: 0.7,
})
// # 初始化Tavily客户端
const tavilyClient = tavily({ apiKey: process.env['TAVILY_API_KEY'] });

// 使用 Annotation.Root 来定义状态
export const SearchState = Annotation.Root({
    // 使用 'messages' 通道来存储历史消息，reducer 用于处理更新
    messages: Annotation<BaseMessage[]>({
        reducer: (a, b) => a.concat(b),
        default: () => [],
    }),
    // 其他状态字段，每个字段都需要定义 reducer 和 default
    userQuery: Annotation<string | undefined>({
        reducer: (a, b) => b ?? a,
        default: () => undefined,
    }),
    searchQuery: Annotation<string | undefined>({
        reducer: (a, b) => b ?? a,
        default: () => undefined,
    }),
    searchResults: Annotation<string | undefined>({
        reducer: (a, b) => b ?? a,
        default: () => undefined,
    }),
    finalAnswer: Annotation<string | undefined>({
        reducer: (a, b) => b ?? a,
        default: () => undefined,
    }),
    step: Annotation<
        'understood' | 'searched' | 'search_failed' | 'completed' | undefined
    >({
        reducer: (a, b) => b ?? a,
        default: () => undefined,
    }),
});
type SearchStateType = typeof SearchState.State;


// ====================================================================

const UNDERSTAND_PROMPT = `
分析用户的查询："{userMessage}"
请完成两个任务：
1. 简洁总结用户想要了解什么
2. 生成最适合搜索引擎的关键词（中英文均可，要精准）

格式：
理解：[用户需求总结]
搜索词：[最佳搜索关键词]
`

/**
 * 步骤1：理解用户查询并生成搜索关键词
 */
export async function understandQueryNode(state: SearchStateType): Promise<Partial<SearchStateType>> {
    // 获取用户最新消息内容
    const userMessage = state.messages[state.messages.length - 1].content! as string;

    // 构建提示词
    const understandPrompt = UNDERSTAND_PROMPT.replace(/\{userMessage\}/g, userMessage);

    // 调用 LLM
    const response = await llm.invoke([
        new SystemMessage(understandPrompt)
    ]);
    const responseText = response.content! as string;

    // 解析 LLM 输出，提取搜索关键词
    let searchQuery = userMessage; // 默认使用原始查询
    if (responseText.includes("搜索词：")) {
        const parts = responseText.split("搜索词：");
        if (parts.length > 1) {
            searchQuery = parts[1].trim();
        }
    }

    // 返回状态更新
    return {
        userQuery: responseText,
        searchQuery: searchQuery,
        step: "understood",
        messages: [new AIMessage(`我将为您搜索：${searchQuery}`)]
    };
}

// ============================================================================
/**
 * 步骤2：使用Tavily API进行真实搜索
 */
export async function tavilySearchNode(state: SearchStateType): Promise<Partial<SearchStateType>> {
    const searchQuery = state.searchQuery || '';

    try {
        console.log(`🔍 正在搜索: ${searchQuery}`);

        // 调用 Tavily API
        const response = await tavilyClient.search(searchQuery, {
            searchDepth: 'basic',
            maxResults: 5,
            includeAnswer: true,
        });

        // 处理和格式化搜索结果
        const searchResults = formatSearchResults(response);

        return {
            searchResults: searchResults,
            step: 'searched',
            messages: [new AIMessage('✅ 搜索完成！正在整理答案...')],
        };
    } catch (error) {
        // 处理错误
        const errorMessage = error instanceof Error ? error.message : '未知错误';
        console.error('搜索失败:', error);

        return {
            searchResults: `搜索失败：${errorMessage}`,
            step: 'search_failed',
            messages: [new AIMessage('❌ 搜索遇到问题...')],
        };
    }
}

/**
 * 格式化 Tavily 搜索结果
 */
function formatSearchResults(response: TavilySearchResponse): string {
    let formattedResults = '';

    // 如果有答案，先添加答案
    if (response.answer) {
        formattedResults += `答案：${response.answer}\n\n`;
    }

    // 格式化搜索结果列表
    if (response.results && response.results.length > 0) {
        formattedResults += '相关搜索结果：\n';
        response.results.forEach((result, index) => {
            formattedResults += `\n${index + 1}. ${result.title || '无标题'}\n`;
            formattedResults += `   内容摘要：${result.content || '无摘要'}\n`;
            if (result.url) {
                formattedResults += `   来源：${result.url}\n`;
            }
            if (result.score !== undefined) {
                formattedResults += `   相关度：${(result.score * 100).toFixed(1)}%\n`;
            }
        });
    } else {
        formattedResults += '未找到相关搜索结果。';
    }
    return formattedResults;
}

// --------------------------------------------------------------------------------

export async function generateAnswerNode(state: SearchStateType): Promise<{
    finalAnswer: string;
    step: "completed";
    messages: BaseMessage[];
}> {
    let prompt: string;

    if (state.step === "search_failed") {
        // 搜索失败：回退策略，基于 LLM 自身知识回答
        prompt = `搜索API暂时不可用，请基于您的知识回答用户的问题：\n用户问题：${state.userQuery}`;
    } else {
        // 搜索成功：基于搜索结果生成答案
        prompt = `基于以下搜索结果为用户提供完整、准确的答案：
            用户问题：${state.userQuery}
            搜索结果：\n${String(state.searchResults ?? "")}
            请综合搜索结果，提供准确、有用的回答...`;
    }

    const response = await llm.invoke([new SystemMessage(prompt)]);

    return {
        finalAnswer: response.content! as string,
        step: "completed",
        messages: [new AIMessage(response.content)],
    };
}

/**
 * 创建搜索助手工作流
 * 使用 LangGraph 构建三个节点的线性流程：
 * 理解查询 → 执行搜索 → 生成答案
 */
function createSearchAssistant() {
    // v1 推荐用法：直接将 Annotation.Root 传递给 StateGraph
    const workflow = new StateGraph(SearchState)
        .addNode('understand', understandQueryNode)
        .addNode('search', tavilySearchNode)
        .addNode('answer', generateAnswerNode)
        .addEdge(START, 'understand')
        .addEdge('understand', 'search')
        .addEdge('search', 'answer')
        .addEdge('answer', END);

    const memory = new MemorySaver();
    // compile 方法保持不变
    const app = workflow.compile({ checkpointer: memory });
    return app;
}

const app = createSearchAssistant();
const inputs = {
    messages: [new HumanMessage("明天我要去北京，天气怎么样？有合适的景点吗")]
};

const stream = await app.stream(inputs, {
    configurable: {
        thread_id: "ai-news-search",
    },
});
for await (const event of stream) {
    console.log(event);
}
