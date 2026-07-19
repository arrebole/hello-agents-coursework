// RAG工具 - 检索增强生成
//
// 数据流程：用户数据 -> 文档解析 -> 向量化存储 -> 智能检索 -> LLM增强问答

import { existsSync, mkdirSync } from "fs";
import { rm, writeFile } from "fs/promises";
import { basename, join } from "path";

import { Tool, ToolParameter } from "../../../7.hello-agents/tools/base";
import { HelloAgentsLLM } from "../../../7.hello-agents/core/llm";
import { createRagPipeline, RagPipeline, RagSearchResult } from "../../memory/rag/pipeline";

const RAG_TOOL_ACTIONS = ["add_document", "add_text", "ask", "search", "stats", "clear"] as const;
type RagToolAction = (typeof RAG_TOOL_ACTIONS)[number];

interface RagToolParams {
    action?: RagToolAction;
    file_path?: string;
    filePath?: string;
    document_id?: string;
    documentId?: string;
    text?: string;
    question?: string;
    query?: string;
    namespace?: string;
    limit?: number;
    min_score?: number;
    minScore?: number;
    chunk_size?: number;
    chunkSize?: number;
    chunk_overlap?: number;
    chunkOverlap?: number;
    enable_advanced_search?: boolean;
    enableAdvancedSearch?: boolean;
    include_citations?: boolean;
    includeCitations?: boolean;
    max_chars?: number;
    maxChars?: number;
    confirm?: boolean;
}

interface RAGToolOptions {
    knowledgeBasePath?: string;
    qdrantUrl?: string | null;
    qdrantApiKey?: string | null;
    collectionName?: string;
    ragNamespace?: string;
    expandable?: boolean;
}

function isRagToolAction(value: unknown): value is RagToolAction {
    return RAG_TOOL_ACTIONS.includes(value as RagToolAction);
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function cleanText(value: unknown): string {
    return String(value ?? "").replace(/\u0000/g, "");
}

function getNumber(value: unknown, fallback: number): number {
    const numericValue = Number(value);
    return Number.isFinite(numericValue) ? numericValue : fallback;
}

export class RAGTool extends Tool {
    private readonly knowledgeBasePath: string;
    private readonly qdrantUrl?: string | null;
    private readonly qdrantApiKey?: string | null;
    private readonly collectionName: string;
    private readonly ragNamespace: string;
    private readonly pipelines = new Map<string, RagPipeline>();
    private llm: HelloAgentsLLM | null = null;
    private initialized = false;
    private initError = "";

    constructor(
        knowledgeBasePathOrOptions: string | RAGToolOptions = "./knowledge_base",
        qdrantUrl: string | null = null,
        qdrantApiKey: string | null = null,
        collectionName: string = "rag_knowledge_base",
        ragNamespace: string = "default",
        expandable: boolean = false,
    ) {
        const options = typeof knowledgeBasePathOrOptions === "string"
            ? { knowledgeBasePath: knowledgeBasePathOrOptions, qdrantUrl, qdrantApiKey, collectionName, ragNamespace, expandable }
            : knowledgeBasePathOrOptions;

        super(
            "rag",
            "RAG工具 - 支持多格式文档检索增强生成，提供智能问答能力",
            options.expandable ?? false,
        );

        this.knowledgeBasePath = options.knowledgeBasePath ?? "./knowledge_base";
        this.qdrantUrl = options.qdrantUrl ?? process.env.QDRANT_URL;
        this.qdrantApiKey = options.qdrantApiKey ?? process.env.QDRANT_API_KEY;
        this.collectionName = options.collectionName ?? "rag_knowledge_base";
        this.ragNamespace = options.ragNamespace ?? "default";

        mkdirSync(this.knowledgeBasePath, { recursive: true });
        this.initComponents();
    }

    private initComponents(): void {
        try {
            const defaultPipeline = createRagPipeline({
                qdrantUrl: this.qdrantUrl,
                qdrantApiKey: this.qdrantApiKey,
                collectionName: this.collectionName,
                ragNamespace: this.ragNamespace,
            });
            this.pipelines.set(this.ragNamespace, defaultPipeline);

            try {
                this.llm = new HelloAgentsLLM();
            } catch (error) {
                this.llm = null;
                this.initError = `LLM初始化失败: ${errorMessage(error)}`;
            }

            this.initialized = true;
            console.log(`✅ RAG工具初始化成功: namespace=${this.ragNamespace}, collection=${this.collectionName}`);
        } catch (error) {
            this.initialized = false;
            this.initError = errorMessage(error);
            console.log(`❌ RAG工具初始化失败: ${this.initError}`);
        }
    }

    private getPipeline(namespace?: string | null): RagPipeline {
        const targetNamespace = namespace || this.ragNamespace;
        const cached = this.pipelines.get(targetNamespace);
        if (cached) return cached;

        const pipeline = createRagPipeline({
            qdrantUrl: this.qdrantUrl,
            qdrantApiKey: this.qdrantApiKey,
            collectionName: this.collectionName,
            ragNamespace: targetNamespace,
        });
        this.pipelines.set(targetNamespace, pipeline);
        return pipeline;
    }

    async run(parameters: Record<string, unknown>): Promise<string> {
        if (!this.validateParameters(parameters)) {
            return "❌ 参数验证失败：缺少必需的参数";
        }

        if (!this.initialized) {
            return `❌ RAG工具未正确初始化，请检查配置: ${this.initError || "未知错误"}`;
        }

        if (!isRagToolAction(parameters.action)) {
            return `❌ 不支持的操作: ${String(parameters.action)}`;
        }

        const params = parameters as RagToolParams;

        try {
            switch (params.action) {
                case "add_document":
                    return await this.addDocumentInternal(
                        params.file_path ?? params.filePath ?? "",
                        params.document_id ?? params.documentId,
                        params.namespace ?? "default",
                        getNumber(params.chunk_size ?? params.chunkSize, 800),
                        getNumber(params.chunk_overlap ?? params.chunkOverlap, 100),
                    );
                case "add_text":
                    return await this.addTextInternal(
                        params.text ?? "",
                        params.document_id ?? params.documentId,
                        params.namespace ?? "default",
                        getNumber(params.chunk_size ?? params.chunkSize, 800),
                        getNumber(params.chunk_overlap ?? params.chunkOverlap, 100),
                    );
                case "ask":
                    return await this.askInternal(
                        params.question ?? params.query ?? "",
                        getNumber(params.limit, 5),
                        params.enable_advanced_search ?? params.enableAdvancedSearch ?? true,
                        params.include_citations ?? params.includeCitations ?? true,
                        getNumber(params.max_chars ?? params.maxChars, 1200),
                        params.namespace ?? "default",
                    );
                case "search":
                    return await this.searchInternal(
                        params.query ?? params.question ?? "",
                        getNumber(params.limit, 5),
                        getNumber(params.min_score ?? params.minScore, 0.1),
                        params.enable_advanced_search ?? params.enableAdvancedSearch ?? true,
                        getNumber(params.max_chars ?? params.maxChars, 1200),
                        params.include_citations ?? params.includeCitations ?? true,
                        params.namespace ?? "default",
                    );
                case "stats":
                    return await this.getStatsInternal(params.namespace ?? "default");
                case "clear":
                    return await this.clearKnowledgeBaseInternal(params.confirm ?? false, params.namespace ?? "default");
                default:
                    return `❌ 不支持的操作: ${String(params.action)}`;
            }
        } catch (error) {
            return `❌ 执行操作 '${params.action}' 时发生错误: ${errorMessage(error)}`;
        }
    }

    getParameters(): ToolParameter[] {
        return [
            new ToolParameter({
                name: "action",
                type: "string",
                description: "操作类型：add_document(添加文档), add_text(添加文本), ask(智能问答), search(搜索), stats(统计), clear(清空)",
                required: true,
            }),
            new ToolParameter({
                name: "file_path",
                type: "string",
                description: "文档文件路径（文本、Markdown、JSON、CSV、HTML等可直接解析；PDF/Office/图片/音频需额外解析能力）",
                required: false,
            }),
            new ToolParameter({
                name: "text",
                type: "string",
                description: "要添加的文本内容",
                required: false,
            }),
            new ToolParameter({
                name: "question",
                type: "string",
                description: "用户问题（用于智能问答）",
                required: false,
            }),
            new ToolParameter({
                name: "query",
                type: "string",
                description: "搜索查询词（用于基础搜索）",
                required: false,
            }),
            new ToolParameter({
                name: "namespace",
                type: "string",
                description: "知识库命名空间（用于隔离不同项目，默认：default）",
                required: false,
                default: "default",
            }),
            new ToolParameter({
                name: "limit",
                type: "integer",
                description: "返回结果数量（默认：5）",
                required: false,
                default: 5,
            }),
            new ToolParameter({
                name: "include_citations",
                type: "boolean",
                description: "是否包含引用来源（默认：true）",
                required: false,
                default: true,
            }),
        ];
    }

    async addDocumentInternal(
        filePath: string,
        documentId: string | null = null,
        namespace: string = "default",
        chunkSize: number = 800,
        chunkOverlap: number = 100,
    ): Promise<string> {
        try {
            if (!filePath || !existsSync(filePath)) {
                return `❌ 文件不存在: ${filePath}`;
            }

            void documentId;
            const pipeline = this.getPipeline(namespace);
            const start = Date.now();
            const chunksAdded = await pipeline.addDocuments([filePath], chunkSize, chunkOverlap);
            const processMs = Date.now() - start;

            if (chunksAdded === 0) {
                return `⚠️ 未能从文件解析内容: ${basename(filePath)}`;
            }

            return [
                `✅ 文档已添加到知识库: ${basename(filePath)}`,
                `📊 分块数量: ${chunksAdded}`,
                `⏱️ 处理时间: ${processMs}ms`,
                `📝 命名空间: ${pipeline.namespace}`,
            ].join("\n");
        } catch (error) {
            return `❌ 添加文档失败: ${errorMessage(error)}`;
        }
    }

    async addTextInternal(
        text: string,
        documentId: string | null = null,
        namespace: string = "default",
        chunkSize: number = 800,
        chunkOverlap: number = 100,
    ): Promise<string> {
        if (!text || !text.trim()) {
            return "❌ 文本内容不能为空";
        }

        const resolvedDocumentId = documentId || `text_${Math.abs(hashString(text)) % 100000}`;
        const tmpPath = join(this.knowledgeBasePath, `${resolvedDocumentId}.md`);

        try {
            await writeFile(tmpPath, text, "utf8");

            const pipeline = this.getPipeline(namespace);
            const start = Date.now();
            const chunksAdded = await pipeline.addDocuments([tmpPath], chunkSize, chunkOverlap);
            const processMs = Date.now() - start;

            if (chunksAdded === 0) {
                return "⚠️ 未能从文本生成有效分块";
            }

            return [
                `✅ 文本已添加到知识库: ${resolvedDocumentId}`,
                `📊 分块数量: ${chunksAdded}`,
                `⏱️ 处理时间: ${processMs}ms`,
                `📝 命名空间: ${pipeline.namespace}`,
            ].join("\n");
        } catch (error) {
            return `❌ 添加文本失败: ${errorMessage(error)}`;
        } finally {
            await rm(tmpPath, { force: true }).catch(() => undefined);
        }
    }

    async searchInternal(
        query: string,
        limit: number = 5,
        minScore: number = 0.1,
        enableAdvancedSearch: boolean = true,
        maxChars: number = 1200,
        includeCitations: boolean = true,
        namespace: string = "default",
    ): Promise<string> {
        try {
            if (!query || !query.trim()) {
                return "❌ 搜索查询不能为空";
            }

            const pipeline = this.getPipeline(namespace);
            const results = enableAdvancedSearch
                ? await pipeline.searchAdvanced(query, limit, { enableMqe: true, enableHyde: true, scoreThreshold: minScore > 0 ? minScore : undefined })
                : await pipeline.search(query, limit, minScore > 0 ? minScore : undefined);

            if (!results.length) {
                return `🔍 未找到与 '${query}' 相关的内容`;
            }

            const output = ["搜索结果："];
            results.forEach((result, index) => {
                const metadata = result.metadata ?? {};
                const rawContent = cleanText(metadata.content);
                const content = rawContent.length > maxChars ? `${rawContent.slice(0, maxChars)}...` : rawContent;
                const source = cleanText(metadata.source_path ?? metadata.source_name ?? "unknown");

                output.push(`\n${index + 1}. 文档: **${source}** (相似度: ${result.score.toFixed(3)})`);
                output.push(`   ${content}`);

                if (includeCitations && metadata.heading_path) {
                    output.push(`   章节: ${cleanText(metadata.heading_path)}`);
                }
            });

            return output.join("\n");
        } catch (error) {
            return `❌ 搜索失败: ${errorMessage(error)}`;
        }
    }

    async askInternal(
        question: string,
        limit: number = 5,
        enableAdvancedSearch: boolean = true,
        includeCitations: boolean = true,
        maxChars: number = 1200,
        namespace: string = "default",
    ): Promise<string> {
        try {
            if (!question || !question.trim()) {
                return "❌ 请提供要询问的问题";
            }

            const userQuestion = question.trim();
            console.log(`🔍 智能问答: ${userQuestion}`);

            const pipeline = this.getPipeline(namespace);
            const searchStart = Date.now();
            const results = enableAdvancedSearch
                ? await pipeline.searchAdvanced(userQuestion, limit, { enableMqe: true, enableHyde: true })
                : await pipeline.search(userQuestion, limit);
            const searchTime = Date.now() - searchStart;

            if (!results.length) {
                return [
                    `🤔 抱歉，我在知识库中没有找到与「${userQuestion}」相关的信息。`,
                    "",
                    "💡 建议：",
                    "• 尝试使用更简洁的关键词",
                    "• 检查是否已添加相关文档",
                    "• 使用 stats 操作查看知识库状态",
                ].join("\n");
            }

            const contextParts: string[] = [];
            const citations: Array<{ index: number; source: string; score: number }> = [];
            let totalScore = 0;

            results.forEach((result, index) => {
                const metadata = result.metadata ?? {};
                const content = cleanText(metadata.content).trim();
                const source = cleanText(metadata.source_path ?? metadata.source_name ?? "unknown");
                totalScore += result.score;

                if (content) {
                    contextParts.push(`片段 ${index + 1}：${this.cleanContentForContext(content)}`);
                    if (includeCitations) {
                        citations.push({ index: index + 1, source: basename(source), score: result.score });
                    }
                }
            });

            let context = contextParts.join("\n\n");
            if (context.length > maxChars) {
                context = this.smartTruncateContext(context, maxChars);
            }

            const answerStart = Date.now();
            const answer = this.llm
                ? await this.llm.invoke([
                    { role: "system", content: this.buildSystemPrompt() },
                    { role: "user", content: this.buildUserPrompt(userQuestion, context) },
                ])
                : this.buildExtractiveAnswer(userQuestion, context);
            const llmTime = Date.now() - answerStart;

            if (!answer || !answer.trim()) {
                return "❌ LLM未能生成有效答案，请稍后重试";
            }

            return this.formatFinalAnswer(
                userQuestion,
                answer.trim(),
                includeCitations ? citations : null,
                searchTime,
                llmTime,
                totalScore / results.length,
            );
        } catch (error) {
            return `❌ 智能问答失败: ${errorMessage(error)}\n💡 请检查知识库状态或稍后重试`;
        }
    }

    private cleanContentForContext(content: string): string {
        const cleaned = content.split(/\s+/).join(" ");
        return cleaned.length > 300 ? `${cleaned.slice(0, 300)}...` : cleaned;
    }

    private smartTruncateContext(context: string, maxChars: number): string {
        if (context.length <= maxChars) return context;
        const truncated = context.slice(0, maxChars);
        const lastBreak = truncated.lastIndexOf("\n\n");
        if (lastBreak > maxChars * 0.7) {
            return `${truncated.slice(0, lastBreak)}\n\n[...更多内容被截断]`;
        }
        return `${truncated.slice(0, Math.max(0, maxChars - 20))}...[内容被截断]`;
    }

    private buildSystemPrompt(): string {
        return [
            "你是一个专业的知识助手，具备以下能力：",
            "1. 精准理解：仔细理解用户问题的核心意图",
            "2. 可信回答：严格基于提供的上下文信息回答，不编造内容",
            "3. 信息整合：从多个片段中提取关键信息，形成完整答案",
            "4. 清晰表达：用简洁明了的语言回答，适当使用结构化格式",
            "5. 诚实表达：如果上下文不足以回答问题，请坦诚说明",
            "",
            "回答格式要求：",
            "• 直接回答核心问题",
            "• 必要时使用要点或步骤",
            "• 引用关键原文时使用引号",
            "• 避免重复和冗余",
        ].join("\n");
    }

    private buildUserPrompt(question: string, context: string): string {
        return [
            "请基于以下上下文信息回答问题：",
            "",
            `【问题】${question}`,
            "",
            `【相关上下文】\n${context}`,
            "",
            "【要求】请提供准确、有帮助的回答。如果上下文信息不足，请说明需要什么额外信息。",
        ].join("\n");
    }

    private buildExtractiveAnswer(question: string, context: string): string {
        void question;
        return [
            "当前未配置可用的 LLM，以下是从知识库检索到的相关片段：",
            "",
            context || "没有可用于回答的上下文。",
        ].join("\n");
    }

    private formatFinalAnswer(
        question: string,
        answer: string,
        citations: Array<{ index: number; source: string; score: number }> | null = null,
        searchTime: number = 0,
        llmTime: number = 0,
        avgScore: number = 0,
    ): string {
        void question;
        const result = ["🤖 **智能问答结果**\n", answer];

        if (citations?.length) {
            result.push("\n\n📚 **参考来源**");
            citations.forEach((citation) => {
                const scoreEmoji = citation.score > 0.8 ? "🟢" : citation.score > 0.6 ? "🟡" : "🔵";
                result.push(`${scoreEmoji} [${citation.index}] ${citation.source} (相似度: ${citation.score.toFixed(3)})`);
            });
        }

        result.push(`\n⚡ 检索: ${searchTime}ms | 生成: ${llmTime}ms | 平均相似度: ${avgScore.toFixed(3)}`);
        return result.join("\n");
    }

    async clearKnowledgeBaseInternal(confirm: boolean = false, namespace: string = "default"): Promise<string> {
        try {
            if (!confirm) {
                return [
                    "⚠️ 危险操作：清空知识库将删除所有数据！",
                    "请使用 confirm=true 参数确认执行。",
                ].join("\n");
            }

            const pipeline = this.getPipeline(namespace);
            const success = await pipeline.store.clearCollection();
            if (success) {
                this.pipelines.set(namespace, createRagPipeline({
                    qdrantUrl: this.qdrantUrl,
                    qdrantApiKey: this.qdrantApiKey,
                    collectionName: this.collectionName,
                    ragNamespace: namespace,
                }));
                return `✅ 知识库已成功清空（命名空间：${namespace}）`;
            }

            return "❌ 清空知识库失败";
        } catch (error) {
            return `❌ 清空知识库失败: ${errorMessage(error)}`;
        }
    }

    async getStatsInternal(namespace: string = "default"): Promise<string> {
        try {
            const pipeline = this.getPipeline(namespace);
            const stats = await pipeline.getStats();
            const output = [
                "📊 **RAG 知识库统计**",
                `📝 命名空间: ${pipeline.namespace}`,
                `📋 集合名称: ${this.collectionName}`,
                `📂 存储根路径: ${this.knowledgeBasePath}`,
            ];

            if (stats) {
                const config = typeof stats.config === "object" && stats.config !== null
                    ? stats.config as Record<string, unknown>
                    : {};
                const totalVectors = stats.points_count ?? stats.vectors_count ?? stats.count ?? 0;
                output.push(`📦 存储类型: ${stats.store_type ?? "unknown"}`);
                output.push(`📊 文档分块数: ${Number(totalVectors) || 0}`);
                output.push(`🔢 向量维度: ${config.vector_size ?? "unknown"}`);
                output.push(`📎 距离度量: ${config.distance ?? "unknown"}`);
            }

            output.push("");
            output.push("🟢 **系统状态**");
            output.push(`✅ RAG 管道: ${this.initialized ? "正常" : "异常"}`);
            output.push(`✅ LLM 连接: ${this.llm ? "正常" : "异常"}`);
            return output.join("\n");
        } catch (error) {
            return `❌ 获取统计信息失败: ${errorMessage(error)}`;
        }
    }

    async getRelevantContext(query: string, limit: number = 3, maxChars: number = 1200, namespace?: string | null): Promise<string> {
        try {
            if (!query) return "";
            const pipeline = this.getPipeline(namespace);
            const results = await pipeline.search(query, limit);
            if (!results.length) return "";

            let mergedContext = results
                .map((result) => cleanText(result.metadata?.content))
                .filter(Boolean)
                .join("\n\n");

            if (mergedContext.length > maxChars) {
                mergedContext = `${mergedContext.slice(0, maxChars)}...`;
            }

            return mergedContext;
        } catch (error) {
            return `获取上下文失败: ${errorMessage(error)}`;
        }
    }

    async batchAddTexts(
        texts: string[],
        documentIds: string[] | null = null,
        chunkSize: number = 800,
        chunkOverlap: number = 100,
        namespace: string | null = null,
    ): Promise<string> {
        try {
            if (!texts.length) return "❌ 文本列表不能为空";
            if (documentIds && documentIds.length !== texts.length) {
                return "❌ 文本数量和文档ID数量不匹配";
            }

            const start = Date.now();
            let totalChunks = 0;
            const successfulFiles: string[] = [];

            for (let i = 0; i < texts.length; i += 1) {
                if (!texts[i]?.trim()) continue;
                const docId = documentIds?.[i] ?? `batch_text_${i}`;
                const result = await this.addTextInternal(texts[i], docId, namespace ?? "default", chunkSize, chunkOverlap);
                const chunks = this.extractChunkCount(result);
                if (chunks > 0) {
                    totalChunks += chunks;
                    successfulFiles.push(docId);
                }
            }

            return [
                "✅ 批量添加完成",
                `📊 成功文件: ${successfulFiles.length}/${texts.length}`,
                `📊 总分块数: ${totalChunks}`,
                `⏱️ 处理时间: ${Date.now() - start}ms`,
            ].join("\n");
        } catch (error) {
            return `❌ 批量添加失败: ${errorMessage(error)}`;
        }
    }

    async clearAllNamespaces(): Promise<string> {
        try {
            for (const pipeline of this.pipelines.values()) {
                await pipeline.store.clearCollection();
            }
            this.pipelines.clear();
            this.initComponents();
            return "✅ 所有命名空间数据已清空并重新初始化";
        } catch (error) {
            return `❌ 清空所有命名空间失败: ${errorMessage(error)}`;
        }
    }

    addDocument(filePath: string, namespace: string = "default"): Promise<string> {
        return this.run({ action: "add_document", file_path: filePath, namespace });
    }

    addText(text: string, namespace: string = "default", documentId: string | null = null): Promise<string> {
        return this.run({ action: "add_text", text, namespace, document_id: documentId ?? undefined });
    }

    ask(question: string, namespace: string = "default", kwargs: Record<string, unknown> = {}): Promise<string> {
        return this.run({ action: "ask", question, namespace, ...kwargs });
    }

    search(query: string, namespace: string = "default", kwargs: Record<string, unknown> = {}): Promise<string> {
        return this.run({ action: "search", query, namespace, ...kwargs });
    }

    async addDocumentsBatch(filePaths: string[], namespace: string = "default"): Promise<string> {
        if (!filePaths.length) return "❌ 文件路径列表不能为空";

        const failedResults: string[] = [];
        let successful = 0;
        let totalChunks = 0;
        const start = Date.now();

        for (let i = 0; i < filePaths.length; i += 1) {
            const filePath = filePaths[i];
            console.log(`📄 处理文档 ${i + 1}/${filePaths.length}: ${basename(filePath)}`);
            const result = await this.addDocument(filePath, namespace);
            const chunks = this.extractChunkCount(result);
            if (chunks > 0) {
                successful += 1;
                totalChunks += chunks;
            } else {
                failedResults.push(`❌ ${basename(filePath)}: 处理失败`);
            }
        }

        const output = [
            "📊 **批量处理完成**",
            `✅ 成功: ${successful}/${filePaths.length} 个文档`,
            `📊 总分块数: ${totalChunks}`,
            `⏱️ 总耗时: ${Date.now() - start}ms`,
            `📝 命名空间: ${namespace}`,
        ];

        if (failedResults.length > 0) {
            output.push(`❌ 失败: ${failedResults.length} 个文档`);
            output.push("\n**失败详情:**");
            output.push(...failedResults);
        }

        return output.join("\n");
    }

    async addTextsBatch(texts: string[], namespace: string = "default", documentIds: string[] | null = null): Promise<string> {
        if (!texts.length) return "❌ 文本列表不能为空";
        if (documentIds && documentIds.length !== texts.length) {
            return "❌ 文本数量和文档ID数量不匹配";
        }

        const failedResults: string[] = [];
        let successful = 0;
        let totalChunks = 0;
        const start = Date.now();

        for (let i = 0; i < texts.length; i += 1) {
            const docId = documentIds?.[i] ?? `batch_text_${i + 1}`;
            console.log(`📝 处理文本 ${i + 1}/${texts.length}: ${docId}`);
            const result = await this.addText(texts[i], namespace, docId);
            const chunks = this.extractChunkCount(result);
            if (chunks > 0) {
                successful += 1;
                totalChunks += chunks;
            } else {
                failedResults.push(`❌ ${docId}: 处理失败`);
            }
        }

        const output = [
            "📊 **批量文本处理完成**",
            `✅ 成功: ${successful}/${texts.length} 个文本`,
            `📊 总分块数: ${totalChunks}`,
            `⏱️ 总耗时: ${Date.now() - start}ms`,
            `📝 命名空间: ${namespace}`,
        ];

        if (failedResults.length > 0) {
            output.push(`❌ 失败: ${failedResults.length} 个文本`);
            output.push("\n**失败详情:**");
            output.push(...failedResults);
        }

        return output.join("\n");
    }

    private extractChunkCount(result: string): number {
        const match = result.match(/分块数量:\s*(\d+)/);
        return match ? Number(match[1]) : 0;
    }
}

function hashString(text: string): number {
    let hash = 0;
    for (let i = 0; i < text.length; i += 1) {
        hash = ((hash << 5) - hash + text.charCodeAt(i)) | 0;
    }
    return hash;
}

export { RAGTool as RagTool };
export type { RagSearchResult };

