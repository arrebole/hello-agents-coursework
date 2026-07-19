/**
 * ContextBuilder - GSSC 流水线实现
 *
 * 实现 Gather-Select-Structure-Compress 上下文构建流程：
 * 1. Gather: 从多源收集候选信息（历史、记忆、RAG、工具结果）
 * 2. Select: 基于优先级、相关性、多样性筛选
 * 3. Structure: 组织成结构化上下文模板
 * 4. Compress: 在预算内压缩与规范化
 */

import type { Message } from "../../7.hello-agents/core/message";
import type { MemoryTool } from "../../8.memory-and-RAG/tools/builtin/memory_tool";
import type { RAGTool } from "../../8.memory-and-RAG/tools/builtin/rag_tools";

export type ContextMetadata = Record<string, unknown>;

export interface ContextPacketOptions {
    content: string;
    timestamp?: Date;
    metadata?: ContextMetadata;
    tokenCount?: number;
    relevanceScore?: number;
}

export interface ContextConfigOptions {
    maxTokens?: number;
    reserveRatio?: number;
    minRelevance?: number;
    enableMmr?: boolean;
    mmrLambda?: number;
    systemPromptTemplate?: string;
    enableCompression?: boolean;
}

export interface ContextBuildOptions {
    userQuery: string;
    conversationHistory?: Message[];
    systemInstructions?: string;
    additionalPackets?: ContextPacket[];
}

export interface ContextBuilderOptions {
    memoryTool?: MemoryTool;
    ragTool?: RAGTool;
    config?: ContextConfig | ContextConfigOptions;
}

interface ScoredPacket {
    score: number;
    packet: ContextPacket;
}

/**
 * 上下文信息包。
 *
 * `tokenCount` 未显式传入时会用本地估算函数自动计算，便于后续预算控制。
 */
export class ContextPacket {
    readonly content: string;
    readonly timestamp: Date;
    readonly metadata: ContextMetadata;
    readonly tokenCount: number;
    relevanceScore: number;

    constructor(options: ContextPacketOptions) {
        this.content = options.content;
        this.timestamp = options.timestamp ?? new Date();
        this.metadata = options.metadata ?? {};
        this.tokenCount = options.tokenCount && options.tokenCount > 0
            ? options.tokenCount
            : countTokens(options.content);
        this.relevanceScore = options.relevanceScore ?? 0;
    }
}

/** 上下文构建配置。 */
export class ContextConfig {
    /** 总 token 预算。 */
    readonly maxTokens: number;
    /** 生成余量，通常保留 10%-20%。 */
    readonly reserveRatio: number;
    /** 非系统上下文包的最小相关性阈值。 */
    readonly minRelevance: number;
    /** 是否启用最大边际相关性（多样性）排序。 */
    readonly enableMmr: boolean;
    /** MMR 平衡参数：0=纯多样性，1=纯相关性。 */
    readonly mmrLambda: number;
    /** 默认系统提示模板。 */
    readonly systemPromptTemplate: string;
    /** 是否启用超预算压缩。 */
    readonly enableCompression: boolean;

    constructor(options: ContextConfigOptions = {}) {
        this.maxTokens = options.maxTokens ?? 8000;
        this.reserveRatio = options.reserveRatio ?? 0.15;
        this.minRelevance = options.minRelevance ?? 0.3;
        this.enableMmr = options.enableMmr ?? true;
        this.mmrLambda = options.mmrLambda ?? 0.7;
        this.systemPromptTemplate = options.systemPromptTemplate ?? "";
        this.enableCompression = options.enableCompression ?? true;
    }

    /** 获取可用 token 预算（扣除生成余量）。 */
    getAvailableTokens(): number {
        return Math.floor(this.maxTokens * (1 - this.reserveRatio));
    }
}

/**
 * 上下文构建器 - GSSC 流水线。
 *
 * 用法示例：
 * ```ts
 * const builder = new ContextBuilder({ memoryTool, ragTool, config });
 * const context = await builder.build({
 *   userQuery: "用户问题",
 *   conversationHistory: messages,
 *   systemInstructions: "系统指令",
 * });
 * ```
 */
export class ContextBuilder {
    private readonly memoryTool?: MemoryTool;
    private readonly ragTool?: RAGTool;
    private readonly config: ContextConfig;

    constructor(options?: ContextBuilderOptions);
    constructor(memoryTool?: MemoryTool, ragTool?: RAGTool, config?: ContextConfig | ContextConfigOptions);
    constructor(
        optionsOrMemoryTool: ContextBuilderOptions | MemoryTool = {},
        ragTool?: RAGTool,
        config?: ContextConfig | ContextConfigOptions,
    ) {
        const options = isContextBuilderOptions(optionsOrMemoryTool)
            ? optionsOrMemoryTool
            : { memoryTool: optionsOrMemoryTool, ragTool, config };

        this.memoryTool = options.memoryTool;
        this.ragTool = options.ragTool;
        this.config = options.config instanceof ContextConfig
            ? options.config
            : new ContextConfig(options.config);
    }

    /**
     * 构建完整上下文。
     *
     * 支持对象参数与位置参数两种调用方式：
     * - `build({ userQuery, conversationHistory, systemInstructions })`
     * - `build(userQuery, conversationHistory, systemInstructions, additionalPackets)`
     */
    async build(options: ContextBuildOptions): Promise<string>;
    async build(
        userQuery: string,
        conversationHistory?: Message[],
        systemInstructions?: string,
        additionalPackets?: ContextPacket[],
    ): Promise<string>;
    async build(
        optionsOrUserQuery: ContextBuildOptions | string,
        conversationHistory: Message[] = [],
        systemInstructions?: string,
        additionalPackets: ContextPacket[] = [],
    ): Promise<string> {
        const options = typeof optionsOrUserQuery === "string"
            ? {
                userQuery: optionsOrUserQuery,
                conversationHistory,
                systemInstructions,
                additionalPackets,
            }
            : optionsOrUserQuery;

        const effectiveSystemInstructions = (
            options.systemInstructions ?? this.config.systemPromptTemplate
        ) || undefined;

        const packets = await this.gather(
            options.userQuery,
            options.conversationHistory ?? [],
            effectiveSystemInstructions,
            options.additionalPackets ?? [],
        );

        const selectedPackets = this.select(packets, options.userQuery);
        const structuredContext = this.structure(
            selectedPackets,
            options.userQuery,
            effectiveSystemInstructions,
        );

        return this.compress(structuredContext);
    }

    private async gather(
        userQuery: string,
        conversationHistory: Message[],
        systemInstructions: string | undefined,
        additionalPackets: ContextPacket[],
    ): Promise<ContextPacket[]> {
        const packets: ContextPacket[] = [];

        // P0: 系统指令（强约束）
        if (systemInstructions) {
            packets.push(new ContextPacket({
                content: systemInstructions,
                metadata: { type: "instructions" },
            }));
        }

        // P1: 从记忆中获取任务状态与关键结论
        if (this.memoryTool) {
            try {
                const stateResults = await Promise.resolve(this.memoryTool.run({
                    action: "search",
                    query: "(任务状态 OR 子目标 OR 结论 OR 阻塞)",
                    minImportance: 0.7,
                    limit: 5,
                }));

                if (isUsefulResult(stateResults)) {
                    packets.push(new ContextPacket({
                        content: stateResults,
                        metadata: { type: "task_state", importance: "high" },
                    }));
                }

                const relatedResults = await Promise.resolve(this.memoryTool.run({
                    action: "search",
                    query: userQuery,
                    limit: 5,
                }));

                if (isUsefulResult(relatedResults)) {
                    packets.push(new ContextPacket({
                        content: relatedResults,
                        metadata: { type: "related_memory" },
                    }));
                }
            } catch (error) {
                console.warn(`记忆检索失败: ${errorMessage(error)}`);
            }
        }

        // P2: 从 RAG 中获取事实证据
        if (this.ragTool) {
            try {
                const ragResults = await Promise.resolve(this.ragTool.run({
                    action: "search",
                    query: userQuery,
                    limit: 5,
                }));

                if (isUsefulResult(ragResults) && !ragResults.includes("错误")) {
                    packets.push(new ContextPacket({
                        content: ragResults,
                        metadata: { type: "knowledge_base" },
                    }));
                }
            } catch (error) {
                console.warn(`RAG检索失败: ${errorMessage(error)}`);
            }
        }

        // P3: 对话历史（辅助材料）
        if (conversationHistory.length > 0) {
            const recentHistory = conversationHistory.slice(-10);
            const historyText = recentHistory
                .map((message) => `[${message.role}] ${message.content}`)
                .join("\n");

            packets.push(new ContextPacket({
                content: historyText,
                metadata: { type: "history", count: recentHistory.length },
            }));
        }

        packets.push(...additionalPackets);
        return packets;
    }

    private select(packets: ContextPacket[], userQuery: string): ContextPacket[] {
        const queryTokens = tokenizeForSimilarity(userQuery);

        for (const packet of packets) {
            const contentTokens = tokenizeForSimilarity(packet.content);
            packet.relevanceScore = queryTokens.size > 0
                ? intersectionSize(queryTokens, contentTokens) / queryTokens.size
                : 0;
        }

        const scoredPackets = packets.map((packet) => ({
            score: 0.7 * packet.relevanceScore + 0.3 * recencyScore(packet.timestamp),
            packet,
        }));

        const systemPackets = scoredPackets
            .filter(({ packet }) => packet.metadata.type === "instructions")
            .map(({ packet }) => packet);

        const remaining = scoredPackets
            .filter(({ packet }) => packet.metadata.type !== "instructions")
            .sort((left, right) => right.score - left.score);

        const filtered = remaining.filter(({ packet }) => (
            packet.relevanceScore >= this.config.minRelevance
        ));

        const availableTokens = this.config.getAvailableTokens();
        const selected: ContextPacket[] = [];
        let usedTokens = 0;

        for (const packet of systemPackets) {
            if (usedTokens + packet.tokenCount <= availableTokens) {
                selected.push(packet);
                usedTokens += packet.tokenCount;
            }
        }

        if (this.config.enableMmr) {
            usedTokens = this.fillWithMmr(filtered, selected, usedTokens, availableTokens);
        } else {
            for (const { packet } of filtered) {
                if (usedTokens + packet.tokenCount > availableTokens) continue;
                selected.push(packet);
                usedTokens += packet.tokenCount;
            }
        }

        return selected;
    }

    private fillWithMmr(
        candidates: ScoredPacket[],
        selected: ContextPacket[],
        usedTokens: number,
        availableTokens: number,
    ): number {
        const remaining = [...candidates];
        const selectedNonSystem = selected.filter((packet) => (
            packet.metadata.type !== "instructions"
        ));

        while (remaining.length > 0) {
            let bestIndex = -1;
            let bestScore = Number.NEGATIVE_INFINITY;

            for (let index = 0; index < remaining.length; index += 1) {
                const candidate = remaining[index];
                const maxSimilarity = selectedNonSystem.length === 0
                    ? 0
                    : Math.max(...selectedNonSystem.map((packet) => (
                        jaccardSimilarity(candidate.packet.content, packet.content)
                    )));

                const mmrScore = this.config.mmrLambda * candidate.score
                    - (1 - this.config.mmrLambda) * maxSimilarity;

                if (mmrScore > bestScore) {
                    bestScore = mmrScore;
                    bestIndex = index;
                }
            }

            if (bestIndex < 0) break;

            const [best] = remaining.splice(bestIndex, 1);
            if (usedTokens + best.packet.tokenCount > availableTokens) continue;

            selected.push(best.packet);
            selectedNonSystem.push(best.packet);
            usedTokens += best.packet.tokenCount;
        }

        return usedTokens;
    }

    private structure(
        selectedPackets: ContextPacket[],
        userQuery: string,
        _systemInstructions?: string,
    ): string {
        const sections: string[] = [];

        const p0Packets = selectedPackets.filter((packet) => (
            packet.metadata.type === "instructions"
        ));
        if (p0Packets.length > 0) {
            sections.push([
                "[Role & Policies]",
                p0Packets.map((packet) => packet.content).join("\n"),
            ].join("\n"));
        }

        sections.push(`[Task]\n用户问题：${userQuery}`);

        const p1Packets = selectedPackets.filter((packet) => (
            packet.metadata.type === "task_state"
        ));
        if (p1Packets.length > 0) {
            sections.push([
                "[State]",
                "关键进展与未决问题：",
                p1Packets.map((packet) => packet.content).join("\n"),
            ].join("\n"));
        }

        const evidenceTypes = new Set(["related_memory", "knowledge_base", "retrieval", "tool_result"]);
        const p2Packets = selectedPackets.filter((packet) => (
            evidenceTypes.has(String(packet.metadata.type))
        ));
        if (p2Packets.length > 0) {
            const evidenceContent = p2Packets
                .map((packet) => `\n${packet.content}\n`)
                .join("");
            sections.push(`[Evidence]\n事实与引用：\n${evidenceContent}`);
        }

        const p3Packets = selectedPackets.filter((packet) => (
            packet.metadata.type === "history"
        ));
        if (p3Packets.length > 0) {
            sections.push([
                "[Context]",
                "对话历史与背景：",
                p3Packets.map((packet) => packet.content).join("\n"),
            ].join("\n"));
        }

        sections.push([
            "[Output]",
            "请按以下格式回答：",
            "1. 结论（简洁明确）",
            "2. 依据（列出支撑证据及来源）",
            "3. 风险与假设（如有）",
            "4. 下一步行动建议（如适用）",
        ].join("\n"));

        return sections.join("\n\n");
    }

    private compress(context: string): string {
        if (!this.config.enableCompression) return context;

        const currentTokens = countTokens(context);
        const availableTokens = this.config.getAvailableTokens();

        if (currentTokens <= availableTokens) return context;

        console.warn(`上下文超预算 (${currentTokens} > ${availableTokens})，执行截断`);

        const compressedLines: string[] = [];
        let usedTokens = 0;

        for (const line of context.split("\n")) {
            const lineTokens = countTokens(line);
            if (usedTokens + lineTokens > availableTokens) break;
            compressedLines.push(line);
            usedTokens += lineTokens;
        }

        return compressedLines.join("\n");
    }
}

/** 计算文本 token 数。当前项目未引入 tiktoken，因此采用 1 token ≈ 4 字符的粗略估算。 */
export function countTokens(text: string): number {
    if (!text) return 0;
    return Math.ceil(Array.from(text).length / 4);
}

function recencyScore(timestamp: Date): number {
    const deltaSeconds = Math.max((Date.now() - timestamp.getTime()) / 1000, 0);
    const tau = 3600;
    return Math.exp(-deltaSeconds / tau);
}

function tokenizeForSimilarity(text: string): Set<string> {
    return new Set(
        text
            .toLowerCase()
            .split(/\s+/)
            .map((token) => token.trim())
            .filter(Boolean),
    );
}

function intersectionSize(left: Set<string>, right: Set<string>): number {
    let count = 0;
    for (const value of left) {
        if (right.has(value)) count += 1;
    }
    return count;
}

function jaccardSimilarity(leftText: string, rightText: string): number {
    const left = tokenizeForSimilarity(leftText);
    const right = tokenizeForSimilarity(rightText);
    const union = new Set([...left, ...right]);
    if (union.size === 0) return 0;
    return intersectionSize(left, right) / union.size;
}

function isUsefulResult(value: string): boolean {
    return value.length > 0 && !value.includes("未找到");
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function isContextBuilderOptions(value: ContextBuilderOptions | MemoryTool): value is ContextBuilderOptions {
    return Boolean(
        value
        && typeof value === "object"
        && (
            "memoryTool" in value
            || "ragTool" in value
            || "config" in value
        ),
    );
}
