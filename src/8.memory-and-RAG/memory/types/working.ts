// 工作记忆（TTL管理，纯内存）

import { MemoryConfig } from "../config";
import { MemoryStore } from "../storage/store";
import { Memory, MemoryItem } from "./memory";
import { Heap } from "heap-js";

/** 堆中的单条记录。 */
type MemoryHeapEntry = { priority: number; timestamp: number; memoryItem: MemoryItem };
/** 检索阶段按记忆 ID 存放的向量分数。 */
type VectorScores = Map<string, number>;

/**
 * 工作记忆实现。
 *
 * 特点：
 * - 容量有限
 * - 纯内存存储
 * - 支持 TTL 自动清理
 * - 支持优先级管理与轻量混合检索
 */
export class WorkingMemory extends Memory {
    private readonly config: MemoryConfig;
    private maxCapacity: number;
    private maxAgeMinutes: number;
    private maxTokens: number

    private currentTokens: number
    private sessionStart: number

    // 内存存储（工作记忆不需要持久化）
    private memories: MemoryItem[];
    // 使用优先级队列管理记忆
    private memoryHeap: MemoryHeapEntry[];


    /**
     * 创建工作记忆实例。
     *
     * @param config 记忆配置
     * @param store 存储后端
     */
    constructor(config: MemoryConfig, store: MemoryStore) {
        super(config, store);
        this.config = config;
        this.maxCapacity = Math.max(1, config.workingMemoryCapacity);
        // 允许通过额外字段覆盖 TTL，方便和 Python 版配置保持一致。
        this.maxAgeMinutes = config.workingMemoryTTL;
        this.maxTokens = Math.max(1, config.workingMemoryTokens);
        this.memories = [];
        this.memoryHeap = [];
        this.currentTokens = 0;
        this.sessionStart = new Date().getTime();
    }

    /**
     * 添加一条工作记忆。
     *
     * 会同步执行 TTL 清理、优先级计算和容量约束。
     *
     * @param memoryItem 要写入的记忆
     * @returns 记忆 ID
     */
    public add(memoryItem: MemoryItem): string {
        // 先清理过期项，避免把新数据压进一个已经失效的工作集里。
        this.expireOldMemories(); // 过期清理

        // 工作记忆默认没有 timestamp 时，用当前时间补齐，保证后续排序和衰减可用。
        const timestamp = memoryItem.timestamp ?? Date.now();
        memoryItem.timestamp = timestamp;
        // 优先级由“重要性 × 时间衰减”构成，越重要且越新的内容越靠前。
        const priority = this.calculatePriority(memoryItem);

        // 堆只负责快速拿到高优先级项，真正的数据源仍然是 memories 列表。
        Heap.heappush(
            this.memoryHeap,
            { priority, timestamp, memoryItem },
            this.compareMemoryHeapEntries
        );
        // 列表是当前工作记忆的主索引，检索、更新、删除都以它为准。
        this.memories.push(memoryItem);

        // token 统计用于容量控制，防止短期上下文无限膨胀。
        this.currentTokens += this.countTokens(memoryItem.content);

        // 新增后立即做双重约束：条数限制和 token 限制。
        this.enforceCapacityLimits();

        return memoryItem.id;
    }

    /**
     * 检索工作记忆。
     *
     * 先做过期清理，再按用户过滤，最后结合 TF-IDF 近似分数、关键词命中、
     * 时间衰减和重要性做排序。
     *
     * @param query 查询文本
     * @param limit 返回数量上限
     * @param kwargs 额外参数，支持 `userId` / `user_id`
     * @returns 排序后的记忆列表
     */
    public retrieve(
        query: string,
        limit: number = 5,
        kwargs: Record<string, unknown> = {}
    ): MemoryItem[] {

        // 遗忘过期记忆
        this.expireOldMemories();

        // 如果传了 userId，就只检索该用户当前的工作记忆。
        const userId = this.extractUserId(kwargs);
        const activeMemories = this.getActiveMemories();
        const filteredMemories = userId
            ? activeMemories.filter((memory) => memory.userId === userId)
            : activeMemories;

        if (filteredMemories.length === 0) {
            return [];
        }

        // 先做一个轻量 TF-IDF 近似，用来给内容相似的记忆抬分。
        const vectorScores: VectorScores = this.tryTfidfSearch(query, filteredMemories);

        // 再叠加关键词命中、时间衰减和重要性，得到最终排序分。
        const scoredMemories: Array<{ score: number; memory: MemoryItem; }> = [];

        for (const memory of filteredMemories) {
            const vectorScore = vectorScores.get(memory.id) ?? 0.0;
            const keywordScore = this.calculateKeywordScore(query, memory.content);

            // 关键词和向量分数互补：向量负责语义相似，关键词负责精确命中。
            const baseRelevance = vectorScore > 0
                ? vectorScore * 0.7 + keywordScore * 0.3
                : keywordScore;

            // 越新的记忆保留得越久，旧内容会逐渐失去权重。
            const timeDecay = this.calculateTimeDecay(memory.timestamp);
            // 重要性在最终分数里也占一层权重，避免新但无关的内容排到前面。
            const importanceWeight = 0.8 + memory.importance * 0.4;

            const finalScore =
                baseRelevance * timeDecay * importanceWeight;

            if (finalScore > 0) {
                scoredMemories.push({
                    score: finalScore,
                    memory,
                });
            }
        }

        scoredMemories.sort((a, b) => b.score - a.score);

        return scoredMemories
            .slice(0, Math.max(0, Math.floor(limit)))
            .map(({ memory }) => this.cloneMemoryItem(memory));
    }


    /**
     * 更新指定工作记忆。
     *
     * @param memoryId 目标记忆 ID
     * @param content 新内容；不传则不修改
     * @param importance 新重要性；不传则不修改
     * @param metadata 新元数据；不传则不修改
     * @returns 是否更新成功
     */
    public update(
        memoryId: string,
        content: string | null = null,
        importance: number | null = null,
        metadata: Record<string, unknown> | null = null
    ): boolean {
        const index = this.memories.findIndex((memory) => memory.id === memoryId);
        if (index < 0) {
            return false;
        }

        const memory = this.memories[index];
        const oldTokens = this.countTokens(memory.content);

        if (content !== null) {
            memory.content = content;
            // 内容变化会直接影响 token 预算，所以这里要即时回算。
            this.currentTokens = this.currentTokens - oldTokens + this.countTokens(content);
        }

        if (importance !== null) {
            memory.importance = this.clampImportance(importance);
        }

        if (metadata !== null) {
            memory.metadata = {
                ...(memory.metadata ?? {}),
                ...metadata,
            };
        }

        memory.timestamp = Date.now();
        // 任何影响排序的字段变化后，都重建堆，保证堆顶仍然正确。
        this.rebuildHeap();
        return true;
    }

    /**
     * 删除指定工作记忆。
     *
     * @param memoryId 目标记忆 ID
     * @returns 是否删除成功
     */
    public remove(memoryId: string): boolean {
        const index = this.memories.findIndex((memory) => memory.id === memoryId);
        if (index < 0) {
            return false;
        }

        const [removedMemory] = this.memories.splice(index, 1);
        // 删除一条记忆后要同步扣减 token，再把堆按当前列表重建。
        this.currentTokens = Math.max(0, this.currentTokens - this.countTokens(removedMemory.content));
        this.rebuildHeap();
        return true;
    }

    /**
     * 检查记忆是否存在。
     *
     * @param memoryId 目标记忆 ID
     * @returns 是否存在
     */
    public hasMemory(memoryId: string): boolean {
        return this.memories.some((memory) => memory.id === memoryId);
    }

    /**
     * 清空所有工作记忆。
     */
    public clear(): void {
        this.memories = [];
        this.memoryHeap = [];
        this.currentTokens = 0;
    }

    /**
     * 获取工作记忆统计信息。
     *
     * @returns 统计快照
     */
    public getStats(): Record<string, unknown> {
        this.expireOldMemories();

        // 统计只看当前活跃记忆，不把已过期项算进去。
        const activeMemories = this.getActiveMemories();
        const avgImportance = activeMemories.length > 0
            ? activeMemories.reduce((sum, memory) => sum + memory.importance, 0) / activeMemories.length
            : 0;

        return {
            count: activeMemories.length,
            forgottenCount: 0,
            totalCount: this.memories.length,
            currentTokens: this.currentTokens,
            maxCapacity: this.maxCapacity,
            maxTokens: this.maxTokens,
            maxAgeMinutes: this.maxAgeMinutes,
            sessionDurationMinutes: (Date.now() - this.sessionStart) / 60000,
            avgImportance: avgImportance,
            capacityUsage: this.maxCapacity > 0 ? activeMemories.length / this.maxCapacity : 0,
            tokenUsage: this.maxTokens > 0 ? this.currentTokens / this.maxTokens : 0,
            memoryType: "working",
        };
    }

    /**
     * 获取最近的记忆。
     *
     * @param limit 返回数量上限
     * @returns 最近记忆列表
     */
    public getRecent(limit: number = 10): MemoryItem[] {
        return [...this.getActiveMemories()]
            .sort((left, right) => (right.timestamp ?? 0) - (left.timestamp ?? 0))
            .slice(0, Math.max(0, Math.floor(limit)))
            .map((memory) => this.cloneMemoryItem(memory));
    }

    /**
     * 获取最重要的记忆。
     *
     * @param limit 返回数量上限
     * @returns 按重要性排序的记忆列表
     */
    public getImportant(limit: number = 10): MemoryItem[] {
        return [...this.getActiveMemories()]
            .sort((left, right) => right.importance - left.importance || (right.timestamp ?? 0) - (left.timestamp ?? 0))
            .slice(0, Math.max(0, Math.floor(limit)))
            .map((memory) => this.cloneMemoryItem(memory));
    }

    /**
     * 获取当前所有活跃记忆。
     *
     * @returns 工作记忆快照
     */
    public getAll(): MemoryItem[] {
        return this.getActiveMemories().map((memory) => this.cloneMemoryItem(memory));
    }

    /**
     * 获取上下文摘要。
     *
     * @param maxLength 摘要最大长度
     * @returns 可直接拼接到提示词中的摘要文本
     */
    public getContextSummary(maxLength: number = 500): string {
        const activeMemories = this.getActiveMemories();
        if (activeMemories.length === 0) {
            return "No working memories available.";
        }

        // 先按重要性，再按时间排序，确保摘要优先保留最值得带入上下文的内容。
        const sortedMemories = [...activeMemories].sort((left, right) => {
            if (right.importance !== left.importance) {
                return right.importance - left.importance;
            }
            return (right.timestamp ?? 0) - (left.timestamp ?? 0);
        });

        const summaryParts: string[] = [];
        let currentLength = 0;

        for (const memory of sortedMemories) {
            const content = memory.content;
            if (currentLength + content.length <= maxLength) {
                summaryParts.push(content);
                currentLength += content.length;
            } else {
                // 最后一条可以截断，但只在剩余空间足够时保留，避免摘要太碎。
                const remaining = maxLength - currentLength;
                if (remaining > 50) {
                    summaryParts.push(content.slice(0, remaining) + "...");
                }
                break;
            }
        }

        return "Working Memory Context:\n" + summaryParts.join("\n");
    }

    /**
     * 执行工作记忆遗忘。
     *
     * @param strategy 遗忘策略
     * @param threshold 重要性阈值
     * @param maxAgeDays 时间策略下的最大保留天数
     * @returns 实际删除的数量
     */
    public forget(
        strategy: "importance_based" | "time_based" | "capacity_based" = "importance_based",
        threshold: number = 0.1,
        maxAgeDays: number = 1
    ): number {
        let forgottenCount = 0;
        const now = Date.now();

        const toRemove = new Set<string>();

        // TTL 永远优先，其他策略只是在此基础上继续筛选。
        const ttlCutoff = now - this.maxAgeMinutes * 60 * 1000;
        for (const memory of this.memories) {
            if ((memory.timestamp ?? now) < ttlCutoff) {
                toRemove.add(memory.id);
            }
        }

        if (strategy === "importance_based") {
            // 重要性策略：先删明显不重要的内容。
            for (const memory of this.memories) {
                if (memory.importance < threshold) {
                    toRemove.add(memory.id);
                }
            }
        } else if (strategy === "time_based") {
            // 时间策略：按天级保留窗口清理更旧的数据。
            const cutoff = now - maxAgeDays * 24 * 60 * 60 * 1000;
            for (const memory of this.memories) {
                if ((memory.timestamp ?? now) < cutoff) {
                    toRemove.add(memory.id);
                }
            }
        } else if (strategy === "capacity_based") {
            // 容量策略：保留最值得留在工作记忆里的那部分。
            if (this.memories.length > this.maxCapacity) {
                const ordered = [...this.memories].sort((left, right) => this.calculatePriority(left) - this.calculatePriority(right));
                const excessCount = this.memories.length - this.maxCapacity;
                for (const memory of ordered.slice(0, excessCount)) {
                    toRemove.add(memory.id);
                }
            }
        }

        for (const memoryId of toRemove) {
            if (this.remove(memoryId)) {
                forgottenCount += 1;
            }
        }

        return forgottenCount;
    }

    /** 强制执行容量和 token 限制。 */
    private enforceCapacityLimits() {
        // 如果条数超限，就持续移除最低优先级项。
        while (this.memories.length > this.maxCapacity) {
            this.removeLowestPriorityMemory();
        }

        // 如果 token 超限，也用同一套回收逻辑处理。
        while (this.currentTokens > this.maxTokens) {
            this.removeLowestPriorityMemory();
        }
    }

    /**
     * 计算记忆优先级。
     *
     * @param memoryItem 目标记忆
     * @returns 优先级分数
     */
    private calculatePriority(memoryItem: MemoryItem) {
        const importance = this.clampImportance(memoryItem.importance);
        // 时间衰减会把旧内容的权重慢慢压下去。
        return importance * this.calculateTimeDecay(memoryItem.timestamp);
    }

    /**
     * 对当前候选记忆做轻量 TF-IDF 召回。
     *
     * @param query 查询文本
     * @param memories 候选记忆
     * @returns 记忆 ID 到分数的映射
     */
    private tryTfidfSearch(query: string, memories: MemoryItem[]): VectorScores {
        const scores: VectorScores = new Map();
        const normalizedQuery = query.trim().toLowerCase();

        if (!normalizedQuery || memories.length === 0) {
            return scores;
        }

        try {
            // 第一轮构造文档集：查询在前，候选记忆在后。
            const documents = [normalizedQuery, ...memories.map((memory) => memory.content.toLowerCase())];
            const tokenizedDocuments = documents.map((document) => this.tokenize(document));
            const vocabulary = new Set<string>();

            // 收集全量词表，用来做简化版 TF-IDF。
            for (const tokens of tokenizedDocuments) {
                for (const token of tokens) {
                    vocabulary.add(token);
                }
            }

            if (vocabulary.size === 0) {
                return scores;
            }

            const documentFrequency = new Map<string, number>();
            for (const token of vocabulary) {
                let frequency = 0;
                for (const tokens of tokenizedDocuments) {
                    if (tokens.includes(token)) {
                        frequency += 1;
                    }
                }
                documentFrequency.set(token, frequency);
            }

            const idf = new Map<string, number>();
            const documentCount = tokenizedDocuments.length;
            for (const token of vocabulary) {
                const frequency = documentFrequency.get(token) ?? 0;
                idf.set(token, Math.log((documentCount + 1) / (frequency + 1)) + 1);
            }

            // 把每个文档转换成稀疏向量，再用余弦相似度比较。
            const vectors = tokenizedDocuments.map((tokens) => {
                const termCounts = new Map<string, number>();
                for (const token of tokens) {
                    termCounts.set(token, (termCounts.get(token) ?? 0) + 1);
                }

                const vector = new Map<string, number>();
                let normSquared = 0;
                const totalTerms = Math.max(1, tokens.length);

                for (const token of vocabulary) {
                    const tf = (termCounts.get(token) ?? 0) / totalTerms;
                    const weight = tf * (idf.get(token) ?? 0);
                    if (weight !== 0) {
                        vector.set(token, weight);
                        normSquared += weight * weight;
                    }
                }

                return {
                    vector,
                    norm: Math.sqrt(normSquared),
                };
            });

            const queryVector = vectors[0];
            const queryWeights = queryVector.vector;

            for (let index = 1; index < vectors.length; index += 1) {
                const candidate = vectors[index];
                if (queryVector.norm === 0 || candidate.norm === 0) {
                    scores.set(memories[index - 1].id, 0);
                    continue;
                }

                // 利用稀疏向量的交集部分计算点积，避免不必要的全量遍历。
                let dotProduct = 0;
                const smaller = queryWeights.size <= candidate.vector.size ? queryWeights : candidate.vector;
                const larger = smaller === queryWeights ? candidate.vector : queryWeights;

                for (const [token, weight] of smaller.entries()) {
                    const otherWeight = larger.get(token);
                    if (otherWeight !== undefined) {
                        dotProduct += weight * otherWeight;
                    }
                }

                scores.set(memories[index - 1].id, dotProduct / (queryVector.norm * candidate.norm));
            }
        } catch {
            return new Map();
        }

        return scores;
    }

    /**
     * 计算关键词匹配分数。
     *
     * @param query 查询文本
     * @param content 记忆内容
     * @returns 关键词命中分数
     */
    private calculateKeywordScore(query: string, content: string) {
        const queryLower = query.trim().toLowerCase();
        const contentLower = content.toLowerCase();

        if (!queryLower || !contentLower) {
            return 0;
        }

        if (contentLower.includes(queryLower)) {
            // 完整包含时直接给一个更强的精确命中分。
            return queryLower.length / contentLower.length;
        }

        // 否则退化成词级交集，适合短查询和口语化输入。
        const queryWords = new Set(this.tokenize(queryLower));
        const contentWords = new Set(this.tokenize(contentLower));
        if (queryWords.size === 0 || contentWords.size === 0) {
            return 0;
        }

        let intersectionCount = 0;
        for (const token of queryWords) {
            if (contentWords.has(token)) {
                intersectionCount += 1;
            }
        }

        if (intersectionCount === 0) {
            return 0;
        }

        const unionSize = new Set([...queryWords, ...contentWords]).size;
        return (intersectionCount / unionSize) * 0.8;
    }

    /**
     * 计算时间衰减因子。
     *
     * @param timestamp 记忆时间戳
     * @returns 衰减权重
     */
    private calculateTimeDecay(timestamp?: number) {
        if (timestamp === undefined) {
            return 1;
        }

        const ageMs = Math.max(0, Date.now() - timestamp);
        const hoursPassed = ageMs / 3600000;
        // 这里用指数衰减，确保旧内容的权重下降得更快。
        const baseDecay = this.config.decayFactor > 0 ? this.config.decayFactor : 0.05;
        const decayFactor = Math.pow(baseDecay, hoursPassed / 6);
        return Math.max(0.1, decayFactor);
    }

    /** 清理超时或被标记遗忘的记忆。 */
    private expireOldMemories() {
        if (this.memories.length === 0) {
            return;
        }

        const cutoff = Date.now() - this.maxAgeMinutes * 60 * 1000;
        const kept: MemoryItem[] = [];
        let removedTokens = 0;

        for (const memory of this.memories) {
            const timestamp = memory.timestamp ?? Date.now();
            const forgotten = memory.metadata?.["forgotten"] === true;
            if (!forgotten && timestamp >= cutoff) {
                kept.push(memory);
            } else {
                // 过期项不再保留，但它们占过的 token 需要回收掉。
                removedTokens += this.countTokens(memory.content);
            }
        }

        if (kept.length === this.memories.length) {
            return;
        }

        // 列表是主索引，变更后必须同步重建堆。
        this.memories = kept;
        this.currentTokens = Math.max(0, this.currentTokens - removedTokens);
        this.rebuildHeap();
    }

    /** 删除当前优先级最低的记忆。 */
    private removeLowestPriorityMemory() {
        if (this.memories.length === 0) {
            return;
        }

        // 这里直接扫列表，而不是从堆顶弹出，是为了保持列表和 token 统计一致。
        let lowestIndex = 0;
        let lowestPriority = this.calculatePriority(this.memories[0]);

        for (let index = 1; index < this.memories.length; index += 1) {
            const candidate = this.memories[index];
            const candidatePriority = this.calculatePriority(candidate);
            if (candidatePriority < lowestPriority) {
                lowestPriority = candidatePriority;
                lowestIndex = index;
            }
        }

        const [removedMemory] = this.memories.splice(lowestIndex, 1);
        this.currentTokens = Math.max(0, this.currentTokens - this.countTokens(removedMemory.content));
        this.rebuildHeap();
    }

    /** 根据当前记忆列表重建堆。 */
    private rebuildHeap() {
        // 任何增删改之后都重建一次，避免堆里残留旧优先级。
        this.memoryHeap = [];
        for (const memory of this.memories) {
            const timestamp = memory.timestamp ?? Date.now();
            Heap.heappush(
                this.memoryHeap,
                { priority: this.calculatePriority(memory), timestamp, memoryItem: memory },
                this.compareMemoryHeapEntries
            );
        }
    }

    /** 获取未被遗忘标记的活跃记忆。 */
    private getActiveMemories(): MemoryItem[] {
        // 工作记忆里被标记 forgotten 的内容不再参与检索和摘要。
        return this.memories.filter((memory) => memory.metadata?.["forgotten"] !== true);
    }

    /** 从检索参数中提取用户 ID。 */
    private extractUserId(kwargs: Record<string, unknown>): string | null {
        // 兼容驼峰和下划线两种传参方式。
        const candidate = kwargs.userId ?? kwargs.user_id;
        return typeof candidate === "string" && candidate.trim() ? candidate : null;
    }

    /** 按空白切分文本。 */
    private tokenize(text: string): string[] {
        // 这是轻量 fallback 分词，不依赖额外库。
        return text
            .split(/\s+/)
            .map((part) => part.trim().toLowerCase())
            .filter(Boolean);
    }

    /** 统计简单 token 数。 */
    private countTokens(content: string): number {
        // 工作记忆的 token 预算直接按空白切词近似估算。
        const trimmed = content.trim();
        return trimmed ? trimmed.split(/\s+/).length : 0;
    }

    /** 将重要性夹紧到 0 到 1 之间。 */
    private clampImportance(value: number): number {
        if (!Number.isFinite(value)) {
            return 0.5;
        }

        // 统一把重要性控制在合法区间，避免排序和阈值判断失真。
        return Math.max(0, Math.min(1, value));
    }

    /** 克隆记忆对象，避免外部修改内部状态。 */
    private cloneMemoryItem(memory: MemoryItem): MemoryItem {
        // 只做浅层结构复制，metadata 单独复制一份，避免引用泄漏。
        return {
            ...memory,
            metadata: memory.metadata ? { ...memory.metadata } : undefined,
        };
    }

    // 堆比较器：优先级高的排前面，优先级相同时更早的时间戳优先。
    private readonly compareMemoryHeapEntries = (a: MemoryHeapEntry, b: MemoryHeapEntry): number => {
        return b.priority - a.priority || a.timestamp - b.timestamp;
    };
}
