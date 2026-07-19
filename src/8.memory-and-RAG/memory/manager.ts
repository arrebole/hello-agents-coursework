// 记忆管理器（统一协调调度）

import { MemoryConfig } from "./config";
import { MemoryStore } from "./storage/store";
import { EpisodicMemory } from "./types/episodic";
import { Memory, MemoryItem } from "./types/memory";
import { PerceptualMemory } from "./types/perceptual";
import { SemanticMemory } from "./types/semantic";
import { WorkingMemory } from "./types/working";

export { MemoryConfig } from "./config";

/**
 * 记忆类型枚举的字符串联合值。
 *
 * 这四类对应系统中不同生命周期和用途的记忆层：
 * - working: 短期工作记忆
 * - episodic: 情景记忆
 * - semantic: 语义记忆
 * - perceptual: 感知记忆
 */
export type MemoryType = "working" | "episodic" | "semantic" | "perceptual";

/**
 * 记忆遗忘策略。
 *
 * - importance_based: 按重要性阈值清理
 * - time_based: 按时间过期清理
 * - capacity_based: 按容量上限回收
 */
export type ForgetType = "importance_based" | "time_based" | "capacity_based";

/**
 * 记忆类型中文标签映射。
 *
 * 主要用于展示层，把内部类型键转换成更适合输出给用户的文本。
 */
export const MemoryTypeLabels: Record<string, string> = {
    working: "工作记忆",
    episodic: "情景记忆",
    semantic: "语义记忆",
    perceptual: "感知记忆",
};


/**
 * 单类型记忆统计。
 *
 * 只保留最常用的两个指标：数量和平均重要性，便于上层快速展示系统状态。
 */
export interface MemoryTypeStats {
    count: number;
    avgImportance: number;
}

/**
 * 记忆系统统计结果。
 *
 * 这份结构是 `getMemoryStats` 的返回值，供 `memory_tool` 的 `stats` / `summary`
 * 直接使用。
 */
export interface MemoryStats {
    userId: string;
    totalMemories: number;
    memoriesByType: Record<string, MemoryTypeStats>;
}

/**
 * 每一种记忆类型在管理器内部对应一个桶。
 *
 * - instance: 该类型的具体实现实例，保留兼容性
 * - records: 当前用户下这类记忆的内存索引
 */
type MemoryBucket = {
    instance: Memory;
    records: Map<string, MemoryItem>;
};

/**
 * 默认重要性。
 *
 * 当调用方没有显式传值时，避免出现“重要性缺失”的空状态。
 */
const DEFAULT_IMPORTANCE = 0.5;
/**
 * 工作记忆默认容量上限。
 *
 * 超过后会优先丢弃低重要性、较旧的记录。
 */
const DEFAULT_CAPACITY_LIMIT = 20;

/**
 * 记忆检索器的占位实现。
 *
 * 当前版本的检索逻辑主要在 `MemoryManager` 内部完成，这个类先保留为稳定入口，
 * 方便以后替换成向量检索、全文检索或外部存储检索实现。
 */
export class MemoryRetriever {
    /**
     * 创建一个检索器。
     *
     * @param store 底层记忆存储抽象
     * @param config 记忆系统配置
     */
    constructor(private store: MemoryStore, private config: MemoryConfig) {
        // 当前版本的检索逻辑主要在 MemoryManager 内部完成；
        // 这里保留对象是为了后续如果接入向量检索或外部存储时可以平滑扩展。
        void this.store;
        void this.config;
    }
}

/**
 * 记忆管理器。
 *
 * 负责统一协调不同类型记忆的写入、检索、更新、删除、遗忘、整合和统计。
 * 对外暴露的是“统一入口”，对内则按记忆类型维护独立的记录桶。
 */
export class MemoryManager {
    private config: MemoryConfig;
    private userId: string;
    private store: MemoryStore;
    private retriever: MemoryRetriever;
    private memoryTypes: Map<MemoryType, MemoryBucket>;

    /**
     * 创建记忆管理器。
     *
     * @param config 系统配置；不传则使用默认配置
     * @param userId 记忆所属用户 ID
     * @param enableWorking 是否启用工作记忆
     * @param enableEpisodic 是否启用情景记忆
     * @param enableSemantic 是否启用语义记忆
     * @param enablePerceptual 是否启用感知记忆
     */
    constructor(
        config: MemoryConfig | null = null,
        userId: string = "default_user",
        enableWorking: boolean = true,
        enableEpisodic: boolean = true,
        enableSemantic: boolean = true,
        enablePerceptual: boolean = false,
    ) {
        this.config = config || new MemoryConfig();
        this.userId = userId;

        // 存储层与检索层先初始化出来，后续接数据库、向量库时只需要替换实现。
        this.store = new MemoryStore(this.config);
        this.retriever = new MemoryRetriever(this.store, this.config);

        // 这里不是把“类实例”直接当作存储，而是统一包成 bucket，
        // 这样每种记忆类型都能持有自己的记录索引，方便统计、更新和迁移。
        this.memoryTypes = new Map<MemoryType, MemoryBucket>();
        if (enableWorking) {
            this.memoryTypes.set("working", this.createBucket(new WorkingMemory(this.config, this.store)));
        }
        if (enableEpisodic) {
            this.memoryTypes.set("episodic", this.createBucket(new EpisodicMemory(this.config, this.store)));
        }
        if (enableSemantic) {
            this.memoryTypes.set("semantic", this.createBucket(new SemanticMemory(this.config, this.store)));
        }
        if (enablePerceptual) {
            this.memoryTypes.set("perceptual", this.createBucket(new PerceptualMemory(this.config, this.store)));
        }
    }

    /**
     * 添加一条记忆。
     *
     * 该方法负责补齐公共字段、写入对应桶、执行工作记忆容量控制，
     * 并在需要时触发工作记忆到情景记忆的整合。
     *
     * @param content 记忆内容
     * @param memoryType 目标记忆类型
     * @param importance 记忆重要性，范围 0 到 1
     * @param metadata 附加元数据
     * @param skipConsolidation 是否跳过自动整合
     * @returns 新增记忆的 ID
     */
    public addMemory(
        content: string,
        memoryType: MemoryType,
        importance: number = DEFAULT_IMPORTANCE,
        metadata: Record<string, unknown> = {},
        skipConsolidation: boolean = false,
    ): string {
        // 第一步：确认目标记忆类型已经启用。
        // 如果调用方传了一个未启用的类型，这里直接抛错，比“悄悄写丢”更安全。
        const bucket = this.getBucket(memoryType);
        if (!bucket) {
            throw new Error(`不支持的记忆类型: ${memoryType}`);
        }

        // 统一在管理器层补齐公共字段：ID、所属用户、时间戳、重要性。
        // 这样上层只关心“写什么”，不用重复构造完整记录。
        const record: MemoryItem = {
            id: this.generateId(),
            content,
            memoryType,
            importance: this.clampImportance(importance),
            metadata: { ...metadata },
            userId: this.userId,
            timestamp: Date.now(),
        };

        bucket.records.set(record.id, this.cloneRecord(record));

        // 工作记忆是短期缓冲区，先做容量约束，避免无限增长。
        // 这里不会影响其它类型，因为长期记忆本来就允许积累更多内容。
        if (memoryType === "working") {
            this.enforceWorkingCapacity(bucket);
        }

        // 工作记忆达到较高重要性时，自动尝试整合到情景记忆。
        // skipConsolidation 给内部批量迁移场景使用，避免重复触发整合。
        if (!skipConsolidation && memoryType === "working" && this.shouldConsolidate(record)) {
            this.consolidateMemories("working", "episodic", this.getConsolidationThreshold());
        }

        return record.id;
    }

    /**
     * 检索记忆。
     *
     * 该实现采用轻量评分模型：
     * - 重要性作为基础分
     * - 内容和元数据命中会加分
     * - 最终按相关度、重要性、时间排序
     *
     * @param query 查询文本
     * @param limit 返回数量上限
     * @param memoryTypes 限定搜索的记忆类型列表；不传则搜索所有启用类型
     * @param minImportance 最低重要性阈值
     * @returns 排序后的记忆记录列表
     */
    public retrieveMemories(
        query: string,
        limit: number = 10,
        memoryTypes: string[] | null = null,
        minImportance: number = 0,
    ): MemoryItem[] {
        // memoryTypes 为空时，默认在所有启用的类型里搜索。
        const typesToSearch = this.normalizeMemoryTypes(memoryTypes);
        // 查询串统一转成小写，后续比较时只做简单包含判断，降低噪声。
        const normalizedQuery = query.trim().toLowerCase();
        const results: Array<{ record: MemoryItem; score: number }> = [];

        for (const memoryType of typesToSearch) {
            const bucket = this.getBucket(memoryType);
            if (!bucket) continue;

            for (const record of bucket.records.values()) {
                // 重要性低于阈值的记录直接跳过，避免把明显不重要的内容带到结果里。
                if (record.importance < minImportance) continue;

                // 这里没有接入真正的向量召回，所以用一个轻量评分：
                // 重要性作为基础分，内容命中和元数据命中再加分。
                const score = this.scoreRecord(record, normalizedQuery);
                if (normalizedQuery && score <= 0) continue;
                results.push({ record: this.cloneRecord(record), score });
            }
        }

        // 先按相关度，再按重要性，再按时间排序，保证“更像答案”的结果在前面。
        // 这种排序方式适合记忆工具：既考虑“像不像”，也考虑“重不重要”，还兼顾“新不新”。
        results.sort((left, right) => {
            if (right.score !== left.score) return right.score - left.score;
            if (right.record.importance !== left.record.importance) {
                return right.record.importance - left.record.importance;
            }
            return (right.record.timestamp ?? 0) - (left.record.timestamp ?? 0);
        });

        return results.slice(0, limit).map((item) => item.record);
    }

    /**
     * 获取当前记忆系统统计信息。
     *
     * 返回总记忆数、按类型的分布以及各类型的平均重要性。
     * 统计对象只计算当前仍保留在内存中的活跃记忆。
     *
     * @returns 记忆统计快照
     */
    public getMemoryStats(): MemoryStats {
        // 先创建一个空壳统计对象，再逐个 memory bucket 填充。
        // 这种写法比边遍历边拼字符串更适合后续扩展字段。
        const stats: MemoryStats = {
            userId: this.userId,
            totalMemories: 0,
            memoriesByType: {},
        };

        for (const [memoryType, bucket] of this.memoryTypes.entries()) {
            // 平均重要性只统计当前仍然保留在桶里的活跃记忆。
            let importanceSum = 0;
            for (const record of bucket.records.values()) {
                importanceSum += record.importance;
            }

            const count = bucket.records.size;
            // 每种类型都单独记录 count / avgImportance，方便上层按类型单独展示。
            stats.memoriesByType[memoryType] = {
                count,
                avgImportance: count > 0 ? importanceSum / count : 0,
            };
            stats.totalMemories += count;
        }

        return stats;
    }

    /**
     * 执行遗忘/清理。
     *
     * @param strategy 遗忘策略
     * @param threshold 重要性阈值或容量策略的辅助参数
     * @param maxAgeDays 时间策略下的最大保留天数
     * @returns 实际删除的记忆数量
     */
    public forgetMemories(strategy: ForgetType, threshold: number, maxAgeDays: number): number {
        let forgotten = 0;
        const now = Date.now();
        const maxAgeMs = maxAgeDays * 24 * 60 * 60 * 1000;

        for (const bucket of this.memoryTypes.values()) {
            for (const [memoryId, record] of bucket.records.entries()) {
                let shouldRemove = false;

                // 三种遗忘策略分别对应：重要性过低、时间过旧、容量清理。
                // 这里把判断逻辑集中在一起，便于以后再加新的遗忘策略。
                if (strategy === "importance_based") {
                    shouldRemove = record.importance < threshold;
                } else if (strategy === "time_based") {
                    shouldRemove = now - (record.timestamp ?? now) > maxAgeMs;
                } else if (strategy === "capacity_based") {
                    shouldRemove = false;
                }

                if (shouldRemove) {
                    bucket.records.delete(memoryId);
                    forgotten += 1;
                }
            }
        }

        if (strategy === "capacity_based") {
            // 容量策略不直接按单条记录判断，而是先让每个桶回收到目标容量。
            for (const bucket of this.memoryTypes.values()) {
                forgotten += this.trimToCapacity(bucket, this.getCapacityLimit());
            }
        }

        return forgotten;
    }

    /**
     * 将一类记忆整合到另一类记忆中。
     *
     * 常见场景是把重要的工作记忆提升到情景记忆中，模拟短期信息固化。
     *
     * @param fromType 源记忆类型
     * @param toType 目标记忆类型
     * @param importanceThreshold 参与整合的最低重要性
     * @returns 被整合的记忆数量
     */
    public consolidateMemories(fromType: MemoryType, toType: MemoryType, importanceThreshold: number): number {
        if (fromType === toType) {
            // 同类型之间没有迁移意义，直接返回即可。
            return 0;
        }

        const sourceBucket = this.getBucket(fromType);
        const targetBucket = this.getBucket(toType);
        if (!sourceBucket || !targetBucket) {
            // 源类型或目标类型未启用时，不做任何迁移。
            return 0;
        }

        // 只把高于阈值的源记忆迁移出去。
        // 这样可以把“短期里已经足够重要的信息”提升到更稳定的记忆层。
        const candidates = [...sourceBucket.records.values()]
            .filter((record) => record.importance >= importanceThreshold)
            .sort((left, right) => right.importance - left.importance);

        let consolidated = 0;
        for (const record of candidates) {
            // 迁移前先从源桶移除，避免同一条记录在两个桶里重复存在。
            if (!sourceBucket.records.delete(record.id)) continue;

            // 迁移到目标类型时保留原 ID，便于上层继续引用同一条记忆。
            // 同时略微提升重要性，表示它已经被固化进更长期的记忆层。
            const movedRecord = this.cloneRecord({
                ...record,
                memoryType: toType,
                importance: this.clampImportance(record.importance * 1.1),
            });

            targetBucket.records.set(movedRecord.id, movedRecord);
            consolidated += 1;
        }

        return consolidated;
    }

    /**
     * 更新已有记忆。
     *
     * 只修改调用方显式传入的字段，其余字段保持原值。
     *
     * @param memoryId 目标记忆 ID
     * @param content 新内容；传 null 表示不改
     * @param importance 新重要性；传 null 表示不改
     * @param metadata 新元数据；传 null 表示不改
     * @returns 是否更新成功
     */
    public updateMemory(
        memoryId: string,
        content: string | null = null,
        importance: number | null = null,
        metadata: Record<string, unknown> | null = null,
    ): boolean {
        const found = this.findMemory(memoryId);
        if (!found) {
            // ID 不存在时直接失败，由上层决定是否提示“未找到”。
            return false;
        }

        const { bucket, record } = found;
        // 只改调用方明确传入的字段，其他字段保持原值。
        const updated: MemoryItem = {
            ...record,
            content: content ?? record.content,
            importance: importance === null ? record.importance : this.clampImportance(importance),
            metadata: metadata === null ? record.metadata : { ...(record.metadata ?? {}), ...metadata },
            timestamp: Date.now(),
        };

        bucket.records.set(memoryId, this.cloneRecord(updated));
        return true;
    }

    /**
     * 删除指定 ID 的记忆。
     *
     * 由于调用方通常只知道 ID，不知道它属于哪个类型，所以这里会跨所有桶查找。
     *
     * @param memoryId 目标记忆 ID
     * @returns 是否删除成功
     */
    public removeMemory(memoryId: string): boolean {
        // 删除操作需要跨所有类型查找，因为调用方通常只知道 ID，不知道它属于哪一类记忆。
        for (const bucket of this.memoryTypes.values()) {
            if (bucket.records.delete(memoryId)) {
                return true;
            }
        }

        return false;
    }

    /**
     * 清空所有记忆类型中的数据。
     *
     * 注意：这里只清理记录，不会改变当前启用的记忆类型开关。
     */
    public clearAllMemories(): void {
        // 逐桶清空即可，不需要重建 Map；这样外部持有的类型开关状态也不变。
        for (const bucket of this.memoryTypes.values()) {
            bucket.records.clear();
        }
    }

    /**
     * 仅清理工作记忆。
     *
     * 这通常用于开始新对话时重置短期上下文，而保留长期记忆。
     */
    public clearWorkingMemory(): void {
        // 这里只清工作记忆，不碰长期记忆。
        // 这是“开启新对话”时最常见的操作：忘掉上下文缓存，但保留知识和经历。
        const bucket = this.getBucket("working");
        if (bucket) {
            bucket.records.clear();
        }
    }

    /**
     * 创建一个记忆桶。
     *
     * @param instance 该类型的具体实现实例
     * @returns 包含实例引用和空记录索引的桶
     */
    private createBucket(instance: Memory): MemoryBucket {
        // 每个类型的 bucket 都有自己的实例引用和记录表。
        // 记录表才是真正的数据源，instance 主要用于未来扩展。
        return {
            instance,
            records: new Map<string, MemoryItem>(),
        };
    }

    /**
     * 根据类型获取记忆桶。
     *
     * @param memoryType 记忆类型
     * @returns 对应桶；如果该类型未启用则返回 undefined
     */
    private getBucket(memoryType: MemoryType): MemoryBucket | undefined {
        // Map.get 直接封装一层，避免外部代码到处直接碰 memoryTypes。
        return this.memoryTypes.get(memoryType);
    }

    /**
     * 归一化检索类型列表。
     *
     * 规则是：
     * - 不传时，搜索所有启用类型
     * - 传入非法值时，过滤掉
     * - 只保留当前已经启用的类型
     *
     * @param memoryTypes 外部传入的类型列表
     * @returns 可安全使用的记忆类型列表
     */
    private normalizeMemoryTypes(memoryTypes: string[] | null): MemoryType[] {
        // 不传类型时，搜索所有已启用类型。
        if (!memoryTypes || memoryTypes.length === 0) {
            return [...this.memoryTypes.keys()];
        }

        // 只保留合法且已启用的类型，避免外部传错字符串导致检索异常。
        const validTypes = new Set<MemoryType>();
        for (const memoryType of memoryTypes) {
            if (this.isMemoryType(memoryType) && this.memoryTypes.has(memoryType)) {
                validTypes.add(memoryType);
            }
        }

        return validTypes.size > 0 ? [...validTypes] : [...this.memoryTypes.keys()];
    }

    /**
     * 按 ID 查找记忆。
     *
     * 因为 ID 是全局唯一的，所以可以直接在所有桶里顺序查找。
     *
     * @param memoryId 目标记忆 ID
     * @returns 找到时返回桶和记录，否则返回 null
     */
    private findMemory(memoryId: string): { bucket: MemoryBucket; record: MemoryItem } | null {
        // 因为 ID 全局唯一，所以顺序扫一遍所有桶即可定位到目标记录。
        for (const bucket of this.memoryTypes.values()) {
            const record = bucket.records.get(memoryId);
            if (record) {
                return { bucket, record };
            }
        }

        return null;
    }

    /**
     * 为单条记录计算检索分数。
     *
     * 当前是轻量规则评分，不依赖向量库：
     * - 空查询时直接返回重要性
     * - 内容整段命中加高分
     * - 分词命中内容或元数据再加分
     *
     * @param record 目标记录
     * @param query 标准化后的查询文本
     * @returns 该记录的检索分数
     */
    private scoreRecord(record: MemoryItem, query: string): number {
        if (!query) {
            // 空查询时，直接退化成按重要性排序，常用于 summary 之类的场景。
            return record.importance;
        }

        const content = record.content.toLowerCase();
        const metadataText = JSON.stringify(record.metadata ?? {}).toLowerCase();
        let score = record.importance;

        // 直接整段命中时给更高权重，适合短查询和标签式内容。
        if (content.includes(query)) {
            score += 1.5;
        }

        // 分词后做轻量命中，避免必须整句完全一致才返回结果。
        for (const token of this.tokenize(query)) {
            if (!token) continue;
            if (content.includes(token)) score += 0.4;
            if (metadataText.includes(token)) score += 0.2;
        }

        return score;
    }

    /**
     * 对查询文本做最轻量的分词处理。
     *
     * 这里仅按空白切分，适合作为 fallback 规则；
     * 如果未来接入更复杂的中文分词，可以在这里替换。
     *
     * @param text 输入文本
     * @returns 分词结果
     */
    private tokenize(text: string): string[] {
        // 这里是最简单的分词策略：按空白切分。
        // 对中文来说它不够聪明，但足够作为轻量 fallback。
        return text
            .split(/\s+/)
            .map((part) => part.trim().toLowerCase())
            .filter(Boolean);
    }

    /**
     * 判断工作记忆是否应该自动整合到长期记忆。
     *
     * @param record 待判断的记录
     * @returns 是否需要整合
     */
    private shouldConsolidate(record: MemoryItem): boolean {
        // 高重要性工作记忆会被进一步提到情景记忆，模拟“短期固化”。
        return record.importance >= this.getConsolidationThreshold();
    }

    /**
     * 获取整合阈值。
     *
     * 如果配置里没有明确值，则回退到默认阈值。
     *
     * @returns 重要性阈值
     */
    private getConsolidationThreshold(): number {
        // 如果配置里没填，就回退到 0.7，这个值和上层工具默认值保持一致。
        return this.clampImportance(this.config.importanceThreshold ?? 0.7);
    }

    /**
     * 获取容量上限。
     *
     * 配置值必须至少为 1，避免错误配置导致所有记忆都被立刻清空。
     *
     * @returns 当前允许的最大记忆条数
     */
    private getCapacityLimit(): number {
        // capacity 是一个正整数上限，至少保证为 1，避免被错误配置成 0 或负数。
        const configured = this.config.maxCapacity ?? DEFAULT_CAPACITY_LIMIT;
        return Math.max(1, Math.floor(configured));
    }

    /**
     * 对工作记忆执行容量约束。
     *
     * 超过上限时，优先删除低重要性、较旧的条目。
     *
     * @param bucket 工作记忆桶
     */
    private enforceWorkingCapacity(bucket: MemoryBucket): void {
        const workingLimit = this.getCapacityLimit();
        if (bucket.records.size <= workingLimit) {
            return;
        }

        // 工作记忆超量时，优先删除低重要性、较旧的项。
        // 这和人类短期记忆“先忘掉不重要的、旧的”这个直觉保持一致。
        const ordered = [...bucket.records.values()].sort((left, right) => {
            if (left.importance !== right.importance) {
                return left.importance - right.importance;
            }
            return (left.timestamp ?? 0) - (right.timestamp ?? 0);
        });

        while (bucket.records.size > workingLimit && ordered.length > 0) {
            const record = ordered.shift();
            if (record) {
                bucket.records.delete(record.id);
            }
        }
    }

    /**
     * 将一个桶裁剪到指定容量。
     *
     * 这是通用回收逻辑，和工作记忆的收缩规则保持一致：
     * 先删低重要性、再删更旧的。
     *
     * @param bucket 目标桶
     * @param maxSize 容量上限
     * @returns 实际删除的条数
     */
    private trimToCapacity(bucket: MemoryBucket, maxSize: number): number {
        const resolvedMaxSize = Math.max(1, Math.floor(maxSize));
        if (bucket.records.size <= resolvedMaxSize) {
            return 0;
        }

        // 容量回收和工作记忆收缩使用同一套排序规则，保持行为一致。
        // 先删最不重要、最旧的记录，是最简单也最稳妥的降载方式。
        const ordered = [...bucket.records.values()].sort((left, right) => {
            if (left.importance !== right.importance) {
                return left.importance - right.importance;
            }
            return (left.timestamp ?? 0) - (right.timestamp ?? 0);
        });

        let removed = 0;
        while (bucket.records.size > resolvedMaxSize && ordered.length > 0) {
            const record = ordered.shift();
            if (record && bucket.records.delete(record.id)) {
                removed += 1;
            }
        }

        return removed;
    }

    /**
     * 运行时类型守卫。
     *
     * @param value 任意字符串
     * @returns 是否属于 MemoryType
     */
    private isMemoryType(value: string): value is MemoryType {
        // 运行时类型守卫：把普通字符串收窄成合法的 MemoryType。
        return value === "working" || value === "episodic" || value === "semantic" || value === "perceptual";
    }

    /**
     * 将重要性限制在 0 到 1 之间。
     *
     * @param value 原始重要性值
     * @returns 夹紧后的重要性值
     */
    private clampImportance(value: number): number {
        // 重要性统一夹在 0~1 之间，避免上层传入越界值破坏排序和策略判断。
        if (Number.isNaN(value)) {
            return DEFAULT_IMPORTANCE;
        }
        return Math.max(0, Math.min(1, value));
    }

    /**
     * 生成记忆 ID。
     *
     * 优先使用标准 UUID；如果运行环境不支持，则退回到时间戳加随机串。
     *
     * @returns 全局唯一风格的字符串 ID
     */
    private generateId(): string {
        // 优先使用标准 UUID；没有可用实现时再退回到时间戳 + 随机串。
        if (typeof globalThis.crypto?.randomUUID === "function") {
            return globalThis.crypto.randomUUID();
        }

        return `memory_${Date.now()}_${Math.random().toString(16).slice(2)}`;
    }

    /**
     * 克隆记忆记录。
     *
     * 这是一个防御性拷贝，避免外部拿到内部对象后直接修改 Map 里的原始数据。
     *
     * @param record 原始记录
     * @returns 复制后的记录
     */
    private cloneRecord(record: MemoryItem): MemoryItem {
        // 返回拷贝而不是原对象，避免上层误改内部索引里的数据。
        // 这里尤其要保护 metadata，因为它是引用类型，直接透传容易被意外修改。
        return {
            ...record,
            metadata: record.metadata ? { ...record.metadata } : undefined,
        };
    }

    /**
     * 返回当前管理器的简短字符串表示。
     *
     * @returns 适合日志输出的摘要字符串
     */
    public toString(): string {
        const stats = this.getMemoryStats();
        return `MemoryManager(user=${this.userId}, total=${stats.totalMemories})`;
    }
}
