// 工作记忆（TTL管理，纯内存）

import { MemoryConfig } from "../config";
import { MemoryStore } from "../storage/store";
import { Memory, MemoryItem } from "./memory";
import { Heap } from "heap-js";

type MemoryHeapEntry = { priority: number; timestamp: number; memoryItem: MemoryItem };

/**
 *  工作记忆实现
    特点：
    - 容量有限（默认50条）+ TTL自动清理
    - 纯内存存储，访问速度极快
    - 混合检索：TF-IDF向量化 + 关键词匹配
 */
export class WorkingMemory extends Memory {
    private maxCapacity: number;
    private maxAgeMinutes: number;
    private maxTokens: number

    private currentTokens: number
    private sessionStart: number

    // 内存存储（工作记忆不需要持久化）
    private memories: MemoryItem[];
    // 使用优先级队列管理记忆
    private memoryHeap: MemoryHeapEntry[];


    constructor(config: MemoryConfig, store: MemoryStore) {
        super(config, store);
        this.maxCapacity = config.workingMemoryCapacity;
        this.maxAgeMinutes = config.workingMemoryTTL;
        this.maxTokens = config.workingMemoryTokens;
        this.memories = []
        this.memoryHeap = []
        this.currentTokens = 0;
        this.sessionStart = new Date().getTime();
    }

    /**
     * 添加工作记忆
     * @param memoryItem 
     * @returns 
     */
    public add(memoryItem: MemoryItem): string {
        // 添加工作记忆
        this.expireOldMemories(); // 过期清理

        // 计算优先级（重要性 + 时间衰减）
        const priority = this.calculatePriority(memoryItem)
        const timestamp = memoryItem.timestamp ?? Date.now();
        memoryItem.timestamp = timestamp;

        // 添加到堆中
        Heap.heappush(this.memoryHeap, 
            { priority, timestamp, memoryItem },
            (a, b) => b.priority - a.priority || a.timestamp - b.timestamp
        );
        // 添加到队列中
        this.memories.push(memoryItem)

        // 更新 token 计数
        this.currentTokens += memoryItem.content.trim()
            ? memoryItem.content.trim().split(/\s+/).length
            : 0;

        // # 检查容量限制
        this.enforceCapacityLimits()

        return memoryItem.id;
    }

    /**
     * 混合检索：TF-IDF 向量化 + 关键词匹配
     * @param query 
     * @param limit 
     * @param kwargs 
     * @returns 
     */
    public retrieve(
        query: string,
        limit: number = 5,
        kwargs: Record<string, unknown> = {}
    ): MemoryItem[] {

        // 遗忘过期记忆
        this.expireOldMemories();

        // 尝试 TF-IDF 向量检索
        const vectorScores: VectorScores = this.tryTfidfSearch(query);

        // 计算综合分数
        const scoredMemories: Array<{ score: number; memory: MemoryItem; }> = [];

        for (const memory of this.memories) {
            const vectorScore = vectorScores.get(memory.id) ?? 0.0;
            const keywordScore = this.calculateKeywordScore(query, memory.content);

            // 混合评分
            const baseRelevance = vectorScore > 0
                ? vectorScore * 0.7 + keywordScore * 0.3
                : keywordScore;

            const timeDecay = this.calculateTimeDecay(memory.timestamp);
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
            .slice(0, limit)
            .map(({ memory }) => memory);
    }


    private enforceCapacityLimits() {

    }

    private calculatePriority(memoryItem: MemoryItem) {
        return 0;
    }

    private tryTfidfSearch(query: string) {

    }

    private calculateKeywordScore(query: string, content: string) {
        return 0;
    }

    private calculateTimeDecay(timestamp?: number) {
        return 0;
    }


    private expireOldMemories() {

    }

    private removeLowestPriorityMemory() {

    }
}
