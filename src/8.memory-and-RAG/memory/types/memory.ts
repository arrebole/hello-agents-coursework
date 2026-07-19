import { MemoryConfig } from "../config";
import { MemoryType } from "../manager";
import { MemoryStore } from "../storage/store";

/**
 * 单条记忆记录。
 *
 * 这是管理器内部最核心的数据结构：每次 `addMemory` 最终都会落成一条记录。
 * 记录中保留内容、归属类型、重要性、元数据，以及用于统计和清理的辅助字段。
 */
export interface MemoryItem {
    id: string;
    content: string;
    memoryType: MemoryType;
    importance: number;
    metadata?: Record<string, unknown>;
    userId?: string;
    timestamp?: number;
}


// 记忆类接口
export abstract class Memory {

    constructor(config: MemoryConfig, store: MemoryStore) { }

    // 增加记忆
    public abstract add(memoryItem: MemoryItem): string;

    // 索引记忆 
    public abstract retrieve(query: string, limit: number, kwargs: any): MemoryItem[];
}