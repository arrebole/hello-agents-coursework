// 记忆管理器（统一协调调度）
// 工作记忆，情感记忆, 语义记忆，感知记忆
export type MemoryType = "working" | "episodic" | "semantic" | "perceptual"

export const MemoryTypeLabels: Record<string, string> = {
    "working": "工作记忆",
    "episodic": "情景记忆",
    "semantic": "语义记忆",
    "perceptual": "感知记忆"
};


export class MemoryConfig { }
export class MemoryStore {
    constructor(config: MemoryConfig) { }
}
export class MemoryRetriever {
    constructor(store: MemoryStore, config: MemoryConfig) { }
}


abstract class Memory {
    constructor(config: MemoryConfig, store: MemoryStore) { }
}
class WorkingMemory extends Memory { }
class EpisodicMemory extends Memory { }
class SemanticMemory extends Memory { }
class PerceptualMemory extends Memory { }


export class MemoryManager {

    private config: MemoryConfig
    private userId: string
    private store: MemoryStore
    private retriever: MemoryRetriever
    private memoryTypes: Map<MemoryType, Memory>

    constructor(
        config: MemoryConfig | null = null,
        userId: string = "default_user",
        enableWorking: boolean = true,
        enableEpisodic: boolean = true,
        enableSemantic: boolean = true,
        enablePerceptual: boolean = false
    ) {
        this.config = config || new MemoryConfig();
        this.userId = userId;

        // 初始化存储和检索组件
        this.store = new MemoryStore(this.config);
        this.retriever = new MemoryRetriever(this.store, this.config);

        // 初始化记忆仓库
        this.memoryTypes = new Map<MemoryType, Memory>();
        if (enableWorking) {
            this.memoryTypes.set('working', new WorkingMemory(this.config, this.store));
        }
        if (enableEpisodic) {
            this.memoryTypes.set('episodic', new EpisodicMemory(this.config, this.store));
        }
        if (enableSemantic) {
            this.memoryTypes.set('semantic', new SemanticMemory(this.config, this.store));
        }
        if (enablePerceptual) {
            this.memoryTypes.set('perceptual', new PerceptualMemory(this.config, this.store));
        }
    }
}