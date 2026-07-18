//  记忆工具（Agent记忆能力）

import { Tool, ToolParameter } from "../../../7.hello-agents/tools/base";
import { MemoryConfig, MemoryManager, MemoryType, MemoryTypeLabels } from "../../memory/manager";

// - add: 添加记忆（支持4种类型: working/episodic/semantic/perceptual）
// - search: 搜索记忆
// - summary: 获取记忆摘要
// - stats: 获取统计信息
// - update: 更新记忆
// - remove: 删除记忆
// - forget: 遗忘记忆（多种策略）
// - consolidate: 整合记忆（短期→长期）
// - clear_all: 清空所有记忆
type MemoryToolAction = "add" | "search" | "summary" | "stats" | "update" | "remove" | "forget" | "consolidate" | "clear_all"

// 遗忘类型
type ForgetType = "importance_based" | "time_based" | "capacity_based";


// 记忆工具
export class MemoryTool extends Tool {
    // 当前的会话ID
    private currentSessionId: string = ""
    private metadata: Map<string, string> = new Map();
    private memoryManager: MemoryManager
    private memoryConfig: MemoryConfig
    private memoryTypes: MemoryType[]

    constructor(
        userId: string = "default_user",
        memoryConfig: MemoryConfig | null = null,
        memoryTypes: MemoryType[] | null = null
    ) {
        super(
            "memory",
            "记忆工具 - 可以存储和检索对话历史、知识和经验"
        );

        // 初始化记忆管理器
        this.memoryConfig = memoryConfig || new MemoryConfig();
        this.memoryTypes = memoryTypes || ["working", "episodic", "semantic"];
        this.memoryManager = new MemoryManager({
            config: this.memoryConfig,
            userId: userId,
            enableWorking: this.memoryTypes.includes("working"),
            enableEpisodic: this.memoryTypes.includes("episodic"),
            enableSemantic: this.memoryTypes.includes("semantic"),
            enablePerceptual: this.memoryTypes.includes("perceptual")
        });
    }

    getParameters(): ToolParameter[] {
        return [
            new ToolParameter({
                name: "action",
                type: "string",
                description: "要执行的操作：add(添加记忆), search(搜索记忆), summary(获取摘要), stats(获取统计), update(更新记忆), remove(删除记忆), forget(遗忘记忆), consolidate(整合记忆), clear_all(清空所有记忆)",
                required: true,
            }),
            new ToolParameter({
                name: "content",
                type: "string",
                description: "记忆内容（add/update时可用；感知记忆可作描述）",
                required: false,
            }),
            new ToolParameter({
                name: "query",
                type: "string",
                description: "搜索查询（search时可用）",
                required: false,
            }),
            new ToolParameter({
                name: "memory_type",
                type: "string",
                description: "记忆类型：working, episodic, semantic, perceptual（默认：working）",
                required: false,
                default: "working",
            }),
            new ToolParameter({
                name: "importance",
                type: "number",
                description: "重要性分数，0.0-1.0（add/update时可用）",
                required: false,
            }),
            new ToolParameter({
                name: "limit",
                type: "integer",
                description: "搜索结果数量限制（默认：5）",
                required: false,
                default: 5,
            }),
            new ToolParameter({
                name: "memory_id",
                type: "string",
                description: "目标记忆ID（update/remove时必需）",
                required: false,
            }),
            new ToolParameter({
                name: "file_path",
                type: "string",
                description: "感知记忆：本地文件路径（image/audio）",
                required: false,
            }),
            new ToolParameter({
                name: "modality",
                type: "string",
                description: "感知记忆模态：text/image/audio（不传则按扩展名推断）",
                required: false,
            }),
            new ToolParameter({
                name: "strategy",
                type: "string",
                description: "遗忘策略：importance_based/time_based/capacity_based（forget时可用）",
                required: false,
                default: "importance_based",
            }),
            new ToolParameter({
                name: "threshold",
                type: "number",
                description: "遗忘阈值（forget时可用，默认0.1）",
                required: false,
                default: 0.1,
            }),
            new ToolParameter({
                name: "max_age_days",
                type: "integer",
                description: "最大保留天数（forget策略为time_based时可用）",
                required: false,
                default: 30,
            }),
            new ToolParameter({
                name: "from_type",
                type: "string",
                description: "整合来源类型（consolidate时可用，默认working）",
                required: false,
                default: "working",
            }),
            new ToolParameter({
                name: "to_type",
                type: "string",
                description: "整合目标类型（consolidate时可用，默认episodic）",
                required: false,
                default: "episodic",
            }),
            new ToolParameter({
                name: "importance_threshold",
                type: "number",
                description: "整合重要性阈值（默认0.7）",
                required: false,
                default: 0.7,
            }),
        ];
    }

    // 执行记忆操作
    run(parameters: Record<string, unknown>): string {
        if (!this.validateParameters(parameters)) {
            return "❌ 参数验证失败：缺少必需的参数";
        }

        const action = parameters["action"];

        switch (action) {
            case "add":
                return this.addMemory({
                    content: parameters["content"],
                    memoryType: parameters["memory_type"] ?? "working",
                    importance: parameters["importance"] ?? 0.5,
                    filePath: parameters["file_path"],
                    modality: parameters["modality"]
                });

            case "search":
                return this._searchMemory({
                    query: parameters.get("query"),
                    limit: parameters.get("limit", 5),
                    memoryType: parameters.get("memory_type"),
                    minImportance: parameters.get("min_importance", 0.1)
                });

            case "summary":
                return this._getSummary({
                    limit: parameters.get("limit", 10)
                });

            case "stats":
                return this._getStats();

            case "update":
                return this._updateMemory({
                    memoryId: parameters.get("memory_id"),
                    content: parameters.get("content"),
                    importance: parameters.get("importance")
                });

            case "remove":
                return this._removeMemory({
                    memoryId: parameters.get("memory_id")
                });

            case "forget":
                return this._forget({
                    strategy: parameters.get("strategy", "importance_based"),
                    threshold: parameters.get("threshold", 0.1),
                    maxAgeDays: parameters.get("max_age_days", 30)
                });

            case "consolidate":
                return this._consolidate({
                    fromType: parameters.get("from_type", "working"),
                    toType: parameters.get("to_type", "episodic"),
                    importanceThreshold: parameters.get("importance_threshold", 0.7)
                });

            case "clear_all":
                return this._clearAll();

            default:
                return `❌ 不支持的操作: ${action}`;
        }
    }

    private addMemory(
        content: string = "",
        memoryType: MemoryType = "working",
        importance: number = 0.5,
        filePath: string | null = null,
        modality: string | null = null,
        metadata: any
    ) {
        // 如果当前不存在会话ID，则通过时间初始化一个会话ID
        if (!this.currentSessionId) {
            const now = new Date();
            const timestamp = now.toISOString()
                .replace(/[-:]/g, '')
                .replace(/\.\d{3}Z$/, '')
                .slice(0, 14); // YYYYMMDD_HHMMSS
            this.currentSessionId = `session_${timestamp}`;
        }

        // 感知记忆文件支持
        if (memoryType === "perceptual" && filePath) {
            const inferred = modality || this.inferModality(filePath);
            metadata.setdefault("modality", inferred)
            metadata.setdefault("raw_data", filePath)
        }

        // 添加会话信息到元数据
        metadata.update({
            "session_id": this.currentSessionId,
            "timestamp": new Date().valueOf()
        })

        // 调用记忆管理器 添加记忆
        const memoryId = this.memoryManager.addMemory(
            content,
            memoryType,
            importance,
            metadata,
            false,
        )

        return `✅ 记忆已添加 (ID: ${memoryId.slice(0, 8)}...)`;
    }

    // 搜索记忆
    private search(
        query: string,
        limit: number = 5,
        memoryTypes: string[] | null = null,
        memoryType: string | null = null,
        minImportance: number = 0.1
    ): string {

        let typesToUse: string[] | null = memoryTypes;
        if (memoryType && !memoryTypes) {
            typesToUse = [memoryType];
        }

        // 通过记忆管理器查询记忆
        const results: any = this.memoryManager.retrieveMemories(
            query,
            limit,
            typesToUse,
            minImportance
        );

        if (!results || results.length === 0) {
            return `🔍 未找到与 '${query}' 相关的记忆`;
        }

        // 将查询到的记忆结果格式化
        const formattedResults: string[] = [];
        formattedResults.push(`🔍 找到 ${results.length} 条相关记忆:`);


        results.forEach((memory: any, index: number) => {
            const label = MemoryTypeLabels[memory.memoryType] || memory.memoryType;

            let contentPreview = memory.content;
            if (memory.content.length > 80) {
                contentPreview = memory.content.slice(0, 80) + "...";
            }

            formattedResults.push(
                `${index + 1}. [${label}] ${contentPreview} (重要性: ${memory.importance.toFixed(2)})`
            );
        });

        return formattedResults.join("\n");
    }

    // 遗忘记忆（支持多种策略）
    private forget(
        strategy: ForgetType = "importance_based",
        threshold: number = 0.1,
        maxAgeDays: number = 30
    ): string {
        const count = this.memoryManager.forgetMemories(
            strategy,
            threshold,
            maxAgeDays
        );
        return `🧹 已遗忘 ${count} 条记忆（策略: ${strategy}）`;
    }

    // 整合记忆（将重要的短期记忆提升为长期记忆）
    // 借鉴了神经科学中的记忆固化概念，模拟人类大脑将短期记忆转化为长期记忆的过程。默认设置是将重要性超过0.7的工作记忆转换为情景记忆，这个阈值确保只有真正重要的信息才会被长期保
    private consolidate(
        fromType: MemoryType = "working",
        toType: MemoryType = "episodic",
        importanceThreshold: number = 0.7
    ): string {
        const count = this.memoryManager.consolidateMemories(
            fromType,
            toType,
            importanceThreshold
        );
        return `🔄 已整合 ${count} 条记忆为长期记忆（${fromType} → ${toType}，阈值=${importanceThreshold}）`;
    }

    // 根据扩展名推断模态(默认image/audio/text)
    private inferModality(path: string): string {
        try {
            // 获取文件后缀名
            const ext = (path.split('.').pop() || '').toLowerCase();

            if (["png", "jpg", "jpeg", "bmp", "gif", "webp"].includes(ext)) {
                return "image";
            }
            if (["mp3", "wav", "flac", "m4a", "ogg"].includes(ext)) {
                return "audio";
            }
            return "text";
        } catch {
            return "text";
        }
    }
}

