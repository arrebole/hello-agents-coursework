// 记忆工具（Agent 记忆能力）
//
// 本文件把「记忆系统」封装成一个标准工具（Tool），可挂到任意 Agent 上。
// 分层关系：
//   Agent → MemoryTool（本文件：参数解析、分发、结果格式化）→ MemoryManager（底层存取，见 ../../memory/manager）
// 对外有两条入口：
//   1) run(parameters)  —— 面向 LLM function-calling：传入 { action, ... } 字典，按 action 分发到各私有方法。
//   2) 便捷方法（addKnowledge / getContextForQuery / autoRecordConversation / ...）—— 面向代码直接调用（Agent 内部）。

import { Tool, ToolParameter } from "../../../7.hello-agents/tools/base";
import { ForgetType, MemoryConfig, MemoryManager, MemoryRecord, MemoryStats, MemoryType, MemoryTypeLabels } from "../../memory/manager";

// 支持的操作集合。
// 用 `as const` 数组作为「单一事实来源」：类型 MemoryToolAction 与运行时守卫 isMemoryToolAction 都从它派生，
// 增删一个操作只改这一处，避免「类型改了、守卫忘改」这类不一致。
const MEMORY_TOOL_ACTIONS = [
    "add", "search", "summary", "stats", "update", "remove", "forget", "consolidate", "clearAll",
] as const;

// 从上面的数组派生出联合字面量类型：等价于 "add" | "search" | ... | "clearAll"
type MemoryToolAction = (typeof MEMORY_TOOL_ACTIONS)[number];

// 记忆工具参数（统一使用驼峰命名）。
// 所有字段都可选：因为一次调用只会用到某个 action 对应的少数字段，
// 且 action 已由 isMemoryToolAction 在 run 入口单独收窄，故此处声明为可选以便后续安全断言。
interface MemoryToolParams {
    action?: MemoryToolAction;
    content?: string;            // 记忆内容（add/update）
    query?: string;              // 搜索查询（search）
    memoryType?: MemoryType;     // 记忆类型：working/episodic/semantic/perceptual
    importance?: number;         // 重要性 0.0-1.0（add/update）
    limit?: number;              // 结果数量上限（search/summary）
    memoryId?: string;           // 目标记忆 ID（update/remove）
    filePath?: string;           // 感知记忆的本地文件路径
    modality?: string;           // 感知记忆模态：text/image/audio
    minImportance?: number;      // 搜索时的最低重要性阈值
    strategy?: ForgetType;       // 遗忘策略（forget）
    threshold?: number;          // 遗忘阈值（forget）
    maxAgeDays?: number;         // 最大保留天数（forget 的 time_based 策略）
    fromType?: MemoryType;       // 整合来源类型（consolidate）
    toType?: MemoryType;         // 整合目标类型（consolidate）
    importanceThreshold?: number;// 整合重要性阈值（consolidate）
}

// 运行时类型守卫：判断任意值是否为受支持的操作。
// 返回类型 `value is MemoryToolAction` 让 TS 在调用点自动收窄——
// 通过此函数后，编译器即认为该值是合法的 action 字面量。
function isMemoryToolAction(value: unknown): value is MemoryToolAction {
    return MEMORY_TOOL_ACTIONS.includes(value as MemoryToolAction);
}


// 记忆工具：对外暴露为名为 "memory" 的工具
export class MemoryTool extends Tool {
    // 当前会话 ID。首次添加记忆时按时间戳惰性生成（见 addMemory），clearSession 时重置为空。
    private currentSessionId: string = ""
    // 对话轮次计数。仅 autoRecordConversation 会递增它，summary/stats 展示时读取。
    private conversationCount: number = 0
    // 底层记忆管理器：真正的存取、检索、遗忘、整合都委托给它
    private memoryManager: MemoryManager
    // 记忆系统配置（当前为占位配置对象）
    private memoryConfig: MemoryConfig
    // 本工具实例启用的记忆类型（构造时决定，供 stats 展示与初始化管理器用）
    private memoryTypes: MemoryType[]

    /**
     * @param userId       记忆归属的用户 ID（多用户隔离时使用）
     * @param memoryConfig 记忆配置；不传则用默认配置
     * @param memoryTypes  启用哪些记忆类型；不传则默认启用 工作/情景/语义 三类（不含感知记忆）
     */
    constructor(
        userId: string = "default_user",
        memoryConfig: MemoryConfig | null = null,
        memoryTypes: MemoryType[] | null = null
    ) {
        // 向基类注册工具名与描述，LLM 通过这段描述认识本工具
        super(
            "memory",
            "记忆工具 - 可以存储和检索对话历史、知识和经验"
        );

        // 初始化记忆管理器
        this.memoryConfig = memoryConfig || new MemoryConfig();
        this.memoryTypes = memoryTypes || ["working", "episodic", "semantic"];
        // 把「启用哪些类型」转成 4 个布尔开关传给管理器：includes 命中即启用对应记忆子系统
        this.memoryManager = new MemoryManager(
            this.memoryConfig,
            userId,
            this.memoryTypes.includes("working"),
            this.memoryTypes.includes("episodic"),
            this.memoryTypes.includes("semantic"),
            this.memoryTypes.includes("perceptual")
        );
    }

    /**
     * 声明本工具的参数 schema（Tool 基类要求实现）。
     * 这份列表有双重用途：
     *   1) 供 run() 前的 validateParameters 校验必填项（这里只有 action 必填）；
     *   2) 生成 function-calling schema 交给 LLM，让模型知道每个参数的名字/类型/默认值。
     * 注意：参数 name 已统一为驼峰，需与 MemoryToolParams 的字段、以及调用方传入的键一致。
     */
    getParameters(): ToolParameter[] {
        return [
            new ToolParameter({
                name: "action",
                type: "string",
                description: "要执行的操作：add(添加记忆), search(搜索记忆), summary(获取摘要), stats(获取统计), update(更新记忆), remove(删除记忆), forget(遗忘记忆), consolidate(整合记忆), clearAll(清空所有记忆)",
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
                name: "memoryType",
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
                name: "memoryId",
                type: "string",
                description: "目标记忆ID（update/remove时必需）",
                required: false,
            }),
            new ToolParameter({
                name: "filePath",
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
                name: "maxAgeDays",
                type: "integer",
                description: "最大保留天数（forget策略为time_based时可用）",
                required: false,
                default: 30,
            }),
            new ToolParameter({
                name: "fromType",
                type: "string",
                description: "整合来源类型（consolidate时可用，默认working）",
                required: false,
                default: "working",
            }),
            new ToolParameter({
                name: "toType",
                type: "string",
                description: "整合目标类型（consolidate时可用，默认episodic）",
                required: false,
                default: "episodic",
            }),
            new ToolParameter({
                name: "importanceThreshold",
                type: "number",
                description: "整合重要性阈值（默认0.7）",
                required: false,
                default: 0.7,
            }),
        ];
    }

    /**
     * 工具统一入口：解析参数字典 → 按 action 分发 → 返回给用户/LLM 的结果字符串。
     * 这是 Tool 基类要求实现的抽象方法，LLM 每次调用本工具最终都走到这里。
     *
     * 两道前置关卡：
     *   1) validateParameters —— 基类校验必填项（缺 action 直接拒绝）；
     *   2) isMemoryToolAction —— 校验 action 取值合法，顺便把类型收窄为 MemoryToolAction。
     * 过关后才把 parameters 断言为 MemoryToolParams，用 `?? 默认值` 兜底可选参数。
     */
    run(parameters: Record<string, unknown>): string {
        // 关卡一：必填参数是否齐全（本工具仅 action 必填）
        if (!this.validateParameters(parameters)) {
            return "❌ 参数验证失败：缺少必需的参数";
        }

        // 关卡二：action 是否为受支持的操作；不合法则直接返回，避免落到 default 之外的未知状态
        if (!isMemoryToolAction(parameters["action"])) {
            return `❌ 不支持的操作: ${String(parameters["action"])}`;
        }

        // 参数名已统一为驼峰，且两道关卡已通过，安全收窄为受约束的参数类型
        const params = parameters as MemoryToolParams;

        // 按 action 分发到对应私有方法；每个 case 用 `?? 默认值` 补齐未传的可选参数
        switch (params.action) {
            case "add":
                return this.addMemory(
                    params.content ?? "",
                    params.memoryType ?? "working",
                    params.importance ?? 0.5,
                    params.filePath ?? null,
                    params.modality ?? null
                );

            case "search":
                return this.search(
                    params.query ?? "",
                    params.limit ?? 5,
                    null,
                    params.memoryType ?? null,
                    params.minImportance ?? 0.1
                );

            case "summary":
                return this.getSummary(params.limit ?? 10);

            case "stats":
                return this.getStats();

            case "update":
                return this.updateMemory(
                    params.memoryId ?? "",
                    params.content ?? null,
                    params.importance ?? null
                );

            case "remove":
                return this.removeMemory(params.memoryId ?? "");

            case "forget":
                return this.forget(
                    params.strategy ?? "importance_based",
                    params.threshold ?? 0.1,
                    params.maxAgeDays ?? 30
                );

            case "consolidate":
                return this.consolidate(
                    params.fromType ?? "working",
                    params.toType ?? "episodic",
                    params.importanceThreshold ?? 0.7
                );

            case "clearAll":
                return this.clearAll();

            // 理论上不可达（action 已被 isMemoryToolAction 收窄），保留作类型兜底与防御
            default:
                return `❌ 不支持的操作: ${String(params.action)}`;
        }
    }

    /**
     * 添加一条记忆（add 操作的实现，也被便捷方法复用）。
     * @param content    记忆内容
     * @param memoryType 记忆类型，默认工作记忆
     * @param importance 重要性 0.0-1.0
     * @param filePath   感知记忆的文件路径（仅 perceptual 时有意义）
     * @param modality   感知记忆模态；不传则按扩展名推断
     * @param metadata   附加元数据；便捷方法通过它透传 knowledgeType/source/type 等自定义字段
     * @returns 形如 "✅ 记忆已添加 (ID: xxxxxxxx...)" 的结果
     */
    private addMemory(
        content: string = "",
        memoryType: MemoryType = "working",
        importance: number = 0.5,
        filePath: string | null = null,
        modality: string | null = null,
        metadata: Record<string, unknown> = {}
    ): string {
        // 惰性初始化会话 ID：首次写入记忆时按当前时间生成，形如 session_YYYYMMDDHHMMSS
        if (!this.currentSessionId) {
            const now = new Date();
            const timestamp = now.toISOString()
                .replace(/[-:]/g, '')          // 去掉日期/时间中的 - 和 :
                .replace(/\.\d{3}Z$/, '')      // 去掉毫秒和结尾的 Z
                .slice(0, 14); // YYYYMMDD_HHMMSS
            this.currentSessionId = `session_${timestamp}`;
        }

        // 拷贝一份传入的 metadata 再补充，避免直接修改调用方对象
        const memoryMetadata: Record<string, unknown> = { ...metadata };
        // 感知记忆文件支持：记录模态与原始文件路径。
        // 用 `??` 而非直接赋值，是为了尊重调用方已显式给出的值，只在缺失时才填推断结果。
        if (memoryType === "perceptual" && filePath) {
            const inferred = modality || this.inferModality(filePath);
            memoryMetadata["modality"] = memoryMetadata["modality"] ?? inferred;
            memoryMetadata["raw_data"] = memoryMetadata["raw_data"] ?? filePath;
        }

        // 统一附加会话信息：会话 ID + 写入时间戳（毫秒），便于后续按会话/时间检索
        memoryMetadata["session_id"] = this.currentSessionId;
        memoryMetadata["timestamp"] = new Date().valueOf();

        // 委托底层管理器落库。最后一参 false = 不跳过整合（保留自动整合逻辑）
        const memoryId = this.memoryManager.addMemory(
            content,
            memoryType,
            importance,
            memoryMetadata,
            false,
        )

        // 只回显 ID 前 8 位，避免结果过长
        return `✅ 记忆已添加 (ID: ${memoryId.slice(0, 8)}...)`;
    }

    /**
     * 搜索记忆（search 操作的实现）。
     * @param query        查询文本
     * @param limit        返回数量上限
     * @param memoryTypes  限定的记忆类型数组；与 memoryType 二选一
     * @param memoryType   限定的单个记忆类型（便捷入口）
     * @param minImportance 最低重要性阈值，过滤掉不重要的记忆
     * @returns 格式化的多行结果，或「未找到」提示
     */
    private search(
        query: string,
        limit: number = 5,
        memoryTypes: string[] | null = null,
        memoryType: string | null = null,
        minImportance: number = 0.1
    ): string {

        // 归一化类型过滤条件：只传了单个 memoryType 时，包装成数组交给管理器
        let typesToUse: string[] | null = memoryTypes;
        if (memoryType && !memoryTypes) {
            typesToUse = [memoryType];
        }

        // 委托底层检索
        const results: any = this.memoryManager.retrieveMemories(
            query,
            limit,
            typesToUse,
            minImportance
        );

        // 空结果早返回，避免下面输出一个只有标题的空列表
        if (!results || results.length === 0) {
            return `🔍 未找到与 '${query}' 相关的记忆`;
        }

        // 将查询到的记忆逐条格式化为「序号. [类型标签] 内容预览 (重要性)」
        const formattedResults: string[] = [];
        formattedResults.push(`🔍 找到 ${results.length} 条相关记忆:`);


        results.forEach((memory: any, index: number) => {
            // 把内部类型键（working…）映射成中文标签；缺失时退回原键
            const label = MemoryTypeLabels[memory.memoryType] || memory.memoryType;

            // 内容超过 80 字截断并加省略号，保持列表整洁
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

    /**
     * 获取记忆摘要（summary 操作）：统计概览 + 类型分布 + Top-N 重要记忆。
     * @param limit 展示的重要记忆条数上限
     * @returns 多行摘要文本
     */
    private getSummary(limit: number = 10): string {
        // 先取统计快照，拼出概览区（总数 / 当前会话 / 对话轮次）
        const stats = this.memoryManager.getMemoryStats();

        const summaryParts: string[] = [
            "📊 记忆系统摘要",
            `总记忆数: ${stats.totalMemories}`,
            `当前会话: ${this.currentSessionId || "未开始"}`,
            `对话轮次: ${this.conversationCount}`,
        ];

        // 各类型记忆分布：遍历 { 类型 -> {count, avgImportance} }，逐类输出中文标签与均值
        const typeEntries = Object.entries(stats.memoriesByType);
        if (typeEntries.length > 0) {
            summaryParts.push("\n📋 记忆类型分布:");
            for (const [memoryType, typeStats] of typeEntries) {
                const count = typeStats.count ?? 0;
                const avgImportance = typeStats.avgImportance ?? 0;
                const typeLabel = MemoryTypeLabels[memoryType] || memoryType;
                summaryParts.push(
                    `  • ${typeLabel}: ${count} 条 (平均重要性: ${avgImportance.toFixed(2)})`
                );
            }
        }

        // 取重要记忆候选：故意多取 limit*3 条，因为下面去重会淘汰一部分，多取才能凑够 limit 条
        const importantMemories = this.memoryManager.retrieveMemories(
            "",            // 空查询：不按相关性过滤，纯按重要性取
            limit * 3,
            null,          // 不限类型
            0.5            // 只要重要性 ≥ 0.5 的
        );

        if (importantMemories && importantMemories.length > 0) {
            // 双重去重：既防同一条记忆重复出现（ID），也防不同 ID 但内容相同的记忆（内容）
            const seenIds = new Set<string>();
            const seenContents = new Set<string>();
            const uniqueMemories: MemoryRecord[] = [];

            for (const memory of importantMemories) {
                // 1) ID 去重：同一条记忆只保留一次
                if (seenIds.has(memory.id)) {
                    continue;
                }
                // 2) 内容去重：trim+小写后比较，过滤大小写/空白差异导致的重复
                const contentKey = memory.content.trim().toLowerCase();
                if (seenContents.has(contentKey)) {
                    continue;
                }
                seenIds.add(memory.id);
                seenContents.add(contentKey);
                uniqueMemories.push(memory);
            }

            // 去重后按重要性降序排列，最重要的在前
            uniqueMemories.sort((a, b) => b.importance - a.importance);
            // 标题里的条数取 limit 与实际数量的较小值，避免显示「前10条」却只有 3 条
            summaryParts.push(
                `\n⭐ 重要记忆 (前${Math.min(limit, uniqueMemories.length)}条):`
            );

            // 只取前 limit 条输出，内容超 60 字截断
            uniqueMemories.slice(0, limit).forEach((memory, index) => {
                const contentPreview =
                    memory.content.length > 60
                        ? memory.content.slice(0, 60) + "..."
                        : memory.content;
                summaryParts.push(
                    `  ${index + 1}. ${contentPreview} (重要性: ${memory.importance.toFixed(2)})`
                );
            });
        }

        return summaryParts.join("\n");
    }

    /**
     * 获取统计信息（stats 操作）：比 summary 更精简，只给系统级指标，不列具体记忆。
     * 注意：「启用的记忆类型」直接读本工具的 this.memoryTypes（构造时确定），
     * 而非从 getMemoryStats 里取——这个信息工具自己就持有，无需绕到底层。
     */
    private getStats(): string {
        const stats = this.memoryManager.getMemoryStats();

        const statsInfo: string[] = [
            "📈 记忆系统统计",
            `总记忆数: ${stats.totalMemories}`,
            `启用的记忆类型: ${this.memoryTypes.join(", ")}`,
            `会话ID: ${this.currentSessionId || "未开始"}`,
            `对话轮次: ${this.conversationCount}`,
        ];

        return statsInfo.join("\n");
    }

    /**
     * 更新记忆（update 操作）。
     * 底层 updateMemory 返回布尔：true=更新成功，false=没找到该 ID。
     * try/catch 兜住底层异常，转成用户可读的失败提示。
     * @param memoryId   目标记忆 ID
     * @param content    新内容；null 表示不改
     * @param importance 新重要性；null 表示不改
     */
    private updateMemory(
        memoryId: string,
        content: string | null = null,
        importance: number | null = null
    ): string {
        try {
            const success = this.memoryManager.updateMemory(
                memoryId,
                content,
                importance,
                null            // 暂不透传额外 metadata
            );
            // 三态回显：成功 / 未命中 / 异常
            return success ? "✅ 记忆已更新" : "⚠️ 未找到要更新的记忆";
        } catch (e) {
            return `❌ 更新记忆失败: ${String(e)}`;
        }
    }

    /**
     * 删除记忆（remove 操作）。
     * 底层 removeMemory 返回布尔：true=删除成功，false=没找到该 ID。
     * @param memoryId 目标记忆 ID
     */
    private removeMemory(memoryId: string): string {
        try {
            const success = this.memoryManager.removeMemory(memoryId);
            return success ? "✅ 记忆已删除" : "⚠️ 未找到要删除的记忆";
        } catch (e) {
            return `❌ 删除记忆失败: ${String(e)}`;
        }
    }

    /**
     * 遗忘记忆（forget 操作）：按策略批量清理。
     * @param strategy    遗忘策略：importance_based(按重要性) / time_based(按时间) / capacity_based(按容量)
     * @param threshold   重要性阈值（importance_based 用）
     * @param maxAgeDays  最大保留天数（time_based 用）
     * @returns 形如「🧹 已遗忘 N 条记忆（策略: xxx）」
     */
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

    /**
     * 便捷方法：遗忘旧记忆。
     * 是 forget 的一个「预设入口」——固定用 time_based 策略，只需给出保留天数。
     * 直接返回被遗忘的条数（number），供代码调用方使用，而非格式化字符串。
     */
    public forgetOldMemories(maxAgeDays: number = 30): number {
        return this.memoryManager.forgetMemories("time_based", 0.1, maxAgeDays);
    }

    /**
     * 便捷方法：清除当前会话。
     * 重置会话 ID 与对话计数，并清空工作记忆（短期记忆），但保留长期记忆（情景/语义）。
     * 用于「开启新对话」时把临时上下文清干净。
     */
    public clearSession(): void {
        this.currentSessionId = "";
        this.conversationCount = 0;
        // 只清工作记忆：长期记忆不受影响
        this.memoryManager.clearWorkingMemory();
    }

    // 清空所有记忆（危险操作：会抹掉全部类型的记忆，不可恢复，谨慎调用）
    private clearAll(): string {
        this.memoryManager.clearAllMemories();
        return "🧽 已清空所有记忆";
    }

    /**
     * 整合记忆（consolidate 操作）：把重要的短期记忆「提升」为长期记忆。
     *
     * 借鉴神经科学的「记忆固化」概念，模拟大脑将短期记忆转化为长期记忆的过程：
     * 默认把重要性 ≥ 0.7 的工作记忆转成情景记忆——阈值确保只有真正重要的信息才被长期保留。
     * @param fromType            来源类型，默认 working（工作/短期）
     * @param toType              目标类型，默认 episodic（情景/长期）
     * @param importanceThreshold 触发整合的重要性下限
     * @returns 形如「🔄 已整合 N 条记忆...」
     */
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

    /**
     * 便捷方法：把一条「知识」写入语义记忆。
     * 相比通用 add，这里固定 memoryType=semantic、默认高重要性 0.9，
     * 并通过 metadata 打上 knowledgeType/source 标记，便于区分「人工录入的事实性知识」。
     */
    public addKnowledge(content: string, importance: number = 0.9): string {
        return this.addMemory(
            content,
            "semantic",
            importance,
            null,
            null,
            { knowledgeType: "factual", source: "manual" }   // 透传给底层的自定义元数据
        );
    }

    /**
     * 便捷方法：为一次查询取回相关记忆，拼成可直接塞进提示词的上下文块。
     * 供 Agent 在回答前调用，实现「带记忆的对话」。
     * @returns 形如「相关记忆:\n- ...\n- ...」；无命中时返回空串（调用方可据此决定是否拼接）
     */
    public getContextForQuery(query: string, limit: number = 3): string {
        // 阈值 0.3 比 summary(0.5) 更低：上下文召回宁可多带一点，也不漏掉可能相关的记忆
        const results = this.memoryManager.retrieveMemories(query, limit, null, 0.3);

        if (!results || results.length === 0) {
            return "";
        }

        const contextParts: string[] = ["相关记忆:"];
        for (const memory of results) {
            contextParts.push(`- ${memory.content}`);
        }
        return contextParts.join("\n");
    }

    /**
     * 便捷方法：自动记录一轮对话。供 Agent 在每次问答后调用。
     * 效果：
     *   - 对话轮次 +1（这是 conversationCount 唯一的写入点，summary/stats 的轮次靠它）；
     *   - 用户输入、助手响应各存为一条工作记忆（短期）；
     *   - 若判定为「重要对话」，再额外存一条情景记忆（长期）。
     */
    public autoRecordConversation(userInput: string, agentResponse: string): void {
        this.conversationCount += 1;

        // 记录用户输入（工作记忆，重要性 0.6）
        this.addMemory(
            `用户: ${userInput}`,
            "working",
            0.6,
            null,
            null,
            { type: "user_input", conversationId: this.conversationCount }
        );

        // 记录 Agent 响应（工作记忆，重要性略高 0.7）
        this.addMemory(
            `助手: ${agentResponse}`,
            "working",
            0.7,
            null,
            null,
            { type: "agent_response", conversationId: this.conversationCount }
        );

        // 重要对话判定：响应较长(>100)、或用户明确说了「重要」「记住」→ 额外固化为情景记忆
        if (agentResponse.length > 100 || userInput.includes("重要") || userInput.includes("记住")) {
            const interactionContent = `对话 - 用户: ${userInput}\n助手: ${agentResponse}`;
            this.addMemory(
                interactionContent,
                "episodic",
                0.8,
                null,
                null,
                { type: "interaction", conversationId: this.conversationCount }
            );
        }
    }

    /**
     * 根据文件扩展名推断感知记忆的模态。
     * 图片类 → image，音频类 → audio，其余一律按 text 处理（兜底，绝不抛错）。
     */
    private inferModality(path: string): string {
        // 取最后一个「.」之后的部分作为扩展名，并转小写
        const ext = (path.split('.').pop() || '').toLowerCase();

        if (["png", "jpg", "jpeg", "bmp", "gif", "webp"].includes(ext)) {
            return "image";
        }
        if (["mp3", "wav", "flac", "m4a", "ogg"].includes(ext)) {
            return "audio";
        }
        return "text";
    }
}
