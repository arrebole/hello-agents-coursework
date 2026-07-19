/**
 * NoteTool - 结构化笔记工具
 *
 * 为 Agent 提供结构化笔记能力，支持：
 * - 创建/读取/更新/删除笔记
 * - 按类型组织（任务状态、结论、阻塞项、行动计划等）
 * - 持久化存储（Markdown 格式，带 YAML 前置元数据）
 * - 搜索与过滤
 * - 与 MemoryTool 集成（可选）
 */

import {
    copyFileSync,
    existsSync,
    mkdirSync,
    readFileSync,
    unlinkSync,
    writeFileSync,
} from "fs";
import { basename, join, resolve } from "path";

import { Tool, ToolParameter, toolAction } from "../../../7.hello-agents/tools/base";

const NOTE_ACTIONS = ["create", "read", "update", "delete", "list", "search", "summary"] as const;
type NoteAction = (typeof NOTE_ACTIONS)[number];

type NoteType = "task_state" | "conclusion" | "blocker" | "action" | "reference" | "general" | string;

interface NoteToolParams {
    action?: NoteAction;
    title?: string;
    content?: string;
    note_type?: NoteType;
    noteType?: NoteType;
    tags?: unknown;
    note_id?: string;
    noteId?: string;
    query?: string;
    limit?: number;
}

interface NoteToolOptions {
    workspace?: string;
    autoBackup?: boolean;
    maxNotes?: number;
    expandable?: boolean;
}

interface NoteIndexEntry {
    id: string;
    title: string;
    type: NoteType;
    tags: string[];
    created_at: string;
}

interface NotesIndex {
    notes: NoteIndexEntry[];
    metadata: {
        created_at: string;
        total_notes: number;
    };
}

interface StructuredNote {
    id: string;
    title: string;
    content: string;
    type: NoteType;
    tags: string[];
    created_at: string;
    updated_at: string;
    metadata: {
        word_count: number;
        status: string;
    };
}

function isNoteAction(value: unknown): value is NoteAction {
    return NOTE_ACTIONS.includes(value as NoteAction);
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

function nowIso(): string {
    return new Date().toISOString();
}

function pad2(value: number): string {
    return String(value).padStart(2, "0");
}

function normalizeTags(value: unknown): string[] {
    if (Array.isArray(value)) {
        return value
            .map((item) => String(item).trim())
            .filter(Boolean);
    }

    if (typeof value === "string") {
        return value
            .split(",")
            .map((item) => item.trim())
            .filter(Boolean);
    }

    return [];
}

function normalizeLimit(value: unknown, fallback = 10): number {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
    return Math.floor(parsed);
}

function parseFrontmatterValue(value: string): unknown {
    const trimmed = value.trim();
    if (!trimmed) return "";

    try {
        return JSON.parse(trimmed);
    } catch {
        return trimmed;
    }
}

function toFrontmatterScalar(value: string): string {
    return JSON.stringify(value);
}

/**
 * 笔记工具。
 *
 * 笔记类型约定：
 * - task_state: 任务状态
 * - conclusion: 关键结论
 * - blocker: 阻塞项
 * - action: 行动计划
 * - reference: 参考资料
 * - general: 通用笔记
 */
export class NoteTool extends Tool {
    private readonly workspace: string;
    private readonly autoBackup: boolean;
    private readonly maxNotes: number;
    private readonly indexFile: string;
    private readonly backupDir: string;
    private notesIndex: NotesIndex;

    constructor(workspace?: string, autoBackup?: boolean, maxNotes?: number, expandable?: boolean);
    constructor(options?: NoteToolOptions);
    constructor(
        workspaceOrOptions: string | NoteToolOptions = "./notes",
        autoBackup = true,
        maxNotes = 1000,
        expandable = false,
    ) {
        const options = typeof workspaceOrOptions === "string"
            ? { workspace: workspaceOrOptions, autoBackup, maxNotes, expandable }
            : workspaceOrOptions;

        super(
            "note",
            "笔记工具 - 创建、读取、更新、删除结构化笔记，支持任务状态、结论、阻塞项等类型",
            options.expandable ?? false,
        );

        this.workspace = resolve(options.workspace ?? "./notes");
        this.autoBackup = options.autoBackup ?? true;
        this.maxNotes = options.maxNotes ?? 1000;
        this.indexFile = join(this.workspace, "notes_index.json");
        this.backupDir = join(this.workspace, ".backups");

        mkdirSync(this.workspace, { recursive: true });
        if (this.autoBackup) mkdirSync(this.backupDir, { recursive: true });

        this.notesIndex = this.loadIndex();
    }

    run(parameters: Record<string, unknown>): string {
        if (!this.validateParameters(parameters)) {
            return "❌ 参数验证失败";
        }

        if (!isNoteAction(parameters.action)) {
            return `❌ 不支持的操作: ${String(parameters.action)}`;
        }

        const params = parameters as NoteToolParams;

        try {
            switch (params.action) {
                case "create":
                    return this.createNote(
                        params.title,
                        params.content,
                        params.note_type ?? params.noteType ?? "general",
                        normalizeTags(params.tags),
                    );
                case "read":
                    return this.readNote(params.note_id ?? params.noteId);
                case "update":
                    return this.updateNote(
                        params.note_id ?? params.noteId,
                        params.title,
                        params.content,
                        params.note_type ?? params.noteType,
                        params.tags === undefined ? undefined : normalizeTags(params.tags),
                    );
                case "delete":
                    return this.deleteNote(params.note_id ?? params.noteId);
                case "list":
                    return this.listNotes(
                        params.note_type ?? params.noteType,
                        normalizeLimit(params.limit),
                    );
                case "search":
                    return this.searchNotes(params.query, normalizeLimit(params.limit));
                case "summary":
                    return this.getSummary();
                default:
                    return `❌ 不支持的操作: ${String(params.action)}`;
            }
        } catch (error) {
            return `❌ 笔记操作失败: ${errorMessage(error)}`;
        }
    }

    getParameters(): ToolParameter[] {
        return [
            new ToolParameter({
                name: "action",
                type: "string",
                description: "操作类型: create(创建), read(读取), update(更新), delete(删除), list(列表), search(搜索), summary(摘要)",
                required: true,
            }),
            new ToolParameter({
                name: "title",
                type: "string",
                description: "笔记标题（create/update时必需）",
                required: false,
            }),
            new ToolParameter({
                name: "content",
                type: "string",
                description: "笔记内容（create/update时必需）",
                required: false,
            }),
            new ToolParameter({
                name: "note_type",
                type: "string",
                description: "笔记类型: task_state(任务状态), conclusion(结论), blocker(阻塞项), action(行动计划), reference(参考), general(通用)",
                required: false,
                default: "general",
            }),
            new ToolParameter({
                name: "tags",
                type: "array",
                description: "标签列表（可选）",
                required: false,
            }),
            new ToolParameter({
                name: "note_id",
                type: "string",
                description: "笔记ID（read/update/delete时必需）",
                required: false,
            }),
            new ToolParameter({
                name: "query",
                type: "string",
                description: "搜索关键词（search时必需）",
                required: false,
            }),
            new ToolParameter({
                name: "limit",
                type: "integer",
                description: "返回结果数量限制（默认10）",
                required: false,
                default: 10,
            }),
        ];
    }

    @toolAction("note_create", "创建一条新的结构化笔记")
    createNote(title?: string, content?: string, noteType: NoteType = "general", tags: string[] = []): string {
        if (!title || !content) {
            return "❌ 创建笔记需要提供 title 和 content";
        }

        if (this.notesIndex.notes.length >= this.maxNotes) {
            return `❌ 笔记数量已达上限 (${this.maxNotes})`;
        }

        const noteId = this.generateNoteId();
        const timestamp = nowIso();
        const note: StructuredNote = {
            id: noteId,
            title,
            content,
            type: noteType || "general",
            tags,
            created_at: timestamp,
            updated_at: timestamp,
            metadata: {
                word_count: content.length,
                status: "active",
            },
        };

        writeFileSync(this.getNotePath(noteId), this.noteToMarkdown(note), "utf-8");

        this.notesIndex.notes.push({
            id: noteId,
            title,
            type: note.type,
            tags,
            created_at: note.created_at,
        });
        this.notesIndex.metadata.total_notes = this.notesIndex.notes.length;
        this.saveIndex();

        return `✅ 笔记创建成功\nID: ${noteId}\n标题: ${title}\n类型: ${note.type}`;
    }

    @toolAction("note_read", "读取指定ID的笔记")
    readNote(noteId?: string): string {
        if (!noteId) {
            return "❌ 读取笔记需要提供 note_id";
        }

        const notePath = this.getNotePath(noteId);
        if (!existsSync(notePath)) {
            return `❌ 笔记不存在: ${noteId}`;
        }

        const note = this.markdownToNote(readFileSync(notePath, "utf-8"));
        return this.formatNote(note);
    }

    @toolAction("note_update", "更新已存在的笔记")
    updateNote(
        noteId?: string,
        title?: string,
        content?: string,
        noteType?: NoteType,
        tags?: string[],
    ): string {
        if (!noteId) {
            return "❌ 更新笔记需要提供 note_id";
        }

        const notePath = this.getNotePath(noteId);
        if (!existsSync(notePath)) {
            return `❌ 笔记不存在: ${noteId}`;
        }

        this.backupNote(noteId);

        const note = this.markdownToNote(readFileSync(notePath, "utf-8"));
        if (title) note.title = title;
        if (content) {
            note.content = content;
            note.metadata.word_count = content.length;
        }
        if (noteType) note.type = noteType;
        if (tags !== undefined) note.tags = tags;
        note.updated_at = nowIso();

        writeFileSync(notePath, this.noteToMarkdown(note), "utf-8");

        const indexEntry = this.notesIndex.notes.find((item) => item.id === noteId);
        if (indexEntry) {
            indexEntry.title = note.title;
            indexEntry.type = note.type;
            indexEntry.tags = note.tags;
        }
        this.saveIndex();

        return `✅ 笔记更新成功: ${noteId}`;
    }

    @toolAction("note_delete", "删除指定ID的笔记")
    deleteNote(noteId?: string): string {
        if (!noteId) {
            return "❌ 删除笔记需要提供 note_id";
        }

        const notePath = this.getNotePath(noteId);
        if (!existsSync(notePath)) {
            return `❌ 笔记不存在: ${noteId}`;
        }

        this.backupNote(noteId);
        unlinkSync(notePath);

        this.notesIndex.notes = this.notesIndex.notes.filter((note) => note.id !== noteId);
        this.notesIndex.metadata.total_notes = this.notesIndex.notes.length;
        this.saveIndex();

        return `✅ 笔记已删除: ${noteId}`;
    }

    @toolAction("note_list", "列出所有笔记或指定类型的笔记")
    listNotes(noteType?: NoteType, limit = 10): string {
        let filteredNotes = this.notesIndex.notes;
        if (noteType) {
            filteredNotes = filteredNotes.filter((note) => note.type === noteType);
        }

        filteredNotes = filteredNotes.slice(0, normalizeLimit(limit));

        if (filteredNotes.length === 0) {
            return "📝 暂无笔记";
        }

        let result = `📝 笔记列表（共 ${filteredNotes.length} 条）\n\n`;
        for (const note of filteredNotes) {
            result += `• [${note.type}] ${note.title}\n`;
            result += `  ID: ${note.id}\n`;
            if (note.tags.length > 0) {
                result += `  标签: ${note.tags.join(", ")}\n`;
            }
            result += `  创建时间: ${note.created_at}\n\n`;
        }

        return result;
    }

    @toolAction("note_search", "搜索包含关键词的笔记")
    searchNotes(query?: string, limit = 10): string {
        if (!query) {
            return "❌ 搜索需要提供 query";
        }

        const queryLower = query.toLowerCase();
        const matchedNotes: StructuredNote[] = [];

        for (const indexNote of this.notesIndex.notes) {
            const notePath = this.getNotePath(indexNote.id);
            if (!existsSync(notePath)) continue;

            try {
                const note = this.markdownToNote(readFileSync(notePath, "utf-8"));
                const matches = note.title.toLowerCase().includes(queryLower)
                    || note.content.toLowerCase().includes(queryLower)
                    || note.tags.some((tag) => tag.toLowerCase().includes(queryLower));

                if (matches) matchedNotes.push(note);
            } catch (error) {
                console.warn(`解析笔记失败 ${indexNote.id}: ${errorMessage(error)}`);
            }
        }

        const limitedNotes = matchedNotes.slice(0, normalizeLimit(limit));
        if (limitedNotes.length === 0) {
            return `📝 未找到匹配 '${query}' 的笔记`;
        }

        let result = `🔍 搜索结果（共 ${limitedNotes.length} 条）\n\n`;
        for (const note of limitedNotes) {
            result += `${this.formatNote(note, true)}\n`;
        }

        return result;
    }

    @toolAction("note_summary", "获取笔记系统的摘要统计信息")
    getSummary(): string {
        const typeCounts = new Map<string, number>();
        for (const note of this.notesIndex.notes) {
            typeCounts.set(note.type, (typeCounts.get(note.type) ?? 0) + 1);
        }

        let result = "📊 笔记摘要\n\n";
        result += `总笔记数: ${this.notesIndex.notes.length}\n\n`;
        result += "按类型统计:\n";

        for (const [noteType, count] of [...typeCounts.entries()].sort(([left], [right]) => left.localeCompare(right))) {
            result += `  • ${noteType}: ${count}\n`;
        }

        return result;
    }

    private loadIndex(): NotesIndex {
        if (!existsSync(this.indexFile)) {
            const emptyIndex: NotesIndex = {
                notes: [],
                metadata: {
                    created_at: nowIso(),
                    total_notes: 0,
                },
            };
            this.notesIndex = emptyIndex;
            this.saveIndex();
            return emptyIndex;
        }

        const parsed = JSON.parse(readFileSync(this.indexFile, "utf-8")) as Partial<NotesIndex>;
        return {
            notes: Array.isArray(parsed.notes) ? parsed.notes as NoteIndexEntry[] : [],
            metadata: {
                created_at: parsed.metadata?.created_at ?? nowIso(),
                total_notes: Array.isArray(parsed.notes) ? parsed.notes.length : 0,
            },
        };
    }

    private saveIndex(): void {
        writeFileSync(this.indexFile, JSON.stringify(this.notesIndex, null, 2), "utf-8");
    }

    private generateNoteId(): string {
        const now = new Date();
        const timestamp = [
            now.getFullYear(),
            pad2(now.getMonth() + 1),
            pad2(now.getDate()),
            "_",
            pad2(now.getHours()),
            pad2(now.getMinutes()),
            pad2(now.getSeconds()),
        ].join("");
        return `note_${timestamp}_${this.notesIndex.notes.length}`;
    }

    private getNotePath(noteId: string): string {
        const safeName = basename(noteId);
        const notePath = resolve(this.workspace, `${safeName}.md`);
        if (!notePath.startsWith(`${this.workspace}/`) && notePath !== join(this.workspace, `${safeName}.md`)) {
            throw new Error(`非法笔记ID: ${noteId}`);
        }
        return notePath;
    }

    private backupNote(noteId: string): void {
        if (!this.autoBackup) return;

        const notePath = this.getNotePath(noteId);
        if (!existsSync(notePath)) return;

        const backupPath = join(this.backupDir, `${basename(noteId)}.${Date.now()}.md`);
        copyFileSync(notePath, backupPath);
    }

    private noteToMarkdown(note: StructuredNote): string {
        const frontmatter = [
            "---",
            `id: ${toFrontmatterScalar(note.id)}`,
            `title: ${toFrontmatterScalar(note.title)}`,
            `type: ${toFrontmatterScalar(note.type)}`,
            `tags: ${JSON.stringify(note.tags)}`,
            `created_at: ${toFrontmatterScalar(note.created_at)}`,
            `updated_at: ${toFrontmatterScalar(note.updated_at)}`,
            "---",
            "",
        ].join("\n");

        return `${frontmatter}# ${note.title}\n\n${note.content}`;
    }

    private markdownToNote(markdownText: string): StructuredNote {
        const frontmatterMatch = markdownText.match(/^---\s*\n([\s\S]*?)\n---\s*\n/);
        if (!frontmatterMatch || frontmatterMatch.index !== 0) {
            throw new Error("无效的笔记格式：缺少YAML前置元数据");
        }

        const frontmatterText = frontmatterMatch[1];
        const noteData: Record<string, unknown> = {};

        for (const line of frontmatterText.split("\n")) {
            const separatorIndex = line.indexOf(":");
            if (separatorIndex < 0) continue;

            const key = line.slice(0, separatorIndex).trim();
            const value = line.slice(separatorIndex + 1).trim();
            noteData[key] = key === "tags"
                ? normalizeTags(parseFrontmatterValue(value))
                : parseFrontmatterValue(value);
        }

        let markdownContent = markdownText.slice(frontmatterMatch[0].length).trim();
        const lines = markdownContent.split("\n");
        if (lines[0]?.startsWith("# ")) {
            markdownContent = lines.slice(1).join("\n").trim();
        }

        const content = markdownContent;
        return {
            id: String(noteData.id ?? ""),
            title: String(noteData.title ?? ""),
            content,
            type: String(noteData.type ?? "general"),
            tags: normalizeTags(noteData.tags),
            created_at: String(noteData.created_at ?? ""),
            updated_at: String(noteData.updated_at ?? ""),
            metadata: {
                word_count: content.length,
                status: "active",
            },
        };
    }

    private formatNote(note: StructuredNote, compact = false): string {
        if (compact) {
            const preview = note.content.length > 100
                ? `${note.content.slice(0, 100)}...`
                : note.content;
            return [
                `[${note.type}] ${note.title}`,
                `ID: ${note.id}`,
                `内容: ${preview}`,
            ].join("\n");
        }

        const lines = [
            "📝 笔记详情",
            "",
            `ID: ${note.id}`,
            `标题: ${note.title}`,
            `类型: ${note.type}`,
        ];

        if (note.tags.length > 0) {
            lines.push(`标签: ${note.tags.join(", ")}`);
        }

        lines.push(
            `创建时间: ${note.created_at}`,
            `更新时间: ${note.updated_at}`,
            "",
            `内容:\n${note.content}`,
        );

        return `${lines.join("\n")}\n`;
    }
}
