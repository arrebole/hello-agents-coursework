// 文档存储实现
//
// 支持多种文档数据库后端：
// - SQLite: 轻量级关系型数据库
// - PostgreSQL: 企业级关系型数据库（可扩展）

import { randomUUID } from "crypto";
import { existsSync, mkdirSync } from "fs";
import { dirname, resolve } from "path";
import { DatabaseSync } from "node:sqlite";

type JsonRecord = Record<string, unknown>;

export interface DocumentStore {
    add_memory(
        memory_id: string,
        user_id: string,
        content: string,
        memory_type: string,
        timestamp: number,
        importance: number,
        properties?: JsonRecord,
    ): string;

    get_memory(memory_id: string): JsonRecord | null;

    search_memories(
        user_id?: string | null,
        memory_type?: string | null,
        start_time?: number | null,
        end_time?: number | null,
        importance_threshold?: number | null,
        limit?: number,
    ): JsonRecord[];

    update_memory(
        memory_id: string,
        content?: string | null,
        importance?: number | null,
        properties?: JsonRecord | null,
    ): boolean;

    delete_memory(memory_id: string): boolean;

    get_database_stats(): JsonRecord;

    add_document(content: string, metadata?: JsonRecord): string;

    get_document(document_id: string): JsonRecord | null;

    close(): void;
}

function normalizeJson(value: unknown): string | null {
    if (value === undefined || value === null) {
        return null;
    }

    return JSON.stringify(value);
}

function parseJson(value: unknown, fallback: JsonRecord = {}): JsonRecord {
    if (typeof value !== "string" || value.length === 0) {
        return { ...fallback };
    }

    try {
        const parsed = JSON.parse(value);
        if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
            return parsed as JsonRecord;
        }
        return { ...fallback };
    } catch {
        return { ...fallback };
    }
}

function ensureDirectoryForFile(dbPath: string): void {
    const absolutePath = resolve(dbPath);
    const directory = dirname(absolutePath);
    if (directory && !existsSync(directory)) {
        mkdirSync(directory, { recursive: true });
    }
}

function asNumber(value: unknown, fallback = 0): number {
    const num = typeof value === "number" ? value : Number(value);
    return Number.isFinite(num) ? num : fallback;
}

class SQLiteDocumentStore implements DocumentStore {
    private static instances = new Map<string, SQLiteDocumentStore>();
    private static initializedDbs = new Set<string>();

    private readonly dbPath: string;
    private readonly database: DatabaseSync;
    private closed = false;

    constructor(dbPath: string = "./memory.db") {
        const absolutePath = resolve(dbPath);
        const cached = SQLiteDocumentStore.instances.get(absolutePath);
        if (cached && !cached.closed) {
            return cached;
        }

        ensureDirectoryForFile(absolutePath);
        this.dbPath = absolutePath;
        this.database = new DatabaseSync(absolutePath);

        if (!SQLiteDocumentStore.initializedDbs.has(absolutePath)) {
            this.initDatabase();
            SQLiteDocumentStore.initializedDbs.add(absolutePath);
        }

        SQLiteDocumentStore.instances.set(absolutePath, this);
    }

    private initDatabase(): void {
        this.database.exec("PRAGMA foreign_keys = ON");

        this.database.exec(`
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT,
                properties TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        `);

        this.database.exec(`
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                memory_type TEXT NOT NULL,
                timestamp INTEGER NOT NULL,
                importance REAL NOT NULL,
                properties TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        `);

        this.database.exec(`
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                properties TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        `);

        this.database.exec(`
            CREATE TABLE IF NOT EXISTS memory_concepts (
                memory_id TEXT NOT NULL,
                concept_id TEXT NOT NULL,
                relevance_score REAL DEFAULT 1.0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (memory_id, concept_id),
                FOREIGN KEY (memory_id) REFERENCES memories (id) ON DELETE CASCADE,
                FOREIGN KEY (concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        `);

        this.database.exec(`
            CREATE TABLE IF NOT EXISTS concept_relationships (
                from_concept_id TEXT NOT NULL,
                to_concept_id TEXT NOT NULL,
                relationship_type TEXT NOT NULL,
                strength REAL DEFAULT 1.0,
                properties TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (from_concept_id, to_concept_id, relationship_type),
                FOREIGN KEY (from_concept_id) REFERENCES concepts (id) ON DELETE CASCADE,
                FOREIGN KEY (to_concept_id) REFERENCES concepts (id) ON DELETE CASCADE
            )
        `);

        const indexes = [
            "CREATE INDEX IF NOT EXISTS idx_memories_user_id ON memories (user_id)",
            "CREATE INDEX IF NOT EXISTS idx_memories_type ON memories (memory_type)",
            "CREATE INDEX IF NOT EXISTS idx_memories_timestamp ON memories (timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_memories_importance ON memories (importance)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_memory ON memory_concepts (memory_id)",
            "CREATE INDEX IF NOT EXISTS idx_memory_concepts_concept ON memory_concepts (concept_id)",
        ];

        for (const indexSql of indexes) {
            this.database.exec(indexSql);
        }
    }

    add_memory(
        memory_id: string,
        user_id: string,
        content: string,
        memory_type: string,
        timestamp: number,
        importance: number,
        properties: JsonRecord = {},
    ): string {
        const userProps = normalizeJson({ name: user_id });
        this.database.prepare(
            "INSERT OR IGNORE INTO users (id, name, properties) VALUES (?, ?, ?)",
        ).run(user_id, user_id, userProps);

        this.database.prepare(`
            INSERT OR REPLACE INTO memories
            (id, user_id, content, memory_type, timestamp, importance, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        `).run(
            memory_id,
            user_id,
            content,
            memory_type,
            timestamp,
            importance,
            normalizeJson(properties),
        );

        return memory_id;
    }

    get_memory(memory_id: string): JsonRecord | null {
        const row = this.database.prepare(`
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at, updated_at
            FROM memories
            WHERE id = ?
        `).get(memory_id) as JsonRecord | undefined;

        if (!row) {
            return null;
        }

        return {
            memory_id: row.id,
            user_id: row.user_id,
            content: row.content,
            memory_type: row.memory_type,
            timestamp: asNumber(row.timestamp),
            importance: asNumber(row.importance),
            properties: parseJson(row.properties),
            created_at: row.created_at,
            updated_at: row.updated_at,
        };
    }

    search_memories(
        user_id: string | null = null,
        memory_type: string | null = null,
        start_time: number | null = null,
        end_time: number | null = null,
        importance_threshold: number | null = null,
        limit: number = 10,
    ): JsonRecord[] {
        const where: string[] = [];
        const params: Array<string | number> = [];

        if (user_id !== null && user_id !== undefined) {
            where.push("user_id = ?");
            params.push(user_id);
        }
        if (memory_type !== null && memory_type !== undefined) {
            where.push("memory_type = ?");
            params.push(memory_type);
        }
        if (start_time !== null && start_time !== undefined) {
            where.push("timestamp >= ?");
            params.push(start_time);
        }
        if (end_time !== null && end_time !== undefined) {
            where.push("timestamp <= ?");
            params.push(end_time);
        }
        if (importance_threshold !== null && importance_threshold !== undefined) {
            where.push("importance >= ?");
            params.push(importance_threshold);
        }

        const whereClause = where.length > 0 ? `WHERE ${where.join(" AND ")}` : "";
        const rows = this.database.prepare(`
            SELECT id, user_id, content, memory_type, timestamp, importance, properties, created_at, updated_at
            FROM memories
            ${whereClause}
            ORDER BY importance DESC, timestamp DESC
            LIMIT ?
        `).all(...params, limit) as JsonRecord[];

        return rows.map((row) => ({
            memory_id: row.id,
            user_id: row.user_id,
            content: row.content,
            memory_type: row.memory_type,
            timestamp: asNumber(row.timestamp),
            importance: asNumber(row.importance),
            properties: parseJson(row.properties),
            created_at: row.created_at,
            updated_at: row.updated_at,
        }));
    }

    update_memory(
        memory_id: string,
        content: string | null = null,
        importance: number | null = null,
        properties: JsonRecord | null = null,
    ): boolean {
        const fields: string[] = [];
        const params: Array<string | number | null> = [];

        if (content !== null) {
            fields.push("content = ?");
            params.push(content);
        }
        if (importance !== null) {
            fields.push("importance = ?");
            params.push(importance);
        }
        if (properties !== null) {
            fields.push("properties = ?");
            params.push(normalizeJson(properties));
        }

        if (fields.length === 0) {
            return false;
        }

        fields.push("updated_at = CURRENT_TIMESTAMP");
        params.push(memory_id);

        const result = this.database.prepare(`
            UPDATE memories
            SET ${fields.join(", ")}
            WHERE id = ?
        `).run(...params);

        return result.changes > 0;
    }

    delete_memory(memory_id: string): boolean {
        const result = this.database.prepare("DELETE FROM memories WHERE id = ?").run(memory_id);
        return result.changes > 0;
    }

    get_database_stats(): JsonRecord {
        const stats: JsonRecord = {};
        const tables = ["users", "memories", "concepts", "memory_concepts", "concept_relationships"];

        for (const table of tables) {
            const row = this.database.prepare(`SELECT COUNT(*) as count FROM ${table}`).get() as JsonRecord;
            stats[`${table}_count`] = asNumber(row.count);
        }

        const memoryTypeRows = this.database.prepare(`
            SELECT memory_type, COUNT(*) as count
            FROM memories
            GROUP BY memory_type
        `).all() as JsonRecord[];

        const memoryTypes: JsonRecord = {};
        for (const row of memoryTypeRows) {
            memoryTypes[String(row.memory_type)] = asNumber(row.count);
        }
        stats.memory_types = memoryTypes;

        const topUserRows = this.database.prepare(`
            SELECT user_id, COUNT(*) as count
            FROM memories
            GROUP BY user_id
            ORDER BY count DESC
            LIMIT 10
        `).all() as JsonRecord[];

        const topUsers: JsonRecord = {};
        for (const row of topUserRows) {
            topUsers[String(row.user_id)] = asNumber(row.count);
        }
        stats.top_users = topUsers;

        stats.store_type = "sqlite";
        stats.db_path = this.dbPath;
        return stats;
    }

    add_document(content: string, metadata: JsonRecord = {}): string {
        const documentId = randomUUID();
        const userId = typeof metadata.user_id === "string" && metadata.user_id.length > 0
            ? metadata.user_id
            : "system";

        return this.add_memory(
            documentId,
            userId,
            content,
            "document",
            Math.floor(Date.now() / 1000),
            0.5,
            metadata,
        );
    }

    get_document(document_id: string): JsonRecord | null {
        return this.get_memory(document_id);
    }

    close(): void {
        if (this.closed) {
            return;
        }

        this.database.close();
        this.closed = true;
        SQLiteDocumentStore.instances.delete(this.dbPath);
    }
}

class PostgreSQLDocumentStore implements DocumentStore {
    constructor() {
        throw new Error("PostgreSQLDocumentStore 暂未实现。请先安装 pg 并补充连接配置。");
    }

    add_memory(): string { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    get_memory(): JsonRecord | null { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    search_memories(): JsonRecord[] { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    update_memory(): boolean { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    delete_memory(): boolean { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    get_database_stats(): JsonRecord { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    add_document(): string { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    get_document(): JsonRecord | null { throw new Error("PostgreSQLDocumentStore 暂未实现。") }
    close(): void { return; }
}

export function createDocumentStore(dbPath: string = "./memory.db", backend: "sqlite" | "postgresql" = "sqlite"): DocumentStore {
    if (backend === "postgresql") {
        return new PostgreSQLDocumentStore();
    }

    return new SQLiteDocumentStore(dbPath);
}

export { PostgreSQLDocumentStore, SQLiteDocumentStore };
