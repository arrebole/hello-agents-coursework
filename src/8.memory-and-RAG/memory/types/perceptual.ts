// 感知记忆（多模态，SQLite+Qdrant）

import { createHash, randomUUID } from "crypto";
import { mkdirSync } from "fs";
import { resolve } from "path";

import type { MemoryConfig } from "../config.ts";
import type { MemoryStore } from "../storage/store.ts";
import { createDocumentStore, SQLiteDocumentStore } from "../storage/document_store.ts";
import { QdrantVectorStore } from "../storage/qdrant_store.ts";
import { Memory } from "./memory.ts";
import type { MemoryItem } from "./memory.ts";

type JsonRecord = Record<string, unknown>;

export class Perception {
    public readonly perceptionId: string;
    public readonly data: unknown;
    public readonly modality: string;
    public readonly timestamp: Date;
    public readonly dataHash: string;
    public encoding: number[];
    public metadata: JsonRecord;

    constructor(
        perceptionId: string,
        data: unknown,
        modality: string,
        encoding: number[] = [],
        metadata: JsonRecord = {},
    ) {
        this.perceptionId = perceptionId;
        this.data = data;
        this.modality = modality;
        this.encoding = encoding;
        this.metadata = metadata;
        this.timestamp = new Date();
        this.dataHash = this.calculateHash();
    }

    private calculateHash(): string {
        if (typeof this.data === "string") {
            return createHash("md5").update(this.data).digest("hex");
        }

        if (this.data instanceof Uint8Array) {
            return createHash("md5").update(Buffer.from(this.data)).digest("hex");
        }

        return createHash("md5").update(String(this.data)).digest("hex");
    }
}

function asDate(value: unknown): Date {
    if (value instanceof Date) {
        return value;
    }

    if (typeof value === "number" && Number.isFinite(value)) {
        const millis = value > 1e12 ? value : value * 1000;
        return new Date(millis);
    }

    return new Date();
}

function toNumber(value: unknown, fallback = 0.5): number {
    const num = typeof value === "number" ? value : Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function cloneRecord(record: JsonRecord | undefined): JsonRecord {
    return record ? { ...record } : {};
}

function tokenize(text: string): string[] {
    return text
        .toLowerCase()
        .split(/\s+/)
        .map((part) => part.trim())
        .filter(Boolean);
}

function normalizeModality(value: unknown): string {
    return typeof value === "string" && value.trim() ? value.trim().toLowerCase() : "text";
}

function cosineSimilarity(left: number[], right: number[]): number {
    const size = Math.min(left.length, right.length);
    if (size === 0) {
        return 0;
    }

    let dot = 0;
    let leftNorm = 0;
    let rightNorm = 0;

    for (let i = 0; i < size; i += 1) {
        const a = left[i] ?? 0;
        const b = right[i] ?? 0;
        dot += a * b;
        leftNorm += a * a;
        rightNorm += b * b;
    }

    if (leftNorm === 0 || rightNorm === 0) {
        return 0;
    }

    return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

export class PerceptualMemory extends Memory {
    private readonly config: MemoryConfig;
    private readonly perceptions = new Map<string, Perception>();
    private readonly perceptualMemories: MemoryItem[] = [];
    private readonly modalityIndex = new Map<string, string[]>();
    private readonly supportedModalities: Set<string>;
    private readonly docStore: SQLiteDocumentStore;
    private readonly vectorStores: Map<string, QdrantVectorStore> = new Map();
    private readonly vectorIndex: Map<string, number[]> = new Map();
    private readonly vectorDim: number;
    private readonly imageDim: number;
    private readonly audioDim: number;
    private readonly textEmbedder = null;

    constructor(config: MemoryConfig, storageBackend: MemoryStore) {
        super(config, storageBackend);
        this.config = config;

        this.supportedModalities = new Set(["text", "image", "audio", "video", "structured"]);

        const storageRoot = resolve("./memory_data");
        mkdirSync(storageRoot, { recursive: true });
        this.docStore = createDocumentStore(resolve(storageRoot, "memory.db"), "sqlite") as SQLiteDocumentStore;

        this.vectorDim = 384;
        this.imageDim = 384;
        this.audioDim = 384;

        const qdrantUrl = process.env.QDRANT_URL;
        const qdrantApiKey = process.env.QDRANT_API_KEY;
        const baseCollection = process.env.QDRANT_COLLECTION ?? "hello_agents_vectors";
        const distance = process.env.QDRANT_DISTANCE ?? "cosine";

        this.vectorStores.set("text", new QdrantVectorStore(qdrantUrl, qdrantApiKey, `${baseCollection}_perceptual_text`, this.vectorDim, distance));
        this.vectorStores.set("image", new QdrantVectorStore(qdrantUrl, qdrantApiKey, `${baseCollection}_perceptual_image`, this.imageDim, distance));
        this.vectorStores.set("audio", new QdrantVectorStore(qdrantUrl, qdrantApiKey, `${baseCollection}_perceptual_audio`, this.audioDim, distance));

        for (const store of this.vectorStores.values()) {
            store.initialize().catch(() => undefined);
        }
    }

    public add(memoryItem: MemoryItem): string {
        const modality = normalizeModality(memoryItem.metadata?.modality ?? "text");
        if (!this.supportedModalities.has(modality)) {
            throw new Error(`不支持的模态类型: ${modality}`);
        }

        const rawData = memoryItem.metadata?.raw_data ?? memoryItem.content;
        const perception = this.encodePerception(rawData, modality, memoryItem.id);

        this.perceptions.set(perception.perceptionId, perception);
        const ids = this.modalityIndex.get(modality) ?? [];
        ids.push(perception.perceptionId);
        this.modalityIndex.set(modality, ids);

        const storedItem: MemoryItem = {
            ...memoryItem,
            metadata: {
                ...(memoryItem.metadata ?? {}),
                perception_id: perception.perceptionId,
                modality,
            },
        };
        this.perceptualMemories.push(storedItem);
        this.vectorIndex.set(memoryItem.id, perception.encoding);

        this.docStore.add_memory(
            memoryItem.id,
            memoryItem.userId ?? "default_user",
            memoryItem.content,
            "perceptual",
            Math.floor((asDate(memoryItem.timestamp).getTime()) / 1000),
            memoryItem.importance,
            {
                perception_id: perception.perceptionId,
                modality,
                context: cloneRecord(memoryItem.metadata?.context as JsonRecord | undefined),
                tags: Array.isArray(memoryItem.metadata?.tags) ? memoryItem.metadata.tags : [],
                raw_data_hash: perception.dataHash,
            },
        );

        const store = this.getVectorStoreForModality(modality);
        store.addVectors(
            [perception.encoding],
            [{
                memory_id: memoryItem.id,
                user_id: memoryItem.userId ?? "default_user",
                memory_type: "perceptual",
                modality,
                importance: memoryItem.importance,
                content: memoryItem.content,
            }],
            [memoryItem.id],
        ).catch(() => undefined);

        return memoryItem.id;
    }

    public retrieve(query: string, limit: number = 5, kwargs: Record<string, unknown> = {}): MemoryItem[] {
        const userId = this.extractUserId(kwargs);
        const targetModality = this.optionalModality(kwargs.target_modality ?? kwargs.targetModality);
        const queryModality = this.optionalModality(kwargs.query_modality ?? kwargs.queryModality) ?? targetModality ?? "text";
        const effectiveModality = targetModality ?? queryModality;

        let hits = this.vectorSearch(query, queryModality, effectiveModality, userId, limit);
        const nowTs = Math.floor(Date.now() / 1000);
        const results: Array<{ score: number; item: MemoryItem }> = [];
        const seen = new Set<string>();

        for (const hit of hits) {
            const metadata = hit.metadata ?? {};
            const memoryId = typeof metadata.memory_id === "string" ? metadata.memory_id : String(hit.id);
            if (seen.has(memoryId)) {
                continue;
            }

            const doc = this.docStore.get_memory(memoryId);
            if (!doc) {
                continue;
            }

            if (targetModality) {
                const modality = normalizeModality(doc.properties?.modality ?? metadata.modality ?? effectiveModality);
                if (modality !== targetModality) {
                    continue;
                }
            }

            const vecScore = toNumber(hit.score, 0);
            const recencyScore = this.calculateRecencyScore(nowTs, doc.timestamp);
            const importance = toNumber(doc.importance, 0.5);
            const combined = this.combineScores(vecScore, recencyScore, importance);

            results.push({
                score: combined,
                item: {
                    id: doc.memory_id,
                    content: doc.content,
                    memoryType: "perceptual",
                    userId: doc.user_id,
                    timestamp: doc.timestamp,
                    importance,
                    metadata: {
                        ...(doc.properties ?? {}),
                        relevance_score: combined,
                        vector_score: vecScore,
                        recency_score: recencyScore,
                    },
                },
            });
            seen.add(memoryId);
        }

        if (results.length === 0) {
            const fallback = this.fallbackSearch(query, queryModality, effectiveModality, userId, limit);
            results.push(...fallback);
        }

        return results
            .sort((left, right) => right.score - left.score)
            .slice(0, Math.max(0, Math.floor(limit)))
            .map(({ item }) => this.cloneMemoryItem(item));
    }

    public update(memoryId: string, content: string | null = null, importance: number | null = null, metadata: JsonRecord | null = null): boolean {
        const memoryIndex = this.perceptualMemories.findIndex((memory) => memory.id === memoryId);
        if (memoryIndex < 0) {
            return false;
        }

        const current = this.perceptualMemories[memoryIndex];
        const modality = normalizeModality(metadata?.modality ?? current.metadata?.modality ?? "text");
        const rawData = metadata?.raw_data ?? content ?? current.content;

        if (content !== null) {
            current.content = content;
        }
        if (importance !== null) {
            current.importance = this.clampImportance(importance);
        }
        if (metadata !== null) {
            current.metadata = {
                ...(current.metadata ?? {}),
                ...metadata,
            };
        }

        const perception = this.encodePerception(rawData, modality, memoryId);
        this.perceptions.set(perception.perceptionId, perception);
        this.vectorIndex.set(memoryId, perception.encoding);
        current.metadata = {
            ...(current.metadata ?? {}),
            perception_id: perception.perceptionId,
            modality,
        };

        this.docStore.update_memory(
            memoryId,
            content,
            importance,
            {
                ...(metadata ?? {}),
                perception_id: perception.perceptionId,
                modality,
            },
        );

        const doc = this.docStore.get_memory(memoryId);
        const store = this.getVectorStoreForModality(modality);
        store.addVectors(
            [perception.encoding],
            [{
                memory_id: memoryId,
                user_id: doc?.user_id ?? current.userId ?? "default_user",
                memory_type: "perceptual",
                modality,
                importance: doc ? toNumber(doc.importance, current.importance) : current.importance,
                content: content ?? current.content,
            }],
            [memoryId],
        ).catch(() => undefined);

        return true;
    }

    public remove(memoryId: string): boolean {
        const index = this.perceptualMemories.findIndex((memory) => memory.id === memoryId);
        if (index < 0) {
            return false;
        }

        const [removed] = this.perceptualMemories.splice(index, 1);
        const perceptionId = removed.metadata?.perception_id;
        if (typeof perceptionId === "string") {
            this.perceptions.delete(perceptionId);
            for (const [modality, ids] of this.modalityIndex.entries()) {
                const next = ids.filter((id) => id !== perceptionId);
                if (next.length > 0) {
                    this.modalityIndex.set(modality, next);
                } else {
                    this.modalityIndex.delete(modality);
                }
            }
        }

        this.vectorIndex.delete(memoryId);
        this.docStore.delete_memory(memoryId);
        for (const store of this.vectorStores.values()) {
            store.delete_memories([memoryId]).catch(() => undefined);
        }

        return true;
    }

    public hasMemory(memoryId: string): boolean {
        return this.perceptualMemories.some((memory) => memory.id === memoryId);
    }

    public forget(strategy: "importance_based" | "time_based" | "capacity_based" = "importance_based", threshold: number = 0.1, maxAgeDays: number = 30): number {
        const currentTime = new Date();
        const toRemove: string[] = [];
        const maxCapacity = Math.max(1, Math.floor(this.config.maxCapacity));

        for (const memory of this.perceptualMemories) {
            let shouldForget = false;

            if (strategy === "importance_based") {
                shouldForget = memory.importance < threshold;
            } else if (strategy === "time_based") {
                shouldForget = memory.timestamp < new Date(currentTime.getTime() - maxAgeDays * 24 * 60 * 60 * 1000);
            } else if (strategy === "capacity_based") {
                shouldForget = this.perceptualMemories.length > maxCapacity;
            }

            if (shouldForget) {
                toRemove.push(memory.id);
            }
        }

        if (strategy === "capacity_based" && toRemove.length > 0) {
            const ordered = [...this.perceptualMemories].sort((left, right) => left.importance - right.importance || left.timestamp.getTime() - right.timestamp.getTime());
            const excess = this.perceptualMemories.length - maxCapacity;
            toRemove.length = 0;
            for (const memory of ordered.slice(0, Math.max(0, excess))) {
                toRemove.push(memory.id);
            }
        }

        let forgotten = 0;
        for (const id of toRemove) {
            if (this.remove(id)) {
                forgotten += 1;
            }
        }

        return forgotten;
    }

    public clear(): void {
        const ids = [...this.perceptualMemories.map((memory) => memory.id)];
        this.perceptualMemories.length = 0;
        this.perceptions.clear();
        this.modalityIndex.clear();
        this.vectorIndex.clear();

        for (const id of ids) {
            this.docStore.delete_memory(id);
        }

        for (const store of this.vectorStores.values()) {
            store.delete_memories(ids).catch(() => undefined);
        }
    }

    public getAll(): MemoryItem[] {
        return this.perceptualMemories.map((memory) => this.cloneMemoryItem(memory));
    }

    public getStats(): JsonRecord {
        const active = this.perceptualMemories;
        const modalityCounts: JsonRecord = {};
        for (const [modality, ids] of this.modalityIndex.entries()) {
            modalityCounts[modality] = ids.length;
        }

        const vectorStores: JsonRecord = {};
        for (const [modality] of this.vectorStores.entries()) {
            vectorStores[modality] = {
                store_type: "qdrant",
                collection_name: `${process.env.QDRANT_COLLECTION ?? "hello_agents_vectors"}_perceptual_${modality}`,
                indexed_vectors: this.getVectorCountForModality(modality),
            };
        }

        return {
            count: active.length,
            forgotten_count: 0,
            total_count: this.perceptualMemories.length,
            perceptions_count: this.perceptions.size,
            modality_counts: modalityCounts,
            supported_modalities: [...this.supportedModalities],
            avg_importance: active.length > 0
                ? active.reduce((sum, memory) => sum + memory.importance, 0) / active.length
                : 0,
            memory_type: "perceptual",
            vector_stores: vectorStores,
            document_store: this.docStore.get_database_stats(),
        };
    }

    public cross_modal_search(query: unknown, queryModality: string, targetModality: string | null = null, limit: number = 5): MemoryItem[] {
        return this.retrieve(String(query), limit, {
            query_modality: queryModality,
            target_modality: targetModality,
        });
    }

    public get_by_modality(modality: string, limit: number = 10): MemoryItem[] {
        const normalized = normalizeModality(modality);
        const ids = this.modalityIndex.get(normalized) ?? [];
        const selected = new Set(ids);
        return this.perceptualMemories
            .filter((memory) => selected.has(String(memory.metadata?.perception_id)))
            .slice(0, Math.max(0, Math.floor(limit)))
            .map((memory) => this.cloneMemoryItem(memory));
    }

    public generate_content(prompt: string, target_modality: string): string | null {
        const normalized = normalizeModality(target_modality);
        if (!this.supportedModalities.has(normalized)) {
            return null;
        }

        const relevant = this.retrieve(prompt, 3, { target_modality: normalized, query_modality: "text" });
        if (relevant.length === 0) {
            return null;
        }

        if (normalized === "text") {
            return `基于感知记忆生成的内容：\n${relevant.map((memory) => memory.content).join("\n")}`;
        }

        return `生成的${normalized}内容（基于${relevant.length}个相关记忆）`;
    }

    public close(): void {
        this.docStore.close();
    }

    private encodePerception(data: unknown, modality: string, memoryId: string): Perception {
        const encoding = this.encodeData(data, modality);
        return new Perception(`perception_${memoryId}`, data, modality, encoding, { source: "memory_system" });
    }

    private encodeData(data: unknown, modality: string): number[] {
        const dim = this.getDimForModality(modality);
        if (normalizeModality(modality) === "text") {
            return this.textEncoder(String(data ?? ""), dim);
        }

        if (normalizeModality(modality) === "image") {
            return this.imageEncoder(data, dim);
        }

        if (normalizeModality(modality) === "audio") {
            return this.audioEncoder(data, dim);
        }

        return this.defaultEncoder(String(data ?? ""), dim);
    }

    private textEncoder(text: string, dim: number): number[] {
        return this.hashToVector(`text:${text}`, dim);
    }

    private imageEncoder(data: unknown, dim: number): number[] {
        return this.hashToVector(`image:${this.dataToString(data)}`, dim);
    }

    private audioEncoder(data: unknown, dim: number): number[] {
        return this.hashToVector(`audio:${this.dataToString(data)}`, dim);
    }

    private defaultEncoder(data: string, dim: number): number[] {
        return this.hashToVector(`default:${data}`, dim);
    }

    private hashToVector(value: string, dim: number): number[] {
        const vector = new Array<number>(dim).fill(0);
        const hash = createHash("sha256").update(value).digest();

        for (let i = 0; i < dim; i += 1) {
            const a = hash[i % hash.length] ?? 0;
            const b = hash[(i * 7) % hash.length] ?? 0;
            vector[i] = ((a << 8) | b) / 65535;
        }

        const norm = Math.sqrt(vector.reduce((sum, current) => sum + current * current, 0));
        if (norm > 0) {
            for (let i = 0; i < vector.length; i += 1) {
                vector[i] /= norm;
            }
        }

        return vector;
    }

    private dataToString(data: unknown): string {
        if (typeof data === "string") {
            return data;
        }

        if (data instanceof Uint8Array) {
            return Buffer.from(data).toString("base64");
        }

        if (Array.isArray(data)) {
            return JSON.stringify(data);
        }

        if (data && typeof data === "object") {
            try {
                return JSON.stringify(data);
            } catch {
                return String(data);
            }
        }

        return String(data ?? "");
    }

    private getDimForModality(modality: string): number {
        const normalized = normalizeModality(modality);
        if (normalized === "image") {
            return this.imageDim;
        }
        if (normalized === "audio") {
            return this.audioDim;
        }
        return this.vectorDim;
    }

    private getVectorStoreForModality(modality: string): QdrantVectorStore {
        const normalized = normalizeModality(modality);
        return this.vectorStores.get(normalized) ?? this.vectorStores.get("text")!;
    }

    private vectorSearch(query: string, queryModality: string, targetModality: string, userId: string | null, limit: number): Array<{ id: string | number; score: number; metadata?: JsonRecord }> {
        const normalizedTarget = normalizeModality(targetModality);
        const normalizedQuery = normalizeModality(queryModality);
        const queryVector = this.encodeData(query, normalizedQuery);
        const hits: Array<{ id: string | number; score: number; metadata?: JsonRecord }> = [];

        for (const [memoryId, vector] of this.vectorIndex.entries()) {
            const memory = this.perceptualMemories.find((item) => item.id === memoryId);
            if (!memory) {
                continue;
            }

            const modality = normalizeModality(memory.metadata?.modality ?? "text");
            if (normalizedTarget && modality !== normalizedTarget) {
                continue;
            }

            if (userId && memory.userId !== userId) {
                continue;
            }

            const score = cosineSimilarity(queryVector, vector);
            if (score <= 0) {
                continue;
            }

            hits.push({
                id: memoryId,
                score,
                metadata: {
                    memory_id: memoryId,
                    user_id: memory.userId,
                    modality,
                    memory_type: "perceptual",
                    perception_id: memory.metadata?.perception_id,
                },
            });
        }

        return hits
            .sort((left, right) => right.score - left.score)
            .slice(0, Math.max(20, limit * 5));
    }

    private fallbackSearch(query: string, queryModality: string, targetModality: string, userId: string | null, limit: number): Array<{ score: number; item: MemoryItem }> {
        const normalizedQuery = query.trim().toLowerCase();
        const nowTs = Math.floor(Date.now() / 1000);
        const matches: Array<{ score: number; item: MemoryItem }> = [];

        for (const memory of this.perceptualMemories) {
            if (userId && memory.userId !== userId) {
                continue;
            }

            const modality = normalizeModality(memory.metadata?.modality ?? "text");
            if (targetModality && modality !== targetModality) {
                continue;
            }

            const contentLower = memory.content.toLowerCase();
            const metadataText = JSON.stringify(memory.metadata ?? {}).toLowerCase();
            let keywordScore = 0;

            if (!normalizedQuery) {
                keywordScore = memory.importance;
            } else if (contentLower.includes(normalizedQuery)) {
                keywordScore = Math.min(1, 0.7 + normalizedQuery.length / Math.max(1, contentLower.length));
            } else {
                for (const token of tokenize(normalizedQuery)) {
                    if (contentLower.includes(token)) {
                        keywordScore += 0.2;
                    }
                    if (metadataText.includes(token)) {
                        keywordScore += 0.1;
                    }
                }
            }

            if (keywordScore <= 0) {
                continue;
            }

            const recencyScore = this.calculateRecencyScore(nowTs, asDate(memory.timestamp));
            const combined = this.combineScores(keywordScore, recencyScore, memory.importance);

            matches.push({
                score: combined,
                item: this.cloneMemoryItem({
                    ...memory,
                    timestamp: asDate(memory.timestamp).getTime(),
                    metadata: {
                        ...(memory.metadata ?? {}),
                        relevance_score: combined,
                        keyword_score: keywordScore,
                        recency_score: recencyScore,
                        query_modality: queryModality,
                    },
                }),
            });
        }

        return matches.sort((left, right) => right.score - left.score).slice(0, Math.max(0, Math.floor(limit)));
    }

    private calculateRecencyScore(nowTs: number, timestamp: Date): number {
        const ageDays = Math.max(0, (nowTs - Math.floor(timestamp.getTime() / 1000)) / 86400);
        return 1 / (1 + ageDays);
    }

    private combineScores(primaryScore: number, recencyScore: number, importance: number): number {
        const base = primaryScore * 0.8 + recencyScore * 0.2;
        const importanceWeight = 0.8 + (this.clampImportance(importance) * 0.4);
        return base * importanceWeight;
    }

    private extractUserId(kwargs: Record<string, unknown>): string | null {
        const candidate = kwargs.userId ?? kwargs.user_id;
        return typeof candidate === "string" && candidate.trim() ? candidate : null;
    }

    private optionalModality(value: unknown): string | null {
        if (typeof value !== "string") {
            return null;
        }

        const normalized = value.trim().toLowerCase();
        return normalized ? normalized : null;
    }

    private clampImportance(value: number): number {
        if (!Number.isFinite(value)) {
            return 0.5;
        }

        return Math.max(0, Math.min(1, value));
    }

    private safeCollectionStats(store: QdrantVectorStore): JsonRecord {
        void store;
        return { store_type: "qdrant" };
    }

    private getVectorCountForModality(modality: string): number {
        const normalized = normalizeModality(modality);
        const ids = this.modalityIndex.get(normalized) ?? [];
        return ids.length;
    }

    private cloneMemoryItem(memory: MemoryItem): MemoryItem {
        return {
            ...memory,
            metadata: memory.metadata ? { ...memory.metadata } : undefined,
        };
    }
}
