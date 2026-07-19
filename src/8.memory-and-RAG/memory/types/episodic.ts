// 情景记忆（事件序列，SQLite+Qdrant）

import { randomUUID } from "crypto";
import { mkdirSync } from "fs";
import { resolve } from "path";

import type { MemoryConfig } from "../config.ts";
import type { MemoryStore } from "../storage/store.ts";
import { createDocumentStore, SQLiteDocumentStore } from "../storage/document_store.ts";
import { QdrantVectorStore } from "../storage/qdrant_store.ts";
import { Memory } from "./memory.ts";
import type { MemoryItem } from "./memory.ts";

type JsonRecord = Record<string, unknown>;

export class Episode {
    public readonly episodeId: string;
    public readonly userId: string;
    public readonly sessionId: string;
    public timestamp: Date;
    public content: string;
    public context: JsonRecord;
    public outcome: string | null;
    public importance: number;
    public metadata: JsonRecord;

    constructor(
        episodeId: string,
        userId: string,
        sessionId: string,
        timestamp: Date,
        content: string,
        context: JsonRecord,
        outcome: string | null = null,
        importance: number = 0.5,
        metadata: JsonRecord = {},
    ) {
        this.episodeId = episodeId;
        this.userId = userId;
        this.sessionId = sessionId;
        this.timestamp = timestamp;
        this.content = content;
        this.context = context;
        this.outcome = outcome;
        this.importance = importance;
        this.metadata = metadata;
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

function getTextEmbedding(text: string, dimension: number): number[] {
    const vector = new Array<number>(dimension).fill(0);
    const normalized = text.trim().toLowerCase();
    if (!normalized) {
        return vector;
    }

    const tokens = tokenize(normalized);
    if (tokens.length === 0) {
        return vector;
    }

    for (const token of tokens) {
        let hash = 0;
        for (let i = 0; i < token.length; i += 1) {
            hash = ((hash << 5) - hash + token.charCodeAt(i)) | 0;
        }
        const index = Math.abs(hash) % dimension;
        vector[index] += 1;
    }

    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    if (norm > 0) {
        for (let i = 0; i < vector.length; i += 1) {
            vector[i] /= norm;
        }
    }

    return vector;
}

export class EpisodicMemory extends Memory {
    private readonly config: MemoryConfig;
    private readonly episodes = new Map<string, Episode>();
    private readonly sessions = new Map<string, string[]>();
    private readonly patternsCache = new Map<string, JsonRecord[]>();
    private lastPatternAnalysis: Date | null = null;
    private readonly docStore: SQLiteDocumentStore;
    private readonly vectorStore: QdrantVectorStore;
    private readonly vectorIndex = new Map<string, number[]>();
    private readonly vectorSize: number;

    constructor(config: MemoryConfig, storageBackend: MemoryStore) {
        super(config, storageBackend);
        this.config = config;

        const storageRoot = resolve("./memory_data");
        mkdirSync(storageRoot, { recursive: true });
        this.docStore = createDocumentStore(resolve(storageRoot, "memory.db"), "sqlite") as SQLiteDocumentStore;

        this.vectorSize = 384;
        this.vectorStore = new QdrantVectorStore(
            process.env.QDRANT_URL,
            process.env.QDRANT_API_KEY,
            process.env.QDRANT_COLLECTION ?? "hello_agents_vectors",
            this.vectorSize,
            process.env.QDRANT_DISTANCE ?? "cosine",
        );
        this.vectorStore.initialize().catch(() => undefined);
    }

    public add(memoryItem: MemoryItem): string {
        const sessionId = typeof memoryItem.metadata?.session_id === "string"
            ? memoryItem.metadata.session_id
            : "default_session";

        const context = cloneRecord(typeof memoryItem.metadata?.context === "object" && memoryItem.metadata?.context !== null
            ? (memoryItem.metadata.context as JsonRecord)
            : undefined);
        const outcome = typeof memoryItem.metadata?.outcome === "string"
            ? memoryItem.metadata.outcome
            : null;
        const participants = Array.isArray(memoryItem.metadata?.participants)
            ? memoryItem.metadata.participants
            : [];
        const tags = Array.isArray(memoryItem.metadata?.tags)
            ? memoryItem.metadata.tags
            : [];

        const episode = new Episode(
            memoryItem.id,
            memoryItem.userId ?? "default_user",
            sessionId,
            asDate(memoryItem.timestamp),
            memoryItem.content,
            context,
            outcome,
            toNumber(memoryItem.importance, 0.5),
            { ...(memoryItem.metadata ?? {}) },
        );

        this.episodes.set(episode.episodeId, episode);
        const sessionEpisodes = this.sessions.get(sessionId) ?? [];
        sessionEpisodes.push(episode.episodeId);
        this.sessions.set(sessionId, sessionEpisodes);

        this.docStore.add_memory(
            memoryItem.id,
            episode.userId,
            memoryItem.content,
            "episodic",
            Math.floor(episode.timestamp.getTime() / 1000),
            episode.importance,
            {
                session_id: sessionId,
                context,
                outcome,
                participants,
                tags,
                metadata: { ...(memoryItem.metadata ?? {}) },
            },
        );

        const embedding = getTextEmbedding(memoryItem.content, this.vectorSize);
        this.vectorIndex.set(memoryItem.id, embedding);
        this.vectorStore.addVectors(
            [embedding],
            [{
                memory_id: memoryItem.id,
                user_id: episode.userId,
                memory_type: "episodic",
                importance: episode.importance,
                session_id: sessionId,
                content: memoryItem.content,
            }],
            [memoryItem.id],
        ).catch(() => undefined);

        return memoryItem.id;
    }

    public retrieve(query: string, limit: number = 5, kwargs: Record<string, unknown> = {}): MemoryItem[] {
        const userId = this.extractUserId(kwargs);
        const sessionId = typeof kwargs.session_id === "string" ? kwargs.session_id : null;
        const timeRange = this.extractTimeRange(kwargs.time_range);
        const importanceThreshold = typeof kwargs.importance_threshold === "number"
            ? kwargs.importance_threshold
            : null;

        const candidateIds = this.collectCandidateIds(userId, timeRange, importanceThreshold);
        const normalizedQuery = query.trim().toLowerCase();
        const nowTs = Math.floor(Date.now() / 1000);
        const scored: Array<{ score: number; item: MemoryItem }> = [];
        const seen = new Set<string>();

        const vectorHits = this.searchVectorHits(query, userId, sessionId, candidateIds);
        for (const hit of vectorHits) {
            const metadata = hit.metadata ?? {};
            const memoryId = typeof metadata.memory_id === "string" ? metadata.memory_id : String(hit.id);
            if (seen.has(memoryId)) {
                continue;
            }

            if (candidateIds && !candidateIds.has(memoryId)) {
                continue;
            }

            const episode = this.episodes.get(memoryId);
            if (!episode || this.isForgotten(episode)) {
                continue;
            }

            if (sessionId && episode.sessionId !== sessionId) {
                continue;
            }

            const item = this.toMemoryItem(episode);
            const vecScore = toNumber(hit.score, 0);
            const recencyScore = this.calculateRecencyScore(nowTs, episode.timestamp);
            const combined = this.combineScores(vecScore, recencyScore, episode.importance);

            item.metadata = {
                ...(item.metadata ?? {}),
                relevance_score: combined,
                vector_score: vecScore,
                recency_score: recencyScore,
            };

            scored.push({ score: combined, item });
            seen.add(memoryId);
        }

        if (scored.length === 0) {
            for (const episode of this.filterEpisodes(userId, sessionId, timeRange)) {
                if (candidateIds && !candidateIds.has(episode.episodeId)) {
                    continue;
                }

                if (this.isForgotten(episode)) {
                    continue;
                }

                const keywordScore = this.calculateKeywordScore(normalizedQuery, episode.content, episode.context);
                if (keywordScore <= 0) {
                    continue;
                }

                const recencyScore = this.calculateRecencyScore(nowTs, episode.timestamp);
                const combined = this.combineScores(keywordScore, recencyScore, episode.importance);
                const item = this.toMemoryItem(episode);
                item.metadata = {
                    ...(item.metadata ?? {}),
                    relevance_score: combined,
                    keyword_score: keywordScore,
                    recency_score: recencyScore,
                };

                scored.push({ score: combined, item });
            }
        }

        return scored
            .sort((left, right) => right.score - left.score)
            .slice(0, Math.max(0, Math.floor(limit)))
            .map(({ item }) => this.cloneMemoryItem(item));
    }

    public update(memoryId: string, content: string | null = null, importance: number | null = null, metadata: JsonRecord | null = null): boolean {
        const episode = this.episodes.get(memoryId);
        if (!episode) {
            return false;
        }

        if (content !== null) {
            episode.content = content;
        }

        if (importance !== null) {
            episode.importance = this.clampImportance(importance);
        }

        if (metadata !== null) {
            episode.metadata = {
                ...episode.metadata,
                ...metadata,
            };

            if (metadata.context && typeof metadata.context === "object") {
                episode.context = {
                    ...episode.context,
                    ...(metadata.context as JsonRecord),
                };
            }

            if (typeof metadata.outcome === "string" || metadata.outcome === null) {
                episode.outcome = metadata.outcome as string | null;
            }
        }

        episode.timestamp = new Date();

        const docUpdated = this.docStore.update_memory(
            memoryId,
            content,
            importance,
            metadata,
        );

        if (content !== null) {
            const embedding = getTextEmbedding(content, this.vectorSize);
            this.vectorIndex.set(memoryId, embedding);
            const doc = this.docStore.get_memory(memoryId);
            const payload = {
                memory_id: memoryId,
                user_id: doc?.user_id ?? episode.userId,
                memory_type: "episodic",
                importance: doc ? toNumber(doc.importance, episode.importance) : episode.importance,
                session_id: episode.sessionId,
                content,
            };

            this.vectorStore.addVectors([embedding], [payload], [memoryId]).catch(() => undefined);
        }

        return docUpdated || true;
    }

    public remove(memoryId: string): boolean {
        const episode = this.episodes.get(memoryId);
        if (!episode) {
            return false;
        }

        this.episodes.delete(memoryId);
        this.vectorIndex.delete(memoryId);
        const sessionEpisodes = this.sessions.get(episode.sessionId);
        if (sessionEpisodes) {
            const next = sessionEpisodes.filter((id) => id !== memoryId);
            if (next.length > 0) {
                this.sessions.set(episode.sessionId, next);
            } else {
                this.sessions.delete(episode.sessionId);
            }
        }

        this.docStore.delete_memory(memoryId);
        this.vectorStore.delete_memories([memoryId]).catch(() => undefined);
        return true;
    }

    public hasMemory(memoryId: string): boolean {
        return this.episodes.has(memoryId);
    }

    public clear(): void {
        const ids = [...this.episodes.keys()];
        this.episodes.clear();
        this.sessions.clear();
        this.patternsCache.clear();
        this.lastPatternAnalysis = null;
        this.vectorIndex.clear();

        for (const id of ids) {
            this.docStore.delete_memory(id);
        }

        this.vectorStore.delete_memories(ids).catch(() => undefined);
    }

    public forget(strategy: "importance_based" | "time_based" | "capacity_based" = "importance_based", threshold: number = 0.1, maxAgeDays: number = 30): number {
        let forgotten = 0;
        const now = new Date();
        const cutoff = new Date(now.getTime() - Math.max(1, maxAgeDays) * 24 * 60 * 60 * 1000);

        const candidates = [...this.episodes.values()];
        let toRemove: Episode[] = [];

        if (strategy === "importance_based") {
            toRemove = candidates.filter((episode) => episode.importance < threshold);
        } else if (strategy === "time_based") {
            toRemove = candidates.filter((episode) => episode.timestamp < cutoff);
        } else if (strategy === "capacity_based") {
            const maxCapacity = Math.max(1, Math.floor(this.config.maxCapacity));
            if (candidates.length > maxCapacity) {
                toRemove = [...candidates]
                    .sort((left, right) => left.importance - right.importance || left.timestamp.getTime() - right.timestamp.getTime())
                    .slice(0, candidates.length - maxCapacity);
            }
        }

        for (const episode of toRemove) {
            if (this.remove(episode.episodeId)) {
                forgotten += 1;
            }
        }

        return forgotten;
    }

    public getAll(): MemoryItem[] {
        return [...this.episodes.values()]
            .sort((left, right) => right.timestamp.getTime() - left.timestamp.getTime())
            .map((episode) => this.cloneMemoryItem(this.toMemoryItem(episode)));
    }

    public getStats(): JsonRecord {
        const active = [...this.episodes.values()].filter((episode) => !this.isForgotten(episode));
        const avgImportance = active.length > 0
            ? active.reduce((sum, episode) => sum + episode.importance, 0) / active.length
            : 0;

        return {
            count: active.length,
            forgottenCount: 0,
            totalCount: this.episodes.size,
            sessionsCount: this.sessions.size,
            avgImportance,
            timeSpanDays: this.calculateTimeSpanDays(),
            memoryType: "episodic",
            vectorStore: {
                store_type: "qdrant",
                collection_name: process.env.QDRANT_COLLECTION ?? "hello_agents_vectors",
                indexed_vectors: this.vectorIndex.size,
            },
            documentStore: this.docStore.get_database_stats(),
        };
    }

    public getSessionEpisodes(sessionId: string): Episode[] {
        const ids = this.sessions.get(sessionId) ?? [];
        return ids.map((id) => this.episodes.get(id)).filter((episode): episode is Episode => Boolean(episode));
    }

    public findPatterns(userId: string | null = null, minFrequency: number = 2): JsonRecord[] {
        const cacheKey = `${userId ?? "all"}:${minFrequency}`;
        const now = new Date();
        const cached = this.patternsCache.get(cacheKey);
        if (cached && this.lastPatternAnalysis && (now.getTime() - this.lastPatternAnalysis.getTime()) < 3600000) {
            return cached;
        }

        const episodes = [...this.episodes.values()].filter((episode) => !userId || episode.userId === userId);
        const keywordPatterns = new Map<string, number>();
        const contextPatterns = new Map<string, number>();

        for (const episode of episodes) {
            for (const token of tokenize(episode.content)) {
                if (token.length > 3) {
                    keywordPatterns.set(token, (keywordPatterns.get(token) ?? 0) + 1);
                }
            }

            for (const [key, value] of Object.entries(episode.context)) {
                const pattern = `${key}:${String(value)}`;
                contextPatterns.set(pattern, (contextPatterns.get(pattern) ?? 0) + 1);
            }
        }

        const patterns: JsonRecord[] = [];
        const baseCount = Math.max(1, episodes.length);

        for (const [pattern, frequency] of keywordPatterns.entries()) {
            if (frequency >= minFrequency) {
                patterns.push({
                    type: "keyword",
                    pattern,
                    frequency,
                    confidence: frequency / baseCount,
                });
            }
        }

        for (const [pattern, frequency] of contextPatterns.entries()) {
            if (frequency >= minFrequency) {
                patterns.push({
                    type: "context",
                    pattern,
                    frequency,
                    confidence: frequency / baseCount,
                });
            }
        }

        patterns.sort((left, right) => Number(right.frequency ?? 0) - Number(left.frequency ?? 0));
        this.patternsCache.set(cacheKey, patterns);
        this.lastPatternAnalysis = now;
        return patterns;
    }

    public getTimeline(userId: string | null = null, limit: number = 50): JsonRecord[] {
        return [...this.episodes.values()]
            .filter((episode) => !userId || episode.userId === userId)
            .sort((left, right) => right.timestamp.getTime() - left.timestamp.getTime())
            .slice(0, Math.max(0, Math.floor(limit)))
            .map((episode) => ({
                episode_id: episode.episodeId,
                timestamp: episode.timestamp.toISOString(),
                content: episode.content.length > 100 ? `${episode.content.slice(0, 100)}...` : episode.content,
                session_id: episode.sessionId,
                importance: episode.importance,
                outcome: episode.outcome,
            }));
    }

    public _filter_episodes(userId: string | null = null, sessionId: string | null = null, timeRange: [Date, Date] | null = null): Episode[] {
        return this.filterEpisodes(userId, sessionId, timeRange);
    }

    public _calculate_time_span(): number {
        return this.calculateTimeSpanDays();
    }

    private searchVectorHits(query: string, userId: string | null, limit: number): Array<{ id: string | number; score: number; metadata?: JsonRecord }> {
        const queryVector = getTextEmbedding(query, this.vectorSize);
        const hits: Array<{ id: string | number; score: number; metadata?: JsonRecord }> = [];

        for (const [memoryId, vector] of this.vectorIndex.entries()) {
            const episode = this.episodes.get(memoryId);
            if (!episode || this.isForgotten(episode)) {
                continue;
            }

            if (userId && episode.userId !== userId) {
                continue;
            }

            const score = this.cosineSimilarity(queryVector, vector);
            if (score <= 0) {
                continue;
            }

            hits.push({
                id: memoryId,
                score,
                metadata: {
                    memory_id: memoryId,
                    user_id: episode.userId,
                    session_id: episode.sessionId,
                    memory_type: "episodic",
                },
            });
        }

        return hits.sort((left, right) => right.score - left.score).slice(0, Math.max(20, limit * 5));
    }

    private collectCandidateIds(userId: string | null, timeRange: [Date, Date] | null, importanceThreshold: number | null): Set<string> | null {
        if (!timeRange && importanceThreshold === null) {
            return null;
        }

        const startTime = timeRange ? Math.floor(timeRange[0].getTime() / 1000) : null;
        const endTime = timeRange ? Math.floor(timeRange[1].getTime() / 1000) : null;
        const docs = this.docStore.search_memories(
            userId,
            "episodic",
            startTime,
            endTime,
            importanceThreshold,
            1000,
        );
        return new Set(docs.map((doc) => String(doc.memory_id)));
    }

    private filterEpisodes(userId: string | null, sessionId: string | null, timeRange: [Date, Date] | null): Episode[] {
        let episodes = [...this.episodes.values()];
        if (userId) {
            episodes = episodes.filter((episode) => episode.userId === userId);
        }
        if (sessionId) {
            episodes = episodes.filter((episode) => episode.sessionId === sessionId);
        }
        if (timeRange) {
            const [start, end] = timeRange;
            episodes = episodes.filter((episode) => episode.timestamp >= start && episode.timestamp <= end);
        }
        return episodes;
    }

    private extractUserId(kwargs: JsonRecord): string | null {
        const candidate = kwargs.userId ?? kwargs.user_id;
        return typeof candidate === "string" && candidate.trim() ? candidate : null;
    }

    private extractTimeRange(value: unknown): [Date, Date] | null {
        if (!Array.isArray(value) || value.length !== 2) {
            return null;
        }

        const start = asDate(value[0]);
        const end = asDate(value[1]);
        if (Number.isNaN(start.getTime()) || Number.isNaN(end.getTime())) {
            return null;
        }

        return [start, end];
    }

    private calculateRecencyScore(nowTs: number, timestamp: Date): number {
        const ageDays = Math.max(0, (nowTs - Math.floor(timestamp.getTime() / 1000)) / 86400);
        return 1 / (1 + ageDays);
    }

    private calculateKeywordScore(query: string, content: string, context: JsonRecord): number {
        if (!query) {
            return 0;
        }

        const contentLower = content.toLowerCase();
        const contextText = JSON.stringify(context).toLowerCase();
        let score = 0;

        if (contentLower.includes(query)) {
            score += 0.7;
        }

        for (const token of tokenize(query)) {
            if (contentLower.includes(token)) {
                score += 0.2;
            }
            if (contextText.includes(token)) {
                score += 0.1;
            }
        }

        return Math.min(1, score);
    }

    private combineScores(primaryScore: number, recencyScore: number, importance: number): number {
        const baseRelevance = primaryScore * 0.8 + recencyScore * 0.2;
        const importanceWeight = 0.8 + (this.clampImportance(importance) * 0.4);
        return baseRelevance * importanceWeight;
    }

    private calculateTimeSpanDays(): number {
        if (this.episodes.size === 0) {
            return 0;
        }

        const timestamps = [...this.episodes.values()].map((episode) => episode.timestamp.getTime());
        const min = Math.min(...timestamps);
        const max = Math.max(...timestamps);
        return Math.max(0, Math.floor((max - min) / 86400000));
    }

    private isForgotten(episode: Episode): boolean {
        return episode.context["forgotten"] === true || episode.metadata["forgotten"] === true;
    }

    private clampImportance(value: number): number {
        if (!Number.isFinite(value)) {
            return 0.5;
        }

        return Math.max(0, Math.min(1, value));
    }

    private toMemoryItem(episode: Episode): MemoryItem {
        return {
            id: episode.episodeId,
            content: episode.content,
            memoryType: "episodic",
            importance: episode.importance,
            userId: episode.userId,
            timestamp: episode.timestamp.getTime(),
            metadata: {
                session_id: episode.sessionId,
                context: cloneRecord(episode.context),
                outcome: episode.outcome,
                ...episode.metadata,
            },
        };
    }

    private cosineSimilarity(left: number[], right: number[]): number {
        const size = Math.min(left.length, right.length);
        let dot = 0;
        let leftNorm = 0;
        let rightNorm = 0;

        for (let i = 0; i < size; i += 1) {
            const leftValue = left[i] ?? 0;
            const rightValue = right[i] ?? 0;
            dot += leftValue * rightValue;
            leftNorm += leftValue * leftValue;
            rightNorm += rightValue * rightValue;
        }

        if (leftNorm === 0 || rightNorm === 0) {
            return 0;
        }

        return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
    }

    private cloneMemoryItem(memory: MemoryItem): MemoryItem {
        return {
            ...memory,
            metadata: memory.metadata ? { ...memory.metadata } : undefined,
        };
    }

    public close(): void {
        this.docStore.close();
    }
}
