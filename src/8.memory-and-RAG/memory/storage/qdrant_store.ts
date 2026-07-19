// Qdrant向量存储（高性能向量检索）

import { randomUUID } from "crypto";

type Primitive = string | number | boolean;

type QdrantDistance = "Cosine" | "Dot" | "Euclid";

type QdrantHit = {
    id: string | number;
    score: number;
    payload?: Record<string, unknown>;
};

type QueryVector = number[];

interface QdrantPoint {
    id: string | number;
    vector: number[];
    payload: Record<string, unknown>;
}

interface QdrantFilterCondition {
    key: string;
    match: { value: Primitive };
}

interface QdrantFilter {
    must?: QdrantFilterCondition[];
    should?: QdrantFilterCondition[];
}

interface QdrantCollectionInfo {
    vectors_count?: number;
    indexed_vectors_count?: number;
    points_count?: number;
    segments_count?: number;
}

const DEFAULT_COLLECTION = "hello_agents_vectors";
const DEFAULT_TIMEOUT_MS = 30_000;

function toQdrantDistance(distance: string): QdrantDistance {
    switch (distance.toLowerCase()) {
        case "dot":
            return "Dot";
        case "euclidean":
        case "euclid":
            return "Euclid";
        case "cosine":
        default:
            return "Cosine";
    }
}

function isPrimitive(value: unknown): value is Primitive {
    return typeof value === "string" || typeof value === "number" || typeof value === "boolean";
}

function normalizePayload(payload: Record<string, unknown>): Record<string, unknown> {
    const normalized: Record<string, unknown> = { ...payload };

    if (normalized.external !== undefined && typeof normalized.external !== "boolean") {
        const raw = normalized.external;
        normalized.external = ["1", "true", "yes", "on"].includes(String(raw).toLowerCase());
    }

    if (normalized.timestamp !== undefined && typeof normalized.timestamp !== "number") {
        const ts = Number(normalized.timestamp);
        if (!Number.isNaN(ts)) {
            normalized.timestamp = ts;
        }
    }

    normalized.timestamp = typeof normalized.timestamp === "number"
        ? normalized.timestamp
        : Math.floor(Date.now() / 1000);
    normalized.added_at = Math.floor(Date.now() / 1000);

    return normalized;
}

function buildFilter(where?: Record<string, unknown>): QdrantFilter | undefined {
    if (!where || Object.keys(where).length === 0) {
        return undefined;
    }

    const must: QdrantFilterCondition[] = [];
    for (const [key, value] of Object.entries(where)) {
        if (isPrimitive(value)) {
            must.push({ key, match: { value } });
        }
    }

    return must.length > 0 ? { must } : undefined;
}

function buildShouldFilter(memoryIds: string[]): QdrantFilter | undefined {
    const should = memoryIds.map((memoryId) => ({
        key: "memory_id",
        match: { value: memoryId },
    }));
    return should.length > 0 ? { should } : undefined;
}

async function requestJson<T>(url: string, init: RequestInit, timeoutMs: number): Promise<T> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);

    try {
        const response = await fetch(url, {
            ...init,
            signal: controller.signal,
            headers: {
                "content-type": "application/json",
                ...(init.headers ?? {}),
            },
        });

        if (!response.ok) {
            const body = await response.text();
            throw new Error(`Qdrant HTTP ${response.status}: ${body}`);
        }

        if (response.status === 204) {
            return undefined as T;
        }

        return (await response.json()) as T;
    } finally {
        clearTimeout(timer);
    }
}

export class QdrantVectorStore {
    private readonly baseUrl: string;
    private readonly apiKey?: string;
    private readonly collectionName: string;
    private readonly vectorSize: number;
    private readonly distance: QdrantDistance;
    private readonly timeoutMs: number;
    private readonly hnswM: number;
    private readonly hnswEfConstruct: number;
    private readonly searchEf: number;
    private readonly searchExact: boolean;

    constructor(
        url?: string,
        apiKey?: string,
        collectionName: string = DEFAULT_COLLECTION,
        vectorSize: number = 384,
        distance: string = "cosine",
        timeout: number = 30,
    ) {
        const rawUrl = url ?? process.env.QDRANT_URL ?? "http://localhost:6333";
        this.baseUrl = rawUrl.replace(/\/$/, "");
        this.apiKey = apiKey ?? process.env.QDRANT_API_KEY;
        this.collectionName = collectionName;
        this.vectorSize = vectorSize;
        this.distance = toQdrantDistance(distance);
        this.timeoutMs = Math.max(1, timeout) * 1000;
        this.hnswM = Number.parseInt(process.env.QDRANT_HNSW_M ?? "32", 10) || 32;
        this.hnswEfConstruct = Number.parseInt(process.env.QDRANT_HNSW_EF_CONSTRUCT ?? "256", 10) || 256;
        this.searchEf = Number.parseInt(process.env.QDRANT_SEARCH_EF ?? "128", 10) || 128;
        this.searchExact = (process.env.QDRANT_SEARCH_EXACT ?? "0") === "1";
    }

    private get headers(): Record<string, string> {
        return this.apiKey ? { "api-key": this.apiKey } : {};
    }

    private async request<T>(path: string, init: RequestInit = {}): Promise<T> {
        return requestJson<T>(`${this.baseUrl}${path}`, {
            ...init,
            headers: {
                ...this.headers,
                ...(init.headers ?? {}),
            },
        }, this.timeoutMs);
    }

    async initialize(): Promise<void> {
        await this.request("/collections", { method: "GET" });
        await this.ensureCollection();
    }

    async ensureCollection(): Promise<void> {
        const collections = await this.request<{ result?: { collections?: Array<{ name: string }> } }>("/collections", {
            method: "GET",
        });
        const collectionNames = collections.result?.collections?.map((item) => item.name) ?? [];

        if (!collectionNames.includes(this.collectionName)) {
            await this.request(`/collections/${encodeURIComponent(this.collectionName)}`, {
                method: "PUT",
                body: JSON.stringify({
                    vectors: {
                        size: this.vectorSize,
                        distance: this.distance,
                    },
                    hnsw_config: {
                        m: this.hnswM,
                        ef_construct: this.hnswEfConstruct,
                    },
                }),
            });
        } else {
            await this.request(`/collections/${encodeURIComponent(this.collectionName)}`, {
                method: "PATCH",
                body: JSON.stringify({
                    hnsw_config: {
                        m: this.hnswM,
                        ef_construct: this.hnswEfConstruct,
                    },
                }),
            }).catch(() => undefined);
        }

        await this.ensurePayloadIndexes();
    }

    private async ensurePayloadIndexes(): Promise<void> {
        const indexes = [
            ["memory_type", "keyword"],
            ["user_id", "keyword"],
            ["memory_id", "keyword"],
            ["timestamp", "integer"],
            ["modality", "keyword"],
            ["source", "keyword"],
            ["external", "bool"],
            ["namespace", "keyword"],
            ["is_rag_data", "bool"],
            ["rag_namespace", "keyword"],
            ["data_source", "keyword"],
        ] as const;

        for (const [fieldName, fieldSchema] of indexes) {
            await this.request(`/collections/${encodeURIComponent(this.collectionName)}/index`, {
                method: "PUT",
                body: JSON.stringify({
                    field_name: fieldName,
                    field_schema: fieldSchema,
                }),
            }).catch(() => undefined);
        }
    }

    async addVectors(
        vectors: number[][],
        metadata: Array<Record<string, unknown>>,
        ids?: string[],
    ): Promise<boolean> {
        if (!vectors.length) {
            return false;
        }

        const pointIds = ids ?? vectors.map(() => randomUUID());
        const points: QdrantPoint[] = [];

        for (let index = 0; index < vectors.length; index += 1) {
            const vector = vectors[index];
            if (!Array.isArray(vector) || vector.length !== this.vectorSize) {
                continue;
            }

            const payload = normalizePayload({ ...(metadata[index] ?? {}) });
            const pointId = pointIds[index] ?? randomUUID();
            points.push({
                id: pointId,
                vector,
                payload,
            });
        }

        if (!points.length) {
            return false;
        }

        await this.request(`/collections/${encodeURIComponent(this.collectionName)}/points?wait=true`, {
            method: "PUT",
            body: JSON.stringify({ points }),
        });

        return true;
    }

    async searchSimilar(
        queryVector: QueryVector,
        limit: number = 10,
        scoreThreshold?: number,
        where?: Record<string, unknown>,
    ): Promise<Array<{ id: string | number; score: number; metadata: Record<string, unknown> }>> {
        if (!Array.isArray(queryVector) || queryVector.length !== this.vectorSize) {
            return [];
        }

        const body: Record<string, unknown> = {
            vector: queryVector,
            limit,
            with_payload: true,
            with_vector: false,
            params: {
                hnsw_ef: this.searchEf,
                exact: this.searchExact,
            },
        };

        const filter = buildFilter(where);
        if (filter) {
            body.filter = filter;
        }
        if (scoreThreshold !== undefined) {
            body.score_threshold = scoreThreshold;
        }

        const response = await this.request<{ result?: QdrantHit[] }>(
            `/collections/${encodeURIComponent(this.collectionName)}/points/search`,
            {
                method: "POST",
                body: JSON.stringify(body),
            },
        );

        const hits = response.result ?? [];
        return hits.map((hit) => ({
            id: hit.id,
            score: hit.score,
            metadata: hit.payload ?? {},
        }));
    }

    async deleteVectors(ids: Array<string | number>): Promise<boolean> {
        if (!ids.length) {
            return true;
        }

        await this.request(`/collections/${encodeURIComponent(this.collectionName)}/points/delete?wait=true`, {
            method: "POST",
            body: JSON.stringify({
                points: ids,
            }),
        });

        return true;
    }

    async deleteMemories(memoryIds: string[]): Promise<void> {
        if (!memoryIds.length) {
            return;
        }

        const filter = buildShouldFilter(memoryIds);
        if (!filter) {
            return;
        }

        await this.request(`/collections/${encodeURIComponent(this.collectionName)}/points/delete?wait=true`, {
            method: "POST",
            body: JSON.stringify({
                filter,
            }),
        });
    }

    async clearCollection(): Promise<boolean> {
        await this.request(`/collections/${encodeURIComponent(this.collectionName)}`, {
            method: "DELETE",
        });

        await this.ensureCollection();
        return true;
    }

    async getCollectionInfo(): Promise<Record<string, unknown>> {
        const response = await this.request<{ result?: QdrantCollectionInfo }>(`/collections/${encodeURIComponent(this.collectionName)}`, {
            method: "GET",
        });

        const info = response.result ?? {};
        return {
            name: this.collectionName,
            vectors_count: info.vectors_count ?? 0,
            indexed_vectors_count: info.indexed_vectors_count ?? 0,
            points_count: info.points_count ?? 0,
            segments_count: info.segments_count ?? 0,
            config: {
                vector_size: this.vectorSize,
                distance: this.distance,
            },
        };
    }

    async getCollectionStats(): Promise<Record<string, unknown>> {
        const info = await this.getCollectionInfo();
        return {
            ...info,
            store_type: "qdrant",
        };
    }

    async healthCheck(): Promise<boolean> {
        try {
            await this.request("/collections", { method: "GET" });
            return true;
        } catch {
            return false;
        }
    }

    async close(): Promise<void> {
        return;
    }
}

export class QdrantConnectionManager {
    private static readonly instances = new Map<string, QdrantVectorStore>();
    private static readonly pending = new Map<string, Promise<QdrantVectorStore>>();

    static async getInstance(
        url?: string,
        apiKey?: string,
        collectionName: string = DEFAULT_COLLECTION,
        vectorSize: number = 384,
        distance: string = "cosine",
        timeout: number = 30,
    ): Promise<QdrantVectorStore> {
        const key = `${url ?? process.env.QDRANT_URL ?? "local"}::${collectionName}`;

        const cached = this.instances.get(key);
        if (cached) {
            return cached;
        }

        const pending = this.pending.get(key);
        if (pending) {
            return pending;
        }

        const created = (async () => {
            const store = new QdrantVectorStore(url, apiKey, collectionName, vectorSize, distance, timeout);
            await store.initialize();
            this.instances.set(key, store);
            this.pending.delete(key);
            return store;
        })().catch((error) => {
            this.pending.delete(key);
            throw error;
        });

        this.pending.set(key, created);
        return created;
    }
}

