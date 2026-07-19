// 语义记忆（知识图谱，Qdrant+Neo4j）

import type { MemoryConfig } from "../config.ts";
import type { MemoryStore } from "../storage/store.ts";
import { QdrantVectorStore } from "../storage/qdrant_store.ts";
import { Neo4jGraphStore } from "../storage/neo4j_store.ts";
import { Memory } from "./memory.ts";
import type { MemoryItem } from "./memory.ts";

type JsonRecord = Record<string, unknown>;

class SemanticEmbedder {
    public readonly dimension: number;

    constructor(dimension: number = 384) {
        this.dimension = Math.max(1, Math.floor(dimension));
    }

    encode(text: string): number[] {
        const vector = new Array<number>(this.dimension).fill(0);
        const tokens = text
            .toLowerCase()
            .split(/\s+/)
            .map((token) => token.trim())
            .filter(Boolean);

        if (tokens.length === 0) {
            return vector;
        }

        for (const token of tokens) {
            let hash = 0;
            for (let i = 0; i < token.length; i += 1) {
                hash = ((hash << 5) - hash + token.charCodeAt(i)) | 0;
            }
            const index = Math.abs(hash) % this.dimension;
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
}

export class Entity {
    public readonly entityId: string;
    public name: string;
    public entityType: string;
    public description: string;
    public properties: JsonRecord;
    public createdAt: Date;
    public updatedAt: Date;
    public frequency: number;

    constructor(
        entityId: string,
        name: string,
        entityType: string = "MISC",
        description: string = "",
        properties: JsonRecord = {},
    ) {
        this.entityId = entityId;
        this.name = name;
        this.entityType = entityType;
        this.description = description;
        this.properties = { ...properties };
        this.createdAt = new Date();
        this.updatedAt = new Date();
        this.frequency = 1;
    }

    toDict(): JsonRecord {
        return {
            entity_id: this.entityId,
            name: this.name,
            entity_type: this.entityType,
            description: this.description,
            properties: { ...this.properties },
            frequency: this.frequency,
        };
    }
}

export class Relation {
    public readonly fromEntity: string;
    public readonly toEntity: string;
    public readonly relationType: string;
    public strength: number;
    public evidence: string;
    public properties: JsonRecord;
    public createdAt: Date;
    public frequency: number;

    constructor(
        fromEntity: string,
        toEntity: string,
        relationType: string,
        strength: number = 1.0,
        evidence: string = "",
        properties: JsonRecord = {},
    ) {
        this.fromEntity = fromEntity;
        this.toEntity = toEntity;
        this.relationType = relationType;
        this.strength = strength;
        this.evidence = evidence;
        this.properties = { ...properties };
        this.createdAt = new Date();
        this.frequency = 1;
    }

    toDict(): JsonRecord {
        return {
            from_entity: this.fromEntity,
            to_entity: this.toEntity,
            relation_type: this.relationType,
            strength: this.strength,
            evidence: this.evidence,
            properties: { ...this.properties },
            frequency: this.frequency,
        };
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

function normalizeRelationType(value: string): string {
    const normalized = value.trim().toUpperCase().replace(/[^A-Z0-9_]/g, "_");
    return normalized || "RELATED_TO";
}

function normalizeEntityType(value: string): string {
    const normalized = value.trim().toUpperCase();
    return normalized || "MISC";
}

export class SemanticMemory extends Memory {
    private readonly config: MemoryConfig;
    private readonly embeddingModel: SemanticEmbedder;
    private readonly vectorStore: QdrantVectorStore;
    private readonly graphStore: Neo4jGraphStore;
    private readonly entities = new Map<string, Entity>();
    private relations: Relation[] = [];
    private readonly semanticMemories: MemoryItem[] = [];
    private readonly memoryEmbeddings = new Map<string, number[]>();
    private readonly entityUsage = new Map<string, number>();
    private readonly relationUsage = new Map<string, number>();
    private readonly memoryEntityIndex = new Map<string, string[]>();

    constructor(config: MemoryConfig, storageBackend: MemoryStore) {
        super(config, storageBackend);
        this.config = config;

        this.embeddingModel = new SemanticEmbedder(384);

        const qdrantUrl = process.env.QDRANT_URL;
        const qdrantApiKey = process.env.QDRANT_API_KEY;
        const collection = process.env.QDRANT_COLLECTION ?? "hello_agents_vectors";
        const distance = process.env.QDRANT_DISTANCE ?? "cosine";
        this.vectorStore = new QdrantVectorStore(qdrantUrl, qdrantApiKey, `${collection}_semantic`, this.embeddingModel.dimension, distance);
        this.vectorStore.initialize().catch(() => undefined);

        const neo4jUri = process.env.NEO4J_URI ?? "bolt://localhost:7687";
        const neo4jUsername = process.env.NEO4J_USERNAME ?? "neo4j";
        const neo4jPassword = process.env.NEO4J_PASSWORD ?? "hello-agents-password";
        const neo4jDatabase = process.env.NEO4J_DATABASE ?? "neo4j";
        this.graphStore = new Neo4jGraphStore(neo4jUri, neo4jUsername, neo4jPassword, neo4jDatabase);
        this.graphStore.initialize().catch(() => undefined);
    }

    public add(memoryItem: MemoryItem): string {
        const embedding = this.embeddingModel.encode(memoryItem.content);
        this.memoryEmbeddings.set(memoryItem.id, embedding);

        const extractedEntities = this.extractEntities(memoryItem.content, memoryItem.metadata ?? {});
        const extractedRelations = this.extractRelations(memoryItem.content, extractedEntities);

        for (const entity of extractedEntities) {
            this.addOrUpdateEntity(entity);
            this.addEntityToGraph(entity, memoryItem);
        }

        for (const relation of extractedRelations) {
            this.addOrUpdateRelation(relation);
            this.addRelationToGraph(relation, memoryItem);
        }

        const metadata = {
            ...(memoryItem.metadata ?? {}),
            entities: extractedEntities.map((entity) => entity.entityId),
            relations: extractedRelations.map((relation) => `${relation.fromEntity}-${relation.relationType}-${relation.toEntity}`),
            entity_count: extractedEntities.length,
            relation_count: extractedRelations.length,
        };

        const storedItem: MemoryItem = {
            ...memoryItem,
            metadata,
        };

        this.semanticMemories.push(storedItem);
        this.memoryEntityIndex.set(memoryItem.id, extractedEntities.map((entity) => entity.entityId));

        this.vectorStore.addVectors(
            [embedding],
            [{
                memory_id: memoryItem.id,
                user_id: memoryItem.userId ?? "default_user",
                memory_type: "semantic",
                content: memoryItem.content,
                importance: memoryItem.importance,
                entity_count: extractedEntities.length,
                relation_count: extractedRelations.length,
                entities: extractedEntities.map((entity) => entity.entityId),
            }],
            [memoryItem.id],
        ).catch(() => undefined);

        return memoryItem.id;
    }

    public retrieve(query: string, limit: number = 5, kwargs: Record<string, unknown> = {}): MemoryItem[] {
        const userId = typeof kwargs.user_id === "string"
            ? kwargs.user_id
            : typeof kwargs.userId === "string"
                ? kwargs.userId
                : null;

        const vectorResults = this.vectorSearch(query, limit * 2, userId);
        const graphResults = this.graphSearch(query, limit * 2, userId);
        const combinedResults = this.combineAndRankResults(vectorResults, graphResults, query, limit);

        const scores = combinedResults.map((result) => result.combined_score ?? result.vector_score ?? 0);
        const probs = this.softmax(scores);

        const resultMemories: MemoryItem[] = [];
        for (let index = 0; index < combinedResults.length; index += 1) {
            const result = combinedResults[index];
            const memoryId = String(result.memory_id);
            const memory = this.semanticMemories.find((item) => item.id === memoryId);
            if (memory && memory.metadata?.forgotten === true) {
                continue;
            }

            const timestamp = asDate(result.timestamp ?? memory?.timestamp);
            const memoryItem: MemoryItem = {
                id: memoryId,
                content: String(result.content ?? memory?.content ?? ""),
                memoryType: "semantic",
                userId: String(result.user_id ?? memory?.userId ?? "default"),
                timestamp: timestamp.getTime(),
                importance: Number(result.importance ?? memory?.importance ?? 0.5),
                metadata: {
                    ...(cloneRecord(result.metadata as JsonRecord | undefined)),
                    combined_score: result.combined_score ?? 0,
                    vector_score: result.vector_score ?? 0,
                    graph_score: result.graph_score ?? 0,
                    probability: probs[index] ?? 0,
                },
            };
            resultMemories.push(memoryItem);
        }

        return resultMemories.slice(0, Math.max(0, Math.floor(limit)));
    }

    public update(memoryId: string, content: string | null = null, importance: number | null = null, metadata: JsonRecord | null = null): boolean {
        const memory = this.semanticMemories.find((item) => item.id === memoryId);
        if (!memory) {
            return false;
        }

        if (content !== null) {
            memory.content = content;
            const embedding = this.embeddingModel.encode(content);
            this.memoryEmbeddings.set(memoryId, embedding);

            const oldEntities = this.memoryEntityIndex.get(memoryId) ?? [];
            this.cleanupEntitiesAndRelations(oldEntities, memoryId);

            const extractedEntities = this.extractEntities(content, memory.metadata ?? {});
            const extractedRelations = this.extractRelations(content, extractedEntities);

            for (const entity of extractedEntities) {
                this.addOrUpdateEntity(entity);
                this.addEntityToGraph(entity, memory);
            }

            for (const relation of extractedRelations) {
                this.addOrUpdateRelation(relation);
                this.addRelationToGraph(relation, memory);
            }

            memory.metadata = {
                ...(memory.metadata ?? {}),
                entities: extractedEntities.map((entity) => entity.entityId),
                relations: extractedRelations.map((relation) => `${relation.fromEntity}-${relation.relationType}-${relation.toEntity}`),
                entity_count: extractedEntities.length,
                relation_count: extractedRelations.length,
            };
            this.memoryEntityIndex.set(memoryId, extractedEntities.map((entity) => entity.entityId));

            this.vectorStore.addVectors(
                [embedding],
                [{
                    memory_id: memoryId,
                    user_id: memory.userId ?? "default_user",
                    memory_type: "semantic",
                    content,
                    importance: importance ?? memory.importance,
                    entities: extractedEntities.map((entity) => entity.entityId),
                    entity_count: extractedEntities.length,
                    relation_count: extractedRelations.length,
                }],
                [memoryId],
            ).catch(() => undefined);
        }

        if (importance !== null) {
            memory.importance = this.clampImportance(importance);
        }

        if (metadata !== null) {
            memory.metadata = {
                ...(memory.metadata ?? {}),
                ...metadata,
            };
        }

        return true;
    }

    public remove(memoryId: string): boolean {
        const index = this.semanticMemories.findIndex((item) => item.id === memoryId);
        if (index < 0) {
            return false;
        }

        const [removed] = this.semanticMemories.splice(index, 1);
        const entityIds = this.memoryEntityIndex.get(memoryId) ?? [];
        this.cleanupEntitiesAndRelations(entityIds, memoryId);
        this.memoryEntityIndex.delete(memoryId);
        this.memoryEmbeddings.delete(memoryId);

        this.vectorStore.delete_memories([memoryId]).catch(() => undefined);
        return Boolean(removed);
    }

    public hasMemory(memoryId: string): boolean {
        return this.semanticMemories.some((item) => item.id === memoryId);
    }

    public forget(strategy: "importance_based" | "time_based" | "capacity_based" = "importance_based", threshold: number = 0.1, maxAgeDays: number = 30): number {
        let forgottenCount = 0;
        const now = new Date();
        const cutoff = new Date(now.getTime() - Math.max(1, maxAgeDays) * 24 * 60 * 60 * 1000);

        const toRemove: string[] = [];
        for (const memory of this.semanticMemories) {
            let shouldForget = false;

            if (strategy === "importance_based") {
                shouldForget = memory.importance < threshold;
            } else if (strategy === "time_based") {
                shouldForget = asDate(memory.timestamp) < cutoff;
            } else if (strategy === "capacity_based") {
                shouldForget = this.semanticMemories.length > Math.max(1, Math.floor(this.config.maxCapacity));
            }

            if (shouldForget) {
                toRemove.push(memory.id);
            }
        }

        if (strategy === "capacity_based" && toRemove.length > 0) {
            const ordered = [...this.semanticMemories].sort((left, right) => left.importance - right.importance || asDate(left.timestamp).getTime() - asDate(right.timestamp).getTime());
            const excessCount = this.semanticMemories.length - Math.max(1, Math.floor(this.config.maxCapacity));
            toRemove.length = 0;
            for (const memory of ordered.slice(0, Math.max(0, excessCount))) {
                toRemove.push(memory.id);
            }
        }

        for (const memoryId of toRemove) {
            if (this.remove(memoryId)) {
                forgottenCount += 1;
            }
        }

        return forgottenCount;
    }

    public clear(): void {
        this.semanticMemories.length = 0;
        this.memoryEmbeddings.clear();
        this.entities.clear();
        this.relations.length = 0;
        this.entityUsage.clear();
        this.relationUsage.clear();
        this.memoryEntityIndex.clear();

        this.vectorStore.clearCollection().catch(() => undefined);
        this.graphStore.clearAll().catch(() => undefined);
    }

    public getAll(): MemoryItem[] {
        return this.semanticMemories.map((memory) => this.cloneMemoryItem(memory));
    }

    public getStats(): JsonRecord {
        const graphStats = this.safeGraphStats();
        const activeMemories = this.semanticMemories;

        return {
            count: activeMemories.length,
            forgotten_count: 0,
            total_count: this.semanticMemories.length,
            entities_count: this.entities.size,
            relations_count: this.relations.length,
            graph_nodes: graphStats.total_nodes ?? 0,
            graph_edges: graphStats.total_relationships ?? 0,
            avg_importance: activeMemories.length > 0
                ? activeMemories.reduce((sum, memory) => sum + memory.importance, 0) / activeMemories.length
                : 0,
            memory_type: "semantic",
        };
    }

    public getEntity(entityId: string): Entity | undefined {
        return this.entities.get(entityId);
    }

    public searchEntities(query: string, limit: number = 10): Entity[] {
        const normalized = query.toLowerCase();
        const scored: Array<{ score: number; entity: Entity }> = [];

        for (const entity of this.entities.values()) {
            let score = 0;
            if (entity.name.toLowerCase().includes(normalized)) {
                score += 2;
            }
            if (entity.entityType.toLowerCase().includes(normalized)) {
                score += 1;
            }
            if (entity.description.toLowerCase().includes(normalized)) {
                score += 0.5;
            }

            score *= Math.log(1 + entity.frequency);
            if (score > 0) {
                scored.push({ score, entity });
            }
        }

        return scored
            .sort((left, right) => right.score - left.score)
            .slice(0, Math.max(0, Math.floor(limit)))
            .map(({ entity }) => entity);
    }

    public getRelatedEntities(entityId: string, relationTypes: string[] | null = null, maxHops: number = 2): Array<{ entity: Entity; relation_type: string; strength: number; distance: number }> {
        const related = new Map<string, { entity: Entity; relation_type: string; strength: number; distance: number }>();
        const visited = new Set<string>([entityId]);
        const queue: Array<{ id: string; distance: number }> = [{ id: entityId, distance: 0 }];
        const relationFilter = relationTypes ? new Set(relationTypes.map((item) => item.toUpperCase())) : null;

        while (queue.length > 0) {
            const current = queue.shift();
            if (!current || current.distance >= maxHops) {
                continue;
            }

            for (const relation of this.relations) {
                const matchesType = !relationFilter || relationFilter.has(relation.relationType.toUpperCase());
                if (!matchesType) {
                    continue;
                }

                let nextEntityId: string | null = null;
                if (relation.fromEntity === current.id) {
                    nextEntityId = relation.toEntity;
                } else if (relation.toEntity === current.id) {
                    nextEntityId = relation.fromEntity;
                }

                if (!nextEntityId || visited.has(nextEntityId)) {
                    continue;
                }

                const entity = this.entities.get(nextEntityId);
                if (!entity) {
                    continue;
                }

                visited.add(nextEntityId);
                queue.push({ id: nextEntityId, distance: current.distance + 1 });
                related.set(nextEntityId, {
                    entity,
                    relation_type: relation.relationType,
                    strength: relation.strength,
                    distance: current.distance + 1,
                });
            }
        }

        return [...related.values()].sort((left, right) => left.distance - right.distance || right.strength - left.strength);
    }

    public exportKnowledgeGraph(): JsonRecord {
        const graphStats = this.safeGraphStats();
        return {
            entities: Object.fromEntries([...this.entities.entries()].map(([id, entity]) => [id, entity.toDict()])),
            relations: this.relations.map((relation) => relation.toDict()),
            graph_stats: {
                total_nodes: graphStats.total_nodes ?? 0,
                entity_nodes: graphStats.entity_nodes ?? 0,
                memory_nodes: graphStats.memory_nodes ?? 0,
                total_relationships: graphStats.total_relationships ?? 0,
                cached_entities: this.entities.size,
                cached_relations: this.relations.length,
            },
        };
    }

    private extractEntities(text: string, metadata: JsonRecord): Entity[] {
        const entities: Entity[] = [];
        const tokens = tokenize(text);
        const seen = new Set<string>();

        for (const token of tokens) {
            if (token.length < 2) {
                continue;
            }

            const normalized = token.replace(/[^\p{L}\p{N}_-]/gu, "").trim();
            if (!normalized || seen.has(normalized.toLowerCase())) {
                continue;
            }

            seen.add(normalized.toLowerCase());
            const entityType = this.inferEntityType(normalized, metadata);
            const entityId = `entity_${this.hashString(normalized + entityType)}`;
            const description = `从文本中识别的${entityType}实体`;

            entities.push(new Entity(entityId, normalized, entityType, description, {
                source: "semantic_memory",
                mentions: 1,
            }));
        }

        return entities;
    }

    private extractRelations(text: string, entities: Entity[]): Relation[] {
        const relations: Relation[] = [];
        for (let i = 0; i < entities.length; i += 1) {
            for (let j = i + 1; j < entities.length; j += 1) {
                relations.push(new Relation(
                    entities[i].entityId,
                    entities[j].entityId,
                    "CO_OCCURS",
                    0.5,
                    text.slice(0, 120),
                    { evidence_type: "co_occurrence" },
                ));
            }
        }

        return relations;
    }

    private inferEntityType(token: string, metadata: JsonRecord): string {
        const lower = token.toLowerCase();
        const metadataText = JSON.stringify(metadata).toLowerCase();

        if (/(team|company|corp|org|organization|group|lab|studio|school|university)/.test(lower)) {
            return "ORG";
        }
        if (/(skill|ability|tool|framework|library|method|pattern|algorithm|concept|idea)/.test(lower)) {
            return "CONCEPT";
        }
        if (/(person|people|name|user|member|author|designer|engineer)/.test(lower) || /[A-Z][a-z]+/.test(token)) {
            return "PERSON";
        }
        if (/(product|service|app|platform|feature|project)/.test(lower) || metadataText.includes("product")) {
            return "PRODUCT";
        }

        return "MISC";
    }

    private addEntityToGraph(entity: Entity, memoryItem: MemoryItem): void {
        try {
            this.graphStore.addEntity(
                entity.entityId,
                entity.name,
                entity.entityType,
                {
                    name: entity.name,
                    description: entity.description,
                    frequency: entity.frequency,
                    memory_id: memoryItem.id,
                    user_id: memoryItem.userId,
                    importance: memoryItem.importance,
                    ...entity.properties,
                },
            ).catch(() => undefined);

            this.addOrUpdateEntity(entity);
        } catch {
            // 图数据库失败不影响主记忆落盘
        }
    }

    private addRelationToGraph(relation: Relation, memoryItem: MemoryItem): void {
        try {
            this.graphStore.addRelationship(
                relation.fromEntity,
                relation.toEntity,
                relation.relationType,
                {
                    strength: relation.strength,
                    memory_id: memoryItem.id,
                    user_id: memoryItem.userId,
                    importance: memoryItem.importance,
                    evidence: relation.evidence,
                    ...relation.properties,
                },
            ).catch(() => undefined);

            this.addOrUpdateRelation(relation);
        } catch {
            // 图数据库失败不影响主记忆落盘
        }
    }

    private addOrUpdateEntity(entity: Entity): void {
        const existing = this.entities.get(entity.entityId);
        if (existing) {
            existing.frequency += 1;
            existing.updatedAt = new Date();
            existing.properties = {
                ...existing.properties,
                ...entity.properties,
            };
            return;
        }

        this.entities.set(entity.entityId, entity);
        this.entityUsage.set(entity.entityId, (this.entityUsage.get(entity.entityId) ?? 0) + 1);
    }

    private addOrUpdateRelation(relation: Relation): void {
        const existing = this.relations.find((item) =>
            item.fromEntity === relation.fromEntity
            && item.toEntity === relation.toEntity
            && item.relationType === relation.relationType,
        );

        if (existing) {
            existing.frequency += 1;
            existing.strength = Math.min(1, existing.strength + 0.1);
            existing.evidence = relation.evidence || existing.evidence;
            existing.properties = {
                ...existing.properties,
                ...relation.properties,
            };
            return;
        }

        this.relations.push(relation);
        const key = this.relationKey(relation.fromEntity, relation.toEntity, relation.relationType);
        this.relationUsage.set(key, (this.relationUsage.get(key) ?? 0) + 1);
    }

    private cleanupEntitiesAndRelations(entityIds: string[], excludeMemoryId: string | null = null): void {
        if (!entityIds.length) {
            return;
        }

        const referenced = new Map<string, number>();
        for (const ids of this.memoryEntityIndex.values()) {
            if (excludeMemoryId && ids === this.memoryEntityIndex.get(excludeMemoryId)) {
                continue;
            }
            for (const id of ids) {
                referenced.set(id, (referenced.get(id) ?? 0) + 1);
            }
        }

        for (const entityId of entityIds) {
            const count = referenced.get(entityId) ?? 0;
            if (count <= 1) {
                this.entities.delete(entityId);
                this.entityUsage.delete(entityId);
                this.memoryEntityIndex.delete(entityId);
            } else {
                this.entityUsage.set(entityId, count - 1);
            }
        }

        this.relations = this.relations.filter((relation) => this.entities.has(relation.fromEntity) && this.entities.has(relation.toEntity));
    }

    private vectorSearch(query: string, limit: number, userId: string | null): Array<{ memory_id: string; content: string; user_id: string; importance: number; timestamp: number; vector_score: number; metadata: JsonRecord }> {
        const queryEmbedding = this.embeddingModel.encode(query);
        const scored: Array<{ memory_id: string; content: string; user_id: string; importance: number; timestamp: number; vector_score: number; metadata: JsonRecord }> = [];

        for (const memory of this.semanticMemories) {
            if (userId && memory.userId !== userId) {
                continue;
            }

            const memoryEmbedding = this.memoryEmbeddings.get(memory.id);
            if (!memoryEmbedding) {
                continue;
            }

            const score = this.cosineSimilarity(queryEmbedding, memoryEmbedding);
            if (score <= 0) {
                continue;
            }

            scored.push({
                memory_id: memory.id,
                content: memory.content,
                user_id: memory.userId ?? "default_user",
                importance: memory.importance,
                timestamp: asDate(memory.timestamp).getTime(),
                vector_score: score,
                metadata: cloneRecord(memory.metadata),
            });
        }

        return scored.sort((left, right) => right.vector_score - left.vector_score).slice(0, Math.max(0, Math.floor(limit)));
    }

    private graphSearch(query: string, limit: number, userId: string | null): Array<{ memory_id: string; content: string; user_id: string; importance: number; timestamp: number; graph_score: number; metadata: JsonRecord }> {
        const queryEntities = this.extractEntities(query, {});
        const relatedMemoryIds = new Set<string>();

        if (queryEntities.length === 0) {
            const entitiesByName = this.searchEntities(query, 5).slice(0, 3);
            queryEntities.push(...entitiesByName);
        }

        for (const entity of queryEntities) {
            const related = this.getRelatedEntities(entity.entityId, null, 2);
            for (const item of related) {
                const entityObj = item.entity;
                if (!entityObj) {
                    continue;
                }

                for (const memory of this.semanticMemories) {
                    const entityIds = memory.metadata?.entities;
                    if (Array.isArray(entityIds) && entityIds.includes(entityObj.entityId)) {
                        relatedMemoryIds.add(memory.id);
                    }
                }
            }
        }

        const results: Array<{ memory_id: string; content: string; user_id: string; importance: number; timestamp: number; graph_score: number; metadata: JsonRecord }> = [];
        for (const memoryId of relatedMemoryIds) {
            const memory = this.semanticMemories.find((item) => item.id === memoryId);
            if (!memory) {
                continue;
            }

            if (userId && memory.userId !== userId) {
                continue;
            }

            const metadata = cloneRecord(memory.metadata);
            const graphScore = this.calculateGraphRelevance(metadata, queryEntities);
            results.push({
                memory_id: memory.id,
                content: memory.content,
                user_id: memory.userId ?? "default_user",
                importance: memory.importance,
                timestamp: asDate(memory.timestamp).getTime(),
                graph_score: graphScore,
                metadata,
            });
        }

        return results.sort((left, right) => right.graph_score - left.graph_score).slice(0, Math.max(0, Math.floor(limit)));
    }

    private combineAndRankResults(
        vectorResults: Array<{ memory_id: string; content: string; user_id: string; importance: number; timestamp: number; vector_score: number; metadata: JsonRecord }>,
        graphResults: Array<{ memory_id: string; content: string; user_id: string; importance: number; timestamp: number; graph_score: number; metadata: JsonRecord }>,
        query: string,
        limit: number,
    ): Array<{ memory_id: string; content: string; user_id: string; importance: number; timestamp: number; vector_score: number; graph_score: number; combined_score: number; metadata: JsonRecord }> {
        const combined = new Map<string, { memory_id: string; content: string; user_id: string; importance: number; timestamp: number; vector_score: number; graph_score: number; combined_score: number; metadata: JsonRecord }>();
        const contentSeen = new Set<string>();
        const normalizedQuery = query.trim().toLowerCase();

        for (const result of vectorResults) {
            const contentHash = this.hashString(result.content.trim().toLowerCase());
            if (contentSeen.has(contentHash)) {
                continue;
            }

            contentSeen.add(contentHash);
            combined.set(result.memory_id, {
                memory_id: result.memory_id,
                content: result.content,
                user_id: result.user_id,
                importance: result.importance,
                timestamp: result.timestamp,
                vector_score: result.vector_score,
                graph_score: 0,
                combined_score: 0,
                metadata: cloneRecord(result.metadata),
            });
        }

        for (const result of graphResults) {
            const contentHash = this.hashString(result.content.trim().toLowerCase());
            const existing = combined.get(result.memory_id);
            if (existing) {
                existing.graph_score = result.graph_score;
                existing.metadata = {
                    ...existing.metadata,
                    ...result.metadata,
                };
                continue;
            }

            if (contentSeen.has(contentHash)) {
                continue;
            }

            contentSeen.add(contentHash);
            combined.set(result.memory_id, {
                memory_id: result.memory_id,
                content: result.content,
                user_id: result.user_id,
                importance: result.importance,
                timestamp: result.timestamp,
                vector_score: 0,
                graph_score: result.graph_score,
                combined_score: 0,
                metadata: cloneRecord(result.metadata),
            });
        }

        const ranked = [...combined.values()].map((result) => {
            const exactMatchBonus = normalizedQuery && result.content.toLowerCase().includes(normalizedQuery) ? 0.2 : 0;
            const keywordBonus = this.keywordBonus(normalizedQuery, result.content, result.metadata);
            const entityTypeBonus = Array.isArray(result.metadata.entities) ? Math.min(0.2, result.metadata.entities.length * 0.05) : 0;
            const companyBonus = typeof result.metadata.entity_type === "string" && /ORG|COMPANY|ORGANIZATION/.test(String(result.metadata.entity_type).toUpperCase()) ? 0.1 : 0;

            const baseRelevance = result.vector_score * 0.7 + result.graph_score * 0.3 + exactMatchBonus + keywordBonus + entityTypeBonus + companyBonus;
            const importanceWeight = 0.8 + (this.clampImportance(result.importance) * 0.4);
            const combinedScore = baseRelevance * importanceWeight;

            return {
                ...result,
                combined_score: combinedScore,
                metadata: {
                    ...result.metadata,
                    debug_info: {
                        base_relevance: baseRelevance,
                        importance_weight: importanceWeight,
                        combined_score: combinedScore,
                    },
                    exact_match_bonus: exactMatchBonus,
                    keyword_bonus: keywordBonus,
                    company_bonus: companyBonus,
                    entity_type_bonus: entityTypeBonus,
                },
            };
        }).filter((result) => result.combined_score >= 0.1);

        ranked.sort((left, right) => right.combined_score - left.combined_score);
        return ranked.slice(0, Math.max(0, Math.floor(limit)));
    }

    private extractEntities(text: string, metadata: JsonRecord): Entity[] {
        const entities: Entity[] = [];
        const tokens = tokenize(text);
        const seen = new Set<string>();

        for (const token of tokens) {
            const normalized = token.replace(/[^\p{L}\p{N}_-]/gu, "").trim();
            if (!normalized || normalized.length < 2) {
                continue;
            }

            const key = normalized.toLowerCase();
            if (seen.has(key)) {
                continue;
            }

            seen.add(key);
            const entityType = this.inferEntityType(normalized, metadata);
            const entityId = `entity_${this.hashString(`${normalized}:${entityType}`)}`;

            entities.push(new Entity(entityId, normalized, entityType, `从文本中识别的${entityType}实体`, {
                source: "semantic_memory",
                mentions: 1,
            }));
        }

        return entities;
    }

    private extractRelations(text: string, entities: Entity[]): Relation[] {
        const relations: Relation[] = [];
        for (let i = 0; i < entities.length; i += 1) {
            for (let j = i + 1; j < entities.length; j += 1) {
                relations.push(new Relation(
                    entities[i].entityId,
                    entities[j].entityId,
                    "CO_OCCURS",
                    0.5,
                    text.slice(0, 120),
                    { evidence_type: "co_occurrence" },
                ));
            }
        }

        return relations;
    }

    private inferEntityType(token: string, metadata: JsonRecord): string {
        const lower = token.toLowerCase();
        const metadataText = JSON.stringify(metadata).toLowerCase();

        if (/(team|company|corp|org|organization|group|lab|studio|school|university)/.test(lower)) {
            return "ORG";
        }
        if (/(skill|ability|tool|framework|library|method|pattern|algorithm|concept|idea)/.test(lower)) {
            return "CONCEPT";
        }
        if (/(person|people|name|user|member|author|designer|engineer)/.test(lower) || /[A-Z][a-z]+/.test(token)) {
            return "PERSON";
        }
        if (/(product|service|app|platform|feature|project)/.test(lower) || metadataText.includes("product")) {
            return "PRODUCT";
        }

        return "MISC";
    }

    private calculateGraphRelevance(memoryMetadata: JsonRecord, queryEntities: Entity[]): number {
        const memoryEntities = Array.isArray(memoryMetadata.entities) ? memoryMetadata.entities : [];
        if (memoryEntities.length === 0 || queryEntities.length === 0) {
            return 0;
        }

        const queryEntityIds = new Set(queryEntities.map((entity) => entity.entityId));
        const matching = memoryEntities.filter((entityId) => queryEntityIds.has(String(entityId))).length;
        const entityScore = queryEntityIds.size > 0 ? matching / queryEntityIds.size : 0;
        const entityDensity = Math.min(Number(memoryMetadata.entity_count ?? 0) / 10, 1);
        const relationDensity = Math.min(Number(memoryMetadata.relation_count ?? 0) / 5, 1);

        return Math.min(entityScore * 0.6 + entityDensity * 0.2 + relationDensity * 0.2, 1);
    }

    private addOrUpdateEntity(entity: Entity): void {
        const existing = this.entities.get(entity.entityId);
        if (existing) {
            existing.frequency += 1;
            existing.updatedAt = new Date();
            existing.properties = {
                ...existing.properties,
                ...entity.properties,
            };
            return;
        }

        this.entities.set(entity.entityId, entity);
    }

    private addOrUpdateRelation(relation: Relation): void {
        const existing = this.relations.find((item) =>
            item.fromEntity === relation.fromEntity
            && item.toEntity === relation.toEntity
            && item.relationType === relation.relationType,
        );

        if (existing) {
            existing.frequency += 1;
            existing.strength = Math.min(1, existing.strength + 0.1);
            existing.evidence = relation.evidence || existing.evidence;
            existing.properties = {
                ...existing.properties,
                ...relation.properties,
            };
            return;
        }

        this.relations.push(relation);
    }

    private safeGraphStats(): JsonRecord {
        return {
            total_nodes: this.entities.size,
            entity_nodes: this.entities.size,
            memory_nodes: this.semanticMemories.length,
            total_relationships: this.relations.length,
        };
    }

    private keywordBonus(query: string, content: string, metadata: JsonRecord): number {
        if (!query) {
            return 0;
        }

        const contentLower = content.toLowerCase();
        const metadataText = JSON.stringify(metadata).toLowerCase();
        let score = 0;

        if (contentLower.includes(query)) {
            score += 0.3;
        }

        for (const token of tokenize(query)) {
            if (contentLower.includes(token)) {
                score += 0.1;
            }
            if (metadataText.includes(token)) {
                score += 0.05;
            }
        }

        return Math.min(score, 0.3);
    }

    private softmax(values: number[]): number[] {
        if (values.length === 0) {
            return [];
        }

        const maxValue = Math.max(...values);
        const exps = values.map((value) => Math.exp(value - maxValue));
        const denom = exps.reduce((sum, value) => sum + value, 0) || 1;
        return exps.map((value) => value / denom);
    }

    private cosineSimilarity(left: number[], right: number[]): number {
        const size = Math.min(left.length, right.length);
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

    private clampImportance(value: number): number {
        if (!Number.isFinite(value)) {
            return 0.5;
        }

        return Math.max(0, Math.min(1, value));
    }

    private relationKey(fromEntity: string, toEntity: string, relationType: string): string {
        return `${fromEntity}:${toEntity}:${relationType}`;
    }

    private hashString(value: string): string {
        let hash = 0;
        for (let i = 0; i < value.length; i += 1) {
            hash = ((hash << 5) - hash + value.charCodeAt(i)) | 0;
        }
        return Math.abs(hash).toString(36);
    }

    private cloneMemoryItem(memory: MemoryItem): MemoryItem {
        return {
            ...memory,
            metadata: memory.metadata ? { ...memory.metadata } : undefined,
        };
    }
}
