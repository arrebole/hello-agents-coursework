// Neo4j图存储（知识图谱管理）

type Neo4jSession = {
    run: (query: string, params?: Record<string, unknown>) => Promise<{ single: () => Record<string, unknown> | null; records?: Array<{ toObject: () => Record<string, unknown> }>; consume: () => { counters: { nodesDeleted: number; relationshipsDeleted: number } } }>;
    close: () => Promise<void>;
};

type Neo4jDriver = {
    session: (options?: { database?: string }) => Neo4jSession;
    verifyConnectivity: () => Promise<void>;
    close: () => Promise<void>;
};

type Neo4jModule = {
    driver: (
        uri: string,
        auth: { username: string; password: string },
        config?: Record<string, unknown>,
    ) => Neo4jDriver;
    auth?: {
        basic: (username: string, password: string) => { username: string; password: string };
    };
};

type GraphRecord = Record<string, unknown>;

const DEFAULT_URI = "bolt://localhost:7687";
const DEFAULT_USERNAME = "neo4j";
const DEFAULT_PASSWORD = "hello-agents-password";
const DEFAULT_DATABASE = "neo4j";

function nowIso(): string {
    return new Date().toISOString();
}

function normalizeProps(properties?: GraphRecord): GraphRecord {
    return { ...(properties ?? {}) };
}

function sanitizeRelationshipType(relationshipType: string): string {
    const trimmed = relationshipType.trim();
    if (!trimmed) {
        throw new Error("relationship_type 不能为空");
    }

    if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(trimmed)) {
        throw new Error(`非法关系类型: ${relationshipType}`);
    }

    return trimmed;
}

export class Neo4jGraphStore {
    private readonly uri: string;
    private readonly username: string;
    private readonly password: string;
    private readonly database: string;
    private driver: Neo4jDriver | null = null;
    private initPromise: Promise<void> | null = null;

    constructor(
        uri: string = DEFAULT_URI,
        username: string = DEFAULT_USERNAME,
        password: string = DEFAULT_PASSWORD,
        database: string = DEFAULT_DATABASE,
        maxConnectionLifetime: number = 3600,
        maxConnectionPoolSize: number = 50,
        connectionAcquisitionTimeout: number = 60,
        ..._kwargs: unknown[]
    ) {
        this.uri = uri;
        this.username = username;
        this.password = password;
        this.database = database;

        void maxConnectionLifetime;
        void maxConnectionPoolSize;
        void connectionAcquisitionTimeout;

        this.initPromise = this._initializeDriver({
            maxConnectionLifetime,
            maxConnectionPoolSize,
            connectionAcquisitionTimeout,
        });
        this.initPromise.catch(() => undefined);
    }

    async initialize(): Promise<void> {
        await this.ensureInitialized();
    }

    private async loadNeo4jModule(): Promise<Neo4jModule> {
        try {
            return (await import("neo4j-driver")) as Neo4jModule;
        } catch (error) {
            throw new Error("neo4j-driver 未安装。请运行: npm install neo4j-driver");
        }
    }

    private async _initializeDriver(config: Record<string, unknown>): Promise<void> {
        try {
            const neo4j = await this.loadNeo4jModule();
            const auth = neo4j.auth?.basic
                ? neo4j.auth.basic(this.username, this.password)
                : { username: this.username, password: this.password };
            this.driver = neo4j.driver(
                this.uri,
                auth,
                config,
            );

            await this.driver.verifyConnectivity();
            if (this.uri.includes("neo4j.io") || this.uri.toLowerCase().includes("aura")) {
                console.info(`✅ 成功连接到Neo4j云服务: ${this.uri}`);
            } else {
                console.info(`✅ 成功连接到Neo4j服务: ${this.uri}`);
            }

            await this._createIndexes();
        } catch (error) {
            this.driver = null;
            throw error;
        }
    }

    private async ensureInitialized(): Promise<void> {
        if (this.driver) {
            return;
        }

        if (!this.initPromise) {
            this.initPromise = this._initializeDriver({});
            this.initPromise.catch(() => undefined);
        }

        await this.initPromise;
    }

    private async withSession<T>(operation: (session: Neo4jSession) => Promise<T>): Promise<T> {
        await this.ensureInitialized();

        if (!this.driver) {
            throw new Error("Neo4j driver 未初始化");
        }

        const session = this.driver.session({ database: this.database });
        try {
            return await operation(session);
        } finally {
            await session.close();
        }
    }

    private async _createIndexes(): Promise<void> {
        const indexes = [
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.id)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
            "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
        ];

        await this.withSession(async (session) => {
            for (const indexQuery of indexes) {
                try {
                    await session.run(indexQuery);
                } catch (error) {
                    console.debug(`索引创建跳过 (可能已存在): ${String(error)}`);
                }
            }
        });

        console.info("✅ Neo4j索引创建完成");
    }

    async addEntity(entityId: string, name: string, entityType: string, properties: GraphRecord = {}): Promise<boolean> {
        try {
            const props = normalizeProps(properties);
            const mergedProps = {
                ...props,
                id: entityId,
                name,
                type: entityType,
                created_at: nowIso(),
                updated_at: nowIso(),
            };

            const query = `
            MERGE (e:Entity {id: $entity_id})
            SET e += $properties
            RETURN e
            `;

            const record = await this.withSession(async (session) => {
                const result = await session.run(query, { entity_id: entityId, properties: mergedProps });
                return result.single();
            });

            if (record) {
                console.debug(`✅ 添加实体: ${name} (${entityType})`);
                return true;
            }
            return false;
        } catch (error) {
            console.error(`❌ 添加实体失败: ${String(error)}`);
            return false;
        }
    }

    async addRelationship(
        fromEntityId: string,
        toEntityId: string,
        relationshipType: string,
        properties: GraphRecord = {},
    ): Promise<boolean> {
        try {
            const safeType = sanitizeRelationshipType(relationshipType);
            const mergedProps = {
                ...normalizeProps(properties),
                type: safeType,
                created_at: nowIso(),
                updated_at: nowIso(),
            };

            const query = `
            MATCH (from:Entity {id: $from_id})
            MATCH (to:Entity {id: $to_id})
            MERGE (from)-[r:${safeType}]->(to)
            SET r += $properties
            RETURN r
            `;

            const record = await this.withSession(async (session) => {
                const result = await session.run(query, {
                    from_id: fromEntityId,
                    to_id: toEntityId,
                    properties: mergedProps,
                });
                return result.single();
            });

            if (record) {
                console.debug(`✅ 添加关系: ${fromEntityId} -${safeType}-> ${toEntityId}`);
                return true;
            }
            return false;
        } catch (error) {
            console.error(`❌ 添加关系失败: ${String(error)}`);
            return false;
        }
    }

    async findRelatedEntities(
        entityId: string,
        relationshipTypes: string[] | null = null,
        maxDepth: number = 2,
        limit: number = 50,
    ): Promise<GraphRecord[]> {
        try {
            const relFilter = relationshipTypes && relationshipTypes.length > 0
                ? `:${relationshipTypes.map(sanitizeRelationshipType).join("|")}`
                : "";

            const query = `
            MATCH path = (start:Entity {id: $entity_id})-[r${relFilter}*1..${Math.max(1, maxDepth)}]-(related:Entity)
            WHERE start.id <> related.id
            RETURN DISTINCT related,
                   length(path) as distance,
                   [rel in relationships(path) | type(rel)] as relationship_path
            ORDER BY distance, related.name
            LIMIT $limit
            `;

            return await this.withSession(async (session) => {
                const result = await session.run(query, { entity_id: entityId, limit });
                const entities: GraphRecord[] = [];
                const records = result.records ?? [];

                for (const record of records) {
                    const related = record.toObject().related as GraphRecord | undefined;
                    const entityData = { ...(related ?? {}) };
                    entityData.distance = record.toObject().distance;
                    entityData.relationship_path = record.toObject().relationship_path;
                    entities.push(entityData);
                }

                console.debug(`🔍 找到 ${entities.length} 个相关实体`);
                return entities;
            });
        } catch (error) {
            console.error(`❌ 查找相关实体失败: ${String(error)}`);
            return [];
        }
    }

    async searchEntitiesByName(
        namePattern: string,
        entityTypes: string[] | null = null,
        limit: number = 20,
    ): Promise<GraphRecord[]> {
        try {
            const params: Record<string, unknown> = {
                pattern: `.*${namePattern}.*`,
                limit,
            };

            const typeFilter = entityTypes && entityTypes.length > 0
                ? "AND e.type IN $types"
                : "";

            if (entityTypes && entityTypes.length > 0) {
                params.types = entityTypes;
            }

            const query = `
            MATCH (e:Entity)
            WHERE e.name =~ $pattern ${typeFilter}
            RETURN e
            ORDER BY e.name
            LIMIT $limit
            `;

            return await this.withSession(async (session) => {
                const result = await session.run(query, params);
                const entities: GraphRecord[] = [];
                const records = result.records ?? [];

                for (const record of records) {
                    const data = record.toObject().e as GraphRecord | undefined;
                    entities.push({ ...(data ?? {}) });
                }

                console.debug(`🔍 按名称搜索到 ${entities.length} 个实体`);
                return entities;
            });
        } catch (error) {
            console.error(`❌ 按名称搜索实体失败: ${String(error)}`);
            return [];
        }
    }

    async getEntityRelationships(entityId: string): Promise<GraphRecord[]> {
        try {
            const query = `
            MATCH (e:Entity {id: $entity_id})-[r]-(other:Entity)
            RETURN r, other,
                   CASE WHEN startNode(r).id = $entity_id THEN 'outgoing' ELSE 'incoming' END as direction
            `;

            return await this.withSession(async (session) => {
                const result = await session.run(query, { entity_id: entityId });
                const relationships: GraphRecord[] = [];
                const records = result.records ?? [];

                for (const record of records) {
                    const row = record.toObject();
                    relationships.push({
                        relationship: { ...(row.r as GraphRecord | undefined ?? {}) },
                        other_entity: { ...(row.other as GraphRecord | undefined ?? {}) },
                        direction: row.direction,
                    });
                }

                return relationships;
            });
        } catch (error) {
            console.error(`❌ 获取实体关系失败: ${String(error)}`);
            return [];
        }
    }

    async deleteEntity(entityId: string): Promise<boolean> {
        try {
            const query = `
            MATCH (e:Entity {id: $entity_id})
            DETACH DELETE e
            `;

            return await this.withSession(async (session) => {
                const result = await session.run(query, { entity_id: entityId });
                const summary = result.consume();
                const deletedCount = summary.counters.nodesDeleted;
                console.info(`✅ 删除实体: ${entityId} (删除 ${deletedCount} 个节点)`);
                return deletedCount > 0;
            });
        } catch (error) {
            console.error(`❌ 删除实体失败: ${String(error)}`);
            return false;
        }
    }

    async clearAll(): Promise<boolean> {
        try {
            const query = "MATCH (n) DETACH DELETE n";

            return await this.withSession(async (session) => {
                const result = await session.run(query);
                const summary = result.consume();
                console.info(`✅ 清空Neo4j数据库: 删除 ${summary.counters.nodesDeleted} 个节点, ${summary.counters.relationshipsDeleted} 个关系`);
                return true;
            });
        } catch (error) {
            console.error(`❌ 清空数据库失败: ${String(error)}`);
            return false;
        }
    }

    async getStats(): Promise<Record<string, number>> {
        try {
            const queries = {
                total_nodes: "MATCH (n) RETURN count(n) as count",
                total_relationships: "MATCH ()-[r]->() RETURN count(r) as count",
                entity_nodes: "MATCH (n:Entity) RETURN count(n) as count",
                memory_nodes: "MATCH (n:Memory) RETURN count(n) as count",
            };

            return await this.withSession(async (session) => {
                const stats: Record<string, number> = {};

                for (const [key, query] of Object.entries(queries)) {
                    const result = await session.run(query);
                    const record = result.single();
                    stats[key] = Number(record?.count ?? 0);
                }

                return stats;
            });
        } catch (error) {
            console.error(`❌ 获取统计信息失败: ${String(error)}`);
            return {};
        }
    }

    async healthCheck(): Promise<boolean> {
        try {
            return await this.withSession(async (session) => {
                const result = await session.run("RETURN 1 as health");
                const record = result.single();
                return Number(record?.health ?? 0) === 1;
            });
        } catch (error) {
            console.error(`❌ Neo4j健康检查失败: ${String(error)}`);
            return false;
        }
    }

    async close(): Promise<void> {
        if (this.driver) {
            await this.driver.close();
            this.driver = null;
        }
    }

    async __del__(): Promise<void> {
        await this.close();
    }
}
