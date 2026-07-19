// 统一嵌入模块（实现 + Provider）
//
// 支持：
// - DashScope/OpenAI 兼容 REST embedding
// - 本地 Transformer（可选依赖 @xenova/transformers 或 @huggingface/transformers）
// - 纯 TypeScript TF-IDF 兜底
//
// 环境变量：
// - EMBED_MODEL_TYPE: "dashscope" | "local" | "tfidf"（默认 dashscope）
// - EMBED_MODEL_NAME: 模型名称（dashscope 默认 text-embedding-v3；local 默认 sentence-transformers/all-MiniLM-L6-v2）
// - EMBED_API_KEY: Embedding API Key
// - EMBED_BASE_URL: Embedding Base URL（可选，按 OpenAI 兼容 /embeddings 调用）
// - EMBED_DIMENSION: 统一向量维度（默认 384）

export const DEFAULT_EMBEDDING_DIMENSION = readPositiveInteger(
    process.env.EMBED_DIMENSION,
    384,
);

export type EmbeddingInput = string | string[];
export type EmbeddingVector = number[];

interface EmbeddingModelOptions {
    modelName?: string;
    apiKey?: string;
    baseUrl?: string;
    dimension?: number;
    maxFeatures?: number;
}

export abstract class EmbeddingModel {
    abstract encode(texts: string): Promise<EmbeddingVector>;
    abstract encode(texts: string[]): Promise<EmbeddingVector[]>;
    abstract encode(texts: EmbeddingInput): Promise<EmbeddingVector | EmbeddingVector[]>;
    abstract get dimension(): number;
}

type LocalBackend = {
    name: string;
    extractor: (inputs: string[]) => Promise<EmbeddingVector[]>;
};

export class LocalTransformerEmbedding extends EmbeddingModel {
    readonly modelName: string;
    private backend: LocalBackend | null = null;
    private backendPromise: Promise<LocalBackend> | null = null;
    private resolvedDimension: number;

    constructor(modelName: string = "sentence-transformers/all-MiniLM-L6-v2", dimension: number = DEFAULT_EMBEDDING_DIMENSION) {
        super();
        this.modelName = modelName;
        this.resolvedDimension = dimension;
    }

    async encode(texts: string): Promise<EmbeddingVector>;
    async encode(texts: string[]): Promise<EmbeddingVector[]>;
    async encode(texts: EmbeddingInput): Promise<EmbeddingVector | EmbeddingVector[]> {
        const { inputs, single } = normalizeInputs(texts);
        const backend = await this.loadBackend();
        const vectors = await backend.extractor(inputs);
        const normalized = vectors.map((vector) => resizeAndNormalizeVector(vector, this.resolvedDimension));
        return single ? normalized[0] : normalized;
    }

    get dimension(): number {
        return this.resolvedDimension;
    }

    private async loadBackend(): Promise<LocalBackend> {
        if (this.backend) return this.backend;
        if (this.backendPromise) return this.backendPromise;

        this.backendPromise = this.createBackend().then((backend) => {
            this.backend = backend;
            return backend;
        });
        return this.backendPromise;
    }

    private async createBackend(): Promise<LocalBackend> {
        const candidates = ["@xenova/transformers", "@huggingface/transformers"];

        for (const packageName of candidates) {
            try {
                const mod = await optionalImport(packageName);
                const pipeline = mod?.pipeline;
                if (typeof pipeline !== "function") continue;

                const extractor = await pipeline("feature-extraction", this.modelName);
                return {
                    name: packageName,
                    extractor: async (inputs: string[]) => {
                        const output = await extractor(inputs, {
                            pooling: "mean",
                            normalize: true,
                        });
                        return coerceTransformerOutput(output);
                    },
                };
            } catch {
                continue;
            }
        }

        throw new Error("未找到可用的本地嵌入后端，请安装 @xenova/transformers 或 @huggingface/transformers");
    }
}

export class TFIDFEmbedding extends EmbeddingModel {
    readonly maxFeatures: number;
    private vocabulary = new Map<string, number>();
    private idf: number[] = [];
    private fitted = false;

    constructor(maxFeatures: number = 1000) {
        super();
        this.maxFeatures = Math.max(1, Math.floor(maxFeatures));
    }

    fit(texts: string[]): void {
        const documents = texts.map((text) => [...new Set(tokenize(text))]).filter((tokens) => tokens.length > 0);
        const documentFrequency = new Map<string, number>();

        for (const tokens of documents) {
            for (const token of tokens) {
                documentFrequency.set(token, (documentFrequency.get(token) ?? 0) + 1);
            }
        }

        const selectedTokens = [...documentFrequency.entries()]
            .sort((a, b) => b[1] - a[1] || a[0].localeCompare(b[0]))
            .slice(0, this.maxFeatures)
            .map(([token]) => token);

        this.vocabulary = new Map(selectedTokens.map((token, index) => [token, index]));
        const documentCount = Math.max(1, documents.length);
        this.idf = selectedTokens.map((token) => {
            const df = documentFrequency.get(token) ?? 0;
            return Math.log((1 + documentCount) / (1 + df)) + 1;
        });
        this.fitted = true;
    }

    async encode(texts: string): Promise<EmbeddingVector>;
    async encode(texts: string[]): Promise<EmbeddingVector[]>;
    async encode(texts: EmbeddingInput): Promise<EmbeddingVector | EmbeddingVector[]> {
        const { inputs, single } = normalizeInputs(texts);
        if (!this.fitted) {
            this.fit(inputs);
        }

        const vectors = inputs.map((text) => this.transform(text));
        return single ? vectors[0] : vectors;
    }

    get dimension(): number {
        return this.maxFeatures;
    }

    private transform(text: string): EmbeddingVector {
        const vector = new Array<number>(this.maxFeatures).fill(0);
        const tokens = tokenize(text);
        if (!tokens.length || !this.vocabulary.size) return vector;

        for (const token of tokens) {
            const index = this.vocabulary.get(token);
            if (index === undefined) continue;
            vector[index] += 1;
        }

        for (let i = 0; i < vector.length; i += 1) {
            vector[i] *= this.idf[i] ?? 1;
        }

        return normalizeVector(vector);
    }
}

export class DashScopeEmbedding extends EmbeddingModel {
    readonly modelName: string;
    readonly apiKey?: string;
    readonly baseUrl?: string;
    private resolvedDimension: number;

    constructor(
        modelName: string = "text-embedding-v3",
        apiKey?: string,
        baseUrl?: string,
        dimension: number = DEFAULT_EMBEDDING_DIMENSION,
    ) {
        super();
        this.modelName = modelName;
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
        this.resolvedDimension = dimension;

        if (!this.apiKey) {
            throw new Error("DashScope Embedding 需要 EMBED_API_KEY");
        }
    }

    async encode(texts: string): Promise<EmbeddingVector>;
    async encode(texts: string[]): Promise<EmbeddingVector[]>;
    async encode(texts: EmbeddingInput): Promise<EmbeddingVector | EmbeddingVector[]> {
        const { inputs, single } = normalizeInputs(texts);
        const url = `${(this.baseUrl ?? "https://dashscope.aliyuncs.com/compatible-mode/v1").replace(/\/$/, "")}/embeddings`;
        const response = await fetch(url, {
            method: "POST",
            headers: {
                authorization: `Bearer ${this.apiKey}`,
                "content-type": "application/json",
            },
            body: JSON.stringify({
                model: this.modelName,
                input: inputs,
                dimensions: this.resolvedDimension,
            }),
        });

        if (!response.ok) {
            const body = await response.text();
            throw new Error(`Embedding REST 调用失败: ${response.status} ${body}`);
        }

        const data = await response.json() as { data?: Array<{ embedding?: unknown; index?: number }> };
        const items = data.data ?? [];
        if (!items.length) {
            throw new Error("DashScope 返回为空或格式不匹配");
        }

        const vectors = items
            .slice()
            .sort((a, b) => (a.index ?? 0) - (b.index ?? 0))
            .map((item) => resizeAndNormalizeVector(coerceVector(item.embedding), this.resolvedDimension));

        return single ? vectors[0] : vectors;
    }

    get dimension(): number {
        return this.resolvedDimension;
    }
}

export async function createEmbeddingModel(
    modelType: string = "local",
    options: EmbeddingModelOptions = {},
): Promise<EmbeddingModel> {
    const normalizedType = normalizeModelType(modelType);
    const dimension = options.dimension ?? DEFAULT_EMBEDDING_DIMENSION;

    if (normalizedType === "local") {
        const modelName = options.modelName ?? "sentence-transformers/all-MiniLM-L6-v2";
        const model = new LocalTransformerEmbedding(modelName, dimension);
        await model.encode("health_check");
        return model;
    }

    if (normalizedType === "dashscope") {
        const modelName = options.modelName ?? "text-embedding-v3";
        const model = new DashScopeEmbedding(modelName, options.apiKey, options.baseUrl, dimension);
        await model.encode("health_check");
        return model;
    }

    if (normalizedType === "tfidf") {
        return new TFIDFEmbedding(options.maxFeatures ?? dimension);
    }

    throw new Error(`不支持的模型类型: ${modelType}`);
}

export async function createEmbeddingModelWithFallback(
    preferredType: string = "dashscope",
    options: EmbeddingModelOptions = {},
): Promise<EmbeddingModel> {
    const preferred = normalizeModelType(preferredType);
    const fallback = ["dashscope", "local", "tfidf"];
    const orderedTypes = fallback.includes(preferred)
        ? [preferred, ...fallback.filter((type) => type !== preferred)]
        : fallback;

    let lastError: unknown = null;
    for (const modelType of orderedTypes) {
        try {
            return await createEmbeddingModel(modelType, providerOptions(modelType, options));
        } catch (error) {
            lastError = error;
        }
    }

    throw new Error(`所有嵌入模型都不可用，请安装依赖或检查配置: ${errorMessage(lastError)}`);
}

let embedder: EmbeddingModel | null = null;
let pendingEmbedder: Promise<EmbeddingModel> | null = null;

async function buildEmbedder(): Promise<EmbeddingModel> {
    const preferred = (process.env.EMBED_MODEL_TYPE ?? "dashscope").trim() || "dashscope";
    const explicitModelName = process.env.EMBED_MODEL_NAME?.trim();
    const apiKey = process.env.EMBED_API_KEY;
    const baseUrl = process.env.EMBED_BASE_URL;
    const dimension = readPositiveInteger(process.env.EMBED_DIMENSION, DEFAULT_EMBEDDING_DIMENSION);

    return createEmbeddingModelWithFallback(preferred, {
        modelName: explicitModelName || undefined,
        apiKey,
        baseUrl,
        dimension,
        maxFeatures: dimension,
    });
}

export async function getTextEmbedder(): Promise<EmbeddingModel> {
    if (embedder) return embedder;
    if (pendingEmbedder) return pendingEmbedder;

    pendingEmbedder = buildEmbedder().then((created) => {
        embedder = created;
        pendingEmbedder = null;
        return created;
    }).catch((error) => {
        pendingEmbedder = null;
        throw error;
    });

    return pendingEmbedder;
}

export async function getDimension(defaultDimension: number = DEFAULT_EMBEDDING_DIMENSION): Promise<number> {
    try {
        const model = await getTextEmbedder();
        return Number.isFinite(model.dimension) && model.dimension > 0
            ? model.dimension
            : defaultDimension;
    } catch {
        return defaultDimension;
    }
}

export async function refreshEmbedder(): Promise<EmbeddingModel> {
    embedder = null;
    pendingEmbedder = null;
    return getTextEmbedder();
}

export async function getTextEmbedding(
    text: string,
    dimension: number = DEFAULT_EMBEDDING_DIMENSION,
): Promise<EmbeddingVector> {
    try {
        const model = await getTextEmbedder();
        const vector = await model.encode(text);
        return resizeAndNormalizeVector(vector, dimension);
    } catch {
        return getHashingEmbedding(text, dimension);
    }
}

export async function getTextEmbeddings(
    texts: string[],
    dimension: number = DEFAULT_EMBEDDING_DIMENSION,
): Promise<EmbeddingVector[]> {
    try {
        const model = await getTextEmbedder();

        if (model instanceof TFIDFEmbedding) {
            model.fit(texts);
        }

        const vectors = await model.encode(texts);
        return vectors.map((vector) => resizeAndNormalizeVector(vector, dimension));
    } catch {
        return texts.map((text) => getHashingEmbedding(text, dimension));
    }
}

// Python 风格别名，方便课程示例迁移。
export const create_embedding_model = createEmbeddingModel;
export const create_embedding_model_with_fallback = createEmbeddingModelWithFallback;
export const get_text_embedder = getTextEmbedder;
export const get_dimension = getDimension;
export const refresh_embedder = refreshEmbedder;
export const get_text_embedding = getTextEmbedding;
export const get_text_embeddings = getTextEmbeddings;

function providerOptions(modelType: string, options: EmbeddingModelOptions): EmbeddingModelOptions {
    const result = { ...options };
    if (!result.modelName) {
        result.modelName = modelType === "dashscope"
            ? "text-embedding-v3"
            : "sentence-transformers/all-MiniLM-L6-v2";
    }
    return result;
}

function normalizeModelType(modelType: string): string {
    const normalized = modelType.trim().toLowerCase();
    if (normalized === "sentence_transformer" || normalized === "huggingface") return "local";
    return normalized;
}

function normalizeInputs(texts: EmbeddingInput): { inputs: string[]; single: boolean } {
    if (typeof texts === "string") {
        return { inputs: [texts], single: true };
    }
    return { inputs: [...texts], single: false };
}

function tokenize(text: string): string[] {
    const normalized = text.trim().toLowerCase();
    if (!normalized) return [];

    const rawTokens = normalized.match(/[\p{L}\p{N}_]+/gu) ?? [];
    const tokens: string[] = [];

    for (const token of rawTokens) {
        if (ENGLISH_STOP_WORDS.has(token)) continue;
        tokens.push(token);

        if (/[\p{Script=Han}\p{Script=Hiragana}\p{Script=Katakana}]/u.test(token)) {
            const chars = Array.from(token);
            for (let i = 0; i < chars.length - 1; i += 1) {
                tokens.push(`${chars[i]}${chars[i + 1]}`);
            }
        }
    }

    return tokens;
}

function getHashingEmbedding(text: string, dimension: number): EmbeddingVector {
    const vectorSize = Math.max(1, Math.floor(dimension));
    const vector = new Array<number>(vectorSize).fill(0);
    const tokens = tokenize(text);

    for (const token of tokens) {
        vector[hashToken(token) % vectorSize] += 1;
    }

    return normalizeVector(vector);
}

function hashToken(token: string): number {
    let hash = 0;
    for (let i = 0; i < token.length; i += 1) {
        hash = ((hash << 5) - hash + token.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
}

function normalizeVector(vector: EmbeddingVector): EmbeddingVector {
    const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
    if (norm === 0) return vector;
    return vector.map((value) => value / norm);
}

function resizeAndNormalizeVector(vector: EmbeddingVector, dimension: number): EmbeddingVector {
    const targetDimension = Math.max(1, Math.floor(dimension));
    const resized = new Array<number>(targetDimension).fill(0);
    const length = Math.min(vector.length, targetDimension);
    for (let i = 0; i < length; i += 1) {
        resized[i] = Number.isFinite(vector[i]) ? vector[i] : 0;
    }
    return normalizeVector(resized);
}

function coerceVector(value: unknown): EmbeddingVector {
    if (!Array.isArray(value)) {
        throw new Error("Embedding 向量格式不正确");
    }
    return value.map((item) => Number(item)).filter((item) => Number.isFinite(item));
}

function coerceTransformerOutput(output: unknown): EmbeddingVector[] {
    if (output && typeof output === "object" && "tolist" in output && typeof output.tolist === "function") {
        return coerceNestedVectors(output.tolist());
    }

    return coerceNestedVectors(output);
}

function coerceNestedVectors(value: unknown): EmbeddingVector[] {
    if (!Array.isArray(value)) {
        throw new Error("本地 Transformer 返回格式不正确");
    }

    if (value.length > 0 && typeof value[0] === "number") {
        return [coerceVector(value)];
    }

    return value.map((item) => {
        if (Array.isArray(item) && item.length > 0 && Array.isArray(item[0])) {
            return coerceVector(item[0]);
        }
        return coerceVector(item);
    });
}

async function optionalImport(packageName: string): Promise<any> {
    const importer = new Function("packageName", "return import(packageName)") as (name: string) => Promise<any>;
    return importer(packageName);
}

function readPositiveInteger(value: string | undefined, fallback: number): number {
    const parsed = Number.parseInt(value ?? "", 10);
    return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback;
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}

const ENGLISH_STOP_WORDS = new Set([
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
    "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were",
    "will", "with",
]);
