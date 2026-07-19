// RAG 管道：通用文档加载 -> Markdown 化 -> 分块 -> 向量化 -> 检索/排序/合并。

import { createHash, randomUUID } from "crypto";
import { existsSync } from "fs";
import { readFile } from "fs/promises";
import { basename, extname } from "path";

import {
    DEFAULT_EMBEDDING_DIMENSION,
    getDimension,
    getTextEmbedding,
    getTextEmbeddings,
} from "../embedding";
import { QdrantVectorStore } from "../storage/qdrant_store";

type JsonRecord = Record<string, unknown>;

export interface RagChunk {
    id: string;
    content: string;
    metadata: JsonRecord;
}

export interface RagSearchResult {
    id: string | number;
    score: number;
    metadata: JsonRecord;
    rerank_score?: number;
}

export interface RankedItem {
    memory_id: string | number;
    score: number;
    vector_score: number;
    graph_score: number;
    content: string;
    metadata: JsonRecord;
    rerank_score?: number;
}

interface VectorStoreLike {
    addVectors?(vectors: number[][], metadata: JsonRecord[], ids?: string[]): Promise<boolean> | boolean;
    add_vectors?(args: { vectors: number[][]; metadata: JsonRecord[]; ids?: string[] }): Promise<boolean> | boolean;
    searchSimilar?(
        queryVector: number[],
        limit?: number,
        scoreThreshold?: number,
        where?: JsonRecord,
    ): Promise<RagSearchResult[]> | RagSearchResult[];
    search_similar?(args: {
        query_vector: number[];
        limit?: number;
        score_threshold?: number;
        where?: JsonRecord;
    }): Promise<RagSearchResult[]> | RagSearchResult[];
    clearCollection?(): Promise<boolean> | boolean;
    clear_collection?(): Promise<boolean> | boolean;
    getCollectionStats?(): Promise<JsonRecord> | JsonRecord;
    get_collection_stats?(): Promise<JsonRecord> | JsonRecord;
}

export interface RagPipeline {
    namespace: string;
    store: VectorStoreLike;
    addDocuments(filePaths: string[], chunkSize?: number, chunkOverlap?: number): Promise<number>;
    add_documents(filePaths: string[], chunkSize?: number, chunkOverlap?: number): Promise<number>;
    search(query: string, topK?: number, scoreThreshold?: number): Promise<RankedItem[]>;
    searchAdvanced(
        query: string,
        topK?: number,
        options?: { enableMqe?: boolean; enableHyde?: boolean; scoreThreshold?: number },
    ): Promise<RankedItem[]>;
    search_advanced(
        query: string,
        topK?: number,
        enableMqe?: boolean,
        enableHyde?: boolean,
        scoreThreshold?: number,
    ): Promise<RankedItem[]>;
    getStats(): Promise<JsonRecord>;
    get_stats(): Promise<JsonRecord>;
}

export interface CreateRagPipelineOptions {
    qdrantUrl?: string | null;
    qdrantApiKey?: string | null;
    collectionName?: string;
    ragNamespace?: string;
    vectorSize?: number;
}

interface Paragraph {
    content: string;
    heading_path: string | null;
    start: number;
    end: number;
}

interface ChunkSpan {
    content: string;
    start: number;
    end: number;
    heading_path: string | null;
}

class InMemoryVectorStore implements VectorStoreLike {
    private readonly points: Array<{ id: string; vector: number[]; metadata: JsonRecord }> = [];
    private readonly vectorSize: number;

    constructor(vectorSize: number = DEFAULT_EMBEDDING_DIMENSION) {
        this.vectorSize = vectorSize;
    }

    async addVectors(vectors: number[][], metadata: JsonRecord[], ids?: string[]): Promise<boolean> {
        for (let index = 0; index < vectors.length; index += 1) {
            const vector = normalizeVectorDimension(vectors[index], this.vectorSize);
            this.points.push({
                id: ids?.[index] ?? randomUUID(),
                vector,
                metadata: { ...(metadata[index] ?? {}) },
            });
        }
        return vectors.length > 0;
    }

    async add_vectors(args: { vectors: number[][]; metadata: JsonRecord[]; ids?: string[] }): Promise<boolean> {
        return this.addVectors(args.vectors, args.metadata, args.ids);
    }

    async searchSimilar(
        queryVector: number[],
        limit: number = 10,
        scoreThreshold?: number,
        where?: JsonRecord,
    ): Promise<RagSearchResult[]> {
        const normalizedQuery = normalizeVectorDimension(queryVector, this.vectorSize);
        const results: RagSearchResult[] = [];

        for (const point of this.points) {
            if (where && !Object.entries(where).every(([key, value]) => point.metadata[key] === value)) {
                continue;
            }

            const score = cosineSimilarity(normalizedQuery, point.vector);
            if (scoreThreshold !== undefined && score < scoreThreshold) {
                continue;
            }
            results.push({ id: point.id, score, metadata: { ...point.metadata } });
        }

        return results.sort((a, b) => b.score - a.score).slice(0, limit);
    }

    async search_similar(args: {
        query_vector: number[];
        limit?: number;
        score_threshold?: number;
        where?: JsonRecord;
    }): Promise<RagSearchResult[]> {
        return this.searchSimilar(args.query_vector, args.limit, args.score_threshold, args.where);
    }

    async clearCollection(): Promise<boolean> {
        this.points.length = 0;
        return true;
    }

    async clear_collection(): Promise<boolean> {
        return this.clearCollection();
    }

    async getCollectionStats(): Promise<JsonRecord> {
        return {
            store_type: "memory",
            points_count: this.points.length,
            vectors_count: this.points.length,
            config: {
                vector_size: this.vectorSize,
                distance: "Cosine",
            },
        };
    }

    async get_collection_stats(): Promise<JsonRecord> {
        return this.getCollectionStats();
    }
}

export async function getMarkitdownInstance(): Promise<any | null> {
    try {
        const mod = await optionalImport("markitdown");
        const MarkItDown = mod?.MarkItDown ?? mod?.default;
        return typeof MarkItDown === "function" ? new MarkItDown() : null;
    } catch {
        console.warn("[WARNING] MarkItDown not available. Install a compatible JS package or rely on fallback reader.");
        return null;
    }
}

export function isMarkitdownSupportedFormat(path: string): boolean {
    const extension = extname(path).toLowerCase();
    return MARKITDOWN_SUPPORTED_FORMATS.has(extension);
}

export async function convertToMarkdown(path: string): Promise<string> {
    if (!existsSync(path)) return "";

    const extension = extname(path).toLowerCase();
    if (extension === ".pdf") {
        return enhancedPdfProcessing(path);
    }

    const mdInstance = await getMarkitdownInstance();
    if (!mdInstance) {
        return fallbackTextReader(path);
    }

    try {
        const result = await mdInstance.convert(path);
        const text = result?.text_content ?? result?.textContent ?? result?.markdown ?? result?.text;
        return typeof text === "string" && text.trim() ? text : "";
    } catch (error) {
        console.warn(`[WARNING] MarkItDown failed for ${path}: ${errorMessage(error)}`);
        return fallbackTextReader(path);
    }
}

export async function enhancedPdfProcessing(path: string): Promise<string> {
    console.log(`[RAG] Using enhanced PDF processing for: ${path}`);
    const mdInstance = await getMarkitdownInstance();
    if (!mdInstance) {
        return fallbackTextReader(path);
    }

    try {
        const result = await mdInstance.convert(path);
        const rawText = result?.text_content ?? result?.textContent ?? result?.markdown ?? result?.text;
        if (typeof rawText !== "string" || !rawText.trim()) {
            return "";
        }

        const cleanedText = postProcessPdfText(rawText);
        console.log(`[RAG] PDF post-processing completed: ${rawText.length} -> ${cleanedText.length} chars`);
        return cleanedText;
    } catch (error) {
        console.warn(`[WARNING] Enhanced PDF processing failed for ${path}: ${errorMessage(error)}`);
        return fallbackTextReader(path);
    }
}

export function postProcessPdfText(text: string): string {
    const cleanedLines: string[] = [];
    for (const rawLine of text.split(/\r?\n/)) {
        const line = rawLine.trim();
        if (!line) continue;
        if (line.length <= 2 && !/^\d+$/.test(line)) continue;
        if (/^\d+$/.test(line)) continue;
        if (["github", "project", "forks", "stars", "language"].includes(line.toLowerCase())) continue;
        cleanedLines.push(line);
    }

    const mergedLines: string[] = [];
    let index = 0;
    while (index < cleanedLines.length) {
        const currentLine = cleanedLines[index];
        const nextLine = cleanedLines[index + 1];

        if (
            currentLine.length < 60 &&
            nextLine &&
            !currentLine.endsWith("：") &&
            !currentLine.endsWith(":") &&
            !currentLine.startsWith("#") &&
            !nextLine.startsWith("#") &&
            nextLine.length < 120
        ) {
            mergedLines.push(`${currentLine} ${nextLine}`);
            index += 2;
            continue;
        }

        mergedLines.push(currentLine);
        index += 1;
    }

    const paragraphs: string[] = [];
    let currentParagraph: string[] = [];
    for (const line of mergedLines) {
        const startsNewParagraph =
            line.startsWith("#") ||
            line.endsWith("：") ||
            line.endsWith(":") ||
            line.length > 150 ||
            currentParagraph.length === 0;

        if (startsNewParagraph) {
            if (currentParagraph.length > 0) {
                paragraphs.push(currentParagraph.join(" "));
                currentParagraph = [];
            }
            paragraphs.push(line);
        } else {
            currentParagraph.push(line);
        }
    }

    if (currentParagraph.length > 0) {
        paragraphs.push(currentParagraph.join(" "));
    }

    return paragraphs.join("\n\n");
}

export async function fallbackTextReader(path: string): Promise<string> {
    try {
        return await readFile(path, "utf8");
    } catch {
        try {
            return await readFile(path, "latin1");
        } catch {
            return "";
        }
    }
}

export function detectLang(sample: string): string {
    if (!sample.trim()) return "unknown";
    if (/[\u4E00-\u9FFF]/u.test(sample)) return "zh";
    if (/[\u3040-\u30FF]/u.test(sample)) return "ja";
    if (/[가-힣]/u.test(sample)) return "ko";
    return "unknown";
}

export function isCjk(character: string): boolean {
    const code = character.codePointAt(0) ?? 0;
    return (
        (0x4E00 <= code && code <= 0x9FFF) ||
        (0x3400 <= code && code <= 0x4DBF) ||
        (0x20000 <= code && code <= 0x2A6DF) ||
        (0x2A700 <= code && code <= 0x2B73F) ||
        (0x2B740 <= code && code <= 0x2B81F) ||
        (0x2B820 <= code && code <= 0x2CEAF) ||
        (0xF900 <= code && code <= 0xFAFF)
    );
}

export function approxTokenLen(text: string): number {
    const cjk = Array.from(text).filter(isCjk).length;
    const nonCjkTokens = text.split(/\s+/).filter(Boolean).length;
    return cjk + nonCjkTokens;
}

export function splitParagraphsWithHeadings(text: string): Paragraph[] {
    const lines = text.split(/\r?\n/);
    let headingStack: string[] = [];
    const paragraphs: Paragraph[] = [];
    let buffer: string[] = [];
    let charPosition = 0;

    const flushBuffer = (endPosition: number): void => {
        if (!buffer.length) return;
        const content = buffer.join("\n").trim();
        if (!content) return;

        paragraphs.push({
            content,
            heading_path: headingStack.length ? headingStack.join(" > ") : null,
            start: Math.max(0, endPosition - content.length),
            end: endPosition,
        });
        buffer = [];
    };

    for (const line of lines) {
        if (line.trim().startsWith("#")) {
            flushBuffer(charPosition);
            let level = line.length - line.replace(/^#+/, "").length;
            const title = line.replace(/^#+/, "").trim();
            if (level <= 0) level = 1;
            if (level <= headingStack.length) {
                headingStack = headingStack.slice(0, level - 1);
            }
            headingStack.push(title);
            charPosition += line.length + 1;
            continue;
        }

        if (line.trim() === "") {
            flushBuffer(charPosition);
        } else {
            buffer.push(line);
        }
        charPosition += line.length + 1;
    }

    flushBuffer(charPosition);
    return paragraphs.length ? paragraphs : [{ content: text, heading_path: null, start: 0, end: text.length }];
}

export function chunkParagraphs(
    paragraphs: Paragraph[],
    chunkTokens: number,
    overlapTokens: number,
): ChunkSpan[] {
    const chunks: ChunkSpan[] = [];
    let current: Paragraph[] = [];
    let currentTokens = 0;
    let index = 0;

    const emitCurrent = (): void => {
        if (!current.length) return;
        chunks.push({
            content: current.map((item) => item.content).join("\n\n"),
            start: current[0].start,
            end: current[current.length - 1].end,
            heading_path: [...current].reverse().find((item) => item.heading_path)?.heading_path ?? null,
        });
    };

    while (index < paragraphs.length) {
        const paragraph = paragraphs[index];
        const paragraphTokens = approxTokenLen(paragraph.content) || 1;

        if (currentTokens + paragraphTokens <= chunkTokens || !current.length) {
            current.push(paragraph);
            currentTokens += paragraphTokens;
            index += 1;
        } else {
            emitCurrent();

            if (overlapTokens > 0 && current.length) {
                const kept: Paragraph[] = [];
                let keptTokens = 0;
                for (const item of [...current].reverse()) {
                    const tokens = approxTokenLen(item.content) || 1;
                    if (keptTokens + tokens > overlapTokens) break;
                    kept.push(item);
                    keptTokens += tokens;
                }
                current = kept.reverse();
                currentTokens = keptTokens;
            } else {
                current = [];
                currentTokens = 0;
            }
        }
    }

    emitCurrent();
    return chunks;
}

export async function loadAndChunkTexts(
    paths: string[],
    chunkSize: number = 800,
    chunkOverlap: number = 100,
    namespace: string | null = null,
    sourceLabel: string = "rag",
): Promise<RagChunk[]> {
    console.log(`[RAG] Universal loader start: files=${paths.length} chunk_size=${chunkSize} overlap=${chunkOverlap} ns=${namespace || "default"}`);
    const chunks: RagChunk[] = [];
    const seenHashes = new Set<string>();

    for (const path of paths) {
        if (!existsSync(path)) {
            console.warn(`[WARNING] File not found: ${path}`);
            continue;
        }

        console.log(`[RAG] Processing: ${path}`);
        const extension = extname(path).toLowerCase();
        const markdownText = await convertToMarkdown(path);

        if (!markdownText.trim()) {
            console.warn(`[WARNING] No content extracted from: ${path}`);
            continue;
        }

        const lang = detectLang(markdownText);
        const docId = md5(`${path}|${markdownText.length}`);
        const paragraphs = splitParagraphsWithHeadings(markdownText);
        const tokenChunks = chunkParagraphs(paragraphs, Math.max(1, chunkSize), Math.max(0, chunkOverlap));

        for (const chunk of tokenChunks) {
            const content = chunk.content;
            const start = chunk.start ?? 0;
            const end = chunk.end ?? start + content.length;
            const normalized = content.trim();
            if (!normalized) continue;

            const contentHash = md5(normalized);
            if (seenHashes.has(contentHash)) continue;
            seenHashes.add(contentHash);

            const chunkId = md5(`${docId}|${start}|${end}|${contentHash}`);
            chunks.push({
                id: chunkId,
                content,
                metadata: {
                    source_path: path,
                    source_name: basename(path),
                    file_ext: extension,
                    doc_id: docId,
                    lang,
                    start,
                    end,
                    content_hash: contentHash,
                    namespace: namespace || "default",
                    source: sourceLabel,
                    external: true,
                    heading_path: chunk.heading_path,
                    format: "markdown",
                },
            });
        }
    }

    console.log(`[RAG] Universal loader done: total_chunks=${chunks.length}`);
    return chunks;
}

export async function buildGraphFromChunks(neo4j: any, chunks: RagChunk[]): Promise<void> {
    const createdDocs = new Set<string>();

    for (const chunk of chunks) {
        const memoryId = chunk.id;
        const metadata = chunk.metadata ?? {};
        const sourcePath = asString(metadata.source_path);
        const docId = asString(metadata.doc_id);

        if (docId && !createdDocs.has(docId)) {
            createdDocs.add(docId);
            await callOptional(neo4j, ["addEntity", "add_entity"], {
                entity_id: docId,
                name: basename(sourcePath || docId),
                entity_type: "Document",
                properties: { source_path: sourcePath, lang: metadata.lang },
            });
        }

        await callOptional(neo4j, ["addEntity", "add_entity"], {
            entity_id: memoryId,
            name: memoryId,
            entity_type: "Memory",
            properties: {
                source_path: sourcePath,
                doc_id: docId,
                start: metadata.start,
                end: metadata.end,
            },
        });

        if (docId) {
            await callOptional(neo4j, ["addRelationship", "add_relationship"], {
                from_id: docId,
                to_id: memoryId,
                rel_type: "HAS_CHUNK",
                properties: {},
            });
        }
    }
}

export function preprocessMarkdownForEmbedding(text: string): string {
    return text
        .replace(/^#{1,6}\s+/gm, "")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .replace(/\*\*([^*]+)\*\*/g, "$1")
        .replace(/\*([^*]+)\*/g, "$1")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/```[^\n]*\n([\s\S]*?)```/g, "$1")
        .replace(/\n\s*\n/g, "\n\n")
        .replace(/[ \t]+/g, " ")
        .trim();
}

export function createDefaultVectorStore(dimension: number = DEFAULT_EMBEDDING_DIMENSION): VectorStoreLike {
    const qdrantUrl = process.env.QDRANT_URL;
    if (!qdrantUrl) {
        return new InMemoryVectorStore(dimension);
    }

    const store = new QdrantVectorStore(
        qdrantUrl,
        process.env.QDRANT_API_KEY,
        process.env.QDRANT_COLLECTION ?? "hello_agents_rag_vectors",
        dimension,
        "cosine",
    );
    store.initialize().catch((error) => {
        console.warn(`[WARNING] Qdrant initialization failed: ${errorMessage(error)}`);
    });
    return store;
}

export async function indexChunks(
    store: VectorStoreLike | null = null,
    chunks: RagChunk[] | null = null,
    cacheDb: string | null = null,
    batchSize: number = 64,
    ragNamespace: string = "default",
): Promise<void> {
    void cacheDb;
    if (!chunks?.length) {
        console.log("[RAG] No chunks to index");
        return;
    }

    const dimension = await getDimension(DEFAULT_EMBEDDING_DIMENSION);
    const targetStore = store ?? createDefaultVectorStore(dimension);
    const processedTexts = chunks.map((chunk) => preprocessMarkdownForEmbedding(chunk.content));
    const vectors: number[][] = [];

    console.log(`[RAG] Embedding start: total_texts=${processedTexts.length} batch_size=${batchSize}`);

    for (let index = 0; index < processedTexts.length; index += batchSize) {
        const batch = processedTexts.slice(index, index + batchSize);
        try {
            const batchVectors = await getTextEmbeddings(batch, dimension);
            vectors.push(...batchVectors.map((vector) => normalizeVectorDimension(vector, dimension)));
        } catch (error) {
            console.warn(`[WARNING] Batch ${index} encoding failed: ${errorMessage(error)}`);
            console.log(`[RAG] Retrying batch ${index} with smaller chunks...`);

            for (let smallIndex = 0; smallIndex < batch.length; smallIndex += 8) {
                const smallBatch = batch.slice(smallIndex, smallIndex + 8);
                try {
                    await sleep(200);
                    const smallVectors = await getTextEmbeddings(smallBatch, dimension);
                    vectors.push(...smallVectors.map((vector) => normalizeVectorDimension(vector, dimension)));
                } catch (smallError) {
                    console.warn(`[WARNING] 小批次 ${Math.floor(smallIndex / 8)} 仍然失败: ${errorMessage(smallError)}`);
                    for (let i = 0; i < smallBatch.length; i += 1) {
                        vectors.push(new Array<number>(dimension).fill(0));
                    }
                }
            }
        }

        console.log(`[RAG] Embedding progress: ${Math.min(index + batchSize, processedTexts.length)}/${processedTexts.length}`);
    }

    const metadata = chunks.map((chunk) => ({
        memory_id: chunk.id,
        user_id: "rag_user",
        memory_type: "rag_chunk",
        content: chunk.content,
        data_source: "rag_pipeline",
        rag_namespace: ragNamespace,
        is_rag_data: true,
        ...chunk.metadata,
    }));
    const ids = chunks.map((chunk) => chunk.id);

    console.log(`[RAG] Qdrant upsert start: n=${vectors.length}`);
    const success = await addVectorsToStore(targetStore, vectors, metadata, ids);
    if (!success) {
        console.log("[RAG] Qdrant upsert failed");
        throw new Error("Failed to index vectors to Qdrant");
    }
    console.log(`[RAG] Qdrant upsert done: ${vectors.length} vectors indexed`);
}

export async function embedQuery(query: string): Promise<number[]> {
    const dimension = await getDimension(DEFAULT_EMBEDDING_DIMENSION);
    try {
        const vector = await getTextEmbedding(query, dimension);
        return normalizeVectorDimension(vector, dimension);
    } catch (error) {
        console.warn(`[WARNING] Query embedding failed: ${errorMessage(error)}`);
        return new Array<number>(dimension).fill(0);
    }
}

export async function searchVectors(
    store: VectorStoreLike | null = null,
    query: string = "",
    topK: number = 8,
    ragNamespace: string | null = null,
    onlyRagData: boolean = true,
    scoreThreshold?: number,
): Promise<RagSearchResult[]> {
    if (!query) return [];

    const targetStore = store ?? createDefaultVectorStore(await getDimension(DEFAULT_EMBEDDING_DIMENSION));
    const queryVector = await embedQuery(query);
    const where: JsonRecord = { memory_type: "rag_chunk" };
    if (onlyRagData) {
        where.is_rag_data = true;
        where.data_source = "rag_pipeline";
    }
    if (ragNamespace) {
        where.rag_namespace = ragNamespace;
    }

    try {
        return await searchStore(targetStore, queryVector, topK, scoreThreshold, where);
    } catch (error) {
        console.warn(`[WARNING] RAG search failed: ${errorMessage(error)}`);
        return [];
    }
}

async function promptMqe(query: string, n: number): Promise<string[]> {
    try {
        const { HelloAgentsLLM } = await import("../../../7.hello-agents/core/llm");
        const llm = new HelloAgentsLLM();
        const text = await llm.invoke([
            { role: "system", content: "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。" },
            { role: "user", content: `原始查询：${query}\n请给出${n}个不同表述的查询，每行一个。` },
        ]);
        const outputs = (text || "")
            .split(/\r?\n/)
            .map((line) => line.replace(/^[-\s\t]+/, "").trim())
            .filter(Boolean);
        return outputs.slice(0, n).length ? outputs.slice(0, n) : [query];
    } catch {
        return [query];
    }
}

async function promptHyde(query: string): Promise<string | null> {
    try {
        const { HelloAgentsLLM } = await import("../../../7.hello-agents/core/llm");
        const llm = new HelloAgentsLLM();
        return await llm.invoke([
            { role: "system", content: "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。" },
            { role: "user", content: `问题：${query}\n请直接写一段中等长度、客观、包含关键术语的段落。` },
        ]);
    } catch {
        return null;
    }
}

export async function searchVectorsExpanded(
    store: VectorStoreLike | null = null,
    query: string = "",
    topK: number = 8,
    ragNamespace: string | null = null,
    onlyRagData: boolean = true,
    scoreThreshold?: number,
    enableMqe: boolean = false,
    mqeExpansions: number = 2,
    enableHyde: boolean = false,
    candidatePoolMultiplier: number = 4,
): Promise<RagSearchResult[]> {
    if (!query) return [];

    const targetStore = store ?? createDefaultVectorStore(await getDimension(DEFAULT_EMBEDDING_DIMENSION));
    const expansions = [query];
    if (enableMqe && mqeExpansions > 0) {
        expansions.push(...await promptMqe(query, mqeExpansions));
    }
    if (enableHyde) {
        const hydeText = await promptHyde(query);
        if (hydeText) expansions.push(hydeText);
    }

    const uniqueExpansions = [...new Set(expansions.filter(Boolean))];
    const pool = Math.max(topK * candidatePoolMultiplier, 20);
    const perExpansion = Math.max(1, Math.floor(pool / Math.max(1, uniqueExpansions.length)));
    const where: JsonRecord = { memory_type: "rag_chunk" };
    if (onlyRagData) {
        where.is_rag_data = true;
        where.data_source = "rag_pipeline";
    }
    if (ragNamespace) {
        where.rag_namespace = ragNamespace;
    }

    const aggregate = new Map<string | number, RagSearchResult>();
    for (const expandedQuery of uniqueExpansions) {
        const queryVector = await embedQuery(expandedQuery);
        const hits = await searchStore(targetStore, queryVector, perExpansion, scoreThreshold, where);
        for (const hit of hits) {
            const memoryId = asId(hit.metadata?.memory_id ?? hit.id);
            const existing = aggregate.get(memoryId);
            if (!existing || hit.score > existing.score) {
                aggregate.set(memoryId, hit);
            }
        }
    }

    return [...aggregate.values()].sort((a, b) => b.score - a.score).slice(0, topK);
}

export async function rerankWithCrossEncoder(
    query: string,
    items: RagSearchResult[],
    modelName: string = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    topK: number = 10,
): Promise<RagSearchResult[]> {
    void query;
    void modelName;
    return items.slice(0, topK);
}

export function computeGraphSignalsFromPool(
    vectorHits: RagSearchResult[],
    sameDocWeight: number = 1.0,
    proximityWeight: number = 1.0,
    proximityWindowChars: number = 1600,
): Record<string, number> {
    const byDoc = new Map<string, RagSearchResult[]>();
    for (const hit of vectorHits) {
        const metadata = hit.metadata ?? {};
        const docId = asString(metadata.doc_id ?? metadata.memory_id ?? hit.id) || asString(hit.id);
        const items = byDoc.get(docId) ?? [];
        items.push(hit);
        byDoc.set(docId, items);
    }

    const docCounts = new Map([...byDoc.entries()].map(([docId, items]) => [docId, items.length]));
    const maxCount = Math.max(1, ...docCounts.values());
    const graphSignal: Record<string, number> = {};

    for (const [docId, items] of byDoc.entries()) {
        items.sort((a, b) => toNumber(a.metadata?.start, 0) - toNumber(b.metadata?.start, 0));
        const density = (docCounts.get(docId) ?? 1) / maxCount;

        for (let i = 0; i < items.length; i += 1) {
            const hit = items[i];
            const memoryId = asString(hit.metadata?.memory_id ?? hit.id);
            const position = toNumber(hit.metadata?.start, 0);
            let proximity = 0;

            for (let j = i - 1; j >= 0; j -= 1) {
                const dist = Math.abs(position - toNumber(items[j].metadata?.start, 0));
                if (dist > proximityWindowChars) break;
                proximity += Math.max(0, 1 - dist / Math.max(1, proximityWindowChars));
            }
            for (let j = i + 1; j < items.length; j += 1) {
                const dist = Math.abs(position - toNumber(items[j].metadata?.start, 0));
                if (dist > proximityWindowChars) break;
                proximity += Math.max(0, 1 - dist / Math.max(1, proximityWindowChars));
            }

            graphSignal[memoryId] = (graphSignal[memoryId] ?? 0) + sameDocWeight * density + proximityWeight * proximity;
        }
    }

    const maxSignal = Math.max(0, ...Object.values(graphSignal));
    if (maxSignal > 0) {
        for (const key of Object.keys(graphSignal)) {
            graphSignal[key] /= maxSignal;
        }
    }
    return graphSignal;
}

export function rank(
    vectorHits: RagSearchResult[],
    graphSignals: Record<string, number> | null = null,
    wVector: number = 0.7,
    wGraph: number = 0.3,
): RankedItem[] {
    const signals = graphSignals ?? {};
    const items = vectorHits.map((hit) => {
        const memoryId = asId(hit.metadata?.memory_id ?? hit.id);
        const graphScore = Number(signals[String(memoryId)] ?? 0);
        const vectorScore = Number(hit.score ?? 0);
        return {
            memory_id: memoryId,
            score: wVector * vectorScore + wGraph * graphScore,
            vector_score: vectorScore,
            graph_score: graphScore,
            content: asString(hit.metadata?.content),
            metadata: hit.metadata ?? {},
        };
    });

    return items.sort((a, b) => b.score - a.score);
}

export function mergeSnippets(rankedItems: RankedItem[], maxChars: number = 1200): string {
    const output: string[] = [];
    let total = 0;

    for (const item of rankedItems) {
        const text = item.content.trim();
        if (!text) continue;
        if (total + text.length > maxChars) {
            const remain = maxChars - total;
            if (remain <= 0) break;
            output.push(text.slice(0, remain));
            break;
        }
        output.push(text);
        total += text.length;
    }

    return output.join("\n\n");
}

export function expandNeighborsFromPool(
    selected: RankedItem[],
    pool: RankedItem[],
    neighbors: number = 1,
    maxAdditions: number = 5,
): RankedItem[] {
    if (!selected.length || !pool.length || neighbors <= 0) return selected;

    const byDoc = new Map<string, RankedItem[]>();
    for (const item of pool) {
        const docId = asString(item.metadata?.doc_id);
        if (!docId) continue;
        const items = byDoc.get(docId) ?? [];
        items.push(item);
        byDoc.set(docId, items);
    }
    for (const items of byDoc.values()) {
        items.sort((a, b) => toNumber(a.metadata?.start, 0) - toNumber(b.metadata?.start, 0));
    }

    const selectedIds = new Set(selected.map((item) => String(item.memory_id)));
    const additions: RankedItem[] = [];

    for (const item of selected) {
        const docId = asString(item.metadata?.doc_id);
        const docItems = byDoc.get(docId);
        if (!docItems) continue;

        const itemIndex = docItems.findIndex((candidate) => String(candidate.memory_id) === String(item.memory_id));
        if (itemIndex < 0) continue;

        for (let offset = 1; offset <= neighbors; offset += 1) {
            for (const nextIndex of [itemIndex - offset, itemIndex + offset]) {
                const candidate = docItems[nextIndex];
                if (!candidate) continue;
                const memoryId = String(candidate.memory_id);
                if (selectedIds.has(memoryId)) continue;
                additions.push(candidate);
                selectedIds.add(memoryId);
                if (additions.length >= maxAdditions) break;
            }
            if (additions.length >= maxAdditions) break;
        }
        if (additions.length >= maxAdditions) break;
    }

    return [...selected, ...additions].sort((a, b) => (b.rerank_score ?? b.score) - (a.rerank_score ?? a.score));
}

export function mergeSnippetsGrouped(
    rankedItems: RankedItem[],
    maxChars: number = 1200,
    includeCitations: boolean = true,
): string {
    const byDoc = new Map<string, RankedItem[]>();
    const docScores = new Map<string, number>();

    for (const item of rankedItems) {
        const docId = asString(item.metadata?.doc_id ?? item.metadata?.source_path) || "unknown";
        const items = byDoc.get(docId) ?? [];
        items.push(item);
        byDoc.set(docId, items);
        docScores.set(docId, (docScores.get(docId) ?? 0) + item.score);
    }

    const orderedDocs = [...byDoc.keys()].sort((a, b) => (docScores.get(b) ?? 0) - (docScores.get(a) ?? 0));
    for (const docId of orderedDocs) {
        byDoc.get(docId)?.sort((a, b) => toNumber(a.metadata?.start, 0) - toNumber(b.metadata?.start, 0));
    }

    const output: string[] = [];
    const citations: JsonRecord[] = [];
    let total = 0;
    let citationIndex = 1;

    for (const docId of orderedDocs) {
        const parts = byDoc.get(docId) ?? [];
        for (const item of parts) {
            const text = item.content.trim();
            if (!text) continue;
            const suffix = includeCitations ? ` [${citationIndex}]` : "";
            const required = text.length + suffix.length;

            if (total + required > maxChars) {
                const remain = maxChars - total;
                if (remain <= 0) break;
                const clipped = text.slice(0, Math.max(0, remain - suffix.length));
                if (clipped) {
                    output.push(clipped + suffix);
                    total += clipped.length + suffix.length;
                    if (includeCitations) citations.push(makeCitation(item, citationIndex++));
                }
                break;
            }

            output.push(text + suffix);
            total += required;
            if (includeCitations) citations.push(makeCitation(item, citationIndex++));
        }
        if (total >= maxChars) break;
    }

    const merged = output.join("\n\n");
    if (!includeCitations || !citations.length) return merged;

    const lines = [merged, "", "References:"];
    for (const citation of citations) {
        const loc = citation.start !== undefined && citation.end !== undefined ? ` (${citation.start}-${citation.end})` : "";
        const heading = citation.heading_path ? ` - ${citation.heading_path}` : "";
        const source = citation.source_path ?? citation.doc_id ?? "source";
        lines.push(`[${citation.index}] ${source}${loc}${heading}`);
    }
    return lines.join("\n");
}

export function compressRankedItems(
    rankedItems: RankedItem[],
    enableCompression: boolean = true,
    maxPerDoc: number = 2,
    joinGap: number = 200,
): RankedItem[] {
    if (!enableCompression) return rankedItems;

    const byDocCount = new Map<string, number>();
    const lastByDoc = new Map<string, RankedItem>();
    const newItems: RankedItem[] = [];

    for (const item of rankedItems) {
        const docId = asString(item.metadata?.doc_id ?? item.metadata?.source_path) || "unknown";
        const start = toNumber(item.metadata?.start, 0);
        const end = toNumber(item.metadata?.end, start + item.content.length);
        const last = lastByDoc.get(docId);

        if (!last) {
            lastByDoc.set(docId, item);
            byDocCount.set(docId, 1);
            newItems.push(item);
            continue;
        }

        const lastStart = toNumber(last.metadata?.start, 0);
        const lastEnd = toNumber(last.metadata?.end, lastStart + last.content.length);

        if (start - lastEnd <= joinGap && start >= lastStart) {
            const addText = item.content.trim();
            if (addText) {
                last.content = last.content.trim() ? `${last.content.trim()}\n\n${addText}` : addText;
                last.metadata.end = Math.max(lastEnd, end);
                last.score = Math.max(last.score, item.score);
            }
            lastByDoc.set(docId, last);
        } else {
            const count = byDocCount.get(docId) ?? 0;
            if (count >= maxPerDoc) continue;
            newItems.push(item);
            lastByDoc.set(docId, item);
            byDocCount.set(docId, count + 1);
        }
    }

    return newItems;
}

export async function tldrSummarize(text: string, bullets: number = 3): Promise<string | null> {
    try {
        if (!text.trim()) return null;
        const { HelloAgentsLLM } = await import("../../../7.hello-agents/core/llm");
        const llm = new HelloAgentsLLM();
        return await llm.invoke([
            { role: "system", content: "请将以下内容概括为简洁的要点列表（最多3-5条），用中文，避免重复，突出关键信息。" },
            { role: "user", content: `请用 ${Math.max(1, Math.min(5, Math.floor(bullets)))} 条要点总结：\n\n${text}` },
        ]);
    } catch {
        return null;
    }
}

export function createRagPipeline(options: CreateRagPipelineOptions = {}): RagPipeline {
    const namespace = options.ragNamespace ?? "default";
    const collectionName = options.collectionName ?? "hello_agents_rag_vectors";
    const vectorSize = options.vectorSize ?? DEFAULT_EMBEDDING_DIMENSION;
    const qdrantUrl = options.qdrantUrl ?? process.env.QDRANT_URL;
    const qdrantApiKey = options.qdrantApiKey ?? process.env.QDRANT_API_KEY;

    let store: VectorStoreLike = qdrantUrl
        ? new QdrantVectorStore(qdrantUrl, qdrantApiKey ?? undefined, collectionName, vectorSize, "cosine")
        : new InMemoryVectorStore(vectorSize);

    let initialization = Promise.resolve();
    if (store instanceof QdrantVectorStore) {
        initialization = store.initialize().catch((error) => {
            console.warn(`[WARNING] Qdrant initialization failed, using in-memory store: ${errorMessage(error)}`);
            store = new InMemoryVectorStore(vectorSize);
        });
    }

    const pipeline: RagPipeline = {
        namespace,
        get store() {
            return store;
        },

        async addDocuments(filePaths: string[], chunkSize: number = 800, chunkOverlap: number = 100): Promise<number> {
            await initialization;
            const chunks = await loadAndChunkTexts(filePaths, chunkSize, chunkOverlap, namespace, "rag");
            await indexChunks(store, chunks, null, 64, namespace);
            return chunks.length;
        },

        async add_documents(filePaths: string[], chunkSize: number = 800, chunkOverlap: number = 100): Promise<number> {
            return this.addDocuments(filePaths, chunkSize, chunkOverlap);
        },

        async search(query: string, topK: number = 8, scoreThreshold?: number): Promise<RankedItem[]> {
            await initialization;
            const hits = await searchVectors(store, query, topK, namespace, true, scoreThreshold);
            return rank(hits);
        },

        async searchAdvanced(
            query: string,
            topK: number = 8,
            options: { enableMqe?: boolean; enableHyde?: boolean; scoreThreshold?: number } = {},
        ): Promise<RankedItem[]> {
            await initialization;
            const hits = await searchVectorsExpanded(
                store,
                query,
                topK,
                namespace,
                true,
                options.scoreThreshold,
                options.enableMqe ?? false,
                2,
                options.enableHyde ?? false,
            );
            const graphSignals = computeGraphSignalsFromPool(hits);
            return rank(hits, graphSignals);
        },

        async search_advanced(
            query: string,
            topK: number = 8,
            enableMqe: boolean = false,
            enableHyde: boolean = false,
            scoreThreshold?: number,
        ): Promise<RankedItem[]> {
            return this.searchAdvanced(query, topK, { enableMqe, enableHyde, scoreThreshold });
        },

        async getStats(): Promise<JsonRecord> {
            await initialization;
            return getStoreStats(store);
        },

        async get_stats(): Promise<JsonRecord> {
            return this.getStats();
        },
    };

    return pipeline;
}

export const _get_markitdown_instance = getMarkitdownInstance;
export const _is_markitdown_supported_format = isMarkitdownSupportedFormat;
export const _convert_to_markdown = convertToMarkdown;
export const _enhanced_pdf_processing = enhancedPdfProcessing;
export const _post_process_pdf_text = postProcessPdfText;
export const _fallback_text_reader = fallbackTextReader;
export const _detect_lang = detectLang;
export const _is_cjk = isCjk;
export const _approx_token_len = approxTokenLen;
export const _split_paragraphs_with_headings = splitParagraphsWithHeadings;
export const _chunk_paragraphs = chunkParagraphs;
export const load_and_chunk_texts = loadAndChunkTexts;
export const build_graph_from_chunks = buildGraphFromChunks;
export const _preprocess_markdown_for_embedding = preprocessMarkdownForEmbedding;
export const _create_default_vector_store = createDefaultVectorStore;
export const index_chunks = indexChunks;
export const embed_query = embedQuery;
export const search_vectors = searchVectors;
export const search_vectors_expanded = searchVectorsExpanded;
export const rerank_with_cross_encoder = rerankWithCrossEncoder;
export const compute_graph_signals_from_pool = computeGraphSignalsFromPool;
export const merge_snippets = mergeSnippets;
export const expand_neighbors_from_pool = expandNeighborsFromPool;
export const merge_snippets_grouped = mergeSnippetsGrouped;
export const compress_ranked_items = compressRankedItems;
export const tldr_summarize = tldrSummarize;
export const create_rag_pipeline = createRagPipeline;

const MARKITDOWN_SUPPORTED_FORMATS = new Set([
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm",
    ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
    ".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg",
    ".zip", ".tar", ".gz", ".rar",
    ".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".css", ".scss",
    ".log", ".conf", ".ini", ".cfg", ".yaml", ".yml", ".toml",
]);

function md5(text: string): string {
    return createHash("md5").update(text, "utf8").digest("hex");
}

function cosineSimilarity(a: number[], b: number[]): number {
    const length = Math.min(a.length, b.length);
    let dot = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < length; i += 1) {
        dot += a[i] * b[i];
        normA += a[i] * a[i];
        normB += b[i] * b[i];
    }

    return normA > 0 && normB > 0 ? dot / (Math.sqrt(normA) * Math.sqrt(normB)) : 0;
}

function normalizeVectorDimension(vector: unknown, dimension: number): number[] {
    const targetDimension = Math.max(1, Math.floor(dimension));
    const source = Array.isArray(vector) ? vector : [];
    const normalized = new Array<number>(targetDimension).fill(0);

    for (let i = 0; i < Math.min(source.length, targetDimension); i += 1) {
        const value = Number(source[i]);
        normalized[i] = Number.isFinite(value) ? value : 0;
    }

    return normalized;
}

async function addVectorsToStore(
    store: VectorStoreLike,
    vectors: number[][],
    metadata: JsonRecord[],
    ids: string[],
): Promise<boolean> {
    if (store.addVectors) {
        return Boolean(await store.addVectors(vectors, metadata, ids));
    }
    if (store.add_vectors) {
        return Boolean(await store.add_vectors({ vectors, metadata, ids }));
    }
    throw new Error("Vector store does not implement addVectors/add_vectors");
}

async function searchStore(
    store: VectorStoreLike,
    queryVector: number[],
    limit: number,
    scoreThreshold: number | undefined,
    where: JsonRecord,
): Promise<RagSearchResult[]> {
    if (store.searchSimilar) {
        return await store.searchSimilar(queryVector, limit, scoreThreshold, where);
    }
    if (store.search_similar) {
        return await store.search_similar({
            query_vector: queryVector,
            limit,
            score_threshold: scoreThreshold,
            where,
        });
    }
    throw new Error("Vector store does not implement searchSimilar/search_similar");
}

async function getStoreStats(store: VectorStoreLike): Promise<JsonRecord> {
    if (store.getCollectionStats) {
        return await store.getCollectionStats();
    }
    if (store.get_collection_stats) {
        return await store.get_collection_stats();
    }
    return { store_type: "unknown" };
}

async function optionalImport(packageName: string): Promise<any> {
    const importer = new Function("packageName", "return import(packageName)") as (name: string) => Promise<any>;
    return importer(packageName);
}

async function callOptional(target: any, methodNames: string[], args: JsonRecord): Promise<void> {
    for (const methodName of methodNames) {
        if (typeof target?.[methodName] === "function") {
            try {
                await target[methodName](args);
            } catch {
                return;
            }
            return;
        }
    }
}

function makeCitation(item: RankedItem, index: number): JsonRecord {
    return {
        index,
        source_path: item.metadata.source_path,
        doc_id: item.metadata.doc_id,
        start: item.metadata.start,
        end: item.metadata.end,
        heading_path: item.metadata.heading_path,
    };
}

function asString(value: unknown): string {
    if (value === null || value === undefined) return "";
    return String(value);
}

function asId(value: unknown): string | number {
    return typeof value === "number" || typeof value === "string" ? value : String(value ?? "");
}

function toNumber(value: unknown, fallback: number): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
}

function sleep(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function errorMessage(error: unknown): string {
    return error instanceof Error ? error.message : String(error);
}
