// 文档处理模块

import { createHash } from "crypto";
import { basename, extname, resolve } from "path";
import { readFile } from "fs/promises";

type Metadata = Record<string, unknown>;

export class Document {
    content: string;
    metadata: Metadata;
    docId: string;

    constructor(content: string, metadata: Metadata = {}, docId: string | null = null) {
        this.content = content;
        this.metadata = { ...metadata };
        this.docId = docId ?? md5(content);
    }

    get doc_id(): string {
        return this.docId;
    }
}

export class DocumentChunk {
    content: string;
    metadata: Metadata;
    chunkId: string;
    docId: string | null;
    chunkIndex: number;

    constructor(
        content: string,
        metadata: Metadata = {},
        chunkId: string | null = null,
        docId: string | null = null,
        chunkIndex: number = 0,
    ) {
        this.content = content;
        this.metadata = { ...metadata };
        this.docId = docId;
        this.chunkIndex = chunkIndex;

        const chunkContent = `${this.docId ?? ""}_${this.chunkIndex}_${this.content.slice(0, 50)}`;
        this.chunkId = chunkId ?? md5(chunkContent);
    }

    get chunk_id(): string {
        return this.chunkId;
    }

    get doc_id(): string | null {
        return this.docId;
    }

    get chunk_index(): number {
        return this.chunkIndex;
    }
}

export interface ParsedDocument {
    filePath: string;
    fileName: string;
    content: string;
    metadata: Metadata;
    docId: string;
    doc_id: string;
}

export interface DocumentProcessorOptions {
    chunkSize?: number;
    chunkOverlap?: number;
    separators?: string[] | null;
}

export class DocumentProcessor {
    chunkSize: number;
    chunkOverlap: number;
    separators: string[];

    constructor(
        chunkSizeOrOptions: number | DocumentProcessorOptions = 1000,
        chunkOverlap: number = 200,
        separators: string[] | null = null,
    ) {
        if (typeof chunkSizeOrOptions === "object") {
            this.chunkSize = sanitizePositiveInteger(chunkSizeOrOptions.chunkSize, 1000);
            this.chunkOverlap = sanitizeOverlap(chunkSizeOrOptions.chunkOverlap, this.chunkSize, 200);
            this.separators = chunkSizeOrOptions.separators ?? ["\n\n", "\n", "。", ".", " "];
        } else {
            this.chunkSize = sanitizePositiveInteger(chunkSizeOrOptions, 1000);
            this.chunkOverlap = sanitizeOverlap(chunkOverlap, this.chunkSize, 200);
            this.separators = separators ?? ["\n\n", "\n", "。", ".", " "];
        }
    }

    processDocument(document: Document): DocumentChunk[] {
        const chunks = this.splitText(document.content);
        const processedAt = new Date().toISOString();

        return chunks.map((chunkContent, index) => {
            const chunkMetadata: Metadata = {
                ...document.metadata,
                doc_id: document.docId,
                chunk_index: index,
                chunk_count: chunks.length,
                total_chunks: chunks.length,
                processed_at: processedAt,
                content: chunkContent,
            };

            return new DocumentChunk(
                chunkContent,
                chunkMetadata,
                null,
                document.docId,
                index,
            );
        });
    }

    processDocuments(documents: Document[]): DocumentChunk[] {
        const allChunks: DocumentChunk[] = [];
        for (const document of documents) {
            allChunks.push(...this.processDocument(document));
        }
        return allChunks;
    }

    splitText(text: string): string[] {
        const normalized = normalizePlainText(text);
        if (!normalized) return [];
        if (normalized.length <= this.chunkSize) return [normalized];

        const chunks: string[] = [];
        let start = 0;

        while (start < normalized.length) {
            const end = start + this.chunkSize;

            if (end >= normalized.length) {
                chunks.push(normalized.slice(start).trim());
                break;
            }

            let splitPoint = this.findSplitPoint(normalized, start, end);
            if (splitPoint === -1) {
                splitPoint = end;
            }

            const chunk = normalized.slice(start, splitPoint).trim();
            if (chunk) chunks.push(chunk);

            start = Math.max(start + 1, splitPoint - this.chunkOverlap);
        }

        return chunks;
    }

    findSplitPoint(text: string, start: number, end: number): number {
        for (const separator of this.separators) {
            if (!separator) continue;

            const searchStart = Math.max(start, end - 100);
            for (let i = end - separator.length; i >= searchStart; i -= 1) {
                if (text.slice(i, i + separator.length) === separator) {
                    return i + separator.length;
                }
            }
        }

        return -1;
    }

    mergeChunks(chunks: DocumentChunk[], maxLength: number = 2000): DocumentChunk[] {
        if (!chunks.length) return [];

        const mergedChunks: DocumentChunk[] = [];
        let currentChunk = cloneChunk(chunks[0]);

        for (const nextChunk of chunks.slice(1)) {
            const combinedLength = currentChunk.content.length + nextChunk.content.length;

            if (combinedLength <= maxLength && currentChunk.docId === nextChunk.docId) {
                currentChunk.content = `${currentChunk.content}\n${nextChunk.content}`;
                currentChunk.metadata.content = currentChunk.content;
                currentChunk.metadata.total_chunks = Number(currentChunk.metadata.total_chunks ?? 1) + 1;
                currentChunk.metadata.merged_chunk_ids = [
                    ...asStringArray(currentChunk.metadata.merged_chunk_ids),
                    nextChunk.chunkId,
                ];
            } else {
                mergedChunks.push(currentChunk);
                currentChunk = cloneChunk(nextChunk);
            }
        }

        mergedChunks.push(currentChunk);
        return mergedChunks;
    }

    filterChunks(chunks: DocumentChunk[], minLength: number = 50): DocumentChunk[] {
        return chunks.filter((chunk) => chunk.content.trim().length >= minLength);
    }

    addChunkMetadata(chunks: DocumentChunk[], metadata: Metadata): DocumentChunk[] {
        for (const chunk of chunks) {
            chunk.metadata = { ...chunk.metadata, ...metadata };
        }
        return chunks;
    }

    // Python 风格别名，便于示例迁移。
    process_document(document: Document): DocumentChunk[] {
        return this.processDocument(document);
    }

    process_documents(documents: Document[]): DocumentChunk[] {
        return this.processDocuments(documents);
    }

    _split_text(text: string): string[] {
        return this.splitText(text);
    }

    _find_split_point(text: string, start: number, end: number): number {
        return this.findSplitPoint(text, start, end);
    }

    merge_chunks(chunks: DocumentChunk[], maxLength: number = 2000): DocumentChunk[] {
        return this.mergeChunks(chunks, maxLength);
    }

    filter_chunks(chunks: DocumentChunk[], minLength: number = 50): DocumentChunk[] {
        return this.filterChunks(chunks, minLength);
    }

    add_chunk_metadata(chunks: DocumentChunk[], metadata: Metadata): DocumentChunk[] {
        return this.addChunkMetadata(chunks, metadata);
    }
}

const TEXT_EXTENSIONS = new Set([
    ".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".html", ".htm", ".xml",
    ".ts", ".tsx", ".js", ".jsx", ".py", ".java", ".go", ".rs", ".c", ".cpp", ".h",
    ".css", ".scss", ".sql", ".yaml", ".yml", ".toml", ".ini", ".log",
]);

const KNOWN_BINARY_EXTENSIONS = new Set([
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".png", ".jpg", ".jpeg",
    ".gif", ".webp", ".bmp", ".tiff", ".mp3", ".wav", ".m4a", ".mp4", ".mov", ".avi",
]);

export async function parseDocument(filePath: string): Promise<ParsedDocument> {
    const absolutePath = resolve(filePath);
    const fileName = basename(absolutePath);
    const extension = extname(fileName).toLowerCase();
    const buffer = await readFile(absolutePath);

    let content = "";
    if (TEXT_EXTENSIONS.has(extension)) {
        content = normalizeText(buffer.toString("utf8"), extension);
    } else if (!KNOWN_BINARY_EXTENSIONS.has(extension)) {
        const decoded = buffer.toString("utf8");
        content = printableRatio(decoded) > 0.85 ? normalizeText(decoded, extension) : "";
    }

    const document = new Document(content, {
        source: absolutePath,
        source_path: absolutePath,
        source_name: fileName,
        type: "text_file",
        extension: extension || "unknown",
        loaded_at: new Date().toISOString(),
    });

    return {
        filePath: absolutePath,
        fileName,
        content: document.content,
        metadata: document.metadata,
        docId: document.docId,
        doc_id: document.docId,
    };
}

export function splitTextIntoChunks(
    text: string,
    chunkSize: number = 800,
    chunkOverlap: number = 100,
): string[] {
    return new DocumentProcessor(chunkSize, chunkOverlap)._split_text(text);
}

export async function loadAndChunkDocuments(
    filePaths: string[],
    chunkSize: number = 800,
    chunkOverlap: number = 100,
): Promise<DocumentChunk[]> {
    const processor = new DocumentProcessor(chunkSize, chunkOverlap);
    const documents: Document[] = [];

    for (const filePath of filePaths) {
        const parsed = await parseDocument(filePath);
        documents.push(new Document(parsed.content, parsed.metadata, parsed.docId));
    }

    return processor.processDocuments(documents);
}

export async function loadTextFile(filePath: string, encoding: BufferEncoding = "utf8"): Promise<Document> {
    const content = await readFile(filePath, { encoding });
    const absolutePath = resolve(filePath);

    return new Document(content, {
        source: absolutePath,
        type: "text_file",
        loaded_at: new Date().toISOString(),
    });
}

export function createDocument(content: string, metadata: Metadata = {}): Document {
    return new Document(content, metadata);
}

export const load_text_file = loadTextFile;
export const create_document = createDocument;

function md5(text: string): string {
    return createHash("md5").update(text).digest("hex");
}

function normalizeText(text: string, extension: string): string {
    const withoutBom = text.replace(/^\uFEFF/, "");
    if (extension === ".html" || extension === ".htm") {
        return stripHtml(withoutBom);
    }
    return normalizePlainText(withoutBom);
}

function normalizePlainText(text: string): string {
    return text.replace(/\r\n/g, "\n").replace(/\r/g, "\n").trim();
}

function stripHtml(text: string): string {
    return text
        .replace(/<script[\s\S]*?<\/script>/gi, " ")
        .replace(/<style[\s\S]*?<\/style>/gi, " ")
        .replace(/<[^>]+>/g, " ")
        .replace(/&nbsp;/g, " ")
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/\s+/g, " ")
        .trim();
}

function printableRatio(text: string): number {
    if (!text) return 0;
    const printable = Array.from(text).filter((char) => {
        const code = char.charCodeAt(0);
        return code === 9 || code === 10 || code === 13 || (code >= 32 && code !== 127);
    }).length;
    return printable / text.length;
}

function sanitizePositiveInteger(value: unknown, fallback: number): number {
    const parsed = Number(value);
    return Number.isFinite(parsed) && parsed > 0 ? Math.floor(parsed) : fallback;
}

function sanitizeOverlap(value: unknown, chunkSize: number, fallback: number): number {
    const parsed = Number(value);
    const overlap = Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : fallback;
    return Math.min(overlap, Math.max(0, chunkSize - 1));
}

function cloneChunk(chunk: DocumentChunk): DocumentChunk {
    return new DocumentChunk(
        chunk.content,
        { ...chunk.metadata },
        chunk.chunkId,
        chunk.docId,
        chunk.chunkIndex,
    );
}

function asStringArray(value: unknown): string[] {
    return Array.isArray(value)
        ? value.map((item) => String(item))
        : [];
}

