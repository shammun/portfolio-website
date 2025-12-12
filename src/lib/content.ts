import fs from "fs";
import path from "path";
import matter from "gray-matter";

const CONTENT_DIR = path.join(process.cwd(), "blogs", "ai-paper-implementations");

export interface SeriesMetadata {
  paper_id: string;
  title: string;
  short_title: string;
  authors: string[];
  arxiv: string;
  github: string;
  year: number;
  venue: string;
  series_info: {
    title: string;
    description: string;
    total_chunks: number;
    estimated_time: string;
    difficulty: string;
    last_updated: string;
  };
  prerequisites: string[];
  learning_outcomes: string[];
  chunks: ChunkMetadata[];
}

export interface ChunkMetadata {
  id: string;
  title: string;
  subtitle?: string;
  order: number;
  estimated_time: string;
  description: string;
  topics: string[];
  is_extended?: boolean;
}

export interface ChunkContent {
  metadata: ChunkMetadata;
  content: string;
  codeBlocks: CodeBlock[];
}

export interface CodeBlock {
  id: string;
  title?: string;
  language: string;
  code: string;
}

export async function getSeriesMetadata(seriesId: string): Promise<SeriesMetadata | null> {
  try {
    const metadataPath = path.join(CONTENT_DIR, seriesId, "metadata.json");
    const content = fs.readFileSync(metadataPath, "utf-8");
    return JSON.parse(content) as SeriesMetadata;
  } catch {
    return null;
  }
}

export async function getAllSeries(): Promise<SeriesMetadata[]> {
  const series: SeriesMetadata[] = [];

  try {
    const dirs = fs.readdirSync(CONTENT_DIR);

    for (const dir of dirs) {
      const metadataPath = path.join(CONTENT_DIR, dir, "metadata.json");
      if (fs.existsSync(metadataPath)) {
        const content = fs.readFileSync(metadataPath, "utf-8");
        series.push(JSON.parse(content) as SeriesMetadata);
      }
    }
  } catch {
    console.error("Error reading series");
  }

  return series;
}

export async function getChunkContent(
  seriesId: string,
  chunkId: string
): Promise<{ content: string; metadata: ChunkMetadata } | null> {
  try {
    // Get series metadata to find chunk info
    const seriesMetadata = await getSeriesMetadata(seriesId);
    if (!seriesMetadata) return null;

    // Normalize chunk ID (chunk0, chunk1, etc.)
    const normalizedChunkId = chunkId.startsWith("chunk") ? chunkId : `chunk${chunkId}`;
    const chunkNumber = parseInt(normalizedChunkId.replace("chunk", ""));

    // Find chunk metadata - try different matching strategies
    let chunkMetadata = seriesMetadata.chunks.find(
      (c) => c.id === normalizedChunkId || c.id === chunkId || c.order === chunkNumber
    );

    // If chunk0 (introduction), create synthetic metadata if not found
    if (!chunkMetadata && chunkNumber === 0) {
      chunkMetadata = {
        id: "chunk0",
        title: "Introduction",
        subtitle: "Overview and motivation",
        order: 0,
        estimated_time: "15 min",
        description: "Introduction to the paper and tutorial series",
        topics: ["Overview", "Motivation"],
      };
    }

    if (!chunkMetadata) return null;

    // Try to find the chunk content file
    const chunksDir = path.join(CONTENT_DIR, seriesId, "chunks");
    const chunkDir = path.join(chunksDir, normalizedChunkId);

    // Look for markdown files in order of preference
    const possibleFiles = [
      path.join(chunkDir, `${normalizedChunkId}_complete.md`),
      path.join(chunkDir, `${normalizedChunkId}.md`),
      path.join(chunkDir, "theory.md"),
      path.join(chunkDir, `chunk${chunkNumber}_complete.md`),
    ];

    let content = "";
    for (const filePath of possibleFiles) {
      if (fs.existsSync(filePath)) {
        content = fs.readFileSync(filePath, "utf-8");
        break;
      }
    }

    if (!content) {
      console.log(`No content file found for ${normalizedChunkId}. Tried:`, possibleFiles);
      return null;
    }

    return {
      content,
      metadata: chunkMetadata,
    };
  } catch (error) {
    console.error(`Error loading chunk ${chunkId}:`, error);
    return null;
  }
}

export function getChunkIds(seriesId: string): string[] {
  try {
    const chunksDir = path.join(CONTENT_DIR, seriesId, "chunks");
    if (!fs.existsSync(chunksDir)) return [];

    const dirs = fs.readdirSync(chunksDir);
    return dirs.filter((d) => d.startsWith("chunk")).sort((a, b) => {
      const numA = parseInt(a.replace("chunk", "")) || 0;
      const numB = parseInt(b.replace("chunk", "")) || 0;
      return numA - numB;
    });
  } catch {
    return [];
  }
}

export function extractCodeBlocks(content: string): { cleanContent: string; codeBlocks: CodeBlock[] } {
  const codeBlocks: CodeBlock[] = [];
  let blockIndex = 0;

  // Match code blocks with optional language and title
  const codeBlockRegex = /```(\w+)?(?:\s+title="([^"]*)")?\n([\s\S]*?)```/g;

  const cleanContent = content.replace(codeBlockRegex, (match, lang, title, code) => {
    const id = `code-block-${blockIndex++}`;
    codeBlocks.push({
      id,
      title: title || undefined,
      language: lang || "python",
      code: code.trim(),
    });

    // Return a placeholder that we'll replace with the interactive editor
    return `<CodeBlock id="${id}" />`;
  });

  return { cleanContent, codeBlocks };
}

export function getChunkFigures(seriesId: string, chunkId: string): string[] {
  try {
    const figuresDir = path.join(CONTENT_DIR, seriesId, "chunks", chunkId, "figures");
    if (!fs.existsSync(figuresDir)) return [];

    const files = fs.readdirSync(figuresDir);
    return files
      .filter((f) => /\.(png|jpg|jpeg|gif|webp|svg)$/i.test(f))
      .map((f) => `/blog-images/${seriesId}/${chunkId}/${f}`);
  } catch {
    return [];
  }
}
