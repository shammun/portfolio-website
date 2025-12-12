import Link from "next/link";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface ChunkInfo {
  id: string;
  title: string;
  order: number;
}

interface ChunkNavigationProps {
  seriesId: string;
  currentChunkId: string;
  chunks: ChunkInfo[];
}

export function ChunkNavigation({
  seriesId,
  currentChunkId,
  chunks,
}: ChunkNavigationProps) {
  const sortedChunks = [...chunks].sort((a, b) => a.order - b.order);
  const currentIndex = sortedChunks.findIndex(
    (c) => c.id === currentChunkId || c.id === `chunk${currentChunkId.replace("chunk", "")}`
  );

  const prevChunk = currentIndex > 0 ? sortedChunks[currentIndex - 1] : null;
  const nextChunk =
    currentIndex < sortedChunks.length - 1 ? sortedChunks[currentIndex + 1] : null;

  const getChunkPath = (chunk: ChunkInfo) => {
    return chunk.id.startsWith("chunk") ? chunk.id : `chunk${chunk.order}`;
  };

  return (
    <div className="border-t border-border pt-8 mt-12">
      {/* Progress indicator */}
      <div className="mb-6">
        <div className="flex items-center justify-between text-sm text-muted-foreground mb-2">
          <span>Progress</span>
          <span>
            {currentIndex + 1} of {sortedChunks.length}
          </span>
        </div>
        <div className="flex gap-1">
          {sortedChunks.map((chunk, index) => (
            <Link
              key={chunk.id}
              href={`/blog/ai-paper-implementations/${seriesId}/${getChunkPath(chunk)}`}
              className={cn(
                "flex-1 h-2 rounded-full transition-colors",
                index <= currentIndex
                  ? "bg-primary"
                  : "bg-muted hover:bg-muted-foreground/30"
              )}
              title={`Chapter ${chunk.order}: ${chunk.title}`}
            />
          ))}
        </div>
      </div>

      {/* Navigation buttons */}
      <div className="flex items-stretch gap-4">
        {prevChunk ? (
          <Link
            href={`/blog/ai-paper-implementations/${seriesId}/${getChunkPath(prevChunk)}`}
            className="flex-1 p-4 rounded-lg border border-border hover:border-primary/50 hover:bg-muted/50 transition-colors group"
          >
            <div className="flex items-center gap-2 text-sm text-muted-foreground mb-1">
              <ChevronLeft className="h-4 w-4" />
              Previous
            </div>
            <p className="font-medium text-foreground group-hover:text-primary transition-colors">
              {prevChunk.title}
            </p>
          </Link>
        ) : (
          <div className="flex-1" />
        )}

        {nextChunk ? (
          <Link
            href={`/blog/ai-paper-implementations/${seriesId}/${getChunkPath(nextChunk)}`}
            className="flex-1 p-4 rounded-lg border border-border hover:border-primary/50 hover:bg-muted/50 transition-colors group text-right"
          >
            <div className="flex items-center justify-end gap-2 text-sm text-muted-foreground mb-1">
              Next
              <ChevronRight className="h-4 w-4" />
            </div>
            <p className="font-medium text-foreground group-hover:text-primary transition-colors">
              {nextChunk.title}
            </p>
          </Link>
        ) : (
          <Link
            href={`/blog/ai-paper-implementations/${seriesId}`}
            className="flex-1 p-4 rounded-lg border border-primary bg-primary/10 hover:bg-primary/20 transition-colors group text-right"
          >
            <div className="flex items-center justify-end gap-2 text-sm text-primary mb-1">
              Complete!
              <ChevronRight className="h-4 w-4" />
            </div>
            <p className="font-medium text-primary">Back to Series Overview</p>
          </Link>
        )}
      </div>
    </div>
  );
}
