"use client";

import { Maximize2, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

interface InteractiveVisualizationProps {
  src: string;
  title?: string;
  caption?: string;
  height?: string;
  className?: string;
}

// Simple iframe wrapper without complex state management
export function InteractiveVisualization({
  src,
  title,
  caption,
  height = "800px",
  className,
}: InteractiveVisualizationProps) {

  return (
    <figure className={cn("my-8 not-prose", className)}>
      <div className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-900 shadow-sm group">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-4 py-2.5 bg-slate-50 dark:bg-slate-800/80 border-b border-slate-200 dark:border-slate-700/50">
          <div className="flex items-center gap-3">
            {/* Window Controls (decorative) */}
            <div className="flex items-center gap-1.5">
              <div className="w-3 h-3 rounded-full bg-red-400" />
              <div className="w-3 h-3 rounded-full bg-yellow-400" />
              <div className="w-3 h-3 rounded-full bg-green-400" />
            </div>

            {title && (
              <span className="text-sm font-medium text-slate-600 dark:text-slate-300">
                {title}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1">
            <a
              href={src}
              target="_blank"
              rel="noopener noreferrer"
              className="p-1.5 rounded-md hover:bg-slate-200 dark:hover:bg-slate-700 text-slate-500 hover:text-slate-700 dark:hover:text-slate-300 transition-colors"
              title="Open in new tab"
            >
              <ExternalLink className="h-4 w-4" />
            </a>
          </div>
        </div>

        {/* Iframe Container */}
        <div className="relative" style={{ height, minHeight: "400px" }}>
          <iframe
            src={src}
            className="w-full h-full border-0"
            style={{ minHeight: "400px" }}
            title={title || "Interactive Visualization"}
            loading="lazy"
          />
        </div>
      </div>

      {caption && (
        <figcaption className="text-sm text-muted-foreground text-center mt-3 italic">
          {caption}
        </figcaption>
      )}
    </figure>
  );
}

export default InteractiveVisualization;
