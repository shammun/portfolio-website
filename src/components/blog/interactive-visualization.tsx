"use client";

import { useEffect, useRef, useState } from "react";
import { ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";

interface InteractiveVisualizationProps {
  src: string;
  title?: string;
  caption?: string;
  height?: string;
  className?: string;
}

// Auto-resizing iframe wrapper for embedded HTML visualizations
export function InteractiveVisualization({
  src,
  title,
  caption,
  height = "800px",
  className,
}: InteractiveVisualizationProps) {
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const [iframeHeight, setIframeHeight] = useState<string>(height);

  useEffect(() => {
    const iframe = iframeRef.current;
    if (!iframe) return;

    const resizeIframe = () => {
      try {
        const iframeDoc = iframe.contentDocument || iframe.contentWindow?.document;
        if (iframeDoc) {
          // Get the full content height
          const body = iframeDoc.body;
          const html = iframeDoc.documentElement;

          if (body && html) {
            const contentHeight = Math.max(
              body.scrollHeight,
              body.offsetHeight,
              html.clientHeight,
              html.scrollHeight,
              html.offsetHeight
            );

            // Add some padding and set the height
            if (contentHeight > 100) {
              setIframeHeight(`${contentHeight + 20}px`);
            }
          }
        }
      } catch (e) {
        // Cross-origin restriction - use default height
        console.log("Could not auto-resize iframe:", e);
      }
    };

    // Resize on load
    iframe.addEventListener("load", resizeIframe);

    // Also try after a short delay for dynamic content
    const timeoutId = setTimeout(resizeIframe, 500);

    return () => {
      iframe.removeEventListener("load", resizeIframe);
      clearTimeout(timeoutId);
    };
  }, [src]);

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

        {/* Iframe Container - auto-resizing */}
        <div className="relative overflow-visible">
          <iframe
            ref={iframeRef}
            src={src}
            className="w-full border-0"
            style={{
              height: iframeHeight,
              minHeight: "400px",
              overflow: "visible"
            }}
            title={title || "Interactive Visualization"}
            loading="lazy"
            scrolling="no"
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
