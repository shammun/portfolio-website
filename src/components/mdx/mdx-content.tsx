"use client";

import dynamic from "next/dynamic";
import type { MDXRemoteSerializeResult } from "next-mdx-remote";

// Dynamic import with SSR disabled to fix useState error in MDXRemote
const MDXRenderer = dynamic(
  () => import("@/components/mdx/mdx-renderer").then(mod => mod.MDXRenderer),
  {
    ssr: false,
    loading: () => (
      <div className="flex flex-col items-center py-16 bg-muted rounded-xl border border-dashed border-border">
        <p className="text-muted-foreground text-sm">Loading content...</p>
        <div className="mt-4 flex space-x-1">
          <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "0ms" }}></div>
          <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "150ms" }}></div>
          <div className="w-2 h-2 rounded-full bg-primary/60 animate-bounce" style={{ animationDelay: "300ms" }}></div>
        </div>
      </div>
    )
  }
);

interface MDXContentProps {
  source: MDXRemoteSerializeResult;
}

export function MDXContent({ source }: MDXContentProps) {
  return <MDXRenderer source={source} />;
}
