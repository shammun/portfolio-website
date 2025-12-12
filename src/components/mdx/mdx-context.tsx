"use client";

import { createContext, useContext, type ReactNode } from "react";

interface MDXContextValue {
  seriesId: string;
  chunkId: string;
}

const MDXContext = createContext<MDXContextValue | null>(null);

export function MDXProvider({
  children,
  seriesId,
  chunkId,
}: {
  children: ReactNode;
  seriesId: string;
  chunkId: string;
}) {
  return (
    <MDXContext.Provider value={{ seriesId, chunkId }}>
      {children}
    </MDXContext.Provider>
  );
}

export function useMDXContext() {
  return useContext(MDXContext);
}
