import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { serialize } from "next-mdx-remote/serialize";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypePrettyCode from "rehype-pretty-code";
import { Clock, ArrowLeft } from "lucide-react";
import { getSeriesMetadata, getChunkContent, getChunkIds, getAllSeries } from "@/lib/content";
import { MDXContent } from "@/components/mdx/mdx-content";
import { ChunkNavigation } from "@/components/blog/chunk-navigation";
import { TableOfContents } from "@/components/blog/table-of-contents";

// Generate static params for all series/chunk combinations
export async function generateStaticParams() {
  const allSeries = await getAllSeries();
  const params: { series: string; chunk: string }[] = [];

  for (const series of allSeries) {
    const chunkIds = getChunkIds(series.paper_id);
    for (const chunkId of chunkIds) {
      params.push({ series: series.paper_id, chunk: chunkId });
    }
  }

  return params;
}

interface PageProps {
  params: Promise<{ series: string; chunk: string }>;
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { series, chunk } = await params;
  const seriesMetadata = await getSeriesMetadata(series);
  const chunkData = await getChunkContent(series, chunk);

  if (!seriesMetadata || !chunkData) {
    return { title: "Chapter Not Found" };
  }

  return {
    title: `${chunkData.metadata.title} | ${seriesMetadata.short_title}`,
    description: chunkData.metadata.description,
  };
}

// Use static generation - pages are pre-rendered at build time

export default async function ChunkPage({ params }: PageProps) {
  const { series, chunk } = await params;
  const seriesMetadata = await getSeriesMetadata(series);
  const chunkData = await getChunkContent(series, chunk);

  if (!seriesMetadata || !chunkData || !chunkData.content) {
    notFound();
  }

  // Preprocess content to fix image paths
  const imageBasePath = `/blog-images/${series}/${chunk}`;
  let processedContent = chunkData.content;

  // Fix figures/ relative paths
  processedContent = processedContent.replace(
    /!\[([^\]]*)\]\(figures\/([^)]+)\)/g,
    `![$1](${imageBasePath}/$2)`
  );

  // Fix standalone image names (like 01_spectral_conv_1d_basic.png)
  processedContent = processedContent.replace(
    /!\[([^\]]*)\]\((\d+_[^)]+\.png)\)/g,
    `![$1](${imageBasePath}/$2)`
  );

  // Fix chunk-prefixed image names (like chunk3_01_architecture.png)
  processedContent = processedContent.replace(
    /!\[([^\]]*)\]\((chunk\d+_[^)]+\.png)\)/g,
    `![$1](${imageBasePath}/$2)`
  );

  // Serialize MDX with math and code highlighting
  const mdxSource = await serialize(processedContent, {
    mdxOptions: {
      remarkPlugins: [remarkMath],
      rehypePlugins: [
        rehypeKatex,
        [
          rehypePrettyCode,
          {
            theme: "github-light",
            keepBackground: false,
          },
        ],
      ],
    },
  });

  // Get chunk info for navigation
  const chunkInfo = seriesMetadata.chunks.map((c) => ({
    id: c.id,
    title: c.title,
    order: c.order,
  }));

  return (
    <>
      {/* KaTeX CSS */}
      <link
        rel="stylesheet"
        href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
        integrity="sha384-n8MVd4RsNIU0tAv4ct0nTaAbDJwPJzDEaqSD1odI+WdtXRGWt2kTvGFasHpSy3SV"
        crossOrigin="anonymous"
      />

      <div className="py-12 md:py-20">
        <div className="container-wide">
          {/* Header Section */}
          <div className="mb-12 border-b border-border pb-10">
            {/* Breadcrumb and Back Link Row */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-10">
              <nav aria-label="Breadcrumb" className="flex text-sm">
                <ol className="flex items-center space-x-2 text-muted-foreground">
                  <li>
                    <Link href="/blog/ai-paper-implementations" className="hover:text-foreground transition-colors">
                      AI Papers
                    </Link>
                  </li>
                  <li className="text-border">/</li>
                  <li>
                    <Link href={`/blog/ai-paper-implementations/${series}`} className="hover:text-foreground transition-colors">
                      {seriesMetadata.short_title}
                    </Link>
                  </li>
                  <li className="text-border">/</li>
                  <li>
                    <span className="text-foreground font-medium truncate max-w-[150px] sm:max-w-none">
                      {chunkData.metadata.title}
                    </span>
                  </li>
                </ol>
              </nav>
              <Link
                href={`/blog/ai-paper-implementations/${series}`}
                className="group inline-flex items-center text-sm font-medium text-muted-foreground hover:text-primary transition-colors"
              >
                <ArrowLeft className="h-4 w-4 mr-1.5 transition-transform group-hover:-translate-x-1" />
                Back to {seriesMetadata.short_title}
              </Link>
            </div>

            {/* Chapter Info */}
            <div className="space-y-6">
              <div className="flex items-center gap-3">
                <span className="text-xs font-bold tracking-wider text-primary uppercase bg-primary/5 px-2 py-1 rounded">
                  Chapter {chunkData.metadata.order}
                </span>
                <span className="text-border">•</span>
                <div className="flex items-center text-xs font-medium text-muted-foreground uppercase tracking-wide">
                  <Clock className="h-4 w-4 mr-1.5" />
                  {chunkData.metadata.estimated_time}
                </div>
              </div>

              <h1 className="text-4xl sm:text-5xl font-extrabold text-foreground tracking-tight leading-tight">
                {chunkData.metadata.title}
              </h1>

              {chunkData.metadata.subtitle && (
                <p className="text-xl text-muted-foreground font-light leading-relaxed max-w-3xl">
                  {chunkData.metadata.subtitle}
                </p>
              )}

              {chunkData.metadata.topics && chunkData.metadata.topics.length > 0 && (
                <div className="flex flex-wrap gap-2.5 pt-2">
                  {chunkData.metadata.topics.map((topic) => (
                    <span
                      key={topic}
                      className="inline-flex items-center px-3 py-1.5 rounded-md text-xs font-medium bg-muted text-muted-foreground border border-border transition-colors hover:bg-muted/80"
                    >
                      {topic}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="grid gap-12 xl:grid-cols-[1fr_250px]">
            {/* Main Content */}
            <div>

              {/* MDX Content */}
              <article className="max-w-none">
                <MDXContent source={mdxSource} />
              </article>

              {/* Navigation */}
              <ChunkNavigation
                seriesId={series}
                currentChunkId={chunk}
                chunks={chunkInfo}
              />
            </div>

            {/* Table of Contents Sidebar */}
            <aside className="hidden xl:block">
              <TableOfContents content={chunkData.content} />
            </aside>
          </div>
        </div>
      </div>
    </>
  );
}
