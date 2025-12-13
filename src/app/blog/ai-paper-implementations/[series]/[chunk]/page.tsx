import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import { serialize } from "next-mdx-remote/serialize";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypePrettyCode from "rehype-pretty-code";
import { Clock, BookOpen, ArrowLeft } from "lucide-react";
import { getSeriesMetadata, getChunkContent, getChunkIds } from "@/lib/content";
import { MDXRenderer } from "@/components/mdx/mdx-renderer";
import { ChunkNavigation } from "@/components/blog/chunk-navigation";
import { TableOfContents } from "@/components/blog/table-of-contents";

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

// Dynamic rendering to handle MDX processing at request time
export const dynamic = "force-dynamic";

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
          {/* Breadcrumb */}
          <div className="flex flex-wrap items-center gap-2 mb-8 text-sm">
            <Link
              href="/blog/ai-paper-implementations"
              className="text-muted-foreground hover:text-primary"
            >
              AI Papers
            </Link>
            <span className="text-muted-foreground">/</span>
            <Link
              href={`/blog/ai-paper-implementations/${series}`}
              className="text-muted-foreground hover:text-primary"
            >
              {seriesMetadata.short_title}
            </Link>
            <span className="text-muted-foreground">/</span>
            <span className="text-primary">{chunkData.metadata.title}</span>
          </div>

          <div className="grid gap-12 xl:grid-cols-[1fr_250px]">
            {/* Main Content */}
            <div>
              {/* Header */}
              <div className="mb-8">
                <Link
                  href={`/blog/ai-paper-implementations/${series}`}
                  className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-primary mb-4"
                >
                  <ArrowLeft className="h-4 w-4" />
                  Back to {seriesMetadata.short_title}
                </Link>

                <div className="flex items-center gap-2 text-sm text-muted-foreground mb-2">
                  <span>Chapter {chunkData.metadata.order}</span>
                  <span>•</span>
                  <span className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {chunkData.metadata.estimated_time}
                  </span>
                </div>

                <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-2">
                  {chunkData.metadata.title}
                </h1>

                {chunkData.metadata.subtitle && (
                  <p className="text-xl text-muted-foreground">
                    {chunkData.metadata.subtitle}
                  </p>
                )}

                {chunkData.metadata.topics && chunkData.metadata.topics.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-4">
                    {chunkData.metadata.topics.map((topic) => (
                      <span
                        key={topic}
                        className="px-2 py-1 text-xs rounded-full bg-primary/10 text-primary"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>
                )}
              </div>

              {/* MDX Content */}
              <article className="max-w-none">
                <MDXRenderer source={mdxSource} />
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
