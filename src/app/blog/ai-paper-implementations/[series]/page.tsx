import Link from "next/link";
import { notFound } from "next/navigation";
import type { Metadata } from "next";
import {
  Clock,
  BookOpen,
  ArrowRight,
  ExternalLink,
  Github,
  CheckCircle,
  Circle,
  Users,
  Target,
  Sparkles,
} from "lucide-react";
import { getSeriesMetadata, getChunkIds } from "@/lib/content";

interface PageProps {
  params: Promise<{ series: string }>;
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  const { series } = await params;
  const metadata = await getSeriesMetadata(series);

  if (!metadata) {
    return { title: "Series Not Found" };
  }

  return {
    title: metadata.series_info.title,
    description: metadata.series_info.description,
  };
}

export default async function SeriesPage({ params }: PageProps) {
  const { series } = await params;
  const metadata = await getSeriesMetadata(series);

  if (!metadata) {
    notFound();
  }

  const chunkIds = getChunkIds(series);

  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="max-w-3xl mb-12">
          <div className="flex flex-wrap items-center gap-2 mb-4">
            <Link
              href="/blog/ai-paper-implementations"
              className="text-sm text-muted-foreground hover:text-primary"
            >
              AI Paper Implementations
            </Link>
            <span className="text-muted-foreground">/</span>
            <span className="text-sm text-primary">{metadata.short_title}</span>
          </div>

          <h1 className="text-3xl md:text-4xl lg:text-5xl font-bold text-foreground mb-4">
            {metadata.series_info.title}
          </h1>

          <div className="flex flex-wrap items-center gap-4 text-sm text-muted-foreground mb-6">
            <span className="flex items-center gap-1">
              <Users className="h-4 w-4" />
              {metadata.authors.slice(0, 3).join(", ")}
              {metadata.authors.length > 3 && " et al."}
            </span>
            <span>{metadata.venue}</span>
            <a
              href={`https://arxiv.org/abs/${metadata.arxiv}`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-primary hover:underline flex items-center gap-1"
            >
              arXiv:{metadata.arxiv}
              <ExternalLink className="h-3 w-3" />
            </a>
          </div>

          <p className="text-lg text-muted-foreground mb-6">
            {metadata.series_info.description}
          </p>

          <div className="flex flex-wrap items-center gap-6 text-sm">
            <span className="flex items-center gap-2 text-muted-foreground">
              <BookOpen className="h-4 w-4" />
              {metadata.series_info.total_chunks} chapters
            </span>
            <span className="flex items-center gap-2 text-muted-foreground">
              <Clock className="h-4 w-4" />
              {metadata.series_info.estimated_time}
            </span>
            <span className="px-3 py-1 rounded-full bg-primary/10 text-primary">
              {metadata.series_info.difficulty}
            </span>
          </div>

          <div className="flex flex-wrap gap-3 mt-6">
            <Link
              href={`/blog/ai-paper-implementations/${series}/chunk0`}
              className="btn btn-primary"
            >
              <Sparkles className="h-5 w-5" />
              Start Tutorial
            </Link>
            <a
              href={metadata.github}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
            >
              <Github className="h-5 w-5" />
              View Code
            </a>
          </div>
        </div>

        <div className="grid gap-12 lg:grid-cols-3">
          {/* Main Content - Chapters */}
          <div className="lg:col-span-2">
            <h2 className="text-2xl font-bold text-foreground mb-6">
              Tutorial Chapters
            </h2>

            <div className="space-y-4">
              {metadata.chunks.map((chunk, index) => {
                const chunkPath = chunk.id.startsWith("chunk")
                  ? chunk.id
                  : `chunk${chunk.order}`;
                const isAvailable = chunkIds.includes(chunkPath);

                return (
                  <div
                    key={chunk.id}
                    className={`relative pl-8 ${
                      index !== metadata.chunks.length - 1
                        ? "pb-4 border-l-2 border-border ml-3"
                        : "ml-3"
                    }`}
                  >
                    {/* Timeline node */}
                    <div
                      className={`absolute -left-3 top-0 w-6 h-6 rounded-full flex items-center justify-center ${
                        isAvailable
                          ? "bg-primary text-primary-foreground"
                          : "bg-muted text-muted-foreground"
                      }`}
                    >
                      {isAvailable ? (
                        <CheckCircle className="h-4 w-4" />
                      ) : (
                        <Circle className="h-4 w-4" />
                      )}
                    </div>

                    {/* Chapter card */}
                    {isAvailable ? (
                      <Link
                        href={`/blog/ai-paper-implementations/${series}/${chunkPath}`}
                        className="block"
                      >
                        <div className="card hover:border-primary/50 transition-colors group">
                          <div className="flex items-start justify-between gap-4">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-1">
                                <span className="text-xs text-muted-foreground">
                                  Chapter {chunk.order}
                                </span>
                                <span className="text-xs text-muted-foreground">
                                  • {chunk.estimated_time}
                                </span>
                              </div>
                              <h3 className="text-lg font-semibold text-foreground group-hover:text-primary transition-colors">
                                {chunk.title}
                              </h3>
                              {chunk.subtitle && (
                                <p className="text-sm text-muted-foreground mt-1">
                                  {chunk.subtitle}
                                </p>
                              )}
                              <p className="text-sm text-muted-foreground mt-2">
                                {chunk.description}
                              </p>

                              {chunk.topics && chunk.topics.length > 0 && (
                                <div className="flex flex-wrap gap-1 mt-3">
                                  {chunk.topics.slice(0, 4).map((topic) => (
                                    <span
                                      key={topic}
                                      className="px-2 py-0.5 text-xs rounded-full bg-muted text-muted-foreground"
                                    >
                                      {topic}
                                    </span>
                                  ))}
                                </div>
                              )}
                            </div>
                            <ArrowRight className="h-5 w-5 text-muted-foreground group-hover:text-primary transition-colors flex-shrink-0 mt-1" />
                          </div>
                        </div>
                      </Link>
                    ) : (
                      <div className="card opacity-60">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs text-muted-foreground">
                            Chapter {chunk.order}
                          </span>
                          <span className="text-xs text-muted-foreground">
                            • {chunk.estimated_time}
                          </span>
                          <span className="px-2 py-0.5 text-xs rounded-full bg-muted text-muted-foreground">
                            Coming Soon
                          </span>
                        </div>
                        <h3 className="text-lg font-semibold text-foreground">
                          {chunk.title}
                        </h3>
                        {chunk.subtitle && (
                          <p className="text-sm text-muted-foreground mt-1">
                            {chunk.subtitle}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Prerequisites */}
            <div className="card">
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <Target className="h-5 w-5 text-primary" />
                Prerequisites
              </h3>
              <ul className="space-y-2">
                {metadata.prerequisites.map((prereq, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-muted-foreground"
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-primary mt-1.5 flex-shrink-0" />
                    {prereq}
                  </li>
                ))}
              </ul>
            </div>

            {/* Learning Outcomes */}
            <div className="card">
              <h3 className="font-semibold text-foreground mb-4 flex items-center gap-2">
                <CheckCircle className="h-5 w-5 text-green-500" />
                What You&apos;ll Learn
              </h3>
              <ul className="space-y-2">
                {metadata.learning_outcomes.map((outcome, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-2 text-sm text-muted-foreground"
                  >
                    <CheckCircle className="h-4 w-4 text-green-500 mt-0.5 flex-shrink-0" />
                    {outcome}
                  </li>
                ))}
              </ul>
            </div>

            {/* Paper Info */}
            <div className="card bg-muted">
              <h3 className="font-semibold text-foreground mb-4">
                Original Paper
              </h3>
              <p className="text-sm font-medium text-foreground mb-2">
                {metadata.title}
              </p>
              <p className="text-sm text-muted-foreground mb-4">
                {metadata.authors.join(", ")}
              </p>
              <div className="flex flex-wrap gap-2">
                <a
                  href={`https://arxiv.org/abs/${metadata.arxiv}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                >
                  <ExternalLink className="h-3 w-3" />
                  arXiv
                </a>
                <a
                  href={metadata.github}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                >
                  <Github className="h-3 w-3" />
                  Code
                </a>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
