"use client";

import Link from "next/link";
import Image from "next/image";
import { type ReactNode, isValidElement, Children } from "react";
import { Copy, ExternalLink, FileCode } from "lucide-react";
import { cn } from "@/lib/utils";
import { InteractiveVisualization } from "@/components/blog/interactive-visualization";

// Helper function to extract text content from React children
function extractTextContent(node: ReactNode): string {
  if (typeof node === "string") return node;
  if (typeof node === "number") return String(node);
  if (!node) return "";

  if (Array.isArray(node)) {
    return node.map(extractTextContent).join("");
  }

  if (isValidElement(node)) {
    const props = node.props as { children?: ReactNode };
    return extractTextContent(props.children);
  }

  return "";
}

// Helper to detect language from code element class
function detectLanguage(children: ReactNode): string | null {
  if (!isValidElement(children)) return null;

  const props = children.props as { className?: string };
  const className = props.className || "";

  const match = className.match(/language-(\w+)/);
  return match ? match[1] : null;
}

// Helper to generate heading ID
function generateHeadingId(children: ReactNode): string | undefined {
  const text = extractTextContent(children);
  if (!text) return undefined;
  return text.toLowerCase().replace(/[^\w\s-]/g, "").replace(/\s+/g, "-").trim();
}

// Language display names and icons
const languageConfig: Record<string, { name: string; color: string }> = {
  python: { name: "Python", color: "#3776ab" },
  py: { name: "Python", color: "#3776ab" },
  javascript: { name: "JavaScript", color: "#f7df1e" },
  js: { name: "JavaScript", color: "#f7df1e" },
  typescript: { name: "TypeScript", color: "#3178c6" },
  ts: { name: "TypeScript", color: "#3178c6" },
  bash: { name: "Bash", color: "#4eaa25" },
  shell: { name: "Shell", color: "#4eaa25" },
  json: { name: "JSON", color: "#292929" },
  html: { name: "HTML", color: "#e34c26" },
  css: { name: "CSS", color: "#264de4" },
  sql: { name: "SQL", color: "#f29111" },
  markdown: { name: "Markdown", color: "#083fa1" },
  md: { name: "Markdown", color: "#083fa1" },
  yaml: { name: "YAML", color: "#cb171e" },
  rust: { name: "Rust", color: "#dea584" },
  go: { name: "Go", color: "#00add8" },
  java: { name: "Java", color: "#b07219" },
  cpp: { name: "C++", color: "#f34b7d" },
  c: { name: "C", color: "#555555" },
};

// Heading components
function H1({ children, id, ...props }: { children?: ReactNode; id?: string; className?: string }) {
  const headingId = id || generateHeadingId(children);
  return (
    <h1
      id={headingId}
      className="scroll-mt-24 text-2xl sm:text-3xl font-bold text-foreground mt-12 mb-6 pb-3 border-b border-border/50"
      {...props}
    >
      {children}
    </h1>
  );
}

function H2({ children, id, ...props }: { children?: ReactNode; id?: string; className?: string }) {
  const headingId = id || generateHeadingId(children);
  return (
    <h2
      id={headingId}
      className="scroll-mt-24 text-xl sm:text-2xl font-bold text-foreground mt-10 mb-4"
      {...props}
    >
      {children}
    </h2>
  );
}

function H3({ children, id, ...props }: { children?: ReactNode; id?: string; className?: string }) {
  const headingId = id || generateHeadingId(children);
  return (
    <h3
      id={headingId}
      className="scroll-mt-24 text-lg sm:text-xl font-semibold text-foreground mt-8 mb-3"
      {...props}
    >
      {children}
    </h3>
  );
}

function H4({ children, id, ...props }: { children?: ReactNode; id?: string; className?: string }) {
  const headingId = id || generateHeadingId(children);
  return (
    <h4
      id={headingId}
      className="scroll-mt-24 text-base sm:text-lg font-semibold text-foreground mt-6 mb-2"
      {...props}
    >
      {children}
    </h4>
  );
}

// Educational Code Block Component - clean, readable styling with syntax highlighting
function CodeBlock({ children, className, ...props }: { children?: ReactNode; className?: string }) {
  const language = detectLanguage(children);
  const codeContent = extractTextContent(children).trim();
  const lineCount = codeContent.split("\n").length;

  const langConfig = language ? languageConfig[language.toLowerCase()] : null;
  const displayLang = langConfig?.name || language?.toUpperCase() || "CODE";

  // Get the code element to preserve syntax highlighting
  const codeElement = isValidElement(children) ? children : null;

  return (
    <div className="my-8 group code-block-educational">
      {/* Code Container - Educational Style */}
      <div className="rounded-xl overflow-hidden shadow-md border-2 border-teal-200/60 dark:border-teal-700/40 bg-gradient-to-br from-slate-50 to-white dark:from-slate-800 dark:to-slate-850">
        {/* Header Bar - Teal accent theme */}
        <div className="flex items-center justify-between px-4 py-3 bg-gradient-to-r from-teal-50 to-cyan-50 dark:from-teal-900/30 dark:to-cyan-900/20 border-b-2 border-teal-200/60 dark:border-teal-700/40">
          <div className="flex items-center gap-3">
            {/* Code icon with teal styling */}
            <div className="flex items-center justify-center w-8 h-8 rounded-lg bg-teal-100 dark:bg-teal-800/50">
              <FileCode className="h-4 w-4 text-teal-600 dark:text-teal-400" />
            </div>

            {/* Language Badge */}
            <div className="flex items-center gap-2">
              <span
                className="text-xs font-bold px-3 py-1 rounded-full uppercase tracking-wide"
                style={{
                  backgroundColor: langConfig?.color ? `${langConfig.color}20` : 'rgb(13 115 119 / 0.15)',
                  color: langConfig?.color || '#0D7377',
                  border: `1px solid ${langConfig?.color ? `${langConfig.color}40` : 'rgb(13 115 119 / 0.3)'}`
                }}
              >
                {displayLang}
              </span>
              <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">
                {lineCount} {lineCount === 1 ? 'line' : 'lines'}
              </span>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Copy button - teal theme */}
            <button
              onClick={() => navigator.clipboard.writeText(codeContent)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold bg-teal-100 hover:bg-teal-200 dark:bg-teal-800/50 dark:hover:bg-teal-700/50 text-teal-700 dark:text-teal-300 transition-all border border-teal-200 dark:border-teal-700"
              title="Copy code"
            >
              <Copy className="h-3.5 w-3.5" />
              <span>Copy</span>
            </button>
          </div>
        </div>

        {/* Code Content - Educational readable styling */}
        <div className="overflow-x-auto code-content">
          <pre
            className={cn(
              "p-5 text-sm leading-7 font-mono",
              "bg-gradient-to-br from-amber-50/50 via-slate-50 to-cyan-50/30",
              "dark:from-slate-800/90 dark:via-slate-850 dark:to-slate-800/80",
              "text-slate-700 dark:text-slate-200",
              className
            )}
            {...props}
          >
            {codeElement || (
              <code className="block">
                {codeContent.split("\n").map((line, i) => (
                  <div key={i} className="table-row hover:bg-teal-50/50 dark:hover:bg-teal-900/10 transition-colors">
                    <span className="table-cell pr-5 pl-1 text-right text-teal-400/70 dark:text-teal-600/70 select-none w-[3.5ch] text-xs font-medium border-r border-teal-200/50 dark:border-teal-700/30 mr-4">
                      {i + 1}
                    </span>
                    <span className="table-cell pl-4">{line || " "}</span>
                  </div>
                ))}
              </code>
            )}
          </pre>
        </div>

        {/* Bottom accent bar */}
        <div className="h-1 bg-gradient-to-r from-teal-400 via-cyan-400 to-teal-500 dark:from-teal-600 dark:via-cyan-600 dark:to-teal-700" />
      </div>
    </div>
  );
}

// Inline code - Educational teal theme
function InlineCode({ children, className, ...props }: { children?: ReactNode; className?: string }) {
  if (className?.includes("language-")) {
    return <code className={className} {...props}>{children}</code>;
  }

  return (
    <code
      className="bg-teal-50 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300 px-1.5 py-0.5 rounded-md text-[0.9em] font-mono border border-teal-200 dark:border-teal-700/50 font-medium"
      {...props}
    >
      {children}
    </code>
  );
}

// Custom link component
function CustomLink({ href, children, ...props }: { href?: string; children?: ReactNode; className?: string }) {
  const isExternal = href?.startsWith("http");

  if (isExternal) {
    return (
      <a
        href={href}
        target="_blank"
        rel="noopener noreferrer"
        className="text-primary hover:text-primary/80 underline underline-offset-2 decoration-primary/30 hover:decoration-primary/60 transition-colors inline-flex items-center gap-1"
        {...props}
      >
        {children}
        <ExternalLink className="h-3 w-3 flex-shrink-0" />
      </a>
    );
  }

  return (
    <Link
      href={href || "#"}
      className="text-primary hover:text-primary/80 underline underline-offset-2 decoration-primary/30 hover:decoration-primary/60 transition-colors"
      {...props}
    >
      {children}
    </Link>
  );
}

// Custom image component (simplified without hooks for SSR compatibility)
function CustomImage({ src, alt, ...props }: { src?: string; alt?: string; className?: string }) {
  if (!src) return null;

  // Handle relative paths for local images
  const imageSrc = src.startsWith("http") || src.startsWith("/") ? src : `/${src}`;

  return (
    <figure className="my-8">
      <div className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700 bg-slate-50 dark:bg-slate-900 shadow-sm">
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          src={imageSrc}
          alt={alt || ""}
          className="w-full h-auto"
          loading="lazy"
          {...props}
        />
      </div>
      {alt && (
        <figcaption className="text-sm text-muted-foreground text-center mt-3 italic">
          {alt}
        </figcaption>
      )}
    </figure>
  );
}

// Blockquote - styled as a callout
function Blockquote({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <blockquote
      className="my-6 px-5 py-4 border-l-4 border-primary/60 bg-primary/5 rounded-r-lg text-muted-foreground"
      {...props}
    >
      {children}
    </blockquote>
  );
}

// Table components with improved styling
function Table({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <div className="my-8 overflow-x-auto rounded-xl border border-slate-200 dark:border-slate-700 shadow-sm">
      <table className="w-full text-sm" {...props}>
        {children}
      </table>
    </div>
  );
}

function TableHead({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <thead className="bg-slate-50 dark:bg-slate-800/80 border-b border-slate-200 dark:border-slate-700" {...props}>{children}</thead>;
}

function TableRow({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <tr className="border-b border-slate-200 dark:border-slate-700/50 last:border-b-0 hover:bg-slate-50 dark:hover:bg-slate-800/50 transition-colors" {...props}>{children}</tr>;
}

function TableCell({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <td className="px-4 py-3 text-muted-foreground" {...props}>{children}</td>;
}

function TableHeader({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <th className="px-4 py-3 text-left font-semibold text-foreground" {...props}>{children}</th>;
}

// List components
function UnorderedList({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <ul className="my-4 ml-6 list-disc space-y-2 marker:text-primary/60" {...props}>
      {children}
    </ul>
  );
}

function OrderedList({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <ol className="my-4 ml-6 list-decimal space-y-2 marker:text-primary/60 marker:font-semibold" {...props}>
      {children}
    </ol>
  );
}

function ListItem({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <li className="text-muted-foreground pl-2 leading-relaxed" {...props}>{children}</li>;
}

// Paragraph - handles images specially
function Paragraph({ children, ...props }: { children?: ReactNode; className?: string }) {
  const childArray = Children.toArray(children);

  // Check if any child is an image
  const hasImage = childArray.some(child => {
    if (isValidElement(child)) {
      const type = child.type;
      if (typeof type === 'string' && type === 'img') return true;
      const childProps = child.props as { src?: string };
      if (childProps.src) return true;
    }
    return false;
  });

  // Images get a div wrapper to avoid hydration issues
  if (hasImage) {
    return <div className="my-4" {...props}>{children}</div>;
  }

  return (
    <p className="my-4 text-muted-foreground leading-7" {...props}>
      {children}
    </p>
  );
}

// Horizontal rule
function HorizontalRule(props: { className?: string }) {
  return (
    <div className="my-10 flex items-center gap-4">
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
      <div className="w-2 h-2 rounded-full bg-border" />
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
    </div>
  );
}

// Strong/Bold text
function Strong({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <strong className="font-semibold text-foreground" {...props}>{children}</strong>;
}

// Emphasis/Italic text
function Em({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <em className="italic" {...props}>{children}</em>;
}

// Export all MDX components
export const mdxComponents = {
  h1: H1,
  h2: H2,
  h3: H3,
  h4: H4,
  p: Paragraph,
  a: CustomLink,
  img: CustomImage,
  pre: CodeBlock,
  code: InlineCode,
  blockquote: Blockquote,
  table: Table,
  thead: TableHead,
  tr: TableRow,
  td: TableCell,
  th: TableHeader,
  ul: UnorderedList,
  ol: OrderedList,
  li: ListItem,
  hr: HorizontalRule,
  strong: Strong,
  em: Em,
  InteractiveVisualization,
};
