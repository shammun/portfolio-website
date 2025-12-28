"use client";

import Link from "next/link";
import Image from "next/image";
import dynamic from "next/dynamic";
import { type ReactNode, isValidElement, Children } from "react";
import { Copy, ExternalLink, FileCode } from "lucide-react";
import { cn } from "@/lib/utils";

// Dynamic import with SSR disabled to prevent hooks issues during server rendering
const InteractiveVisualization = dynamic(
  () => import("@/components/blog/interactive-visualization").then(mod => mod.InteractiveVisualization),
  {
    ssr: false,
    loading: () => (
      <div className="my-8 rounded-xl border border-slate-200 dark:border-slate-700 bg-slate-100 dark:bg-slate-800 animate-pulse" style={{ height: "400px" }}>
        <div className="flex items-center justify-center h-full text-muted-foreground">
          Loading visualization...
        </div>
      </div>
    )
  }
);

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

// Language display names and icons - matching Clean Educational colors
const languageConfig: Record<string, { name: string; color: string }> = {
  python: { name: "Python", color: "#3776ab" },
  py: { name: "Python", color: "#3776ab" },
  javascript: { name: "JavaScript", color: "#f7df1e" },
  js: { name: "JavaScript", color: "#f7df1e" },
  typescript: { name: "TypeScript", color: "#3178c6" },
  ts: { name: "TypeScript", color: "#3178c6" },
  bash: { name: "Bash", color: "#4eaa25" },
  shell: { name: "Shell", color: "#4eaa25" },
  json: { name: "JSON", color: "#8b5cf6" },
  html: { name: "HTML", color: "#e34c26" },
  css: { name: "CSS", color: "#264de4" },
  sql: { name: "SQL", color: "#f97316" },
  markdown: { name: "Markdown", color: "#083fa1" },
  md: { name: "Markdown", color: "#083fa1" },
  yaml: { name: "YAML", color: "#cb171e" },
  rust: { name: "Rust", color: "#dea584" },
  go: { name: "Go", color: "#00add8" },
  java: { name: "Java", color: "#b07219" },
  cpp: { name: "C++", color: "#f34b7d" },
  c: { name: "C", color: "#555555" },
};

// Heading components with Clean Educational styling
function H1({ children, id, ...props }: { children?: ReactNode; id?: string; className?: string }) {
  const headingId = id || generateHeadingId(children);
  return (
    <h1
      id={headingId}
      className="scroll-mt-24 text-2xl sm:text-3xl font-bold text-foreground mt-12 mb-6 pb-3 border-b border-border"
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

// Code Block Component - Clean Educational style with purple-tinted shadow
function CodeBlock({ children, className, ...props }: { children?: ReactNode; className?: string }) {
  const language = detectLanguage(children);
  const codeContent = extractTextContent(children).trim();
  const lineCount = codeContent.split("\n").length;

  const langConfig = language ? languageConfig[language.toLowerCase()] : null;
  const displayLang = langConfig?.name || language?.toUpperCase() || "CODE";

  return (
    <div className="my-6 group max-w-full overflow-hidden">
      {/* Code Container with Clean Educational shadow */}
      <div className="rounded-[16px] sm:rounded-[16px] overflow-hidden border border-border shadow-soft">
        {/* Header Bar */}
        <div className="flex items-center justify-between px-3 sm:px-4 py-2 bg-muted border-b border-border">
          <div className="flex items-center gap-2 sm:gap-3 min-w-0">
            <FileCode className="h-4 w-4 text-muted-foreground flex-shrink-0" />
            <span
              className="text-xs font-semibold px-2 py-0.5 rounded-[8px] truncate"
              style={{
                backgroundColor: langConfig?.color ? `${langConfig.color}20` : 'var(--muted)',
                color: langConfig?.color || 'var(--muted-foreground)'
              }}
            >
              {displayLang}
            </span>
            <span className="text-xs text-muted-foreground hidden sm:inline">
              {lineCount} {lineCount === 1 ? 'line' : 'lines'}
            </span>
          </div>

          <button
            onClick={() => navigator.clipboard.writeText(codeContent)}
            className="flex items-center gap-1 px-2 py-1 rounded-lg text-xs font-medium bg-card hover:bg-primary/10 text-muted-foreground hover:text-primary transition-colors flex-shrink-0"
            title="Copy code"
          >
            <Copy className="h-3.5 w-3.5" />
            <span className="hidden sm:inline">Copy</span>
          </button>
        </div>

        {/* Code Content - Render children directly to preserve syntax highlighting */}
        <div className="overflow-x-auto code-block-content">
          <pre
            className="p-3 sm:p-4 text-xs sm:text-sm leading-5 sm:leading-6 font-mono bg-card"
            {...props}
          >
            {children}
          </pre>
        </div>
      </div>
    </div>
  );
}

// Inline code - Clean Educational styling
function InlineCode({ children, className, ...props }: { children?: ReactNode; className?: string }) {
  if (className?.includes("language-")) {
    return <code className={className} {...props}>{children}</code>;
  }

  return (
    <code
      className="bg-muted text-foreground px-1.5 py-0.5 rounded-[4px] text-[0.9em] font-mono"
      {...props}
    >
      {children}
    </code>
  );
}

// Custom link component with primary color
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

// Custom image component with Clean Educational styling
function CustomImage({ src, alt, ...props }: { src?: string; alt?: string; className?: string }) {
  if (!src) return null;

  // Handle relative paths for local images
  const imageSrc = src.startsWith("http") || src.startsWith("/") ? src : `/${src}`;

  return (
    <figure className="my-8">
      <div className="relative rounded-[16px] overflow-hidden border border-border bg-card shadow-soft">
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

// Blockquote - styled as callout with primary accent
function Blockquote({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <blockquote
      className="my-6 px-5 py-4 border-l-4 border-primary bg-primary/5 rounded-r-[8px] text-muted-foreground"
      {...props}
    >
      {children}
    </blockquote>
  );
}

// Table components with Clean Educational styling
function Table({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <div className="my-8 overflow-x-auto rounded-[16px] border border-border shadow-soft">
      <table className="w-full text-sm" {...props}>
        {children}
      </table>
    </div>
  );
}

function TableHead({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <thead className="bg-muted border-b border-border" {...props}>{children}</thead>;
}

function TableRow({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <tr className="border-b border-border last:border-b-0 hover:bg-muted/50 transition-colors" {...props}>{children}</tr>;
}

function TableCell({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <td className="px-4 py-3 text-muted-foreground" {...props}>{children}</td>;
}

function TableHeader({ children, ...props }: { children?: ReactNode; className?: string }) {
  return <th className="px-4 py-3 text-left font-semibold text-foreground" {...props}>{children}</th>;
}

// List components with primary accent markers
function UnorderedList({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <ul className="my-4 ml-6 list-disc space-y-2 marker:text-primary" {...props}>
      {children}
    </ul>
  );
}

function OrderedList({ children, ...props }: { children?: ReactNode; className?: string }) {
  return (
    <ol className="my-4 ml-6 list-decimal space-y-2 marker:text-primary marker:font-semibold" {...props}>
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

// Horizontal rule - decorative Clean Educational style
function HorizontalRule(props: { className?: string }) {
  return (
    <div className="my-10 flex items-center gap-4">
      <div className="flex-1 h-px bg-gradient-to-r from-transparent via-border to-transparent" />
      <div className="w-2 h-2 rounded-full bg-primary/40" />
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
