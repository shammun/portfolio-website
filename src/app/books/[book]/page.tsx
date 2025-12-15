import Link from "next/link";
import type { Metadata } from "next";
import { notFound } from "next/navigation";
import {
  BookOpen,
  Clock,
  ArrowLeft,
  ExternalLink,
  CheckCircle,
  Code,
  Github,
  FileText,
} from "lucide-react";

// Book data - can be moved to a separate data file later
const booksData: Record<string, BookData> = {
  "llm-from-scratch": {
    id: "llm-from-scratch",
    title: "Build a Large Language Model (From Scratch)",
    shortTitle: "LLM from Scratch",
    author: "Sebastian Raschka",
    publisher: "Manning Publications",
    publishYear: 2024,
    description:
      "My learning journey through Professor Sebastian Raschka's excellent book on building LLMs from scratch. These are my notes, code implementations, and explanations as I worked through each chapter.",
    longDescription: `This section documents my learning journey through Professor Sebastian Raschka's excellent book "Build a Large Language Model (From Scratch)" published by Manning Publications. All credit for the original content, concepts, and code goes to Professor Raschka.

As I worked through each chapter, I created these notebooks with my own notes, explanations, and implementations to solidify my understanding. The book provides a unique opportunity to understand the inner workings of large language models by building one from scratch—starting with text processing and tokenization, progressing through attention mechanisms and transformer architecture, and culminating in pretraining and fine-tuning.

I hope sharing my learning process and simplified explanations helps others who are also working through this fantastic resource.`,
    totalChapters: 7,
    estimatedTime: "40-50 hours",
    difficulty: "Intermediate to Advanced",
    topics: [
      "Transformers",
      "Attention Mechanisms",
      "Tokenization",
      "Text Embeddings",
      "Pretraining",
      "Fine-tuning",
      "PyTorch",
      "GPT Architecture",
    ],
    prerequisites: [
      "Python programming",
      "Basic linear algebra",
      "Familiarity with PyTorch basics",
      "Understanding of neural networks",
    ],
    learningOutcomes: [
      "Implement a complete GPT-style language model",
      "Understand attention mechanisms deeply",
      "Build custom tokenizers and embeddings",
      "Pretrain models on text data",
      "Fine-tune models for classification",
      "Create instruction-following models",
    ],
    links: {
      book: "https://www.manning.com/books/build-a-large-language-model-from-scratch",
      github: "https://github.com/rasbt/LLMs-from-scratch",
    },
    chapters: [
      {
        id: "00",
        title: "Prerequisites: PyTorch Basics",
        htmlFile: "PyTorch_Basics.html",
        notebookFile: "PyTorch_Basics.ipynb",
        description: "Foundation concepts in PyTorch needed for the rest of the book",
        estimatedTime: "2-3 hours",
      },
      {
        id: "02",
        title: "Working with Text Data",
        htmlFile: "chapter_2.html",
        notebookFile: "chapter2_working_with_text_data.ipynb",
        description: "Tokenization, text processing, and creating data loaders",
        estimatedTime: "4-5 hours",
      },
      {
        id: "03",
        title: "Coding Attention Mechanisms",
        htmlFile: "chapter_3.html",
        notebookFile: "chapter3_Coding Attention Mechanism.ipynb",
        description: "Self-attention, causal attention, and multi-head attention",
        estimatedTime: "6-8 hours",
      },
      {
        id: "04",
        title: "Implementing GPT from Scratch",
        htmlFile: "chapter_4.html",
        notebookFile: "chapter4_Implementing a GPT model from scratch to generate text.ipynb",
        description: "Building the complete GPT architecture and generating text",
        estimatedTime: "6-8 hours",
      },
      {
        id: "05",
        title: "Pretraining on Unlabeled Data",
        htmlFile: "chapter_5.html",
        notebookFile: "chapter5_pretraining_on_unlabeled_data.ipynb",
        description: "Training loops, loss functions, and pretraining strategies",
        estimatedTime: "6-8 hours",
      },
      {
        id: "06",
        title: "Fine-tuning for Classification",
        htmlFile: "chapter_6.html",
        notebookFile: "chapter6_Fine-tuning for Classification.ipynb",
        description: "Adapting pretrained models for classification tasks",
        estimatedTime: "5-6 hours",
      },
      {
        id: "07",
        title: "Fine-tuning to Follow Instructions",
        htmlFile: "chapter_7.html",
        notebookFile: "chapter7_Fine-tuning to follow instructions.ipynb",
        description: "Creating instruction-following models through fine-tuning",
        estimatedTime: "5-6 hours",
      },
    ],
  },
};

interface BookData {
  id: string;
  title: string;
  shortTitle: string;
  author: string;
  publisher: string;
  publishYear: number;
  description: string;
  longDescription: string;
  totalChapters: number;
  estimatedTime: string;
  difficulty: string;
  topics: string[];
  prerequisites: string[];
  learningOutcomes: string[];
  links: {
    book: string;
    github: string;
  };
  chapters: {
    id: string;
    title: string;
    htmlFile: string;
    notebookFile: string;
    description: string;
    estimatedTime: string;
  }[];
}

type Props = {
  params: Promise<{ book: string }>;
};

export async function generateMetadata({ params }: Props): Promise<Metadata> {
  const { book } = await params;
  const bookData = booksData[book];

  if (!bookData) {
    return {
      title: "Book Not Found",
    };
  }

  return {
    title: `${bookData.shortTitle} | Book Implementation`,
    description: bookData.description,
  };
}

export default async function BookPage({ params }: Props) {
  const { book } = await params;
  const bookData = booksData[book];

  if (!bookData) {
    notFound();
  }

  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Breadcrumb */}
        <Link
          href="/books"
          className="inline-flex items-center gap-2 text-muted-foreground hover:text-foreground mb-8 transition-colors"
        >
          <ArrowLeft className="h-4 w-4" />
          Back to Books
        </Link>

        {/* Header */}
        <div className="max-w-4xl mb-12">
          <div className="flex flex-wrap items-center gap-2 mb-4">
            <span className="badge badge-cool">Completed</span>
            <span className="badge badge-primary">Featured</span>
          </div>

          <h1 className="text-3xl md:text-4xl font-bold text-foreground mb-3">
            {bookData.title}
          </h1>
          <p className="text-lg text-teal font-medium mb-4">
            by {bookData.author} - {bookData.publisher} ({bookData.publishYear})
          </p>
          <p className="text-muted-foreground mb-6 whitespace-pre-line">
            {bookData.longDescription}
          </p>

          {/* Quick Stats */}
          <div className="flex flex-wrap items-center gap-6 text-sm text-muted-foreground mb-6">
            <span className="flex items-center gap-2">
              <BookOpen className="h-4 w-4" />
              {bookData.totalChapters} chapters
            </span>
            <span className="flex items-center gap-2">
              <Clock className="h-4 w-4" />
              {bookData.estimatedTime}
            </span>
            <span className="badge badge-sm badge-muted">
              {bookData.difficulty}
            </span>
          </div>

          {/* External Links */}
          <div className="flex flex-wrap gap-3">
            <a
              href={bookData.links.book}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-secondary"
            >
              <ExternalLink className="h-4 w-4" />
              Official Book
            </a>
            <a
              href={bookData.links.github}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost"
            >
              <Github className="h-4 w-4" />
              Author's GitHub
            </a>
          </div>
        </div>

        <div className="grid gap-8 lg:grid-cols-3">
          {/* Main Content - Chapters */}
          <div className="lg:col-span-2">
            <div className="flex items-center gap-3 mb-6">
              <div className="icon-container-cool">
                <BookOpen className="h-6 w-6" />
              </div>
              <h2 className="text-2xl font-bold text-foreground">
                My Chapter Notes
              </h2>
            </div>

            <div className="space-y-4">
              {bookData.chapters.map((chapter, index) => (
                <div
                  key={chapter.id}
                  className="card group"
                >
                  <div className="flex items-start gap-4">
                    <div className="flex-shrink-0 mt-1">
                      <CheckCircle className="h-5 w-5 text-teal" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-sm text-muted-foreground">
                          Chapter {chapter.id}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          ({chapter.estimatedTime})
                        </span>
                      </div>
                      <h3 className="font-semibold text-foreground mb-2">
                        {chapter.title}
                      </h3>
                      <p className="text-sm text-muted-foreground mb-3">
                        {chapter.description}
                      </p>
                      <div className="flex flex-wrap gap-2">
                        <a
                          href={`/books/llm-from-scratch/${chapter.htmlFile}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
                        >
                          <FileText className="h-3 w-3" />
                          View HTML
                        </a>
                        <a
                          href={`/books/llm-from-scratch/${chapter.notebookFile}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
                        >
                          <Code className="h-3 w-3" />
                          Download Notebook
                        </a>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Topics */}
            <div className="card">
              <h3 className="font-semibold text-foreground mb-4">
                Topics Covered
              </h3>
              <div className="flex flex-wrap gap-2">
                {bookData.topics.map((topic) => (
                  <span key={topic} className="badge badge-sm badge-primary">
                    {topic}
                  </span>
                ))}
              </div>
            </div>

            {/* Prerequisites */}
            <div className="card">
              <h3 className="font-semibold text-foreground mb-4">
                Prerequisites
              </h3>
              <ul className="space-y-2">
                {bookData.prerequisites.map((prereq, index) => (
                  <li
                    key={index}
                    className="text-sm text-muted-foreground flex items-start gap-2"
                  >
                    <span className="text-primary">-</span>
                    {prereq}
                  </li>
                ))}
              </ul>
            </div>

            {/* Learning Outcomes */}
            <div className="card">
              <h3 className="font-semibold text-foreground mb-4">
                Key Takeaways
              </h3>
              <ul className="space-y-2">
                {bookData.learningOutcomes.map((outcome, index) => (
                  <li
                    key={index}
                    className="text-sm text-muted-foreground flex items-start gap-2"
                  >
                    <CheckCircle className="h-4 w-4 text-teal flex-shrink-0 mt-0.5" />
                    {outcome}
                  </li>
                ))}
              </ul>
            </div>

            {/* Quick Access */}
            <div className="rounded-2xl bg-gradient-to-br from-[#ecfdf5] to-[#f0fdfa] dark:from-teal/10 dark:to-teal/5 border border-[#6ee7b7]/30 p-6">
              <div className="flex items-center gap-2 mb-4">
                <div className="w-8 h-8 rounded-lg bg-[#14b8a6] flex items-center justify-center">
                  <BookOpen className="h-4 w-4 text-white" />
                </div>
                <h3 className="font-semibold text-foreground">
                  Browse Notes
                </h3>
              </div>

              <p className="text-sm text-muted-foreground mb-5 leading-relaxed">
                Access my complete chapter-by-chapter notes with code implementations and explanations.
              </p>

              <a
                href="/books/llm-from-scratch/index.html"
                target="_blank"
                rel="noopener noreferrer"
                className="group flex items-center justify-between w-full px-4 py-3 rounded-xl bg-white dark:bg-card border border-[#6ee7b7]/40 hover:border-[#14b8a6] hover:shadow-md transition-all duration-200"
              >
                <span className="font-medium text-[#0d9488]">Open Full Index</span>
                <ExternalLink className="h-4 w-4 text-[#14b8a6] group-hover:translate-x-0.5 transition-transform" />
              </a>

              <div className="mt-4 pt-4 border-t border-[#6ee7b7]/20">
                <p className="text-xs text-[#0d9488]/70 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-[#14b8a6]"></span>
                  Opens in a new tab with navigation
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
