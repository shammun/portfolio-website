import Link from "next/link";
import type { Metadata } from "next";
import {
  BookOpen,
  Clock,
  ArrowRight,
  Code,
  Brain,
  Layers,
  ExternalLink,
  Sparkles,
  Rocket,
} from "lucide-react";

export const metadata: Metadata = {
  title: "Technical Book Learning Journey",
  description:
    "My learning journey through technical AI/ML books. Notes, implementations, and explanations as I work through building LLMs and other models from scratch.",
};

const completedBooks = [
  {
    id: "llm-from-scratch",
    title: "Build a Large Language Model (From Scratch)",
    shortTitle: "LLM from Scratch",
    author: "Sebastian Raschka",
    publisher: "Manning Publications",
    publishYear: 2024,
    description:
      "My notes and implementations from working through Professor Sebastian Raschka's excellent book. Includes my explanations, code, and insights gained while learning to build a GPT-style model from scratch.",
    totalChapters: 7,
    estimatedTime: "40-50 hours",
    difficulty: "Intermediate to Advanced",
    status: "completed",
    topics: [
      "Transformers",
      "Attention Mechanisms",
      "Tokenization",
      "Pretraining",
      "Fine-tuning",
      "PyTorch",
    ],
    links: {
      book: "https://www.manning.com/books/build-a-large-language-model-from-scratch",
      github: "https://github.com/rasbt/LLMs-from-scratch",
    },
    featured: true,
  },
];

const plannedBooks = [
  {
    title: "Build a Reasoning Model (From Scratch)",
    author: "Sebastian Raschka",
    description: "Deep dive into building models capable of complex reasoning and chain-of-thought processes.",
    priority: "high",
    status: "next",
  },
  {
    title: "Build a DeepSeek Model (From Scratch)",
    author: "Dandekar et al.",
    description: "Implementation guide for building DeepSeek-style efficient language models.",
    priority: "high",
    status: "planned",
  },
];

const features = [
  {
    icon: Brain,
    title: "Key Insights",
    description:
      "Important concepts and takeaways distilled from each chapter",
    color: "cool",
  },
  {
    icon: Code,
    title: "Code Examples",
    description:
      "Working implementations where applicable, with annotations",
    color: "primary",
  },
  {
    icon: Layers,
    title: "Chapter Notes",
    description:
      "Structured notes following the book's progression",
    color: "warm",
  },
];

const getIconContainerClass = (color: string) => {
  switch (color) {
    case "cool":
      return "icon-container-cool";
    case "warm":
      return "icon-container-warm";
    case "primary":
      return "icon-container-primary";
    default:
      return "icon-container-primary";
  }
};

export default function BooksPage() {
  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="max-w-3xl mb-12">
          <div className="badge badge-cool mb-4">
            <BookOpen className="h-4 w-4" />
            Book Learning Journey
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            Technical Book Notes
          </h1>
          <p className="text-lg text-muted-foreground">
            My journey through technical books in AI, ML, Climate Science, Statistics, Remote Sensing,
            and related fields. Each entry includes personal notes, key insights, and where applicable,
            code implementations. All credit goes to the respective authors—these are my learning notes.
          </p>
        </div>

        {/* Features */}
        <div className="grid gap-4 sm:grid-cols-3 mb-16">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="card"
            >
              <div className={`${getIconContainerClass(feature.color)} w-fit mb-3`}>
                <feature.icon className="h-5 w-5" />
              </div>
              <h3 className="font-semibold text-foreground mb-1">
                {feature.title}
              </h3>
              <p className="text-sm text-muted-foreground">
                {feature.description}
              </p>
            </div>
          ))}
        </div>

        {/* Completed Books */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="icon-container-cool">
              <Sparkles className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Books Completed
            </h2>
          </div>

          <div className="space-y-6">
            {completedBooks.map((book) => (
              <Link
                key={book.id}
                href={`/books/${book.id}`}
                className="block hover:no-underline"
              >
                <div className="card-featured group">
                  <div className="flex flex-wrap items-center gap-2 mb-3">
                    {book.featured && (
                      <span className="badge badge-sm badge-primary">
                        Featured
                      </span>
                    )}
                    <span className="badge badge-sm badge-cool">
                      Completed
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {book.publisher} ({book.publishYear})
                    </span>
                  </div>

                  <h3 className="text-2xl font-bold text-foreground mb-2 group-hover:text-primary transition-colors">
                    {book.title}
                  </h3>
                  <p className="text-teal font-medium mb-3">
                    by {book.author}
                  </p>
                  <p className="text-muted-foreground mb-4">{book.description}</p>

                  <div className="flex flex-wrap items-center gap-6 mb-4 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <BookOpen className="h-4 w-4" />
                      {book.totalChapters} chapters
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      {book.estimatedTime}
                    </span>
                    <span className="badge badge-sm badge-muted">
                      {book.difficulty}
                    </span>
                  </div>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {book.topics.map((topic) => (
                      <span
                        key={topic}
                        className="badge badge-sm badge-primary"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>

                  <div className="inline-flex items-center gap-2 text-primary font-medium group-hover:gap-3 transition-all">
                    View Implementation
                    <ArrowRight className="h-4 w-4" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Coming Soon */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="icon-container-warm">
              <Rocket className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Coming Soon
            </h2>
          </div>
          <p className="text-muted-foreground mb-6">
            These books are next on my learning journey. I'll add my notes and implementations as I work through them.
          </p>

          <div className="grid gap-4 sm:grid-cols-2">
            {plannedBooks.map((book, index) => (
              <div
                key={index}
                className="card opacity-75"
              >
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className={`badge badge-sm ${
                      book.status === "next"
                        ? "badge-warm"
                        : "badge-muted"
                    }`}
                  >
                    {book.status === "next" ? "Up Next" : "Planned"}
                  </span>
                </div>
                <h3 className="font-semibold text-foreground mb-1">
                  {book.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-2">{book.author}</p>
                <p className="text-sm text-muted-foreground">{book.description}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Info Section */}
        <section className="py-12 px-6 rounded-[16px] bg-card shadow-soft">
          <h2 className="text-2xl font-bold text-foreground mb-6 text-center">
            Why Document the Learning Journey?
          </h2>
          <div className="grid gap-6 sm:grid-cols-3 max-w-3xl mx-auto">
            <div className="text-center">
              <div className="icon-container-cool mx-auto mb-3">
                <Brain className="h-6 w-6" />
              </div>
              <h3 className="font-semibold text-foreground mb-1">
                Deeper Understanding
              </h3>
              <p className="text-sm text-muted-foreground">
                Writing notes forces active engagement with the material
              </p>
            </div>
            <div className="text-center">
              <div className="icon-container-warm mx-auto mb-3">
                <BookOpen className="h-6 w-6" />
              </div>
              <h3 className="font-semibold text-foreground mb-1">Future Reference</h3>
              <p className="text-sm text-muted-foreground">
                Personal notes are easier to revisit than re-reading entire books
              </p>
            </div>
            <div className="text-center">
              <div className="icon-container-primary mx-auto mb-3">
                <Layers className="h-6 w-6" />
              </div>
              <h3 className="font-semibold text-foreground mb-1">
                Knowledge Sharing
              </h3>
              <p className="text-sm text-muted-foreground">
                Helping others who may benefit from simplified explanations
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
