import Link from "next/link";
import type { Metadata } from "next";
import {
  Sparkles,
  Clock,
  BookOpen,
  ArrowRight,
  Code,
  Brain,
} from "lucide-react";

export const metadata: Metadata = {
  title: "AI Paper Implementations from Scratch",
  description:
    "Interactive tutorials explaining foundational AI papers with theory and runnable Python code in the browser. Learn by understanding and implementing.",
};

const paperSeries = [
  {
    id: "fno-paper",
    title: "Fourier Neural Operator (FNO)",
    shortTitle: "FNO from Scratch",
    authors: "Li et al.",
    venue: "ICLR 2021",
    arxiv: "2010.08895",
    description:
      "A comprehensive tutorial with hands-on approach covering all the necessary theory and their implementation in Python on Fourier Neural Operators. Learn how to solve PDEs using neural networks that operate in Fourier space.",
    totalChunks: 7,
    estimatedTime: "8-12 hours",
    difficulty: "Intermediate",
    status: "published",
    topics: [
      "Neural Operators",
      "Fourier Transforms",
      "PDE Solvers",
      "Physics-Informed ML",
    ],
    featured: true,
  },
];

const plannedPapers = [
  {
    title: "Physics-Informed Neural Networks (PINNs)",
    authors: "Raissi, Perdikaris & Karniadakis",
    year: 2019,
    arxiv: "1711.10561",
    priority: "high",
  },
  {
    title: "Neural Ordinary Differential Equations",
    authors: "Chen et al.",
    year: 2018,
    arxiv: "1806.07366",
    priority: "high",
  },
  {
    title: "Pangu-Weather",
    authors: "Bi et al.",
    year: 2023,
    arxiv: "2211.02556",
    priority: "high",
  },
  {
    title: "FourCastNet",
    authors: "Pathak et al.",
    year: 2022,
    arxiv: "2202.11214",
    priority: "high",
  },
  {
    title: "DeepONet: Learning Nonlinear Operators",
    authors: "Lu et al.",
    year: 2021,
    arxiv: "1910.03193",
    priority: "medium",
  },
  {
    title: "Attention Is All You Need (Transformers)",
    authors: "Vaswani et al.",
    year: 2017,
    arxiv: "1706.03762",
    priority: "medium",
  },
];

const features = [
  {
    icon: BookOpen,
    title: "Clear Theory",
    description:
      "Mathematical concepts explained step-by-step with intuitive explanations",
  },
  {
    icon: Code,
    title: "Working Code",
    description:
      "Complete Python implementations you can modify and experiment with",
  },
  {
    icon: Brain,
    title: "Deep Understanding",
    description:
      "Learn the 'why' behind the 'how' for lasting comprehension",
  },
];

export default function AIPaperImplementationsPage() {
  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="max-w-3xl mb-12">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
            <Sparkles className="h-4 w-4" />
            Interactive Tutorials
          </div>
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            AI Paper Implementations from Scratch
          </h1>
          <p className="text-lg text-muted-foreground">
            Learn foundational AI papers by understanding the theory and
            implementing the code yourself. Each tutorial breaks down complex
            papers into digestible chunks with Python code.
          </p>
        </div>

        {/* Features */}
        <div className="grid gap-4 sm:grid-cols-3 mb-16">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="p-4 rounded-xl border border-border bg-card"
            >
              <div className="p-2 rounded-lg bg-primary/10 text-primary w-fit mb-3">
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

        {/* Published Tutorials */}
        <section className="mb-16">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-6">
            Available Tutorials
          </h2>

          <div className="space-y-6">
            {paperSeries.map((paper) => (
              <Link
                key={paper.id}
                href={`/blog/ai-paper-implementations/${paper.id}`}
                className="block"
              >
                <div className="card border-primary/30 bg-primary/5 hover:border-primary transition-colors group">
                  <div className="flex flex-wrap items-center gap-2 mb-3">
                    {paper.featured && (
                      <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-primary text-primary-foreground">
                        Featured
                      </span>
                    )}
                    <span className="px-2 py-0.5 text-xs font-medium rounded-full bg-green-500 text-white">
                      Published
                    </span>
                    <span className="text-sm text-muted-foreground">
                      {paper.venue}
                    </span>
                  </div>

                  <h3 className="text-2xl font-bold text-foreground mb-2 group-hover:text-primary transition-colors">
                    {paper.title}
                  </h3>
                  <p className="text-primary font-medium mb-3">
                    {paper.authors} • arXiv:{paper.arxiv}
                  </p>
                  <p className="text-muted-foreground mb-4">{paper.description}</p>

                  <div className="flex flex-wrap items-center gap-6 mb-4 text-sm text-muted-foreground">
                    <span className="flex items-center gap-1">
                      <BookOpen className="h-4 w-4" />
                      {paper.totalChunks} chapters
                    </span>
                    <span className="flex items-center gap-1">
                      <Clock className="h-4 w-4" />
                      {paper.estimatedTime}
                    </span>
                    <span className="px-2 py-0.5 rounded-full bg-muted text-muted-foreground text-xs">
                      {paper.difficulty}
                    </span>
                  </div>

                  <div className="flex flex-wrap gap-2 mb-4">
                    {paper.topics.map((topic) => (
                      <span
                        key={topic}
                        className="px-2 py-1 text-xs rounded-full bg-primary/10 text-primary"
                      >
                        {topic}
                      </span>
                    ))}
                  </div>

                  <div className="inline-flex items-center gap-2 text-primary font-medium group-hover:gap-3 transition-all">
                    Start Learning
                    <ArrowRight className="h-4 w-4" />
                  </div>
                </div>
              </Link>
            ))}
          </div>
        </section>

        {/* Coming Soon */}
        <section className="mb-16">
          <h2 className="text-2xl md:text-3xl font-bold text-foreground mb-6">
            Coming Soon
          </h2>
          <p className="text-muted-foreground mb-6">
            These papers are planned for future tutorials. Stay tuned!
          </p>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {plannedPapers.map((paper, index) => (
              <div
                key={index}
                className="p-4 rounded-xl border border-border bg-card opacity-75"
              >
                <div className="flex items-center gap-2 mb-2">
                  <span
                    className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                      paper.priority === "high"
                        ? "bg-accent/20 text-accent"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {paper.priority === "high" ? "High Priority" : "Planned"}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {paper.year}
                  </span>
                </div>
                <h3 className="font-semibold text-foreground mb-1">
                  {paper.title}
                </h3>
                <p className="text-sm text-muted-foreground">{paper.authors}</p>
                <p className="text-xs text-primary mt-2">arXiv:{paper.arxiv}</p>
              </div>
            ))}
          </div>
        </section>

        {/* Target Audience */}
        <section className="py-12 px-6 rounded-2xl bg-muted">
          <h2 className="text-2xl font-bold text-foreground mb-4 text-center">
            Who Are These Tutorials For?
          </h2>
          <div className="grid gap-4 sm:grid-cols-3 max-w-3xl mx-auto">
            <div className="text-center">
              <div className="text-3xl mb-2">🎓</div>
              <h3 className="font-semibold text-foreground mb-1">
                Graduate Students
              </h3>
              <p className="text-sm text-muted-foreground">
                Studying ML, climate science, or computational physics
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">🔬</div>
              <h3 className="font-semibold text-foreground mb-1">Researchers</h3>
              <p className="text-sm text-muted-foreground">
                Looking to apply neural operators to their domain
              </p>
            </div>
            <div className="text-center">
              <div className="text-3xl mb-2">💻</div>
              <h3 className="font-semibold text-foreground mb-1">
                ML Practitioners
              </h3>
              <p className="text-sm text-muted-foreground">
                Wanting to understand physics-informed methods
              </p>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
