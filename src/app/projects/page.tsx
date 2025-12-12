import Link from "next/link";
import type { Metadata } from "next";
import { Github, ExternalLink, Star, Code } from "lucide-react";

export const metadata: Metadata = {
  title: "Projects",
  description:
    "Open source projects and tools for climate science, machine learning, and geospatial analysis by Shammunul Islam.",
};

const projects = [
  {
    id: "fno-lst-prediction",
    title: "Urban Heat Island Prediction with FNO",
    description:
      "Fourier Neural Operator model for predicting land surface temperature across New York City at 70m resolution using ECOSTRESS satellite data.",
    longDescription:
      "My MS thesis project applies ML model to urban climate prediction. The model processes ECOSTRESS thermal observations, ERA5 reanalysis, NDVI vegetation data, and urban morphology parameters to predict spatial temperature patterns.",
    status: "active",
    featured: true,
    github: "https://github.com/shammun/fno-lst-prediction",
    tags: ["PyTorch", "FNO", "Climate", "Remote Sensing", "ECOSTRESS"],
    year: 2024,
  },
  {
    id: "ai-paper-tutorials",
    title: "AI Paper Implementations from Scratch",
    description:
      "Interactive tutorials explaining foundational AI papers with theory and runnable Python code in the browser.",
    longDescription:
      "A collection of comprehensive tutorials that break down complex AI papers into digestible chunks with mathematical explanations and working code implementations. Currently featuring the Fourier Neural Operator paper.",
    status: "active",
    featured: true,
    github: "https://github.com/shammun/ai-paper-tutorials",
    demo: "/blog/ai-paper-implementations",
    tags: ["Education", "Python", "Neural Networks", "Interactive"],
    year: 2024,
  },
  {
    id: "extreme-precipitation-gnn",
    title: "Extreme Precipitation Prediction with GNN",
    description:
      "Predicting extreme precipitation events using remote sensing and ERA5 data with Graph Neural Networks.",
    longDescription:
      "Uses Google Earth Engine for remote sensing data extraction, Google Cloud Storage for big geospatial data handling, and implements Temporal Fusion Transformers and Graph Neural Networks for prediction.",
    status: "completed",
    featured: false,
    github: "https://github.com/shammun/extreme-precipitation-gnn",
    tags: ["GNN", "Climate", "Google Earth Engine", "PyTorch"],
    year: 2024,
  },
  {
    id: "rossby-deep-learning",
    title: "Rossby Wave Deep Learning",
    description:
      "A Streamlit application for simulating and predicting Rossby waves using various deep learning models.",
    longDescription:
      "Allows users to simulate and predict Rossby waves using LSTM, CNN, GAN, and Physics-Informed Neural Networks (PINNs). Includes interactive visualizations and model comparisons.",
    status: "completed",
    featured: false,
    github: "https://github.com/shammun/rossby_deepLearning",
    tags: ["Streamlit", "LSTM", "PINNs", "Atmospheric Science"],
    year: 2024,
  },
  {
    id: "satellite-image-classifier",
    title: "Satellite Image Classifier",
    description: "An AI-based application for satellite image classification.",
    status: "completed",
    featured: false,
    github: "https://github.com/shammun/image-classification-ai",
    demo: "https://shammun.github.io/image-classification-ai/",
    tags: ["Computer Vision", "Deep Learning", "Remote Sensing"],
    year: 2023,
  },
  {
    id: "cmip6-downscaling",
    title: "CMIP6 Statistical Downscaling",
    description:
      "Statistical downscaling of CMIP6 climate projections for Virginia using PRISM data with bias correction and quantile mapping.",
    status: "completed",
    featured: false,
    tags: ["Climate", "R", "Statistics", "CMIP6"],
    year: 2024,
  },
];

const featuredProjects = projects.filter((p) => p.featured);
const otherProjects = projects.filter((p) => !p.featured);

export default function ProjectsPage() {
  return (
    <div className="py-12 md:py-20">
      <div className="container-default">
        {/* Header */}
        <div className="max-w-2xl mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-foreground mb-4">
            Projects
          </h1>
          <p className="text-lg text-muted-foreground">
            Open source projects and tools for climate science, machine learning,
            and geospatial analysis. Most projects are available on GitHub.
          </p>
        </div>

        {/* Featured Projects */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Star className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Featured Projects
            </h2>
          </div>

          <div className="grid gap-6 md:grid-cols-2">
            {featuredProjects.map((project) => (
              <div
                key={project.id}
                className="card border-primary/30 bg-primary/5 group"
              >
                <div className="flex flex-wrap items-center gap-2 mb-3">
                  <span
                    className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                      project.status === "active"
                        ? "bg-green-500 text-white"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {project.status === "active" ? "Active" : "Completed"}
                  </span>
                  <span className="text-sm text-muted-foreground">
                    {project.year}
                  </span>
                </div>

                <h3 className="text-xl font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
                  {project.title}
                </h3>
                <p className="text-muted-foreground mb-4">
                  {project.longDescription || project.description}
                </p>

                <div className="flex flex-wrap gap-2 mb-4">
                  {project.tags.map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-1 text-xs rounded-full bg-primary/10 text-primary"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="flex flex-wrap gap-3">
                  {project.github && (
                    <a
                      href={project.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-sm text-foreground hover:text-primary transition-colors"
                    >
                      <Github className="h-4 w-4" />
                      GitHub
                    </a>
                  )}
                  {project.demo && (
                    <Link
                      href={project.demo}
                      className="inline-flex items-center gap-1 text-sm text-foreground hover:text-primary transition-colors"
                    >
                      <ExternalLink className="h-4 w-4" />
                      Demo
                    </Link>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* Other Projects */}
        <section className="mb-16">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 rounded-lg bg-primary/10 text-primary">
              <Code className="h-6 w-6" />
            </div>
            <h2 className="text-2xl md:text-3xl font-bold text-foreground">
              Other Projects
            </h2>
          </div>

          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {otherProjects.map((project) => (
              <div key={project.id} className="card group">
                <div className="flex flex-wrap items-center gap-2 mb-2">
                  <span
                    className={`px-2 py-0.5 text-xs font-medium rounded-full ${
                      project.status === "active"
                        ? "bg-green-500 text-white"
                        : "bg-muted text-muted-foreground"
                    }`}
                  >
                    {project.status === "active" ? "Active" : "Completed"}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {project.year}
                  </span>
                </div>

                <h3 className="text-lg font-semibold text-foreground mb-2 group-hover:text-primary transition-colors">
                  {project.title}
                </h3>
                <p className="text-sm text-muted-foreground mb-3">
                  {project.description}
                </p>

                <div className="flex flex-wrap gap-1 mb-3">
                  {project.tags.slice(0, 3).map((tag) => (
                    <span
                      key={tag}
                      className="px-2 py-0.5 text-xs rounded-full bg-muted text-muted-foreground"
                    >
                      {tag}
                    </span>
                  ))}
                </div>

                <div className="flex flex-wrap gap-3">
                  {project.github && (
                    <a
                      href={project.github}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-primary transition-colors"
                    >
                      <Github className="h-4 w-4" />
                      Code
                    </a>
                  )}
                  {project.demo && (
                    <a
                      href={project.demo}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-primary transition-colors"
                    >
                      <ExternalLink className="h-4 w-4" />
                      Demo
                    </a>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* GitHub CTA */}
        <div className="text-center py-12 px-6 rounded-2xl bg-muted">
          <h2 className="text-2xl font-bold text-foreground mb-4">
            More on GitHub
          </h2>
          <p className="text-muted-foreground max-w-xl mx-auto mb-6">
            Check out my GitHub profile for more projects, contributions, and code
            samples.
          </p>
          <a
            href="https://github.com/shammun"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-primary"
          >
            <Github className="h-5 w-5" />
            View GitHub Profile
          </a>
        </div>
      </div>
    </div>
  );
}
