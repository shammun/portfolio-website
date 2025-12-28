import Link from "next/link";
import {
  ArrowRight,
  BookOpen,
  Code,
  FlaskConical,
  Sparkles,
} from "lucide-react";
import ClimateAIBackground from "@/components/ClimateAIBackground";

export default function HomePage() {
  return (
    <div className="flex flex-col overflow-x-hidden">
      {/* Hero Section with Animated Background */}
      <section className="relative min-h-[80vh] overflow-hidden bg-gradient-to-br from-[#f0f9ff] via-[#f5f7ff] to-[#fdfeff]">
        {/* Background Animation Layer */}
        <div className="absolute inset-0 z-0">
          <ClimateAIBackground />
        </div>

        {/* Glass Overlay */}
        <div className="absolute inset-0 z-0 bg-white/40 pointer-events-none backdrop-blur-[1px]" />

        {/* Content Layer */}
        <div className="relative z-10 container-default py-20 md:py-32 flex items-center min-h-[80vh]">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-slate-800">
              Hi, I&apos;m{" "}
              <span className="text-primary">Shammunul Islam</span>
            </h1>

            <p className="mt-6 text-xl md:text-2xl text-slate-600 leading-relaxed">
              Exploring Climate Dynamics and AI
            </p>

            <p className="mt-4 text-lg text-slate-600 leading-relaxed max-w-2xl">
              I am studying{" "}
              <strong className="text-slate-800">
                Climate Science
              </strong>{" "}
              and exploring how{" "}
              <strong className="text-slate-800">
                AI
              </strong>{" "}
              can be used to build better climate models and solutions. Author of
              2 books on geospatial analysis, now researching urban climate.
            </p>

            <div className="mt-8 flex flex-wrap gap-4">
              <Link href="/blog/ai-paper-implementations" className="btn btn-primary">
                <BookOpen className="h-5 w-5" />
                Explore AI Tutorials
              </Link>
              <Link href="/about" className="btn btn-secondary">
                Learn More About Me
                <ArrowRight className="h-5 w-5" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Section - What I Do */}
      <section className="py-16 md:py-24 bg-card">
        <div className="container-default">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <span className="section-label">What I Do</span>
            <h2 className="mt-2 text-3xl md:text-4xl font-bold text-foreground">
              Bridging AI Research &amp; Climate Science
            </h2>
            <p className="mt-4 text-muted-foreground">
              Bridging the gap between cutting-edge AI research and climate
              science applications
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {/* Card 1: AI Paper Tutorials - Cool/Teal theme */}
            <div className="card group">
              <div className="flex items-center gap-3 mb-4">
                <div className="icon-container-cool">
                  <Sparkles className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">
                  AI Paper Tutorials
                </h3>
              </div>
              <p className="text-muted-foreground mb-4">
                Interactive tutorials explaining foundational AI papers with
                theory and runnable Python code in the browser. Learn by doing.
              </p>
              <Link
                href="/blog/ai-paper-implementations"
                className="inline-flex items-center gap-2 text-teal font-medium group-hover:gap-3 transition-all hover:no-underline"
              >
                Start Learning
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            {/* Card 2: Research - Warm/Orange theme */}
            <div className="card group">
              <div className="flex items-center gap-3 mb-4">
                <div className="icon-container-warm">
                  <FlaskConical className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">
                  Climate Research
                </h3>
              </div>
              <p className="text-muted-foreground mb-4">
                Applying neural operators and physics-informed ML to predict
                urban land surface temperature and understand climate dynamics.
              </p>
              <Link
                href="/research"
                className="inline-flex items-center gap-2 text-orange font-medium group-hover:gap-3 transition-all hover:no-underline"
              >
                View Publications
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            {/* Card 3: Projects - Primary/Purple theme */}
            <div className="card group">
              <div className="flex items-center gap-3 mb-4">
                <div className="icon-container-primary">
                  <Code className="h-6 w-6" />
                </div>
                <h3 className="text-xl font-semibold text-foreground">
                  Open Source
                </h3>
              </div>
              <p className="text-muted-foreground mb-4">
                Tools and implementations for climate data analysis, remote
                sensing, and machine learning. All available on GitHub.
              </p>
              <Link
                href="/projects"
                className="inline-flex items-center gap-2 text-primary font-medium group-hover:gap-3 transition-all hover:no-underline"
              >
                Browse Projects
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Book Tutorial */}
      <section className="py-16 md:py-24">
        <div className="container-default">
          <div className="grid gap-8 lg:grid-cols-2 items-center">
            <div className="order-2 lg:order-1">
              {/* Book preview card with purple-tinted shadow */}
              <div className="relative">
                <div className="bg-[#1E1E1E] rounded-[16px] p-6 shadow-soft-lg">
                  <div className="flex items-center gap-2 mb-4">
                    <div className="w-3 h-3 rounded-full bg-red-500" />
                    <div className="w-3 h-3 rounded-full bg-yellow-500" />
                    <div className="w-3 h-3 rounded-full bg-green-500" />
                    <span className="ml-2 text-sm text-gray-400">
                      tokenizer.py
                    </span>
                  </div>
                  <pre className="text-sm text-gray-300 overflow-x-auto">
                    <code>{`class SimpleTokenizer:
    """A simple text tokenizer for LLMs"""

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text):
        # Text → Token IDs
        tokens = re.split(r'([,.?!]|\\s)', text)
        return [self.str_to_int[t] for t in tokens
                if t.strip()]

    def decode(self, ids):
        # Token IDs → Text
        return " ".join(self.int_to_str[i] for i in ids)`}</code>
                  </pre>
                </div>
                {/* Decorative blur - warm themed */}
                <div className="absolute -z-10 top-4 left-4 right-4 bottom-4 bg-orange/20 rounded-xl blur-xl" />
              </div>
            </div>

            <div className="order-1 lg:order-2">
              <div className="badge badge-warm mb-4">
                <BookOpen className="h-4 w-4" />
                Featured Book Tutorial
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Build a Large Language Model from Scratch
              </h2>
              <p className="mt-4 text-lg text-muted-foreground">
                My learning journey through Sebastian Raschka&apos;s book with
                interactive visualizations, detailed explanations, and hands-on
                code examples. All chapters completed with interactive visualizations
                up to Chapter 2: Working with Text Data. Working on adding interactive
                components for all the remaining chapters.
              </p>

              <ul className="mt-6 space-y-3">
                {[
                  "Interactive visualizations explaining core concepts",
                  "Step-by-step tokenization and embedding tutorials",
                  "BPE, Word2Vec, and positional encoding explained",
                  "Clean Educational style with beautiful diagrams",
                ].map((item, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-3 text-muted-foreground"
                  >
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-warm-light text-warm-text flex items-center justify-center text-xs font-bold">
                      &#10003;
                    </span>
                    {item}
                  </li>
                ))}
              </ul>

              <div className="mt-8">
                <Link
                  href="/books/llm-from-scratch"
                  className="btn btn-warm"
                >
                  Explore the Book Tutorial
                  <ArrowRight className="h-5 w-5" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Tutorial */}
      <section className="py-16 md:py-24 bg-card">
        <div className="container-default">
          <div className="grid gap-8 lg:grid-cols-2 items-center">
            <div>
              <div className="badge badge-cool mb-4">
                <Sparkles className="h-4 w-4" />
                Featured Tutorial
              </div>
              <h2 className="text-3xl md:text-4xl font-bold text-foreground">
                Fourier Neural Operator (FNO) from Scratch
              </h2>
              <p className="mt-4 text-lg text-muted-foreground">
                A comprehensive, interactive tutorial on understanding and
                implementing Fourier Neural Operators. Start from mathematical
                foundations and build up to a complete implementation.
              </p>

              <ul className="mt-6 space-y-3">
                {[
                  "7 in-depth chapters covering theory and code",
                  "Interactive Python code editor in the browser",
                  "Real-time visualizations",
                  "8-12 hours of comprehensive learning content",
                ].map((item, i) => (
                  <li
                    key={i}
                    className="flex items-start gap-3 text-muted-foreground"
                  >
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-cool-light text-cool-text flex items-center justify-center text-xs font-bold">
                      &#10003;
                    </span>
                    {item}
                  </li>
                ))}
              </ul>

              <div className="mt-8">
                <Link
                  href="/blog/ai-paper-implementations/fno-paper"
                  className="btn btn-teal"
                >
                  Start the Tutorial
                  <ArrowRight className="h-5 w-5" />
                </Link>
              </div>
            </div>

            {/* Code preview card with purple-tinted shadow */}
            <div className="relative">
              <div className="bg-[#1E1E1E] rounded-[16px] p-6 shadow-soft-lg">
                <div className="flex items-center gap-2 mb-4">
                  <div className="w-3 h-3 rounded-full bg-red-500" />
                  <div className="w-3 h-3 rounded-full bg-yellow-500" />
                  <div className="w-3 h-3 rounded-full bg-green-500" />
                  <span className="ml-2 text-sm text-gray-400">
                    fno_layer.py
                  </span>
                </div>
                <pre className="text-sm text-gray-300 overflow-x-auto">
                  <code>{`class SpectralConv2d(nn.Module):
    """Spectral convolution layer for FNO"""

    def __init__(self, in_channels, out_channels,
                 modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2

        # Complex weights for Fourier modes
        self.weights = nn.Parameter(
            torch.randn(in_channels, out_channels,
                       modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        # FFT -> Multiply -> IFFT
        x_ft = torch.fft.rfft2(x)
        out_ft = self.compl_mul(x_ft, self.weights)
        return torch.fft.irfft2(out_ft)`}</code>
                </pre>
              </div>
              {/* Decorative blur - purple themed */}
              <div className="absolute -z-10 top-4 left-4 right-4 bottom-4 bg-primary/20 rounded-xl blur-xl" />
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 md:py-24 bg-card">
        <div className="container-default text-center">
          <h2 className="text-3xl md:text-4xl font-bold text-foreground">
            Let&apos;s Connect
          </h2>
          <p className="mt-4 text-lg text-muted-foreground max-w-2xl mx-auto">
            I&apos;m always interested in collaborations, research
            opportunities, and discussions about climate science and ML.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <Link href="/contact" className="btn btn-primary">
              Get in Touch
            </Link>
            <Link href="/cv" className="btn btn-secondary">
              Download CV
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
