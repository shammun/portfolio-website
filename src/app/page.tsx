import Link from "next/link";
import {
  ArrowRight,
  BookOpen,
  Code,
  FlaskConical,
  Sparkles,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="flex flex-col">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        {/* Background gradient */}
        <div className="absolute inset-0 gradient-hero opacity-5" />

        <div className="container-default relative py-20 md:py-32">
          <div className="max-w-3xl">
            <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold tracking-tight text-foreground">
              Hi, I&apos;m{" "}
              <span className="text-primary">Shammunul Islam</span>
            </h1>

            <p className="mt-6 text-xl md:text-2xl text-muted-foreground leading-relaxed">
              MS Candidate in Climate Dynamics · Columbia Climate &amp; Society Alum
            </p>

            <p className="mt-4 text-lg text-muted-foreground leading-relaxed max-w-2xl">
              I combine{" "}
              <strong className="text-foreground">
                machine learning
              </strong>{" "}
              with{" "}
              <strong className="text-foreground">
                satellite data
              </strong>{" "}
              to understand how cities heat up. Author of two books on geospatial
              analysis, now researching high-resolution urban heat island prediction.
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

      {/* Featured Section */}
      <section className="py-16 md:py-24 bg-muted">
        <div className="container-default">
          <div className="text-center max-w-2xl mx-auto mb-12">
            <h2 className="text-3xl md:text-4xl font-bold text-foreground">
              What I Do
            </h2>
            <p className="mt-4 text-muted-foreground">
              Bridging the gap between cutting-edge AI research and climate
              science applications
            </p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
            {/* Card 1: AI Paper Tutorials */}
            <div className="card group">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-primary/10 text-primary">
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
                className="inline-flex items-center gap-2 text-primary font-medium group-hover:gap-3 transition-all"
              >
                Start Learning
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            {/* Card 2: Research */}
            <div className="card group">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-accent/10 text-accent">
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
                className="inline-flex items-center gap-2 text-primary font-medium group-hover:gap-3 transition-all"
              >
                View Publications
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>

            {/* Card 3: Projects */}
            <div className="card group">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-blue-500/10 text-blue-500">
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
                className="inline-flex items-center gap-2 text-primary font-medium group-hover:gap-3 transition-all"
              >
                Browse Projects
                <ArrowRight className="h-4 w-4" />
              </Link>
            </div>
          </div>
        </div>
      </section>

      {/* Featured Tutorial */}
      <section className="py-16 md:py-24">
        <div className="container-default">
          <div className="grid gap-8 lg:grid-cols-2 items-center">
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/10 text-primary text-sm font-medium mb-4">
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
                    <span className="flex-shrink-0 w-5 h-5 rounded-full bg-green-500/20 text-green-500 flex items-center justify-center text-xs">
                      ✓
                    </span>
                    {item}
                  </li>
                ))}
              </ul>

              <div className="mt-8">
                <Link
                  href="/blog/ai-paper-implementations/fno-paper"
                  className="btn btn-primary"
                >
                  Start the Tutorial
                  <ArrowRight className="h-5 w-5" />
                </Link>
              </div>
            </div>

            {/* Code preview card */}
            <div className="relative">
              <div className="bg-[#1E1E1E] rounded-xl p-6 shadow-xl">
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
              {/* Decorative elements */}
              <div className="absolute -z-10 top-4 left-4 right-4 bottom-4 bg-primary/20 rounded-xl blur-xl" />
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-16 md:py-24 bg-muted">
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
