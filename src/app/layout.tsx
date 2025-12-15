import type { Metadata } from "next";
import { Inter, JetBrains_Mono } from "next/font/google";
import { ThemeProvider } from "@/components/theme-provider";
import { Header } from "@/components/layout/header";
import { Footer } from "@/components/layout/footer";
import "./globals.css";

const inter = Inter({
  variable: "--font-inter",
  subsets: ["latin"],
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  variable: "--font-jetbrains-mono",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: {
    default: "Shammunul Islam | Climate Science & AI Explorer",
    template: "%s | Shammunul Islam",
  },
  description:
    "Climate scientist and ML researcher. Interactive tutorials on AI papers with runnable code. Specializing in physics-informed machine learning, neural operators, and urban climate prediction.",
  keywords: [
    "climate science",
    "machine learning",
    "neural operators",
    "FNO",
    "urban heat island",
    "remote sensing",
    "AI tutorials",
    "physics-informed ML",
  ],
  authors: [{ name: "Shammunul Islam" }],
  creator: "Shammunul Islam",
  openGraph: {
    type: "website",
    locale: "en_US",
    siteName: "Shammunul Islam",
    title: "Shammunul Islam | Climate Science & AI Explorer",
    description:
      "Climate scientist and ML researcher. Interactive tutorials on AI papers with runnable code.",
  },
  twitter: {
    card: "summary_large_image",
    title: "Shammunul Islam | Climate Science & AI Explorer",
    description:
      "Climate scientist and ML researcher. Interactive tutorials on AI papers with runnable code.",
  },
  robots: {
    index: true,
    follow: true,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}
      >
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <a href="#main-content" className="skip-link">
            Skip to main content
          </a>
          <div className="flex min-h-screen flex-col">
            <Header />
            <main id="main-content" className="flex-1">
              {children}
            </main>
            <Footer />
          </div>
        </ThemeProvider>
      </body>
    </html>
  );
}
