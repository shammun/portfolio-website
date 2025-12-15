import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Exclude large files from serverless function bundles
  outputFileTracingExcludes: {
    "*": [
      "public/books/**/*",
      "public/blog-images/**/*",
      ".next/cache/**/*",
    ],
  },
  // Optimize for Vercel deployment
  experimental: {
    optimizePackageImports: ["lucide-react"],
  },
};

export default nextConfig;
