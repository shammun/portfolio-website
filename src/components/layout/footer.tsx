import Link from "next/link";
import { Github, Linkedin, Mail, GraduationCap, Twitter } from "lucide-react";

const socialLinks = [
  {
    name: "GitHub",
    href: "https://github.com/shammun",
    icon: Github,
  },
  {
    name: "LinkedIn",
    href: "https://www.linkedin.com/in/shammunul/",
    icon: Linkedin,
  },
  {
    name: "Google Scholar",
    href: "https://scholar.google.com/citations?user=YOUR_ID",
    icon: GraduationCap,
  },
  {
    name: "Twitter",
    href: "https://twitter.com/YOUR_HANDLE",
    icon: Twitter,
  },
  {
    name: "Email",
    href: "mailto:sha_is13@yahoo.com",
    icon: Mail,
  },
];

const footerNavigation = {
  main: [
    { name: "About", href: "/about" },
    { name: "Research", href: "/research" },
    { name: "Projects", href: "/projects" },
    { name: "Contact", href: "/contact" },
  ],
  blog: [
    { name: "AI Paper Implementations", href: "/blog/ai-paper-implementations" },
    { name: "FNO Tutorial", href: "/blog/ai-paper-implementations/fno-paper" },
  ],
};

export function Footer() {
  const currentYear = new Date().getFullYear();

  return (
    <footer className="border-t border-border bg-muted">
      <div className="container-default py-12">
        <div className="grid grid-cols-1 gap-8 md:grid-cols-4">
          {/* Brand */}
          <div className="md:col-span-2">
            <Link
              href="/"
              className="text-lg font-semibold text-foreground hover:text-primary hover:no-underline"
            >
              Shammunul Islam
            </Link>
            <p className="mt-2 text-sm text-muted-foreground max-w-md">
              Graduate student in climate science at George Mason University.
              Building interactive tutorials on AI papers and applying machine
              learning to urban climate problems.
            </p>

            {/* Social Links */}
            <div className="mt-4 flex gap-3">
              {socialLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="p-2 rounded-lg text-muted-foreground hover:text-primary hover:bg-card transition-colors"
                  aria-label={link.name}
                >
                  <link.icon className="h-5 w-5" />
                </a>
              ))}
            </div>
          </div>

          {/* Navigation */}
          <div>
            <h3 className="text-sm font-semibold text-foreground">
              Navigation
            </h3>
            <ul className="mt-3 space-y-2">
              {footerNavigation.main.map((item) => (
                <li key={item.name}>
                  <Link
                    href={item.href}
                    className="text-sm text-muted-foreground hover:text-primary"
                  >
                    {item.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>

          {/* Blog */}
          <div>
            <h3 className="text-sm font-semibold text-foreground">
              Tutorials
            </h3>
            <ul className="mt-3 space-y-2">
              {footerNavigation.blog.map((item) => (
                <li key={item.name}>
                  <Link
                    href={item.href}
                    className="text-sm text-muted-foreground hover:text-primary"
                  >
                    {item.name}
                  </Link>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="mt-10 pt-6 border-t border-border">
          <div className="flex flex-col items-center justify-between gap-4 sm:flex-row">
            <p className="text-sm text-muted-foreground">
              &copy; {currentYear} Shammunul Islam. All rights reserved.
            </p>
            <p className="text-sm text-muted-foreground">
              Built with Next.js &amp; Tailwind CSS
            </p>
          </div>
        </div>
      </div>
    </footer>
  );
}
