"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { Menu, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { ThemeToggle } from "@/components/theme-toggle";

const navigation = [
  { name: "Home", href: "/" },
  { name: "About", href: "/about" },
  { name: "AI Papers", href: "/blog/ai-paper-implementations", highlight: true },
  { name: "Research", href: "/research" },
  { name: "Projects", href: "/projects" },
  { name: "CV", href: "/cv" },
  { name: "Contact", href: "/contact" },
];

export function Header() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 w-full border-b border-border bg-card/80 backdrop-blur-md shadow-soft">
      <nav className="container-default flex h-16 items-center justify-between">
        {/* Logo */}
        <Link
          href="/"
          className="flex items-center gap-2 text-lg font-bold text-foreground hover:text-primary hover:no-underline transition-colors"
        >
          <span className="hidden sm:inline">Shammunul Islam</span>
          <span className="sm:hidden">SI</span>
        </Link>

        {/* Desktop Navigation */}
        <div className="hidden md:flex md:items-center md:gap-1">
          {navigation.map((item) => {
            const isActive = pathname === item.href ||
              (item.href !== "/" && pathname.startsWith(item.href));

            return (
              <Link
                key={item.name}
                href={item.href}
                className={cn(
                  "px-3 py-2 text-sm font-medium rounded-lg transition-colors hover:no-underline",
                  isActive
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted",
                  item.highlight && !isActive && "text-teal"
                )}
              >
                {item.name}
              </Link>
            );
          })}
        </div>

        {/* Right side: Theme toggle + Mobile menu button */}
        <div className="flex items-center gap-2">
          <ThemeToggle />

          {/* Mobile menu button */}
          <button
            type="button"
            className="md:hidden p-2 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            aria-expanded={mobileMenuOpen}
            aria-label="Toggle navigation menu"
          >
            {mobileMenuOpen ? (
              <X className="h-6 w-6" />
            ) : (
              <Menu className="h-6 w-6" />
            )}
          </button>
        </div>
      </nav>

      {/* Mobile Navigation */}
      {mobileMenuOpen && (
        <div className="md:hidden border-t border-border bg-card">
          <div className="container-default py-4 space-y-1">
            {navigation.map((item) => {
              const isActive = pathname === item.href ||
                (item.href !== "/" && pathname.startsWith(item.href));

              return (
                <Link
                  key={item.name}
                  href={item.href}
                  className={cn(
                    "block px-4 py-3 text-base font-medium rounded-lg transition-colors hover:no-underline",
                    isActive
                      ? "text-primary bg-primary/10"
                      : "text-muted-foreground hover:text-foreground hover:bg-muted",
                    item.highlight && !isActive && "text-teal"
                  )}
                  onClick={() => setMobileMenuOpen(false)}
                >
                  {item.name}
                </Link>
              );
            })}
          </div>
        </div>
      )}
    </header>
  );
}
