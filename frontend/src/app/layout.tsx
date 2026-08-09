import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "Socbench — Scientific Dataset Intelligence",
  description: "Examine first, train later. Multi-dimension dataset quality scoring, contamination checking, and training impact measurement.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b-2 border-arxiv-border bg-white sticky top-0 z-50">
          <div className="max-w-6xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h1 className="text-xl font-sans font-bold tracking-tight">
                  <span className="text-arxiv-red">Soc</span>
                  <span className="text-arxiv-dark">bench</span>
                </h1>
                <span className="text-[10px] font-sans text-arxiv-gray italic hidden sm:inline">
                  &ldquo;The unexamined dataset is not worth training on.&rdquo;
                </span>
              </div>
              <nav className="flex gap-4 text-sm font-sans">
                <Link href="/" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Leaderboard
                </Link>
                <Link href="/evals" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Evals
                </Link>
                <Link href="/trending" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Trending
                </Link>
                <Link href="/about" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  About
                </Link>
              </nav>
            </div>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-6">{children}</main>
        <footer className="border-t border-arxiv-border mt-12 px-4 py-5 text-center text-xs font-sans text-arxiv-gray">
          <p>Scientific dataset intelligence. Examine first, train later.</p>
        </footer>
      </body>
    </html>
  );
}
