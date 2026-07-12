import type { Metadata } from "next";
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
                <a href="/" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Leaderboard
                </a>
                <a href="/evals" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Evals
                </a>
                <a href="/trending" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Trending
                </a>
                <a href="/about" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  About
                </a>
              </nav>
            </div>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-6">{children}</main>
        <footer className="border-t border-arxiv-border mt-12 py-6 text-center text-xs font-sans text-arxiv-gray">
          <div className="max-w-3xl mx-auto space-y-2">
            <p>
              <span className="text-arxiv-red font-bold">Soc</span>
              <span className="text-arxiv-dark font-bold">bench</span> — Scientific dataset intelligence. Examine first, train later.
            </p>
            <p className="text-[11px] leading-relaxed max-w-2xl mx-auto">
              The name <em>Socbench</em> honors Socrates, who held that &ldquo;the unexamined life is not worth living.&rdquo;
              We apply the same principle to data: <strong>the unexamined dataset is not worth training on.</strong>
              Just as Socrates examined beliefs through rigorous questioning, Socbench examines datasets through
              multi-dimension scoring, contamination checking, and training impact measurement — because in AI,
              as in philosophy, the quality of what you feed in determines the quality of what comes out.
            </p>
          </div>
        </footer>
      </body>
    </html>
  );
}
