import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Socbench — 'The unexamined dataset is not worth training on.'",
  description: "Scientific dataset intelligence. Examine first, train later.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen">
        <header className="border-b-2 border-arxiv-border bg-white">
          <div className="max-w-6xl mx-auto px-4 py-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h1 className="text-xl font-sans font-bold tracking-tight">
                  <span className="text-arxiv-red">Soc</span>
                  <span className="text-arxiv-dark">bench</span>
                </h1>
                <span className="text-[10px] font-sans text-arxiv-gray italic hidden sm:inline">
                  "The unexamined dataset is not worth training on."
                </span>
              </div>
              <nav className="flex gap-4 text-sm font-sans">
                <a href="/" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Leaderboard
                </a>
                <a href="/discover" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  Discover
                </a>
                <a href="/about" className="text-arxiv-gray hover:text-arxiv-red no-underline transition-colors">
                  About
                </a>
              </nav>
            </div>
          </div>
        </header>
        <main className="max-w-6xl mx-auto px-4 py-6">{children}</main>
        <footer className="border-t border-arxiv-border mt-12 py-4 text-center text-xs font-sans text-arxiv-gray">
          Socbench — Scientific dataset intelligence. Examine first, train later.
        </footer>
      </body>
    </html>
  );
}