"use client";

import { useState, useEffect } from "react";

interface TrainingEntry {
  training_rank: number | null;
  hf_id: string;
  name: string;
  category: string;
  category_label?: string;
  training_score: number | null;
  combined_score: number | null;
  quality: number | null;
  final_val_loss: number | null;
  perplexity: number | null;
  tokens_seen: number | null;
  convergence_steps: number | null;
  loss_curve: number[] | null;
  downloads: number | null;
  downloads_all_time: number | null;
  likes: number | null;
  created_at: string | null;
  status: "trained" | "pending";
  source: "trained" | "trending" | "most_used";
}

function formatNumber(n: number | null) {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

export default function TrainingPage() {
  const [entries, setEntries] = useState<TrainingEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    setError(null);
    fetch("/api/training-leaderboard?limit=100")
      .then((r) => {
        if (!r.ok) throw new Error(`Backend returned ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setEntries(data as TrainingEntry[]);
        setLoading(false);
      })
      .catch((e) => {
        setLoading(false);
        setError(e.message || "Could not connect to backend.");
      });
  }, []);

  const trained = entries.filter((e) => e.status === "trained");
  const pending = entries.filter((e) => e.status === "pending");

  if (loading) {
    return (
      <div className="p-8 border border-arxiv-border rounded bg-arxiv-lightgray text-center">
        <p className="text-arxiv-gray font-sans">Loading training data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-8 border border-red-200 rounded bg-red-50 text-center">
        <p className="text-red-700 font-sans font-medium">{error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark">Training Leaderboard</h2>
        <p className="text-sm font-sans text-arxiv-gray mt-1 italic">
          &ldquo;The unexamined dataset is not worth training on.&rdquo;
        </p>
        <p className="text-xs font-sans text-arxiv-gray mt-1">
          Datasets are trained on a GPT-2 124M model. The training score measures relative validation-loss improvement.
        </p>
      </div>

      <div>
        <h3 className="text-sm font-sans font-bold text-arxiv-dark mb-3 uppercase tracking-wide">
          Pending Training <span className="text-arxiv-gray font-normal">({pending.length})</span>
        </h3>
        {pending.length === 0 ? (
          <p className="text-xs text-arxiv-gray">No pending datasets.</p>
        ) : (
          <div className="overflow-x-auto border border-arxiv-border rounded">
            <table className="w-full text-xs font-sans">
              <thead className="bg-arxiv-lightgray text-arxiv-gray uppercase tracking-wide">
                <tr>
                  <th className="text-left px-3 py-2">#</th>
                  <th className="text-left px-3 py-2">Dataset</th>
                  <th className="text-left px-3 py-2">Category</th>
                  <th className="text-left px-3 py-2">Source</th>
                  <th className="text-right px-3 py-2">Downloads</th>
                </tr>
              </thead>
              <tbody>
                {pending.map((e, i) => (
                  <tr key={e.hf_id} className={i % 2 ? "bg-arxiv-lightgray/40" : ""}>
                    <td className="px-3 py-2 font-mono text-arxiv-gray">{i + 1}</td>
                    <td className="px-3 py-2">
                      <a href={`/datasets/${encodeURIComponent(e.hf_id)}`} className="text-arxiv-link hover:text-arxiv-hover no-underline font-medium">
                        {e.hf_id}
                      </a>
                    </td>
                    <td className="px-3 py-2 text-arxiv-gray">{e.category_label || e.category}</td>
                    <td className="px-3 py-2 text-arxiv-gray capitalize">{e.source}</td>
                    <td className="px-3 py-2 text-right font-mono text-arxiv-gray">{formatNumber(e.downloads_all_time)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div>
        <h3 className="text-sm font-sans font-bold text-arxiv-dark mb-3 uppercase tracking-wide">
          Trained Datasets <span className="text-arxiv-gray font-normal">({trained.length})</span>
        </h3>
        {trained.length === 0 ? (
          <p className="text-xs text-arxiv-gray">No trained datasets yet.</p>
        ) : (
          <div className="overflow-x-auto border border-arxiv-border rounded">
            <table className="w-full text-xs font-sans">
              <thead className="bg-arxiv-lightgray text-arxiv-gray uppercase tracking-wide">
                <tr>
                  <th className="text-left px-3 py-2">#</th>
                  <th className="text-left px-3 py-2">Dataset</th>
                  <th className="text-left px-3 py-2">Category</th>
                  <th className="text-right px-3 py-2">Training Score</th>
                  <th className="text-right px-3 py-2">Val Loss</th>
                  <th className="text-right px-3 py-2">PPL</th>
                  <th className="text-right px-3 py-2">Tokens</th>
                </tr>
              </thead>
              <tbody>
                {trained.map((e, i) => (
                  <tr key={e.hf_id} className={i % 2 ? "bg-arxiv-lightgray/40" : ""}>
                    <td className="px-3 py-2 font-mono text-arxiv-gray">{e.training_rank ?? i + 1}</td>
                    <td className="px-3 py-2">
                      <a href={`/datasets/${encodeURIComponent(e.hf_id)}`} className="text-arxiv-link hover:text-arxiv-hover no-underline font-medium">
                        {e.hf_id}
                      </a>
                    </td>
                    <td className="px-3 py-2 text-arxiv-gray">{e.category_label || e.category}</td>
                    <td className="px-3 py-2 text-right font-mono">{e.training_score != null ? e.training_score.toFixed(1) : "—"}</td>
                    <td className="px-3 py-2 text-right font-mono">{e.final_val_loss != null ? e.final_val_loss.toFixed(4) : "—"}</td>
                    <td className="px-3 py-2 text-right font-mono">{e.perplexity != null ? e.perplexity.toFixed(2) : "—"}</td>
                    <td className="px-3 py-2 text-right font-mono text-arxiv-gray">{e.tokens_seen != null ? (e.tokens_seen / 1e6).toFixed(1) + "M" : "—"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
