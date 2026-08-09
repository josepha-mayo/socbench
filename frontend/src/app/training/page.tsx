"use client";

import { useState, useEffect } from "react";
import { TrainingImpact, type TrainingEntry } from "@/components/training-impact";

export default function TrainingPage() {
  const [entries, setEntries] = useState<TrainingEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [requestId, setRequestId] = useState(0);

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
  }, [requestId]);

  return (
    <div className="space-y-8">
      <header>
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark">Training Leaderboard</h2>
        <p className="text-sm font-sans text-arxiv-gray mt-1 italic">
          &ldquo;The unexamined dataset is not worth training on.&rdquo;
        </p>
        <p className="text-xs font-sans text-arxiv-gray mt-1">
          GPT-2 124M proxy runs, scored only when final validation loss improves.
        </p>
      </header>
      <TrainingImpact entries={entries} loading={loading} error={error} onRetry={() => setRequestId((value) => value + 1)} />
    </div>
  );
}
