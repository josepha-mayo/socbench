"use client";

import { useState, useEffect } from "react";

interface TrendingDataset {
  hf_id: string;
  name: string;
  downloads: number;
  likes: number;
  trending_score: number;
  languages: string[];
  tags: string[];
  created_at: string | null;
  category: string;
  category_label: string;
  qualified: boolean;
}

const CAT_LABELS: Record<string, string> = {
  "pretraining-web": "Web",
  "pretraining-code": "Code",
  "pretraining-math": "Math",
  "pretraining-science": "Science",
  "pretraining-books": "Books",
  "pretraining-multilingual": "Multilingual",
  "posttraining-sft": "SFT",
  "posttraining-agent": "Agent",
  "posttraining-preference": "Pref Opt",
  "posttraining-tooluse": "Tool Call",
  "posttraining-safety": "Safety",
  "posttraining-reasoning": "Reasoning",
  "evaluation": "Eval",
  "task-qa": "QA",
  "task-summarization": "Summarize",
  "task-translation": "Translate",
  "task-classification": "Classify",
  "multimodal": "Multimodal",
};

function formatNumber(n: number | null) {
  if (n === null || n === undefined) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

function formatDate(dateStr: string | null) {
  if (!dateStr) return "—";
  try {
    const d = new Date(dateStr);
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric" });
  } catch {
    return "—";
  }
}

export default function TrendingPage() {
  const [datasets, setDatasets] = useState<TrendingDataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");
  const [days, setDays] = useState(7);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/trending?limit=50&days=${days}`)
      .then((r) => {
        if (!r.ok) throw new Error(`Backend returned ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setDatasets(data || []);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message || "Could not connect to backend.");
        setLoading(false);
      });
  }, [days]);

  const filtered = datasets.filter((ds) => {
    if (search && !ds.hf_id.toLowerCase().includes(search.toLowerCase())) return false;
    if (filter === "qualified" && !ds.qualified) return false;
    if (filter === "rejected" && ds.qualified) return false;
    return true;
  });

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark">
          Trending Datasets
        </h2>
        <p className="text-sm font-sans text-arxiv-gray mt-1">
          Real-time trending datasets from HuggingFace. Updated dynamically — not hardcoded.
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-3 mb-4">
        <input
          type="text"
          placeholder="Search datasets..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 min-w-[200px] px-3 py-2 border border-arxiv-border rounded text-sm font-sans focus:outline-none focus:border-arxiv-red"
        />
        <div className="flex gap-1.5">
          {["all", "qualified", "rejected"].map((f) => (
            <button key={f} className={`category-tab ${filter === f ? "active" : ""}`} onClick={() => setFilter(f)}>
              {f === "all" ? "All" : f === "qualified" ? "Qualified" : "Rejected"}
            </button>
          ))}
        </div>
        <div className="flex gap-1.5">
          {[1, 7, 30].map((d) => (
            <button
              key={d}
              className={`category-tab ${days === d ? "active" : ""}`}
              onClick={() => setDays(d)}
            >
              {d === 1 ? "24h" : d === 7 ? "7d" : "30d"}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-4 p-3 border border-arxiv-border rounded bg-arxiv-lightgray text-xs font-sans text-arxiv-gray">
        <strong className="text-arxiv-dark">Live data:</strong> Fetched from HuggingFace API sorted by trending score.
        Qualification: Downloads &ge; 1K, Likes &ge; 10, public, non-gated.
      </div>

      {loading ? (
        <div className="p-8 border border-arxiv-border rounded bg-arxiv-lightgray text-center">
          <p className="text-arxiv-gray font-sans">Fetching trending datasets from HuggingFace...</p>
        </div>
      ) : error ? (
        <div className="p-8 border border-red-200 rounded bg-red-50 text-center">
          <p className="text-red-700 font-sans font-medium">{error}</p>
        </div>
      ) : filtered.length === 0 ? (
        <div className="p-8 border border-arxiv-border rounded bg-arxiv-lightgray text-center">
          <p className="text-arxiv-gray font-sans">No trending datasets found for this period.</p>
        </div>
      ) : (
        <div className="overflow-x-auto border border-arxiv-border rounded">
          <table className="arxiv-table">
            <thead>
              <tr>
                <th className="w-10">#</th>
                <th className="min-w-[180px]">Dataset</th>
                <th className="w-[90px]">Category</th>
                <th className="text-right w-[80px]">Downloads</th>
                <th className="text-right w-[60px]">Likes</th>
                <th className="text-right w-[70px]">Trending</th>
                <th className="w-[90px]">Created</th>
                <th className="w-[80px]">Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((ds, i) => (
                <tr key={ds.hf_id}>
                  <td className="font-mono text-arxiv-gray text-xs">{i + 1}</td>
                  <td>
                    <div className="flex items-center gap-1.5">
                      <a
                        href={`/datasets/${encodeURIComponent(ds.hf_id)}`}
                        className="font-sans text-sm font-medium hover:text-arxiv-red no-underline"
                      >
                        {ds.hf_id}
                      </a>
                      <a
                        href={`https://huggingface.co/datasets/${ds.hf_id}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-[10px] text-arxiv-link hover:text-arxiv-hover no-underline"
                      >
                        HF↗
                      </a>
                    </div>
                    <div className="flex gap-1 mt-0.5">
                      {(ds.tags || []).slice(0, 3).map((t) => (
                        <span key={t} className="text-[9px] font-sans bg-arxiv-lightgray border border-arxiv-border px-1 rounded text-arxiv-gray">
                          {t}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>
                    <span className="text-[10px] font-sans bg-arxiv-red/10 text-arxiv-red px-2 py-0.5 rounded-full whitespace-nowrap">
                      {CAT_LABELS[ds.category] || ds.category_label || ds.category || "—"}
                    </span>
                  </td>
                  <td className="text-right font-mono text-xs">{formatNumber(ds.downloads)}</td>
                  <td className="text-right font-mono text-xs">{ds.likes}</td>
                  <td className="text-right font-mono text-xs font-bold text-arxiv-dark">{ds.trending_score.toFixed(1)}</td>
                  <td className="font-mono text-[10px] text-arxiv-gray">{formatDate(ds.created_at)}</td>
                  <td>
                    {ds.qualified ? (
                      <span className="score-high">QUALIFIED</span>
                    ) : (
                      <span className="score-low">REJECTED</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
