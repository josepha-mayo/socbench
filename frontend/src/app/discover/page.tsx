"use client";

import { useState, useEffect } from "react";

interface DiscoveredDataset {
  hf_id: string;
  name: string;
  downloads: number;
  likes: number;
  trending_score: number;
  languages: string[];
  tags: string[];
  created_at: string | null;
  qualified: boolean;
  category: string;
  category_label: string;
}

const CAT_LABELS: Record<string, string> = {
  "pretraining-web": "Web",
  "pretraining-code": "Code",
  "pretraining-math": "Math",
  "pretraining-science": "Science",
  "pretraining-books": "Books",
  "pretraining-multilingual": "Multilingual",
  "posttraining-sft": "SFT",
  "posttraining-agent": "Agent Traces",
  "posttraining-preference": "Pref Opt",
  "posttraining-tooluse": "Tool Calling",
  "posttraining-safety": "Safety",
  "posttraining-reasoning": "Reasoning",
  "evaluation": "Eval",
  "task-qa": "QA",
  "task-summarization": "Summarize",
  "task-translation": "Translate",
  "task-classification": "Classify",
  "multimodal": "Multimodal",
};

export default function DiscoverPage() {
  const [datasets, setDatasets] = useState<DiscoveredDataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    setLoading(true);
    fetch("/api/discover")
      .then((r) => r.json())
      .then((data) => {
        setDatasets(data as DiscoveredDataset[]);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, []);

  const filtered = datasets.filter((ds) => {
    if (search && !ds.hf_id.toLowerCase().includes(search.toLowerCase())) return false;
    if (filter === "qualified" && !ds.qualified) return false;
    if (filter === "rejected" && ds.qualified) return false;
    return true;
  });

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark">Dataset Discovery</h2>
        <p className="text-sm font-sans text-arxiv-gray mt-1">
          Daily scan of HuggingFace for new and trending datasets. Filtered by minimum quality thresholds.
        </p>
      </div>

      <div className="flex gap-4 mb-6">
        <input
          type="text"
          placeholder="Search datasets..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="flex-1 px-3 py-2 border border-arxiv-border rounded text-sm font-sans focus:outline-none focus:border-arxiv-red"
        />
        <div className="flex gap-2">
          {["all", "qualified", "rejected"].map((f) => (
            <button key={f} className={`category-tab ${filter === f ? "active" : ""}`} onClick={() => setFilter(f)}>
              {f}
            </button>
          ))}
        </div>
      </div>

      <div className="mb-6 p-3 border border-arxiv-border rounded bg-arxiv-lightgray text-xs font-sans text-arxiv-gray">
        <strong className="text-arxiv-dark">Thresholds:</strong> Downloads ≥ 1,000 · Likes ≥ 10 · Rows ≥ 1,000 · Bytes ≥ 1 MB · Public, non-gated
      </div>

      {loading ? (
        <p className="text-arxiv-gray font-sans">Loading...</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="arxiv-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th className="text-right">Downloads</th>
                <th className="text-right">Likes</th>
                <th className="text-right">Trending</th>
                <th>Category</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((ds) => (
                <tr key={ds.hf_id}>
                  <td>
                    <a href={`/datasets/${encodeURIComponent(ds.hf_id)}`} className="font-sans font-medium text-sm">{ds.hf_id}</a>
                    <div className="flex gap-1 mt-1">
                      {ds.tags.slice(0, 3).map((t) => (
                        <span key={t} className="text-[9px] font-sans bg-arxiv-lightgray border border-arxiv-border px-1 rounded">{t}</span>
                      ))}
                    </div>
                  </td>
                  <td className="text-right font-mono text-xs">{ds.downloads.toLocaleString()}</td>
                  <td className="text-right font-mono text-xs">{ds.likes}</td>
                  <td className="text-right font-mono text-xs">{ds.trending_score.toFixed(1)}</td>
                  <td><span className="text-xs font-sans bg-arxiv-red/10 text-arxiv-red px-2 py-0.5 rounded-full">{ds.category_label}</span></td>
                  <td>{ds.qualified ? <span className="score-high">QUALIFIED</span> : <span className="score-low">REJECTED</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}