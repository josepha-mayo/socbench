"use client";

import { useState, useEffect } from "react";

interface Dataset {
  rank: number;
  hf_id: string;
  name: string;
  tags: string[];
  quality: number | null;
  diversity: number | null;
  utility: number | null;
  documentation: number | null;
  popularity: number | null;
  freshness: number | null;
  contamination: number | null;
  downloads: number | null;
  likes: number | null;
  category: string;
}

const CATEGORIES = [
  "All",
  "Pretraining",
  "Post-Training",
  "Code",
  "Instruction/SFT",
  "Agent",
  "Evaluation",
  "Multimodal",
];

function scoreBar(score: number | null, width: number = 60) {
  if (score === null) return <span className="text-arxiv-gray text-xs">—</span>;
  const pct = Math.round((score || 0) * 100);
  const color =
    score >= 0.7 ? "bg-green-500" : score >= 0.4 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-1">
      <div className={`flex-1 h-2 rounded-full overflow-hidden bg-arxiv-lightgray ${width ? `w-[${width}px]` : ''}`}>
        <div
          className={`h-full ${color} rounded-full`}
          style={{ width: `${pct}%` }}
        />
      </div>
      <span className="font-mono text-[10px] text-arxiv-gray w-8 text-right">
        {score.toFixed(2)}
      </span>
    </div>
  );
}

function dimensionBar(score: number | null) {
  return scoreBar(score, 60);
}

function formatNumber(n: number | null) {
  if (n === null) return "—";
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toLocaleString();
}

export default function LeaderboardPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [loading, setLoading] = useState(true);
  const [category, setCategory] = useState("All");

  useEffect(() => {
    const mockData: Dataset[] = [
      { rank: 1, hf_id: "bigcode/starcoderdata", name: "starcoderdata", tags: ["code"], quality: 0.87, diversity: 0.72, utility: 0.85, documentation: 0.90, popularity: 0.72, freshness: 0.85, contamination: 0.02, downloads: 45000, likes: 180, category: "pretraining-code" },
      { rank: 2, hf_id: "teknium/OpenHermes-2.5", name: "OpenHermes-2.5", tags: ["instruction", "sft"], quality: 0.84, diversity: 0.78, utility: 0.91, documentation: 0.95, popularity: 0.88, freshness: 0.90, contamination: 0.03, downloads: 89000, likes: 220, category: "posttraining-sft" },
      { rank: 3, hf_id: "HuggingFaceFW/fineweb", name: "fineweb", tags: ["web", "pretraining"], quality: 0.82, diversity: 0.91, utility: 0.80, documentation: 0.88, popularity: 0.98, freshness: 0.95, contamination: 0.02, downloads: 450000, likes: 890, category: "pretraining-web" },
      { rank: 4, hf_id: "bigcode/the-stack-v2", name: "the-stack-v2", tags: ["code", "pretraining"], quality: 0.78, diversity: 0.89, utility: 0.76, documentation: 0.85, popularity: 0.96, freshness: 0.88, contamination: 0.04, downloads: 230000, likes: 520, category: "pretraining-code" },
      { rank: 5, hf_id: "ise-uiuc/Magicoder-OSS-Instruct-110K", name: "OSS-Instruct-110K", tags: ["code", "instruction"], quality: 0.81, diversity: 0.74, utility: 0.88, documentation: 0.82, popularity: 0.55, freshness: 0.80, contamination: 0.03, downloads: 32000, likes: 95, category: "posttraining-sft" },
      { rank: 6, hf_id: "nvidia/Nemotron-SFT-OpenCode-v1", name: "Nemotron-SFT-OpenCode", tags: ["code", "sft"], quality: 0.80, diversity: 0.70, utility: 0.87, documentation: 0.80, popularity: 0.45, freshness: 0.85, contamination: 0.05, downloads: 18000, likes: 65, category: "posttraining-sft" },
      { rank: 7, hf_id: "openai/openai_humaneval", name: "humaneval", tags: ["code", "evaluation"], quality: 0.91, diversity: 0.55, utility: 0.95, documentation: 0.88, popularity: 0.94, freshness: 0.70, contamination: 0.0, downloads: 120000, likes: 340, category: "evaluation" },
      { rank: 8, hf_id: "databricks/databricks-dolly-15k", name: "dolly-15k", tags: ["instruction"], quality: 0.82, diversity: 0.65, utility: 0.85, documentation: 0.92, popularity: 0.91, freshness: 0.75, contamination: 0.03, downloads: 150000, likes: 280, category: "posttraining-sft" },
      { rank: 9, hf_id: "livecodebench/code_generation", name: "livecodebench", tags: ["code", "evaluation"], quality: 0.85, diversity: 0.60, utility: 0.92, documentation: 0.85, popularity: 0.50, freshness: 0.98, contamination: 0.01, downloads: 25000, likes: 88, category: "evaluation" },
      { rank: 10, hf_id: "princeton-nlp/SWE-bench", name: "SWE-bench", tags: ["code", "agent"], quality: 0.88, diversity: 0.58, utility: 0.93, documentation: 0.90, popularity: 0.85, freshness: 0.82, contamination: 0.02, downloads: 78000, likes: 310, category: "evaluation" },
    ];
    setDatasets(mockData);
    setLoading(false);
  }, []);

  const filtered = category === "All" ? datasets : datasets.filter(ds => {
    const cat = ds.category || "";
    const tag = ds.tags.map(t => t.toLowerCase());
    if (category === "Code") return cat.includes("code") || tag.includes("code");
    if (category === "Pretraining") return cat.includes("pretraining");
    if (category === "Post-Training") return cat.includes("posttraining");
    if (category === "Instruction/SFT") return cat.includes("sft") || tag.includes("instruction") || tag.includes("sft");
    if (category === "Agent") return cat.includes("agent");
    return true;
  });

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark">
          Dataset Intelligence
        </h2>
        <p className="text-sm font-sans text-arxiv-gray mt-1 italic">
          "The unexamined dataset is not worth training on."
        </p>
        <p className="text-xs font-sans text-arxiv-gray mt-1">
          Multi-dimension quality scoring: quality, diversity, utility, documentation, freshness, and contamination.
        </p>
      </div>

      <div className="flex flex-wrap gap-2 mb-6">
        {CATEGORIES.map((cat) => (
          <button
            key={cat}
            className={`category-tab ${category === cat ? "active" : ""}`}
            onClick={() => setCategory(cat)}
          >
            {cat}
          </button>
        ))}
      </div>

      {loading ? (
        <p className="text-arxiv-gray font-sans">Loading...</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="arxiv-table">
            <thead>
              <tr>
                <th className="w-10">#</th>
                <th className="min-w-[200px]">Dataset</th>
                <th className="w-[90px]">Quality</th>
                <th className="w-[90px]">Diversity</th>
                <th className="w-[90px]">Utility</th>
                <th className="w-[90px]">Documentation</th>
                <th className="w-[90px]">Freshness</th>
                <th className="text-right w-[80px]">Downloads</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((ds) => (
                <tr key={ds.hf_id}>
                  <td className="font-mono text-arxiv-gray text-xs">{ds.rank}</td>
                  <td>
                    <a
                      href={`/datasets/${encodeURIComponent(ds.hf_id)}`}
                      className="font-sans text-sm font-medium block"
                    >
                      {ds.hf_id}
                    </a>
                    <div className="flex gap-1 mt-1">
                      {ds.tags.slice(0, 3).map((t) => (
                        <span key={t} className="text-[9px] font-sans bg-arxiv-lightgray border border-arxiv-border px-1 rounded text-arxiv-gray">
                          {t}
                        </span>
                      ))}
                    </div>
                  </td>
                  <td>{dimensionBar(ds.quality)}</td>
                  <td>{dimensionBar(ds.diversity)}</td>
                  <td>{dimensionBar(ds.utility)}</td>
                  <td>{dimensionBar(ds.documentation)}</td>
                  <td>{dimensionBar(ds.freshness)}</td>
                  <td className="text-right font-mono text-xs">
                    {formatNumber(ds.downloads)}
                    <div className="text-[9px] text-arxiv-gray">{formatNumber(ds.likes)} likes</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-8 p-4 border border-arxiv-border rounded bg-arxiv-lightgray text-xs font-sans text-arxiv-gray">
        <strong className="text-arxiv-dark">Methodology:</strong> Multi-dimension scoring using
        Gopher quality rules (Rae et al., 2021), FineWeb filters (Penedo et al., NeurIPS 2024),
        DCLM filtering (Li et al., NeurIPS 2024), MinHash deduplication (SlimPajama),
        and 13-gram contamination checking (GPT-3 methodology). Categories are hierarchically
        classified — different metrics apply to code, instruction, agent, and evaluation datasets.
      </div>
    </div>
  );
}