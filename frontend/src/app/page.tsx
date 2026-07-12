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
  contaminated: boolean;
  repetition_pct: number | null;
  combined_score: number | null;
  training_score: number | null;
  downloads: number | null;
  likes: number | null;
  category: string;
  created_at: string | null;
}

interface TrainingEntry {
  training_rank: number;
  hf_id: string;
  name: string;
  category: string;
  training_score: number | null;
  combined_score: number | null;
  quality: number | null;
  final_val_loss: number | null;
  perplexity: number | null;
  tokens_seen: number | null;
  convergence_steps: number | null;
  loss_curve: number[] | null;
  downloads: number | null;
  likes: number | null;
}

const CATEGORIES = [
  "All",
  "Pretraining",
  "SFT",
  "Preference Opt.",
  "Agent Traces",
  "Tool Calling",
  "Reasoning",
  "Task",
  "Multimodal",
  "Safety",
  "Evaluation",
];

const CAT_LABEL: Record<string, string> = {
  "pretraining-web": "Web",
  "pretraining-code": "Code",
  "pretraining-math": "Math",
  "pretraining-science": "Science",
  "pretraining-books": "Books",
  "pretraining-multilingual": "Multilingual",
  "posttraining-sft": "SFT",
  "posttraining-preference": "Pref Opt",
  "posttraining-tooluse": "Tool Call",
  "posttraining-agent": "Agent Traces",
  "posttraining-safety": "Safety",
  "posttraining-reasoning": "Reasoning",
  "evaluation": "Eval",
  "task-classification": "Classify",
  "task-translation": "Translate",
  "task-qa": "QA",
  "task-summarization": "Summarize",
  "multimodal": "Multimodal",
};

const SORT_OPTIONS = [
  { key: "quality", label: "Quality" },
  { key: "diversity", label: "Diversity" },
  { key: "combined_score", label: "Combined" },
  { key: "training_score", label: "Training" },
  { key: "popularity", label: "Popularity" },
  { key: "freshness", label: "Freshness" },
];

const VIEW_MODES = ["Table", "Chart"];

function score100(score: number | null) {
  if (score === null || score === undefined) return <span className="text-arxiv-gray text-xs">—</span>;
  const val = Math.round(score);
  const color = val >= 70 ? "text-green-700" : val >= 40 ? "text-yellow-700" : "text-red-700";
  return <span className={`font-mono text-xs font-bold ${color}`}>{val}</span>;
}

function scoreBar(score: number | null) {
  if (score === null || score === undefined) return <span className="text-arxiv-gray text-xs">—</span>;
  const pct = Math.round(score);
  const color = pct >= 70 ? "bg-green-500" : pct >= 40 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-12 h-2 rounded-full overflow-hidden bg-arxiv-lightgray">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[10px] font-bold text-arxiv-dark w-7">{pct}</span>
    </div>
  );
}

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

function categorySlug(cat: string): string {
  const map: Record<string, string> = {
    "Pretraining": "pretraining",
    "SFT": "sft",
    "Preference Opt.": "preference",
    "Agent Traces": "agent",
    "Tool Calling": "tooluse",
    "Reasoning": "reasoning",
    "Task": "task",
    "Multimodal": "multimodal",
    "Safety": "safety",
    "Evaluation": "evaluation",
  };
  return map[cat] || "";
}

function LossCurve({ losses }: { losses: number[] | null }) {
  if (!losses || !Array.isArray(losses) || losses.length === 0) return null;
  const clean = losses.filter((x) => typeof x === "number" && !Number.isNaN(x));
  if (clean.length === 0) return null;
  const maxLoss = Math.max(...clean);
  const minLoss = Math.min(...clean);
  const range = maxLoss - minLoss || 1;
  return (
    <div className="flex items-end gap-px h-12 mt-1">
      {clean.map((l, i) => {
        const h = ((l - minLoss) / range) * 100;
        return (
          <div
            key={i}
            className="flex-1 bg-arxiv-red rounded-t-sm min-w-[2px] opacity-80"
            style={{ height: `${Math.max(h, 2)}%` }}
            title={`Step ${i}: ${l.toFixed(4)}`}
          />
        );
      })}
    </div>
  );
}

export default function LeaderboardPage() {
  const [datasets, setDatasets] = useState<Dataset[]>([]);
  const [trainingData, setTrainingData] = useState<TrainingEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [category, setCategory] = useState("All");
  const [sortBy, setSortBy] = useState("quality");
  const [viewMode, setViewMode] = useState("Table");
  const [showEvalModal, setShowEvalModal] = useState(false);
  const [evalHfId, setEvalHfId] = useState("");
  const [evalVisibility, setEvalVisibility] = useState("public");
  const [evalEmail, setEvalEmail] = useState("");
  const [evalName, setEvalName] = useState("");
  const [evalStatus, setEvalStatus] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  useEffect(() => {
    setLoading(true);
    setError(null);
    const categoryParam = category === "All" ? "" : categorySlug(category);
    fetch(`/api/leaderboard?category=${encodeURIComponent(categoryParam)}&sort=${sortBy}&limit=200`)
      .then((r) => {
        if (!r.ok) throw new Error(`Backend returned ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setDatasets(data as Dataset[]);
        setLoading(false);
      })
      .catch((e) => {
        setLoading(false);
        setError(e.message || "Could not connect to backend.");
      });
  }, [category, sortBy]);

  useEffect(() => {
    fetch("/api/training-leaderboard?limit=100")
      .then((r) => r.ok ? r.json() : [])
      .then((data) => setTrainingData(data || []))
      .catch(() => {});
  }, []);

  const stats = (() => {
    if (!datasets.length) return null;
    const n = datasets.length;
    const avgQ = Math.round(datasets.reduce((s, d) => s + (d.quality ?? 0), 0) / n);
    const contaminated = datasets.filter(d => d.contaminated).length;
    const highRep = datasets.filter(d => (d.repetition_pct ?? 0) > 20).length;
    const trained = datasets.filter(d => (d.training_score ?? null) !== null).length;
    return { n, avgQ, contaminated, highRep, trained };
  })();

  const filtered = datasets.filter(ds => {
    if (search && !ds.hf_id.toLowerCase().includes(search.toLowerCase())) return false;
    if (category === "All") return true;
    const cat = ds.category || "";
    if (category === "SFT") return cat.includes("sft");
    if (category === "Pretraining") return cat.includes("pretraining");
    if (category === "Preference Opt.") return cat.includes("preference");
    if (category === "Agent Traces") return cat.includes("agent");
    if (category === "Tool Calling") return cat.includes("tooluse");
    if (category === "Reasoning") return cat.includes("reasoning");
    if (category === "Task") return cat.startsWith("task-");
    if (category === "Multimodal") return cat.includes("multimodal");
    if (category === "Safety") return cat.includes("safety");
    if (category === "Evaluation") return cat.includes("evaluation");
    return true;
  });

  const NOTIFY_EMAIL = "ayandajoseph390@gmail.com";

  async function submitEvalRequest() {
    if (!evalHfId.trim()) return;
    setEvalStatus(null);
    try {
      // 1. Persist to backend DB
      const resp = await fetch("/api/request-evaluation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          hf_id: evalHfId.trim(),
          visibility: evalVisibility,
          requester_email: evalEmail.trim(),
          requester_name: evalName.trim(),
        }),
      });
      const data = await resp.json();
      const requestId = data.request_id || "?";

      // 2. Open user's email client with pre-filled message to the maintainer
      const subject = `[Socbench] Evaluation Request #${requestId}: ${evalHfId.trim()}`;
      const body = [
        `Request ID: ${requestId}`,
        `Dataset: ${evalHfId.trim()}`,
        `Visibility: ${evalVisibility}`,
        `Requester: ${evalName.trim() || "N/A"}`,
        `Requester Email: ${evalEmail.trim() || "N/A"}`,
        ``,
        `Notes:`,
        ``,
      ].join("\n");
      const mailto = `mailto:${NOTIFY_EMAIL}?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`;
      window.open(mailto, "_blank");

      setEvalStatus(`Request #${requestId} saved. Your email client should open — please send it to notify the maintainer.`);
    } catch {
      setEvalStatus("Failed to submit request.");
    }
  }

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark">
          Dataset Intelligence
        </h2>
        <p className="text-sm font-sans text-arxiv-gray mt-1 italic">
          &ldquo;The unexamined dataset is not worth training on.&rdquo;
        </p>
        <p className="text-xs font-sans text-arxiv-gray mt-1">
          Multi-dimension quality scoring (0-100): quality, diversity, utility, documentation, freshness, contamination, repetition, and training impact.
        </p>
      </div>

      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Datasets</div>
            <div className="text-xl font-mono font-bold text-arxiv-dark">{stats.n}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Avg Quality</div>
            <div className="text-xl font-mono font-bold text-arxiv-dark">{stats.avgQ}<span className="text-xs text-arxiv-gray">/100</span></div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Training Evals</div>
            <div className="text-xl font-mono font-bold text-arxiv-dark">{stats.trained}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Contaminated</div>
            <div className={`text-xl font-mono font-bold ${stats.contaminated > 0 ? "text-red-700" : "text-green-700"}`}>{stats.contaminated}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">High Repetition</div>
            <div className={`text-xl font-mono font-bold ${stats.highRep > 5 ? "text-red-700" : stats.highRep > 0 ? "text-yellow-700" : "text-green-700"}`}>{stats.highRep}</div>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 mb-4">
        <div className="flex flex-wrap gap-1.5">
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
        <div className="flex items-center gap-2">
          <input
            type="text"
            placeholder="Search datasets..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="text-xs font-sans border border-arxiv-border rounded px-3 py-1.5 bg-white w-40 focus:outline-none focus:border-arxiv-red"
          />
          <span className="text-xs font-sans text-arxiv-gray">Sort:</span>
          <select
            className="text-xs font-sans border border-arxiv-border rounded px-2 py-1.5 bg-white"
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value)}
          >
            {SORT_OPTIONS.map((opt) => (
              <option key={opt.key} value={opt.key}>{opt.label}</option>
            ))}
          </select>
          <div className="flex gap-1.5 border-l border-arxiv-border pl-2">
            {VIEW_MODES.map((v) => (
              <button
                key={v}
                className={`category-tab ${viewMode === v ? "active" : ""}`}
                onClick={() => setViewMode(v)}
              >
                {v}
              </button>
            ))}
          </div>
          <button
            className="text-xs font-sans bg-arxiv-red text-white px-3 py-1.5 rounded font-medium hover:bg-arxiv-darkred transition-colors"
            onClick={() => { setShowEvalModal(true); setEvalStatus(null); }}
          >
            Request Evaluation
          </button>
        </div>
      </div>

      {loading ? (
        <div className="p-8 border border-arxiv-border rounded bg-arxiv-lightgray text-center">
          <p className="text-arxiv-gray font-sans">Loading datasets...</p>
        </div>
      ) : error ? (
        <div className="p-8 border border-red-200 rounded bg-red-50 text-center">
          <p className="text-red-700 font-sans font-medium">{error}</p>
          <p className="text-red-600 text-xs font-sans mt-2">
            Make sure the backend is running on <code className="bg-white px-1 rounded">http://localhost:8000</code> or the API is reachable.
          </p>
        </div>
      ) : datasets.length === 0 ? (
        <div className="p-8 border border-arxiv-border rounded bg-arxiv-lightgray text-center">
          <p className="text-arxiv-gray font-sans">No datasets found. Start the backend to load the leaderboard.</p>
        </div>
      ) : viewMode === "Table" ? (
        <div className="overflow-x-auto border border-arxiv-border rounded">
          <table className="arxiv-table">
            <thead>
              <tr>
                <th className="w-10">#</th>
                <th className="min-w-[180px]">Dataset</th>
                <th className="w-[90px]">Category</th>
                <th className="w-[70px]">Quality</th>
                <th className="w-[70px]">Diversity</th>
                <th className="w-[70px]">Combined</th>
                <th className="w-[70px]">Training</th>
                <th className="w-[60px]">Repetition</th>
                <th className="w-[60px]">Contam.</th>
                <th className="w-[90px]">Created</th>
                <th className="text-right w-[80px]">Downloads</th>
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
                        title="View on HuggingFace"
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
                      {CAT_LABEL[ds.category || ""] || ds.category || "—"}
                    </span>
                  </td>
                  <td>{scoreBar(ds.quality)}</td>
                  <td>{scoreBar(ds.diversity)}</td>
                  <td>{score100(ds.combined_score)}</td>
                  <td>{score100(ds.training_score)}</td>
                  <td className="font-mono text-xs">
                    {ds.repetition_pct != null ? (
                      <span className={ds.repetition_pct > 20 ? "text-red-700 font-bold" : ds.repetition_pct > 5 ? "text-yellow-700" : "text-green-700"}>
                        {ds.repetition_pct.toFixed(1)}%
                      </span>
                    ) : "—"}
                  </td>
                  <td className="font-mono text-xs">
                    {ds.contaminated ? (
                      <span className="text-red-700 font-bold" title={`Contamination: ${ds.contamination ?? 0}%`}>
                        YES
                      </span>
                    ) : (
                      <span className="text-green-700">clean</span>
                    )}
                  </td>
                  <td className="font-mono text-[10px] text-arxiv-gray">
                    {formatDate(ds.created_at)}
                  </td>
                  <td className="text-right font-mono text-xs">
                    {formatNumber(ds.downloads)}
                    <div className="text-[9px] text-arxiv-gray">{formatNumber(ds.likes)} likes</div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        /* Chart view: horizontal bar chart of combined scores */
        <div className="border border-arxiv-border rounded p-4 bg-white">
          <h3 className="text-sm font-serif font-bold mb-4">Combined Score by Dataset (higher = better)</h3>
          <div className="space-y-1 max-h-[600px] overflow-y-auto">
            {filtered.map((ds, i) => (
              <div key={ds.hf_id} className="flex items-center gap-3">
                <div className="w-48 text-xs font-sans font-medium text-right truncate" title={ds.hf_id}>
                  {i + 1}. <a href={`/datasets/${encodeURIComponent(ds.hf_id)}`} className="hover:text-arxiv-red no-underline">{ds.hf_id}</a>
                </div>
                <div className="flex-1 h-5 bg-arxiv-lightgray rounded-full overflow-hidden relative">
                  <div
                    className={`h-full rounded-full ${(ds.combined_score ?? 0) >= 70 ? "bg-green-500" : (ds.combined_score ?? 0) >= 40 ? "bg-yellow-500" : "bg-red-500"}`}
                    style={{ width: `${ds.combined_score ?? 0}%` }}
                  />
                  <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] font-mono font-bold text-arxiv-dark">
                    {Math.round(ds.combined_score ?? 0)}
                  </span>
                </div>
                {ds.training_score != null && (
                  <div className="w-12 text-[10px] font-mono text-blue-700 font-bold" title="Training score">
                    T:{Math.round(ds.training_score)}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training Impact Leaderboard — separate table below */}
      {trainingData.length > 0 && (
        <div className="mt-12">
          <h3 className="section-header">Training Impact Leaderboard — GPT-2 124M (ranked by training score, 1 = best)</h3>
          <p className="text-xs font-sans text-arxiv-gray mb-4">
            Datasets trained from scratch with GPT-2 124M. Training score = normalized relative quality (lower loss = better).
            Perplexity = e^(val_loss). Loss curves show validation loss over training steps.
          </p>
          <div className="overflow-x-auto border border-arxiv-border rounded">
            <table className="arxiv-table">
              <thead>
                <tr>
                  <th className="w-10">#</th>
                  <th className="min-w-[180px]">Dataset</th>
                  <th className="w-[90px]">Category</th>
                  <th className="w-[70px]">Train Score</th>
                  <th className="w-[70px]">Combined</th>
                  <th className="w-[70px]">Quality</th>
                  <th className="w-[80px]">Val Loss</th>
                  <th className="w-[80px]">Perplexity</th>
                  <th className="w-[80px]">Tokens</th>
                  <th className="w-[150px]">Loss Curve</th>
                </tr>
              </thead>
              <tbody>
                {trainingData.map((t) => (
                  <tr key={t.hf_id}>
                    <td className="font-mono text-arxiv-gray text-xs font-bold">{t.training_rank}</td>
                    <td>
                      <div className="flex items-center gap-1.5">
                        <a
                          href={`/datasets/${encodeURIComponent(t.hf_id)}`}
                          className="font-sans text-sm font-medium hover:text-arxiv-red no-underline"
                        >
                          {t.hf_id}
                        </a>
                        <a
                          href={`https://huggingface.co/datasets/${t.hf_id}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-[10px] text-arxiv-link hover:text-arxiv-hover no-underline"
                        >
                          HF↗
                        </a>
                      </div>
                    </td>
                    <td>
                      <span className="text-[10px] font-sans bg-arxiv-red/10 text-arxiv-red px-2 py-0.5 rounded-full whitespace-nowrap">
                        {CAT_LABEL[t.category || ""] || t.category || "—"}
                      </span>
                    </td>
                    <td>{scoreBar(t.training_score)}</td>
                    <td>{score100(t.combined_score)}</td>
                    <td>{score100(t.quality)}</td>
                    <td className="font-mono text-xs font-bold text-arxiv-dark">
                      {t.final_val_loss != null ? t.final_val_loss.toFixed(4) : "—"}
                    </td>
                    <td className="font-mono text-xs text-arxiv-dark">
                      {t.perplexity != null ? t.perplexity.toFixed(2) : "—"}
                    </td>
                    <td className="font-mono text-[10px] text-arxiv-gray">
                      {t.tokens_seen != null ? formatNumber(t.tokens_seen) : "—"}
                    </td>
                    <td>
                      <LossCurve losses={t.loss_curve} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      <div className="mt-8 info-banner">
        <strong className="text-arxiv-dark">Methodology:</strong> Multi-dimension scoring using
        Gopher quality rules, FineWeb filters, DCLM filtering, MinHash deduplication,
        13-gram contamination checking, and GPT-2 124M training impact measurement.
        Scores are on a 0-100 scale. Contamination flag triggers when benchmark overlap exceeds 1%.
        Repetition % measures exact-row duplication. Categories are hierarchically classified.
        Training score = normalized relative quality (avg_loss / this_loss, scaled 0-100).
      </div>

      {showEvalModal && (
        <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-50" onClick={() => setShowEvalModal(false)}>
          <div className="bg-white border border-arxiv-border rounded-lg p-6 max-w-md w-full mx-4" onClick={(e) => e.stopPropagation()}>
            <h3 className="text-lg font-serif font-bold text-arxiv-dark mb-4">Request Dataset Evaluation</h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs font-sans text-arxiv-gray block mb-1">HuggingFace Dataset ID</label>
                <input
                  type="text"
                  placeholder="org/dataset-name"
                  className="w-full text-sm font-sans border border-arxiv-border rounded px-3 py-2"
                  value={evalHfId}
                  onChange={(e) => setEvalHfId(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs font-sans text-arxiv-gray block mb-1">Your Email (optional — for notification)</label>
                <input
                  type="email"
                  placeholder="you@example.com"
                  className="w-full text-sm font-sans border border-arxiv-border rounded px-3 py-2"
                  value={evalEmail}
                  onChange={(e) => setEvalEmail(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs font-sans text-arxiv-gray block mb-1">Your Name (optional)</label>
                <input
                  type="text"
                  placeholder="Your name"
                  className="w-full text-sm font-sans border border-arxiv-border rounded px-3 py-2"
                  value={evalName}
                  onChange={(e) => setEvalName(e.target.value)}
                />
              </div>
              <div>
                <label className="text-xs font-sans text-arxiv-gray block mb-1">Visibility</label>
                <div className="flex gap-3">
                  <label className="flex items-center gap-1.5 text-sm font-sans">
                    <input
                      type="radio"
                      name="visibility"
                      value="public"
                      checked={evalVisibility === "public"}
                      onChange={(e) => setEvalVisibility(e.target.value)}
                    />
                    Public (results published on leaderboard)
                  </label>
                  <label className="flex items-center gap-1.5 text-sm font-sans">
                    <input
                      type="radio"
                      name="visibility"
                      value="private"
                      checked={evalVisibility === "private"}
                      onChange={(e) => setEvalVisibility(e.target.value)}
                    />
                    Private (results shared with you only)
                  </label>
                </div>
              </div>
              {evalStatus && (
                <p className="text-xs font-sans text-green-700 bg-green-50 border border-green-200 rounded px-3 py-2">
                  {evalStatus}
                </p>
              )}
              <div className="flex gap-2 justify-end pt-2">
                <button
                  className="text-sm font-sans px-4 py-2 border border-arxiv-border rounded text-arxiv-gray hover:bg-arxiv-lightgray"
                  onClick={() => setShowEvalModal(false)}
                >
                  Cancel
                </button>
                <button
                  className="text-sm font-sans px-4 py-2 bg-arxiv-red text-white rounded font-medium hover:bg-arxiv-darkred"
                  onClick={submitEvalRequest}
                  disabled={!evalHfId.trim()}
                >
                  Submit Request
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
