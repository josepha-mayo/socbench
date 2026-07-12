"use client";

import { useState, useEffect, Fragment } from "react";

interface EvalEntry {
  key: string;
  name: string;
  category: string;
  open_source: boolean;
  hf_id: string | null;
  description: string;
  paper: string;
  num_examples: number;
  sota_pass_rate: number;
  ceiling: number;
  saturation_index: number;
  contamination_risk: number;
  discriminative_power: number;
  annotation_quality: number;
  coverage_breadth: number;
  effective_quality: number;
  headroom: number;
  status: string;
  data_publicly_available: boolean;
  deprecated: boolean;
  successor: string | null;
  year: number;
  actively_maintained: boolean;
  contamination_incidents: string[];
  overlap_with_training: number;
  overlap_benchmarks: { dataset: string; overlap_rate: number; overlap_count: number; total_eval: number }[];
}

interface EvalSummary {
  total: number;
  open_source: number;
  closed_source: number;
  saturated: number;
  contaminated: number;
  deprecated: number;
  fresh: number;
  categories: Record<string, { total: number; open: number; closed: number }>;
}

const STATUS_COLORS: Record<string, string> = {
  fresh: "bg-green-100 text-green-800 border-green-300",
  maturing: "bg-yellow-100 text-yellow-800 border-yellow-300",
  saturated: "bg-orange-100 text-orange-800 border-orange-300",
  contaminated: "bg-red-100 text-red-800 border-red-300",
  deprecated: "bg-gray-200 text-gray-600 border-gray-400 line-through",
};

const CATEGORY_LABELS: Record<string, string> = {
  knowledge: "Knowledge",
  code: "Code",
  math: "Math",
  reasoning: "Reasoning",
  safety: "Safety",
  chat: "Chat",
  agent: "Agent",
};

const CAT_FILTERS = ["All", "knowledge", "code", "math", "reasoning", "safety", "chat", "agent"];
const SOURCE_FILTERS = ["All", "Open Source", "Closed Source"];
const VIEW_MODES = ["Table", "Chart", "Graph"];

function metricBar(val: number, reverseColors = false) {
  const pct = Math.round(val);
  const color = reverseColors
    ? pct >= 60 ? "bg-red-500" : pct >= 30 ? "bg-yellow-500" : "bg-green-500"
    : pct >= 70 ? "bg-green-500" : pct >= 40 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-14 h-2 rounded-full overflow-hidden bg-arxiv-lightgray">
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[10px] font-bold text-arxiv-dark w-7">{pct}</span>
    </div>
  );
}

export default function EvalsPage() {
  const [evals, setEvals] = useState<EvalEntry[]>([]);
  const [summary, setSummary] = useState<EvalSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [catFilter, setCatFilter] = useState("All");
  const [sourceFilter, setSourceFilter] = useState("All");
  const [viewMode, setViewMode] = useState("Table");
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    setLoading(true);
    fetch("/api/evals")
      .then((r) => {
        if (!r.ok) throw new Error(`Backend returned ${r.status}`);
        return r.json();
      })
      .then((data) => {
        setEvals(data.evals || []);
        setSummary(data.summary || null);
        setLoading(false);
      })
      .catch((e) => {
        setError(e.message || "Could not connect to backend.");
        setLoading(false);
      });
  }, []);

  const filtered = evals.filter((e) => {
    if (catFilter !== "All" && e.category !== catFilter) return false;
    if (sourceFilter === "Open Source" && !e.open_source) return false;
    if (sourceFilter === "Closed Source" && e.open_source) return false;
    return true;
  });

  return (
    <div>
      <div className="mb-6">
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark">
          Evaluation Benchmark Intelligence
        </h2>
        <p className="text-sm font-sans text-arxiv-gray mt-1">
          Contamination risk, saturation, and quality analysis for the top 20 evaluation benchmarks — open and closed source.
          Ranked by effective quality (discriminative power, contamination resistance, annotation quality, coverage).
        </p>
      </div>

      {summary && (
        <div className="grid grid-cols-2 md:grid-cols-6 gap-3 mb-6">
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Total</div>
            <div className="text-xl font-mono font-bold text-arxiv-dark">{summary.total}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Open</div>
            <div className="text-xl font-mono font-bold text-green-700">{summary.open_source}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Closed</div>
            <div className="text-xl font-mono font-bold text-arxiv-dark">{summary.closed_source}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Saturated</div>
            <div className={`text-xl font-mono font-bold ${summary.saturated > 5 ? "text-red-700" : "text-yellow-700"}`}>{summary.saturated}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Contaminated</div>
            <div className={`text-xl font-mono font-bold ${summary.contaminated > 3 ? "text-red-700" : "text-green-700"}`}>{summary.contaminated}</div>
          </div>
          <div className="stat-card">
            <div className="text-[10px] font-sans uppercase tracking-wide text-arxiv-gray">Deprecated</div>
            <div className="text-xl font-mono font-bold text-gray-600">{summary.deprecated}</div>
          </div>
        </div>
      )}

      <div className="flex flex-wrap items-center gap-2 mb-4">
        <div className="flex flex-wrap gap-1.5">
          {CAT_FILTERS.map((cat) => (
            <button
              key={cat}
              className={`category-tab ${catFilter === cat ? "active" : ""}`}
              onClick={() => setCatFilter(cat)}
            >
              {cat === "All" ? "All" : CATEGORY_LABELS[cat] || cat}
            </button>
          ))}
        </div>
        <div className="flex gap-1.5 ml-auto">
          {SOURCE_FILTERS.map((s) => (
            <button
              key={s}
              className={`category-tab ${sourceFilter === s ? "active" : ""}`}
              onClick={() => setSourceFilter(s)}
            >
              {s}
            </button>
          ))}
        </div>
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
      </div>

      {loading ? (
        <div className="p-8 border border-arxiv-border rounded bg-arxiv-lightgray text-center">
          <p className="text-arxiv-gray font-sans">Analyzing evaluation benchmarks...</p>
        </div>
      ) : error ? (
        <div className="p-8 border border-red-200 rounded bg-red-50 text-center">
          <p className="text-red-700 font-sans font-medium">{error}</p>
          <p className="text-red-600 text-xs font-sans mt-2">
            Make sure the backend is running on <code className="bg-white px-1 rounded">http://localhost:8000</code>.
          </p>
        </div>
      ) : viewMode === "Table" ? (
        <div className="overflow-x-auto border border-arxiv-border rounded">
          <table className="arxiv-table">
            <thead>
              <tr>
                <th className="w-10">#</th>
                <th className="min-w-[160px]">Benchmark</th>
                <th className="w-[80px]">Category</th>
                <th className="w-[60px]">Source</th>
                <th className="w-[60px]">SOTA</th>
                <th className="w-[60px]">Ceiling</th>
                <th className="w-[90px]">Saturation</th>
                <th className="w-[90px]">Contam. Risk</th>
                <th className="w-[80px]">Discrim.</th>
                <th className="w-[80px]">Eff. Quality</th>
                <th className="w-[90px]">Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((e, i) => (
                <Fragment key={e.key}>
                  <tr
                    className="cursor-pointer"
                    onClick={() => setExpanded(expanded === e.key ? null : e.key)}
                  >
                    <td className="font-mono text-arxiv-gray text-xs">{i + 1}</td>
                    <td>
                      <div className="flex items-center gap-1.5">
                        <span className={`font-sans text-sm font-medium ${e.deprecated ? "line-through text-arxiv-gray" : ""}`}>{e.name}</span>
                        {e.hf_id && (
                          <a
                            href={`https://huggingface.co/datasets/${e.hf_id}`}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-[10px] text-arxiv-link hover:text-arxiv-hover no-underline"
                            onClick={(ev) => ev.stopPropagation()}
                          >
                            HF↗
                          </a>
                        )}
                      </div>
                      <div className="text-[10px] text-arxiv-gray mt-0.5">{e.description}</div>
                    </td>
                    <td>
                      <span className="text-[10px] font-sans bg-arxiv-red/10 text-arxiv-red px-2 py-0.5 rounded-full whitespace-nowrap">
                        {CATEGORY_LABELS[e.category] || e.category}
                      </span>
                    </td>
                    <td>
                      {e.open_source ? (
                        <span className="text-[10px] font-mono text-green-700 font-bold">OPEN</span>
                      ) : (
                        <span className="text-[10px] font-mono text-arxiv-gray font-bold">CLOSED</span>
                      )}
                    </td>
                    <td className="font-mono text-xs font-bold text-arxiv-dark">{e.sota_pass_rate}%</td>
                    <td className="font-mono text-xs text-arxiv-gray">{e.ceiling}%</td>
                    <td>{metricBar(e.saturation_index, true)}</td>
                    <td>{metricBar(e.contamination_risk, true)}</td>
                    <td>{metricBar(e.discriminative_power)}</td>
                    <td>{metricBar(e.effective_quality)}</td>
                    <td>
                      <span className={`text-[10px] font-sans px-2 py-0.5 rounded-full border ${STATUS_COLORS[e.status] || ""}`}>
                        {e.status}
                      </span>
                    </td>
                  </tr>
                  {expanded === e.key && (
                    <tr className="bg-arxiv-lightgray">
                      <td colSpan={11} className="p-4">
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs font-sans">
                          <div>
                            <strong className="text-arxiv-dark block mb-1">Details</strong>
                            <div className="text-arxiv-gray">Paper: {e.paper}</div>
                            <div className="text-arxiv-gray">Year: {e.year}</div>
                            <div className="text-arxiv-gray">Examples: {e.num_examples.toLocaleString()}</div>
                            <div className="text-arxiv-gray">Data public: {e.data_publicly_available ? "Yes" : "No"}</div>
                            <div className="text-arxiv-gray">Actively maintained: {e.actively_maintained ? "Yes" : "No"}</div>
                            {e.deprecated && e.successor && (
                              <div className="text-orange-700 font-medium mt-1">Deprecated → {e.successor}</div>
                            )}
                          </div>
                          <div>
                            <strong className="text-arxiv-dark block mb-1">Quality Metrics</strong>
                            <div className="space-y-1">
                              <div>Discriminative power: {metricBar(e.discriminative_power)}</div>
                              <div>Annotation quality: {metricBar(e.annotation_quality)}</div>
                              <div>Coverage breadth: {metricBar(e.coverage_breadth)}</div>
                              <div>Effective quality: {metricBar(e.effective_quality)}</div>
                            </div>
                          </div>
                          <div>
                            <strong className="text-arxiv-dark block mb-1">Contamination Analysis</strong>
                            {e.contamination_incidents.length > 0 ? (
                              <ul className="list-disc pl-4 text-arxiv-gray">
                                {e.contamination_incidents.map((c, j) => (
                                  <li key={j}>{c}</li>
                                ))}
                              </ul>
                            ) : (
                              <span className="text-green-700">No known contamination incidents</span>
                            )}
                            {e.overlap_with_training > 0 && (
                              <div className="mt-1 text-red-700 font-medium">
                                Overlap with our training data: {(e.overlap_with_training * 100).toFixed(2)}%
                              </div>
                            )}
                          </div>
                          <div>
                            <strong className="text-arxiv-dark block mb-1">Saturation Analysis</strong>
                            <div className="text-arxiv-gray">SOTA: {e.sota_pass_rate}% / Ceiling: {e.ceiling}%</div>
                            <div className="text-arxiv-gray">Saturation index: {e.saturation_index}%</div>
                            <div className="text-arxiv-gray">Headroom: {e.headroom}%</div>
                            <div className="text-arxiv-gray mt-1">
                              {e.status === "deprecated"
                                ? "This eval has been deprecated. Use its successor for frontier evaluation."
                                : e.saturation_index > 90
                                ? "Eval is saturated — SOTA near ceiling. Losing discriminative power."
                                : e.saturation_index > 75
                                ? "Eval is maturing — limited headroom remains for differentiation."
                                : "Eval is fresh — significant headroom for model improvement."}
                            </div>
                          </div>
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      ) : viewMode === "Chart" ? (
        /* Chart view: horizontal bar chart of effective quality */
        <div className="border border-arxiv-border rounded p-4 bg-white">
          <h3 className="text-sm font-serif font-bold mb-4">Effective Quality by Benchmark (higher = better)</h3>
          <div className="space-y-2">
            {filtered.map((e, i) => (
              <div key={e.key} className="flex items-center gap-3">
                <div className="w-40 text-xs font-sans font-medium text-right truncate" title={e.name}>{i + 1}. {e.name}</div>
                <div className="flex-1 h-6 bg-arxiv-lightgray rounded-full overflow-hidden relative">
                  <div
                    className={`h-full rounded-full ${e.effective_quality >= 70 ? "bg-green-500" : e.effective_quality >= 40 ? "bg-yellow-500" : "bg-red-500"}`}
                    style={{ width: `${e.effective_quality}%` }}
                  />
                  <span className="absolute inset-0 flex items-center justify-end pr-2 text-[10px] font-mono font-bold text-arxiv-dark">
                    {Math.round(e.effective_quality)}
                  </span>
                </div>
                <div className={`w-20 text-[10px] font-sans px-2 py-0.5 rounded-full border text-center ${STATUS_COLORS[e.status] || ""}`}>
                  {e.status}
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : (
        /* Graph view: scatter plot of saturation vs contamination risk */
        <div className="border border-arxiv-border rounded p-4 bg-white">
          <h3 className="text-sm font-serif font-bold mb-4">Saturation vs Contamination Risk (bottom-left = best)</h3>
          <div className="relative w-full" style={{ height: "500px" }}>
            {/* Axes */}
            <div className="absolute bottom-8 left-32 right-4 h-px bg-arxiv-border" />
            <div className="absolute bottom-8 left-32 top-4 w-px bg-arxiv-border" />
            {/* Axis labels */}
            <div className="absolute bottom-0 left-32 right-4 text-center text-[10px] font-sans text-arxiv-gray">Saturation Index (%) →</div>
            <div className="absolute top-4 bottom-8 left-0 w-32 flex items-center justify-center text-[10px] font-sans text-arxiv-gray" style={{ writingMode: "vertical-rl", transform: "rotate(180deg)" }}>Contamination Risk (%) →</div>
            {/* Grid lines */}
            {[25, 50, 75, 100].map((v) => (
              <div key={`h${v}`} className="absolute bottom-8 left-32 right-4 border-t border-dashed border-arxiv-border/50" style={{ bottom: `${8 + v * 0.84}%` }} />
            ))}
            {[25, 50, 75, 100].map((v) => (
              <div key={`v${v}`} className="absolute bottom-8 top-4 border-l border-dashed border-arxiv-border/50" style={{ left: `${32 + v * 0.85}%` }} />
            ))}
            {/* Points */}
            {filtered.map((e) => {
              const left = `${32 + e.saturation_index * 0.85}%`;
              const bottom = `${8 + e.contamination_risk * 0.84}%`;
              const color = e.status === "fresh" ? "bg-green-500" : e.status === "maturing" ? "bg-yellow-500" : e.status === "saturated" ? "bg-orange-500" : e.status === "contaminated" ? "bg-red-500" : "bg-gray-400";
              return (
                <div
                  key={e.key}
                  className="absolute w-3 h-3 rounded-full cursor-pointer hover:scale-150 transition-transform"
                  style={{ left, bottom, transform: "translate(-50%, 50%)" }}
                  title={`${e.name}: sat=${e.saturation_index}% cont=${e.contamination_risk}% status=${e.status}`}
                >
                  <div className={`w-3 h-3 rounded-full ${color}`} />
                  <div className="absolute top-4 left-4 whitespace-nowrap text-[9px] font-sans text-arxiv-dark bg-white/80 px-1 rounded">
                    {e.name}
                  </div>
                </div>
              );
            })}
          </div>
          <div className="flex gap-4 mt-4 text-[10px] font-sans">
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-green-500" /> fresh</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-yellow-500" /> maturing</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-orange-500" /> saturated</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-red-500" /> contaminated</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-gray-400" /> deprecated</span>
          </div>
        </div>
      )}

      <div className="mt-8 info-banner">
        <strong className="text-arxiv-dark">Methodology:</strong> Saturation Index = SOTA / Ceiling
        (ICML 2026). Contamination risk weighted from data availability (+30%), known incidents
        (+15% each, cap 45%), open-source (+10%), saturation with public data (+15%), stale (+5%),
        small dataset &lt;500 (+10%). Effective quality = 35% discrimination + 25% contamination
        resistance + 15% annotation + 15% coverage + 10% maintenance. Status:{" "}
        <span className="text-green-700 font-medium">fresh</span> = low saturation + low contamination,{" "}
        <span className="text-yellow-700 font-medium">maturing</span> = approaching ceiling,{" "}
        <span className="text-orange-700 font-medium">saturated</span> = SOTA near ceiling,{" "}
        <span className="text-red-700 font-medium">contaminated</span> = high contamination risk,{" "}
        <span className="text-gray-600 font-medium">deprecated</span> = superseded by successor.
        Sources: LiveBench (ICLR 2025), LiveCodeBench, GSM1k study, "Leak, Cheat, Repeat" (EACL 2024).
        Click any row for detailed analysis.
      </div>
    </div>
  );
}
