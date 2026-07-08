"use client";

import { useParams } from "next/navigation";
import { useState, useEffect } from "react";

interface ScoreDetail {
  name: string;
  score: number;
  details: Record<string, any>;
  warnings: string[];
}

interface ContaminationDetail {
  benchmark: string;
  overlap_rate: number;
  details: { overlap_count: number; total_eval: number };
}

interface ProvenanceEntry {
  model_name: string;
  paper_title: string;
  paper_url: string;
  verified: boolean;
}

interface RecEntry {
  category_key: string;
  label: string;
  rating: number;
  confidence: number;
  reasoning: string;
}

interface DatasetDetail {
  hf_id: string;
  name: string;
  category: string;
  category_label: string;
  description: string | null;
  license: string | null;
  tags: string[];
  quality: { score: number; details: Record<string, any> };
  diversity: { score: number; details: Record<string, any> };
  utility: { score: number; details: Record<string, any> };
  documentation: { score: number; details: Record<string, any> };
  popularity: { score: number; details: Record<string, any> };
  freshness: { score: number; details: Record<string, any> };
  pii_safety: { score: number; details: Record<string, any> };
  coverage: Record<string, any>;
  contamination_rate: number;
  provenance: ProvenanceEntry[];
  category_metrics: ScoreDetail[];
  recommendations?: { best_for: RecEntry[]; good_for: RecEntry[]; not_for: RecEntry[] };
  metadata: { downloads: number; likes: number; license: string };
  training: any;
}

function scoreBar(label: string, score: number, maxWidth: number = 100) {
  const pct = Math.round(score * 100);
  const color = score >= 0.7 ? "bg-green-500" : score >= 0.4 ? "bg-yellow-500" : "bg-red-500";
  return (
    <div className="mb-2">
      <div className="flex justify-between text-xs mb-1">
        <span className="font-sans font-medium">{label}</span>
        <span className="font-mono text-arxiv-gray">{score.toFixed(3)}</span>
      </div>
      <div className={`h-2.5 bg-arxiv-lightgray rounded-full overflow-hidden`}>
        <div className={`h-full ${color} rounded-full`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}

function DomainBar({ domains }: { domains: Record<string, number> }) {
  const total = Object.values(domains).reduce((a, b) => a + b, 0) || 1;
  const colors = ["bg-blue-500", "bg-green-500", "bg-yellow-500", "bg-purple-500", "bg-red-500", "bg-indigo-500", "bg-pink-500", "bg-teal-500"];
  return (
    <div className="h-4 flex rounded-full overflow-hidden bg-arxiv-lightgray">
      {Object.entries(domains).slice(0, 8).map(([key, val], i) => {
        const pct = (val / total) * 100;
        return (
          <div
            key={key}
            className={`h-full ${colors[i % colors.length]}`}
            style={{ width: `${pct}%` }}
            title={`${key}: ${(val * 100).toFixed(0)}%`}
          />
        );
      })}
    </div>
  );
}

function LossCurve({ losses }: { losses: number[] }) {
  if (!losses?.length) return null;
  const maxLoss = Math.max(...losses);
  const minLoss = Math.min(...losses);
  const range = maxLoss - minLoss || 1;
  return (
    <div className="flex items-end gap-px h-24 mt-2">
      {losses.map((l, i) => {
        const h = ((l - minLoss) / range) * 100;
        return (
          <div
            key={i}
            className="flex-1 bg-arxiv-red rounded-t-sm min-w-[2px] opacity-80 hover:opacity-100"
            style={{ height: `${Math.max(h, 2)}%` }}
            title={`Step ${i}: ${l.toFixed(4)}`}
          />
        );
      })}
    </div>
  );
}

export default function DatasetDetailPage() {
  const params = useParams();
  const hfId = decodeURIComponent(params.id as string);
  const [dataset, setDataset] = useState<DatasetDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    fetch(`/api/datasets/${encodeURIComponent(hfId)}`)
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((data) => {
        setDataset(data as DatasetDetail);
        setLoading(false);
      })
      .catch(() => setLoading(false));
  }, [hfId]);

  if (loading) return <p className="text-arxiv-gray font-sans">Loading...</p>;
  if (!dataset) return <p className="text-arxiv-gray font-sans">Dataset not found.</p>;

  return (
    <div>
      <div className="mb-6">
        <a href="/" className="text-xs font-sans text-arxiv-gray hover:text-arxiv-red">← Back</a>
        <h2 className="text-2xl font-serif font-bold text-arxiv-dark mt-2">{dataset.hf_id}</h2>
        <div className="flex gap-3 mt-1 text-xs font-sans">
          <span className="bg-arxiv-red text-white px-2 py-0.5 rounded-full">{dataset.category_label}</span>
          {dataset.metadata.license && <span className="text-arxiv-gray">License: {dataset.metadata.license}</span>}
          <span className="text-arxiv-gray">Downloads: {dataset.metadata.downloads.toLocaleString()}</span>
          <span className="text-arxiv-gray">Likes: {dataset.metadata.likes}</span>
        </div>
      </div>

      {/* Core dimensions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
        <div className="border border-arxiv-border rounded p-4">
          <h3 className="text-sm font-serif font-bold mb-3">Core Dimensions</h3>
          {scoreBar("Quality", dataset.quality.score)}
          {scoreBar("Diversity", dataset.diversity.score)}
          {scoreBar("Utility", dataset.utility.score)}
        </div>
        <div className="border border-arxiv-border rounded p-4">
          <h3 className="text-sm font-serif font-bold mb-3">Supporting</h3>
          {scoreBar("Documentation", dataset.documentation.score)}
          {scoreBar("Popularity", dataset.popularity.score)}
          {scoreBar("Freshness", dataset.freshness.score)}
        </div>
      </div>

      {/* Coverage */}
      {dataset.coverage?.domain_distribution && (
        <div className="border border-arxiv-border rounded p-4 mb-8">
          <h3 className="text-sm font-serif font-bold mb-2">Domain Coverage</h3>
          <DomainBar domains={dataset.coverage.domain_distribution} />
          <div className="flex flex-wrap gap-2 mt-2 text-[10px] font-mono text-arxiv-gray">
            {Object.entries(dataset.coverage.domain_distribution as Record<string, number>).slice(0, 8).map(([k, v]) => (
              <span key={k}>{k}: {(v * 100).toFixed(0)}%</span>
            ))}
          </div>
        </div>
      )}

      {/* Safety */}
      <div className="border border-arxiv-border rounded p-4 mb-8">
        <h3 className="text-sm font-serif font-bold mb-2">Safety & Ethics</h3>
        <div className="flex gap-6 text-sm font-sans">
          <div>
            <span className="text-arxiv-gray">PII Safety:</span>{" "}
            <span className="font-mono font-bold text-green-700">{dataset.pii_safety.score.toFixed(3)}</span>
          </div>
          <div>
            <span className="text-arxiv-gray">Contamination:</span>{" "}
            <span className={`font-mono font-bold ${dataset.contamination_rate > 0.1 ? 'text-red-700' : dataset.contamination_rate > 0.05 ? 'text-yellow-700' : 'text-green-700'}`}>
              {(dataset.contamination_rate * 100).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>

      {/* Recommendations */}
      {dataset.recommendations && (
        <div className="border border-arxiv-border rounded p-4 mb-8">
          <h3 className="text-sm font-serif font-bold mb-3">Best for — what this dataset is actually good for</h3>
          {dataset.recommendations.best_for.length > 0 && (
            <div className="mb-3">
              <div className="text-[11px] font-sans uppercase tracking-wide text-green-700 mb-1">Best for</div>
              <div className="flex flex-col gap-1">
                {dataset.recommendations.best_for.map((r) => (
                  <div key={r.category_key} className="flex items-center gap-2 text-xs font-sans">
                    <span className="w-32 shrink-0 font-medium">{r.label}</span>
                    <span className="text-arxiv-red tracking-tight">{"★".repeat(r.rating)}{"☆".repeat(5 - r.rating)}</span>
                    <span className="text-arxiv-gray">— {r.reasoning}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {dataset.recommendations.good_for.length > 0 && (
            <div className="mb-3">
              <div className="text-[11px] font-sans uppercase tracking-wide text-arxiv-gray mb-1">Good for</div>
              <div className="flex flex-col gap-1">
                {dataset.recommendations.good_for.map((r) => (
                  <div key={r.category_key} className="flex items-center gap-2 text-xs font-sans">
                    <span className="w-32 shrink-0 font-medium">{r.label}</span>
                    <span className="text-arxiv-red tracking-tight">{"★".repeat(r.rating)}{"☆".repeat(5 - r.rating)}</span>
                    <span className="text-arxiv-gray">— {r.reasoning}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
          {dataset.recommendations.not_for.length > 0 && (
            <div>
              <div className="text-[11px] font-sans uppercase tracking-wide text-red-700 mb-1">Not recommended for</div>
              <div className="flex flex-col gap-1">
                {dataset.recommendations.not_for.map((r) => (
                  <div key={r.category_key} className="flex items-center gap-2 text-xs font-sans">
                    <span className="w-32 shrink-0 font-medium">{r.label}</span>
                    <span className="text-arxiv-red tracking-tight">{"★".repeat(r.rating)}{"☆".repeat(5 - r.rating)}</span>
                    <span className="text-arxiv-gray">— {r.reasoning}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Provenance */}
      {dataset.provenance?.length > 0 && (
        <div className="border border-arxiv-border rounded p-4 mb-8">
          <h3 className="text-sm font-serif font-bold mb-2">Used By</h3>
          <table className="arxiv-table text-xs">
            <thead>
              <tr><th>Model</th><th>Paper</th><th>Verified</th></tr>
            </thead>
            <tbody>
              {dataset.provenance.map((p, i) => (
                <tr key={i}>
                  <td className="font-mono">{p.model_name}</td>
                  <td>
                    {p.paper_url ? (
                      <a href={p.paper_url} target="_blank" rel="noopener">{p.paper_title}</a>
                    ) : p.paper_title}
                  </td>
                  <td>{p.verified ? <span className="text-green-700">✓</span> : <span className="text-arxiv-gray">?</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Category metrics */}
      {dataset.category_metrics?.length > 0 && (
        <div className="border border-arxiv-border rounded p-4 mb-8">
          <h3 className="text-sm font-serif font-bold mb-2">Category-Specific Metrics</h3>
          <div className="grid grid-cols-2 gap-3">
            {dataset.category_metrics.map((m) => (
              <div key={m.name} className="text-xs">
                {scoreBar(m.name, m.score, 80)}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Training */}
      {dataset.training && (
        <div className="border border-arxiv-border rounded p-4">
          <h3 className="text-sm font-serif font-bold mb-2">Training Impact (GPT-2 124M, 1B tokens)</h3>
          <div className="text-xs font-sans text-arxiv-gray mb-2">
            Final Val Loss: <span className="font-mono font-bold">{dataset.training.final_val_loss.toFixed(4)}</span>
            {" · "}
            Convergence: <span className="font-mono">step {dataset.training.convergence_steps}</span>
          </div>
          <LossCurve losses={dataset.training.loss_curve} />
        </div>
      )}
    </div>
  );
}