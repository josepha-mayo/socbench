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
    const mock: DatasetDetail = {
      hf_id: hfId,
      name: hfId.split("/").pop() || hfId,
      category: "pretraining-code",
      category_label: "Code (Pretraining)",
      description: "High-quality code training dataset.",
      license: "mit",
      tags: ["code", "pretraining"],
      quality: { score: 0.87, details: { gopher_pass: 0.91, dedup: 0.92, format: 0.88 } },
      diversity: { score: 0.72, details: { text_diversity: 0.72, token_spread: 0.68 } },
      utility: { score: 0.85, details: { schema_conformance: 0.95 } },
      documentation: { score: 0.90, details: { has_card: true, has_license: true } },
      popularity: { score: 0.72, details: { downloads: 45000, likes: 180 } },
      freshness: { score: 0.85, details: {} },
      pii_safety: { score: 0.98, details: { pii_rate: 0.002 } },
      coverage: { domain_distribution: { code: 0.82, general: 0.14, math: 0.04 } },
      contamination_rate: 0.02,
      provenance: [
        { model_name: "StarCoder (15.5B)", paper_title: "StarCoder: may the source be with you!", paper_url: "https://arxiv.org/abs/2305.06161", verified: true },
      ],
      category_metrics: [
        { name: "parse_rate", score: 0.94, details: {}, warnings: [] },
        { name: "language_coverage", score: 0.78, details: {}, warnings: [] },
      ],
      metadata: { downloads: 45000, likes: 180, license: "mit" },
      training: { final_val_loss: 3.21, loss_curve: Array.from({ length: 30 }, (_, i) => 5.5 - (i / 30) * 2.3 + Math.random() * 0.1), convergence_steps: 12 },
    };
    setDataset(mock);
    setLoading(false);
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
            {Object.entries(dataset.coverage.domain_distribution).slice(0, 8).map(([k, v]) => (
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