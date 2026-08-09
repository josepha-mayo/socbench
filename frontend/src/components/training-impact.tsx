export type TrainingOutcome =
  | "improved"
  | "stable"
  | "regressed"
  | "diverged"
  | "insufficient_evidence";

export interface TrainingEntry {
  training_rank: number | null;
  hf_id: string;
  name: string;
  category: string;
  category_label?: string;
  training_score: number | null;
  combined_score: number | null;
  quality: number | null;
  initial_val_loss: number | null;
  best_val_loss: number | null;
  final_val_loss: number | null;
  relative_improvement: number | null;
  best_relative_improvement: number | null;
  run_outcome: TrainingOutcome | null;
  outcome_reason: string | null;
  tokens_seen: number | null;
  convergence_steps: number | null;
  completed_steps: number | null;
  loss_curve: number[] | null;
  downloads: number | null;
  downloads_all_time?: number | null;
  likes: number | null;
  created_at: string | null;
  trained_at: string | null;
  status: "trained" | "pending";
  source: "trained" | "trending" | "most_used";
}

const OUTCOME_STYLE: Record<TrainingOutcome, string> = {
  improved: "border-green-200 bg-green-50 text-green-800",
  stable: "border-gray-200 bg-gray-50 text-gray-700",
  regressed: "border-orange-200 bg-orange-50 text-orange-800",
  diverged: "border-red-200 bg-red-50 text-red-800",
  insufficient_evidence: "border-gray-200 bg-gray-50 text-gray-600",
};

function formatOutcome(outcome: TrainingOutcome | null) {
  return (outcome || "insufficient_evidence").replace("_", " ");
}

function formatNumber(value: number | null | undefined) {
  if (value == null) return "-";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
  if (value >= 1_000) return `${(value / 1_000).toFixed(1)}K`;
  return value.toLocaleString();
}

function formatDelta(value: number | null) {
  if (value == null) return "-";
  const percent = value * 100;
  return `${percent > 0 ? "+" : ""}${percent.toFixed(1)}%`;
}

function LossCurve({ losses, outcome }: { losses: number[] | null; outcome: TrainingOutcome | null }) {
  const clean = losses?.filter((value) => Number.isFinite(value)) || [];
  if (!clean.length) return <span className="text-arxiv-gray">-</span>;
  const max = Math.max(...clean);
  const min = Math.min(...clean);
  const range = max - min || 1;
  const color = outcome === "improved" ? "bg-green-600" : outcome === "diverged" ? "bg-red-600" : "bg-gray-500";
  return (
    <div className="flex h-10 w-28 items-end gap-px" aria-label="Validation loss curve">
      {clean.map((loss, index) => (
        <div
          key={`${index}-${loss}`}
          className={`min-w-[3px] flex-1 ${color}`}
          style={{ height: `${Math.max(8, ((loss - min) / range) * 100)}%` }}
          title={`Checkpoint ${index + 1}: ${loss.toFixed(4)}`}
        />
      ))}
    </div>
  );
}

function PendingSection({ title, entries }: { title: string; entries: TrainingEntry[] }) {
  if (!entries.length) return null;
  return (
    <section>
      <h3 className="mb-3 font-sans text-xs font-bold uppercase text-arxiv-dark">
        {title} <span className="font-normal text-arxiv-gray">({entries.length})</span>
      </h3>
      <div className="overflow-x-auto border border-arxiv-border">
        <table className="w-full table-fixed font-sans text-xs">
          <thead className="bg-arxiv-lightgray text-arxiv-gray">
            <tr>
              <th className="w-1/2 px-3 py-2 text-left">Dataset</th>
              <th className="px-3 py-2 text-left">Category</th>
              <th className="px-3 py-2 text-right">Downloads</th>
              <th className="px-3 py-2 text-right">Likes</th>
            </tr>
          </thead>
          <tbody>
            {entries.map((entry) => (
              <tr key={`${entry.source}-${entry.hf_id}`} className="border-t border-arxiv-border">
                <td className="break-words px-3 py-2">
                  <a href={`https://huggingface.co/datasets/${entry.hf_id}`} target="_blank" rel="noopener noreferrer" className="no-underline">
                    {entry.hf_id}
                  </a>
                </td>
                <td className="px-3 py-2 text-arxiv-gray">{entry.category_label || entry.category}</td>
                <td className="px-3 py-2 text-right font-mono">{formatNumber(entry.downloads_all_time ?? entry.downloads)}</td>
                <td className="px-3 py-2 text-right font-mono">{formatNumber(entry.likes)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function TrainingImpact({
  entries,
  loading,
  error,
  onRetry,
}: {
  entries: TrainingEntry[];
  loading: boolean;
  error: string | null;
  onRetry?: () => void;
}) {
  if (loading) return <p className="border-y border-arxiv-border py-8 text-center font-sans text-sm text-arxiv-gray">Loading training evidence...</p>;
  if (error) {
    return (
      <div className="border-y border-red-200 bg-red-50 px-4 py-6 text-center font-sans text-sm text-red-800">
        <p>{error}</p>
        {onRetry && <button type="button" onClick={onRetry} className="mt-3 border border-red-300 bg-white px-3 py-1.5 font-medium">Retry</button>}
      </div>
    );
  }

  const trained = entries.filter((entry) => entry.status === "trained");
  const improved = trained.filter((entry) => entry.run_outcome === "improved").length;
  const diverged = trained.filter((entry) => entry.run_outcome === "diverged").length;
  const pendingTrending = entries.filter((entry) => entry.status === "pending" && entry.source === "trending");
  const pendingMostUsed = entries.filter((entry) => entry.status === "pending" && entry.source === "most_used");

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-3 border-y border-arxiv-border bg-arxiv-lightgray font-sans">
        {[["Completed", trained.length], ["Improved", improved], ["Diverged", diverged]].map(([label, value]) => (
          <div key={label} className="border-r border-arxiv-border px-4 py-3 last:border-r-0">
            <div className="text-[10px] font-semibold uppercase text-arxiv-gray">{label}</div>
            <div className="font-mono text-xl font-bold text-arxiv-dark">{value}</div>
          </div>
        ))}
      </div>

      <section>
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h3 className="font-sans text-xs font-bold uppercase text-arxiv-dark">Validated training results</h3>
          <span className="font-sans text-[11px] text-arxiv-gray">Score = positive final validation-loss reduction</span>
        </div>
        <div className="overflow-x-auto border border-arxiv-border">
          <table className="min-w-[920px] w-full font-sans text-xs">
            <thead className="bg-arxiv-lightgray text-arxiv-gray">
              <tr>
                <th className="px-3 py-2 text-left">#</th>
                <th className="px-3 py-2 text-left">Dataset</th>
                <th className="px-3 py-2 text-left">Outcome</th>
                <th className="px-3 py-2 text-right">Score</th>
                <th className="px-3 py-2 text-right">Initial</th>
                <th className="px-3 py-2 text-right">Best</th>
                <th className="px-3 py-2 text-right">Final</th>
                <th className="px-3 py-2 text-right">Delta</th>
                <th className="px-3 py-2 text-right">Tokens</th>
                <th className="px-3 py-2 text-left">Curve</th>
              </tr>
            </thead>
            <tbody>
              {trained.map((entry, index) => {
                const outcome = entry.run_outcome || "insufficient_evidence";
                return (
                  <tr key={entry.hf_id} className="border-t border-arxiv-border even:bg-arxiv-lightgray/40">
                    <td className="px-3 py-2 font-mono text-arxiv-gray">{entry.training_rank ?? index + 1}</td>
                    <td className="max-w-72 break-words px-3 py-2">
                      <a href={`/datasets/${encodeURIComponent(entry.hf_id)}`} className="font-medium no-underline">{entry.hf_id}</a>
                      <div className="mt-0.5 text-[10px] text-arxiv-gray">{entry.category_label || entry.category}</div>
                    </td>
                    <td className="px-3 py-2">
                      <span title={entry.outcome_reason || undefined} className={`inline-block border px-2 py-0.5 font-mono text-[10px] font-bold uppercase ${OUTCOME_STYLE[outcome]}`}>
                        {formatOutcome(outcome)}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right font-mono font-bold">{entry.training_score?.toFixed(1) ?? "-"}</td>
                    <td className="px-3 py-2 text-right font-mono">{entry.initial_val_loss?.toFixed(4) ?? "-"}</td>
                    <td className="px-3 py-2 text-right font-mono">{entry.best_val_loss?.toFixed(4) ?? "-"}</td>
                    <td className="px-3 py-2 text-right font-mono">{entry.final_val_loss?.toFixed(4) ?? "-"}</td>
                    <td className={`px-3 py-2 text-right font-mono font-semibold ${entry.relative_improvement != null && entry.relative_improvement > 0 ? "text-green-700" : "text-red-700"}`}>
                      {formatDelta(entry.relative_improvement)}
                    </td>
                    <td className="px-3 py-2 text-right font-mono text-arxiv-gray">{formatNumber(entry.tokens_seen)}</td>
                    <td className="px-3 py-2"><LossCurve losses={entry.loss_curve} outcome={outcome} /></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      <PendingSection title="Pending from current trending datasets" entries={pendingTrending} />
      <PendingSection title="Pending from most-used datasets" entries={pendingMostUsed} />
    </div>
  );
}
