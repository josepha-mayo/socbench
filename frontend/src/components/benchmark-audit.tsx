"use client";

import { ChangeEvent, FormEvent, useRef, useState } from "react";

interface AuditScores {
  description_expected_alignment: number;
  policy_expected_alignment: number;
  policy_violations_per_task: number;
  policy_violation_coverage: number;
  semantic_quality: number;
}

interface TaskAudit {
  id: string;
  description_expected_alignment: number;
  policy_expected_alignment: number;
  violated_policy_items: number[];
  diagnostics: string[];
  engine: string;
}

interface PolicyCoverage {
  index: number;
  policy_item: string;
  task_count: number;
  covered: boolean;
}

interface AuditResult {
  benchmark_name: string;
  engine: string;
  api_judge_configured: boolean;
  api_judge_enhanced_tasks: number;
  task_count: number;
  policy_item_count: number;
  coverage_threshold: number;
  scores: AuditScores;
  tasks: TaskAudit[];
  policy_coverage: PolicyCoverage[];
  diagnostics: string[];
  warnings: string[];
}

const EXAMPLE_POLICY = `The agent must verify the user's identity before changing an address.
The agent must not cancel an order after shipment.
Refunds must return to the original payment method.`;

const EXAMPLE_TASKS = JSON.stringify(
  [
    {
      id: "cancel-shipped",
      description: "The user asks to cancel an order that has already shipped.",
      expected_behavior: "The agent must refuse to cancel the shipped order.",
      initial_state: { order_status: "shipped" },
    },
    {
      id: "refund-card",
      description: "The customer requests a refund for an order paid by card.",
      expected_behavior: "Return the refund to the original payment card.",
      initial_state: { payment_method: "card" },
    },
  ],
  null,
  2,
);

function scoreTone(score: number) {
  if (score >= 70) return "text-green-700";
  if (score >= 45) return "text-yellow-700";
  return "text-red-700";
}

function formatEngine(engine: string) {
  return engine === "api_enhanced_semantic_v1" ? "API-enhanced semantic" : "Local semantic";
}

export default function BenchmarkAudit() {
  const [name, setName] = useState("");
  const [policy, setPolicy] = useState("");
  const [tasksJson, setTasksJson] = useState("[]");
  const [coverageThreshold, setCoverageThreshold] = useState(3);
  const [useApiJudge, setUseApiJudge] = useState(true);
  const [result, setResult] = useState<AuditResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  function loadExample() {
    setName("Retail agent benchmark");
    setPolicy(EXAMPLE_POLICY);
    setTasksJson(EXAMPLE_TASKS);
    setCoverageThreshold(1);
    setResult(null);
    setError(null);
  }

  async function loadFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const payload = JSON.parse(await file.text());
      if (!payload || typeof payload !== "object" || !Array.isArray(payload.tasks)) {
        throw new Error("Expected a benchmark JSON object with a tasks array.");
      }
      setName(String(payload.name || file.name.replace(/\.json$/i, "")));
      setPolicy(String(payload.policy || (payload.policy_items || []).join("\n")));
      setTasksJson(JSON.stringify(payload.tasks, null, 2));
      setCoverageThreshold(Number(payload.coverage_threshold || 3));
      setResult(null);
      setError(null);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not read benchmark JSON.");
    }
  }

  async function submit(event: FormEvent) {
    event.preventDefault();
    setError(null);
    setResult(null);
    let tasks: unknown;
    try {
      tasks = JSON.parse(tasksJson);
      if (!Array.isArray(tasks) || tasks.length === 0) {
        throw new Error("Tasks JSON must be a non-empty array.");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Tasks JSON is invalid.");
      return;
    }

    setLoading(true);
    try {
      const response = await fetch("/api/evals/benchmark-audit", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: name || "Untitled benchmark",
          policy,
          tasks,
          coverage_threshold: coverageThreshold,
          use_api_judge: useApiJudge,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        const detail = Array.isArray(payload.detail)
          ? payload.detail.map((item: { msg?: string }) => item.msg || "Invalid input").join("; ")
          : payload.detail;
        throw new Error(detail || `Backend returned ${response.status}`);
      }
      setResult(payload);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Benchmark audit failed.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div>
      <form onSubmit={submit} className="border border-arxiv-border rounded bg-white p-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-arxiv-border pb-3 mb-4">
          <h3 className="text-lg font-serif font-bold">Reference-Free Benchmark Audit</h3>
          <div className="flex flex-wrap gap-2">
            <input ref={fileInput} type="file" accept="application/json,.json" onChange={loadFile} className="hidden" />
            <button type="button" className="category-tab" onClick={() => fileInput.current?.click()}>
              Load JSON
            </button>
            <button type="button" className="category-tab" onClick={loadExample}>
              Load Example
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4">
          <label className="font-sans text-xs font-medium text-arxiv-dark">
            Benchmark name
            <input
              value={name}
              onChange={(event) => setName(event.target.value)}
              maxLength={256}
              className="mt-1 block w-full border border-arxiv-border rounded px-3 py-2 font-sans text-sm focus:outline-none focus:border-arxiv-red"
              placeholder="Customer support benchmark"
            />
          </label>

          <label className="font-sans text-xs font-medium text-arxiv-dark">
            Domain policy
            <textarea
              required
              value={policy}
              onChange={(event) => setPolicy(event.target.value)}
              rows={7}
              maxLength={100000}
              className="mt-1 block w-full resize-y border border-arxiv-border rounded px-3 py-2 font-mono text-xs leading-relaxed focus:outline-none focus:border-arxiv-red"
              placeholder="One policy item per line"
            />
          </label>

          <label className="font-sans text-xs font-medium text-arxiv-dark">
            Tasks JSON
            <textarea
              required
              value={tasksJson}
              onChange={(event) => setTasksJson(event.target.value)}
              rows={12}
              className="mt-1 block w-full resize-y border border-arxiv-border rounded px-3 py-2 font-mono text-xs leading-relaxed focus:outline-none focus:border-arxiv-red"
              spellCheck={false}
            />
          </label>
        </div>

        <div className="mt-4 flex flex-wrap items-end justify-between gap-4">
          <div className="flex flex-wrap items-center gap-5">
            <label className="font-sans text-xs font-medium text-arxiv-dark">
              Coverage threshold
              <input
                type="number"
                min={1}
                max={100}
                required
                value={coverageThreshold}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  if (Number.isFinite(value)) setCoverageThreshold(value);
                }}
                className="mt-1 block w-24 border border-arxiv-border rounded px-3 py-2 font-mono text-sm focus:outline-none focus:border-arxiv-red"
              />
            </label>
            <label className="flex items-center gap-2 pb-2 font-sans text-xs text-arxiv-dark">
              <input
                type="checkbox"
                checked={useApiJudge}
                onChange={(event) => setUseApiJudge(event.target.checked)}
                className="h-4 w-4 accent-arxiv-red"
              />
              Use configured API judge
            </label>
          </div>
          <button
            type="submit"
            disabled={loading}
            className="rounded bg-arxiv-red px-4 py-2 font-sans text-sm font-medium text-white hover:bg-arxiv-darkred disabled:cursor-wait disabled:opacity-60"
          >
            {loading ? "Auditing..." : "Run Audit"}
          </button>
        </div>
      </form>

      {error && (
        <div role="alert" className="mt-4 border border-red-200 rounded bg-red-50 p-3 font-sans text-sm text-red-700">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6">
          <div className="flex flex-wrap items-end justify-between gap-2 mb-3">
            <div>
              <h3 className="text-lg font-serif font-bold">{result.benchmark_name}</h3>
              <div className="font-mono text-[10px] uppercase text-arxiv-gray">
                {formatEngine(result.engine)} · {result.task_count} tasks · {result.policy_item_count} policy items
              </div>
            </div>
            {result.api_judge_enhanced_tasks > 0 && (
              <span className="border border-green-300 rounded-full bg-green-50 px-2 py-1 font-sans text-[10px] text-green-700">
                {result.api_judge_enhanced_tasks} API-enhanced tasks
              </span>
            )}
          </div>

          <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-5">
            <div className="stat-card">
              <div className="text-[10px] font-sans uppercase text-arxiv-gray">Task Alignment</div>
              <div className={`text-xl font-mono font-bold ${scoreTone(result.scores.description_expected_alignment)}`}>
                {result.scores.description_expected_alignment.toFixed(1)}
              </div>
            </div>
            <div className="stat-card">
              <div className="text-[10px] font-sans uppercase text-arxiv-gray">Policy Alignment</div>
              <div className={`text-xl font-mono font-bold ${scoreTone(result.scores.policy_expected_alignment)}`}>
                {result.scores.policy_expected_alignment.toFixed(1)}
              </div>
            </div>
            <div className="stat-card">
              <div className="text-[10px] font-sans uppercase text-arxiv-gray">Violations / Task</div>
              <div className="text-xl font-mono font-bold text-arxiv-dark">
                {result.scores.policy_violations_per_task.toFixed(2)}
              </div>
            </div>
            <div className="stat-card">
              <div className="text-[10px] font-sans uppercase text-arxiv-gray">Policy Coverage</div>
              <div className={`text-xl font-mono font-bold ${scoreTone(result.scores.policy_violation_coverage)}`}>
                {result.scores.policy_violation_coverage.toFixed(1)}%
              </div>
            </div>
            <div className="stat-card col-span-2 md:col-span-1">
              <div className="text-[10px] font-sans uppercase text-arxiv-gray">Semantic Quality</div>
              <div className={`text-xl font-mono font-bold ${scoreTone(result.scores.semantic_quality)}`}>
                {result.scores.semantic_quality.toFixed(1)}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-5 font-sans text-xs">
            <section>
              <h4 className="section-header">Diagnostics</h4>
              <ul className="space-y-1 text-arxiv-gray">
                {result.diagnostics.map((diagnostic) => <li key={diagnostic}>· {diagnostic}</li>)}
                {result.warnings.map((warning) => <li key={warning} className="text-yellow-700">· {warning}</li>)}
              </ul>
            </section>
            <section>
              <h4 className="section-header">Policy Coverage</h4>
              <div className="max-h-44 overflow-y-auto border-y border-arxiv-border">
                {result.policy_coverage.map((item) => (
                  <div key={item.index} className="flex gap-2 border-b border-arxiv-border py-2 last:border-b-0">
                    <span className={`font-mono font-bold ${item.covered ? "text-green-700" : "text-red-700"}`}>
                      {item.task_count}
                    </span>
                    <span className="text-arxiv-gray">{item.policy_item}</span>
                  </div>
                ))}
              </div>
            </section>
          </div>

          <div className="overflow-x-auto border border-arxiv-border rounded">
            <table className="arxiv-table min-w-[760px]">
              <thead>
                <tr>
                  <th>Task</th>
                  <th>Task Alignment</th>
                  <th>Policy Alignment</th>
                  <th>Violations</th>
                  <th>Diagnostics</th>
                </tr>
              </thead>
              <tbody>
                {result.tasks.map((task) => (
                  <tr key={task.id}>
                    <td className="font-mono text-xs font-medium">{task.id}</td>
                    <td className={`font-mono text-xs font-bold ${scoreTone(task.description_expected_alignment)}`}>
                      {task.description_expected_alignment.toFixed(1)}
                    </td>
                    <td className={`font-mono text-xs font-bold ${scoreTone(task.policy_expected_alignment)}`}>
                      {task.policy_expected_alignment.toFixed(1)}
                    </td>
                    <td className="font-mono text-xs">{task.violated_policy_items.join(", ") || "—"}</td>
                    <td className="max-w-sm font-sans text-xs text-arxiv-gray">
                      {task.diagnostics.join(" ") || "No task-level weakness detected."}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
