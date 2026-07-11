export default function AboutPage() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-2xl font-serif font-bold text-arxiv-dark mb-1">
        Socbench
      </h2>
      <p className="text-lg font-serif italic text-arxiv-red mb-6">
        "The unexamined dataset is not worth training on."
      </p>

      <div className="space-y-6 text-sm font-serif leading-relaxed">
        <p>
          Socbench is scientific dataset intelligence — not another leaderboard, not a model benchmark.
          It's the IMDb for datasets. When someone is about to build a model, Socbench answers:
          "Which datasets should I train on, and why?"
        </p>

        <h3 className="text-lg font-serif font-bold pt-2">Multi-Dimension Scoring</h3>
        <p className="text-arxiv-gray">
          Not one composite number. Separate dimensions for separate questions.
          All scores are on a 0-100 scale.
        </p>
        <div className="grid grid-cols-2 gap-3 mt-2">
          {[
            ["Quality", "Gopher rules, FineWeb filters, DCLM criteria, deduplication"],
            ["Diversity", "Text diversity, domain coverage, token spread"],
            ["Utility", "How actionable this dataset is — format quality, documentation"],
            ["Documentation", "Dataset card completeness, license clarity"],
            ["Popularity", "Downloads, likes, community adoption"],
            ["Freshness", "How recently updated, actively maintained"],
            ["Contamination", "13-gram overlap with eval benchmarks (GSM8K, MMLU, HumanEval, MBPP)"],
            ["Repetition", "Exact-row duplication rate — catches slop and synthetic padding"],
          ].map(([label, desc]) => (
            <div key={label} className="border border-arxiv-border rounded p-3">
              <strong className="font-sans">{label}</strong>
              <p className="text-xs text-arxiv-gray mt-0.5">{desc}</p>
            </div>
          ))}
        </div>

        <h3 className="text-lg font-serif font-bold pt-4">Hierarchical Classification</h3>
        <p className="text-arxiv-gray">
          Datasets aren't one thing. A Ferrari isn't "better" than a truck — they're built
          for different jobs. Categories include: Pretraining (web, code, math, science, books),
          SFT (instruction, assistant), Preference Optimization (DPO, RLHF), Tool Calling,
          Agent Traces, Reasoning, Task (QA, summarization, translation), Multimodal, and Safety.
          Each category has its own metrics — code datasets get parseability, agent datasets
          get trajectory completeness and loop detection, tool-calling datasets get format consistency.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Contamination & Repetition</h3>
        <p className="text-arxiv-gray">
          Every dataset is checked for benchmark contamination using 13-gram overlap against
          GSM8K, MMLU, HumanEval, MBPP, and more. Datasets with &gt;1% overlap are flagged
          as contaminated. Repetition % measures exact-row duplication — a key signal for
          detecting synthetic slop and padded datasets. truthy-dpo at 99.9% repetition?
          That's a warning. Ultra-FineWeb at 0.45%? That's clean.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Request Evaluation</h3>
        <p className="text-arxiv-gray">
          Don't see a dataset you need? Use the "Request Evaluation" button on the leaderboard
          to submit any HuggingFace dataset ID for scoring. Choose public (results published
          on the leaderboard) or private (results shared with you only).
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Provenance Tracking</h3>
        <p className="text-arxiv-gray">
          Every dataset page shows which models were trained on it, which papers used it,
          and whether the provenance is verified. Open DeepSeek V4's page and see:
          FineWeb (quality 0.82), OpenCoder (quality 0.80), Stack V2 (quality 0.78).
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Evaluation Levels</h3>
        <p className="text-arxiv-gray">
          Socbench evaluates datasets in three levels, each more expensive and more
          informative than the last.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
          {[
            ["Level 1 — Automatic Scoring", "Cheap, fast heuristics. Gopher rules, FineWeb/DCLM filters, diversity, documentation, freshness, popularity."],
            ["Level 2 — Contamination & Repetition", "13-gram overlap with eval benchmarks (GSM8K, MMLU, HumanEval, MBPP). Exact-row duplication."],
            ["Level 3 — Training Impact", "Train GPT-2 124M on the dataset for 1B tokens on Kaggle 2x T4. Final loss, perplexity, convergence, and relative quality."],
          ].map(([title, desc]) => (
            <div key={title} className="border border-arxiv-border rounded p-3">
              <strong className="font-sans text-sm block mb-1">{title}</strong>
              <p className="text-xs text-arxiv-gray">{desc}</p>
            </div>
          ))}
        </div>

        <h3 className="text-lg font-serif font-bold pt-4">Training Impact (Stage 3)</h3>
        <p className="text-arxiv-gray">
          For top-tier datasets only. Train GPT-2 124M from scratch on each dataset
          for 1B tokens on Kaggle 2x T4. Measure loss curves, perplexity, and convergence.
          Relative comparison across all runs. Research proves 125M models predict data
          quality scaling for 3B models (Ankner et al., 2024). This stage earns its
          compute — it runs only when the result answers a question nobody else can answer.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Discovery Pipeline</h3>
        <p className="text-arxiv-gray">
          We scan HuggingFace for new and trending datasets daily. Only datasets above
          minimum thresholds (≥1K downloads, ≥10 likes, ≥1K rows) enter the pipeline.
          Automated scoring runs instantly. Contamination checks flag leakage.
          Top-tier datasets queue for training impact measurement on Kaggle 2x T4.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Research Sources</h3>
        <ul className="list-disc pl-5 space-y-1 text-arxiv-gray text-xs">
          <li>Gopher quality rules — Rae et al., 2021</li>
          <li>FineWeb filters — Penedo et al., NeurIPS 2024</li>
          <li>DCLM filtering + fastText classifier — Li et al., NeurIPS 2024</li>
          <li>SlimPajama deduplication (13-gram, Jaccard 0.8) — Cerebras, 2023</li>
          <li>RedPajama-V2 46 quality signals — Weber et al., NeurIPS 2024</li>
          <li>Dolma curation methodology — Soldaini et al., ACL 2024</li>
          <li>125M proxy models predict 3B scaling — Ankner et al., 2024</li>
          <li>DoReMi domain reweighting — Xie et al., NeurIPS 2023</li>
          <li>DataComp standardized data evaluation — Gadre et al., NeurIPS 2023</li>
          <li>Perplexity-based data pruning — Ankner et al., 2024</li>
        </ul>
      </div>
    </div>
  );
}