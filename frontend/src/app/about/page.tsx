export default function AboutPage() {
  return (
    <div className="max-w-3xl">
      <h2 className="text-2xl font-serif font-bold text-arxiv-dark mb-1">
        Socbench
      </h2>
      <p className="text-lg font-serif italic text-arxiv-red mb-6">
        &ldquo;The unexamined dataset is not worth training on.&rdquo;
      </p>

      <div className="space-y-6 text-sm font-serif leading-relaxed">
        {/* Socrates section */}
        <div className="border-l-4 border-arxiv-red pl-4 py-2 bg-arxiv-lightgray rounded-r">
          <h3 className="text-lg font-serif font-bold text-arxiv-dark mb-2">Why &ldquo;Soc&rdquo;bench?</h3>
          <p className="text-arxiv-gray">
            Socrates &mdash; the ancient Greek philosopher &mdash; famously held that
            <strong> &ldquo;the unexamined life is not worth living.&rdquo;</strong> He believed that
            without rigorous questioning, self-reflection, and honest examination, life has no value.
            We apply the same principle to AI training data: <strong>the unexamined dataset is not
            worth training on.</strong>
          </p>
          <p className="text-arxiv-gray mt-2">
            Just as Socrates walked the streets of Athens questioning citizens about their beliefs &mdash;
            testing whether they held up to scrutiny &mdash; Socbench walks through every dataset
            questioning its quality, diversity, contamination, and training value. Socrates showed
            that unexamined beliefs lead to unexamined actions. In AI, unexamined training data leads
            to unexamined models &mdash; models that memorize benchmarks, propagate slop, and fail
            silently in production.
          </p>
          <p className="text-arxiv-gray mt-2">
            The Socratic method didn&rsquo;t just find answers &mdash; it found the <em>right questions</em>
            to ask. Socbench does the same: not &ldquo;is this dataset good?&rdquo; but &ldquo;good for
            <em> what</em>?&rdquo; A Ferrari isn&rsquo;t better than a truck. They&rsquo;re built for
            different jobs. Socbench examines datasets along the dimensions that matter for their
            intended purpose.
          </p>
        </div>

        <p>
          Socbench is scientific dataset intelligence &mdash; not another leaderboard, not a model benchmark.
          It&rsquo;s the IMDb for datasets. When someone is about to build a model, Socbench answers:
          &ldquo;Which datasets should I train on, and why?&rdquo;
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
            ["Utility", "How actionable this dataset is &mdash; format quality, documentation"],
            ["Documentation", "Dataset card completeness, license clarity"],
            ["Popularity", "Downloads, likes, community adoption"],
            ["Freshness", "How recently updated, actively maintained"],
            ["Contamination", "13-gram overlap with eval benchmarks (GSM8K, MMLU, HumanEval, MBPP)"],
            ["Repetition", "Exact-row duplication rate &mdash; catches slop and synthetic padding"],
            ["Training Impact", "GPT-2 124M loss curves, perplexity, convergence"],
          ].map(([label, desc]) => (
            <div key={label} className="border border-arxiv-border rounded p-3">
              <strong className="font-sans">{label}</strong>
              <p className="text-xs text-arxiv-gray mt-0.5" dangerouslySetInnerHTML={{ __html: desc }} />
            </div>
          ))}
        </div>

        <h3 className="text-lg font-serif font-bold pt-4">Hierarchical Classification</h3>
        <p className="text-arxiv-gray">
          Datasets aren&rsquo;t one thing. A Ferrari isn&rsquo;t &ldquo;better&rdquo; than a truck &mdash; they&rsquo;re built
          for different jobs. Categories include: Pretraining (web, code, math, science, books),
          SFT (instruction, assistant), Preference Optimization (DPO, RLHF), Tool Calling,
          Agent Traces, Reasoning, Task (QA, summarization, translation), Multimodal, and Safety.
          Each category has its own metrics &mdash; code datasets get parseability, agent datasets
          get trajectory completeness and loop detection, tool-calling datasets get format consistency.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Evaluation Benchmark Intelligence</h3>
        <p className="text-arxiv-gray">
          Not just datasets &mdash; Socbench also examines the evals themselves. The
          <a href="/evals" className="text-arxiv-link hover:text-arxiv-hover"> Evals page</a> analyzes
          the top 20 evaluation benchmarks (both open and closed source) for contamination risk and
          saturation. When an eval&rsquo;s data is publicly available, training datasets can
          accidentally include it &mdash; inflating scores and making the eval useless. When SOTA
          approaches the ceiling, the eval loses discriminative power. Socbench tracks both.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Contamination & Repetition</h3>
        <p className="text-arxiv-gray">
          Every dataset is checked for benchmark contamination using 13-gram overlap against
          GSM8K, MMLU, HumanEval, MBPP, and more. Datasets with &gt;1% overlap are flagged
          as contaminated. Repetition % measures exact-row duplication &mdash; a key signal for
          detecting synthetic slop and padded datasets. truthy-dpo at 99.9% repetition?
          That&rsquo;s a warning. Ultra-FineWeb at 0.45%? That&rsquo;s clean.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Trending Discovery</h3>
        <p className="text-arxiv-gray">
          The <a href="/trending" className="text-arxiv-link hover:text-arxiv-hover">Trending page</a>
          fetches real-time trending datasets from HuggingFace &mdash; dynamically, not hardcoded.
          Filter by time window (24h, 7d, 30d) and qualification status. New datasets are discovered
          and classified automatically using our hierarchical category system.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Request Evaluation</h3>
        <p className="text-arxiv-gray">
          Don&rsquo;t see a dataset you need? Use the &ldquo;Request Evaluation&rdquo; button on the leaderboard
          to submit any HuggingFace dataset ID for scoring. Choose public (results published
          on the leaderboard) or private (results shared with you only).
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Provenance Tracking</h3>
        <p className="text-arxiv-gray">
          Every dataset page shows which models were trained on it, which papers used it,
          and whether the provenance is verified. Open DeepSeek V4&rsquo;s page and see:
          FineWeb (quality 0.82), OpenCoder (quality 0.80), Stack V2 (quality 0.78).
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Evaluation Levels</h3>
        <p className="text-arxiv-gray">
          Socbench evaluates datasets in three levels, each more expensive and more
          informative than the last.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
          {[
            ["Level 1 &mdash; Automatic Scoring", "Cheap, fast heuristics. Gopher rules, FineWeb/DCLM filters, diversity, documentation, freshness, popularity."],
            ["Level 2 &mdash; Contamination & Repetition", "13-gram overlap with eval benchmarks (GSM8K, MMLU, HumanEval, MBPP). Exact-row duplication."],
            ["Level 3 &mdash; Training Impact", "Train GPT-2 124M on the dataset for 1B tokens. Final loss, perplexity, convergence, and relative quality."],
          ].map(([title, desc]) => (
            <div key={title} className="border border-arxiv-border rounded p-3">
              <strong className="font-sans text-sm block mb-1" dangerouslySetInnerHTML={{ __html: title }} />
              <p className="text-xs text-arxiv-gray" dangerouslySetInnerHTML={{ __html: desc }} />
            </div>
          ))}
        </div>

        <h3 className="text-lg font-serif font-bold pt-4">Training Impact (Stage 3)</h3>
        <p className="text-arxiv-gray">
          For top-tier datasets only. Train GPT-2 124M from scratch on each dataset
          for 1B tokens. Measure loss curves, perplexity, and convergence.
          Relative comparison across all runs. Research proves 125M models predict data
          quality scaling for 3B models (Ankner et al., 2024). This stage earns its
          compute &mdash; it runs only when the result answers a question nobody else can answer.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Discovery Pipeline</h3>
        <p className="text-arxiv-gray">
          We scan HuggingFace for new and trending datasets daily. Only datasets above
          minimum thresholds (&ge;1K downloads, &ge;10 likes, &ge;1K rows) enter the pipeline.
          Automated scoring runs instantly. Contamination checks flag leakage.
          Top-tier datasets queue for training impact measurement.
        </p>

        <h3 className="text-lg font-serif font-bold pt-4">Research Sources</h3>
        <ul className="list-disc pl-5 space-y-1 text-arxiv-gray text-xs">
          <li>Gopher quality rules &mdash; Rae et al., 2021</li>
          <li>FineWeb filters &mdash; Penedo et al., NeurIPS 2024</li>
          <li>DCLM filtering + fastText classifier &mdash; Li et al., NeurIPS 2024</li>
          <li>SlimPajama deduplication (13-gram, Jaccard 0.8) &mdash; Cerebras, 2023</li>
          <li>RedPajama-V2 46 quality signals &mdash; Weber et al., NeurIPS 2024</li>
          <li>Dolma curation methodology &mdash; Soldaini et al., ACL 2024</li>
          <li>125M proxy models predict 3B scaling &mdash; Ankner et al., 2024</li>
          <li>DoReMi domain reweighting &mdash; Xie et al., NeurIPS 2023</li>
          <li>DataComp standardized data evaluation &mdash; Gadre et al., NeurIPS 2023</li>
          <li>Perplexity-based data pruning &mdash; Ankner et al., 2024</li>
          <li>13-gram contamination detection &mdash; Brown et al. (GPT-3 paper), 2020</li>
          <li>Eval saturation analysis &mdash; LiveBench/LiveCodeBench methodology, 2024</li>
        </ul>
      </div>
    </div>
  );
}
