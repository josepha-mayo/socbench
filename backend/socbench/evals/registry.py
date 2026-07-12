"""Registry of top evaluation benchmarks — open and closed source.

Each eval entry includes:
- name, hf_id (if open), category, open_source flag
- known SOTA pass rates (for saturation detection) — sourced from public leaderboards 2024-2025
- contamination risk indicators — based on documented incidents
- description and paper reference
- deprecation status and successor

Saturation is detected by comparing current SOTA to theoretical ceiling.
Contamination risk is assessed by checking if the eval's data is publicly
available (open-source = higher contamination risk from training data overlap).

Sources: LiveBench (ICLR 2025), LiveCodeBench (arXiv:2403.07974),
"Leak, Cheat, Repeat" (EACL 2024), GSM1k study (arXiv:2405.00332),
SWE-Bench Pro, Humanity's Last Exam, and public model cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class EvalBenchmark:
    key: str
    name: str
    category: str  # code, math, reasoning, knowledge, safety, chat, agent
    open_source: bool
    hf_id: Optional[str] = None
    description: str = ""
    paper: str = ""
    num_examples: int = 0
    # SOTA pass rates (fraction, 0-1) — latest known results from public leaderboards (2024-2025)
    sota_pass_rate: float = 0.0
    # Theoretical ceiling (1.0 for most, <1.0 if some questions are ambiguous/noisy)
    ceiling: float = 1.0
    # Is the eval data publicly downloadable? (contamination risk factor)
    data_publicly_available: bool = True
    # Has this eval been deprecated or superseded?
    deprecated: bool = False
    # Successor eval if deprecated
    successor: Optional[str] = None
    # Year introduced
    year: int = 2023
    # Known contamination incidents (with paper references)
    contamination_incidents: list[str] = field(default_factory=list)
    # Discriminative power: how well does this eval separate models? (0=poor, 1=excellent)
    discriminative_power: float = 0.5
    # Annotation quality: expert-curated vs crowdsourced (0=poor, 1=expert)
    annotation_quality: float = 0.5
    # Coverage breadth: how many topics/domains does it cover? (0=narrow, 1=broad)
    coverage_breadth: float = 0.5
    # Is this eval actively maintained/updated?
    actively_maintained: bool = True


# Top 20 evaluation benchmarks — both open and closed source
# SOTA scores sourced from public leaderboards, model cards, and research papers (2024-2025)
EVALS: dict[str, EvalBenchmark] = {
    # --- Open-source evals (data publicly available — HIGH contamination risk) ---
    "mmlu": EvalBenchmark(
        key="mmlu",
        name="MMLU",
        category="knowledge",
        open_source=True,
        hf_id="cais/mmlu",
        description="Massive Multitask Language Understanding — 57 subjects, multiple choice. The most widely used knowledge benchmark.",
        paper="Hendrycks et al., 2021",
        num_examples=14042,
        sota_pass_rate=0.920,  # Claude Opus 4, 2025
        ceiling=0.99,  # ~1% label noise documented
        year=2021,
        deprecated=True,
        successor="mmlu-pro",
        discriminative_power=0.3,  # Saturated — poor at separating frontier models
        annotation_quality=0.7,   # Academic-sourced, some noise
        coverage_breadth=0.9,     # 57 subjects
        actively_maintained=False,
        contamination_incidents=[
            "Found in pretraining corpora of multiple open models (RedPajama, StarCoder)",
            "Academic .docx/.PDF sources easily scraped into training data",
            "GPT-4o scores 90.2% — within annotation noise of ceiling",
            "Deprecated in favor of MMLU-Pro (10 answer choices, harder questions)",
        ],
    ),
    "humaneval": EvalBenchmark(
        key="humaneval",
        name="HumanEval",
        category="code",
        open_source=True,
        hf_id="openai/openai_humaneval",
        description="164 hand-written programming problems with function-level unit tests. The standard code generation benchmark.",
        paper="Chen et al., 2021 (Codex)",
        num_examples=164,
        sota_pass_rate=0.927,  # Claude 3.5 Sonnet / Claude Opus 4
        ceiling=1.0,
        year=2021,
        deprecated=True,
        successor="livecodebench",
        discriminative_power=0.2,  # Fully saturated — 164 problems too few
        annotation_quality=0.8,   # Hand-written by OpenAI
        coverage_breadth=0.3,     # Only 164 problems, narrow
        actively_maintained=False,
        contamination_incidents=[
            "8-18% overlap found in RedPajama-Data-1T and StarCoder-Data (arXiv:2311.04850)",
            "Widely duplicated in code training corpora — only 164 problems, easy to memorize",
            "Models can reproduce solutions verbatim from training data",
            "Successor: LiveCodeBench (time-segmented, contamination-free)",
        ],
    ),
    "gsm8k": EvalBenchmark(
        key="gsm8k",
        name="GSM8K",
        category="math",
        open_source=True,
        hf_id="openai/gsm8k",
        description="Grade School Math 8K — word problems requiring 2-8 reasoning steps. The standard math reasoning benchmark.",
        paper="Cobbe et al., 2021",
        num_examples=1319,
        sota_pass_rate=0.964,  # Claude 3.5 Sonnet / GPT-4o
        ceiling=0.99,
        year=2021,
        deprecated=True,
        successor=" humanitys-last-exam",
        discriminative_power=0.2,  # Fully saturated
        annotation_quality=0.8,   # Human-written, verified
        coverage_breadth=0.4,     # Grade school level only
        actively_maintained=False,
        contamination_incidents=[
            "GSM1k study shows accuracy drops up to 8% on held-out version (arXiv:2405.00332)",
            "Spearman's r²=0.36 between GSM8K generation probability and performance gap",
            "Common in math training data; many models trained on it directly",
            "1,319 problems — small enough to memorize entirely",
        ],
    ),
    "mbpp": EvalBenchmark(
        key="mbpp",
        name="MBPP",
        category="code",
        open_source=True,
        hf_id="google-research-datasets/mbpp",
        description="Mostly Basic Python Problems — 974 entry-level programming tasks with test cases.",
        paper="Austin et al., 2021",
        num_examples=974,
        sota_pass_rate=0.949,  # Saturated
        ceiling=1.0,
        year=2021,
        deprecated=True,
        successor="mbpp-plus",
        discriminative_power=0.2,  # Saturated
        annotation_quality=0.7,
        coverage_breadth=0.3,     # Basic Python only
        actively_maintained=False,
        contamination_incidents=[
            "Similar contamination patterns as HumanEval in code corpora",
            "Models perform significantly better on subset with similar solutions in training (ACL 2024)",
            "Successor: MBPP+ (EvalPlus, 80x more test cases)",
        ],
    ),
    "hellaswag": EvalBenchmark(
        key="hellaswag",
        name="HellaSwag",
        category="reasoning",
        open_source=True,
        hf_id="Rowan/hellaswag",
        description="Adversarial commonsense reasoning — choose the best story continuation. 10K questions.",
        paper="Zellers et al., 2019",
        num_examples=10000,
        sota_pass_rate=0.964,  # GPT-5, 2025
        ceiling=0.95,  # Annotation noise at top
        year=2019,
        deprecated=True,
        successor=None,  # Commonsense reasoning considered saturated
        discriminative_power=0.1,  # Fully saturated — differences within annotation noise
        annotation_quality=0.7,
        coverage_breadth=0.6,
        actively_maintained=False,
        contamination_incidents=[
            "Included in pretraining mix of most LLMs — internet-sourced, high popularity",
            "GPT-5 scores 96.4% — within annotation noise of 95% ceiling",
            "Differences at 95%+ are within annotation noise — no discriminative power",
        ],
    ),
    "arc": EvalBenchmark(
        key="arc",
        name="ARC-Challenge",
        category="knowledge",
        open_source=True,
        hf_id="allenai/ai2_arc",
        description="AI2 Reasoning Challenge — grade-school science questions requiring reasoning. Non-trivial for AI but easy for humans.",
        paper="Clark et al., 2018",
        num_examples=1172,
        sota_pass_rate=0.96,  # With proper evaluation (options shown simultaneously)
        ceiling=0.97,
        year=2018,
        deprecated=False,  # Not officially deprecated but effectively saturated
        successor=None,
        discriminative_power=0.2,
        annotation_quality=0.8,   # Expert-written science questions
        coverage_breadth=0.4,     # Science only
        actively_maintained=False,
        contamination_incidents=[
            "Evaluation artifact: separation vs options scoring changes results dramatically (ACL 2025)",
            "Llama 3.1 70B jumps from 64% to 93% when options shown simultaneously",
            "Academic test-based — less indexed than MMLU but still scrapable",
        ],
    ),
    "truthfulqa": EvalBenchmark(
        key="truthfulqa",
        name="TruthfulQA",
        category="safety",
        open_source=True,
        hf_id="truthfulqa/truthful_qa",
        description="Measures whether models avoid imitating human falsehoods and misconceptions. 817 questions across 38 categories.",
        paper="Lin et al., 2022",
        num_examples=817,
        sota_pass_rate=0.78,  # Claude Opus 4.5, 2025
        ceiling=1.0,
        year=2022,
        deprecated=False,
        successor=None,
        discriminative_power=0.5,  # Still has headroom
        annotation_quality=0.8,   # Expert-written
        coverage_breadth=0.7,     # 38 categories of misconceptions
        actively_maintained=True,
        contamination_incidents=[
            "Manually constructed — lower contamination risk than scraped benchmarks",
            "New binary-choice version (Jan 2025) prevents test-taking heuristics",
            "Generation vs MC gap: ~15-20% (models know correct answer but generate false one)",
        ],
    ),
    "winogrande": EvalBenchmark(
        key="winogrande",
        name="WinoGrande",
        category="reasoning",
        open_source=True,
        hf_id="allenai/winogrande",
        description="Coreference resolution with adversarial filtering — Winograd schema challenge at scale.",
        paper="Sakaguchi et al., 2021",
        num_examples=1267,
        sota_pass_rate=0.91,  # GPT-4o
        ceiling=0.94,  # Human performance
        year=2021,
        deprecated=False,
        successor=None,
        discriminative_power=0.3,  # Near ceiling
        annotation_quality=0.7,
        coverage_breadth=0.4,
        actively_maintained=False,
        contamination_incidents=[
            "Included in pretraining data of most LLMs",
            "Human performance is 94% — SOTA at 91% is approaching ceiling",
        ],
    ),
    "bbh": EvalBenchmark(
        key="bbh",
        name="BBH (Big-Bench Hard)",
        category="reasoning",
        open_source=True,
        hf_id="lukaemon/bbh",
        description="23 difficult Big-Bench tasks where models significantly underperform humans. 6,511 questions.",
        paper="Suzgun et al., 2022",
        num_examples=6511,
        sota_pass_rate=0.889,  # Qwen3 235B, 2025
        ceiling=1.0,
        year=2022,
        deprecated=False,
        successor=None,
        discriminative_power=0.5,  # Still separates models
        annotation_quality=0.8,   # Curated from Big-Bench
        coverage_breadth=0.7,     # 23 diverse tasks
        actively_maintained=False,
        contamination_incidents=[
            "Public data — included in some training corpora",
            "Still has ~11% headroom to ceiling — moderate discriminative power",
        ],
    ),
    "gpqa": EvalBenchmark(
        key="gpqa",
        name="GPQA Diamond",
        category="knowledge",
        open_source=True,
        hf_id="Idavidrein/GPQA",
        description="Google-Proof Q&A — PhD-level science questions, resistant to web search. 198 questions.",
        paper="Rein et al., 2023",
        num_examples=198,
        sota_pass_rate=0.941,  # Gemini 3.1 Pro Preview, 2026
        ceiling=1.0,
        year=2023,
        deprecated=False,
        successor=None,
        discriminative_power=0.7,  # Good separation — human expert only 65%
        annotation_quality=0.9,   # PhD-level experts
        coverage_breadth=0.5,     # Science domains
        actively_maintained=True,
        contamination_incidents=[
            "Designed to be Google-proof — lower contamination risk",
            "Human expert performance is only 65% — SOTA at 94% suggests possible memorization",
            "Small dataset (198 questions) — risk of memorization despite design",
        ],
    ),
    "ifeval": EvalBenchmark(
        key="ifeval",
        name="IFEval",
        category="chat",
        open_source=True,
        hf_id="google/IFEval",
        description="Instruction Following Evaluation — 540 prompts with 25 types of verifiable instruction-following constraints.",
        paper="Zhou et al., 2023",
        num_examples=540,
        sota_pass_rate=0.926,  # Qwen 3.5, 2025
        ceiling=1.0,
        year=2023,
        deprecated=False,
        successor=None,
        discriminative_power=0.5,  # Still has headroom
        annotation_quality=0.8,   # Verifiable constraints
        coverage_breadth=0.6,     # 25 instruction types
        actively_maintained=True,
        contamination_incidents=[
            "Verifiable constraints — harder to game through memorization",
            "Public data but constraint-based design reduces contamination impact",
        ],
    ),
    "swe-bench": EvalBenchmark(
        key="swe-bench",
        name="SWE-Bench",
        category="agent",
        open_source=True,
        hf_id="princeton-nlp/SWE-bench",
        description="Software engineering tasks from real GitHub issues — agent capability benchmark. 2,294 tasks.",
        paper="Jimenez et al., 2024",
        num_examples=2294,
        sota_pass_rate=0.33,  # Claude 3.5 Sonnet (original)
        ceiling=1.0,
        year=2024,
        deprecated=False,  # Original still used but Verified is deprecated
        successor="swe-bench-pro",
        discriminative_power=0.8,  # Excellent — lots of headroom
        annotation_quality=0.8,   # Real GitHub issues
        coverage_breadth=0.6,     # Multiple repos
        actively_maintained=True,
        contamination_incidents=[
            "Problems sourced from open-source repos used in training",
            "All frontier models can reproduce gold patches verbatim",
            "Original version still has contamination concerns but lots of headroom",
        ],
    ),
    # --- Closed/semi-closed source evals (data not fully public — LOWER contamination risk) ---
    "swe-bench-verified": EvalBenchmark(
        key="swe-bench-verified",
        name="SWE-Bench Verified",
        category="agent",
        open_source=False,
        hf_id="princeton-nlp/SWE-bench_Verified",
        description="Human-verified subset of SWE-Bench with validated test cases. 500 instances.",
        paper="Jimenez et al., 2024",
        num_examples=500,
        sota_pass_rate=0.803,  # Claude Fable 5, 2025 (was 23% in 2024)
        ceiling=1.0,
        data_publicly_available=True,  # data is public but test harness is controlled
        year=2024,
        deprecated=True,
        successor="swe-bench-pro",
        discriminative_power=0.6,  # Was excellent but now saturating
        annotation_quality=0.9,   # Human-verified
        coverage_breadth=0.5,
        actively_maintained=False,
        contamination_incidents=[
            "OpenAI stopped reporting SWE-Bench Verified (Aug 2024) due to contamination",
            "All frontier models can reproduce gold patches verbatim",
            "Progress: 23.3% → 80.3% in 8 months — suggests memorization",
            "Successor: SWE-Bench Pro (private commercial subset, held-out repos)",
        ],
    ),
    "livecodebench": EvalBenchmark(
        key="livecodebench",
        name="LiveCodeBench",
        category="code",
        open_source=True,
        hf_id="livecodebench/livecodebench",
        description="Contamination-free coding eval with problems dated after model training cutoffs. 611 problems from LeetCode, AtCoder, CodeForces.",
        paper="Jain et al., 2024 (arXiv:2403.07974)",
        num_examples=611,
        sota_pass_rate=0.935,  # 2025
        ceiling=1.0,
        data_publicly_available=True,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.7,  # Time-segmented evaluation is excellent
        annotation_quality=0.9,   # Competition problems
        coverage_breadth=0.6,     # Multiple platforms
        actively_maintained=True,
        contamination_incidents=[
            "Designed to be contamination-free via time-segmented evaluation",
            "DeepSeek: performance drop on problems released after Aug 2023",
            "GPT-4o: performance drop on problems released after Oct 2023",
            "Codestral: performance drop on problems released after Jan 2024",
            "Low contamination risk by design",
        ],
    ),
    "livebench": EvalBenchmark(
        key="livebench",
        name="LiveBench",
        category="knowledge",
        open_source=True,
        hf_id="livebench/livebench",
        description="Contamination-free benchmark with monthly updates — questions from recent arXiv papers, news, competitions.",
        paper="White et al., 2024 (ICLR 2025 Spotlight, arXiv:2406.19314)",
        num_examples=1000,
        sota_pass_rate=0.45,  # Low by design — stays fresh
        ceiling=1.0,
        data_publicly_available=True,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.9,  # Excellent — designed to stay fresh
        annotation_quality=0.8,
        coverage_breadth=0.8,     # Multiple domains, monthly updates
        actively_maintained=True,
        contamination_incidents=[
            "Designed to be contamination-free via monthly fresh questions",
            "Top models achieve <70% — intentionally designed to remain unsaturated",
            "Questions from recent sources (arXiv, news, competitions) post-model-cutoff",
            "Very low contamination risk by design",
        ],
    ),
    "arena-hard": EvalBenchmark(
        key="arena-hard",
        name="Arena-Hard-200",
        category="chat",
        open_source=False,
        description="500 hard user queries from Chatbot Arena — LLM judge evaluation. Highest correlation (0.98) with Chatbot Arena.",
        paper="Li et al., 2024",
        num_examples=500,
        sota_pass_rate=0.859,  # o3, 2025
        ceiling=1.0,
        data_publicly_available=False,  # Queries are public but judging is controlled
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.7,  # Good separation
        annotation_quality=0.8,   # Real user queries
        coverage_breadth=0.7,     # Diverse user queries
        actively_maintained=True,
        contamination_incidents=[
            "Queries sourced from Chatbot Arena — semi-private",
            "LLM judge evaluation reduces memorization impact",
            "Correlation 0.98 with Chatbot Arena — gold standard for chat eval",
        ],
    ),
    "alpacaeval2": EvalBenchmark(
        key="alpacaeval2",
        name="AlpacaEval 2.0",
        category="chat",
        open_source=True,
        hf_id="tatsu-lab/alpaca_eval",
        description="Length-controlled win rate against GPT-4 Turbo — 805 open-ended questions. Cost <$10, <3 minutes.",
        paper="Li et al., 2023",
        num_examples=805,
        sota_pass_rate=0.75,  # Claude 3.5
        ceiling=1.0,
        data_publicly_available=True,
        year=2023,
        deprecated=False,
        successor=None,
        discriminative_power=0.6,
        annotation_quality=0.7,
        coverage_breadth=0.6,
        actively_maintained=True,
        contamination_incidents=[
            "Vulnerability: null model achieves 86.5% LC win rate (cheating possible)",
            "Public questions but LLM judge reduces memorization impact",
            "Correlation 0.98 with Chatbot Arena",
        ],
    ),
    "mtbench": EvalBenchmark(
        key="mtbench",
        name="MT-Bench",
        category="chat",
        open_source=True,
        hf_id="HuggingFaceH4/mt_bench",
        description="Multi-turn benchmark — 80 conversations across 8 categories, scored by GPT-4 judge.",
        paper="Zheng et al., 2023",
        num_examples=80,
        sota_pass_rate=0.90,
        ceiling=0.95,
        data_publicly_available=True,
        year=2023,
        deprecated=False,
        successor=None,
        discriminative_power=0.3,  # Small dataset, near ceiling
        annotation_quality=0.7,
        coverage_breadth=0.5,     # 8 categories
        actively_maintained=False,
        contamination_incidents=[
            "Vulnerability: null model achieves 9.55 score (gaming possible)",
            "Only 80 questions — small dataset, easy to memorize",
            "Public data — included in some training corpora",
        ],
    ),
    "agieval": EvalBenchmark(
        key="agieval",
        name="AGIEval",
        category="knowledge",
        open_source=True,
        hf_id="hails/agieval",
        description="Human-centric standardized exam problems (SAT, LSAT, GRE, civil service). 8,000 questions.",
        paper="Zhong et al., 2023",
        num_examples=8000,
        sota_pass_rate=0.70,
        ceiling=1.0,
        year=2023,
        deprecated=False,
        successor=None,
        discriminative_power=0.5,  # Still has headroom
        annotation_quality=0.8,   # Real standardized tests
        coverage_breadth=0.7,     # Multiple exam types
        actively_maintained=True,
        contamination_incidents=[
            "Standardized test problems — publicly available, scrapable",
            "Some overlap with MMLU-style content",
            "Still has 30% headroom — moderate discriminative power",
        ],
    ),
    "agents-last-exam": EvalBenchmark(
        key="agents-last-exam",
        name="Agents Last Exam",
        category="agent",
        open_source=True,
        hf_id="rdi-berkeley/agents-last-exam",
        description="Comprehensive agent capability benchmark — tool use, planning, execution. 1,500+ tasks across 55 industries.",
        paper="arXiv:2606.05405",
        num_examples=1500,
        sota_pass_rate=0.01,  # Hardest tier full pass < 1%
        ceiling=1.0,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.9,
        annotation_quality=0.8,
        coverage_breadth=0.8,
        actively_maintained=True,
        contamination_incidents=[
            "150 public tasks available; full corpus largely private",
            "Economically valuable real-world tasks — low memorization risk",
        ],
    ),
    "bigcodebench": EvalBenchmark(
        key="bigcodebench",
        name="BigCodeBench",
        category="code",
        open_source=True,
        hf_id="bigcode/bigcodebench",
        description="Code generation with diverse function calls and complex instructions — 1,140 tasks across 139 libraries and 7 domains.",
        paper="arXiv:2406.15877 (ICLR 2025)",
        num_examples=1140,
        sota_pass_rate=0.592,  # DeepSeek V4 Pro
        ceiling=1.0,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.9,
        coverage_breadth=0.7,
        actively_maintained=True,
        contamination_incidents=[
            "Public dataset — but function-level diversity reduces verbatim memorization",
            "Harder than HumanEval: tests 139 libraries and complex instructions",
        ],
    ),
    "swe-bench-pro": EvalBenchmark(
        key="swe-bench-pro",
        name="SWE-Bench Pro",
        category="agent",
        open_source=False,
        hf_id="ScaleAI/SWE-bench_Pro",
        description="Long-horizon enterprise-level software engineering — 1,865 problems across 41 repositories (public + held-out + commercial).",
        paper="arXiv:2509.16941",
        num_examples=1865,
        sota_pass_rate=0.233,  # GPT-5 on public subset
        ceiling=1.0,
        data_publicly_available=False,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.9,
        annotation_quality=0.8,
        coverage_breadth=0.7,
        actively_maintained=True,
        contamination_incidents=[
            "Public set (11 repos) open; held-out and commercial subsets private",
            "Long-horizon tasks reduce memorization of gold patches",
        ],
    ),
    "deepswe": EvalBenchmark(
        key="deepswe",
        name="DeepSWE",
        category="agent",
        open_source=True,
        hf_id="datacurve/deep-swe",
        description="Long-horizon software engineering — 113 original tasks across TypeScript, Go, Python, JavaScript, Rust.",
        paper="arXiv:2607.07946",
        num_examples=113,
        sota_pass_rate=0.25,  # Frontier models below 25%
        ceiling=1.0,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.9,
        coverage_breadth=0.5,
        actively_maintained=True,
        contamination_incidents=[
            "Contamination-free: tasks written from scratch",
            "Hand-written program-based verifiers",
        ],
    ),
    "programbench": EvalBenchmark(
        key="programbench",
        name="ProgramBench",
        category="code",
        open_source=True,
        hf_id="programbench/ProgramBench-Tests",
        description="Program reconstruction from executables — rebuild programs like jq, ripgrep, FFmpeg, SQLite from scratch (200 tasks).",
        paper="arXiv:2605.03546",
        num_examples=200,
        sota_pass_rate=0.15,  # Frontier models struggle
        ceiling=1.0,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.85,
        annotation_quality=0.8,
        coverage_breadth=0.5,
        actively_maintained=True,
        contamination_incidents=[
            "Reconstruction from binaries is not in standard training corpora",
            "Open-source tasks but low overlap with text training data",
        ],
    ),
    "terminalbench": EvalBenchmark(
        key="terminalbench",
        name="TerminalBench 2.0",
        category="agent",
        open_source=True,
        hf_id="harbor-framework/terminal-bench-2",
        description="AI agents in terminal environments — 89 tasks across software engineering, ML, security, data science.",
        paper="arXiv:2601.11868",
        num_examples=89,
        sota_pass_rate=0.65,  # Frontier models below 65%
        ceiling=1.0,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.8,
        coverage_breadth=0.6,
        actively_maintained=True,
        contamination_incidents=[
            "Terminal environment tasks are hard to scrape into training corpora",
            "Requires multi-step execution in sandbox",
        ],
    ),
    "aider-polyglot": EvalBenchmark(
        key="aider-polyglot",
        name="Aider Polyglot",
        category="code",
        open_source=True,
        hf_id="Aider-AI/polyglot-benchmark",
        description="Code editing across multiple languages — 225 Exercism exercises in C++, Go, Java, JavaScript, Python, Rust.",
        paper="https://aider.chat/2024/12/21/polyglot.html",
        num_examples=225,
        sota_pass_rate=0.75,  # o1 high reasoning
        ceiling=1.0,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.7,
        annotation_quality=0.8,
        coverage_breadth=0.6,
        actively_maintained=True,
        contamination_incidents=[
            "Exercism problems are public, but multi-language editing reduces verbatim risk",
            "Focus on code editing rather than generation",
        ],
    ),
    "humaneval-plus": EvalBenchmark(
        key="humaneval-plus",
        name="HumanEval+",
        category="code",
        open_source=True,
        hf_id="evalplus/humanevalplus",
        description="HumanEval with 80x more rigorous test coverage — 164 problems.",
        paper="EvalPlus (NeurIPS 2023)",
        num_examples=164,
        sota_pass_rate=0.90,
        ceiling=1.0,
        year=2023,
        deprecated=False,
        successor=None,
        discriminative_power=0.5,
        annotation_quality=0.9,
        coverage_breadth=0.3,
        actively_maintained=True,
        contamination_incidents=[
            "Expanded tests reduce false positives from memorized solutions",
            "Still based on public HumanEval problems",
        ],
    ),
    "mbpp-plus": EvalBenchmark(
        key="mbpp-plus",
        name="MBPP+",
        category="code",
        open_source=True,
        hf_id="evalplus/mbppplus",
        description="MBPP with 35x more rigorous test coverage — 974 problems.",
        paper="EvalPlus (COLM 2024)",
        num_examples=974,
        sota_pass_rate=0.85,
        ceiling=1.0,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.5,
        annotation_quality=0.9,
        coverage_breadth=0.4,
        actively_maintained=True,
        contamination_incidents=[
            "Expanded tests reduce false positives from memorized solutions",
            "Still based on public MBPP problems",
        ],
    ),
    "mmlu-pro": EvalBenchmark(
        key="mmlu-pro",
        name="MMLU-Pro",
        category="knowledge",
        open_source=True,
        hf_id="TIGER-Lab/MMLU-Pro",
        description="MMLU with 12K complex questions and 10 answer choices (vs 4 in MMLU).",
        paper="arXiv:2406.01574 (NeurIPS 2024)",
        num_examples=12000,
        sota_pass_rate=0.80,
        ceiling=1.0,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.6,
        annotation_quality=0.8,
        coverage_breadth=0.9,
        actively_maintained=True,
        contamination_incidents=[
            "Public dataset — higher contamination risk than closed evals",
            "More choices and harder questions make memorization harder",
        ],
    ),
    "humanitys-last-exam": EvalBenchmark(
        key="humanitys-last-exam",
        name="Humanity's Last Exam",
        category="knowledge",
        open_source=True,
        hf_id="cais/hle",
        description="Expert-level academic questions across 100+ subjects — 2,500 multi-modal questions.",
        paper="Nature 2026",
        num_examples=2500,
        sota_pass_rate=0.645,  # Claude Fable 5 / Mythos 5
        ceiling=1.0,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.9,
        coverage_breadth=0.9,
        actively_maintained=True,
        contamination_incidents=[
            "Public dataset on HuggingFace; private test set held back",
            "Designed as final closed-ended academic benchmark",
        ],
    ),
    "frontierbench": EvalBenchmark(
        key="frontierbench",
        name="FrontierBench",
        category="agent",
        open_source=False,
        description="Generalist agents on novel problems beyond familiar search-type tasks.",
        paper="OpenReview 2025",
        num_examples=300,
        sota_pass_rate=0.50,  # GPT-5 fails ~50% of execution tasks
        ceiling=1.0,
        data_publicly_available=False,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.85,
        annotation_quality=0.8,
        coverage_breadth=0.6,
        actively_maintained=True,
        contamination_incidents=[
            "Automated problem generation workflow reduces contamination",
            "Private/full evaluation not publicly available",
        ],
    ),
    "cursorbench": EvalBenchmark(
        key="cursorbench",
        name="CursorBench",
        category="code",
        open_source=False,
        description="Multi-file coding-agent tasks from real Cursor sessions — ambiguous, realistic developer workflows.",
        paper="https://cursor.com/blog/cursorbench",
        num_examples=500,
        sota_pass_rate=0.705,  # Claude Fable 5 Max
        ceiling=1.0,
        data_publicly_available=False,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.9,
        coverage_breadth=0.6,
        actively_maintained=True,
        contamination_incidents=[
            "First-party benchmark — internal to Cursor, not public",
            "Tasks from real Cursor sessions, lower contamination risk",
        ],
    ),
    "swe-lancer": EvalBenchmark(
        key="swe-lancer",
        name="SWE-Lancer",
        category="agent",
        open_source=True,
        hf_id="openai/frontier-evals",
        description="Real freelance software engineering tasks from Upwork — 1,400+ tasks valued at $1M.",
        paper="arXiv:2502.12115 (ICML 2025)",
        num_examples=1400,
        sota_pass_rate=0.25,  # Frontier models unable to solve majority
        ceiling=1.0,
        data_publicly_available=False,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.9,
        annotation_quality=0.8,
        coverage_breadth=0.7,
        actively_maintained=True,
        contamination_incidents=[
            "Public evaluation split (Diamond) available; full dataset partially restricted",
            "Real Upwork tasks unlikely to be in training corpora",
        ],
    ),
    "paperbench": EvalBenchmark(
        key="paperbench",
        name="PaperBench",
        category="agent",
        open_source=True,
        hf_id="openai/frontier-evals",
        description="AI agents replicating AI research papers — 20 ICML 2024 papers, 8,316 gradable tasks.",
        paper="arXiv:2504.01848",
        num_examples=8316,
        sota_pass_rate=0.21,  # Claude 3.5 Sonnet with open scaffolding
        ceiling=1.0,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.9,
        annotation_quality=0.9,
        coverage_breadth=0.7,
        actively_maintained=True,
        contamination_incidents=[
            "Open code but dataset requires Git LFS",
            "Paper replication is not in standard training data",
        ],
    ),
    "frontiermath": EvalBenchmark(
        key="frontiermath",
        name="FrontierMath",
        category="math",
        open_source=False,
        description="Advanced mathematical reasoning — hundreds of original research-level problems.",
        paper="arXiv:2411.04872",
        num_examples=300,
        sota_pass_rate=0.02,  # SOTA < 2%
        ceiling=1.0,
        data_publicly_available=False,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.9,
        annotation_quality=0.95,
        coverage_breadth=0.6,
        actively_maintained=True,
        contamination_incidents=[
            "Sample problems public; full benchmark requires contact",
            "Original problems by 60+ mathematicians — low contamination",
        ],
    ),
    "wildbench": EvalBenchmark(
        key="wildbench",
        name="WildBench",
        category="chat",
        open_source=True,
        hf_id="allenai/WildBench",
        description="Real-world user queries from WildChat logs — 1,024 tasks with automated evaluation.",
        paper="arXiv:2406.04069",
        num_examples=1024,
        sota_pass_rate=0.80,
        ceiling=1.0,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.8,
        coverage_breadth=0.8,
        actively_maintained=True,
        contamination_incidents=[
            "WB-Reward correlates 0.98 with Chatbot Arena Elo",
            "Public dataset but based on real user queries",
        ],
    ),
    "simplebench": EvalBenchmark(
        key="simplebench",
        name="SimpleBench",
        category="knowledge",
        open_source=True,
        hf_id="Impulse2000/simple_bench_public-20-12-2024",
        description="Text benchmark where unspecialized human performance exceeds frontier models.",
        paper="simple-bench.com",
        num_examples=500,
        sota_pass_rate=0.45,
        ceiling=1.0,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.7,
        coverage_breadth=0.5,
        actively_maintained=True,
        contamination_incidents=[
            "Public dataset but designed to be hard for LLMs",
            "Focuses on tasks where humans still outperform models",
        ],
    ),
    "evmbench": EvalBenchmark(
        key="evmbench",
        name="EVMbench",
        category="code",
        open_source=True,
        hf_id="paradigmxyz/evmbench",
        description="Smart contract security — detecting, patching, exploiting vulnerabilities.",
        paper="https://paradigm.xyz/evmbench",
        num_examples=200,
        sota_pass_rate=0.35,
        ceiling=1.0,
        year=2025,
        deprecated=False,
        successor=None,
        discriminative_power=0.8,
        annotation_quality=0.8,
        coverage_breadth=0.4,
        actively_maintained=True,
        contamination_incidents=[
            "Open source but focused on smart contract security",
            "Requires execution in EVM sandbox",
        ],
    ),
    "math-500": EvalBenchmark(
        key="math-500",
        name="MATH-500",
        category="math",
        open_source=True,
        hf_id="HuggingFaceH4/MATH-500",
        description="Mathematical problem-solving — 500 problems from MATH benchmark across 6 domains.",
        paper="https://github.com/openai/prm800k",
        num_examples=500,
        sota_pass_rate=0.85,
        ceiling=1.0,
        year=2024,
        deprecated=False,
        successor=None,
        discriminative_power=0.6,
        annotation_quality=0.8,
        coverage_breadth=0.5,
        actively_maintained=True,
        contamination_incidents=[
            "Subset of MATH dataset — public",
            "Widely used in math training; contamination possible",
        ],
    ),
}


def get_eval(key: str) -> Optional[EvalBenchmark]:
    return EVALS.get(key)


def get_all_evals() -> list[EvalBenchmark]:
    return list(EVALS.values())


def get_evals_by_category(category: str) -> list[EvalBenchmark]:
    return [e for e in EVALS.values() if e.category == category]
