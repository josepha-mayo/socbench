"""Category definitions for hierarchical dataset classification.

Datasets are classified into hierarchical categories with different
quality metrics per category. Like cars — a Ferrari isn't "better" than
a truck, they're built for different jobs.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    icon: str = ""
    description: str = ""
    parent: str | None = None
    metrics: list[str] = field(default_factory=list)


# Hierarchical category tree
CATEGORIES = {
    "pretraining": Category(
        key="pretraining",
        label="Pretraining",
        description="Large-scale unsupervised training data",
        metrics=["quality", "diversity", "dedup_rate", "coverage", "freshness"],
    ),
    "pretraining-web": Category(
        key="pretraining-web",
        label="General Web",
        parent="pretraining",
        metrics=["quality", "diversity", "dedup_rate", "pii_safety", "language_purity", "freshness"],
    ),
    "pretraining-code": Category(
        key="pretraining-code",
        label="Code",
        parent="pretraining",
        metrics=["quality", "diversity", "parse_rate", "boilerplate_rate", "language_coverage", "repo_diversity"],
    ),
    "pretraining-math": Category(
        key="pretraining-math",
        label="Math",
        parent="pretraining",
        metrics=["quality", "diversity", "difficulty", "format_consistency"],
    ),
    "pretraining-science": Category(
        key="pretraining-science",
        label="Science",
        parent="pretraining",
        metrics=["quality", "diversity", "domain_coverage", "language_purity"],
    ),
    "pretraining-books": Category(
        key="pretraining-books",
        label="Books",
        parent="pretraining",
        metrics=["quality", "diversity", "language_purity", "narrative_coherence"],
    ),
    "pretraining-multilingual": Category(
        key="pretraining-multilingual",
        label="Multilingual",
        parent="pretraining",
        metrics=["quality", "diversity", "language_coverage", "translation_quality"],
    ),
    "posttraining": Category(
        key="posttraining",
        label="Post-Training",
        description="SFT, preference, alignment data",
        metrics=["quality", "diversity", "instruction_variety", "safety"],
    ),
    "posttraining-sft": Category(
        key="posttraining-sft",
        label="Instruction / SFT",
        parent="posttraining",
        metrics=["quality", "diversity", "instruction_variety", "response_length", "reasoning_depth", "refusal_balance"],
    ),
    "posttraining-preference": Category(
        key="posttraining-preference",
        label="Preference / DPO",
        parent="posttraining",
        metrics=["quality", "diversity", "pair_quality", "preference_consistency", "safety"],
    ),
    "posttraining-tooluse": Category(
        key="posttraining-tooluse",
        label="Tool Use",
        parent="posttraining",
        metrics=["quality", "diversity", "tool_correctness", "format_consistency"],
    ),
    "posttraining-agent": Category(
        key="posttraining-agent",
        label="Agent Trajectories",
        parent="posttraining",
        metrics=["quality", "diversity", "trajectory_completeness", "recovery_rate", "loop_detection", "tool_correctness"],
    ),
    "posttraining-safety": Category(
        key="posttraining-safety",
        label="Safety",
        parent="posttraining",
        metrics=["quality", "diversity", "safety_coverage", "refusal_rate"],
    ),
    "posttraining-reasoning": Category(
        key="posttraining-reasoning",
        label="Reasoning",
        parent="posttraining",
        metrics=["quality", "diversity", "reasoning_depth", "step_count", "format_consistency"],
    ),
    "evaluation": Category(
        key="evaluation",
        label="Evaluation",
        description="Benchmarks for measuring model capability",
        metrics=["quality", "contamination", "difficulty", "discrimination"],
    ),
    "task": Category(
        key="task",
        label="Task Datasets",
        description="Structured task data (classification, translation, etc.)",
        metrics=["quality", "label_quality", "format_consistency", "class_balance"],
    ),
    "task-classification": Category(
        key="task-classification",
        label="Classification",
        parent="task",
        metrics=["quality", "label_quality", "class_balance", "format_consistency"],
    ),
    "task-translation": Category(
        key="task-translation",
        label="Translation",
        parent="task",
        metrics=["quality", "translation_quality", "language_pair_coverage"],
    ),
    "task-qa": Category(
        key="task-qa",
        label="Question Answering",
        parent="task",
        metrics=["quality", "answer_accuracy", "question_diversity", "format_consistency"],
    ),
    "task-summarization": Category(
        key="task-summarization",
        label="Summarization",
        parent="task",
        metrics=["quality", "summary_quality", "length_ratio", "format_consistency"],
    ),
    "multimodal": Category(
        key="multimodal",
        label="Multimodal",
        description="Image-text, video, audio, VLM data",
        metrics=["quality", "diversity", "modality_balance", "resolution_consistency"],
    ),
}

# Mapping from HuggingFace task tags to our categories
HF_TAG_TO_CATEGORY = {
    "text-generation": "pretraining-web",  # Default for text gen
    "code": "pretraining-code",
    "math": "pretraining-math",
    "question-answering": "task-qa",
    "summarization": "task-summarization",
    "translation": "task-translation",
    "text-classification": "task-classification",
    "visual-question-answering": "multimodal",
    "image-text-to-text": "multimodal",
    "automatic-speech-recognition": "multimodal",
}


# Known evaluation benchmark name fragments.
_EVAL_NAMES = (
    "humaneval", "mmlu", "hellaswag", "truthfulqa", "arc-", "arc_", "mbpp",
    "winogrande", "bigbench", "big-bench", "bbh", "agieval", "gpqa", "ifeval",
    "swe-bench", "swebench", "livecodebench", "-bench", "_bench", "benchmark",
)
# Name fragments signalling a dataset's post-training purpose.
_SFT_NAMES = (
    "hermes", "orca", "alpaca", "dolly", "vicuna", "wizardlm", "wizard-",
    "instruct", "ultrachat", "sharegpt", "oasst", "guanaco", "tulu", "flan",
    "self-instruct", "sft",
)
_PREFERENCE_NAMES = (
    "rlhf", "dpo", "preference", "helpsteer", "ultrafeedback", "hh-",
    "reward", "-feedback",
)
_AGENT_NAMES = ("agent", "trajector", "webarena", "toolbench")


def classify_dataset(
    tags: list[str], description: str = "", dataset_id: str = ""
) -> str:
    """Infer a dataset's primary category.

    Priority: benchmark > multimodal > post-training purpose (name/tags) >
    code/math > reasoning > generic task tags > multilingual/books >
    description > default web. Generic HF task tags (summarization, QA) are
    ranked BELOW purpose signals, since instruction/preference datasets are
    routinely mistagged with them.
    """
    tag_list = [t.lower() for t in tags]
    name = (dataset_id or "").lower()

    def in_name(fragments) -> bool:
        return any(f in name for f in fragments)

    # 1. Evaluation / benchmark
    if any("benchmark" in t or "eval" in t for t in tag_list) or in_name(_EVAL_NAMES):
        return "evaluation"

    # 2. Multimodal
    if any(
        "image-text" in t or "visual-question" in t or "vqa" in t
        or "modality:image" in t or "modality:video" in t or "modality:audio" in t
        for t in tag_list
    ):
        return "multimodal"

    # 3. Post-training purpose from NAME (beats generic task tags)
    if in_name(_AGENT_NAMES):
        return "posttraining-agent"
    if in_name(_PREFERENCE_NAMES):
        return "posttraining-preference"
    if in_name(_SFT_NAMES):
        return "posttraining-sft"

    # 4. Post-training purpose from TAGS
    if any("agent" in t or "trajectory" in t for t in tag_list):
        return "posttraining-agent"
    if any("tool" in t and "use" in t for t in tag_list):
        return "posttraining-tooluse"
    if any("safety" in t for t in tag_list):
        return "posttraining-safety"
    if any(
        "preference" in t or "dpo" in t or "rlhf" in t or "human-feedback" in t
        for t in tag_list
    ):
        return "posttraining-preference"
    if any("sft" in t or "instruction" in t or "alpaca" in t for t in tag_list):
        return "posttraining-sft"
    if any("reasoning" in t or "cot" in t or "chain-of-thought" in t for t in tag_list):
        return "posttraining-reasoning"

    # 5. Code / math
    if any("code" in t for t in tag_list) or in_name(("code", "stack", "starcoder")):
        return "pretraining-code"
    if any("math" in t for t in tag_list) or in_name(("math", "gsm")):
        return "pretraining-math"

    # 6. Generic task tags
    if any("summarization" in t for t in tag_list):
        return "task-summarization"
    if any("translation" in t for t in tag_list):
        return "task-translation"
    if any("question-answering" in t for t in tag_list):
        return "task-qa"
    if any("classification" in t for t in tag_list):
        return "task-classification"

    # 7. Multilingual (avoid "multilinguality:monolingual") / books
    if any("multilingual" in t and "monolingual" not in t for t in tag_list):
        return "pretraining-multilingual"
    if any("science" in t or "books" in t or "book" in t for t in tag_list):
        return "pretraining-books"

    # 8. Description fallback
    desc = description.lower()
    if "instruction" in desc or "sft" in desc:
        return "posttraining-sft"
    if "preference" in desc or "rlhf" in desc:
        return "posttraining-preference"
    if "code" in desc or "programming" in desc:
        return "pretraining-code"

    # 9. Default
    return "pretraining-web"


def get_category_metrics(category_key: str) -> list[str]:
    """Get the relevant quality metrics for a category."""
    cat = CATEGORIES.get(category_key)
    if cat:
        return cat.metrics
    return CATEGORIES["pretraining-web"].metrics