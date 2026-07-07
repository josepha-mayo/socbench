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


def classify_dataset(tags: list[str], description: str = "") -> str:
    """Infer a dataset's primary category from HF tags.

    Returns category key (e.g., 'pretraining-code').
    """
    tag_list = [t.lower() for t in tags]

    # Check for specific tags first
    if any("code" in t for t in tag_list):
        return "pretraining-code"
    if any("math" in t for t in tag_list):
        return "pretraining-math"
    if any("agent" in t or "trajectory" in t for t in tag_list):
        return "posttraining-agent"
    if any("tool" in t and "use" in t for t in tag_list):
        return "posttraining-tooluse"
    if any("safety" in t for t in tag_list):
        return "posttraining-safety"
    if any("preference" in t or "dpo" in t or "rlhf" in t for t in tag_list):
        return "posttraining-preference"
    if any("sft" in t or "instruction" in t or "alpaca" in t for t in tag_list):
        return "posttraining-sft"
    if any("reasoning" in t or "cot" in t or "chain-of-thought" in t for t in tag_list):
        return "posttraining-reasoning"
    if any("multilingual" in t for t in tag_list):
        return "pretraining-multilingual"
    if any("science" in t or "books" in t or "book" in t for t in tag_list):
        return "pretraining-books"

    # Check description
    desc = description.lower()
    if "instruction" in desc or "sft" in desc:
        return "posttraining-sft"
    if "code" in desc or "programming" in desc:
        return "pretraining-code"

    # Default: web pretraining
    return "pretraining-web"


def get_category_metrics(category_key: str) -> list[str]:
    """Get the relevant quality metrics for a category."""
    cat = CATEGORIES.get(category_key)
    if cat:
        return cat.metrics
    return CATEGORIES["pretraining-web"].metrics