"""Provenance tracking — dataset→paper→model mapping.

Track which datasets feed which models and papers.
The "Used by" feature — IMDb for datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ProvenanceEntry:
    dataset_id: str
    model_name: Optional[str] = None
    paper_title: Optional[str] = None
    paper_url: Optional[str] = None
    source: str = "manual"  # manual, hf_papers, codesota
    verified: bool = False


# Known dataset→model mappings (research-backed)
KNOWN_PROVENANCE: list[ProvenanceEntry] = [
    ProvenanceEntry(
        dataset_id="bigcode/starcoderdata",
        model_name="StarCoder (15.5B)",
        paper_title="StarCoder: may the source be with you!",
        paper_url="https://arxiv.org/abs/2305.06161",
        verified=True,
    ),
    ProvenanceEntry(
        dataset_id="bigcode/the-stack-v2",
        model_name="StarCoder2 (15B)",
        paper_title="StarCoder 2 and The Stack v2: The Next Generation",
        paper_url="https://arxiv.org/abs/2402.19197",
        verified=True,
    ),
    ProvenanceEntry(
        dataset_id="HuggingFaceFW/fineweb",
        model_name="FineWeb models",
        paper_title="FineWeb: Decanting the web for the finest text data at scale",
        paper_url="https://arxiv.org/abs/2406.17557",
        verified=True,
    ),
    ProvenanceEntry(
        dataset_id="allenai/dolma",
        model_name="OLMo (7B, 1B)",
        paper_title="Dolma: an Open Corpus of Three Trillion Tokens",
        paper_url="https://arxiv.org/abs/2402.00159",
        verified=True,
    ),
    ProvenanceEntry(
        dataset_id="princeton-nlp/SWE-bench",
        model_name="SWE-agent, Devin, etc.",
        paper_title="SWE-bench: Can Language Models Resolve Real-World GitHub Issues?",
        paper_url="https://arxiv.org/abs/2310.06770",
        verified=True,
    ),
    ProvenanceEntry(
        dataset_id="openai/openai_humaneval",
        model_name="Codex, GPT-4, Claude, DeepSeek, and many others",
        paper_title="Evaluating Large Language Models Trained on Code",
        paper_url="https://arxiv.org/abs/2107.03374",
        verified=True,
    ),
    ProvenanceEntry(
        dataset_id="teknium/OpenHermes-2.5",
        model_name="Nous Hermes 2, OpenHermes models",
        paper_title="OpenHermes 2.5: An Open Dataset for Synthetic Data",
        source="hf_papers",
        verified=False,
    ),
    ProvenanceEntry(
        dataset_id="ise-uiuc/Magicoder-OSS-Instruct-110K",
        model_name="Magicoder models",
        paper_title="Magicoder: Empowering Code Generation with OSS-Instruct",
        paper_url="https://arxiv.org/abs/2312.02120",
        verified=True,
    ),
    ProvenanceEntry(
        dataset_id="databricks/databricks-dolly-15k",
        model_name="Dolly 2.0",
        paper_title="Free Dolly: Introducing the World's First Truly Open Instruction-Tuned LLM",
        source="manual",
        verified=False,
    ),
    ProvenanceEntry(
        dataset_id="Open-Orca/OpenOrca",
        model_name="Orca, Orca 2, Mistral-Orca",
        paper_title="Orca: Progressive Learning from Complex Explanation Traces",
        paper_url="https://arxiv.org/abs/2306.02707",
        verified=True,
    ),
]


def get_provenance(dataset_id: str) -> list[ProvenanceEntry]:
    """Get known provenance entries for a dataset."""
    return [e for e in KNOWN_PROVENANCE if e.dataset_id == dataset_id]


def get_all_models() -> dict[str, list[str]]:
    """Get a mapping of model → datasets used."""
    models: dict[str, list[str]] = {}
    for entry in KNOWN_PROVENANCE:
        if entry.model_name:
            models.setdefault(entry.model_name, []).append(entry.dataset_id)
    return models


async def discover_paper_linked_datasets() -> list[dict]:
    """Discover datasets linked to papers on HuggingFace.

    Scrapes HF Papers to find paper→dataset links.
    """
    import httpx

    results: list[dict] = []
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                "https://huggingface.co/api/papers",
                params={"limit": 100},
            )
            if resp.status_code == 200:
                papers = resp.json()
                for paper in papers:
                    datasets = paper.get("datasets", [])
                    if datasets:
                        results.append({
                            "paper_id": paper.get("id"),
                            "paper_title": paper.get("title"),
                            "paper_url": paper.get("upvotes"),
                            "datasets": datasets,
                        })
        except Exception:
            pass

    return results