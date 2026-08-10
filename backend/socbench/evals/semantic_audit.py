"""Reference-free semantic quality audits for conversational-agent benchmarks.

The local engine is deterministic and requires no model or API key. An optional
OpenAI-compatible judge can refine task-level decisions when explicitly enabled
by the server operator.
"""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PAPER_REFERENCE = "Koren, Bar-Haim & Goldsteen, arXiv:2608.06329"
LOCAL_ENGINE = "local_semantic_v1"
API_ENGINE = "api_enhanced_semantic_v1"
MAX_TASKS = 200
MAX_POLICY_ITEMS = 200


class BenchmarkTaskInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    id: str | None = Field(default=None, max_length=128)
    description: str = Field(min_length=1, max_length=20_000)
    expected_behavior: str = Field(min_length=1, max_length=20_000)
    initial_state: dict[str, Any] | list[Any] | str | None = None


class BenchmarkAuditInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    name: str = Field(default="Untitled benchmark", min_length=1, max_length=256)
    policy: str = Field(default="", max_length=100_000)
    policy_items: list[str] = Field(default_factory=list)
    tasks: list[BenchmarkTaskInput] = Field(min_length=1, max_length=MAX_TASKS)
    coverage_threshold: int = Field(default=3, ge=1, le=100)
    use_api_judge: bool = True

    @field_validator("policy_items")
    @classmethod
    def validate_policy_items(cls, items: list[str]) -> list[str]:
        cleaned = [item.strip() for item in items if item.strip()]
        if len(cleaned) > MAX_POLICY_ITEMS:
            raise ValueError(f"policy_items cannot exceed {MAX_POLICY_ITEMS} entries")
        if any(len(item) > 10_000 for item in cleaned):
            raise ValueError("each policy item must be at most 10,000 characters")
        return cleaned

    @model_validator(mode="after")
    def require_policy(self) -> "BenchmarkAuditInput":
        if not self.policy.strip() and not self.policy_items:
            raise ValueError("policy or policy_items is required")
        return self


_STOPWORDS = {
    "a", "agent", "an", "and", "are", "as", "at", "be", "been", "being", "by", "for", "from",
    "has", "have", "if", "in", "into", "is", "it", "its", "of", "on", "or", "that",
    "the", "their", "then", "this", "to", "was", "were", "when", "with", "would",
}
_NEGATORS = {
    "cannot", "cant", "decline", "deny", "disallow", "forbid", "never", "no", "not",
    "prevent", "prohibit", "refuse", "reject", "unable", "without",
}
_REQUEST_WORDS = {"ask", "asks", "attempt", "request", "requested", "requests", "want", "wants", "try", "tries"}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")
_POLICY_SPLIT_RE = re.compile(r"(?<=[.!?;])\s+|\n+")

_CONCEPT_GROUPS: dict[str, set[str]] = {
    "address": {"address", "destination", "location"},
    "allow": {"allow", "authorize", "enable", "permit"},
    "authenticate": {"authenticate", "authentication", "identity", "login", "verify", "verification"},
    "cancel": {"cancel", "cancellation", "terminate", "termination"},
    "change": {"alter", "change", "edit", "modify", "modification", "update"},
    "customer": {"client", "customer", "user"},
    "delete": {"delete", "erase", "remove", "removal"},
    "deliver": {"deliver", "delivery", "ship", "shipment", "shipped"},
    "escalate": {"escalate", "escalation", "handoff", "supervisor"},
    "order": {"booking", "order", "purchase", "reservation"},
    "pay": {"card", "pay", "payment", "purchase", "transaction"},
    "refund": {"reimburse", "reimbursement", "refund", "repay"},
    "require": {"mandatory", "must", "need", "require", "required", "shall"},
    "restrict": {"block", "deny", "forbid", "prohibit", "restrict"},
    "return": {"return", "send", "transfer"},
}
_CANONICAL_TOKEN = {
    token: canonical
    for canonical, tokens in _CONCEPT_GROUPS.items()
    for token in tokens
}
_ACTION_CONCEPTS = set(_CONCEPT_GROUPS) - {"customer", "require"}
_MEANING_EXCLUDED = {"allow", "customer", "require", "restrict"}
_VIOLATION_ACTIONS = {"allow", "authenticate", "cancel", "change", "delete", "deliver", "escalate", "refund", "return"}


def _stem(token: str) -> str:
    for suffix in ("ations", "ation", "ments", "ment", "ingly", "ing", "edly", "ed", "ies", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def _raw_tokens(text: str) -> list[str]:
    return [token.replace("'", "") for token in _TOKEN_RE.findall(text.lower())]


def _canonicalize(token: str) -> str:
    stemmed = _stem(token)
    return _CANONICAL_TOKEN.get(token, _CANONICAL_TOKEN.get(stemmed, _CANONICAL_TOKEN.get(f"{stemmed}e", stemmed)))


def _semantic_tokens(text: str) -> list[str]:
    base = [_canonicalize(token) for token in _raw_tokens(text) if token not in _STOPWORDS]
    bigrams = [f"{left}_{right}" for left, right in zip(base, base[1:]) if left != right]
    return base + bigrams


def _semantic_vector(text: str) -> Counter[str]:
    counts = Counter(_semantic_tokens(text))
    return Counter({token: 1.0 + math.log(count) for token, count in counts.items()})


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    if not left or not right:
        return 0.0
    common = left.keys() & right.keys()
    dot = sum(left[token] * right[token] for token in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def _semantic_similarity(left: str, right: str) -> float:
    left_tokens = set(_semantic_tokens(left))
    right_tokens = set(_semantic_tokens(right))
    if not left_tokens or not right_tokens:
        return 0.0
    cosine = _cosine(_semantic_vector(left), _semantic_vector(right))
    overlap = len(left_tokens & right_tokens) / max(1, min(len(left_tokens), len(right_tokens)))
    return min(1.0, (0.70 * cosine) + (0.30 * overlap))


def _alignment_score(source: str, target: str) -> float:
    similarity = _semantic_similarity(source, target)
    source_concepts = _meaning_concepts(source)
    target_concepts = _meaning_concepts(target)
    if not source_concepts or not target_concepts:
        return round(similarity * 100, 1)
    concept_overlap = len(source_concepts & target_concepts) / min(len(source_concepts), len(target_concepts))
    return round(((0.40 * similarity) + (0.60 * concept_overlap)) * 100, 1)


def _meaning_concepts(text: str) -> set[str]:
    concepts = {_canonicalize(token) for token in _raw_tokens(text)}
    return {concept for concept in concepts if concept in _CONCEPT_GROUPS and concept not in _MEANING_EXCLUDED}


def _action_polarities(text: str) -> dict[str, int]:
    raw = _raw_tokens(text)
    polarities: dict[str, int] = {}
    for index, token in enumerate(raw):
        canonical = _canonicalize(token)
        if canonical not in _ACTION_CONCEPTS:
            continue
        context = raw[max(0, index - 2):index]
        polarities[canonical] = -1 if any(word in _NEGATORS for word in context) else 1
    return polarities


def _is_restrictive(text: str) -> bool:
    raw = set(_raw_tokens(text))
    return bool(raw & _NEGATORS) or "must not" in text.lower() or "may not" in text.lower()


def _policy_expected_score(policy_item: str, expected: str) -> float:
    score = _alignment_score(policy_item, expected)
    policy_polarity = _action_polarities(policy_item)
    expected_polarity = _action_polarities(expected)
    shared_actions = policy_polarity.keys() & expected_polarity.keys()
    if not shared_actions:
        return score
    agreement = sum(policy_polarity[action] == expected_polarity[action] for action in shared_actions) / len(shared_actions)
    if agreement == 1.0:
        return round(max(score, 82.0), 1)
    if agreement == 0.0:
        return round(min(score, 18.0), 1)
    return round((0.6 * score) + (0.4 * agreement * 100), 1)


def _split_policy(policy: str) -> list[str]:
    items: list[str] = []
    for part in _POLICY_SPLIT_RE.split(policy):
        cleaned = re.sub(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)", "", part).strip()
        if len(cleaned) >= 4:
            items.append(cleaned)
    return items[:MAX_POLICY_ITEMS]


def _task_text(task: BenchmarkTaskInput) -> str:
    if task.initial_state is None:
        return task.description
    if isinstance(task.initial_state, str):
        state = task.initial_state
    else:
        state = json.dumps(task.initial_state, sort_keys=True, ensure_ascii=True)
    return f"{task.description}\nInitial state: {state}"


def _violated_policy_items(task_text: str, policy_items: list[str]) -> list[int]:
    task_polarity = _action_polarities(task_text)
    task_raw = set(_raw_tokens(task_text))
    violations: list[int] = []
    for index, item in enumerate(policy_items):
        if not _is_restrictive(item):
            continue
        item_polarity = _action_polarities(item)
        shared_actions = task_polarity.keys() & item_polarity.keys() & _VIOLATION_ACTIONS
        direct_conflict = any(item_polarity[action] < 0 < task_polarity[action] for action in shared_actions)
        semantic_request = any(_stem(word) in _REQUEST_WORDS for word in task_raw)
        if direct_conflict and semantic_request:
            violations.append(index)
    return violations


def _local_task_audit(task: BenchmarkTaskInput, policy_items: list[str], index: int) -> dict[str, Any]:
    task_text = _task_text(task)
    description_alignment = _alignment_score(task.description, task.expected_behavior)
    task_concepts = _meaning_concepts(task_text)
    ranked_policy = sorted(
        (
            (item_index, _semantic_similarity(task_text, item))
            for item_index, item in enumerate(policy_items)
            if task_concepts & _meaning_concepts(item)
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    top_relevance = ranked_policy[0][1] if ranked_policy else 0.0
    relevance_floor = max(0.10, top_relevance * 0.55)
    relevant = [pair for pair in ranked_policy[:3] if pair[1] >= relevance_floor]
    if relevant:
        total_weight = sum(weight for _, weight in relevant)
        policy_alignment = sum(
            _policy_expected_score(policy_items[item_index], task.expected_behavior) * weight
            for item_index, weight in relevant
        ) / total_weight
    else:
        policy_alignment = 50.0

    violated = _violated_policy_items(task_text, policy_items)
    diagnostics: list[str] = []
    if description_alignment < 45:
        diagnostics.append("Expected behavior has weak semantic alignment with the task description.")
    if policy_alignment < 45:
        diagnostics.append("Expected behavior may conflict with or omit applicable policy guidance.")
    if not relevant:
        diagnostics.append("No clearly applicable policy item was detected for this task.")
    if violated:
        diagnostics.append(f"Task probes {len(violated)} restrictive policy item(s).")

    return {
        "id": task.id or f"task-{index + 1}",
        "description_expected_alignment": round(description_alignment, 1),
        "policy_expected_alignment": round(policy_alignment, 1),
        "violated_policy_items": violated,
        "diagnostics": diagnostics,
        "engine": LOCAL_ENGINE,
    }


@dataclass(frozen=True)
class _JudgeConfig:
    url: str
    model: str
    api_key: str
    timeout_seconds: float
    max_tasks: int


def _judge_config() -> tuple[_JudgeConfig | None, str | None]:
    enabled = os.environ.get("SOCBENCH_SEMANTIC_REMOTE_ENABLED", "").strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None, None
    url = os.environ.get("SOCBENCH_SEMANTIC_API_URL", "").strip()
    model = os.environ.get("SOCBENCH_SEMANTIC_MODEL", "").strip()
    api_key = os.environ.get("SOCBENCH_SEMANTIC_API_KEY", "").strip()
    if not all((url, model, api_key)):
        return None, "Remote semantic judging is enabled but URL, model, or API key is missing; local scoring was used."
    parsed = urlparse(url)
    if parsed.scheme != "https" and parsed.hostname not in {"localhost", "127.0.0.1"}:
        return None, "Remote semantic judge URL must use HTTPS unless it targets localhost; local scoring was used."
    try:
        timeout = max(5.0, min(float(os.environ.get("SOCBENCH_SEMANTIC_TIMEOUT_SECONDS", "45")), 120.0))
        max_tasks = max(1, min(int(os.environ.get("SOCBENCH_SEMANTIC_MAX_JUDGE_TASKS", "50")), MAX_TASKS))
    except ValueError:
        return None, "Remote semantic judge limits are invalid; local scoring was used."
    return _JudgeConfig(url=url, model=model, api_key=api_key, timeout_seconds=timeout, max_tasks=max_tasks), None


def _extract_json_object(content: str) -> dict[str, Any]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("judge response did not contain a JSON object")
    value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("judge response JSON must be an object")
    return value


async def _request_api_judgment(
    client: httpx.AsyncClient,
    config: _JudgeConfig,
    task: BenchmarkTaskInput,
    policy_items: list[str],
) -> dict[str, Any]:
    numbered_policy = "\n".join(f"{index}: {item}" for index, item in enumerate(policy_items))
    prompt = (
        "Evaluate this conversational-agent benchmark task. Return JSON only with keys "
        "description_expected_alignment (0-100), policy_expected_alignment (0-100), "
        "violated_policy_items (zero-based integer array), and diagnostics (short string array).\n\n"
        f"POLICY ITEMS:\n{numbered_policy}\n\n"
        f"TASK DESCRIPTION:\n{task.description}\n\n"
        f"INITIAL STATE:\n{json.dumps(task.initial_state, ensure_ascii=True, default=str)}\n\n"
        f"EXPECTED BEHAVIOR:\n{task.expected_behavior}"
    )
    response = await client.post(
        config.url,
        headers={"Authorization": f"Bearer {config.api_key}", "Content-Type": "application/json"},
        json={
            "model": config.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "You are a strict benchmark-quality auditor."},
                {"role": "user", "content": prompt},
            ],
        },
    )
    response.raise_for_status()
    payload = response.json()
    content = payload["choices"][0]["message"]["content"]
    if isinstance(content, list):
        content = "".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    judged = _extract_json_object(str(content))
    description_score = max(0.0, min(float(judged["description_expected_alignment"]), 100.0))
    policy_score = max(0.0, min(float(judged["policy_expected_alignment"]), 100.0))
    violations = sorted({
        int(item) for item in judged.get("violated_policy_items", [])
        if isinstance(item, int) and 0 <= item < len(policy_items)
    })
    diagnostics = [str(item)[:300] for item in judged.get("diagnostics", []) if str(item).strip()][:8]
    return {
        "description_expected_alignment": description_score,
        "policy_expected_alignment": policy_score,
        "violated_policy_items": violations,
        "diagnostics": diagnostics,
    }


async def _enhance_with_api(
    request: BenchmarkAuditInput,
    policy_items: list[str],
    local_results: list[dict[str, Any]],
    config: _JudgeConfig,
) -> tuple[int, list[str]]:
    warnings: list[str] = []
    task_limit = min(len(request.tasks), config.max_tasks)
    if task_limit < len(request.tasks):
        warnings.append(f"Remote judge was limited to the first {task_limit} tasks; every task still received local scoring.")
    semaphore = asyncio.Semaphore(4)

    async with httpx.AsyncClient(timeout=config.timeout_seconds) as client:
        async def judge(index: int) -> tuple[int, dict[str, Any] | None, str | None]:
            try:
                async with semaphore:
                    return index, await _request_api_judgment(client, config, request.tasks[index], policy_items), None
            except Exception as exc:
                return index, None, f"Task {index + 1} remote judgment failed ({type(exc).__name__}); local score retained."

        judged = await asyncio.gather(*(judge(index) for index in range(task_limit)))

    enhanced = 0
    for index, remote, warning in judged:
        if warning:
            warnings.append(warning)
            continue
        assert remote is not None
        local = local_results[index]
        local["description_expected_alignment"] = round(
            (local["description_expected_alignment"] + remote["description_expected_alignment"]) / 2,
            1,
        )
        local["policy_expected_alignment"] = round(
            (local["policy_expected_alignment"] + remote["policy_expected_alignment"]) / 2,
            1,
        )
        local["violated_policy_items"] = sorted(set(local["violated_policy_items"]) | set(remote["violated_policy_items"]))
        local["diagnostics"] = list(dict.fromkeys(local["diagnostics"] + remote["diagnostics"]))[:8]
        local["engine"] = API_ENGINE
        enhanced += 1
    return enhanced, warnings


async def audit_benchmark(request: BenchmarkAuditInput) -> dict[str, Any]:
    """Audit benchmark consistency, policy complexity, and policy coverage."""
    policy_items = request.policy_items or _split_policy(request.policy)
    if not policy_items:
        raise ValueError("policy did not yield any auditable policy items")

    task_results = [
        _local_task_audit(task, policy_items, index)
        for index, task in enumerate(request.tasks)
    ]
    warnings: list[str] = []
    enhanced_count = 0
    judge_config, config_warning = _judge_config()
    if config_warning:
        warnings.append(config_warning)
    if request.use_api_judge and judge_config:
        enhanced_count, judge_warnings = await _enhance_with_api(request, policy_items, task_results, judge_config)
        warnings.extend(judge_warnings)

    description_alignment = sum(item["description_expected_alignment"] for item in task_results) / len(task_results)
    policy_alignment = sum(item["policy_expected_alignment"] for item in task_results) / len(task_results)
    coverage_counts = Counter(
        policy_index
        for task in task_results
        for policy_index in task["violated_policy_items"]
    )
    coverage_rows = [
        {
            "index": index,
            "policy_item": item,
            "task_count": coverage_counts[index],
            "covered": coverage_counts[index] >= request.coverage_threshold,
        }
        for index, item in enumerate(policy_items)
    ]
    violation_count = sum(len(task["violated_policy_items"]) for task in task_results)
    violations_per_task = violation_count / len(task_results)
    coverage = sum(row["covered"] for row in coverage_rows) / len(coverage_rows)
    complexity_score = min(violations_per_task / 2.0, 1.0) * 100
    semantic_quality = (
        (0.35 * description_alignment)
        + (0.35 * policy_alignment)
        + (0.20 * coverage * 100)
        + (0.10 * complexity_score)
    )

    diagnostics: list[str] = []
    weak_description = sum(task["description_expected_alignment"] < 45 for task in task_results)
    weak_policy = sum(task["policy_expected_alignment"] < 45 for task in task_results)
    uncovered = sum(not row["covered"] for row in coverage_rows)
    if weak_description:
        diagnostics.append(f"{weak_description} task(s) have weak description-to-expected-behavior alignment.")
    if weak_policy:
        diagnostics.append(f"{weak_policy} task(s) have weak policy-to-expected-behavior alignment.")
    if uncovered:
        diagnostics.append(f"{uncovered} of {len(policy_items)} policy item(s) are below the coverage threshold.")
    if not diagnostics:
        diagnostics.append("No aggregate semantic consistency or policy coverage weakness was detected.")

    return {
        "benchmark_name": request.name,
        "engine": API_ENGINE if enhanced_count else LOCAL_ENGINE,
        "api_judge_configured": judge_config is not None,
        "api_judge_enhanced_tasks": enhanced_count,
        "task_count": len(task_results),
        "policy_item_count": len(policy_items),
        "coverage_threshold": request.coverage_threshold,
        "scores": {
            "description_expected_alignment": round(description_alignment, 1),
            "policy_expected_alignment": round(policy_alignment, 1),
            "policy_violations_per_task": round(violations_per_task, 3),
            "policy_violation_coverage": round(coverage * 100, 1),
            "semantic_quality": round(semantic_quality, 1),
        },
        "tasks": task_results,
        "policy_coverage": coverage_rows,
        "diagnostics": diagnostics,
        "warnings": list(dict.fromkeys(warnings)),
        "methodology": {
            "reference": PAPER_REFERENCE,
            "local_engine": LOCAL_ENGINE,
            "dimensions": [
                "description_expected_alignment",
                "policy_expected_alignment",
                "policy_violations_per_task",
                "policy_violation_coverage",
            ],
        },
    }


def load_benchmark_input(path: str | Path, *, use_api_judge: bool = True) -> BenchmarkAuditInput:
    """Load and validate a canonical benchmark audit JSON file."""
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("benchmark audit input must be a JSON object")
    payload["use_api_judge"] = use_api_judge
    return BenchmarkAuditInput.model_validate(payload)
