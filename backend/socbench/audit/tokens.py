"""Token-length filter for the audit pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TokenFilterResult:
    """Result of token-length filtering."""

    ok: bool
    token_count: int


def _count_tokens(text: str, char_prefilter: int) -> int:
    """Count tokens for ``text``."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Rough fallback: ~4 characters per token on average.
        return len(text) // 4


def make_token_filter(
    min_tokens: int = 32,
    max_tokens: int = 8192,
    char_prefilter: int = 60_000,
):
    """Return a token filter closure."""

    def token_filter(text: str) -> TokenFilterResult:
        if len(text) > char_prefilter:
            text = text[:char_prefilter]
        count = _count_tokens(text, char_prefilter)
        return TokenFilterResult(ok=min_tokens <= count <= max_tokens, token_count=count)

    return token_filter
