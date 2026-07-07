"""Quick smoke test for core modules."""
import asyncio
import sys
import traceback


async def test_dedup():
    from socbench.scoring.dedup import dedup_scorer

    samples = [
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Machine learning models require large datasets."},
        {"text": "unique data that should have no duplicates at all"},
    ]
    r = await dedup_scorer(samples)
    assert r.score > 0.9, f"dedup score too low: {r.score}"
    print(f"  dedup: score={r.score:.3f}, dedup_rate={r.details['exact_dedup_rate']:.3f}")


async def test_format():
    from socbench.scoring.format import format_scorer

    samples = [
        {"text": "This is a properly formatted sentence. It has punctuation and structure."},
        {"text": "Another good sample with enough words and proper structure to pass checks."},
    ]
    r = await format_scorer(samples)
    assert r.score > 0.3, f"format score too low: {r.score}"
    print(f"  format: score={r.score:.3f}")


async def test_tokens():
    from socbench.scoring.tokens import token_scorer

    samples = [
        {"text": "A short sentence with about ten words in it total."},
        {"text": "Another normal length sentence for testing token analysis properly."},
    ]
    r = await token_scorer(samples)
    assert "mean_tokens" in r.details, f"missing mean_tokens: {r.details}"
    print(f"  tokens: score={r.score:.3f}, mean_tokens={r.details['mean_tokens']:.1f}")


async def test_language():
    from socbench.scoring.language import language_scorer

    samples = [
        {"text": "This is an English text with proper structure and standard vocabulary."},
        {"text": "Another English sample for language detection testing purposes."},
    ]
    r = await language_scorer(samples, expected_language="en")
    print(f"  language: score={r.score:.3f}, purity={r.details['purity']:.3f}")


async def test_pii():
    from socbench.scoring.pii import pii_scorer

    samples = [
        {"text": "Normal text without any PII information whatsoever."},
    ]
    r = await pii_scorer(samples)
    print(f"  pii: score={r.score:.3f}, pii_rate={r.details['pii_rate']:.3f}")


async def test_quality():
    from socbench.scoring.quality import quality_scorer

    samples = [
        {"text": "Normal quality text with proper structure."},
        {"text": "Another good quality text sample for testing."},
    ]
    r = await quality_scorer(samples)
    print(f"  quality: score={r.score:.3f}")


async def test_code():
    from socbench.scoring.code import code_scorer

    samples = [
        {"text": "def add(a, b):\n    return a + b"},
        {"text": "def multiply(x, y):\n    return x * y"},
    ]
    r = await code_scorer(samples)
    print(f"  code: score={r.score:.3f}, parse_rate={r.details['parse_rate']:.3f}")


async def test_categories():
    from socbench.categories import classify_dataset

    assert classify_dataset(["code", "python"], "") == "pretraining-code"
    assert classify_dataset(["instruction", "alpaca"], "") == "posttraining-sft"
    assert classify_dataset(["general"], "") == "pretraining-web"
    assert classify_dataset(["dpo", "preference"], "") == "posttraining-preference"
    print("  categories: all assertions passed")


async def test_score_import():
    from socbench.scoring import run_all_scorers

    samples = [
        {"text": "A test sample with enough words to pass minimum thresholds for all scorers."},
        {"text": "Another sample that is distinct enough not to trigger dedup issues."},
    ]
    composite, results = await run_all_scorers(samples, is_code=False)
    assert len(results) >= 6, f"only {len(results)} scorers ran"
    print(f"  run_all_scorers: composite={composite:.3f}, {len(results)} scorers")


async def run_all():
    tests = [
        ("dedup", test_dedup),
        ("format", test_format),
        ("tokens", test_tokens),
        ("language", test_language),
        ("pii", test_pii),
        ("quality", test_quality),
        ("code", test_code),
        ("categories", test_categories),
        ("run_all_scorers", test_score_import),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            await fn()
            passed += 1
            print(f"  [PASS] {name}")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {name}: {e}")

    print(f"\n{passed}/{len(tests)} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all())
    sys.exit(0 if success else 1)