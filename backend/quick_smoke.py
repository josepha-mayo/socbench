import asyncio, sys, time

async def test_scorer(name, fn, samples):
    try:
        r = await fn(samples)
        print(f"  [OK] {name}: score={r.score:.3f}")
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")

async def main():
    print("= dedup")
    from socbench.scoring.dedup import dedup_scorer
    s = [{"text": "Sample text for dedup testing."}, {"text": "Another distinct sample."}]
    await test_scorer("dedup", dedup_scorer, s)

    print("= format")
    from socbench.scoring.format import format_scorer
    await test_scorer("format", format_scorer, s)

    print("= tokens")
    from socbench.scoring.tokens import token_scorer
    await test_scorer("tokens", token_scorer, s)

    print("= language")
    from socbench.scoring.language import language_scorer
    await test_scorer("language", language_scorer, s)

    print("= quality")
    from socbench.scoring.quality import quality_scorer
    await test_scorer("quality", quality_scorer, s)

    print("= code")
    from socbench.scoring.code import code_scorer
    cs = [{"text": "def add(a,b):\n    return a+b"}, {"text": "def sub(x,y):\n    return x-y"}]
    await test_scorer("code", code_scorer, cs)

    print("= all done")

asyncio.run(main())
sys.exit(0)