"""
Live integration test for the FinSignals Python SDK.

Sends a real request to the API and prints all 7 classification signals,
account usage, and plan info.

Usage:
    python live_test.py
"""

import finsignals
from finsignals import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
)

API_KEY = "fs_live_nuDJfoBJjhjclE-Ig1a9rse1f0yiq85zPm-4DXlrZGo"

TICKER = "MU"
TITLE = "$MU, Micron is barely starting to uncoil. Here's why."
BODY = """Right at the end of January Micron was reaching new ATH.

The next week had it dip down to the $360 range due to a correction in tech, silver pumping like a meme coin, and some paper handed traders. This week there was even more FUD from a Korean news source claiming their HBM4 was postponed in production until next quarter. Welp, they were wrong as hell.

Today Micron participated in the Wolfe Research Auto, Auto Tech and Semiconductor Conference in New York.

-First they dispelled the FUD issued by the Korean news as totally inaccurate.

-They announced their HBM4 production and shipment had begun a quarter earlier than expected

-Their LPDR ram was 60% more power efficient than DRAM

-They announced new fabrication lab acquisitions, one specifically for DRAM & LPDR

-Most importantly they reported their quarterly guidance from last quarter actually undershot greatly how well they performed this quarter.

-They also noted that their supply has began to exhaust for 2027

-Price Analyst are rising their price targets to the $450-600 range after we've already tapped $450 before.

We trade at a forward PE of only 12 at the moment. Seeing that there is no internal company FUD to fear, how does this sway your opinion on micron?

What are your thoughts on micron? What would your price target be?"""


def section(title: str) -> None:
    print(f"\n{'-' * 60}")
    print(f"  {title}")
    print('-' * 60)


def test_classify() -> None:
    section("classify() — single post")

    client = finsignals.Client(api_key=API_KEY)

    try:
        result = client.classify(ticker=TICKER, title=TITLE, body=BODY)
    except AuthenticationError:
        print("FAIL  Invalid API key.")
        return
    except InsufficientCreditsError as e:
        print(f"FAIL  Insufficient credits: {e}")
        return
    except RateLimitError as e:
        print(f"FAIL  Rate limited — retry after {e.retry_after:.1f}s")
        return
    except ValidationError as e:
        print(f"FAIL  Validation error: {e.errors}")
        return

    print(f"  request_id        : {result.request_id}")
    print(f"  model_version     : {result.model_version}")
    print(f"  credits_charged   : {result.credits_charged}")
    print()
    print(f"  sentiment         : {result.sentiment.label}")
    print(f"    positive        : {result.sentiment.positive:.4f}")
    print(f"    negative        : {result.sentiment.negative:.4f}")
    print(f"    neutral         : {result.sentiment.neutral:.4f}")
    print()
    print(f"  directionality    : {result.directionality.label}")
    print(f"    bullish         : {result.directionality.bullish:.4f}")
    print(f"    bearish         : {result.directionality.bearish:.4f}")
    print(f"    neutral_dir     : {result.directionality.neutral_direction:.4f}")
    print()
    print(f"  quality           : {result.quality.label}")
    print(f"    relevant        : {result.quality.relevant:.4f}")
    print(f"    noise           : {result.quality.noise:.4f}")
    print(f"    spam            : {result.quality.spam:.4f}")
    print()
    print(f"  post_type         : {result.post_type.label}")
    print(f"    dd              : {result.post_type.dd:.4f}")
    print(f"    news_reaction   : {result.post_type.news_reaction:.4f}")
    print(f"    technical_anal  : {result.post_type.technical_analysis:.4f}")
    print(f"    fundamentals    : {result.post_type.fundamentals:.4f}")
    print(f"    question        : {result.post_type.question:.4f}")
    print(f"    general         : {result.post_type.general:.4f}")
    print()
    print(f"  relevance_score   : {result.relevance_score:.4f}")
    print(f"  author_confidence : {result.author_confidence:.4f}")
    print(f"  sarcasm           : {result.sarcasm}")
    print()
    print("  PASS")


def test_classify_batch() -> None:
    section("classify_batch() — 2-item batch")

    client = finsignals.Client(api_key=API_KEY)

    items = [
        {"ticker": TICKER, "title": TITLE, "body": BODY},
        {"ticker": "NVDA", "body": "Blackwell demand is insane 🚀🚀 DD inside"},
    ]

    try:
        results = client.classify_batch(items)
    except (AuthenticationError, InsufficientCreditsError, RateLimitError, ValidationError) as e:
        print(f"FAIL  {type(e).__name__}: {e}")
        return

    print(f"  credits_charged   : {results.credits_charged}")
    print(f"  items returned    : {len(results)}")
    print()
    for i, output in enumerate(results):
        label_info = (
            f"sentiment={output.sentiment.label:<10} "
            f"directionality={output.directionality.label:<16} "
            f"quality={output.quality.label:<10} "
            f"relevance={output.relevance_score:.4f}"
        )
        print(f"  [{i}] {label_info}")
    print()
    print("  PASS")


def test_usage_and_plan() -> None:
    section("get_usage() and get_plan()")

    client = finsignals.Client(api_key=API_KEY)

    try:
        usage = client.get_usage()
        print(f"  plan                       : {usage.plan}")
        print(f"  monthly_credits_total      : {usage.monthly_credits_total}")
        print(f"  monthly_credits_used       : {usage.monthly_credits_used}")
        print(f"  monthly_credits_remaining  : {usage.monthly_credits_remaining}")
        print(f"  payg_balance               : {usage.payg_balance}")
        print()

        plan = client.get_plan()
        print(f"  rate_limit single (req/min): {plan.rate_limits.single}")
        print(f"  rate_limit batch  (req/min): {plan.rate_limits.batch}")
        print(f"  batch_max_items            : {plan.batch_max_items}")
        print()
        print("  PASS")
    except (AuthenticationError, finsignals.APIError) as e:
        print(f"FAIL  {type(e).__name__}: {e}")


def test_health() -> None:
    section("health()")
    client = finsignals.Client(api_key=API_KEY)
    ok = client.health()
    status = "PASS  API is healthy" if ok else "FAIL  API health check returned False"
    print(f"  {status}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  FinSignals SDK — Live Integration Test")
    print("=" * 60)

    test_health()
    test_classify()
    test_classify_batch()
    test_usage_and_plan()

    print("\n" + "=" * 60)
    print("  Done")
    print("=" * 60 + "\n")
