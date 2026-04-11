"""
Reddit sentiment scanner — FinSignals Python SDK example.

Pulls the top N posts from a subreddit and filters for high-quality,
non-sarcastic, directional signals for a given ticker.

Usage
-----
    pip install finsignals-api praw
    export FINSIGNALS_API_KEY=fs_your_key_here
    python examples/reddit_scanner.py --ticker NVDA
    python examples/reddit_scanner.py --ticker AAPL --subreddit stocks --limit 200
"""

import argparse
import os
import praw
import finsignals


def get_args():
    parser = argparse.ArgumentParser(description="Reddit sentiment scanner")
    parser.add_argument("--ticker", required=True, help="Ticker symbol (e.g. NVDA)")
    parser.add_argument("--subreddit", default="wallstreetbets")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-relevance", type=float, default=0.65)
    return parser.parse_args()


def main():
    cfg = get_args()

    reddit = praw.Reddit(
        client_id=os.environ.get("REDDIT_CLIENT_ID", "your_client_id"),
        client_secret=os.environ.get("REDDIT_CLIENT_SECRET", "your_client_secret"),
        user_agent="finsignals-scanner/1.0",
    )

    client = finsignals.Client()  # reads FINSIGNALS_API_KEY from env

    print(f"Fetching top {cfg.limit} posts from r/{cfg.subreddit}...")
    posts = list(reddit.subreddit(cfg.subreddit).hot(limit=cfg.limit))
    print(f"  Got {len(posts)} posts. Sending to FinSignals...")

    items = [
        {"ticker": cfg.ticker, "title": p.title, "body": p.selftext[:1500]}
        for p in posts
    ]

    results = client.classify_batch(items)
    print(f"  Done. Credits charged: {results.credits_charged:.1f}\n")

    signals = [
        (post, out)
        for post, out in zip(posts, results)
        if out.quality.label == "relevant"
        and out.directionality.label in ("bullish", "bearish")
        and out.relevance_score >= cfg.min_relevance
        and not out.sarcasm
    ]

    bullish = [(p, o) for p, o in signals if o.directionality.label == "bullish"]
    bearish = [(p, o) for p, o in signals if o.directionality.label == "bearish"]

    print(f"Results for ${cfg.ticker.upper()} — {len(signals)} signals from {len(posts)} posts")
    print(f"  {len(bullish)} bullish, {len(bearish)} bearish\n")

    for direction, group in [("BULLISH", bullish), ("BEARISH", bearish)]:
        if not group:
            continue
        print(f"{direction} SIGNALS:")
        for post, out in sorted(group, key=lambda x: x[1].relevance_score, reverse=True):
            print(f"  [{out.post_type.label}] rel={out.relevance_score:.2f} conf={out.author_confidence:.2f}")
            print(f"  {post.title[:90]}")
            print(f"  https://reddit.com{post.permalink}\n")


if __name__ == "__main__":
    main()