# FinSignals Python SDK

[![PyPI version](https://badge.fury.io/py/finsignals.svg)](https://pypi.org/project/finsignals/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

The official Python client for the [FinSignals API](https://finsignals.ai) — a finance-tuned NLP API that classifies Reddit posts and financial text across 7 dimensions in a single call.

```python
import finsignals

client = finsignals.Client("fs_your_key_here")

result = client.classify(ticker="NVDA", body="Blackwell demand is insane 🚀🚀 DD inside")

print(result.sentiment.label)       # "positive"
print(result.directionality.label)  # "bullish"
print(result.quality.label)         # "relevant"
print(result.relevance_score)       # 0.9137
print(result.sarcasm)               # False
print(result.credits_charged)       # 1.0
```

## Install

```bash
pip install finsignals
```

Python 3.8+ required. No other non-standard dependencies.

## Quick start

**Get a free API key** at [finsignals.ai](https://finsignals.ai) — 1,000 free credits/month, no credit card required.

Set your key as an environment variable (recommended):

```bash
export FINSIGNALS_API_KEY="fs_your_key_here"
```

Or pass it directly:

```python
client = finsignals.Client(api_key="fs_your_key_here")
```

## The 7 classification heads

Every API call returns all seven signals simultaneously:

| Head | Type | Output |
|---|---|---|
| `sentiment` | Classification | `positive` / `negative` / `neutral` + probabilities |
| `directionality` | Classification | `bullish` / `bearish` / `neutral_direction` + probabilities |
| `quality` | Classification | `relevant` / `noise` / `spam` + probabilities |
| `post_type` | Classification | `dd` / `news_reaction` / `technical_analysis` / `fundamentals` / `question` / `general` |
| `relevance_score` | Float [0, 1] | How on-topic is this post for the queried ticker? |
| `author_confidence` | Float [0, 1] | How confident does the author appear? |
| `sarcasm` | Boolean | Sarcasm flag (experimental) |

## Single classification

```python
result = client.classify(
    ticker="AAPL",
    title="Apple crushes Q3 earnings",
    body="Revenue up 12%, guidance raised. Stock up 5% AH.",
)

print(result.sentiment.label)           # "positive"
print(result.sentiment.positive)        # 0.92
print(result.directionality.label)      # "bullish"
print(result.post_type.label)           # "news_reaction"
print(result.relevance_score)           # 0.9412
print(result.author_confidence)         # 0.71
print(result.sarcasm)                   # False
print(result.credits_charged)           # 1.0
```

All four fields (`ticker`, `company_name`, `title`, `body`) are optional — at least one must be non-empty. Including `ticker` improves relevance scoring.

## Batch classification

Send up to 256 posts in a single request. Credit cost: `1.0 + 0.7 × (n - 1)` per call.

```python
results = client.classify_batch([
    {"ticker": "TSLA", "body": "Delivery miss, stock down premarket."},
    {"ticker": "NVDA", "title": "Blackwell demand", "body": "Hyperscaler capex still strong 🚀"},
    {"ticker": "AAPL", "body": "Apple beats Q3, guidance raised."},
])

print(results.credits_charged)     # 2.4 (1.0 + 0.7 + 0.7)
print(len(results))                # 3

for output in results:
    print(output.sentiment.label, output.directionality.label, output.relevance_score)
```

Output objects are in the same order as your input items. `ClassifyBatchResponse` supports `len()`, iteration, and index access (`results[0]`).

## Full pipeline example (Reddit → FinSignals)

```python
import praw
import finsignals

reddit = praw.Reddit(
    client_id="YOUR_CLIENT_ID",
    client_secret="YOUR_CLIENT_SECRET",
    user_agent="my-scanner/1.0",
)
client = finsignals.Client()   # reads FINSIGNALS_API_KEY from env

posts = list(reddit.subreddit("wallstreetbets").hot(limit=100))

items = [
    {"ticker": "NVDA", "title": post.title, "body": post.selftext[:1500]}
    for post in posts
]

results = client.classify_batch(items)

signals = [
    (post, output)
    for post, output in zip(posts, results)
    if output.quality.label == "relevant"
    and output.directionality.label == "bullish"
    and output.relevance_score > 0.65
    and not output.sarcasm
]

print(f"{len(signals)} signals from {len(posts)} posts")
for post, out in signals:
    print(f"  [{out.post_type.label}] {post.title[:80]}")
```

Full tutorial: [How to build a Reddit sentiment scanner in Python](https://finsignals.ai/blog/)

## Usage and plan

```python
usage = client.get_usage()
print(usage.plan)                          # "pro"
print(usage.monthly_credits_remaining)     # 987499.5

plan = client.get_plan()
print(plan.rate_limits.batch)              # 60 (requests/minute)
```

## Error handling

```python
import finsignals
from finsignals import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
    BatchTooLargeError,
)

try:
    result = client.classify(ticker="NVDA", body="test")
except AuthenticationError:
    print("Invalid API key — check finsignals.ai/api-keys")
except InsufficientCreditsError as e:
    print(f"Out of credits: {e}")
except RateLimitError as e:
    print(f"Rate limited — retry after {e.retry_after:.1f}s")
except ValidationError as e:
    print(f"Bad request: {e.errors}")
except finsignals.APIError as e:
    print(f"Unexpected error {e.status_code}: {e}")
```

`BatchTooLargeError` is raised client-side (before the HTTP request) if you pass more than 256 items to `classify_batch()`.

## Configuration

```python
client = finsignals.Client(
    api_key="fs_your_key_here",  # or set FINSIGNALS_API_KEY env var
    timeout=30,                  # request timeout in seconds (default: 30)
    max_retries=2,               # retries on 5xx errors (default: 2)
)
```

## Contributing

Issues and pull requests welcome at [github.com/finsignals/finsignals-python](https://github.com/finsignals/finsignals-python).

To run the test suite locally:

```bash
git clone https://github.com/finsignals/finsignals-python
cd finsignals-python
pip install -e ".[dev]"
pytest tests/ -v
```

## License

MIT
