# FinSignals Python SDK

[![PyPI version](https://badge.fury.io/py/finsignals-api.svg)](https://pypi.org/project/finsignals-api/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)

**7 signals from one API call.** Finance-tuned NLP for Reddit posts and market text — built for algo traders, quant researchers, and anyone piping financial social data into a model or dashboard. Plus daily sector rotation analysis with 1-year and 5-year outlooks.

| Signal | Output |
|---|---|
| `sentiment` | `positive` / `negative` / `neutral` + probabilities |
| `directionality` | `bullish` / `bearish` / `neutral_direction` + probabilities |
| `quality` | `relevant` / `noise` / `spam` + probabilities |
| `post_type` | `dd` / `news_reaction` / `technical_analysis` / `question` / `general` |
| `relevance_score` | Float [0, 1] — how on-topic is this post for your ticker? |
| `author_confidence` | Float [0, 1] — how confident does the author sound? |
| `sarcasm` | Boolean flag (experimental) |

**Free tier: 1,000 credits/month, no credit card required.**
Get a key at [finsignals.ai](https://finsignals.ai).

## Who uses this

- **Algo traders** — filter Reddit for high-quality directional signals on a ticker before market open
- **Quant researchers** — batch-classify thousands of posts for sentiment datasets or backtests
- **Dashboard builders** — enrich a market dashboard with live sentiment and sector rotation data
- **Scraping pipelines** — drop spam and noise from a firehose before it hits your database
- **Finance NLP projects** — get 7-label annotations without labeling your own data

---

```python
import finsignals

client = finsignals.Client("fs_your_key_here")

# Reddit sentiment — classify a post across 7 dimensions
result = client.classify(ticker="NVDA", body="Blackwell demand is insane 🚀🚀 DD inside")
print(result.sentiment.label)       # "positive"
print(result.directionality.label)  # "bullish"

# Sector rotation — daily analysis with 1y and 5y outlooks
rotation = client.get_sector_rotation()
for sector in rotation.outlook_1y.sector_data:
    print(sector.name, sector.phase, sector.rotation_score)
```

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [Reddit Sentiment Classification](#reddit-sentiment-classification)
  - [The 7 classification heads](#the-7-classification-heads)
  - [Single classification](#single-classification)
  - [Batch classification](#batch-classification)
  - [Full pipeline example](#full-pipeline-example)
- [Sector Rotation Analysis](#sector-rotation-analysis)
  - [Quick example](#sector-rotation-quick-example)
  - [Response structure](#response-structure)
  - [Phase values](#phase-values)
  - [1-year vs 5-year outlooks](#1-year-vs-5-year-outlooks)
  - [Handling the not-yet-ready state](#handling-the-not-yet-ready-state)
  - [Sector Rotation API docs](#sector-rotation-api-docs)
- [Account: usage and plan](#account-usage-and-plan)
- [Error handling](#error-handling)
- [Configuration](#configuration)
- [Rate limits](#rate-limits)
- [Code examples in other languages](#code-examples-in-other-languages)
- [Contributing](#contributing)
- [License](#license)

---

## Install

```bash
pip install finsignals-api
```

Python 3.8+ required. No other non-standard dependencies.

---

## Quick start

**Get a free API key** at [finsignals.ai](https://finsignals.ai) — 1,000 free credits/month, no credit card required.

Set your key as an environment variable (recommended):

**Linux / macOS:**
```bash
export FINSIGNALS_API_KEY="fs_your_key_here"
```

**Windows (PowerShell):**
```powershell
$env:FINSIGNALS_API_KEY = "fs_your_key_here"
```

**Windows (Command Prompt):**
```cmd
set FINSIGNALS_API_KEY=fs_your_key_here
```

To set it permanently on Windows, use System Properties → Environment Variables.

Or pass it directly:

```python
client = finsignals.Client(api_key="fs_your_key_here")
```

---

## Reddit Sentiment Classification

Finance-tuned NLP that classifies Reddit posts and financial text across 7 dimensions in a single API call.

### The 7 classification heads

Every `/v1/classify` call returns all seven signals simultaneously:

| Head | Type | Output |
|---|---|---|
| `sentiment` | Classification | `positive` / `negative` / `neutral` + probabilities |
| `directionality` | Classification | `bullish` / `bearish` / `neutral_direction` + probabilities |
| `quality` | Classification | `relevant` / `noise` / `spam` + probabilities |
| `post_type` | Classification | `dd` / `news_reaction` / `technical_analysis` / `fundamentals` / `question` / `general` |
| `relevance_score` | Float [0, 1] | How on-topic is this post for the queried ticker? |
| `author_confidence` | Float [0, 1] | How confident does the author appear? |
| `sarcasm` | Boolean | Sarcasm flag (experimental) |

### Single classification

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

#### What you get back

```json
{
  "sentiment":         { "label": "positive", "positive": 0.91, "negative": 0.04, "neutral": 0.05 },
  "directionality":    { "label": "bullish",  "bullish": 0.87, "bearish": 0.06, "neutral_direction": 0.07 },
  "quality":           { "label": "relevant", "relevant": 0.83, "noise": 0.14, "spam": 0.03 },
  "post_type":         { "label": "news_reaction" },
  "relevance_score":   0.94,
  "author_confidence": 0.71,
  "sarcasm":           false,
  "credits_charged":   1.0
}
```

All four fields (`ticker`, `company_name`, `title`, `body`) are optional — at least one must be non-empty. Including `ticker` improves relevance scoring.

### Batch classification

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

#### Batch timeouts

`classify_batch()` automatically computes a longer timeout based on batch size:
`max(timeout, 30 + n × 3.0)` seconds, giving ~126 s for 32 items and ~798 s for
256 items. You can override it per call:

```python
# Override for a specific large batch
results = client.classify_batch(items, timeout=600)
```

### Full pipeline example

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

---

## Sector Rotation Analysis

`GET /v1/sector-rotation` delivers a daily market-structure snapshot that shows which sectors and industries are accelerating, peaking, or declining relative to SPY. It is completely independent from the Reddit Sentiment endpoint: different data sources, different calculation pipeline, different response schema.

Cost: **10 credits per call.** The report is pre-calculated once per trading day (after midnight ET) and served from cache, so subsequent calls on the same day are fast.

### Sector Rotation Quick Example

```python
import finsignals

client = finsignals.Client()  # reads FINSIGNALS_API_KEY from env

rotation = client.get_sector_rotation()

print(rotation.trading_date)           # "2025-03-24"
print(rotation.credits_charged)        # 10.0

# 1-year outlook — sector table
print("=== 1-year outlook ===")
for sector in rotation.outlook_1y.sector_data:
    print(
        f"  {sector.name:<25} phase={sector.phase:<12} "
        f"score={sector.rotation_score:+.3f}  rs_3m={sector.rs_3m:+.4f}"
    )

# 5-year outlook — top industries by rotation score
print("=== 5-year top industries ===")
top5 = sorted(
    rotation.outlook_5y.industry_data,
    key=lambda x: x.rotation_score or 0,
    reverse=True,
)[:5]
for ind in top5:
    print(f"  {ind.name:<30} parent={ind.parent_sector_etf}  score={ind.rotation_score:+.3f}")

# AI-generated markdown summary
print(rotation.outlook_1y.summary_md[:300])
```

#### Sample output (sector_data, 1-year outlook)

```
Sector                    Phase          Score    RS 3M
─────────────────────────────────────────────────────────
Technology                advancing      +0.42    +0.0312
Financials                advancing      +0.31    +0.0201
Industrials               peaking        +0.18    +0.0088
Consumer Discretionary    peaking        +0.11    +0.0044
Health Care               neutral         0.00    -0.0012
Real Estate               declining      -0.19    -0.0187
Utilities                 declining      -0.27    -0.0241
Energy                    bottoming      -0.38    -0.0390
```

Each sector includes phase classification, rotation score, RS vs SPY across multiple windows, momentum acceleration, volatility ratio, and P/E. The `summary_md` field contains an AI-generated narrative of the current rotation picture.

### Response structure

`client.get_sector_rotation()` returns a `SectorRotationResponse`:

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | Unique request identifier |
| `model_version` | `str` | API model version string |
| `credits_charged` | `float` | Credits deducted (10.0) |
| `trading_date` | `str` | ISO date the data applies to (`"2025-03-24"`) |
| `generated_at` | `str` | ISO datetime the report was computed |
| `outlook_1y` | `RotationPeriod` | 1-year lookback analysis |
| `outlook_5y` | `RotationPeriod` | 5-year lookback analysis |

Each `RotationPeriod` contains:

| Field | Type | Description |
|---|---|---|
| `trading_date` | `str` | Date for this period |
| `generated_at` | `str` | Timestamp this period was computed |
| `spy_metrics` | `SpyMetrics` | SPY benchmark returns (`ret_1m` … `ret_12m`) |
| `sector_data` | `List[SectorEntry]` | One entry per sector ETF |
| `industry_data` | `List[IndustryEntry]` | One entry per industry ETF |
| `summary_md` | `Optional[str]` | AI-generated markdown narrative |
| `weekly_snapshots` | `List[dict]` | Historical weekly RS data (raw dicts) |
| `rs_window_labels` | `Optional[dict]` | Display labels for RS columns (5y only) |

Key fields on `SectorEntry`:

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Sector label (`"Technology"`) |
| `etf` | `str` | Sector ETF ticker (`"XLK"`) |
| `phase` | `str` | Rotation phase (see table below) |
| `confidence` | `float` | Phase classification confidence [0, 1] |
| `rotation_score` | `float` | Composite RS-momentum score |
| `rs_1m` … `rs_12m` | `float` | Return vs SPY over each window |
| `mom_accel` | `float` | Rate of change in momentum |
| `vol_ratio` | `float` | Sector volatility vs SPY |
| `pe` | `float` | Sector P/E ratio |

Key fields on `IndustryEntry` (same as `SectorEntry` plus):

| Field | Type | Description |
|---|---|---|
| `rs_vs_sector_3m` | `float` | Industry RS minus parent sector RS (3M) |
| `parent_sector_etf` | `str` | ETF ticker of the parent sector |

### Phase values

| Phase | Meaning |
|---|---|
| `accumulation` | Early recovery; RS improving from a low base |
| `advancing` | Sustained outperformance vs SPY |
| `peaking` | RS still high but momentum decelerating |
| `distribution` | Outperformance fading; early rotation out |
| `declining` | Sustained underperformance |
| `bottoming` | RS low but momentum stabilising |

### 1-year vs 5-year outlooks

The two outlooks use **different RS calculation windows**. This is intentional — a 5-year view needs proportionally longer lookback periods to surface structural trends rather than short-term noise.

| RS field | 1-year window | 5-year window |
|---|---|---|
| `rs_1m` | 1 month (~21 bars) | 3 months (~64 bars) |
| `rs_3m` | 3 months (~63 bars) | 6 months (~127 bars) |
| `rs_6m` | 6 months (~126 bars) | 1 year (~252 bars) |
| `rs_9m` | 9 months (~189 bars) | 2 years (~504 bars) |
| `rs_12m` | 12 months (~252 bars) | 3 years (~756 bars) |

`outlook_5y.rs_window_labels` provides the display mapping:

```python
print(rotation.outlook_5y.rs_window_labels)
# {"rs_1m": "RS 3M", "rs_3m": "RS 6M", "rs_6m": "RS 1Y", "rs_9m": "RS 2Y", "rs_12m": "RS 3Y"}
```

### Handling the not-yet-ready state

The daily calculation runs after midnight ET on trading days. If you call the endpoint before it has run (e.g. very early morning ET), the API returns **503 Service Unavailable**. The SDK raises `APIError` with `status_code=503`.

```python
import time
import finsignals
from finsignals import APIError

client = finsignals.Client()

for attempt in range(5):
    try:
        rotation = client.get_sector_rotation()
        break
    except APIError as e:
        if e.status_code == 503:
            print(f"Report not yet ready — retrying in 5 minutes (attempt {attempt + 1})")
            time.sleep(300)
        else:
            raise
```

### Sector Rotation API docs

Full REST reference, field descriptions, and interactive examples:
[finsignals.ai/sector-rotation-api/](https://finsignals.ai/sector-rotation-api/)

The live report is published daily at [finsignals.ai/sector-rotation/](https://finsignals.ai/sector-rotation/).

---

## Account: usage and plan

```python
usage = client.get_usage()
print(usage.plan)                          # "pro"
print(usage.monthly_credits_remaining)     # 987499.5

plan = client.get_plan()
print(plan.rate_limits.batch)              # 60 (requests/minute)
```

---

## Error handling

```python
import finsignals
from finsignals import (
    AuthenticationError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
    BatchTooLargeError,
    APIError,
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
except APIError as e:
    if e.status_code == 503:
        print("Sector rotation report not yet ready for today — try again later")
    else:
        print(f"Unexpected error {e.status_code}: {e}")
```

`BatchTooLargeError` is raised client-side (before the HTTP request) if you pass more than 256 items to `classify_batch()`.

---

## Configuration

```python
client = finsignals.Client(
    api_key="fs_your_key_here",  # or set FINSIGNALS_API_KEY env var
    timeout=30,                  # timeout for single/health/usage calls (default: 30 s)
    max_retries=2,               # retries on 5xx errors (default: 2)
)
```

For batch calls, `classify_batch()` overrides `timeout` automatically based on batch
size unless you pass an explicit `timeout` to that call (see [Batch timeouts](#batch-timeouts) above).

---

## Rate limits

Limits are per API key on a **60-second sliding window**:

| Plan | `/v1/classify` (req/min) | `/v1/classify/batch` (req/min) |
|---|---|---|
| Free | 5 | 2 |
| Starter | 30 | 15 |
| Pro | 120 | 60 |
| Scale / Enterprise | 600 | 300 |

Your exact limits are always available from `client.get_plan()`:

```python
plan = client.get_plan()
print(plan.rate_limits.single)   # e.g. 120
print(plan.rate_limits.batch)    # e.g. 60
```

When exceeded the API returns **429 Too Many Requests**; the SDK raises `RateLimitError` with a `retry_after` attribute.

---

## Code examples in other languages

### cURL — sentiment

```bash
curl -sS -X POST "https://api.finsignals.ai/v1/classify" \
  -H "X-API-Key: $FINSIGNALS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"ticker":"NVDA","body":"Blackwell demand is insane 🚀 DD inside"}' | jq .
```

### cURL — sector rotation

```bash
curl -sS "https://api.finsignals.ai/v1/sector-rotation" \
  -H "X-API-Key: $FINSIGNALS_API_KEY" | jq '.outlook_1y.sector_data | to_entries[0]'
```

### Node.js (fetch)

```javascript
const res = await fetch("https://api.finsignals.ai/v1/classify", {
  method: "POST",
  headers: {
    "X-API-Key": process.env.FINSIGNALS_API_KEY,
    "Content-Type": "application/json",
  },
  body: JSON.stringify({ ticker: "TSLA", body: "Production ramp on track." }),
});
const data = await res.json();
if (!res.ok) throw new Error(JSON.stringify(data));
console.log(data.outputs[0].directionality.label);
```

### PHP

```php
<?php
$apiKey  = getenv('FINSIGNALS_API_KEY');
$payload = json_encode([
    'ticker' => 'MSFT',
    'body'   => 'Azure revenue up 21% YoY, cloud momentum continues.',
]);

$ch = curl_init('https://api.finsignals.ai/v1/classify');
curl_setopt_array($ch, [
    CURLOPT_RETURNTRANSFER => true,
    CURLOPT_POST           => true,
    CURLOPT_POSTFIELDS     => $payload,
    CURLOPT_HTTPHEADER     => [
        'X-API-Key: ' . $apiKey,
        'Content-Type: application/json',
    ],
    CURLOPT_TIMEOUT        => 30,
]);

$body   = curl_exec($ch);
$status = curl_getinfo($ch, CURLINFO_HTTP_CODE);
curl_close($ch);

if ($status !== 200) {
    throw new RuntimeException("API error $status: $body");
}

$data = json_decode($body, true);
$out  = $data['outputs'][0];
echo $out['sentiment']['label'] . ' / ' . $out['directionality']['label'] . PHP_EOL;
```

### Go

```go
package main

import (
    "bytes"
    "encoding/json"
    "fmt"
    "net/http"
    "os"
)

func main() {
    payload, _ := json.Marshal(map[string]string{
        "ticker": "AMZN",
        "body":   "AWS margin expansion drives record operating income.",
    })

    req, _ := http.NewRequest("POST", "https://api.finsignals.ai/v1/classify", bytes.NewBuffer(payload))
    req.Header.Set("X-API-Key", os.Getenv("FINSIGNALS_API_KEY"))
    req.Header.Set("Content-Type", "application/json")

    resp, err := http.DefaultClient.Do(req)
    if err != nil {
        panic(err)
    }
    defer resp.Body.Close()

    var result map[string]interface{}
    json.NewDecoder(resp.Body).Decode(&result)

    outputs := result["outputs"].([]interface{})
    first   := outputs[0].(map[string]interface{})
    sentiment := first["sentiment"].(map[string]interface{})
    fmt.Println(sentiment["label"])
}
```

### Ruby

```ruby
require 'net/http'
require 'json'
require 'uri'

uri     = URI('https://api.finsignals.ai/v1/classify')
payload = { ticker: 'GOOGL', body: 'Search ad revenue rebounds, AI overviews expanding.' }

http          = Net::HTTP.new(uri.host, uri.port)
http.use_ssl  = true
http.open_timeout = 10
http.read_timeout = 30

request = Net::HTTP::Post.new(uri.path)
request['X-API-Key']    = ENV['FINSIGNALS_API_KEY']
request['Content-Type'] = 'application/json'
request.body            = payload.to_json

response = http.request(request)
raise "API error #{response.code}: #{response.body}" unless response.code == '200'

data = JSON.parse(response.body)
out  = data['outputs'][0]
puts "#{out['sentiment']['label']} | score: #{out['relevance_score']}"
```

---

## Contributing

Issues and pull requests welcome at [github.com/finsignals/finsignals-python](https://github.com/finsignals/finsignals-python).

To run the test suite locally:

```bash
git clone https://github.com/finsignals/finsignals-python
cd finsignals-python
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
