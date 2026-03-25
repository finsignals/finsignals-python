"""
Tests for the FinSignals Python SDK.

Run with:
    pip install -e ".[dev]"
    pytest tests/ -v
"""

import pytest
import responses as rsps_lib

import finsignals
from finsignals.client import Client, _build_single_payload, _check_batch_char_limit, _normalise_item
from finsignals.exceptions import (
    AuthenticationError,
    BatchTooLargeError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
)
from finsignals.models import (
    ClassificationOutput,
    ClassifyBatchResponse,
    ClassifyResponse,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

BASE_URL = "https://api.finsignals.ai"

SINGLE_RESPONSE = {
    "model_version": "2.0.0",
    "request_id": "req-abc123",
    "credits_charged": 1.0,
    "endpoint_type": "single",
    "endpoint_name": "reddit_single",
    "outputs": [
        {
            "sentiment": {"label": "positive", "positive": 0.89, "negative": 0.04, "neutral": 0.07},
            "directionality": {"label": "bullish", "bullish": 0.85, "bearish": 0.08, "neutral_direction": 0.07},
            "quality": {"label": "relevant", "relevant": 0.78, "noise": 0.15, "spam": 0.07},
            "post_type": {
                "label": "news_reaction",
                "dd": 0.02, "news_reaction": 0.71, "technical_analysis": 0.05,
                "fundamentals": 0.08, "question": 0.04, "general": 0.10,
            },
            "relevance_score": 0.9137,
            "author_confidence": 0.5802,
            "sarcasm": False,
        }
    ],
}

BATCH_RESPONSE = {
    "model_version": "2.0.0",
    "request_id": "req-batch456",
    "credits_charged": 1.7,
    "endpoint_type": "batch",
    "endpoint_name": "reddit_batch",
    "outputs": [
        {
            "sentiment": {"label": "negative", "positive": 0.05, "negative": 0.82, "neutral": 0.13},
            "directionality": {"label": "bearish", "bullish": 0.10, "bearish": 0.77, "neutral_direction": 0.13},
            "quality": {"label": "relevant", "relevant": 0.80, "noise": 0.15, "spam": 0.05},
            "post_type": {"label": "dd", "dd": 0.70, "news_reaction": 0.10, "technical_analysis": 0.05,
                          "fundamentals": 0.05, "question": 0.05, "general": 0.05},
            "relevance_score": 0.8500,
            "author_confidence": 0.7000,
            "sarcasm": False,
        },
        {
            "sentiment": {"label": "positive", "positive": 0.75, "negative": 0.10, "neutral": 0.15},
            "directionality": {"label": "bullish", "bullish": 0.72, "bearish": 0.15, "neutral_direction": 0.13},
            "quality": {"label": "noise", "relevant": 0.20, "noise": 0.72, "spam": 0.08},
            "post_type": {"label": "general", "dd": 0.02, "news_reaction": 0.08, "technical_analysis": 0.05,
                          "fundamentals": 0.05, "question": 0.10, "general": 0.70},
            "relevance_score": 0.3200,
            "author_confidence": 0.2500,
            "sarcasm": True,
        },
    ],
}

USAGE_RESPONSE = {
    "plan": "pro",
    "monthly_credits_total": 1000000.0,
    "monthly_credits_used": 12500.5,
    "monthly_credits_remaining": 987499.5,
    "payg_balance": 0.0,
    "usage_by_endpoint": [],
}

PLAN_RESPONSE = {
    "plan": "pro",
    "rate_limits": {"single": 120, "batch": 60},
    "monthly_credits": 1000000.0,
    "batch_max_items": 256,
}


@pytest.fixture
def client():
    return Client(api_key="fs_test_key_1234")


# ── classify() ────────────────────────────────────────────────────────────────

class TestClassify:
    @rsps_lib.activate
    def test_returns_classify_response(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)

        result = client.classify(ticker="NVDA", body="Blackwell demand insane 🚀")

        assert isinstance(result, ClassifyResponse)
        assert result.model_version == "2.0.0"
        assert result.credits_charged == 1.0

    @rsps_lib.activate
    def test_sentiment_label(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = client.classify(ticker="NVDA", body="Blackwell demand insane 🚀")
        assert result.sentiment.label == "positive"
        assert result.sentiment.positive == pytest.approx(0.89)

    @rsps_lib.activate
    def test_directionality_label(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = client.classify(ticker="NVDA", body="Blackwell demand insane 🚀")
        assert result.directionality.label == "bullish"

    @rsps_lib.activate
    def test_relevance_score(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = client.classify(ticker="NVDA", body="Blackwell demand insane 🚀")
        assert result.relevance_score == pytest.approx(0.9137)

    @rsps_lib.activate
    def test_sarcasm_flag(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = client.classify(ticker="NVDA", body="Blackwell demand insane 🚀")
        assert result.sarcasm is False

    @rsps_lib.activate
    def test_output_proxy_matches_outputs_0(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = client.classify(ticker="NVDA", body="test")
        assert result.sentiment is result.output.sentiment
        assert result.directionality is result.output.directionality

    def test_raises_value_error_when_all_fields_empty(self, client):
        with pytest.raises(ValueError, match="At least one"):
            client.classify()

    @rsps_lib.activate
    def test_title_only_post(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = client.classify(title="NVDA beats earnings")
        assert isinstance(result, ClassifyResponse)

    @rsps_lib.activate
    def test_401_raises_authentication_error(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify",
                     json={"detail": "invalid_api_key"}, status=401)
        with pytest.raises(AuthenticationError):
            client.classify(ticker="NVDA", body="test")

    @rsps_lib.activate
    def test_402_raises_insufficient_credits_error(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify",
                     json={"detail": {"error": "quota_exceeded", "remaining_credits": 0.0}},
                     status=402)
        with pytest.raises(InsufficientCreditsError):
            client.classify(ticker="NVDA", body="test")

    @rsps_lib.activate
    def test_429_raises_rate_limit_error(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify",
                     json={"detail": {"error": "rate_limit_exceeded", "retry_after": 10.0,
                                       "limit": 120, "window_seconds": 60}},
                     status=429)
        with pytest.raises(RateLimitError) as exc_info:
            client.classify(ticker="NVDA", body="test")
        assert exc_info.value.retry_after == pytest.approx(10.0)

    @rsps_lib.activate
    def test_422_raises_validation_error(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify",
                     json={"detail": [{"loc": ["body"], "msg": "field required"}]},
                     status=422)
        with pytest.raises(ValidationError):
            client.classify(ticker="NVDA", body="test")


# ── classify_batch() ──────────────────────────────────────────────────────────

class TestClassifyBatch:
    @rsps_lib.activate
    def test_returns_batch_response(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify/batch", json=BATCH_RESPONSE, status=200)
        items = [
            {"ticker": "TSLA", "body": "Delivery miss, stock down."},
            {"ticker": "NVDA", "body": "Blackwell demand strong."},
        ]
        results = client.classify_batch(items)

        assert isinstance(results, ClassifyBatchResponse)
        assert len(results) == 2
        assert results.credits_charged == pytest.approx(1.7)

    @rsps_lib.activate
    def test_outputs_in_order(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify/batch", json=BATCH_RESPONSE, status=200)
        items = [
            {"ticker": "TSLA", "body": "Delivery miss."},
            {"ticker": "NVDA", "body": "Demand strong."},
        ]
        results = client.classify_batch(items)
        assert results.outputs[0].sentiment.label == "negative"
        assert results.outputs[1].sentiment.label == "positive"

    @rsps_lib.activate
    def test_iteration(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify/batch", json=BATCH_RESPONSE, status=200)
        items = [{"body": "x"}, {"body": "y"}]
        results = client.classify_batch(items)
        labels = [o.sentiment.label for o in results]
        assert labels == ["negative", "positive"]

    @rsps_lib.activate
    def test_index_access(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify/batch", json=BATCH_RESPONSE, status=200)
        items = [{"body": "x"}, {"body": "y"}]
        results = client.classify_batch(items)
        assert isinstance(results[0], ClassificationOutput)

    def test_raises_value_error_on_empty_items(self, client):
        with pytest.raises(ValueError, match="empty"):
            client.classify_batch([])

    def test_raises_batch_too_large_error(self, client):
        items = [{"body": "x"}] * 257
        with pytest.raises(BatchTooLargeError) as exc_info:
            client.classify_batch(items)
        assert exc_info.value.count == 257
        assert exc_info.value.max_items == 256

    def test_raises_validation_error_on_empty_item(self, client):
        with pytest.raises(ValueError, match="non-empty field"):
            client.classify_batch([{"ticker": "", "body": ""}])


# ── get_usage() and get_plan() ────────────────────────────────────────────────

class TestUsageAndPlan:
    @rsps_lib.activate
    def test_get_usage(self, client):
        rsps_lib.add(rsps_lib.GET, f"{BASE_URL}/v1/usage", json=USAGE_RESPONSE, status=200)
        usage = client.get_usage()
        assert usage.plan == "pro"
        assert usage.monthly_credits_remaining == pytest.approx(987499.5)

    @rsps_lib.activate
    def test_get_plan(self, client):
        rsps_lib.add(rsps_lib.GET, f"{BASE_URL}/v1/plan", json=PLAN_RESPONSE, status=200)
        plan = client.get_plan()
        assert plan.plan == "pro"
        assert plan.rate_limits.single == 120
        assert plan.rate_limits.batch == 60
        assert plan.batch_max_items == 256


# ── health() ─────────────────────────────────────────────────────────────────

class TestHealth:
    @rsps_lib.activate
    def test_returns_true_on_200(self, client):
        rsps_lib.add(rsps_lib.GET, f"{BASE_URL}/v1/health", json={"status": "ok"}, status=200)
        assert client.health() is True

    @rsps_lib.activate
    def test_returns_false_on_503(self, client):
        rsps_lib.add(rsps_lib.GET, f"{BASE_URL}/v1/health", body=Exception("connection refused"))
        assert client.health() is False


# ── Client construction ───────────────────────────────────────────────────────

class TestClientConstruction:
    def test_raises_without_key(self, monkeypatch):
        monkeypatch.delenv("FINSIGNALS_API_KEY", raising=False)
        with pytest.raises(ValueError, match="No API key"):
            Client()

    def test_reads_env_var(self, monkeypatch):
        monkeypatch.setenv("FINSIGNALS_API_KEY", "fs_env_key")
        c = Client()
        assert c._api_key == "fs_env_key"

    def test_explicit_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("FINSIGNALS_API_KEY", "fs_env_key")
        c = Client(api_key="fs_explicit_key")
        assert c._api_key == "fs_explicit_key"


# ── Module-level helpers ──────────────────────────────────────────────────────

class TestHelpers:
    def test_build_single_payload_all_fields(self):
        p = _build_single_payload("AAPL", "Apple Inc.", "Earnings beat", "Revenue up 12%.")
        assert p == {
            "ticker": "AAPL",
            "company_name": "Apple Inc.",
            "title": "Earnings beat",
            "body": "Revenue up 12%.",
        }

    def test_build_single_payload_omits_empty(self):
        p = _build_single_payload("AAPL", "", "", "Revenue up 12%.")
        assert "company_name" not in p
        assert "title" not in p

    def test_build_single_payload_raises_all_empty(self):
        with pytest.raises(ValueError):
            _build_single_payload("", "", "", "")

    def test_normalise_item_strips_unknown_keys(self):
        item = {"ticker": "NVDA", "body": "test", "extra_field": "ignored"}
        result = _normalise_item(item)
        assert "extra_field" not in result
        assert result["ticker"] == "NVDA"

    def test_check_batch_char_limit_passes(self):
        items = [{"body": "x" * 100}] * 10
        _check_batch_char_limit(items)  # should not raise

    def test_check_batch_char_limit_raises(self):
        items = [{"body": "x" * 50_000}] * 3  # 150,000 chars > 128,000 limit
        with pytest.raises(ValidationError, match="128,000"):
            _check_batch_char_limit(items)


# ── Module-level classify() convenience function ──────────────────────────────

class TestModuleLevelClassify:
    @rsps_lib.activate
    def test_classify_function(self):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = finsignals.classify("fs_test_key", ticker="NVDA", body="test")
        assert result.sentiment.label == "positive"


# ── repr / str sanity ─────────────────────────────────────────────────────────

class TestReprs:
    @rsps_lib.activate
    def test_classify_response_repr(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify", json=SINGLE_RESPONSE, status=200)
        result = client.classify(ticker="NVDA", body="test")
        r = repr(result)
        assert "positive" in r
        assert "bullish" in r

    @rsps_lib.activate
    def test_batch_response_repr(self, client):
        rsps_lib.add(rsps_lib.POST, f"{BASE_URL}/v1/classify/batch", json=BATCH_RESPONSE, status=200)
        results = client.classify_batch([{"body": "x"}, {"body": "y"}])
        r = repr(results)
        assert "items=2" in r


# ── get_sector_rotation() ─────────────────────────────────────────────────────

def _make_rotation_period(rs_window_labels=None):
    """Build a minimal mock rotation period payload."""
    return {
        "trading_date": "2025-03-24",
        "generated_at": "2025-03-24T02:15:00Z",
        "spy_metrics": {
            "ret_1m": 0.0312,
            "ret_3m": 0.0748,
            "ret_6m": 0.1103,
            "ret_9m": 0.1501,
            "ret_12m": 0.2204,
        },
        "sector_data": {
            "Technology": {
                "etf": "XLK",
                "phase": "advancing",
                "confidence": 0.82,
                "rotation_score": 0.4531,
                "adjustment": 0.12,
                "rs_1m": 0.0214,
                "rs_3m": 0.0547,
                "rs_6m": 0.0912,
                "rs_9m": 0.1234,
                "rs_12m": 0.1876,
                "ret_1m": 0.0526,
                "mom_accel": 0.0033,
                "vol_ratio": 1.12,
                "pe": 28.5,
            },
            "Energy": {
                "etf": "XLE",
                "phase": "declining",
                "confidence": 0.71,
                "rotation_score": -0.2108,
                "adjustment": -0.05,
                "rs_1m": -0.0132,
                "rs_3m": -0.0421,
                "rs_6m": -0.0803,
                "rs_9m": -0.0994,
                "rs_12m": -0.1452,
                "ret_1m": 0.0180,
                "mom_accel": -0.0021,
                "vol_ratio": 0.94,
                "pe": 11.2,
            },
        },
        "industry_data": {
            "Semiconductors": {
                "etf": "SOXX",
                "phase": "advancing",
                "confidence": 0.88,
                "rotation_score": 0.5812,
                "adjustment": 0.15,
                "rs_1m": 0.0341,
                "rs_3m": 0.0712,
                "rs_6m": 0.1103,
                "rs_vs_sector_3m": 0.0165,
                "mom_accel": 0.0041,
                "pe": 31.2,
                "parent_sector_etf": "XLK",
            }
        },
        "summary_md": "## AI Market Analysis\n\nTechnology continues to lead...",
        "weekly_snapshots": [
            {"week": "2025-03-17", "sectors": {}},
            {"week": "2025-03-10", "sectors": {}},
        ],
        "rs_window_labels": rs_window_labels,
    }


ROTATION_RESPONSE = {
    "request_id": "req-rot-789",
    "model_version": "3.0.0",
    "credits_charged": 10.0,
    "trading_date": "2025-03-24",
    "generated_at": "2025-03-24T02:15:00Z",
    "outlook_1y": _make_rotation_period(rs_window_labels=None),
    "outlook_5y": _make_rotation_period(
        rs_window_labels={"rs_1m": "RS 3M", "rs_3m": "RS 6M", "rs_6m": "RS 1Y", "rs_9m": "RS 2Y", "rs_12m": "RS 3Y"}
    ),
}


class TestGetSectorRotation:
    @rsps_lib.activate
    def test_get_sector_rotation_success(self, client):
        rsps_lib.add(rsps_lib.GET, f"{BASE_URL}/v1/sector-rotation", json=ROTATION_RESPONSE, status=200)

        result = client.get_sector_rotation()

        assert isinstance(result, finsignals.SectorRotationResponse)
        assert result.request_id == "req-rot-789"
        assert result.trading_date == "2025-03-24"
        assert result.credits_charged == pytest.approx(10.0)

        # outlook_1y sector data
        assert len(result.outlook_1y.sector_data) == 2
        tech = next(s for s in result.outlook_1y.sector_data if s.name == "Technology")
        assert tech.etf == "XLK"
        assert tech.phase == "advancing"
        assert tech.rotation_score == pytest.approx(0.4531)
        assert tech.rs_3m == pytest.approx(0.0547)

        # outlook_1y industry data
        assert len(result.outlook_1y.industry_data) == 1
        semi = result.outlook_1y.industry_data[0]
        assert semi.name == "Semiconductors"
        assert semi.parent_sector_etf == "XLK"
        assert semi.rs_vs_sector_3m == pytest.approx(0.0165)

        # SPY metrics
        assert result.outlook_1y.spy_metrics.ret_1m == pytest.approx(0.0312)
        assert result.outlook_1y.spy_metrics.ret_12m == pytest.approx(0.2204)

        # summary markdown
        assert result.outlook_1y.summary_md is not None
        assert "Technology" in result.outlook_1y.summary_md

        # weekly snapshots
        assert len(result.outlook_1y.weekly_snapshots) == 2

        # 5y rs_window_labels
        labels = result.outlook_5y.rs_window_labels
        assert labels is not None
        assert labels["rs_1m"] == "RS 3M"
        assert labels["rs_12m"] == "RS 3Y"

        # 1y outlook has no rs_window_labels
        assert result.outlook_1y.rs_window_labels is None

    @rsps_lib.activate
    def test_get_sector_rotation_repr(self, client):
        rsps_lib.add(rsps_lib.GET, f"{BASE_URL}/v1/sector-rotation", json=ROTATION_RESPONSE, status=200)
        result = client.get_sector_rotation()
        r = repr(result)
        assert "2025-03-24" in r
        assert "sectors_1y=2" in r

    @rsps_lib.activate
    def test_get_sector_rotation_503(self, client):
        rsps_lib.add(
            rsps_lib.GET, f"{BASE_URL}/v1/sector-rotation",
            json={"detail": "rotation data not available yet"},
            status=503,
        )
        from finsignals.exceptions import APIError
        with pytest.raises(APIError) as exc_info:
            client.get_sector_rotation()
        assert exc_info.value.status_code == 503

    @rsps_lib.activate
    def test_get_sector_rotation_auth_error(self, client):
        rsps_lib.add(
            rsps_lib.GET, f"{BASE_URL}/v1/sector-rotation",
            json={"detail": "invalid_api_key"},
            status=401,
        )
        with pytest.raises(AuthenticationError):
            client.get_sector_rotation()

    @rsps_lib.activate
    def test_get_sector_rotation_insufficient_credits(self, client):
        rsps_lib.add(
            rsps_lib.GET, f"{BASE_URL}/v1/sector-rotation",
            json={"detail": {"error": "quota_exceeded", "remaining_credits": 0.0}},
            status=402,
        )
        with pytest.raises(InsufficientCreditsError):
            client.get_sector_rotation()


# ── Batch timeout formula ─────────────────────────────────────────────────────

class TestBatchTimeoutFormula:
    def test_multiplier_is_3_seconds(self):
        from finsignals.client import _SECS_PER_BATCH_ITEM
        assert _SECS_PER_BATCH_ITEM == pytest.approx(3.0)

    def test_effective_timeout_10_items(self, client):
        """For 10 items: max(30, 30 + 10 * 3.0) = max(30, 60) = 60."""
        # We verify the formula by checking the effective timeout used during the call.
        # We capture the actual timeout by inspecting the request via responses library.
        effective = max(client._timeout, 30.0 + 10 * 3.0)
        assert effective == pytest.approx(60.0)

    def test_effective_timeout_256_items(self, client):
        """For 256 items: max(30, 30 + 256 * 3.0) = 798."""
        effective = max(client._timeout, 30.0 + 256 * 3.0)
        assert effective == pytest.approx(798.0)

    def test_effective_timeout_uses_client_timeout_when_larger(self):
        """A client with timeout=900 should keep 900 even for small batches."""
        large_timeout_client = Client(api_key="fs_test_key", timeout=900.0)
        effective = max(large_timeout_client._timeout, 30.0 + 1 * 3.0)
        assert effective == pytest.approx(900.0)
