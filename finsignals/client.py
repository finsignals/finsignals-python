"""
FinSignals Python SDK
~~~~~~~~~~~~~~~~~~~~~

A thin, typed wrapper around the FinSignals REST API.

Basic usage::

    import finsignals

    client = finsignals.Client("fs_your_key_here")

    # Single post
    result = client.classify(ticker="NVDA", body="Blackwell demand is insane 🚀")
    print(result.sentiment.label)       # "positive"
    print(result.directionality.label)  # "bullish"
    print(result.relevance_score)       # 0.9137
    print(result.credits_charged)       # 1.0

    # Batch (1.0 + 0.7 × (n-1) credits per request)
    results = client.classify_batch([
        {"ticker": "TSLA", "body": "Delivery miss, stock down premarket."},
        {"ticker": "NVDA", "title": "Blackwell demand", "body": "Hyperscaler capex strong."},
    ])
    for output in results.outputs:
        print(output.sentiment.label, output.directionality.label)
"""

import os
import time
import logging
from typing import Dict, List, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .exceptions import (
    APIError,
    AuthenticationError,
    BatchTooLargeError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
)
from .models import (
    ClassifyBatchResponse,
    ClassifyResponse,
    PlanResponse,
    UsageResponse,
    parse_classify_batch_response,
    parse_classify_response,
    parse_plan_response,
    parse_usage_response,
)

logger = logging.getLogger("finsignals")

_DEFAULT_BASE_URL = "https://api.finsignals.ai"
_DEFAULT_TIMEOUT = 30       # seconds — for single/health/usage calls
_SECS_PER_BATCH_ITEM = 1.5  # conservative budget per item for auto batch timeout
_MAX_BATCH_ITEMS = 256
_MAX_BATCH_CHARS = 128_000

# Per-field length limits applied client-side before the HTTP request.
_MAX_LEN_TICKER       = 20
_MAX_LEN_COMPANY_NAME = 200
_MAX_LEN_TITLE        = 1_000
_MAX_LEN_BODY         = 40_000


def _build_session(timeout: float, max_retries: int) -> requests.Session:
    """Return a requests Session with retry logic on connection errors."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["GET", "POST"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    return session


class Client:
    """
    FinSignals API client.

    Parameters
    ----------
    api_key : str, optional
        Your FinSignals API key. If not provided, the client reads
        the ``FINSIGNALS_API_KEY`` environment variable.
    base_url : str, optional
        Override the API base URL. Defaults to ``https://api.finsignals.ai``.
    timeout : float, optional
        Default timeout in seconds for single classify, usage, plan, and
        health calls. Defaults to 30. For batch calls, ``classify_batch()``
        overrides this automatically based on batch size unless you pass an
        explicit ``timeout`` to that call.
    max_retries : int, optional
        Number of retries on transient server errors (5xx). Defaults to 2.

    Raises
    ------
    ValueError
        If no API key is provided and ``FINSIGNALS_API_KEY`` is not set.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = _DEFAULT_BASE_URL,
        timeout: float = _DEFAULT_TIMEOUT,
        max_retries: int = 2,
    ):
        resolved_key = api_key or os.environ.get("FINSIGNALS_API_KEY")
        if not resolved_key:
            raise ValueError(
                "No API key provided. Pass api_key= or set the "
                "FINSIGNALS_API_KEY environment variable."
            )

        # Only https:// and http://localhost are permitted to prevent API keys
        # from being transmitted over plaintext on non-loopback connections.
        if not (base_url.startswith("https://") or base_url.startswith("http://localhost") or base_url.startswith("http://127.0.0.1")):
            raise ValueError(
                f"base_url must start with 'https://' (or 'http://localhost' for local dev). "
                f"Got: {base_url!r}"
            )

        self._api_key = resolved_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._session = _build_session(timeout, max_retries)
        self._session.headers.update({
            "X-API-Key": self._api_key,
            "Content-Type": "application/json",
            "User-Agent": f"finsignals-python/{_get_version()}",
        })

    # ── Public API ────────────────────────────────────────────────────────────

    def classify(
        self,
        *,
        ticker: str = "",
        company_name: str = "",
        title: str = "",
        body: str = "",
    ) -> ClassifyResponse:
        """
        Classify a single financial post.

        At least one of ``ticker``, ``company_name``, ``title``, or ``body``
        must be non-empty.

        Parameters
        ----------
        ticker : str
            Ticker symbol (e.g. ``"NVDA"``).
        company_name : str
            Company name, if helpful for context.
        title : str
            Post or article headline.
        body : str
            Main body text of the post.

        Returns
        -------
        ClassifyResponse
            Top-level response. Access ``result.sentiment.label``,
            ``result.directionality.label``, ``result.relevance_score``, etc.
            directly — or via ``result.output`` for the full
            ``ClassificationOutput`` object.

        Raises
        ------
        ValueError
            If all four fields are empty.
        AuthenticationError
            If the API key is invalid.
        InsufficientCreditsError
            If the account has no remaining credits.
        RateLimitError
            If the per-key rate limit is exceeded.
        """
        payload = _build_single_payload(ticker, company_name, title, body)
        data = self._post("/v1/classify", payload)
        return parse_classify_response(data)

    def classify_batch(
        self,
        items: List[Dict[str, str]],
        *,
        timeout: Optional[float] = None,
    ) -> ClassifyBatchResponse:
        """
        Classify multiple posts in a single request.

        Credit cost: ``1.0 + 0.7 × (len(items) - 1)`` per call.

        Parameters
        ----------
        items : list of dict
            Each dict may contain any of: ``ticker``, ``company_name``,
            ``title``, ``body``. At least one key must be non-empty per item.
            Maximum 256 items per call.
        timeout : float, optional
            Per-request timeout in seconds. When omitted the client
            automatically computes ``max(self.timeout, 30 + len(items) * 1.5)``,
            which gives ~78 s for 32 items and ~414 s for 256 items. Pass an
            explicit value to override this formula.

        Returns
        -------
        ClassifyBatchResponse
            Response with an ``outputs`` list in the same order as ``items``.
            Supports ``len()``, iteration, and index access directly.

        Raises
        ------
        BatchTooLargeError
            If ``len(items)`` exceeds 256 (checked client-side before the
            request is sent).
        ValidationError
            If any item fails server-side validation.
        """
        if not items:
            raise ValueError("items must not be empty.")

        if len(items) > _MAX_BATCH_ITEMS:
            raise BatchTooLargeError(count=len(items), max_items=_MAX_BATCH_ITEMS)

        _check_batch_char_limit(items)

        effective_timeout = (
            timeout if timeout is not None
            else max(self._timeout, 30.0 + len(items) * _SECS_PER_BATCH_ITEM)
        )

        payload = {"items": [_normalise_item(i) for i in items]}
        data = self._post("/v1/classify/batch", payload, timeout=effective_timeout)
        return parse_classify_batch_response(data)

    def get_usage(self) -> UsageResponse:
        """
        Return credit balances and usage breakdown for the current key.

        Returns
        -------
        UsageResponse
        """
        data = self._get("/v1/usage")
        return parse_usage_response(data)

    def get_plan(self) -> PlanResponse:
        """
        Return plan details and rate limits for the current key.

        Returns
        -------
        PlanResponse
        """
        data = self._get("/v1/plan")
        return parse_plan_response(data)

    def health(self) -> bool:
        """
        Ping the health endpoint. Returns True if the API is reachable.

        Does not require authentication.
        """
        try:
            resp = self._session.get(
                f"{self._base_url}/v1/health",
                timeout=self._timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict, timeout: Optional[float] = None) -> dict:
        url = f"{self._base_url}{path}"
        logger.debug("POST %s  payload_keys=%s", url, list(payload.keys()))

        t = timeout if timeout is not None else self._timeout
        try:
            resp = self._session.post(url, json=payload, timeout=t)
        except requests.Timeout:
            raise APIError(0, f"Request to {url} timed out after {t}s.")
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection error: {exc}")

        return _handle_response(resp)

    def _get(self, path: str) -> dict:
        url = f"{self._base_url}{path}"
        logger.debug("GET %s", url)

        try:
            resp = self._session.get(url, timeout=self._timeout)
        except requests.Timeout:
            raise APIError(0, f"Request to {url} timed out after {self._timeout}s.")
        except requests.ConnectionError as exc:
            raise APIError(0, f"Connection error: {exc}")

        return _handle_response(resp)


# ── Module-level convenience functions ───────────────────────────────────────

def classify(
    api_key: str,
    *,
    ticker: str = "",
    company_name: str = "",
    title: str = "",
    body: str = "",
) -> ClassifyResponse:
    """
    One-shot classify without constructing a ``Client`` object.

    Prefer using ``Client`` when making multiple calls — it reuses
    the underlying HTTP session and is more efficient.
    """
    return Client(api_key=api_key).classify(
        ticker=ticker,
        company_name=company_name,
        title=title,
        body=body,
    )


# ── Private helpers ───────────────────────────────────────────────────────────

def _build_single_payload(
    ticker: str,
    company_name: str,
    title: str,
    body: str,
) -> dict:
    payload = {}
    if ticker:
        payload["ticker"] = str(ticker)[:_MAX_LEN_TICKER]
    if company_name:
        payload["company_name"] = str(company_name)[:_MAX_LEN_COMPANY_NAME]
    if title:
        payload["title"] = str(title)[:_MAX_LEN_TITLE]
    if body:
        payload["body"] = str(body)[:_MAX_LEN_BODY]
    if not payload:
        raise ValueError(
            "At least one of ticker, company_name, title, or body must be non-empty."
        )
    return payload


def _normalise_item(item: Dict[str, str]) -> dict:
    """Strip unknown keys, enforce types and length limits, ensure at least one field is present."""
    allowed = {"ticker", "company_name", "title", "body"}
    _limits = {
        "ticker": _MAX_LEN_TICKER,
        "company_name": _MAX_LEN_COMPANY_NAME,
        "title": _MAX_LEN_TITLE,
        "body": _MAX_LEN_BODY,
    }
    normalised = {
        k: str(v)[:_limits[k]]
        for k, v in item.items()
        if k in allowed and v
    }
    if not normalised:
        safe_repr = repr({k: v for k, v in item.items() if k in allowed})
        if len(safe_repr) > 200:
            safe_repr = safe_repr[:200] + "…"
        raise ValueError(
            f"Each batch item must have at least one non-empty field "
            f"(ticker, company_name, title, body). Got: {safe_repr}"
        )
    return normalised


def _check_batch_char_limit(items: List[Dict[str, str]]) -> None:
    total = sum(
        len(v)
        for item in items
        for k, v in item.items()
        if k in {"ticker", "company_name", "title", "body"} and isinstance(v, str)
    )
    if total > _MAX_BATCH_CHARS:
        raise ValidationError(
            f"Batch payload is {total:,} characters, "
            f"exceeding the {_MAX_BATCH_CHARS:,}-character limit. "
            "Truncate bodies or split into smaller batches."
        )


def _handle_response(resp: requests.Response) -> dict:
    """Parse the response and raise typed exceptions for error status codes."""
    if resp.status_code == 200:
        return resp.json()

    # Try to extract a detail object from the response body
    detail = {}
    try:
        body = resp.json()
        if isinstance(body, dict):
            detail = body.get("detail", body)
    except (ValueError, requests.exceptions.JSONDecodeError):
        logger.debug("Response body was not valid JSON (status=%s)", resp.status_code)

    if resp.status_code == 401:
        raise AuthenticationError()

    if resp.status_code == 402:
        raise InsufficientCreditsError.from_detail(detail if isinstance(detail, dict) else {})

    if resp.status_code == 422:
        raise ValidationError(detail)

    if resp.status_code == 429:
        d = detail if isinstance(detail, dict) else {}
        raise RateLimitError.from_detail(d)

    # Generic fallback — cap message length to avoid leaking large server payloads
    message = detail if isinstance(detail, str) else str(detail)
    if len(message) > 500:
        message = message[:500] + "…"
    raise APIError(resp.status_code, message)


def _get_version() -> str:
    try:
        from importlib.metadata import version
        return version("finsignals")
    except Exception:
        return "0.0.0"
