from dataclasses import dataclass, field
from typing import List, Optional


# ── Classification head results ───────────────────────────────────────────────

@dataclass
class SentimentResult:
    label: str          # "positive" | "negative" | "neutral"
    positive: float
    negative: float
    neutral: float

    _VALID_LABELS = frozenset({"positive", "negative", "neutral"})

    def __post_init__(self):
        if self.label not in self._VALID_LABELS:
            self.label = "neutral"

    def __repr__(self):
        return f"SentimentResult(label={self.label!r}, {self.label}={getattr(self, self.label, float('nan')):.3f})"


@dataclass
class DirectionalityResult:
    label: str          # "bullish" | "bearish" | "neutral_direction"
    bullish: float
    bearish: float
    neutral_direction: float

    _VALID_LABELS = frozenset({"bullish", "bearish", "neutral_direction"})

    def __post_init__(self):
        if self.label not in self._VALID_LABELS:
            self.label = "neutral_direction"

    def __repr__(self):
        return f"DirectionalityResult(label={self.label!r}, {self.label}={getattr(self, self.label, float('nan')):.3f})"


@dataclass
class QualityResult:
    label: str          # "relevant" | "noise" | "spam"
    relevant: float
    noise: float
    spam: float

    _VALID_LABELS = frozenset({"relevant", "noise", "spam"})

    def __post_init__(self):
        if self.label not in self._VALID_LABELS:
            self.label = "noise"

    def __repr__(self):
        return f"QualityResult(label={self.label!r}, {self.label}={getattr(self, self.label, float('nan')):.3f})"


@dataclass
class PostTypeResult:
    label: str          # "dd" | "news_reaction" | "technical_analysis" |
                        # "fundamentals" | "question" | "general"
    dd: float
    news_reaction: float
    technical_analysis: float
    fundamentals: float
    question: float
    general: float

    _VALID_LABELS = frozenset({"dd", "news_reaction", "technical_analysis", "fundamentals", "question", "general"})

    def __post_init__(self):
        if self.label not in self._VALID_LABELS:
            self.label = "general"

    def __repr__(self):
        return f"PostTypeResult(label={self.label!r}, {self.label}={getattr(self, self.label, float('nan')):.3f})"


# ── Per-item output (all 7 heads) ─────────────────────────────────────────────

@dataclass
class ClassificationOutput:
    sentiment: SentimentResult
    directionality: DirectionalityResult
    quality: QualityResult
    post_type: PostTypeResult
    relevance_score: float          # sigmoid output in [0, 1]
    author_confidence: float        # sigmoid output in [0, 1]
    sarcasm: bool

    def __repr__(self):
        return (
            f"ClassificationOutput("
            f"sentiment={self.sentiment.label!r}, "
            f"directionality={self.directionality.label!r}, "
            f"quality={self.quality.label!r}, "
            f"relevance_score={self.relevance_score:.4f}, "
            f"sarcasm={self.sarcasm})"
        )


# ── Top-level response envelopes ──────────────────────────────────────────────

@dataclass
class ClassifyResponse:
    """Response from POST /v1/classify (single item)."""
    model_version: str
    request_id: str
    credits_charged: float
    endpoint_type: str
    endpoint_name: str
    outputs: List[ClassificationOutput]

    @property
    def output(self) -> ClassificationOutput:
        """Convenience accessor — single classify always has exactly one output."""
        return self.outputs[0]

    # Proxy the most common fields directly onto the response object
    # so `result.sentiment` works without `result.output.sentiment`
    @property
    def sentiment(self) -> SentimentResult:
        return self.outputs[0].sentiment

    @property
    def directionality(self) -> DirectionalityResult:
        return self.outputs[0].directionality

    @property
    def quality(self) -> QualityResult:
        return self.outputs[0].quality

    @property
    def post_type(self) -> PostTypeResult:
        return self.outputs[0].post_type

    @property
    def relevance_score(self) -> float:
        return self.outputs[0].relevance_score

    @property
    def author_confidence(self) -> float:
        return self.outputs[0].author_confidence

    @property
    def sarcasm(self) -> bool:
        return self.outputs[0].sarcasm

    def __repr__(self):
        return (
            f"ClassifyResponse("
            f"sentiment={self.sentiment.label!r}, "
            f"directionality={self.directionality.label!r}, "
            f"quality={self.quality.label!r}, "
            f"relevance_score={self.relevance_score:.4f}, "
            f"credits_charged={self.credits_charged})"
        )


@dataclass
class ClassifyBatchResponse:
    """Response from POST /v1/classify/batch (multiple items)."""
    model_version: str
    request_id: str
    credits_charged: float
    endpoint_type: str
    endpoint_name: str
    outputs: List[ClassificationOutput]

    def __len__(self):
        return len(self.outputs)

    def __iter__(self):
        return iter(self.outputs)

    def __getitem__(self, index):
        return self.outputs[index]

    def __repr__(self):
        return (
            f"ClassifyBatchResponse("
            f"items={len(self.outputs)}, "
            f"credits_charged={self.credits_charged})"
        )


# ── Usage / plan responses ────────────────────────────────────────────────────

@dataclass
class UsageResponse:
    plan: str
    monthly_credits_total: float
    monthly_credits_used: float
    monthly_credits_remaining: float
    payg_balance: float
    usage_by_endpoint: List[dict] = field(default_factory=list)


@dataclass
class RateLimits:
    single: int
    batch: int


@dataclass
class PlanResponse:
    plan: str
    rate_limits: RateLimits
    monthly_credits: float
    batch_max_items: int


# ── Internal helpers ──────────────────────────────────────────────────────────

def _parse_sentiment(d: dict) -> SentimentResult:
    return SentimentResult(
        label=d["label"],
        positive=d.get("positive", 0.0),
        negative=d.get("negative", 0.0),
        neutral=d.get("neutral", 0.0),
    )


def _parse_directionality(d: dict) -> DirectionalityResult:
    return DirectionalityResult(
        label=d["label"],
        bullish=d.get("bullish", 0.0),
        bearish=d.get("bearish", 0.0),
        neutral_direction=d.get("neutral_direction", 0.0),
    )


def _parse_quality(d: dict) -> QualityResult:
    return QualityResult(
        label=d["label"],
        relevant=d.get("relevant", 0.0),
        noise=d.get("noise", 0.0),
        spam=d.get("spam", 0.0),
    )


def _parse_post_type(d: dict) -> PostTypeResult:
    return PostTypeResult(
        label=d["label"],
        dd=d.get("dd", 0.0),
        news_reaction=d.get("news_reaction", 0.0),
        technical_analysis=d.get("technical_analysis", 0.0),
        fundamentals=d.get("fundamentals", 0.0),
        question=d.get("question", 0.0),
        general=d.get("general", 0.0),
    )


def _parse_output(d: dict) -> ClassificationOutput:
    return ClassificationOutput(
        sentiment=_parse_sentiment(d["sentiment"]),
        directionality=_parse_directionality(d["directionality"]),
        quality=_parse_quality(d["quality"]),
        post_type=_parse_post_type(d["post_type"]),
        relevance_score=float(d["relevance_score"]),
        author_confidence=float(d["author_confidence"]),
        sarcasm=bool(d["sarcasm"]),
    )


def parse_classify_response(data: dict) -> ClassifyResponse:
    return ClassifyResponse(
        model_version=data["model_version"],
        request_id=data["request_id"],
        credits_charged=float(data["credits_charged"]),
        endpoint_type=data.get("endpoint_type", "single"),
        endpoint_name=data.get("endpoint_name", "reddit_single"),
        outputs=[_parse_output(o) for o in data["outputs"]],
    )


def parse_classify_batch_response(data: dict) -> ClassifyBatchResponse:
    return ClassifyBatchResponse(
        model_version=data["model_version"],
        request_id=data["request_id"],
        credits_charged=float(data["credits_charged"]),
        endpoint_type=data.get("endpoint_type", "batch"),
        endpoint_name=data.get("endpoint_name", "reddit_batch"),
        outputs=[_parse_output(o) for o in data["outputs"]],
    )


def parse_usage_response(data: dict) -> UsageResponse:
    return UsageResponse(
        plan=data["plan"],
        monthly_credits_total=float(data["monthly_credits_total"]),
        monthly_credits_used=float(data["monthly_credits_used"]),
        monthly_credits_remaining=float(data["monthly_credits_remaining"]),
        payg_balance=float(data.get("payg_balance", 0.0)),
        usage_by_endpoint=data.get("usage_by_endpoint", []),
    )


def parse_plan_response(data: dict) -> PlanResponse:
    rl = data.get("rate_limits", {})
    return PlanResponse(
        plan=data["plan"],
        rate_limits=RateLimits(
            single=rl.get("single", 0),
            batch=rl.get("batch", 0),
        ),
        monthly_credits=float(data.get("monthly_credits", 0)),
        batch_max_items=int(data.get("batch_max_items", 256)),
    )


# ── Sector Rotation models ─────────────────────────────────────────────────────
# These are entirely separate from the Reddit Sentiment models above.
# See GET /v1/sector-rotation and https://finsignals.ai/sector-rotation-api/

@dataclass
class SpyMetrics:
    """SPY benchmark returns used as the baseline for relative-strength calculations."""
    ret_1m: Optional[float]
    ret_3m: Optional[float]
    ret_6m: Optional[float]
    ret_9m: Optional[float]
    ret_12m: Optional[float]


@dataclass
class SectorEntry:
    """
    Rotation metrics for a single sector ETF.

    ``name`` is the sector label (e.g. ``"Technology"``).
    RS fields measure return vs SPY over that window.
    For the 5-year outlook the RS windows are longer; see
    ``RotationPeriod.rs_window_labels`` for human-readable column names.
    """
    name: str
    etf: Optional[str]
    phase: Optional[str]
    confidence: Optional[float]
    rotation_score: Optional[float]
    adjustment: Optional[float]
    rs_1m: Optional[float]
    rs_3m: Optional[float]
    rs_6m: Optional[float]
    rs_9m: Optional[float]
    rs_12m: Optional[float]
    ret_1m: Optional[float]
    mom_accel: Optional[float]
    vol_ratio: Optional[float]
    pe: Optional[float]

    def __repr__(self):
        return (
            f"SectorEntry(name={self.name!r}, etf={self.etf!r}, "
            f"phase={self.phase!r}, rotation_score={self.rotation_score})"
        )


@dataclass
class IndustryEntry:
    """
    Rotation metrics for a single industry ETF.

    ``parent_sector_etf`` is the ETF ticker of the parent sector.
    ``rs_vs_sector_3m`` is the industry RS minus the parent sector RS
    over the 3-month window — positive means the industry is leading its sector.
    """
    name: str
    etf: Optional[str]
    phase: Optional[str]
    confidence: Optional[float]
    rotation_score: Optional[float]
    adjustment: Optional[float]
    rs_1m: Optional[float]
    rs_3m: Optional[float]
    rs_6m: Optional[float]
    rs_vs_sector_3m: Optional[float]
    mom_accel: Optional[float]
    pe: Optional[float]
    parent_sector_etf: Optional[str]

    def __repr__(self):
        return (
            f"IndustryEntry(name={self.name!r}, etf={self.etf!r}, "
            f"phase={self.phase!r}, rs_vs_sector_3m={self.rs_vs_sector_3m})"
        )


@dataclass
class RotationPeriod:
    """
    One outlook period (1-year or 5-year) within a SectorRotationResponse.

    ``sector_data`` and ``industry_data`` are lists of typed entries,
    converted from the dict-keyed API payload.

    ``weekly_snapshots`` are kept as raw dicts because their nested structure
    is large and variable (52 weeks for 1y, 260 weeks for 5y).

    ``rs_window_labels`` is only populated on the 5-year outlook and maps
    API field names to display labels, e.g.
    ``{"rs_1m": "RS 3M", "rs_3m": "RS 6M", "rs_6m": "RS 1Y", ...}``.
    """
    trading_date: str
    generated_at: str
    spy_metrics: SpyMetrics
    sector_data: List[SectorEntry]
    industry_data: List[IndustryEntry]
    summary_md: Optional[str]
    weekly_snapshots: List[dict]
    rs_window_labels: Optional[dict]


@dataclass
class SectorRotationResponse:
    """
    Response from GET /v1/sector-rotation.

    Contains both a 1-year and a 5-year outlook. Each outlook has its own
    sector/industry tables, SPY benchmark, AI summary, and historical snapshots.
    The two outlooks differ in how far back the RS windows look:
    the 1-year view uses standard 1M/3M/6M/12M windows, while the 5-year
    view uses proportionally longer windows (3M/6M/1Y/3Y).

    Cost: 10 credits per call.
    """
    request_id: str
    model_version: str
    credits_charged: float
    trading_date: str
    generated_at: str
    outlook_1y: RotationPeriod
    outlook_5y: RotationPeriod

    def __repr__(self):
        return (
            f"SectorRotationResponse("
            f"trading_date={self.trading_date!r}, "
            f"sectors_1y={len(self.outlook_1y.sector_data)}, "
            f"sectors_5y={len(self.outlook_5y.sector_data)}, "
            f"credits_charged={self.credits_charged})"
        )


# ── Sector Rotation parsers ────────────────────────────────────────────────────

def _parse_spy_metrics(d: dict) -> SpyMetrics:
    return SpyMetrics(
        ret_1m=d.get("ret_1m"),
        ret_3m=d.get("ret_3m"),
        ret_6m=d.get("ret_6m"),
        ret_9m=d.get("ret_9m"),
        ret_12m=d.get("ret_12m"),
    )


def _f(d: dict, key: str) -> Optional[float]:
    """Safe float coerce — returns None if key missing or value is None."""
    v = d.get(key)
    return float(v) if v is not None else None


def _parse_sector_entry(name: str, d: dict) -> SectorEntry:
    return SectorEntry(
        name=name,
        etf=d.get("etf"),
        phase=d.get("phase"),
        confidence=_f(d, "confidence"),
        rotation_score=_f(d, "rotation_score"),
        adjustment=_f(d, "adjustment"),
        rs_1m=_f(d, "rs_1m"),
        rs_3m=_f(d, "rs_3m"),
        rs_6m=_f(d, "rs_6m"),
        rs_9m=_f(d, "rs_9m"),
        rs_12m=_f(d, "rs_12m"),
        ret_1m=_f(d, "ret_1m"),
        mom_accel=_f(d, "mom_accel"),
        vol_ratio=_f(d, "vol_ratio"),
        pe=_f(d, "pe"),
    )


def _parse_industry_entry(name: str, d: dict) -> IndustryEntry:
    return IndustryEntry(
        name=name,
        etf=d.get("etf"),
        phase=d.get("phase"),
        confidence=_f(d, "confidence"),
        rotation_score=_f(d, "rotation_score"),
        adjustment=_f(d, "adjustment"),
        rs_1m=_f(d, "rs_1m"),
        rs_3m=_f(d, "rs_3m"),
        rs_6m=_f(d, "rs_6m"),
        rs_vs_sector_3m=_f(d, "rs_vs_sector_3m"),
        mom_accel=_f(d, "mom_accel"),
        pe=_f(d, "pe"),
        parent_sector_etf=d.get("parent_sector_etf") or None,
    )


def _parse_rotation_period(d: dict) -> RotationPeriod:
    raw_sectors    = d.get("sector_data", {}) or {}
    raw_industries = d.get("industry_data", {}) or {}

    # Build SPY metrics from the top-level spy_metrics dict if present;
    # fall back to the flat spy_ret_* keys that the API also provides.
    spy_raw = d.get("spy_metrics") or {}
    spy = _parse_spy_metrics({
        "ret_1m":  spy_raw.get("ret_1m")  or d.get("spy_ret_1m"),
        "ret_3m":  spy_raw.get("ret_3m")  or d.get("spy_ret_3m"),
        "ret_6m":  spy_raw.get("ret_6m")  or d.get("spy_ret_6m"),
        "ret_9m":  spy_raw.get("ret_9m")  or d.get("spy_ret_9m"),
        "ret_12m": spy_raw.get("ret_12m") or d.get("spy_ret_12m"),
    })

    return RotationPeriod(
        trading_date=str(d.get("trading_date", "")),
        generated_at=str(d.get("generated_at", "")),
        spy_metrics=spy,
        sector_data=[
            _parse_sector_entry(name, v)
            for name, v in raw_sectors.items()
            if isinstance(v, dict)
        ],
        industry_data=[
            _parse_industry_entry(name, v)
            for name, v in raw_industries.items()
            if isinstance(v, dict)
        ],
        summary_md=d.get("summary_md"),
        weekly_snapshots=list(d.get("weekly_snapshots") or []),
        rs_window_labels=d.get("rs_window_labels") or None,
    )


def parse_sector_rotation_response(data: dict) -> SectorRotationResponse:
    return SectorRotationResponse(
        request_id=data["request_id"],
        model_version=data["model_version"],
        credits_charged=float(data["credits_charged"]),
        trading_date=str(data["trading_date"]),
        generated_at=str(data.get("generated_at", "")),
        outlook_1y=_parse_rotation_period(data.get("outlook_1y") or {}),
        outlook_5y=_parse_rotation_period(data.get("outlook_5y") or {}),
    )
