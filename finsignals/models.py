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
