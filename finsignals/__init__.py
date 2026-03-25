"""
FinSignals Python SDK

Classify financial social media posts across 7 dimensions in a single API call.
Access daily sector and industry rotation analysis with 1-year and 5-year outlooks.

    import finsignals

    client = finsignals.Client("fs_your_key_here")

    # Reddit sentiment classification
    result = client.classify(ticker="NVDA", body="Blackwell demand insane 🚀 DD inside")
    print(result.sentiment.label)       # "positive"
    print(result.directionality.label)  # "bullish"
    print(result.relevance_score)       # 0.9137
    print(result.sarcasm)               # False
    print(result.credits_charged)       # 1.0

    # Sector rotation analysis
    rotation = client.get_sector_rotation()
    for sector in rotation.outlook_1y.sector_data:
        print(sector.name, sector.phase, sector.rotation_score)

Full documentation: https://finsignals.ai/docs
"""

__version__ = "0.3.0"

from .client import Client, classify
from .exceptions import (
    APIError,
    AuthenticationError,
    BatchTooLargeError,
    FinSignalsError,
    InsufficientCreditsError,
    RateLimitError,
    ValidationError,
)
from .models import (
    ClassificationOutput,
    ClassifyBatchResponse,
    ClassifyResponse,
    DirectionalityResult,
    IndustryEntry,
    PlanResponse,
    PostTypeResult,
    QualityResult,
    RateLimits,
    RotationPeriod,
    SectorEntry,
    SectorRotationResponse,
    SentimentResult,
    SpyMetrics,
    UsageResponse,
)

__all__ = [
    # Client
    "Client",
    "classify",
    # Reddit Sentiment response types
    "ClassifyResponse",
    "ClassifyBatchResponse",
    "ClassificationOutput",
    "SentimentResult",
    "DirectionalityResult",
    "QualityResult",
    "PostTypeResult",
    "UsageResponse",
    "PlanResponse",
    "RateLimits",
    # Sector Rotation response types
    "SectorRotationResponse",
    "RotationPeriod",
    "SectorEntry",
    "IndustryEntry",
    "SpyMetrics",
    # Exceptions
    "FinSignalsError",
    "AuthenticationError",
    "InsufficientCreditsError",
    "RateLimitError",
    "ValidationError",
    "BatchTooLargeError",
    "APIError",
]
