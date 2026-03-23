"""
FinSignals Python SDK

Classify financial social media posts across 7 dimensions in a single API call.

    import finsignals

    client = finsignals.Client("fs_your_key_here")

    result = client.classify(ticker="NVDA", body="Blackwell demand insane 🚀 DD inside")
    print(result.sentiment.label)       # "positive"
    print(result.directionality.label)  # "bullish"
    print(result.relevance_score)       # 0.9137
    print(result.sarcasm)               # False
    print(result.credits_charged)       # 1.0

Full documentation: https://finsignals.ai/docs
"""

__version__ = "0.2.0"

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
    PlanResponse,
    PostTypeResult,
    QualityResult,
    RateLimits,
    SentimentResult,
    UsageResponse,
)

__all__ = [
    # Client
    "Client",
    "classify",
    # Response types
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
    # Exceptions
    "FinSignalsError",
    "AuthenticationError",
    "InsufficientCreditsError",
    "RateLimitError",
    "ValidationError",
    "BatchTooLargeError",
    "APIError",
]
