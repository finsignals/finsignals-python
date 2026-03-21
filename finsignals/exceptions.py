class FinSignalsError(Exception):
    """Base class for all FinSignals SDK errors."""
    pass


class AuthenticationError(FinSignalsError):
    """Raised when the API key is missing or invalid (HTTP 401)."""

    def __init__(self, message="Invalid or missing API key. Check your key at finsignals.ai/api-keys."):
        super().__init__(message)


class InsufficientCreditsError(FinSignalsError):
    """Raised when the account has run out of credits (HTTP 402)."""

    def __init__(self, message="Insufficient credits. Add credits or upgrade your plan at finsignals.ai."):
        super().__init__(message)
        self.detail = {}  # populated by the client with the raw 402 detail dict

    @classmethod
    def from_detail(cls, detail: dict):
        err = cls()
        err.detail = detail
        remaining = detail.get("remaining_credits")
        msg_parts = ["Insufficient credits."]
        if remaining is not None:
            msg_parts.append(f"Remaining: {remaining}.")
        options = detail.get("options")
        if options:
            options_str = str(options)[:200]
            msg_parts.append(f"Options: {options_str}.")
        err.args = (" ".join(msg_parts),)
        return err


class RateLimitError(FinSignalsError):
    """Raised when the rate limit is exceeded (HTTP 429)."""

    def __init__(self, retry_after: float = 0, limit: int = 0, window_seconds: int = 60):
        self.retry_after = retry_after
        self.limit = limit
        self.window_seconds = window_seconds
        msg = f"Rate limit exceeded ({limit} req/{window_seconds}s)."
        if retry_after:
            msg += f" Retry after {retry_after:.1f}s."
        super().__init__(msg)

    @classmethod
    def from_detail(cls, detail: dict):
        return cls(
            retry_after=float(detail.get("retry_after", 0)),
            limit=int(detail.get("limit", 0)),
            window_seconds=int(detail.get("window_seconds", 60)),
        )


class ValidationError(FinSignalsError):
    """Raised when the request payload is invalid (HTTP 422)."""

    def __init__(self, errors):
        self.errors = errors
        super().__init__(f"Validation error: {errors}")


class BatchTooLargeError(ValidationError):
    """Raised when a batch exceeds 256 items or 128,000 characters before sending."""

    def __init__(self, count: int = 0, max_items: int = 256):
        self.count = count
        self.max_items = max_items
        FinSignalsError.__init__(
            self,
            f"Batch has {count} items but the maximum is {max_items}. "
            "Split into smaller batches."
        )


class APIError(FinSignalsError):
    """Raised for unexpected HTTP errors not covered by a more specific class."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        safe_msg = message[:500] if len(message) > 500 else message
        super().__init__(f"API error {status_code}: {safe_msg}")
