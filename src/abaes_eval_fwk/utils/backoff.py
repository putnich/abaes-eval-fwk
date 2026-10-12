"""Shared exponential backoff delays for retry loops."""

from concurrent.futures import Future

from tenacity import RetryCallState, wait_exponential

_wait = wait_exponential(multiplier=1, min=1, max=10)


def default_backoff(attempt: int) -> float:
    """Delay in seconds before retrying after 1-based `attempt` failed: 1, 2, 4, 8, 10, ..."""
    if attempt < 1:
        raise ValueError("attempt must be positive")
    # wait_exponential reads only attempt_number, so the retry object and call are placeholders.
    state = RetryCallState(None, Future(), (), {})  # type: ignore[arg-type]
    state.attempt_number = attempt
    return _wait(state)


def agent_backoff(attempt_idx: int) -> float:
    """Delay for 0-based agent attempt index `i`: 2, 4, 8, 10, ..."""
    return default_backoff(attempt_idx + 2)
