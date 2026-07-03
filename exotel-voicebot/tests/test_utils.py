"""Unit tests for app.utils.retry_async."""

from __future__ import annotations

import pytest

from app.utils import retry_async


class FlakyError(Exception):
    pass


class OtherError(Exception):
    pass


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_succeeds_on_first_try_without_retrying(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            return "ok"

        result = await retry_async(op, retries=2, base_delay_seconds=0.001)
        assert result == "ok"
        assert calls == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            if calls < 3:
                raise FlakyError("transient")
            return "ok"

        result = await retry_async(
            op, retries=3, base_delay_seconds=0.001, retry_on=(FlakyError,)
        )
        assert result == "ok"
        assert calls == 3

    @pytest.mark.asyncio
    async def test_raises_after_exhausting_retries(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise FlakyError("always fails")

        with pytest.raises(FlakyError):
            await retry_async(op, retries=2, base_delay_seconds=0.001, retry_on=(FlakyError,))
        assert calls == 3  # initial attempt + 2 retries

    @pytest.mark.asyncio
    async def test_does_not_retry_unlisted_exceptions(self) -> None:
        calls = 0

        async def op() -> str:
            nonlocal calls
            calls += 1
            raise OtherError("not retryable")

        with pytest.raises(OtherError):
            await retry_async(op, retries=3, base_delay_seconds=0.001, retry_on=(FlakyError,))
        assert calls == 1
