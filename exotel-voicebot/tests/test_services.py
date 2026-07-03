"""Unit tests for app.services.elevenlabs_stt and app.services.openai_llm."""

from __future__ import annotations

import json

import httpx
import pytest

from app.models import ConversationTurn
from app.services import elevenlabs_stt, openai_llm


class TestElevenLabsSTT:
    @pytest.mark.asyncio
    async def test_empty_audio_returns_empty_string(self) -> None:
        assert await elevenlabs_stt.transcribe(b"") == ""

    @pytest.mark.asyncio
    async def test_transcribe_returns_text_on_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")

        class FakeResponse:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {"text": "hello world"}

        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args) -> None:
                return None

            async def post(self, *args, **kwargs) -> FakeResponse:
                return FakeResponse()

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)

        from app.config import get_settings

        get_settings.cache_clear()
        result = await elevenlabs_stt.transcribe(b"\x00\x01" * 100)
        assert result == "hello world"
        get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_transcribe_returns_empty_on_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, *args, **kwargs) -> None:
                pass

            async def __aenter__(self) -> "FakeClient":
                return self

            async def __aexit__(self, *args) -> None:
                return None

            async def post(self, *args, **kwargs):
                raise httpx.ConnectError("boom")

        monkeypatch.setattr(httpx, "AsyncClient", FakeClient)
        result = await elevenlabs_stt.transcribe(b"\x00\x01" * 100)
        assert result == ""


class TestOpenAILLM:
    @pytest.mark.asyncio
    async def test_stream_reply_yields_sentences(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "test-key")

        async def fake_raw_delta_stream(payload, headers):
            for text in ["Hello there. ", "How are you? ", "Great."]:
                yield text

        monkeypatch.setattr(openai_llm, "_raw_delta_stream", fake_raw_delta_stream)

        sentences = [
            s
            async for s in openai_llm.stream_reply(
                history=[], user_text="hi", system_prompt="Be helpful."
            )
        ]
        assert sentences == ["Hello there.", "How are you?", "Great."]

    @pytest.mark.asyncio
    async def test_stream_reply_yields_fallback_on_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def failing_stream(payload, headers):
            raise httpx.ConnectError("boom")
            yield ""  # pragma: no cover - unreachable, makes this an async generator

        monkeypatch.setattr(openai_llm, "_raw_delta_stream", failing_stream)

        sentences = [
            s
            async for s in openai_llm.stream_reply(
                history=[], user_text="hi", system_prompt="Be helpful."
            )
        ]
        assert sentences == ["Sorry, ek moment ruko."]

    @pytest.mark.asyncio
    async def test_history_converted_to_openai_roles(self) -> None:
        history = [
            ConversationTurn(role="user", text="hi"),
            ConversationTurn(role="assistant", text="hello"),
        ]
        messages = openai_llm._history_to_openai_messages(history)
        assert messages == [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
