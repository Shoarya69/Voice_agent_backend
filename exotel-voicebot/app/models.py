"""Pydantic models for Exotel WS events, agent config, and call logs."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Exotel inbound WebSocket events
# ---------------------------------------------------------------------------


class MediaFormat(BaseModel):
    """Audio format metadata sent inside the `start` event."""

    encoding: str = "audio/x-mulaw"
    sample_rate: int = 8000
    channels: int = 1


class StartPayload(BaseModel):
    """Payload of the `start` event, containing call metadata."""

    stream_sid: str
    call_sid: str
    account_sid: str | None = None
    from_: str = Field(default="", alias="from")
    to: str = ""
    custom_parameters: dict[str, str] = Field(default_factory=dict)
    media_format: MediaFormat = Field(default_factory=MediaFormat)

    model_config = {"populate_by_name": True}


class ConnectedEvent(BaseModel):
    """First message Exotel sends once the WebSocket connects."""

    event: Literal["connected"]
    protocol: str | None = None
    version: str | None = None


class StartEvent(BaseModel):
    """Call metadata event."""

    event: Literal["start"]
    sequence_number: str | None = None
    start: StartPayload


class MediaPayload(BaseModel):
    """A single 20ms audio frame."""

    chunk: str | None = None
    timestamp: str | None = None
    payload: str = ""


class MediaEvent(BaseModel):
    """Inbound audio frame event."""

    event: Literal["media"]
    sequence_number: str | None = None
    media: MediaPayload
    stream_sid: str | None = None


class StopPayload(BaseModel):
    """Payload of the `stop` event."""

    call_sid: str | None = None


class StopEvent(BaseModel):
    """Call-ended event."""

    event: Literal["stop"]
    sequence_number: str | None = None
    stop: StopPayload | None = None


class DTMFPayload(BaseModel):
    """DTMF keypad payload."""

    digit: str | None = None


class DTMFEvent(BaseModel):
    """Keypad input event."""

    event: Literal["dtmf"]
    sequence_number: str | None = None
    dtmf: DTMFPayload | None = None


class MarkEvent(BaseModel):
    """Playback mark acknowledgement event (Exotel -> server, rare)."""

    event: Literal["mark"]
    sequence_number: str | None = None
    mark: dict | None = None


ExotelEvent = Annotated[
    Union[ConnectedEvent, StartEvent, MediaEvent, StopEvent, DTMFEvent, MarkEvent],
    Field(discriminator="event"),
]
"""Discriminated union of all inbound Exotel WebSocket events."""


# ---------------------------------------------------------------------------
# Outbound events (server -> Exotel) are built as plain dicts in
# websocket_handler.py, since Exotel expects the exact field names below.
# ---------------------------------------------------------------------------


class OutboundMediaMessage(BaseModel):
    """Server -> Exotel audio frame."""

    event: Literal["media"] = "media"
    stream_sid: str
    media: dict


class OutboundClearMessage(BaseModel):
    """Server -> Exotel barge-in interrupt message."""

    event: Literal["clear"] = "clear"
    stream_sid: str


class OutboundMarkMessage(BaseModel):
    """Server -> Exotel mark message, used to track TTS playback segments."""

    event: Literal["mark"] = "mark"
    stream_sid: str
    mark: dict


# ---------------------------------------------------------------------------
# Agent configuration (fetched from the Lovable control plane)
# ---------------------------------------------------------------------------


class AgentConfig(BaseModel):
    """Voice agent configuration fetched from the Lovable app."""

    agent_id: str
    system_prompt: str = "You are a helpful voice assistant."
    voice_id: str = ""
    first_message: str = "Hello! How can I help you today?"
    language: str = "hin"
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


# ---------------------------------------------------------------------------
# Conversation / call logging
# ---------------------------------------------------------------------------


class ConversationTurn(BaseModel):
    """A single turn in the conversation transcript."""

    role: Literal["user", "assistant"]
    text: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CallLog(BaseModel):
    """Full record of a completed call, posted back to the Lovable app."""

    call_sid: str
    from_: str = Field(default="", alias="from")
    to: str = ""
    agent_id: str = ""
    turns: list[ConversationTurn] = Field(default_factory=list)
    started_at: datetime
    ended_at: datetime | None = None
    duration_seconds: float | None = None

    model_config = {"populate_by_name": True}
