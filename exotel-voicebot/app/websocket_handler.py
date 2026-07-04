"""Core Exotel Voicebot WebSocket session handler.

Owns the full lifecycle of a single call: accepting the connection,
fetching agent config, running the VAD -> STT -> LLM -> TTS pipeline for
each conversational turn, handling barge-in, and reporting the call log
back to the Lovable control plane when the call ends.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import WebSocket, WebSocketDisconnect
from pydantic import TypeAdapter, ValidationError

from app.audio_utils import mulaw_to_pcm16, rms_level
from app.models import (
    AgentConfig,
    ConversationTurn,
    ExotelEvent,
    MediaEvent,
    StartEvent,
    StopEvent,
)
from app.services import elevenlabs_stt, elevenlabs_tts, lovable_api, openai_llm
from app.vad import VADProcessor

logger = structlog.get_logger(__name__)

_event_adapter: TypeAdapter[ExotelEvent] = TypeAdapter(ExotelEvent)

# 8kHz, 16-bit mono => 160 bytes of PCM per 20ms frame; mu-law is half that.
_FRAME_MS = 20
_MULAW_FRAME_BYTES = 160
_PREBUFFER_FRAMES = 5  # ~100ms of pre-speech padding kept for STT accuracy
_MAX_UTTERANCE_SECONDS = 20

# --- Barge-in / echo-suppression tuning -----------------------------------
# Outbound TTS audio can leak back into the inbound audio path (acoustic
# echo on real calls, or full-duplex mic/speaker crosstalk in local
# testing). Without suppression, the bot's own voice gets detected as the
# caller interrupting, causing an infinite cancel/retry loop. Three
# independent guards are combined before a barge-in is allowed to fire:
#
#   1. Echo window: ignore inbound frames for a short period after each
#      outbound audio frame is sent.
#   2. RMS energy threshold: real speech is typically louder than leaked/
#      attenuated echo; frames below this are ignored for barge-in purposes.
#   3. Sustained-speech confirmation: require several consecutive frames
#      that pass both of the above before treating it as genuine barge-in
#      (a few stray loud frames are not enough).
_ECHO_SUPPRESSION_SECONDS = 0.25
_BARGE_IN_RMS_THRESHOLD = 700.0
_BARGE_IN_CONFIRM_MS = 300
_BARGE_IN_CONFIRM_FRAMES = max(1, _BARGE_IN_CONFIRM_MS // _FRAME_MS)


class Metrics:
    """Process-wide counters exposed via the `/metrics` endpoint."""

    def __init__(self) -> None:
        self.active_calls = 0
        self.total_calls = 0
        self._latencies_ms: deque[float] = deque(maxlen=200)

    def record_turn_latency(self, latency_ms: float) -> None:
        """Record time-to-first-audio-byte for one conversational turn."""
        self._latencies_ms.append(latency_ms)

    @property
    def avg_latency_ms(self) -> float:
        """Average of the most recent turn latencies, in milliseconds."""
        if not self._latencies_ms:
            return 0.0
        return sum(self._latencies_ms) / len(self._latencies_ms)


metrics = Metrics()


@dataclass
class SessionState:
    """Mutable state tracked for a single Exotel WebSocket call."""

    websocket: WebSocket
    call_sid: str = ""
    stream_sid: str = ""
    from_number: str = ""
    to_number: str = ""
    agent_id: str = ""
    config: AgentConfig = field(default_factory=lambda: AgentConfig(agent_id=""))
    history: list[ConversationTurn] = field(default_factory=list)
    vad: VADProcessor = field(default_factory=VADProcessor)
    audio_buffer: bytearray = field(default_factory=bytearray)
    prebuffer: deque = field(default_factory=lambda: deque(maxlen=_PREBUFFER_FRAMES))
    is_bot_speaking: bool = False
    turn_task: asyncio.Task | None = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    ended_at: datetime | None = None
    active: bool = True
    turn_start_monotonic: float | None = None
    turn_latency_recorded: bool = False
    # Monotonic timestamp of the last outbound TTS audio frame sent to
    # Exotel. Used for echo-suppression: inbound frames arriving shortly
    # after outbound audio are more likely to be leaked/echoed bot audio
    # than genuine caller speech. Defaults to 0.0 (far in the past) so no
    # suppression applies before the bot has ever spoken.
    last_bot_audio_monotonic: float = 0.0
    # Consecutive inbound frames that have passed both the echo-suppression
    # window and the RMS threshold while the bot is speaking. Used to
    # require sustained speech (not just a stray loud frame) before firing
    # a real barge-in.
    barge_in_speech_frames: int = 0


async def handle_exotel_websocket(websocket: WebSocket) -> None:
    """Handle a single Exotel Voicebot WebSocket connection end-to-end.

    Accepts the connection, waits for `connected`/`start`, fetches the
    agent config, plays the greeting, then runs the VAD/STT/LLM/TTS loop
    for the lifetime of the call. Never raises - all errors are caught
    and logged so a single bad call cannot crash the server.

    Args:
        websocket: The accepted (but not yet `.accept()`-ed) FastAPI
            WebSocket connection from Exotel.
    """
    await websocket.accept()
    session = SessionState(websocket=websocket)
    log = logger.bind(call_sid="pending")
    metrics.active_calls += 1
    metrics.total_calls += 1

    try:
        async for raw_message in websocket.iter_text():
            try:
                event = _event_adapter.validate_json(raw_message)
            except ValidationError as exc:
                log.warning("websocket.invalid_event", error=str(exc))
                continue

            if event.event == "connected":
                log.info("websocket.connected")

            elif event.event == "start":
                await _handle_start(session, event)
                log = logger.bind(call_sid=session.call_sid)

            elif event.event == "media":
                await _handle_media(session, event, log)

            elif event.event == "stop":
                await _handle_stop(session, event, log)
                break

            elif event.event == "dtmf":
                log.info("websocket.dtmf_received")

    except WebSocketDisconnect:
        log.info("websocket.disconnected")
    except Exception as exc:  # noqa: BLE001 - never crash the server
        log.error("websocket.unhandled_error", error=str(exc))
    finally:
        metrics.active_calls = max(0, metrics.active_calls - 1)
        await _cleanup_session(session, log)


async def _handle_start(session: SessionState, event: StartEvent) -> None:
    """Process the `start` event: load config and play the greeting."""
    start = event.start
    session.stream_sid = start.stream_sid
    session.call_sid = start.call_sid
    session.from_number = start.from_
    session.to_number = start.to
    session.agent_id = start.custom_parameters.get("agent_id", "")

    log = logger.bind(call_sid=session.call_sid)
    log.info(
        "websocket.start",
        from_number=session.from_number,
        to_number=session.to_number,
        agent_id=session.agent_id,
    )

    session.config = await lovable_api.fetch_agent_config(session.agent_id)

    if session.config.first_message:
        await _speak(session, session.config.first_message, log)
        session.history.append(
            ConversationTurn(role="assistant", text=session.config.first_message)
        )


async def _handle_media(session: SessionState, event: MediaEvent, log: Any) -> None:
    """Process one 20ms inbound audio frame: feed VAD, detect turns/barge-in."""
    if not session.call_sid:
        return  # media arriving before start; ignore defensively

    try:
        mulaw_bytes = base64.b64decode(event.media.payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("websocket.bad_media_payload", error=str(exc))
        return

    pcm_frame = mulaw_to_pcm16(mulaw_bytes)
    if not pcm_frame:
        return

    rms = rms_level(pcm_frame)
    state = session.vad.process_frame(pcm_frame)

    log.debug(
        "vad.debug",
        rms=round(rms, 1),
        state=state,
        vad_speaking=session.vad.is_speaking,
        bot_speaking=session.is_bot_speaking,
        active=session.active,
    )

    # Barge-in: caller starts talking while the bot is mid-TTS playback.
    # Outbound TTS audio can leak back into the inbound path (real-call
    # acoustic echo, or full-duplex crosstalk in local testing), which the
    # VAD alone cannot distinguish from genuine speech. Three guards are
    # combined: an echo-suppression window right after outbound audio, an
    # RMS energy floor, and a sustained-speech frame count - all three must
    # hold before a barge-in is allowed to fire.
    if session.is_bot_speaking and state in ("speech", "end_of_utterance"):
        since_bot_audio = time.monotonic() - session.last_bot_audio_monotonic
        if since_bot_audio < _ECHO_SUPPRESSION_SECONDS:
            log.debug(
                "barge_in.suppressed_echo_window",
                rms=round(rms, 1),
                state=state,
                since_bot_audio_s=round(since_bot_audio, 3),
            )
            session.barge_in_speech_frames = 0
        elif rms < _BARGE_IN_RMS_THRESHOLD:
            log.debug(
                "barge_in.suppressed_low_rms",
                rms=round(rms, 1),
                state=state,
                threshold=_BARGE_IN_RMS_THRESHOLD,
            )
            session.barge_in_speech_frames = 0
        else:
            session.barge_in_speech_frames += 1
            speech_ms = session.barge_in_speech_frames * _FRAME_MS
            if session.barge_in_speech_frames >= _BARGE_IN_CONFIRM_FRAMES:
                log.warning(
                    "barge_in.triggered",
                    reason="sustained_speech_during_bot_playback",
                    rms=round(rms, 1),
                    state=state,
                    speech_ms=speech_ms,
                    since_bot_audio_s=round(since_bot_audio, 3),
                    turn_task_active=bool(session.turn_task and not session.turn_task.done()),
                )
                session.barge_in_speech_frames = 0
                await _handle_barge_in(session, log)
    else:
        session.barge_in_speech_frames = 0

    if state == "speech":
        if not session.audio_buffer:
            for prebuffered_frame in session.prebuffer:
                session.audio_buffer.extend(prebuffered_frame)
        session.audio_buffer.extend(pcm_frame)

        max_bytes = _MAX_UTTERANCE_SECONDS * 8000 * 2
        if len(session.audio_buffer) > max_bytes:
            await _finalize_utterance(session, log)

    elif state == "end_of_utterance":
        session.audio_buffer.extend(pcm_frame)
        await _finalize_utterance(session, log)

    else:  # silence
        session.prebuffer.append(pcm_frame)


async def _finalize_utterance(session: SessionState, log: Any) -> None:
    """Snapshot the buffered utterance and kick off turn processing."""
    if not session.audio_buffer:
        return
    utterance_pcm = bytes(session.audio_buffer)
    session.audio_buffer = bytearray()
    session.prebuffer.clear()

    if session.turn_task and not session.turn_task.done():
        return  # a turn is already in flight; drop overlapping audio

    session.turn_start_monotonic = time.monotonic()
    session.turn_latency_recorded = False
    session.turn_task = asyncio.create_task(_process_turn(session, utterance_pcm, log))


async def _handle_barge_in(session: SessionState, log: Any) -> None:
    """Interrupt the currently playing TTS and let the caller take the turn.

    Note: we deliberately do NOT reset `session.vad` here. The VAD already
    confirmed the caller is mid-speech (that's what triggered this barge-in),
    so resetting would force it to re-accumulate `start_speech_ms` worth of
    frames before recognizing speech again, dropping the first ~60ms of the
    caller's new utterance from the buffer.
    """
    log.info("websocket.barge_in")
    try:
        await session.websocket.send_json(
            {"event": "clear", "stream_sid": session.stream_sid}
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("websocket.clear_send_failed", error=str(exc))

    if session.turn_task and not session.turn_task.done():
        session.turn_task.cancel()

    session.is_bot_speaking = False


async def _process_turn(session: SessionState, utterance_pcm: bytes, log: Any) -> None:
    """Run one full conversational turn: STT -> LLM -> TTS -> playback."""
    try:
        transcript = await elevenlabs_stt.transcribe(
            utterance_pcm,
            sample_rate=8000,
            call_sid=session.call_sid,
        )
        if not transcript:
            return

        # ElevenLabs Scribe returns non-speech sounds in square brackets,
        # e.g. "[silence]", "[phone ringing]", "[हँसने की आवाज़]".
        # These are not real utterances — skip them to avoid wasting LLM calls
        # and generating bot responses to background noise.
        stripped = transcript.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            log.info("turn.noise_skipped", transcript=stripped)
            return

        log.info("turn.transcript", transcript=transcript)
        session.history.append(ConversationTurn(role="user", text=transcript))

        session.is_bot_speaking = True
        full_reply_parts: list[str] = []

        async for sentence in openai_llm.stream_reply(
            history=session.history[:-1],
            user_text=transcript,
            system_prompt=session.config.system_prompt,
            call_sid=session.call_sid,
        ):
            if not session.active:
                break
            full_reply_parts.append(sentence)
            await _speak(session, sentence, log)

        if full_reply_parts:
            session.history.append(
                ConversationTurn(role="assistant", text=" ".join(full_reply_parts))
            )

    except asyncio.CancelledError:
        log.info("turn.cancelled_by_barge_in")
        raise
    except Exception as exc:  # noqa: BLE001 - never crash the call loop
        log.error("turn.failed", error=str(exc))
    finally:
        session.is_bot_speaking = False


async def _speak(session: SessionState, text: str, log: Any) -> None:
    """Synthesize `text` via ElevenLabs and stream mu-law audio frames to Exotel.

    No fallback TTS provider: if ElevenLabs yields no audio (e.g. transient
    outage or rate limit), this is logged and the turn is skipped.
    """
    if not text.strip():
        return

    sent_any = False
    # ElevenLabs' HTTP stream delivers arbitrary-sized network chunks that do
    # NOT align to our 160-byte (20ms) Exotel frame size. Accumulate raw
    # mu-law bytes here and only hand off exact 160-byte frames downstream -
    # otherwise each network chunk's leftover tail becomes its own
    # short/misaligned "frame", which is what was causing audible
    # noise/static on real Exotel calls despite clean standalone playback.
    pending = bytearray()
    try:
        async for mulaw_chunk in elevenlabs_tts.stream_tts(
            text, voice_id=session.config.voice_id, call_sid=session.call_sid
        ):
            if not session.active:
                return
            pending.extend(mulaw_chunk)
            while len(pending) >= _MULAW_FRAME_BYTES:
                frame = bytes(pending[:_MULAW_FRAME_BYTES])
                del pending[:_MULAW_FRAME_BYTES]
                sent_any = True
                await _send_mulaw_audio(session, frame)

        if not session.active:
            return

        # Flush the trailing partial frame (< 20ms) so the last syllable of
        # this sentence isn't dropped.
        if pending:
            sent_any = True
            await _send_mulaw_audio(session, bytes(pending))

        if not sent_any:
            log.warning("tts.no_audio_produced", text_len=len(text))
            return

        await _send_mark(session, text)

    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - never crash the call loop
        log.error("tts.failed", error=str(exc))


async def _send_mulaw_audio(session: SessionState, mulaw_bytes: bytes) -> None:
    """Chunk mu-law audio into 20ms frames and stream them to Exotel in real time."""
    for frame in _iter_fixed_chunks(mulaw_bytes, _MULAW_FRAME_BYTES):
        if not session.active:
            return
        payload = base64.b64encode(frame).decode("ascii")
        try:
            await session.websocket.send_json(
                {
                    "event": "media",
                    "stream_sid": session.stream_sid,
                    "media": {"payload": payload},
                }
            )
        except Exception:
            # The socket is gone (caller hung up / disconnect race with an
            # in-flight turn task). Mark the session inactive immediately so
            # every other concurrent send/receive path stops instead of
            # retrying against a dead connection.
            session.active = False
            raise

        session.last_bot_audio_monotonic = time.monotonic()
        logger.debug(
            "tts.frame_sent",
            call_sid=session.call_sid,
            bytes=len(frame),
            bot_speaking=session.is_bot_speaking,
        )
        if not session.turn_latency_recorded and session.turn_start_monotonic is not None:
            session.turn_latency_recorded = True
            metrics.record_turn_latency(
                (time.monotonic() - session.turn_start_monotonic) * 1000
            )
        await asyncio.sleep(_FRAME_MS / 1000)


async def _send_mark(session: SessionState, label: str) -> None:
    """Send a `mark` event so playback completion of a TTS segment can be tracked."""
    try:
        await session.websocket.send_json(
            {
                "event": "mark",
                "stream_sid": session.stream_sid,
                "mark": {"name": label[:64]},
            }
        )
    except Exception:  # noqa: BLE001 - marks are best-effort
        pass


def _iter_fixed_chunks(data: bytes, size: int):
    """Yield fixed-size chunks of `data` (final chunk may be shorter)."""
    for i in range(0, len(data), size):
        yield data[i : i + size]


async def _handle_stop(session: SessionState, event: StopEvent, log: Any) -> None:
    """Process the `stop` event: finalize state; cleanup posts the call log."""
    log.info("websocket.stop")
    session.active = False
    if session.turn_task and not session.turn_task.done():
        session.turn_task.cancel()


async def _cleanup_session(session: SessionState, log: Any) -> None:
    """Cancel in-flight work and report the completed call to the Lovable app."""
    session.active = False
    if session.turn_task and not session.turn_task.done():
        session.turn_task.cancel()
        try:
            await session.turn_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001
            pass

    if not session.call_sid:
        return  # never got past `start`; nothing to log

    session.ended_at = datetime.now(timezone.utc)
    duration = (session.ended_at - session.started_at).total_seconds()

    call_data = {
        "call_sid": session.call_sid,
        "from": session.from_number,
        "to": session.to_number,
        "agent_id": session.agent_id,
        "transcript": [turn.model_dump(mode="json") for turn in session.history],
        "started_at": session.started_at.isoformat(),
        "ended_at": session.ended_at.isoformat(),
        "duration_seconds": duration,
    }
    log.info("call.ended", duration_seconds=duration, turns=len(session.history))
    await lovable_api.post_call_log(call_data)
