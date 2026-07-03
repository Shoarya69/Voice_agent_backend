"""Voice Activity Detection (VAD) for streaming 20ms PCM audio frames.

Wraps `webrtcvad` to classify each 20ms frame as speech/silence and expose
higher-level "end of utterance" detection so the WebSocket handler knows
exactly when the caller has stopped talking.
"""

from __future__ import annotations

from typing import Literal

import webrtcvad

FrameState = Literal["speech", "silence", "end_of_utterance"]

# Default aggressiveness: 0 (least aggressive, most permissive) to 3
# (most aggressive at filtering out non-speech). Voicebots benefit from a
# fairly aggressive mode to avoid line-noise triggering false speech.
_DEFAULT_VAD_MODE = 1

# How many ms of continuous silence after speech has started before we
# consider the utterance finished.
_DEFAULT_END_SILENCE_MS = 500

# How many ms of continuous speech are required before we consider the
# caller to have actually started talking (helps ignore blips).
_DEFAULT_START_SPEECH_MS = 60


class VADProcessor:
    """Stateful voice-activity detector for a single call session.

    Feed it consecutive 20ms, 16-bit mono PCM frames via `process_frame`.
    It tracks whether the caller is currently mid-utterance and returns
    `"end_of_utterance"` exactly once, on the frame where enough trailing
    silence has been observed to conclude the caller finished speaking.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        mode: int = _DEFAULT_VAD_MODE,
        end_silence_ms: int = _DEFAULT_END_SILENCE_MS,
        start_speech_ms: int = _DEFAULT_START_SPEECH_MS,
        frame_ms: int = 20,
    ) -> None:
        """Initialize the VAD processor.

        Args:
            sample_rate: Audio sample rate in Hz (webrtcvad supports 8000,
                16000, 32000, 48000).
            mode: webrtcvad aggressiveness, 0-3 (3 = most aggressive).
            end_silence_ms: Trailing silence duration that marks end-of-utterance.
            start_speech_ms: Leading speech duration required to confirm
                that an utterance has actually begun.
            frame_ms: Duration of each frame passed to `process_frame`.
        """
        self._vad = webrtcvad.Vad(mode)
        self._sample_rate = sample_rate
        self._frame_ms = frame_ms
        self._end_silence_frames = max(1, end_silence_ms // frame_ms)
        self._start_speech_frames = max(1, start_speech_ms // frame_ms)

        self._speaking = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    def process_frame(self, pcm_20ms: bytes) -> FrameState:
        """Classify a single 20ms PCM frame and update internal state.

        Args:
            pcm_20ms: 16-bit little-endian mono PCM bytes for exactly one
                frame duration (`frame_ms`, default 20ms).

        Returns:
            `"speech"` while the caller is actively talking, `"silence"`
            when nothing relevant is happening, or `"end_of_utterance"`
            on the single frame where trailing silence confirms the
            caller has finished their turn.
        """
        try:
            is_speech = self._vad.is_speech(pcm_20ms, self._sample_rate)
        except Exception:
            # Malformed/short frame (e.g. last partial chunk) - treat as silence.
            is_speech = False

        if is_speech:
            self._consecutive_speech += 1
            self._consecutive_silence = 0

            if not self._speaking and self._consecutive_speech >= self._start_speech_frames:
                self._speaking = True

            return "speech" if self._speaking else "silence"

        # Frame is silence.
        self._consecutive_speech = 0
        if self._speaking:
            self._consecutive_silence += 1
            if self._consecutive_silence >= self._end_silence_frames:
                self._speaking = False
                self._consecutive_silence = 0
                return "end_of_utterance"
            return "speech"  # still inside an utterance, just a brief pause

        return "silence"

    def reset(self) -> None:
        """Reset internal speech/silence tracking state (e.g. after barge-in)."""
        self._speaking = False
        self._consecutive_speech = 0
        self._consecutive_silence = 0

    @property
    def is_speaking(self) -> bool:
        """Whether the processor currently believes the caller is mid-utterance."""
        return self._speaking
