#!/usr/bin/env python3
"""Talk to the hosted Exotel voicebot WebSocket from your terminal.

This script behaves like a tiny Exotel client:
1. Connects to your hosted WebSocket URL.
2. Sends Exotel-style `connected` and `start` JSON events.
3. Captures microphone audio, converts PCM16 -> mu-law 8kHz, and sends
   `media` frames to the backend.
4. Receives backend `media` events, converts mu-law -> PCM16, and plays the
   bot audio on your speakers.

Install deps on your local machine:
    sudo apt install -y portaudio19-dev
    python3 -m pip install websockets sounddevice numpy

Run:
    python3 talk_to_vps_websocket.py
    python3 talk_to_vps_websocket.py --agent-id YOUR_AGENT_ID
    python3 talk_to_vps_websocket.py --url ws://80.241.209.69:8001/exotel/voicebot

Notes:
    - This is for local debugging, not production.
    - Use headphones to avoid speaker audio feeding back into the mic.
    - Press Ctrl+C to end the test call.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import uuid
from collections import deque

import numpy as np
import sounddevice as sd
import websockets

try:
    import audioop  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    import audioop_lts as audioop  # type: ignore[import-not-found,no-redef]


DEFAULT_WS_URL = "ws://80.241.209.69:8001/exotel/voicebot"
SAMPLE_RATE = 8000
FRAME_MS = 20
SAMPLES_PER_FRAME = int(SAMPLE_RATE * FRAME_MS / 1000)
PCM_FRAME_BYTES = SAMPLES_PER_FRAME * 2


def pcm16_to_mulaw(pcm_bytes: bytes) -> bytes:
    return audioop.lin2ulaw(pcm_bytes, 2)


def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
    return audioop.ulaw2lin(mulaw_bytes, 2)


async def send_exotel_start(websocket, agent_id: str, call_sid: str, stream_sid: str) -> None:
    await websocket.send(
        json.dumps({"event": "connected", "protocol": "websocket", "version": "1.0.0"})
    )
    await websocket.send(
        json.dumps(
            {
                "event": "start",
                "sequence_number": "1",
                "start": {
                    "stream_sid": stream_sid,
                    "call_sid": call_sid,
                    "account_sid": "terminal-account",
                    "from": "+910000000001",
                    "to": "+910000000002",
                    "custom_parameters": {"agent_id": agent_id} if agent_id else {},
                    "media_format": {
                        "encoding": "audio/x-mulaw",
                        "sample_rate": SAMPLE_RATE,
                        "channels": 1,
                    },
                },
            }
        )
    )


async def mic_sender(websocket, stream_sid: str) -> None:
    """Capture mic PCM16 frames and send Exotel-style mu-law media events."""
    audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=100)
    loop = asyncio.get_running_loop()
    sequence = 2
    chunk = 1
    started_at = time.monotonic()

    def callback(indata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"\n[mic warning] {status}")
        pcm = np.asarray(indata[:, 0], dtype=np.int16).tobytes()
        try:
            loop.call_soon_threadsafe(audio_queue.put_nowait, pcm)
        except asyncio.QueueFull:
            pass

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=SAMPLES_PER_FRAME,
        callback=callback,
    ):
        print("Mic live. Speak now. Press Ctrl+C to stop.")
        while True:
            pcm_frame = await audio_queue.get()
            if len(pcm_frame) < PCM_FRAME_BYTES:
                pcm_frame = pcm_frame.ljust(PCM_FRAME_BYTES, b"\x00")
            elif len(pcm_frame) > PCM_FRAME_BYTES:
                pcm_frame = pcm_frame[:PCM_FRAME_BYTES]

            mulaw = pcm16_to_mulaw(pcm_frame)
            payload = base64.b64encode(mulaw).decode("ascii")
            await websocket.send(
                json.dumps(
                    {
                        "event": "media",
                        "sequence_number": str(sequence),
                        "stream_sid": stream_sid,
                        "media": {
                            "chunk": str(chunk),
                            "timestamp": str(int((time.monotonic() - started_at) * 1000)),
                            "payload": payload,
                        },
                    }
                )
            )
            sequence += 1
            chunk += 1


async def bot_audio_receiver(websocket) -> None:
    """Receive backend mu-law media events and play bot audio."""
    playback_queue: deque[bytes] = deque()
    queue_event = asyncio.Event()
    loop = asyncio.get_running_loop()

    def playback_callback(outdata, frames, time_info, status) -> None:  # noqa: ANN001
        if status:
            print(f"\n[playback warning] {status}")

        needed = frames * 2
        chunks: list[bytes] = []
        have = 0
        while playback_queue and have < needed:
            chunk = playback_queue.popleft()
            take = min(len(chunk), needed - have)
            chunks.append(chunk[:take])
            have += take
            if take < len(chunk):
                playback_queue.appendleft(chunk[take:])

        pcm = b"".join(chunks).ljust(needed, b"\x00")
        outdata[:, 0] = np.frombuffer(pcm, dtype=np.int16)

    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=SAMPLES_PER_FRAME,
        callback=playback_callback,
    ):
        async for raw_message in websocket:
            try:
                message = json.loads(raw_message)
            except json.JSONDecodeError:
                print(f"\n[non-json message] {raw_message}")
                continue

            event = message.get("event")
            if event == "media":
                payload = message.get("media", {}).get("payload", "")
                if payload:
                    mulaw = base64.b64decode(payload)
                    playback_queue.append(mulaw_to_pcm16(mulaw))
                    queue_event.set()
            elif event == "mark":
                mark_name = message.get("mark", {}).get("name", "")
                print(f"\n[bot mark] {mark_name}")
            elif event == "clear":
                playback_queue.clear()
                print("\n[bot clear]")
            else:
                print(f"\n[server event] {message}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Voice-chat with hosted Exotel WS backend.")
    parser.add_argument("--url", default=DEFAULT_WS_URL, help="Hosted WebSocket URL")
    parser.add_argument("--agent-id", default="", help="Optional agent_id custom parameter")
    args = parser.parse_args()

    call_sid = uuid.uuid4().hex
    stream_sid = f"stream-{uuid.uuid4().hex[:12]}"

    print(f"Connecting to {args.url}")
    print(f"call_sid={call_sid}")
    if args.agent_id:
        print(f"agent_id={args.agent_id}")
    else:
        print("agent_id is empty; backend will use default config if Lovable fetch fails.")

    async with websockets.connect(args.url, ping_interval=20, ping_timeout=20) as websocket:
        await send_exotel_start(websocket, args.agent_id, call_sid, stream_sid)
        receiver_task = asyncio.create_task(bot_audio_receiver(websocket))
        sender_task = asyncio.create_task(mic_sender(websocket, stream_sid))
        try:
            await asyncio.gather(receiver_task, sender_task)
        finally:
            stop_event = {
                "event": "stop",
                "sequence_number": "999999",
                "stop": {"call_sid": call_sid},
            }
            try:
                await websocket.send(json.dumps(stop_event))
            except Exception:
                pass
            receiver_task.cancel()
            sender_task.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped.")
