#!/usr/bin/env python3
"""Terminal chat helper for the Exotel voicebot backend.

This is a local/debug client. It does not use Exotel's WebSocket audio
protocol. Instead, it loads the same backend code and lets you talk to the
same OpenAI-powered agent from the terminal. Optionally, it can also synthesize
the bot reply with ElevenLabs and save/play a WAV file.

Usage:
    python3 talk_to_voicebot.py
    python3 talk_to_voicebot.py --agent-id YOUR_AGENT_ID
    python3 talk_to_voicebot.py --agent-id YOUR_AGENT_ID --voice
    python3 talk_to_voicebot.py --agent-id YOUR_AGENT_ID --voice --play
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import subprocess
import sys
import wave
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = ROOT_DIR / "exotel-voicebot"
ENV_FILE = PROJECT_DIR / ".env"


def _load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE lines from `.env` without needing python-dotenv."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _prepare_imports() -> None:
    if not PROJECT_DIR.exists():
        raise SystemExit(f"Project folder not found: {PROJECT_DIR}")
    _load_env_file(ENV_FILE)
    os.chdir(PROJECT_DIR)
    sys.path.insert(0, str(PROJECT_DIR))


def _write_ulaw_8000_as_wav(mulaw_audio: bytes, output_path: Path) -> None:
    from app.audio_utils import mulaw_to_pcm16

    pcm16 = mulaw_to_pcm16(mulaw_audio)
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(8000)
        wav_file.writeframes(pcm16)


def _try_play_audio(path: Path) -> None:
    """Best-effort local playback on Linux/macOS if a player is installed."""
    for command in (
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", str(path)],
        ["aplay", str(path)],
        ["paplay", str(path)],
        ["play", str(path)],
    ):
        if shutil.which(command[0]):
            subprocess.run(command, check=False)
            return
    print(f"(Audio saved at {path}, but no player found: install ffmpeg/aplay/paplay)")


async def _speak_reply(text: str, voice_id: str, reply_index: int, play: bool) -> None:
    from app.services import elevenlabs_tts

    audio_parts: list[bytes] = []
    async for chunk in elevenlabs_tts.stream_tts(text, voice_id=voice_id, call_sid="terminal"):
        audio_parts.append(chunk)

    if not audio_parts:
        print("(ElevenLabs TTS returned no audio)")
        return

    output_path = ROOT_DIR / f"voicebot_reply_{reply_index:03d}.wav"
    _write_ulaw_8000_as_wav(b"".join(audio_parts), output_path)
    print(f"(Voice saved: {output_path})")
    if play:
        _try_play_audio(output_path)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Talk to the voicebot agent from terminal.")
    parser.add_argument("--agent-id", default="", help="Agent id to fetch config from Lovable")
    parser.add_argument("--voice", action="store_true", help="Also synthesize replies via ElevenLabs")
    parser.add_argument("--play", action="store_true", help="Play generated WAV replies if a player exists")
    args = parser.parse_args()

    _prepare_imports()

    from app.config import get_settings
    from app.models import AgentConfig, ConversationTurn
    from app.services import lovable_api, openai_llm

    settings = get_settings()
    if not settings.openai_api_key:
        raise SystemExit("OPENAI_API_KEY missing. Add it in exotel-voicebot/.env first.")
    if args.voice and not settings.elevenlabs_api_key:
        raise SystemExit("ELEVENLABS_API_KEY missing. Add it in exotel-voicebot/.env first.")

    if args.agent_id:
        config = await lovable_api.fetch_agent_config(args.agent_id)
    else:
        config = AgentConfig(agent_id="terminal-agent")

    history: list[ConversationTurn] = []
    print("\nConnected to terminal voicebot chat.")
    print("Type your message and press Enter. Type /exit to quit.\n")

    if config.first_message:
        print(f"Bot: {config.first_message}")
        history.append(ConversationTurn(role="assistant", text=config.first_message))
        if args.voice:
            await _speak_reply(config.first_message, config.voice_id, 0, args.play)

    reply_index = 1
    while True:
        try:
            user_text = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            return

        if not user_text:
            continue
        if user_text.lower() in {"/exit", "exit", "quit", "/quit"}:
            print("Bye.")
            return

        print("Bot: ", end="", flush=True)
        reply_parts: list[str] = []
        async for sentence in openai_llm.stream_reply(
            history=history,
            user_text=user_text,
            system_prompt=config.system_prompt,
            call_sid="terminal",
        ):
            print(sentence + " ", end="", flush=True)
            reply_parts.append(sentence)
        print()

        full_reply = " ".join(reply_parts).strip()
        history.append(ConversationTurn(role="user", text=user_text))
        if full_reply:
            history.append(ConversationTurn(role="assistant", text=full_reply))
            if args.voice:
                await _speak_reply(full_reply, config.voice_id, reply_index, args.play)
                reply_index += 1


if __name__ == "__main__":
    asyncio.run(main())
