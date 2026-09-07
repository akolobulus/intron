"""
Whisper-large-v3 via Groq's hosted API (console.groq.com/docs/speech-to-text).
Replaces the local HF transformers pipeline — no local download, no
disk/RAM pressure. This is the "global baseline" comparison model
(no Nigerian-language fine-tuning).

Groq caps uploads at 25MB. Our raw 16kHz PCM16 WAV clips can exceed
that on the longest conversations, so each clip is compressed to MP3
via ffmpeg before upload.
"""
import os
import subprocess
import tempfile
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3"


def _compress_to_mp3(wav_path: str) -> str:
    fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
    os.close(fd)
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-ar", "16000", "-ac", "1", "-b:a", "64k", mp3_path],
        check=True,
        capture_output=True,
    )
    return mp3_path


def transcribe(audio_path: str, language_code: str = None) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GROQ_API_KEY before running the benchmark.")

    start = time.monotonic()
    mp3_path = _compress_to_mp3(audio_path)
    try:
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": MODEL, "response_format": "json"}
        if language_code:
            data["language"] = language_code
        with open(mp3_path, "rb") as f:
            files = {"file": (os.path.basename(mp3_path), f, "audio/mpeg")}
            resp = requests.post(GROQ_URL, headers=headers, data=data, files=files, timeout=300)
        resp.raise_for_status()
        transcript = resp.json().get("text", "")
    finally:
        os.remove(mp3_path)

    return {"transcript": transcript, "latency_sec": round(time.monotonic() - start, 3)}