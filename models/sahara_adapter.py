"""
Sahara v2.5 adapter using the async Upload File API.

This is the confirmed flow from the Intron docs:
  1. POST multipart/form-data to /file/v1/upload -> get file_id
  2. Poll GET /file/v1/status/{file_id} until terminal state
  3. Read audio_transcript from the completed response

This avoids the Runtime/WS protocol problems from the earlier guessed
WebSocket schema and handles long AfriSwitchCare conversation recordings
without needing session chunking.
"""

import os
import time

import requests

UPLOAD_URL = "https://infer.voice.intron.io/file/v1/upload"
STATUS_URL = "https://infer.voice.intron.io/file/v1/status/{file_id}"
POLL_INTERVAL_SEC = 5
POLL_TIMEOUT_SEC = 1800
TERMINAL_STATUSES = {"FILE_TRANSCRIBED", "FILE_PROCESSING_FAILED"}


def _upload(audio_path: str, language_code: str, api_key: str) -> str:
    headers = {"Authorization": f"Bearer {api_key}"}
    file_name = os.path.basename(audio_path)

    with open(audio_path, "rb") as f:
        files = {"audio_file_blob": (file_name, f, "audio/wav")}
        data = {
            "audio_file_name": file_name,
            "use_language_asr_input": language_code,
        }
        resp = requests.post(UPLOAD_URL, headers=headers, files=files, data=data, timeout=120)

    resp.raise_for_status()
    payload = resp.json()
    file_id = payload["data"]["file_id"]
    return file_id


def _poll_until_done(file_id: str, api_key: str) -> dict:
    headers = {"Authorization": f"Bearer {api_key}"}
    url = STATUS_URL.format(file_id=file_id)
    start = time.monotonic()

    while True:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()["data"]
        status = payload.get("processing_status")

        if status in TERMINAL_STATUSES:
            return payload

        if time.monotonic() - start > POLL_TIMEOUT_SEC:
            raise TimeoutError(f"file_id={file_id} did not complete within {POLL_TIMEOUT_SEC}s")

        time.sleep(POLL_INTERVAL_SEC)


def transcribe(audio_path: str, language_code: str = "en") -> dict:
    """Synchronous wrapper. language_code should match Sahara's supported model codes."""
    api_key = os.environ.get("SAHARA_API_KEY")
    if not api_key:
        raise EnvironmentError("Set SAHARA_API_KEY before running the benchmark.")

    start = time.monotonic()
    file_id = _upload(audio_path, language_code, api_key)
    result = _poll_until_done(file_id, api_key)
    latency = time.monotonic() - start

    if result.get("processing_status") == "FILE_PROCESSING_FAILED":
        raise RuntimeError(f"Sahara transcription failed for {audio_path}: {result}")

    return {
        "transcript": result.get("audio_transcript", ""),
        "latency_sec": round(latency, 3),
        "file_id": file_id,
    }
