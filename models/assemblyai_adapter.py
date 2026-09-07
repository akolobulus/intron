"""
AssemblyAI adapter — hosted async upload and poll flow for a global,
general-purpose ASR comparison model.
"""
import os
import time

import requests

BASE_URL = "https://api.assemblyai.com"


def _upload(audio_path: str, api_key: str) -> str:
    headers = {"authorization": api_key}
    with open(audio_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/v2/upload", headers=headers, data=f, timeout=300)
    resp.raise_for_status()
    return resp.json()["upload_url"]


def _submit(upload_url: str, api_key: str) -> str:
    headers = {"authorization": api_key, "content-type": "application/json"}
    resp = requests.post(
        f"{BASE_URL}/v2/transcript",
        json={"audio_url": upload_url},
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["id"]


def _poll(transcript_id: str, api_key: str, timeout_sec: int = 1800, interval_sec: int = 5) -> dict:
    headers = {"authorization": api_key}
    start = time.monotonic()
    while True:
        resp = requests.get(f"{BASE_URL}/v2/transcript/{transcript_id}", headers=headers, timeout=30)
        resp.raise_for_status()
        payload = resp.json()
        status = payload.get("status")
        if status == "completed":
            return payload
        if status == "error":
            raise RuntimeError(f"AssemblyAI transcription failed: {payload.get('error')}")
        if time.monotonic() - start > timeout_sec:
            raise TimeoutError(f"transcript_id={transcript_id} did not complete within {timeout_sec}s")
        time.sleep(interval_sec)


def transcribe(audio_path: str, language_code: str = None) -> dict:
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set ASSEMBLYAI_API_KEY before running the benchmark.")
    start = time.monotonic()
    upload_url = _upload(audio_path, api_key)
    transcript_id = _submit(upload_url, api_key)
    result = _poll(transcript_id, api_key)
    return {"transcript": result.get("text", "") or "", "latency_sec": round(time.monotonic() - start, 3)}