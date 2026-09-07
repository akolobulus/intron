"""
Pulls the 4 target-language configs from intronhealth/AfriSwitchCare,
writes each conversation's audio to a local WAV file, and builds
`data/manifest.csv` for the benchmark harness.

Usage:
    huggingface-cli login
    python build_manifest.py
"""

import csv
import os

import numpy as np
import soundfile as sf
from datasets import load_dataset

TARGET_CONFIGS = ["hausa", "pidgin", "igbo", "yoruba"]
CLIPS_DIR = os.path.join("data", "clips")
MANIFEST_PATH = os.path.join("data", "manifest.csv")
TARGET_SR = 16000


def _extract_audio_and_sr(example):
    """Return (samples, sample_rate) for a dataset row."""
    audio = example["audio"]

    if hasattr(audio, "get_all_samples"):
        samples = audio.get_all_samples()
        array = samples.data.numpy()
        sr = int(audio.metadata.sample_rate)
    elif isinstance(audio, dict):
        array = audio["array"]
        sr = int(audio["sampling_rate"])
    else:
        raise TypeError(f"Unsupported audio object type: {type(audio)!r}")

    arr = np.asarray(array)
    if arr.ndim > 1:
        arr = arr.squeeze(0)

    if arr.dtype.kind == "f":
        arr = np.clip(arr, -1.0, 1.0)
        arr = np.int16(arr * 32767.0)
    elif arr.dtype != np.int16:
        arr = arr.astype(np.int16, copy=False)

    return arr, int(sr)


def main():
    os.makedirs(CLIPS_DIR, exist_ok=True)
    rows = []

    for cfg in TARGET_CONFIGS:
        print(f"Loading config: {cfg}")
        ds = load_dataset("intronhealth/AfriSwitchCare", cfg, split="test")

        for i, example in enumerate(ds):
            audio_array, sr = _extract_audio_and_sr(example)
            if sr != TARGET_SR:
                raise ValueError(
                    f"Unexpected sample rate for {cfg}: got {sr}, expected {TARGET_SR}."
                )

            out_path = os.path.join(CLIPS_DIR, f"{cfg}_{i:02d}.wav")
            sf.write(out_path, audio_array, sr, subtype="PCM_16")

            rows.append(
                {
                    "audio_path": out_path,
                    "reference_text": example["transcription"],
                    "language": f"{cfg}-english",
                    "source": "afriswitchcare",
                    "diagnosis": example["diagnosis"],
                    "duration_sec": round(float(example["duration"]), 1),
                    "cmi": float(example["cmi"]),
                }
            )
            print(f"  saved {out_path} ({float(example['duration']):.1f}s, {example['diagnosis']})")

    if not rows:
        raise RuntimeError("No rows were loaded from AfriSwitchCare.")

    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows)} conversations to {MANIFEST_PATH}")
    print("Manifest includes long-form conversations; sahara_adapter.py handles chunking across the 300s API limit.")


if __name__ == "__main__":
    main()
