"""
Main benchmark runner for the Sahara CodeSwitch Africa Main Challenge.

Usage:
    python benchmark.py --manifest data/manifest.csv --out results/
"""
import argparse
import os

import pandas as pd
from tqdm import tqdm

from metrics import score_clip
from models.sahara_adapter import transcribe as sahara_transcribe
from models.groq_adapter import transcribe as groq_transcribe
from models.assemblyai_adapter import transcribe as assemblyai_transcribe

SAHARA_LANG_MAP = {
    "yoruba": "yo",
    "yoruba-english": "yo",
    "hausa": "ha",
    "hausa-english": "ha",
    "igbo": "ig",
    "igbo-english": "ig",
    "pidgin": "pcm",
    "pidgin-english": "pcm",
}

MODELS = {
    "sahara-v2.5": lambda path, lang: sahara_transcribe(
        path, SAHARA_LANG_MAP.get(lang, "en")
    ),
    "whisper-large-v3-groq": lambda path, lang: groq_transcribe(path),
    "assemblyai-universal": lambda path, lang: assemblyai_transcribe(path),
}


def run_benchmark(manifest_path: str, out_dir: str):
    manifest = pd.read_csv(manifest_path)
    required_cols = {"audio_path", "reference_text", "language", "source"}
    missing = required_cols - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")

    rows = []
    for _, row in tqdm(manifest.iterrows(), total=len(manifest), desc="Clips"):
        for model_name, fn in MODELS.items():
            try:
                result = fn(row["audio_path"], row["language"])
                scores = score_clip(row["reference_text"], result["transcript"])
                rows.append({
                    "audio_path": row["audio_path"],
                    "language": row["language"],
                    "source": row["source"],
                    "model": model_name,
                    "hypothesis": result["transcript"],
                    "reference": row["reference_text"],
                    "latency_sec": result["latency_sec"],
                    **scores,
                })
            except Exception as exc:
                rows.append({
                    "audio_path": row["audio_path"],
                    "language": row["language"],
                    "source": row["source"],
                    "model": model_name,
                    "hypothesis": None,
                    "reference": row["reference_text"],
                    "latency_sec": None,
                    "wer_normalized": None,
                    "cer_normalized": None,
                    "wer_raw": None,
                    "cer_raw": None,
                    "error": str(exc),
                })

    detail_df = pd.DataFrame(rows)
    os.makedirs(out_dir, exist_ok=True)
    detail_path = os.path.join(out_dir, "benchmark_report.csv")
    detail_df.to_csv(detail_path, index=False)

    summary = (
        detail_df.dropna(subset=["wer_normalized"])
        .groupby(["language", "model"])[
            ["wer_normalized", "cer_normalized", "wer_raw", "cer_raw", "latency_sec"]
        ]
        .mean()
        .round(4)
        .reset_index()
    )
    summary_path = os.path.join(out_dir, "summary.csv")
    summary.to_csv(summary_path, index=False)

    print(f"\nPer-clip results -> {detail_path}")
    print(f"Per-language summary -> {summary_path}\n")
    print(summary.pivot(index="language", columns="model", values="wer_normalized"))

    return detail_df, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", default="results/")
    args = parser.parse_args()
    run_benchmark(args.manifest, args.out)