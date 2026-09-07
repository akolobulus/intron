#!/usr/bin/env python3
import argparse
import csv
import json
import os

from models.sahara_adapter import transcribe

LANGUAGE_CODES = {
    "hausa": "ha",
    "hausa-english": "ha",
    "igbo": "ig",
    "igbo-english": "ig",
    "pidgin": "pcm",
    "pidgin-english": "pcm",
    "yoruba": "yo",
    "yoruba-english": "yo",
}


def run_manifest(manifest_path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    results = []
    for idx, row in enumerate(rows, start=1):
        audio_path = row["audio_path"]
        language = row["language"]
        language_code = LANGUAGE_CODES.get(language, language)
        print(f"[{idx}/{len(rows)}] {language} -> {audio_path}")
        result = transcribe(audio_path=audio_path, language_code=language_code)
        print(f"  result: transcript_len={len(result.get('transcript',''))}, sessions={result.get('num_sessions')}, lat={result.get('latency_sec')}")
        out_path = os.path.join(out_dir, f"{os.path.splitext(os.path.basename(audio_path))[0]}_result.json")
        with open(out_path, "w", encoding="utf-8") as out_f:
            json.dump({"audio_path": audio_path, "language": language, **result}, out_f, ensure_ascii=False, indent=2)
        results.append({"audio_path": audio_path, "language": language, **result})

    summary_path = os.path.join(out_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["audio_path", "language", "transcript", "latency_sec", "num_sessions"])
        writer.writeheader()
        for row in results:
            writer.writerow({
                "audio_path": row["audio_path"],
                "language": row["language"],
                "transcript": row.get("transcript", ""),
                "latency_sec": row.get("latency_sec", ""),
                "num_sessions": row.get("num_sessions", ""),
            })

    print(f"Saved results to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Run a small benchmark manifest against Sahara.")
    parser.add_argument("--manifest", required=True, help="CSV manifest path")
    parser.add_argument("--out", required=True, help="Output directory")
    args = parser.parse_args()
    run_manifest(args.manifest, args.out)


if __name__ == "__main__":
    main()
