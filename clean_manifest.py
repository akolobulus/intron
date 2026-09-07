"""
Filters out manifest rows with placeholder/test reference transcripts
(e.g. AfriSwitchCare's pidgin_10 row: "test clip, SKIP SKIP SKIP").
These are dataset quality issues, not model failures, and inflate WER/CER
into meaningless territory if left in.

Usage:
    python clean_manifest.py
Writes data/manifest_clean.csv, and prints what was dropped so it
can be noted in the benchmark report's methodology section.
"""
import re

import pandas as pd

MANIFEST_PATH = "data/manifest.csv"
CLEAN_PATH = "data/manifest_clean.csv"

PLACEHOLDER_PATTERN = re.compile(
    r"\b(?:skip|test\s+clip|placeholder|n\s*/\s*a)\b", re.IGNORECASE
)


def main():
    manifest = pd.read_csv(MANIFEST_PATH)

    is_placeholder = manifest["reference_text"].fillna("").apply(
        lambda text: bool(PLACEHOLDER_PATTERN.search(text))
    )
    words_per_sec = (
        manifest["reference_text"].fillna("").str.split().apply(len)
        / manifest["duration_sec"].clip(lower=1)
    )
    is_too_sparse = words_per_sec < 0.3

    drop_mask = is_placeholder | is_too_sparse
    dropped = manifest[drop_mask]
    clean = manifest[~drop_mask]

    clean.to_csv(CLEAN_PATH, index=False)

    print(f"Kept {len(clean)} / {len(manifest)} rows -> {CLEAN_PATH}")
    if len(dropped):
        print("\nDropped rows (unscorable references):")
        print(
            dropped[
                ["audio_path", "language", "reference_text", "duration_sec"]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()