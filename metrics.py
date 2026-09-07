"""
WER/CER scoring, matching Intron AfriHealth MultiBench's methodology:
report both normalized and unnormalized error rates, since
orthographic variation in low-resource languages penalizes raw scores
unfairly.
"""
import re
import jiwer


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def compute_wer_cer(reference: str, hypothesis: str, normalized: bool) -> dict:
    ref = normalize_text(reference) if normalized else reference
    hyp = normalize_text(hypothesis) if normalized else hypothesis

    if not ref.strip():
        return {"wer": None, "cer": None}

    wer = jiwer.wer(ref, hyp)
    cer = jiwer.cer(ref, hyp)
    return {"wer": round(wer, 4), "cer": round(cer, 4)}


def score_clip(reference: str, hypothesis: str) -> dict:
    """Return both normalized and unnormalized WER/CER for one clip."""
    norm = compute_wer_cer(reference, hypothesis, normalized=True)
    raw = compute_wer_cer(reference, hypothesis, normalized=False)
    return {
        "wer_normalized": norm["wer"],
        "cer_normalized": norm["cer"],
        "wer_raw": raw["wer"],
        "cer_raw": raw["cer"],
    }