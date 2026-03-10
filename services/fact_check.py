"""
Stage 3 — Fact Check Intelligence Engine

Priority order:
  1. Google Fact Check Tools API (if key available)
  2. HuggingFace NLI model (facebook/bart-large-mnli) as fallback
"""

from typing import Any

import httpx

from config import settings

# ── Rating → Score mapping ─────────────────────────────────────────────────
RATING_SCORE_MAP = {
    # Misinformation indicators → high score
    "false": 90,
    "mostly false": 85,
    "pants on fire": 95,
    "misleading": 75,
    "misinformation": 88,
    "inaccurate": 80,
    "debunked": 92,
    "not true": 87,
    "fake": 90,
    # Mixed / unresolved
    "mixed": 60,
    "partly false": 65,
    "half true": 55,
    "unverified": 45,
    "needs context": 50,
    # Authentic indicators → low score
    "true": 10,
    "mostly true": 15,
    "correct": 10,
    "accurate": 10,
    "verified": 8,
    "confirmed": 10,
}

DEFAULT_NO_MATCH_SCORE = 30  # No fact-check record found


def _rating_to_score(rating_text: str) -> int:
    """Convert a textual rating to a numeric misinformation score."""
    if not rating_text:
        return DEFAULT_NO_MATCH_SCORE
    normalized = rating_text.lower().strip()
    for key, score in RATING_SCORE_MAP.items():
        if key in normalized:
            return score
    return DEFAULT_NO_MATCH_SCORE


async def _query_google_fact_check(claim_text: str) -> dict[str, Any] | None:
    """
    Query the Google Fact Check Tools API.

    Returns parsed result dict or None if unavailable/failed.
    """
    if not settings.GOOGLE_FACT_CHECK_KEY:
        return None

    params = {
        "key": settings.GOOGLE_FACT_CHECK_KEY,
        "query": claim_text,
        "languageCode": "en",
        "pageSize": 5
    }

    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
            resp = await client.get(settings.GOOGLE_FACT_CHECK_URL, params=params)
            resp.raise_for_status()
            data = resp.json()

        claims = data.get("claims", [])
        if not claims:
            return None

        # Use the first (most relevant) claim result
        top = claims[0]
        review = top.get("claimReview", [{}])[0]
        rating = review.get("textualRating", "")
        publisher = review.get("publisher", {}).get("name", "Unknown")
        url = review.get("url", "")
        claim_text_found = top.get("text", claim_text)

        score = _rating_to_score(rating)

        return {
            "fact_check_score": score,
            "fact_check_sources": [{"publisher": publisher, "url": url}],
            "fact_check_summary": (
                f"Google Fact Check: '{claim_text_found}' rated as "
                f"'{rating}' by {publisher}. Score: {score}/100."
            )
        }

    except Exception as e:
        print(f"[WARNING] Google Fact Check API failed: {e}")
        return None


# ── NLI fallback model ─────────────────────────────────────────────────────
_nli_pipeline = None


def _load_nli():
    """Lazy-load the NLI pipeline (downloads on first use)."""
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        print("[INFO] Loading NLI model (first run may take a moment)...")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model=settings.NLI_MODEL,
            device=-1  # CPU only
        )
        print("[INFO] NLI model loaded.")
    return _nli_pipeline


def _nli_fact_check(claim_text: str) -> dict[str, Any]:
    """
    Use zero-shot NLI to classify claim plausibility.

    Labels: MISINFORMATION | MISLEADING | AUTHENTIC
    Maps to scores accordingly.
    """
    nli = _load_nli()
    candidate_labels = ["misinformation", "misleading", "authentic information"]

    try:
        result = nli(claim_text, candidate_labels=candidate_labels)
        # result = {'labels': [...], 'scores': [...]}
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        # Map NLI result to fact_check_score
        if "misinformation" in top_label:
            score = int(70 + top_score * 25)  # 70-95
        elif "misleading" in top_label:
            score = int(50 + top_score * 20)  # 50-70
        else:
            score = int(5 + (1 - top_score) * 25)  # 5-30

        score = min(100, max(0, score))

        label_display = top_label.title()
        confidence = round(top_score * 100, 1)

        return {
            "fact_check_score": score,
            "fact_check_sources": [{"publisher": "HuggingFace NLI (bart-large-mnli)", "url": ""}],
            "fact_check_summary": (
                f"Local NLI model classified claim as '{label_display}' "
                f"with {confidence}% confidence. Score: {score}/100."
            )
        }

    except Exception as e:
        print(f"[WARNING] NLI fallback failed: {e}")
        return {
            "fact_check_score": DEFAULT_NO_MATCH_SCORE,
            "fact_check_sources": [],
            "fact_check_summary": "Fact-check NLI model unavailable. Default score applied."
        }


async def run_fact_check(claim_text: str) -> dict[str, Any]:
    """
    Stage 3 main function.

    Tries Google Fact Check API first; falls back to local NLI model.

    Returns:
        {
            fact_check_score: int (0-100),
            fact_check_sources: list[dict],
            fact_check_summary: str
        }
    """
    # ── Attempt Google Fact Check API ─────────────────────────────────────────
    google_result = await _query_google_fact_check(claim_text)
    if google_result is not None:
        return google_result

    # ── Fallback: Local NLI model ─────────────────────────────────────────────
    print("[INFO] Falling back to local NLI model for fact-check.")
    return _nli_fact_check(claim_text)
