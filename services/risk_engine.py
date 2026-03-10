"""
Stage 5 — Risk Score Calculation Engine

Aggregates signals from Stages 2-4 into a final verdict:
  AUTHENTIC | SUSPICIOUS | MISINFORMATION
"""

from typing import Any


def compute_risk_score(
    visual_result: dict[str, Any],
    fact_result: dict[str, Any],
    context_result: dict[str, Any],
    claim_text: str
) -> dict[str, Any]:

    # ── Extract scores ─────────────────────────────────────────
    visual_score = float(visual_result.get("visual_similarity_score", 0.0))
    fact_score = float(fact_result.get("fact_check_score", 0.0))
    context_score = float(context_result.get("context_match_score", 0.0))
    contradiction_score = float(context_result.get("contradiction_score", 0.0))

    alignment_score = float(
        visual_result.get("image_claim_alignment_score", 50.0)
    )

    # Clamp
    visual_score = min(100.0, max(0.0, visual_score))
    fact_score = min(100.0, max(0.0, fact_score))
    context_score = min(100.0, max(0.0, context_score))
    contradiction_score = min(100.0, max(0.0, contradiction_score))
    alignment_score = min(100.0, max(0.0, alignment_score))

    # High alignment = low risk
    alignment_risk = 100.0 - alignment_score

    # Hard misalignment override
    if alignment_risk > 75:
        fact_score = max(fact_score, 60)

    # ── NEW Weighted Formula ───────────────────────────────────
    risk_score = (
        0.30 * alignment_risk +
        0.25 * visual_score +
        0.25 * fact_score +
        0.15 * context_score +
        0.05 * contradiction_score
    )
    risk_score = round(min(100.0, max(0.0, risk_score)), 2)

    # Strong image-claim mismatch override
    if alignment_risk >= 80 and risk_score >= 40:
        verdict = "SUSPICIOUS"
    elif risk_score >= 75:
        verdict = "MISINFORMATION"
    elif risk_score >= 45:
        verdict = "SUSPICIOUS"
    else:
        verdict = "AUTHENTIC"

    # # ── Verdict ────────────────────────────────────────────────
    # if risk_score >= 75:
    #     verdict = "MISINFORMATION"
    # elif risk_score >= 45:
    #     verdict = "SUSPICIOUS"
    # else:
    #     verdict = "AUTHENTIC"

    # ── Build reasoning (keep your old function call) ─────────
    reasoning = _build_reasoning(
        verdict=verdict,
        risk_score=risk_score,
        visual_score=alignment_risk,   # now alignment-based risk
        fact_score=fact_score,
        context_score=context_score,
        contradiction_score=contradiction_score,
        visual_summary=visual_result.get("visual_evidence_summary", ""),
        fact_summary=fact_result.get("fact_check_summary", ""),
        context_summary=context_result.get("context_summary", ""),
        claim_text=claim_text
    )

    # ── Collect matched sources (same as before) ───────────────
    matched_sources: list = []
    matched_sources.extend(visual_result.get("matched_image_sources", []))
    matched_sources.extend([
        s.get("url") or s.get("publisher", "")
        for s in fact_result.get("fact_check_sources", [])
        if isinstance(s, dict)
    ])
    matched_sources.extend([
        a.get("url", "")
        for a in context_result.get("matched_articles", [])
    ])
    matched_sources = list(dict.fromkeys(s for s in matched_sources if s))

    return {
        "verdict": verdict,
        "risk_score": risk_score,
        "image_claim_alignment_score": alignment_score,
        "alignment_risk": alignment_risk,
        "fact_check_score": fact_score,
        "context_match_score": context_score,
        "contradiction_score": contradiction_score,
        "matched_sources": matched_sources,
        "reasoning": reasoning
    }

def _build_reasoning(
    verdict: str,
    risk_score: float,
    visual_score: float,
    fact_score: float,
    context_score: float,
    contradiction_score: float,
    visual_summary: str,
    fact_summary: str,
    context_summary: str,
    claim_text: str
) -> str:
    """
    Generate a human-readable explanation of the verdict.

    Identifies the strongest signals and explains the score.
    """
    # Identify dominant signals
    signal_contributions = {
        "Image-Claim Alignment (30%)": (visual_score, visual_summary),
        "Fact Check (30%)": (fact_score, fact_summary),
        "News Context (25%)": (context_score, context_summary),
        "Contradiction (15%)": (contradiction_score, "")
    }

    high_signals = [
        (name, score, detail)
        for name, (score, detail) in signal_contributions.items()
        if score >= 60
    ]

    low_signals = [
        (name, score, detail)
        for name, (score, detail) in signal_contributions.items()
        if score < 30
    ]

    # ── Verdict intro ──────────────────────────────────────────────────────────
    if verdict == "MISINFORMATION":
        intro = (
            f"The claim '{claim_text[:120]}...' has been assessed as MISINFORMATION "
            f"with a high risk score of {risk_score}/100. "
            "Multiple signals indicate this content is likely false or manipulated."
        )
    elif verdict == "SUSPICIOUS":
        intro = (
            f"The claim '{claim_text[:120]}...' is flagged as SUSPICIOUS "
            f"with a moderate risk score of {risk_score}/100. "
            "Some signals indicate potential manipulation, but evidence is not conclusive."
        )
    else:
        intro = (
            f"The claim '{claim_text[:120]}...' appears AUTHENTIC "
            f"with a low risk score of {risk_score}/100. "
            "No strong indicators of misinformation were found."
        )

    # ── High-risk signal breakdown ────────────────────────────────────────────
    signal_parts = []
    if high_signals:
        signal_parts.append("High-risk signals detected:")
        for name, score, detail in high_signals:
            signal_parts.append(
                f"  • {name}: score {score:.1f}/100. "
                f"{detail[:200] if detail else ''}"
            )

    if low_signals:
        signal_parts.append("Low-risk (authenticating) signals:")
        for name, score, detail in low_signals:
            signal_parts.append(
                f"  • {name}: score {score:.1f}/100. "
                f"{detail[:200] if detail else ''}"
            )

    # ── Score breakdown ────────────────────────────────────────────────────────
    breakdown = (
        f"Score breakdown — "
        f"Visual: {visual_score:.1f} × 0.30 = {visual_score * 0.30:.1f} | "
        f"Fact-check: {fact_score:.1f} × 0.30 = {fact_score * 0.30:.1f} | "
        f"Context: {context_score:.1f} × 0.25 = {context_score * 0.25:.1f} | "
        f"Contradiction: {contradiction_score:.1f} × 0.15 = {contradiction_score * 0.15:.1f} | "
        f"Total: {risk_score:.1f}/100."
    )

    parts = [intro] + (signal_parts if signal_parts else []) + [breakdown]
    return " ".join(parts)
