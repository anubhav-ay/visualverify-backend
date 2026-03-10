"""
Context-Aware Visual Verification System for Misinformation Detection
Main FastAPI Application - Stage 1: User Input Module
"""

import os
import uuid
import tempfile
import asyncio
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

from config import settings
from services.image_analysis import analyze_image
from services.fact_check import run_fact_check
from services.context_analysis import analyze_context
from services.risk_engine import compute_risk_score

# ─────────────────────────────────────────────
# App Initialization
# ─────────────────────────────────────────────
app = FastAPI(
    title="Misinformation Detection API",
    description="Context-Aware Visual Verification System",
    version="1.0.0"
)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "1.0.0"}


@app.post("/verify")
async def verify(
    image: UploadFile = File(..., description="Image file (jpg/png, max 5MB)"),
    claim_text: str = Form(..., description="Claim text to verify (min 5 chars)")
):
    """
    Stage 1: Validate inputs, then run the full 5-stage pipeline.

    Returns verdict, risk score, and detailed reasoning.
    """
    # ── Validate file extension ──────────────────────────────────────────────
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}"
        )

    # ── Validate MIME type ───────────────────────────────────────────────────
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid MIME type '{image.content_type}'. Allowed: {ALLOWED_TYPES}"
        )

    # ── Validate claim text ──────────────────────────────────────────────────
    claim_text = claim_text.strip()
    if len(claim_text) < 5:
        raise HTTPException(
            status_code=400,
            detail="claim_text must be at least 5 characters long."
        )

    # ── Read and validate file size ──────────────────────────────────────────
    image_bytes = await image.read()
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum size is 5MB, got {len(image_bytes) / 1024 / 1024:.2f}MB."
        )

    # ── Store image temporarily ──────────────────────────────────────────────
    tmp_dir = Path(tempfile.gettempdir()) / "misinfo_uploads"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"{uuid.uuid4().hex}{suffix}"

    try:
        tmp_path.write_bytes(image_bytes)

        # ── Stages 2-5: Run pipeline ─────────────────────────────────────────
        # Run Stage 2 (image analysis) and Stage 3 (fact-check) concurrently
        visual_result, fact_result = await asyncio.gather(
            analyze_image(tmp_path, claim_text),
            run_fact_check(claim_text),
            return_exceptions=True
        )
        # Handle exceptions from concurrent tasks gracefully
        if isinstance(visual_result, Exception):
            print(f"[WARNING] Image analysis failed: {visual_result}")
            visual_result = {
                "visual_similarity_score": 0,
                "matched_image_sources": [],
                "visual_evidence_summary": "Image analysis unavailable.",
                "image_claim_alignment_score": 50.0
            }
        if isinstance(fact_result, Exception):
            print(f"[WARNING] Fact check failed: {fact_result}")
            fact_result = {
                "fact_check_score": 30,
                "fact_check_sources": [],
                "fact_check_summary": "Fact check unavailable."
            }

        # Stage 4: Context analysis (needs claim_text + previous results)
        try:
            context_result = await analyze_context(
                claim_text,
                visual_result.get("visual_evidence_summary", "")
            )

        except Exception as e:
            print(f"[WARNING] Context analysis failed: {e}")
            context_result = {
                "context_match_score": 30,
                "matched_articles": [],
                "semantic_score": 0,
                "contradiction_score": 0,
                "context_summary": "Context analysis unavailable."
            }

        # Stage 5: Risk computation + verdict
        final_result = compute_risk_score(
            visual_result=visual_result,
            fact_result=fact_result,
            context_result=context_result,
            claim_text=claim_text
        )

        return JSONResponse(content=final_result)

    finally:
        # Always clean up temp file
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
