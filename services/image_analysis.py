"""
Stage 2 — Contextual Reverse Image Verification

Uses CLIP model embeddings, perceptual hashing, and optionally
Bing Visual Search API to score image authenticity.
"""

import json
import base64
from pathlib import Path
from typing import Any

import numpy as np
import httpx
import imagehash
from PIL import Image

from config import settings
from utils.similarity import cosine_similarity_score

from sklearn.metrics.pairwise import cosine_similarity

def compute_image_claim_alignment(
    image_embedding: np.ndarray,
    claim_text: str,
    clip_model,
    clip_processor
) -> float:
    """
    Compute semantic alignment between image and claim using CLIP.
    Returns 0–100 alignment score.
    """
    import torch

    with torch.no_grad():
        inputs = clip_processor(
            text=[claim_text],
            return_tensors="pt",
            padding=True
        )
        text_features = clip_model.get_text_features(**inputs)
        text_embedding = text_features.squeeze().numpy()

    # Normalize
    text_embedding = text_embedding / (np.linalg.norm(text_embedding) + 1e-9)

    similarity = np.dot(image_embedding, text_embedding)

    # Normalize (-1 to 1) → (0 to 100)
    alignment_score = similarity * 100
    alignment_score = float(max(0.0, min(100.0, alignment_score)))
    return float(max(0.0, min(100.0, alignment_score)))
# ── Lazy model loading (avoids long startup time) ────────────────────────────
_clip_model = None
_clip_processor = None


def _load_clip():
    """Load CLIP model on first use; auto-downloads from HuggingFace."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        print("[INFO] Loading CLIP model (first run may take a moment)...")
        _clip_model = CLIPModel.from_pretrained(settings.CLIP_MODEL)
        _clip_processor = CLIPProcessor.from_pretrained(settings.CLIP_MODEL)
        _clip_model.eval()
        print("[INFO] CLIP model loaded.")
    return _clip_model, _clip_processor


def get_clip_embedding(image_path: Path) -> np.ndarray:
    """
    Generate a 512-dim CLIP image embedding (CPU-safe).

    Args:
        image_path: Path to the image file.

    Returns:
        Normalized numpy array of shape (512,).
    """
    model, processor = _load_clip()
    image = Image.open(image_path).convert("RGB")

    import torch
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        features = model.get_image_features(**inputs)
        embedding = features.squeeze().numpy()

    # L2 normalize
    norm = np.linalg.norm(embedding)
    return embedding / (norm + 1e-9)


def get_phash(image_path: Path) -> imagehash.ImageHash:
    """Compute perceptual hash of the image."""
    image = Image.open(image_path).convert("RGB")
    return imagehash.phash(image)


def load_local_misinfo_embeddings() -> list[dict]:
    """
    Load pre-stored misinformation image embeddings from local dataset.

    Each entry should have:
        - 'embedding': list[float] (512-dim CLIP vector)
        - 'phash': str (hex perceptual hash)
        - 'source': str
        - 'label': str
    """
    dataset_path = Path(settings.MISINFO_DATASET_PATH)
    if not dataset_path.exists():
        return []
    with open(dataset_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_visual_similarity(
    image_embedding: np.ndarray,
    image_phash: imagehash.ImageHash,
    dataset: list[dict]
) -> tuple[float, list[str]]:
    """
    Compare query image against local misinfo dataset.

    Returns:
        (max_similarity_score 0-100, list of matched source labels)
    """
    if not dataset:
        return 0.0, []

    max_score = 0.0
    matched_sources = []

    for entry in dataset:
        # ── CLIP cosine similarity ────────────────────────────────────────────
        stored_emb = np.array(entry.get("embedding", []), dtype=np.float32)
        if stored_emb.size == 0:
            continue

        cos_sim = cosine_similarity_score(image_embedding, stored_emb)  # 0-1

        # ── Perceptual hash similarity ────────────────────────────────────────
        stored_phash_hex = entry.get("phash", "")
        hash_sim = 0.0
        if stored_phash_hex:
            try:
                stored_phash = imagehash.hex_to_hash(stored_phash_hex)
                hash_diff = image_phash - stored_phash  # Hamming distance (0-64)
                hash_sim = max(0.0, 1.0 - hash_diff / 64.0)
            except Exception:
                pass

        # Combined similarity (weighted: 70% CLIP, 30% hash)
        combined = 0.7 * cos_sim + 0.3 * hash_sim
        score = combined * 100

        if score > max_score:
            max_score = score

        if score > 50:
            matched_sources.append(entry.get("source", "unknown"))

    return round(max_score, 2), matched_sources


async def query_bing_visual_search(image_path: Path) -> list[str]:
    """
    Optional: Query Bing Visual Search API for reverse image results.

    Returns list of source URLs from suspicious/fact-check domains.
    Skipped gracefully if API key is not configured.
    """
    if not settings.BING_VISUAL_SEARCH_KEY:
        return []

    suspicious_domains = [
        "snopes.com", "factcheck.org", "politifact.com",
        "reuters.com/fact-check", "apnews.com/hub/ap-fact-check"
    ]

    try:
        image_bytes = image_path.read_bytes()
        b64 = base64.b64encode(image_bytes).decode()

        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
            response = await client.post(
                settings.BING_VISUAL_SEARCH_URL,
                headers={
                    "Ocp-Apim-Subscription-Key": settings.BING_VISUAL_SEARCH_KEY,
                    "Content-Type": "application/json"
                },
                json={"imageInfo": {"imageInsightsToken": b64}}
            )
            response.raise_for_status()
            data = response.json()

        found_urls = []
        tags = data.get("tags", [])
        for tag in tags:
            for action in tag.get("actions", []):
                if action.get("actionType") == "PagesIncluding":
                    for item in action.get("data", {}).get("value", []):
                        url = item.get("contentUrl", "")
                        if any(d in url for d in suspicious_domains):
                            found_urls.append(url)

        return found_urls

    except Exception as e:
        print(f"[WARNING] Bing Visual Search failed: {e}")
        return []


async def analyze_image(image_path: Path, claim_text: str) -> dict[str, Any]:
    """
    Stage 2 main function.

    Runs CLIP embedding, perceptual hash comparison,
    and optional Bing reverse image search.

    Returns:
        {
            visual_similarity_score: float (0-100),
            matched_image_sources: list[str],
            visual_evidence_summary: str
        }
    """
    try:
        # ── Step 1: Compute CLIP embedding ────────────────────────────────────
        embedding = get_clip_embedding(image_path)
        # ── Step 1.5: Compute image-claim alignment ──────────────────────────
        model, processor = _load_clip()

        alignment_score = compute_image_claim_alignment(
            embedding,
            claim_text,
            model,
            processor
        )
        # ── Step 2: Compute perceptual hash ───────────────────────────────────
        phash = get_phash(image_path)

        # ── Step 3: Compare against local dataset ────────────────────────────
        dataset = load_local_misinfo_embeddings()
        local_score, local_sources = compute_visual_similarity(embedding, phash, dataset)

        # ── Step 4: Optional Bing Visual Search ───────────────────────────────
        bing_sources = await query_bing_visual_search(image_path)

        # Boost score if found on suspicious/fact-check sites
        bing_boost = min(20.0, len(bing_sources) * 8.0)
        final_score = min(100.0, local_score + bing_boost)

        all_sources = list(set(local_sources + bing_sources))

        # ── Build evidence summary ─────────────────────────────────────────────
        if final_score >= 70:
            summary = (
                f"Image closely resembles known misinformation samples "
                f"(score: {final_score:.1f}/100). "
                f"Matched sources: {', '.join(all_sources[:3]) or 'local dataset'}."
            )
        elif final_score >= 40:
            summary = (
                f"Image shows moderate visual similarity to flagged content "
                f"(score: {final_score:.1f}/100). Partial match detected."
            )
        else:
            summary = (
                f"Image does not strongly match known misinformation patterns "
                f"(score: {final_score:.1f}/100)."
            )

        return {
            "visual_similarity_score": round(final_score, 2),
            "matched_image_sources": all_sources,
            "visual_evidence_summary": summary,
            "image_claim_alignment_score": round(alignment_score, 2)
        }
    except Exception as e:
        print(f"[ERROR] Image analysis pipeline failed: {e}")
        return {
            "visual_similarity_score": 0.0,
            "matched_image_sources": [],
            "visual_evidence_summary": f"Image analysis error: {str(e)}",
            "image_claim_alignment_score": 50.0
        }