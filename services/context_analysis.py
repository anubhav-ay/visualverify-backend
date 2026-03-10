"""
Stage 4 — News Context & Semantic Correlation

Fetches top news articles, computes semantic similarity with the claim,
runs NLI contradiction detection, and extracts named entity overlap.
"""

import requests
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from typing import Any

import httpx
import numpy as np
import spacy

from config import settings
from utils.similarity import cosine_similarity_score

# ── Lazy-loaded models ──────────────────────────────────────────────────────

_nlp = None
_embed_model = None

def _load_models():
    global _nlp, _embed_model

    if _nlp is None:
        _nlp = spacy.load("en_core_web_sm")

    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _nlp, _embed_model

def extract_entities(text: str):
    nlp, _ = _load_models()
    doc = nlp(text)
    return [ent.text for ent in doc.ents]

def fetch_wikipedia_summary(entity: str):
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{entity}"
        response = requests.get(url, timeout=5)

        if response.status_code == 200:
            data = response.json()
            return data.get("extract", "")
    except:
        pass

    return ""

def compute_external_mismatch(claim_text: str, wiki_text: str):
    _, model = _load_models()

    claim_emb = model.encode([claim_text])
    wiki_emb = model.encode([wiki_text])

    similarity = cosine_similarity(claim_emb, wiki_emb)[0][0]

    mismatch_score = (1 - similarity) * 100

    return max(0.0, min(100.0, mismatch_score))

_sentence_model = None
_nlp = None  # spaCy model


def _load_sentence_model():
    """Load sentence-transformers model on first use."""
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer
        print("[INFO] Loading sentence-transformers model...")
        _sentence_model = SentenceTransformer(settings.SENTENCE_MODEL)
        print("[INFO] Sentence model loaded.")
    return _sentence_model


def _load_spacy():
    """Load spaCy NER model, auto-downloading if missing."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("[INFO] Downloading spaCy model en_core_web_sm...")
            import subprocess
            subprocess.run(
                ["python", "-m", "spacy", "download", "en_core_web_sm"],
                check=True, capture_output=True
            )
            _nlp = spacy.load("en_core_web_sm")
        print("[INFO] spaCy model loaded.")
    return _nlp


# ── News API helpers ────────────────────────────────────────────────────────

async def _fetch_newsapi(claim_text: str) -> list[dict]:
    """Fetch top 5 articles from NewsAPI."""
    if not settings.NEWS_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
            resp = await client.get(
                settings.NEWS_API_URL,
                params={
                    "q": claim_text[:100],  # Limit query length
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": 5,
                    "apiKey": settings.NEWS_API_KEY
                }
            )
            resp.raise_for_status()
            data = resp.json()
        articles = data.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "source": a.get("source", {}).get("name", "")
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        print(f"[WARNING] NewsAPI failed: {e}")
        return []


async def _fetch_mediastack(claim_text: str) -> list[dict]:
    """Fetch top 5 articles from MediaStack (fallback)."""
    if not settings.MEDIASTACK_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=settings.HTTP_TIMEOUT) as client:
            resp = await client.get(
                settings.MEDIASTACK_URL,
                params={
                    "access_key": settings.MEDIASTACK_API_KEY,
                    "keywords": claim_text[:100],
                    "languages": "en",
                    "limit": 5,
                    "sort": "relevancy"
                }
            )
            resp.raise_for_status()
            data = resp.json()
        articles = data.get("data", [])
        return [
            {
                "title": a.get("title", ""),
                "description": a.get("description", ""),
                "url": a.get("url", ""),
                "source": a.get("source", "")
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        print(f"[WARNING] MediaStack failed: {e}")
        return []


async def _fetch_articles(claim_text: str) -> list[dict]:
    """Fetch articles from NewsAPI, falling back to MediaStack."""
    articles = await _fetch_newsapi(claim_text)
    if not articles:
        print("[INFO] Falling back to MediaStack.")
        articles = await _fetch_mediastack(claim_text)
    return articles


# ── Semantic scoring ────────────────────────────────────────────────────────

def _compute_semantic_score(claim_text: str, articles: list[dict]) -> float:
    """
    Compute cosine similarity between claim and article texts.

    Returns max similarity score (0-100).
    """
    if not articles:
        return 0.0

    model = _load_sentence_model()
    claim_emb = model.encode(claim_text, normalize_embeddings=True)

    scores = []
    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}".strip()
        if not text:
            continue
        art_emb = model.encode(text, normalize_embeddings=True)
        sim = cosine_similarity_score(claim_emb, art_emb)  # 0-1
        scores.append(sim)

    if not scores:
        return 0.0

    return round(float(np.max(scores)) * 100, 2)


# ── NLI contradiction detection ─────────────────────────────────────────────

_nli_pipeline = None


def _load_nli():
    """Load NLI pipeline (reuse bart-large-mnli)."""
    global _nli_pipeline
    if _nli_pipeline is None:
        from transformers import pipeline
        print("[INFO] Loading NLI model for contradiction detection...")
        _nli_pipeline = pipeline(
            "zero-shot-classification",
            model=settings.NLI_MODEL,
            device=-1
        )
        print("[INFO] NLI contradiction model loaded.")
    return _nli_pipeline


def _compute_contradiction_score(claim_text: str, articles: list[dict]) -> float:
    """
    Run NLI to check if top articles contradict the claim.

    Returns contradiction score (0-100).
    """
    if not articles:
        return 0.0

    nli = _load_nli()
    candidate_labels = ["supports this claim", "contradicts this claim", "unrelated"]

    contradiction_scores = []
    for art in articles[:3]:  # Check top 3 for performance
        premise = f"{art.get('title', '')} {art.get('description', '')}".strip()
        if not premise:
            continue
        try:
            result = nli(
                sequences=premise,
                candidate_labels=candidate_labels,
                hypothesis_template=f"This article {{}} '{claim_text[:200]}'"
            )
            label_scores = dict(zip(result["labels"], result["scores"]))
            contradiction_prob = label_scores.get("contradicts this claim", 0.0)
            contradiction_scores.append(contradiction_prob)
        except Exception as e:
            print(f"[WARNING] NLI contradiction check failed for article: {e}")

    if not contradiction_scores:
        return 0.0

    max_contradiction = float(np.max(contradiction_scores))
    return round(max_contradiction * 100, 2)


# ── Named entity overlap ────────────────────────────────────────────────────

def _compute_entity_overlap(claim_text: str, articles: list[dict]) -> float:
    """
    Extract named entities from claim and articles, compute Jaccard overlap.

    Returns overlap score (0-100).
    """
    if not articles:
        return 0.0

    nlp = _load_spacy()

    # Entities in claim
    claim_doc = nlp(claim_text)
    claim_entities = {ent.text.lower() for ent in claim_doc.ents}

    if not claim_entities:
        return 0.0

    # Entities in all articles combined
    article_text = " ".join(
        f"{a.get('title', '')} {a.get('description', '')}" for a in articles
    )
    art_doc = nlp(article_text[:10000])  # Limit for performance
    article_entities = {ent.text.lower() for ent in art_doc.ents}

    if not article_entities:
        return 0.0

    # Jaccard similarity
    intersection = claim_entities & article_entities
    union = claim_entities | article_entities
    overlap = len(intersection) / len(union) if union else 0.0

    return round(overlap * 100, 2)


# ── Stage 4 main function ───────────────────────────────────────────────────

async def analyze_context(claim_text: str, image_context: str) -> dict[str, Any]:
    """
    Stage 4 main function.

    Fetches news articles and computes:
    - semantic_similarity_score
    - contradiction_score
    - entity_overlap_score
    - context_match_score (combined)

    Returns:
        {
            context_match_score: float (0-100),
            matched_articles: list[dict],
            semantic_score: float,
            contradiction_score: float,
            context_summary: str
        }
    """
    # ── Fetch articles ──────────────────────────────────────────────────────
    articles = await _fetch_articles(claim_text)

    if not articles:
        print("[WARNING] No news articles fetched. Applying external knowledge grounding only.")

        semantic_score = 0.0
        contradiction_score = 0.0
        entity_overlap = 0.0
        context_match_score = 30.0
        summary = (
            "No news articles were available for context analysis. "
            "Default context score applied."
        )

        # 🔥 Image–Claim Entity Consistency Check
        entities_claim = extract_entities(claim_text)
        entities_image = extract_entities(image_context)

        if entities_image and entities_claim:
            image_entities_lower = [e.lower() for e in entities_image]
            claim_text_lower = claim_text.lower()

            if not any(ent in claim_text_lower for ent in image_entities_lower):
                contradiction_score = 60
                context_match_score = 65
                summary += " Image entity does not match claim context."

        return {
            "context_match_score": context_match_score,
            "matched_articles": [],
            "semantic_score": semantic_score,
            "contradiction_score": contradiction_score,
            "context_summary": summary
        }    

    # ── Compute scores ──────────────────────────────────────────────────────
    semantic_score = _compute_semantic_score(claim_text, articles)
    contradiction_score = _compute_contradiction_score(claim_text, articles)
    entity_overlap = _compute_entity_overlap(claim_text, articles)

    # ───────── Image–Claim Entity Consistency Check ─────────

    entities_claim = extract_entities(claim_text)
    entities_image = extract_entities(image_context)

    if entities_image and entities_claim:
        image_entities_lower = [e.lower() for e in entities_image]
        claim_text_lower = claim_text.lower()

        # If image entity not mentioned in claim → possible mismatch
        if not any(ent in claim_text_lower for ent in image_entities_lower):
            contradiction_score = max(contradiction_score, 60)
            context_match_score = max(context_match_score, 65)
            summary += " Image entity does not match claim context."

    # ── Combine into context_match_score ────────────────────────────────────
    # High semantic similarity to contradicting articles → high risk
    # High entity overlap with news → claim is discussing a real story
    # High contradiction → claim differs from news accounts
    context_match_score = round(
        0.40 * contradiction_score +
        0.35 * semantic_score +
        0.25 * entity_overlap,
        2
    )
    context_match_score = min(100.0, max(0.0, context_match_score))

    # ── Build summary ───────────────────────────────────────────────────────
    top_titles = [a["title"] for a in articles[:3]]
    titles_str = "; ".join(t for t in top_titles if t) or "N/A"

    if contradiction_score >= 60:
        summary_lead = "News articles strongly contradict this claim."
    elif semantic_score >= 60:
        summary_lead = "Claim aligns with recent news context, though accuracy is uncertain."
    else:
        summary_lead = "Claim has weak overlap with current news coverage."

    summary = (
        f"{summary_lead} "
        f"Semantic match: {semantic_score:.1f}/100, "
        f"Contradiction: {contradiction_score:.1f}/100, "
        f"Entity overlap: {entity_overlap:.1f}/100. "
        f"Top articles: [{titles_str}]."
    )

        # ───────── External Knowledge Grounding ─────────

    # ───────── External Knowledge Grounding ─────────

    entities = extract_entities(claim_text)

    external_mismatch_score = 0.0

    for entity in entities:
        wiki_text = fetch_wikipedia_summary(entity)

        if wiki_text:
            mismatch = compute_external_mismatch(claim_text, wiki_text)
            external_mismatch_score = max(external_mismatch_score, mismatch)

    if external_mismatch_score > 60:
        context_match_score = max(context_match_score, 65)
        contradiction_score = max(contradiction_score, 60)

        summary += " External knowledge grounding detected factual inconsistency."

    return {
        "context_match_score": context_match_score,
        "matched_articles": articles,
        "semantic_score": semantic_score,
        "contradiction_score": contradiction_score,
        "context_summary": summary
    }
