# Context-Aware Visual Verification System for Misinformation Detection

A production-grade, 5-stage hybrid AI + API backend that determines whether an image + claim text is **AUTHENTIC**, **SUSPICIOUS**, or **MISINFORMATION**.

---

## Architecture Overview

```
POST /verify  ──►  Stage 1: Input Validation
                        │
                        ▼
               Stage 2: Visual Analysis (CLIP + pHash + Bing)
               Stage 3: Fact Check (Google API → NLI fallback)
                        │
                        ▼
               Stage 4: News Context (NewsAPI + Semantic NLI + spaCy NER)
                        │
                        ▼
               Stage 5: Risk Score + Verdict
                        │
                        ▼
               JSON Response  ◄──────────────────────────────────
```

---

## 1. Prerequisites

- Python 3.10 or newer
- Windows / macOS / Linux
- No GPU required (runs on CPU)
- Internet connection (for model auto-download on first run)

---

## 2. Installation

### Step 1 — Clone / set up the project folder

```bash
cd misinfo_detector
```

### Step 2 — Create and activate a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### Step 3 — Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** On the first run, PyTorch and HuggingFace models (~1.5 GB total) will be
> downloaded automatically and cached. Subsequent runs are fast.

### Step 4 — Download the spaCy NLP model

```bash
python -m spacy download en_core_web_sm
```

### Step 5 — Create your `.env` file

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Then open `.env` and fill in any API keys you have.
**All keys are optional** — the system uses local AI models as fallbacks.

---

## 3. Running the Server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API is now available at: **http://localhost:8000**

Interactive docs: **http://localhost:8000/docs**

---

## 4. Example curl Request

```bash
curl -X POST "http://localhost:8000/verify" \
  -F "image=@/path/to/your/image.jpg" \
  -F "claim_text=Scientists confirm that 5G towers caused the COVID-19 pandemic."
```

### Windows (PowerShell)

```powershell
curl.exe -X POST "http://localhost:8000/verify" `
  -F "image=@C:\path\to\image.jpg" `
  -F "claim_text=Scientists confirm that 5G towers caused the COVID-19 pandemic."
```

---

## 5. Example Response

```json
{
  "verdict": "MISINFORMATION",
  "risk_score": 82.5,
  "visual_similarity_score": 15.0,
  "fact_check_score": 90.0,
  "context_match_score": 71.2,
  "contradiction_score": 85.0,
  "matched_sources": [
    "https://factcheck.org/2020/04/5g-covid19-debunked",
    "https://reuters.com/fact-check/5g-towers-covid"
  ],
  "reasoning": "The claim '...' has been assessed as MISINFORMATION with a high risk score of 82.5/100. Multiple signals indicate this content is likely false or manipulated. High-risk signals detected: • Fact Check (30%): score 90.0/100. Google Fact Check: claim rated as 'False' by Reuters. • Contradiction (15%): score 85.0/100. Score breakdown — Visual: 15.0 × 0.30 = 4.5 | Fact-check: 90.0 × 0.30 = 27.0 | Context: 71.2 × 0.25 = 17.8 | Contradiction: 85.0 × 0.15 = 12.8 | Total: 62.1/100."
}
```

---

## 6. Verdict Thresholds

| Risk Score | Verdict |
|---|---|
| ≥ 75 | **MISINFORMATION** |
| 45–74 | **SUSPICIOUS** |
| < 45 | **AUTHENTIC** |

---

## 7. Adding Known Misinformation Images

Edit `data/local_misinfo_dataset.json` and add entries with:

1. A CLIP embedding (512-dim float list) — generate with:
   ```python
   from services.image_analysis import get_clip_embedding
   emb = get_clip_embedding(Path("path/to/image.jpg"))
   print(emb.tolist())  # Paste into JSON
   ```

2. A perceptual hash — generate with:
   ```python
   import imagehash
   from PIL import Image
   phash = str(imagehash.phash(Image.open("path/to/image.jpg")))
   print(phash)
   ```

---

## 8. Project Structure

```
misinfo_detector/
├── main.py                    # FastAPI app + Stage 1
├── config.py                  # Environment config
├── services/
│   ├── image_analysis.py      # Stage 2: Visual analysis
│   ├── fact_check.py          # Stage 3: Fact-check engine
│   ├── context_analysis.py    # Stage 4: News context
│   └── risk_engine.py         # Stage 5: Risk score + verdict
├── utils/
│   ├── similarity.py          # Cosine similarity helper
│   └── preprocessing.py       # Image/text preprocessing
├── data/
│   └── local_misinfo_dataset.json  # Local misinfo image store
├── requirements.txt
├── .env.example
└── README.md
```

---

## 9. Fallback Strategy

| Stage | Primary | Fallback |
|---|---|---|
| Visual | CLIP + pHash + Bing API | CLIP + pHash only |
| Fact Check | Google Fact Check API | `bart-large-mnli` NLI |
| News Context | NewsAPI | MediaStack → Skip (weight adjusted) |
| Overall | Full pipeline | System never crashes; always returns verdict |

---

## 10. Health Check

```bash
curl http://localhost:8000/health
# → {"status": "healthy", "version": "1.0.0"}
```
