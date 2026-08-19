# VERITAS AI v2 — ARCHITECTURAL & TECHNICAL SPECIFICATION REPORT

## 1. Existing Baseline Architecture vs. New Architecture

### Baseline Architecture
```
User Claim → Tavily Single Query Search → Raw Search Results (5 items) → OpenRouter Claude 3 Haiku → Verdict (REAL/FAKE/UNCERTAIN)
```

### New Architecture (Veritas AI v2)
```
USER CLAIM
    ↓
1. CLAIM NORMALIZATION & KEY ENTITY EXTRACTION
    ↓
2. SEMANTIC KNOWLEDGE BASE SEARCH (FAISS Vector Store)
    │
    ├── FRESH KB HIT (Sim ≥ 0.85 & Entity Match & Fresh)
    │     └─→ Reuse verified evidence & verdict instantly (0 web calls)
    │
    └── MISS or STALE KB HIT
          ↓
3. MULTI-QUERY RETRIEVAL EXPANSION (4 Complementary Queries)
    │   • Direct query
    │   • Official authority query
    │   • Independent news verification query
    │   • Contradiction & debunk check query
    ↓
4. TAVILY RETRIEVAL (Collect candidate pool across queries)
    ↓
5. URL & CONTENT DEDUPLICATION (Exact URL + Normalized URL + Jaccard Title/Snippet similarity)
    ↓
6. EVIDENCE QUALITY SCORING & SELECTION
    │   • Relevance Score (Embedding Cosine Similarity)
    │   • Source Credibility Score (Domain Classification Rules: Official, Fact-Checker, News, Academic, General, Unverified)
    │   • Freshness Score (Publication Date Decay)
    │   • Cross-Source Agreement Score (Consensus vs Conflict Detection)
    │   • Combined Weighted Evidence Score → Select Top 5 Diverse Sources
    ↓
7. CLAUDE 3 HAIKU REASONING & VERDICT GENERATION
    ↓
8. PERSIST KNOWLEDGE BASE RESULT (FAISS Index + JSON Metadata disk persistence)
```

---

## 2. Knowledge Base Concept & Semantic Retrieval

The Knowledge Base is a persistent vector-indexed memory layer for completed fact-checks. It prevents redundant web searching and provides reproducible, fast evidence retrieval.

### Why Semantic Retrieval?
Traditional exact keyword matching fails for paraphrased claims:
- *Original Claim:* "Did India ban the 2000 rupee note?"
- *Paraphrased Claim:* "Is it true that India banned ₹2000 currency notes?"

Semantic embeddings represent claims in a continuous vector space where phrasing variations share high cosine similarity.

### Entity Mismatch Protection
To prevent false-positive semantic matches between distinct claims with identical sentence structures:
- *Claim A:* "India discontinued ₹2000 notes in 2023."
- *Claim B:* "India discontinued ₹500 notes in 2026."

`normalization.py` extracts key numeric entities (currency denominations like 2000 vs 500, years like 2023 vs 2026). If key numeric entities conflict, the match is rejected regardless of high vector embedding similarity.

---

## 3. Vector DB Technical Details

- **Embedding Model:** OpenRouter `text-embedding-3-small` (with local deterministic fallback).
- **Vector Dimensions:** 1536 float32 dimensions.
- **Vector Index:** FAISS `IndexFlatIP` (Inner Product on L2-normalized vectors for exact Cosine Similarity).
- **Persistence Mechanism:** `kb_index.faiss` (FAISS binary index) + `kb_metadata.json` (Structured JSON store) in `./kb_data/`.
- **Similarity Threshold:** `0.85` (configurable via `SIMILARITY_THRESHOLD`).

### Metadata Schema
```json
{
  "id": "uuid-v4-string",
  "original_claim": "User prompt text",
  "normalized_claim": "normalized text",
  "verdict": "REAL | FAKE | UNCERTAIN",
  "confidence": 95,
  "explanation": "Summary text",
  "evidence_snippets": ["snippet 1", "snippet 2"],
  "source_urls": ["https://..."],
  "source_titles": ["title 1"],
  "source_domains": ["pib.gov.in", "reuters.com"],
  "source_credibility_info": [{"domain": "pib.gov.in", "score": 0.95, "category": "fact_checker"}],
  "publication_dates": ["2026-01-01"],
  "verification_timestamp": "2026-08-15T23:40:00",
  "retrieval_timestamp": "2026-08-15T23:40:00",
  "search_queries_used": ["query 1", "query 2"],
  "evidence_selected": [...],
  "metadata": {"is_time_sensitive": false}
}
```

---

## 4. Freshness & Staleness Policy

Fact-checks must not be blindly reused forever. Every entry records `verification_timestamp`.
- **Default TTL:** 30 days (`DEFAULT_FRESHNESS_DAYS = 30`).
- **Time-Sensitive TTL:** 3 days (`TIME_SENSITIVE_FRESHNESS_DAYS = 3`). Automatically triggered if the claim contains keywords like "today", "now", "2026", "breaking", "latest".
- **Stale Entry Behavior:** If similarity ≥ 0.85 but age > TTL, the system tags the match as `STALE_HIT`, executes fresh web retrieval, re-verifies with Claude, and updates the stored KB entry.

---

## 5. Multi-Query Retrieval Expansion

For claims requiring web retrieval, `query_generator.py` expands the input claim into complementary queries:

| Query Category | Targeted Purpose | Example Query |
| :--- | :--- | :--- |
| `direct` | Keyword search | *"India 2000 rupee notes ban"* |
| `official` | Official authorities | *"RBI 2000 rupee notes withdrawal official statement"* |
| `verification` | Independent media | *"India 2000 rupee note withdrawal Reuters BBC"* |
| `contradiction` | Debunk / legal status | *"India 2000 rupee notes legal tender hoax rumor"* |

---

## 6. Evidence Quality Scoring & Selection Formulas

Candidates from all queries are pooled (15–20 candidates), deduplicated by exact URL, normalized URL, and title Jaccard similarity, and then scored:

$$\text{Final Score} = (w_{\text{rel}} \cdot S_{\text{rel}}) + (w_{\text{cred}} \cdot S_{\text{cred}}) + (w_{\text{fresh}} \cdot S_{\text{fresh}}) + (w_{\text{agree}} \cdot S_{\text{agree}})$$

- **Default Weights:**
  - $w_{\text{rel}} = 0.40$ (Relevance via Embedding Cosine Similarity)
  - $w_{\text{cred}} = 0.35$ (Domain Credibility Classification)
  - $w_{\text{fresh}} = 0.15$ (Publication Date Time Decay)
  - $w_{\text{agree}} = 0.10$ (Cross-Source Agreement / Conflict Detection)

### Domain Credibility Categories
- **Official (`1.0`):** `.gov`, `.gov.in`, `nic.in`, `rbi.org.in`, `who.int`, `un.org`
- **Fact Checker (`0.95`):** `altnews.in`, `boomlive.in`, `pib.gov.in`, `snopes.com`, `factly.in`, `politifact.com`
- **Academic (`0.90`):** `.edu`, `.ac.in`, `nature.com`, `sciencedirect.com`
- **Established News (`0.85`):** `reuters.com`, `bbc.com`, `apnews.com`, `ndtv.com`, `thehindu.com`, `indianexpress.com`
- **General (`0.60`):** Standard web domains
- **Unverified (`0.40`):** Social media platforms, personal blogspots

### Evidence Selection
Top 5 highest scoring items are selected with **domain diversity filtering** (max 2 items per unique domain).

---

## 7. Baseline Compatibility & Mode Switcher

The application maintains 100% baseline compatibility via configuration or live UI toggles in the Streamlit sidebar:
- `RETRIEVAL_MODE`: `"enhanced"` (default) or `"baseline"`
- `KNOWLEDGE_BASE_ENABLED`: `True` (default) or `False`

---

## 8. How to Run & Test

### Running the Web Application
```powershell
.\venv\Scripts\python.exe -m streamlit run main.py
```

### Running the Automated Test Suite (10 Validation Scenarios)
```powershell
.\venv\Scripts\python.exe test_veritas_v2.py
```

---

## 9. Known Limitations & Future Work

1. **Date Extraction:** Web articles lacking explicit structured `published_date` meta tags fallback to neutral freshness scores (0.60).
2. **Domain Classification:** Unlisted niche domains fallback to `general` credibility (0.60); custom domain rules can be added to `config.py`.
