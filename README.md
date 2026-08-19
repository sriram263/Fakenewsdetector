# 🛡️ Veritas AI v2 | Agentic Fake News Detector with Persistent Semantic Memory

> **A Next-Gen Fact-Checking Agent combining Persistent Vector Memory, Multi-Query Retrieval Expansion, and Evidence-Quality Ranking.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-purple)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![OpenRouter](https://img.shields.io/badge/AI-Claude%203%20Haiku-orange)
![Tavily](https://img.shields.io/badge/Search-Tavily%20AI-green)
![Status](https://img.shields.io/badge/Status-Active-success)

Veritas AI v2 upgrades traditional Web-RAG architectures by integrating a persistent local semantic knowledge base and an evidence-quality-aware retrieval engine.

---

## 🚀 What's New in v2

1. **🧠 Feature 1 — Persistent Semantic Knowledge Base (FAISS + Structured Metadata):**
   - Stores completed fact-checks persistently using FAISS vector search (`1536` dimensions, cosine similarity).
   - Reuses verified evidence for semantically similar queries without redundant API search calls.
   - Enforces key entity validation (e.g., distinguishing between ₹2000 and ₹500 currency note claims).
   - Configurable Freshness Policy (3-day TTL for time-sensitive breaking news, 30-day default TTL). Automatically forces fresh web verification for stale entries.

2. **🔍 Feature 2 — Multi-Query + Evidence-Quality-Aware Retrieval Engine:**
   - **Multi-Query Expansion:** Autonomously generates 4 complementary search queries (`direct`, `official`, `verification`, `contradiction`).
   - **Deduplication:** Filters exact URLs, normalized URLs, and near-duplicate content/titles.
   - **Evidence Quality Ranking:** Scores candidates transparently using weighted metrics:
     - Relevance (Embedding Cosine Distance)
     - Domain Credibility Classification (`Official: 1.0`, `Fact-Checker: 0.95`, `Academic: 0.90`, `Established News: 0.85`, `General: 0.60`, `Unverified: 0.40`)
     - Freshness Decay
     - Cross-Source Agreement & Conflict Analysis
   - **Diversity Selection:** Selects the top 5 diverse sources for LLM reasoning.

3. **⚙️ Baseline & Enhanced Compatibility:**
   - Interactive sidebar toggle to run in `enhanced` mode or fall back to `baseline` mode.
   - Expandable **"🔍 Evidence Retrieval & Fact-Check Details"** in the Streamlit UI displaying KB status, queries generated, deduplication metrics, and domain credibility scores.

---

## 🛠️ Project Architecture

```
USER CLAIM
    ↓
1. CLAIM NORMALIZATION & ENTITY EXTRACTION
    ↓
2. SEMANTIC KB SEARCH (FAISS Vector Store)
    ├── FRESH HIT ──→ Instant Evidence Reuse
    └── MISS / STALE
          ↓
3. MULTI-QUERY EXPANSION (4 Targeted Queries)
          ↓
4. TAVILY SEARCH RETRIEVAL (Candidate Pool)
          ↓
5. CANDIDATE DEDUPLICATION (URL & Content)
          ↓
6. EVIDENCE QUALITY SCORING & SELECTION (Relevance + Credibility + Freshness + Agreement)
          ↓
7. CLAUDE 3 HAIKU REASONING & VERDICT (REAL / FAKE / UNCERTAIN)
          ↓
8. STORE COMPLETED RESULT IN PERSISTENT KB
```

---

## 📦 Installation & Setup

1. **Clone the Repo:**
   ```bash
   git clone https://github.com/sriram263/Fakenewsdetector.git
   cd Fakenewsdetector
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Keys (`.env`):**
   ```env
   OPENROUTER_API_KEY=your_openrouter_claude_api_key
   TAVILY_API_KEY=your_tavily_api_key
   RETRIEVAL_MODE=enhanced
   KNOWLEDGE_BASE_ENABLED=true
   ```

4. **Run the Streamlit Application:**
   ```bash
   streamlit run main.py
   ```

5. **Run the Automated Test Suite (10 Validation Scenarios):**
   ```bash
   python test_veritas_v2.py
   ```

---

## 📜 Documentation & Specifications

For full architectural blueprints, metadata schemas, scoring formulas, and evaluation results, see:
- [veritas_ai_v2_architecture.md](file:///c:/My_Learning/FND%20Modified/veritas_ai_v2_architecture.md)

---
*Built with ❤️ by Sriram*