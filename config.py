import os
from dotenv import load_dotenv

load_dotenv()

# System Modes & Toggles
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "enhanced")  # "baseline" or "enhanced"
KNOWLEDGE_BASE_ENABLED = os.getenv("KNOWLEDGE_BASE_ENABLED", "true").lower() == "true"

# Knowledge Base Configurations
KB_DIR = os.getenv("KB_DIR", "./kb_data")
KB_INDEX_FILE = os.path.join(KB_DIR, "kb_index.faiss")
KB_METADATA_FILE = os.path.join(KB_DIR, "kb_metadata.json")
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.68"))
DEFAULT_FRESHNESS_DAYS = int(os.getenv("DEFAULT_FRESHNESS_DAYS", "30"))
TIME_SENSITIVE_FRESHNESS_DAYS = int(os.getenv("TIME_SENSITIVE_FRESHNESS_DAYS", "3"))

# Embedding Configurations
EMBEDDING_MODEL = "stable-sha256-ngram"
EMBEDDING_DIM = 1536

# Multi-Query Retrieval Configurations
NUM_QUERIES = int(os.getenv("NUM_QUERIES", "3"))
TAVILY_MAX_RESULTS_PER_QUERY = 5

# Evidence Scoring Weights
WEIGHT_RELEVANCE = float(os.getenv("WEIGHT_RELEVANCE", "0.40"))
WEIGHT_CREDIBILITY = float(os.getenv("WEIGHT_CREDIBILITY", "0.35"))
WEIGHT_FRESHNESS = float(os.getenv("WEIGHT_FRESHNESS", "0.15"))
WEIGHT_AGREEMENT = float(os.getenv("WEIGHT_AGREEMENT", "0.10"))

# Final Evidence Selection Count
FINAL_EVIDENCE_COUNT = int(os.getenv("FINAL_EVIDENCE_COUNT", "5"))

# Domain Credibility Rules
DOMAIN_CREDIBILITY_RULES = {
    "official": {
        "score": 1.0,
        "domains": [
            "gov", "gov.in", "nic.in", "rbi.org.in", "who.int", "un.org",
            "cdc.gov", "fda.gov", "nasa.gov", "sec.gov", "whitehouse.gov"
        ]
    },
    "fact_checker": {
        "score": 0.95,
        "domains": [
            "altnews.in", "boomlive.in", "pib.gov.in", "snopes.com",
            "factly.in", "politifact.com", "fullfact.org", "checkyourfact.com",
            "factcheck.org", "afp.com", "apnews.com/hub/ap-fact-check"
        ]
    },
    "established_news": {
        "score": 0.85,
        "domains": [
            "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "ndtv.com",
            "thehindu.com", "indianexpress.com", "bloomberg.com", "nytimes.com",
            "wsj.com", "theguardian.com", "aljazeera.com", "timesofindia.indiatimes.com",
            "hindustantimes.com", "business-standard.com", "cnbc.com", "ft.com"
        ]
    },
    "academic": {
        "score": 0.90,
        "domains": [
            "edu", "ac.in", "nature.com", "sciencedirect.com", "arxiv.org",
            "springer.com", "ieee.org", "pnas.org", "cell.com"
        ]
    },
    "general": {
        "score": 0.60,
        "domains": []  # Default for normal registered news / blogs
    },
    "unverified": {
        "score": 0.40,
        "domains": [
            "facebook.com", "twitter.com", "x.com", "instagram.com",
            "tiktok.com", "reddit.com", "blogspot.com", "wordpress.com"
        ]
    }
}
