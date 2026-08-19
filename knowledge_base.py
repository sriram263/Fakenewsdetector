import os
import json
import uuid
import shutil
from datetime import datetime
import numpy as np
import faiss

import config
from normalization import normalize_claim, are_claims_entity_compatible
from embeddings import get_embedding, cosine_similarity

KB_VERSION = "v2.1"

class FactCheckKB:
    """
    Persistent Semantic Fact-Checking Knowledge Base (v2.1) using FAISS + JSON metadata store.
    Includes verification version checking to safeguard against obsolete verdicts.
    """
    def __init__(self, kb_dir=config.KB_DIR):
        self.kb_dir = kb_dir
        self.index_file = os.path.join(kb_dir, "kb_index.faiss")
        self.metadata_file = os.path.join(kb_dir, "kb_metadata.json")
        self.dim = config.EMBEDDING_DIM
        
        # Ensure KB directory exists
        os.makedirs(self.kb_dir, exist_ok=True)
        
        # Initialize or load FAISS index and metadata
        self.metadata_store = []
        self.index = None
        self._load_kb()

    def _load_kb(self):
        """Loads FAISS index and JSON metadata file if they exist, else creates new."""
        if os.path.exists(self.index_file) and os.path.exists(self.metadata_file):
            try:
                self.index = faiss.read_index(self.index_file)
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata_store = json.load(f)
                print(f"[KnowledgeBase] Loaded {len(self.metadata_store)} records from {self.kb_dir}")
                return
            except Exception as e:
                print(f"[KnowledgeBase] Error loading persistent KB: {e}. Reinitializing...")
        
        # Re-initialize empty index
        self.index = faiss.IndexFlatIP(self.dim)
        self.metadata_store = []

    def _save_kb(self):
        """Saves FAISS index and metadata to disk for persistence across restarts."""
        try:
            faiss.write_index(self.index, self.index_file)
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata_store, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[KnowledgeBase] Save error: {e}")

    def clear_kb(self):
        """Clears all stored entries and resets the index."""
        self.metadata_store = []
        self.index = faiss.IndexFlatIP(self.dim)
        try:
            if os.path.exists(self.kb_dir):
                shutil.rmtree(self.kb_dir)
            os.makedirs(self.kb_dir, exist_ok=True)
            self._save_kb()
            print("[KnowledgeBase] Reset and cleared Knowledge Base storage.")
        except Exception as e:
            print(f"[KnowledgeBase] Clear error: {e}")

    def is_time_sensitive(self, claim: str) -> bool:
        """Determines if a claim is time-sensitive."""
        time_keywords = [
            "today", "yesterday", "now", "this week", "this month",
            "breaking", "latest", "2026", "current", "recently"
        ]
        norm = claim.lower()
        return any(kw in norm for kw in time_keywords)

    def is_entry_fresh(self, entry: dict, claim: str) -> tuple[bool, float]:
        """
        Evaluates whether a stored KB entry is fresh based on timestamp, claim context, AND verification version.
        Returns (is_fresh: bool, age_days: float)
        """
        # Version check: Entries created before v2.1 must be re-verified
        if entry.get("verification_version") != KB_VERSION:
            return False, 999.0

        verif_str = entry.get("verification_timestamp")
        if not verif_str:
            return False, 999.0
        
        try:
            verif_time = datetime.fromisoformat(verif_str)
            now = datetime.now()
            age_days = (now - verif_time).total_seconds() / 86400.0
            
            if self.is_time_sensitive(claim) or entry.get("metadata", {}).get("is_time_sensitive", False):
                ttl_days = config.TIME_SENSITIVE_FRESHNESS_DAYS
            else:
                ttl_days = config.DEFAULT_FRESHNESS_DAYS
                
            is_fresh = age_days <= ttl_days
            return is_fresh, round(age_days, 2)
        except Exception as e:
            return False, 999.0

    def search_similar_fact_check(self, claim: str, threshold=config.SIMILARITY_THRESHOLD) -> dict:
        """
        Searches the Knowledge Base for a semantically similar previous fact check.
        """
        if not self.metadata_store or self.index is None or self.index.ntotal == 0:
            return {
                "hit": False,
                "entry": None,
                "similarity_score": 0.0,
                "is_fresh": False,
                "age_days": 0.0,
                "status": "MISS"
            }

        query_vec = get_embedding(normalize_claim(claim))
        vec_np = np.array([query_vec], dtype=np.float32)

        k = min(5, self.index.ntotal)
        distances, indices = self.index.search(vec_np, k)

        best_entry = None
        best_score = 0.0

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_store):
                continue
            
            sim_score = float(dist)
            candidate_entry = self.metadata_store[idx]
            cand_claim = candidate_entry.get("original_claim", "")

            if sim_score >= threshold:
                if are_claims_entity_compatible(claim, cand_claim):
                    if sim_score > best_score:
                        best_score = sim_score
                        best_entry = candidate_entry

        if best_entry is not None:
            is_fresh, age_days = self.is_entry_fresh(best_entry, claim)
            status = "FRESH_HIT" if is_fresh else "STALE_HIT"
            return {
                "hit": True,
                "entry": best_entry,
                "similarity_score": round(best_score, 4),
                "is_fresh": is_fresh,
                "age_days": age_days,
                "status": status
            }

        return {
            "hit": False,
            "entry": None,
            "similarity_score": round(float(distances[0][0]) if len(distances[0]) > 0 else 0.0, 4),
            "is_fresh": False,
            "age_days": 0.0,
            "status": "MISS"
        }

    def store_fact_check(
        self,
        original_claim: str,
        verdict: str,
        confidence: int,
        explanation: str,
        evidence_snippets: list[str],
        source_urls: list[str],
        source_titles: list[str],
        source_domains: list[str],
        source_credibility_info: list[dict],
        publication_dates: list[str],
        search_queries_used: list[str],
        evidence_selected: list[dict],
        metadata: dict = None
    ) -> dict:
        """
        Stores a completed fact-check result with verification version v2.1 into the Knowledge Base.
        """
        norm_claim = normalize_claim(original_claim)
        vec = get_embedding(norm_claim)
        
        entry_id = str(uuid.uuid4())
        now_iso = datetime.now().isoformat()
        
        record = {
            "id": entry_id,
            "verification_version": KB_VERSION,
            "original_claim": original_claim,
            "normalized_claim": norm_claim,
            "verdict": verdict,
            "confidence": confidence,
            "explanation": explanation,
            "evidence_snippets": evidence_snippets or [],
            "source_urls": source_urls or [],
            "source_titles": source_titles or [],
            "source_domains": source_domains or [],
            "source_credibility_info": source_credibility_info or [],
            "publication_dates": publication_dates or [],
            "verification_timestamp": now_iso,
            "retrieval_timestamp": now_iso,
            "search_queries_used": search_queries_used or [],
            "evidence_selected": evidence_selected or [],
            "metadata": metadata or {"is_time_sensitive": self.is_time_sensitive(original_claim)}
        }
        
        vec_np = np.array([vec], dtype=np.float32)
        self.index.add(vec_np)
        self.metadata_store.append(record)
        self._save_kb()
        return record
