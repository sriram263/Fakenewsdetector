import re
from urllib.parse import urlparse
from datetime import datetime
import config
from embeddings import get_embedding, cosine_similarity
from normalization import normalize_claim

def get_domain_from_url(url: str) -> str:
    """Extracts base domain from URL."""
    if not url:
        return ""
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""

def score_source_credibility(url: str) -> tuple[float, str]:
    """
    Evaluates domain credibility against DOMAIN_CREDIBILITY_RULES.
    Returns (score: float, category: str)
    """
    domain = get_domain_from_url(url)
    if not domain:
        return config.DOMAIN_CREDIBILITY_RULES["unverified"]["score"], "unverified"

    rules = config.DOMAIN_CREDIBILITY_RULES

    # 1. Check official domains
    for d in rules["official"]["domains"]:
        if domain == d or domain.endswith(f".{d}"):
            return rules["official"]["score"], "official"

    # 2. Check fact checker domains
    for d in rules["fact_checker"]["domains"]:
        if domain == d or domain.endswith(f".{d}"):
            return rules["fact_checker"]["score"], "fact_checker"

    # 3. Check academic domains
    for d in rules["academic"]["domains"]:
        if domain == d or domain.endswith(f".{d}"):
            return rules["academic"]["score"], "academic"

    # 4. Check established news domains
    for d in rules["established_news"]["domains"]:
        if domain == d or domain.endswith(f".{d}"):
            return rules["established_news"]["score"], "established_news"

    # 5. Check unverified / social domains
    for d in rules["unverified"]["domains"]:
        if domain == d or domain.endswith(f".{d}"):
            return rules["unverified"]["score"], "unverified"

    # Default for standard registered web domains
    return rules["general"]["score"], "general"

def score_freshness(published_date: str) -> float:
    """
    Evaluates publication date freshness.
    Returns a score between 0.4 and 1.0.
    """
    if not published_date:
        return 0.6  # Default neutral score for missing dates
    
    try:
        # Try parsing standard ISO or date string formats
        pub_dt = None
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%d %b %Y"):
            try:
                pub_dt = datetime.strptime(published_date[:10], fmt)
                break
            except Exception:
                continue
                
        if not pub_dt:
            return 0.6
            
        now = datetime.now()
        age_days = (now - pub_dt).total_seconds() / 86400.0
        
        if age_days <= 7:
            return 1.0
        elif age_days <= 30:
            return 0.85
        elif age_days <= 90:
            return 0.70
        elif age_days <= 365:
            return 0.55
        else:
            return 0.40
    except Exception:
        return 0.6

def score_relevance(claim_vec: list[float], candidate_text: str) -> float:
    """
    Calculates cosine similarity between claim embedding vector and candidate title/snippet text.
    """
    if not candidate_text:
        return 0.0
    cand_vec = get_embedding(normalize_claim(candidate_text))
    sim = cosine_similarity(claim_vec, cand_vec)
    return max(0.0, min(1.0, float(sim)))

def analyze_cross_source_agreement(candidates: list[dict]) -> tuple[float, bool]:
    """
    Analyzes agreement and potential conflict across candidate sources from independent domains.
    Returns (agreement_score: float, conflict_detected: bool)
    """
    if not candidates:
        return 0.5, False

    domains = set()
    for c in candidates:
        d = get_domain_from_url(c.get("url", ""))
        if d:
            domains.add(d)

    # Check for debunking or conflicting stance keywords
    fake_terms = ["fake", "hoax", "false", "debunk", "misleading", "scam", "rumor", "refutes", "denies"]
    real_terms = ["confirmed", "official", "announces", "approved", "verified", "true", "enacted"]

    fake_count = 0
    real_count = 0

    for c in candidates:
        text = (c.get("title", "") + " " + c.get("content", "")).lower()
        if any(t in text for t in fake_terms):
            fake_count += 1
        if any(t in text for t in real_terms):
            real_count += 1

    conflict_detected = (fake_count >= 1 and real_count >= 1 and len(domains) >= 2)
    
    if len(domains) >= 3 and not conflict_detected:
        agreement_score = 1.0
    elif len(domains) >= 2 and not conflict_detected:
        agreement_score = 0.85
    elif conflict_detected:
        agreement_score = 0.40  # Lower agreement score when sources conflict
    else:
        agreement_score = 0.60

    return agreement_score, conflict_detected

def score_and_rank_candidates(
    candidates: list[dict],
    claim: str,
    max_count: int = config.FINAL_EVIDENCE_COUNT
) -> tuple[list[dict], dict]:
    """
    Computes transparent evidence quality scores for all candidates and selects the top diverse sources.
    Returns: (selected_evidence: list[dict], scoring_summary: dict)
    """
    if not candidates:
        return [], {"conflict_detected": False, "domain_breakdown": {}}

    claim_vec = get_embedding(normalize_claim(claim))
    agreement_score, conflict_detected = analyze_cross_source_agreement(candidates)

    scored_candidates = []
    domain_breakdown = {}

    for c in candidates:
        url = c.get("url", "")
        title = c.get("title", "")
        content = c.get("content", "")
        combined_text = f"{title}. {content}"

        # 1. Relevance Score
        rel_score = score_relevance(claim_vec, combined_text)

        # 2. Source Credibility Score
        cred_score, category = score_source_credibility(url)

        # 3. Freshness Score
        pub_date = c.get("published_date", "")
        fresh_score = score_freshness(pub_date)

        # 4. Final Combined Score
        final_score = (
            (config.WEIGHT_RELEVANCE * rel_score) +
            (config.WEIGHT_CREDIBILITY * cred_score) +
            (config.WEIGHT_FRESHNESS * fresh_score) +
            (config.WEIGHT_AGREEMENT * agreement_score)
        )

        item = dict(c)
        item["relevance_score"] = round(rel_score, 4)
        item["credibility_score"] = round(cred_score, 4)
        item["domain_category"] = category
        item["freshness_score"] = round(fresh_score, 4)
        item["agreement_score"] = round(agreement_score, 4)
        item["final_evidence_score"] = round(final_score, 4)
        item["domain"] = get_domain_from_url(url)

        scored_candidates.append(item)
        domain_breakdown[category] = domain_breakdown.get(category, 0) + 1

    # Sort candidates by final evidence score descending
    scored_candidates.sort(key=lambda x: x["final_evidence_score"], reverse=True)

    # Diversity filtering (max 2 sources per domain)
    selected = []
    domain_counts = {}
    for c in scored_candidates:
        d = c["domain"]
        if domain_counts.get(d, 0) < 2:
            selected.append(c)
            domain_counts[d] = domain_counts.get(d, 0) + 1
        if len(selected) >= max_count:
            break

    summary = {
        "conflict_detected": conflict_detected,
        "domain_breakdown": domain_breakdown,
        "top_score": selected[0]["final_evidence_score"] if selected else 0.0,
        "selected_count": len(selected)
    }

    return selected, summary
