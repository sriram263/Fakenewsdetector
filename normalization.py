import re

STOP_WORDS = {
    "did", "is", "it", "true", "that", "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "done", "can", "could",
    "will", "would", "should", "shall", "may", "might", "must", "question", "news", "headline"
}

def normalize_claim(claim: str) -> str:
    """
    Normalizes user claim for text consistency and comparison.
    - Lowercases text & normalizes currency symbols (₹ -> rupee, $ -> dollar)
    - Strips punctuation & stop words
    - Stems basic plurals and past tenses
    """
    if not claim:
        return ""
    
    text = claim.strip().lower()
    text = text.replace("₹", " rupee ").replace("$", " dollar ").replace("€", " euro ").replace("£", " pound ")
    text = re.sub(r'[^\w\s]', ' ', text)
    
    tokens = text.split()
    cleaned = []
    for w in tokens:
        if w in STOP_WORDS:
            continue
        # Basic suffix stemming
        if len(w) > 4:
            if w.endswith("es"):
                w = w[:-2]
            elif w.endswith("s"):
                w = w[:-1]
            elif w.endswith("ed"):
                w = w[:-2]
            elif w.endswith("ing"):
                w = w[:-3]
        cleaned.append(w)
        
    return " ".join(cleaned)

def extract_key_entities(claim: str) -> dict:
    """
    Extracts key distinct markers from a claim:
    - Numbers (e.g., 2000, 500, 2023, 2026)
    - Currency quantities (e.g., 2000 rupee vs 500 rupee)
    - Years (4-digit numbers starting with 19 or 20)
    """
    norm = claim.lower()
    
    numbers = set(re.findall(r'\b\d+\b', norm))
    years = set(re.findall(r'\b(?:19|20)\d{2}\b', norm))
    currency_denoms = set(re.findall(r'\b(\d+)\s*(?:rupee|rupees|dollar|dollars|note|notes)\b', norm))
    
    return {
        "numbers": numbers,
        "years": years,
        "currency_denoms": currency_denoms
    }

def are_claims_entity_compatible(claim1: str, claim2: str) -> bool:
    """
    Ensures claims with differing key numeric entities (e.g., ₹2000 vs ₹500 or 2023 vs 2026)
    are not falsely matched as identical fact-checks.
    """
    e1 = extract_key_entities(claim1)
    e2 = extract_key_entities(claim2)
    
    if e1["currency_denoms"] and e2["currency_denoms"]:
        if not (e1["currency_denoms"] & e2["currency_denoms"]):
            return False
            
    if e1["years"] and e2["years"]:
        if not (e1["years"] & e2["years"]):
            return False
            
    if e1["numbers"] and e2["numbers"]:
        if not (e1["numbers"] & e2["numbers"]):
            return False
            
    return True
