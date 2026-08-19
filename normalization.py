import os
import re
import difflib

STOP_WORDS = {
    "did", "is", "it", "true", "that", "the", "a", "an", "in", "on", "at", "to", "for", "of",
    "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "done", "can", "could",
    "will", "would", "should", "shall", "may", "might", "must", "question", "news", "headline",
    "chief", "minister", "prime", "president", "governor", "leader", "head", "actor", "politician",
    "india", "tamil", "nadu", "us", "usa", "uk", "delhi", "california", "state", "country", "government", "cm", "pm"
}

EVENT_VERB_CATEGORIES = {
    "death": {"died", "death", "killed", "passed", "dead", "assassinated", "fatal", "dying", "murdered", "demise"},
    "appointment": {"sworn", "appointed", "elected", "became", "assuming", "promoted", "won", "inaugurated", "crowned"},
    "resignation": {"resigned", "stepped", "quit", "fired", "sacked", "ousted", "removed", "vacated"},
    "ban": {"banned", "discontinued", "withdrawn", "scrapped", "illegal", "prohibited", "outlawed", "banning"},
    "launch": {"launched", "started", "begun", "unveiled", "released", "introduced"},
    "disaster": {"crashed", "explosion", "erupted", "collapsed", "sank", "fire", "accident", "wreck", "destroyed"},
    "crime": {"arrested", "jailed", "raided", "investigated", "guilty", "convicted", "sentenced", "bribe", "indicted"},
    "financial": {"bankrupt", "acquired", "merged", "bought", "sold", "profit", "loss", "revenue"}
}

def is_fuzzy_word_match(w1: str, w2: str, threshold: float = 0.75) -> bool:
    """Returns True if two words match exactly or fuzzy match with high character similarity."""
    w1_clean, w2_clean = w1.lower().strip(), w2.lower().strip()
    if w1_clean == w2_clean:
        return True
    if abs(len(w1_clean) - len(w2_clean)) > 3:
        return False
    return difflib.SequenceMatcher(None, w1_clean, w2_clean).ratio() >= threshold

def is_stop_word_or_role(w: str) -> bool:
    """Checks if a word is a stop word or a role title, including fuzzy typos like 'cheif' -> 'chief'."""
    w_clean = w.lower().strip()
    if w_clean in STOP_WORDS:
        return True
    for stop_w in STOP_WORDS:
        if is_fuzzy_word_match(w_clean, stop_w, threshold=0.78):
            return True
    return False

def extract_event_category(text: str) -> str:
    """Extracts explicit action/event domain from text (e.g., death, appointment, ban, disaster, crime)."""
    words = set(re.findall(r'\b\w+\b', text.lower()))
    for cat_name, verb_set in EVENT_VERB_CATEGORIES.items():
        if words & verb_set:
            return cat_name
    return "general"

def stem_word(w: str) -> str:
    """Stems common suffixes and double consonants for high-accuracy semantic vector matching."""
    w = w.lower()
    if len(w) <= 3:
        return w
    w = re.sub(r'([a-z])\1ed$', r'\1', w)
    w = re.sub(r'([a-z])\1ing$', r'\1', w)
    if w.endswith("ed"):
        w = w[:-2]
    elif w.endswith("ing"):
        w = w[:-3]
    elif w.endswith("es"):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith("ss"):
        w = w[:-1]
    return w

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
    cleaned = [stem_word(w) for w in tokens if not is_stop_word_or_role(w)]
        
    return " ".join(cleaned)

def extract_key_entities(claim: str) -> dict:
    """
    Extracts key distinct markers from a claim:
    - Numbers (e.g., 2000, 500, 2023, 2026, 40)
    - Currency quantities (e.g., 2000 rupee vs 500 rupee)
    - Years (4-digit numbers starting with 19 or 20)
    - Event category (death vs appointment vs ban vs disaster vs crime)
    """
    norm = claim.lower()
    
    numbers = set(re.findall(r'\b\d+\b', norm))
    years = set(re.findall(r'\b(?:19|20)\d{2}\b', norm))
    currency_denoms = set(re.findall(r'\b(\d+)\s*(?:rupee|rupees|dollar|dollars|note|notes)\b', norm))
    event_category = extract_event_category(claim)
    
    return {
        "numbers": numbers,
        "years": years,
        "currency_denoms": currency_denoms,
        "event_category": event_category
    }

def are_claims_entity_compatible(claim1: str, claim2: str) -> bool:
    """
    Ensures claims with differing numeric entities or distinct event intents (e.g., 'died' vs 'sworn in')
    are generically prevented from falsely matching as identical fact-checks.
    """
    e1 = extract_key_entities(claim1)
    e2 = extract_key_entities(claim2)
    
    cat1, cat2 = e1["event_category"], e2["event_category"]
    if cat1 != "general" and cat2 != "general" and cat1 != cat2:
        return False

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
