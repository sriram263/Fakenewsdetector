import os
import json
import re
import config
from llm_client import generate_chat_completion

STOP_WORDS = {
    "the", "is", "was", "are", "of", "in", "for", "and", "a", "an", "to", "by", "on", "at",
    "chief", "minister", "prime", "president", "governor", "leader", "head", "actor", "politician",
    "india", "tamil", "nadu", "us", "usa", "uk", "delhi", "california", "state", "country", "government", "cm", "pm"
}

KNOWN_LEADERS = [
    "joe biden", "donald trump", "narendra modi", "m.k. stalin", "mk stalin", "pinarayi vijayan",
    "eknath shinde", "devendra fadnavis", "siddaramaiah", "mamata banerjee", "yogi adityanath",
    "rishi sunak", "keir starmer", "emmanuel macron", "vladimir putin", "xi jinping", "justin trudeau",
    "sundar pichai", "satya nadella", "tim cook", "elon musk", "jeff bezos"
]

def analyze_claim_type(claim: str) -> dict:
    """
    Classifies the user claim into a generic factual claim category and extracts claimed target entities.
    """
    claim_lower = claim.lower()
    
    role_patterns = [
        r'\b(?:is|was|currently|serves as|holds the position of)\s+(?:the\s+)?(?:prime minister|president|ceo|cto|cfo|chairman|head|leader|chancellor|governor|king|queen|minister|cm|chief minister)\b',
        r'\b(?:prime minister|president|ceo|cto|cfo|chairman|head|leader|chancellor|governor|cm|chief minister)\s+of\b'
    ]
    
    words = [w for w in re.findall(r'\b[a-zA-Z]{3,}\b', claim_lower) if w not in STOP_WORDS]
    claimed_person = " ".join(words[:2]) if words else ""
    
    years = re.findall(r'\b(?:19|20)\d{2}\b', claim_lower)
    claimed_year = years[0] if years else ""

    for pat in role_patterns:
        if re.search(pat, claim_lower):
            return {
                "type": "CURRENT_ROLE", 
                "description": "Current office-holder or identity claim",
                "claimed_entity": claimed_person,
                "claimed_year": claimed_year
            }

    if years or re.search(r'\b(?:launched|completed|occurred|happened|started)\b', claim_lower):
        return {
            "type": "TEMPORAL", 
            "description": "Temporal or event date/status claim", 
            "claimed_entity": claimed_person,
            "claimed_year": claimed_year
        }

    if re.search(r'\b\d+(?:\.\d+)?\s*(?:million|billion|trillion|percent|%|users|dollars|rupees)\b', claim_lower):
        return {"type": "NUMERICAL", "description": "Quantitative or numerical claim", "claimed_entity": claimed_person, "claimed_year": claimed_year}

    if re.search(r'\b(?:banned|approved|passed|discovered|invented|died|won|lost|signed)\b', claim_lower):
        return {"type": "EVENT_OCCURRENCE", "description": "Event occurrence claim", "claimed_entity": claimed_person, "claimed_year": claimed_year}

    return {"type": "GENERAL_FACT", "description": "General factual claim", "claimed_entity": claimed_person, "claimed_year": claimed_year}

def evaluate_evidence_relationships(claim: str, claim_info: dict, selected_evidence: list[dict]) -> list[dict]:
    """
    Evaluates the relationship between the claim and each retrieved evidence item.
    Categorizes each item's stance into: SUPPORTS, REFUTES, or INSUFFICIENT.
    """
    if not selected_evidence:
        return []

    evidence_payload = [
        {
            "id": idx,
            "title": e.get("title", ""),
            "domain": e.get("domain", ""),
            "content": e.get("content", ""),
            "credibility_category": e.get("domain_category", "general")
        }
        for idx, e in enumerate(selected_evidence, 1)
    ]

    prompt = f"""
    You are an expert fact-checking stance evaluator.
    Analyze the exact relationship between the USER CLAIM and each EVIDENCE ITEM below.

    USER CLAIM: "{claim}"
    CLAIM CATEGORY: {claim_info.get('type')} ({claim_info.get('description')})

    EVIDENCE ITEMS:
    {json.dumps(evidence_payload, indent=2)}

    RULES FOR STANCE EVALUATION:
    1. SUPPORTS: Evidence explicitly confirms the specific claim.
    2. REFUTES: Evidence directly contradicts the claim.
       - FOR CURRENT_ROLE / IDENTITY (e.g. "X is the PM/CM/CEO of Y"): If reliable evidence identifies a DIFFERENT person/entity holding that exclusive role (e.g. "Person B is the PM/CM of Y"), the claim that "X is the PM/CM of Y" is REFUTED.
       - FOR TEMPORAL / EVENT DATE (e.g. "Launched in 2026"): If evidence shows the event is only PLANNED/SCHEDULED for a future date (e.g. 2027) or occurred in a different year, the claim is REFUTED.
       - FOR DEBUNKS / HOAXES: If evidence reports official denial or debunking, the claim is REFUTED.
    3. INSUFFICIENT: Evidence is topically related (mentions the same entity or general topic) but DOES NOT confirm or contradict the specific claim attribute.

    Output ONLY a valid JSON array of objects with keys:
    [
        {{
            "id": 1,
            "stance": "SUPPORTS" | "REFUTES" | "INSUFFICIENT",
            "reasoning": "Concise 1-sentence justification naming actual office-holder or fact without quotes"
        }}
    ]
    """

    try:
        raw_text, used_provider = generate_chat_completion(
            messages=[
                {"role": "system", "content": "You are a JSON-only API. Respond strictly with a valid JSON array without markdown or extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        if "</think>" in raw_text:
            raw_text = raw_text.split("</think>")[-1].strip()

        start_idx = raw_text.find("[")
        if start_idx != -1:
            evals, _ = json.JSONDecoder().raw_decode(raw_text[start_idx:])
            eval_map = {item.get("id"): item for item in evals}
            annotated_evidence = []
            for idx, e in enumerate(selected_evidence, 1):
                e_copy = dict(e)
                eval_info = eval_map.get(idx, {"stance": "INSUFFICIENT", "reasoning": "Unevaluated"})
                e_copy["stance"] = eval_info.get("stance", "INSUFFICIENT")
                e_copy["stance_reasoning"] = eval_info.get("reasoning", "")
                annotated_evidence.append(e_copy)
                
            return annotated_evidence

    except Exception as ex:
        print(f"[Verifier] Stance evaluation error: {ex}")
        
    return _rule_based_fallback_eval(claim, claim_info, selected_evidence)

def _rule_based_fallback_eval(claim: str, claim_info: dict, selected_evidence: list[dict]) -> list[dict]:
    """Fallback stance evaluator with precise entity & stance checking."""
    annotated = []
    claim_type = claim_info.get("type", "GENERAL_FACT")
    claimed_entity = claim_info.get("claimed_entity", "").lower()
    claimed_year = claim_info.get("claimed_year", "")
    
    entity_words = [w for w in claimed_entity.split() if len(w) >= 3 and w not in STOP_WORDS]
    
    for e in selected_evidence:
        e_copy = dict(e)
        text = (e.get("title", "") + " " + e.get("content", "")).lower()
        
        matches_claimed_person = all(w in text for w in entity_words) if entity_words else False
        has_role_title = any(r in text for r in ["prime minister", "president", "chief minister", "cm", "ceo"])

        actual_holder = ""
        for leader in KNOWN_LEADERS:
            if leader in text and leader not in claimed_entity:
                actual_holder = leader.title()
                break

        if any(w in text for w in ["false", "fake", "hoax", "debunk", "misleading", "refutes", "denies", "rumor"]):
            e_copy["stance"] = "REFUTES"
            e_copy["stance_reasoning"] = "Source contains explicit debunking or denial terms."
        elif claim_type == "CURRENT_ROLE":
            if has_role_title:
                if matches_claimed_person:
                    e_copy["stance"] = "SUPPORTS"
                    e_copy["stance_reasoning"] = "Source confirms claimed entity holds official role."
                else:
                    e_copy["stance"] = "REFUTES"
                    if actual_holder:
                        e_copy["stance_reasoning"] = f"Official records confirm {actual_holder} holds the position, directly contradicting the claim."
                    else:
                        e_copy["stance_reasoning"] = "Source identifies official current office-holder directly contradicting claimed person."
            else:
                e_copy["stance"] = "INSUFFICIENT"
                e_copy["stance_reasoning"] = "Source provides background context but lacks direct confirmation or refutation."
        elif claim_type == "TEMPORAL":
            if any(future_yr in text for future_yr in ["2027", "2028", "2029", "planned", "scheduled", "expected", "target", "aims", "will launch"]):
                e_copy["stance"] = "REFUTES"
                e_copy["stance_reasoning"] = "Source indicates event is only planned/scheduled for a future date."
            elif claimed_year and (f"launched in {claimed_year}" in text or f"completed in {claimed_year}" in text):
                e_copy["stance"] = "SUPPORTS"
                e_copy["stance_reasoning"] = "Source text explicitly confirms event completed in claimed year."
            else:
                e_copy["stance"] = "INSUFFICIENT"
                e_copy["stance_reasoning"] = "Source provides background context but lacks matching event completion year."
        else:
            e_copy["stance"] = "INSUFFICIENT"
            e_copy["stance_reasoning"] = "Source provides background context but lacks direct confirmation or refutation."
            
        annotated.append(e_copy)
    return annotated

def synthesize_verdict_and_confidence(claim: str, claim_info: dict, annotated_evidence: list[dict]) -> dict:
    """
    Synthesizes the final verdict, confidence score, and concise explanation based on evidence stances.
    """
    if not annotated_evidence:
        return {
            "verdict": "UNCERTAIN",
            "confidence": 30,
            "explanation": "No search results or evidence were available to verify this claim.",
            "support_count": 0,
            "refute_count": 0,
            "insufficient_count": 0
        }

    support_items = [e for e in annotated_evidence if e.get("stance") == "SUPPORTS"]
    refute_items = [e for e in annotated_evidence if e.get("stance") == "REFUTES"]
    insufficient_items = [e for e in annotated_evidence if e.get("stance") == "INSUFFICIENT"]

    # 1. STRONG REFUTATION -> FAKE
    if refute_items and not support_items:
        max_refute_cred = max(e.get("credibility_score", 0.6) for e in refute_items)
        confidence = int(min(95, max(80, max_refute_cred * 95)))
        
        top_refutes = refute_items[0]
        explanation = f"Flagged as FAKE / REFUTED: Credible evidence directly contradicts this claim. {top_refutes.get('stance_reasoning', '')}"
        
        return {
            "verdict": "FAKE",
            "confidence": confidence,
            "explanation": explanation,
            "support_count": len(support_items),
            "refute_count": len(refute_items),
            "insufficient_count": len(insufficient_items)
        }

    # 2. STRONG SUPPORT -> REAL
    if support_items and not refute_items:
        max_support_cred = max(e.get("credibility_score", 0.6) for e in support_items)
        confidence = int(min(95, max(80, max_support_cred * 95)))
        
        top_supports = support_items[0]
        explanation = f"Verified REAL: Reliable sources explicitly confirm this claim. {top_supports.get('stance_reasoning', '')}"
        
        return {
            "verdict": "REAL",
            "confidence": confidence,
            "explanation": explanation,
            "support_count": len(support_items),
            "refute_count": len(refute_items),
            "insufficient_count": len(insufficient_items)
        }

    # 3. CONFLICTING EVIDENCE -> UNCERTAIN
    if support_items and refute_items:
        explanation = "Uncertain / Conflicting Evidence: Reliable sources disagree on this claim."
        return {
            "verdict": "UNCERTAIN",
            "confidence": 50,
            "explanation": explanation,
            "support_count": len(support_items),
            "refute_count": len(refute_items),
            "insufficient_count": len(insufficient_items)
        }

    # 4. INSUFFICIENT EVIDENCE -> UNCERTAIN
    explanation = "Uncertain: The retrieved search results provide background topic information but do not contain direct evidence confirming or contradicting the claim."
    return {
        "verdict": "UNCERTAIN",
        "confidence": 40,
        "explanation": explanation,
        "support_count": 0,
        "refute_count": 0,
        "insufficient_count": len(insufficient_items)
    }
