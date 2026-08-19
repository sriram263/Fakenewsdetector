import os
import json
import re
import config
from llm_client import generate_chat_completion

def generate_multi_queries(claim: str, num_queries: int = config.NUM_QUERIES) -> list[dict]:
    """
    Generates complementary search queries with specific targeted purposes:
    1. Direct claim query
    2. Role-holder / Authority query
    3. Independent verification query
    """
    if not claim:
        return []

    prompt = f"""
    You are a professional fact-checker search query optimizer.
    For the claim below, generate exactly {num_queries} distinct, complementary search queries with different purposes.
    If the claim is about an office holder (e.g. PM, President, CM, CEO), ensure one query explicitly asks who currently holds that position.

    CLAIM: "{claim}"

    Provide output ONLY as valid JSON in this structure:
    [
        {{"query": "direct search phrase", "category": "direct", "purpose": "Direct keywords search"}},
        {{"query": "official authority query", "category": "official", "purpose": "Check government or official records"}},
        {{"query": "who is current office holder query", "category": "contradiction", "purpose": "Check actual office holder or status"}}
    ]
    """

    try:
        content, used_provider = generate_chat_completion(
            messages=[
                {"role": "system", "content": "You output only clean, valid JSON arrays. Do not add markdown formatting or conversational text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        if "</think>" in content:
            content = content.split("</think>")[-1].strip()

        start_idx = content.find("[")
        if start_idx != -1:
            queries_data, _ = json.JSONDecoder().raw_decode(content[start_idx:])
            cleaned_queries = []
            if isinstance(queries_data, list):
                for qitem in queries_data:
                    if isinstance(qitem, dict) and qitem.get("query"):
                        qtext = str(qitem.get("query", "")).strip()
                        if len(qtext) > 3 and not qtext.startswith("..."):
                            cleaned_queries.append({
                                "query": qtext,
                                "category": qitem.get("category", "general"),
                                "purpose": qitem.get("purpose", "Verification query")
                            })
            if cleaned_queries:
                return cleaned_queries[:num_queries]
    except Exception as e:
        print(f"[QueryGenerator] Query expansion fallback triggered: {e}")

    return _generate_fallback_queries(claim, num_queries)

def _generate_fallback_queries(claim: str, num_queries: int) -> list[dict]:
    """Fallback rule-based query expansion for high reliability."""
    base_clean = claim.replace("?", "").replace('"', '').strip()
    claim_lower = claim.lower()
    
    role_match = re.search(r'\b(?:prime minister|president|ceo|cto|cfo|chairman|head|leader|governor|king|queen|minister|cm|chief minister)(?:\s+of\s+[a-zA-Z]+)?\b', claim_lower)
    
    fallback_queries = [
        {
            "query": base_clean,
            "category": "direct",
            "purpose": "Direct claim search"
        }
    ]

    if role_match:
        role_str = role_match.group(0)
        fallback_queries.append({
            "query": f"who is current {role_str} official",
            "category": "official",
            "purpose": "Identify current official office holder"
        })
    else:
        fallback_queries.append({
            "query": f"{base_clean} official statement government news",
            "category": "official",
            "purpose": "Official authority verification"
        })

    fallback_queries.append({
        "query": f"{base_clean} Reuters BBC fact check debunked rumor",
        "category": "contradiction",
        "purpose": "Independent verification and debunking"
    })
    
    return fallback_queries[:num_queries]
