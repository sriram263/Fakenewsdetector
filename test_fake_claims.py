import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
from llm_client import generate_chat_completion

claims = [
    "Elon Musk bought WhatsApp in 2024",
    "India won the world cup in 2028",
    "vimal is the prime minister of india",
    "Narendra Modi is the prime minister of india"
]

news_sample = [
    {"domain": "reuters.com", "title": "Meta owns WhatsApp, Instagram and Facebook", "snippet": "Meta Platforms continues to operate WhatsApp worldwide."},
    {"domain": "bbc.com", "title": "Elon Musk owns X (formerly Twitter) and Tesla", "snippet": "Elon Musk leads Tesla, SpaceX, and X."}
]

for claim in claims:
    prompt = f"""
    You are Veritas AI, an expert news fact-checking analyst.

    USER CLAIM: "{claim}"

    RETRIEVED LIVE NEWS ARTICLES:
    {json.dumps(news_sample, indent=2)}

    CRITICAL VERDICT DECISION RULES:
    RULE 1 (CONTRADICTED OR FABRICATED CLAIMS):
    - If the user claim asserts an event, acquisition, role, or date that is false, fabricated, or contradicted by known facts (e.g. claiming Elon Musk bought WhatsApp when Meta owns it; claiming someone holds an office when another person does; or asserting a future event date like 2028), the verdict MUST BE "FAKE". You are STRICTLY FORBIDDEN from setting verdict to "UNCERTAIN" for false or fabricated claims!
    RULE 2 (REAL VERDICT): Set verdict to "REAL" ONLY IF official news explicitly confirms the exact claim.
    RULE 3 (UNCERTAIN VERDICT): Set verdict to "UNCERTAIN" ONLY IF search results are completely empty or uninformative.

    Output STRICT JSON format:
    {{
        "verdict": "REAL" | "FAKE" | "UNCERTAIN",
        "confidence": 95,
        "explanation": "Concise 2-sentence explanation naming the true facts."
    }}
    """

    res, prov = generate_chat_completion([
        {"role": "system", "content": "You are a strict JSON fact-checker."},
        {"role": "user", "content": prompt}
    ], temperature=0.0)

    print(f"\nClaim: \"{claim}\"")
    print("Response:", res)
