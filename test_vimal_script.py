import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
from llm_client import generate_chat_completion

claim = "vimal is the chief minister of tamil nadu"
news = [
    {"domain": "thehindu.com", "title": "M.K. Stalin urges Tamil Nadu CM Vijay to abandon talks", "snippet": "Tamil Nadu CM Vijay addresses meeting."},
    {"domain": "indianexpress.com", "title": "The men around Vijay: Why Tamil Nadu CM inner circle is under scrutiny", "snippet": "Vijay takes oath as chief minister."},
    {"domain": "bbc.com", "title": "Vijay, Tamil Nadu election results 2026: Film star takes oath as chief minister", "snippet": "Film star Vijay has taken charge as Chief Minister of Tamil Nadu."}
]

prompt = f"""
You are Veritas AI, an expert news fact-checking analyst.

USER CLAIM: "{claim}"

RETRIEVED NEWS ARTICLES:
{json.dumps(news, indent=2)}

CRITICAL VERDICT DECISION RULES:
RULE 1 (NAME CONTRADICTION): Check if the person named in the USER CLAIM (e.g. "Vimal") is DIFFERENT from the person named in the NEWS ARTICLES (e.g. "Vijay" / "Stalin"). If the claim names one person, but the news articles confirm a DIFFERENT person holds that position, the verdict MUST BE "FAKE". You are STRICTLY FORBIDDEN from setting verdict to "REAL" when names differ!
RULE 2 (REAL VERDICT): Set verdict to "REAL" ONLY IF the news articles explicitly confirm the EXACT person named in the user claim holds the position.
RULE 3 (UNCERTAIN VERDICT): Set verdict to "UNCERTAIN" ONLY IF news articles have zero relevant information.

Output STRICT JSON format:
<VERDICT_JSON>
{{
    "verdict": "REAL" | "FAKE" | "UNCERTAIN",
    "confidence": 95,
    "explanation": "State clearly that the claim is FAKE because news confirms Vijay is Chief Minister, not Vimal.",
    "sources": "thehindu.com, bbc.com"
}}
</VERDICT_JSON>
"""

res, prov = generate_chat_completion([
    {"role": "system", "content": "You are a strict JSON fact-checking assistant."},
    {"role": "user", "content": prompt}
], temperature=0.0)

print(f"Provider used: {prov}")
print(f"Response:\n{res}")
