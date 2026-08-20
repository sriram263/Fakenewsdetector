import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
import json
from llm_client import generate_chat_completion

claim = "sriram is the cheif minister of tamil nadu"
news = [
    {"domain": "thehindu.com", "title": "What Tamil Nadu CM Vijay really needs to see at Fort St. George", "content": "Film star C. Joseph Vijay has taken charge as the Chief Minister of Tamil Nadu."},
    {"domain": "hindustantimes.com", "title": "Vijay is now chief minister of Tamil Nadu", "content": "Tamil Nadu chief minister Vijay addresses public."}
]

prompt = f"""
You are Veritas, an expert AI Fact-Checking Analyst.

Analyze the USER CLAIM against the RETRIEVED NEWS ARTICLES below to determine if the claim is REAL, FAKE, or UNCERTAIN.

USER CLAIM: "{claim}"

RETRIEVED NEWS ARTICLES:
{json.dumps(news, indent=2)}

EVALUATION INSTRUCTIONS:
1. Compare the user claim directly against the retrieved news articles.
2. If the user claim is contradicted by official news (e.g. claim says Sriram is CM, but news articles confirm Vijay is CM), mark verdict as "FAKE" with 90-95% confidence.
3. In the explanation, explicitly name the true office-holder / fact established by the news articles.
4. Output STRICT JSON format only:
<VERDICT_JSON>
{{
    "verdict": "REAL" | "FAKE" | "UNCERTAIN",
    "confidence": 90,
    "explanation": "Concise 2-sentence explanation naming the true facts from news.",
    "sources": "thehindu.com, hindustantimes.com"
}}
</VERDICT_JSON>
"""

res, prov = generate_chat_completion([
    {"role": "system", "content": "You are a strict JSON fact-checking assistant."},
    {"role": "user", "content": prompt}
], temperature=0.1)

print(f"Provider used: {prov}")
print(f"Response:\n{res}")
