import os
import re
from llm_client import generate_chat_completion

CASUAL_PATTERNS = [
    r'^(?:hi|hello|hey|greetings|good morning|good afternoon|good evening)\b',
    r'^(?:bye|goodbye|cya|bye\s+bye|see\s+you|farewell)\b',
    r'^(?:thanks|thank\s+you|thx|cheers|awesome|cool|great|nice|ok|okay)\b',
    r'^(?:who\s+are\s+you|what\s+can\s+you\s+do|how\s+are\s+you|help)\b',
    r'^(?:bye|goodbye|cya|bye\s+bye|see\s+you)[!\.\s]*$'
]

def classify_input_intent(user_text: str) -> dict:
    """
    First-Stage Intent Classification Layer.
    Determines whether user input is 'CONVERSATIONAL' or 'FACTUAL_CLAIM'.
    """
    if not user_text:
        return {"intent": "CONVERSATIONAL", "type": "empty"}

    text_clean = user_text.strip().lower()

    # Fast pattern heuristic check
    for pat in CASUAL_PATTERNS:
        if re.search(pat, text_clean):
            return {"intent": "CONVERSATIONAL", "type": "casual_chat"}

    # Word count & structural assertion check
    words = re.findall(r'\w+', text_clean)
    if len(words) <= 2 and not any(w in text_clean for w in ["died", "banned", "pm", "cm", "president", "won", "lost"]):
        return {"intent": "CONVERSATIONAL", "type": "short_chat"}

    # Factual assertion indicators (verbs, numbers, role titles, proper nouns)
    news_indicators = [
        "is", "was", "are", "were", "banned", "launched", "died", "appointed", "sworn",
        "prime minister", "president", "chief minister", "cm", "pm", "ceo", "rupee", "dollar",
        "2023", "2024", "2025", "2026", "2027", "india", "us", "uk", "rbi", "isro", "who", "nasa"
    ]

    if any(ind in text_clean for ind in news_indicators):
        return {"intent": "FACTUAL_CLAIM", "type": "news_verification"}

    # Default to FACTUAL_CLAIM if more than 3 words
    if len(words) >= 3:
        return {"intent": "FACTUAL_CLAIM", "type": "general_statement"}

    return {"intent": "CONVERSATIONAL", "type": "casual_chat"}
