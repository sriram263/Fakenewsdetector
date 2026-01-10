import os
import json
from openai import OpenAI
from duckduckgo_search import DDGS
from dotenv import load_dotenv

load_dotenv()

# We use OpenRouter so you can swap models easily (Claude, Mistral, Llama 3)
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

class SmartAgent:
    def __init__(self):
        self.search_tool = DDGS()

    def get_live_info(self, query):
        """Searches the web for real-time verification."""
        try:
            results = list(self.search_tool.text(query, max_results=3))
            return results
        except Exception as e:
            return []

    def process_input(self, user_text):
        """
        The Master Function:
        1. Decides INTENT (Chat vs. News).
        2. If Chat -> Responds politely.
        3. If News -> Performs Search + Fact Check.
        """
        
        # --- PHASE 1: SEARCH ---
        # We search first because even "chat" might need context, 
        # but mostly we need it for news.
        search_results = self.get_live_info(user_text[:300])

        # --- PHASE 2: THE PROMPT ---
        # This prompt forces the AI to be a "Router" AND a "Fact Checker" simultaneously.
        system_prompt = f"""
        You are 'Veritas', an advanced AI News Analyst.
        
        You have two modes of operation based on the user's input:

        MODE 1: CASUAL CHAT / GENERAL QUESTIONS
        - If the user says "Hi", "Who are you?", "Thanks", or asks general questions unrelated to specific news claims.
        - ACTION: Reply naturally, politely, and concisely. 
        - If they ask what you can do, say: "I am Veritas. I use AI and live web search to verify news and detect misinformation."
        - OUTPUT FORMAT: Just the text of your reply.

        MODE 2: NEWS VERIFICATION
        - If the user provides a statement, headline, or article text that sounds like a claim (True or False).
        - ACTION: Cross-reference the input with the SEARCH RESULTS provided below.
        - CRITERIA:
          * If search results confirm it: Verdict is REAL.
          * If search results contradict it: Verdict is FAKE.
          * If no results found but text looks sensationalist/clickbait: Verdict is SUSPICIOUS.
        
        SEARCH RESULTS (Live Evidence):
        {json.dumps(search_results)}

        IMPORTANT: If you detect this is MODE 2 (News), you MUST format your output strictly as specific HTML to fit the app interface:
        
        <VERDICT_JSON>
        {{
            "type": "news_check",
            "verdict": "REAL" | "FAKE" | "UNCERTAIN",
            "confidence": 0-100,
            "explanation": "Your brief fact-check summary here.",
            "sources": "List of sources found in search"
        }}
        </VERDICT_JSON>
        """

        # --- PHASE 3: EXECUTION ---
        try:
            response = client.chat.completions.create(
                model="anthropic/claude-3-haiku", # Or "meta-llama/llama-3-8b-instruct:free"
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.3, # Keep it factual
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error connecting to AI: {e}"