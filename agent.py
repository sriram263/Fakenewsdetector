import os
import json
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv

load_dotenv()

# Initialize Clients
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class SmartAgent:
    def __init__(self):
        pass

    def get_live_info(self, query):
        """
        Uses Tavily AI to get highly accurate, fresh news.
        Returns a LIST of dictionary objects (Title, URL, Content).
        """
        try:
            response = tavily_client.search(
                query=query,
                topic="news",
                days=365,
                max_results=5,
                include_answer=True
            )
            return response.get('results', [])
        except Exception as e:
            return []

    def process_input(self, user_text):
        """
        Returns a DICTIONARY: 
        {
            "ai_response": "The text from Claude...",
            "sources": [List of search result objects] 
        }
        """
        
        # --- PHASE 1: SEARCH ---
        # We search first, but the AI decides if it needs to use it.
        search_results = self.get_live_info(user_text[:300])

        # --- PHASE 2: THE PROMPT ---
        system_prompt = f"""
        You are 'Veritas', an advanced AI News Analyst.
        
        SEARCH RESULTS:
        {json.dumps(search_results)}

        You have two modes. CHOOSE WISELY based on user input:

        MODE 1: CASUAL CHAT (Priority)
        - Trigger: If user says "Hi", "Hello", "Thanks", "Who are you?", or asks for help.
        - RULE: IGNORE the search results above if they are irrelevant to a greeting (e.g., ignore "Hello Kitty" news for "Hello").
        - ACTION: Reply naturally, politely, and concisely.
        - OUTPUT: Just the text of your reply (No JSON).

        MODE 2: NEWS VERIFICATION
        - Trigger: If the user makes a claim, asks a factual question, or pastes a headline.
        - ACTION: Verify the claim using the SEARCH RESULTS.
        - OUTPUT: STRICT JSON format below.
        
        <VERDICT_JSON>
        {{
            "type": "news_check",
            "verdict": "REAL" | "FAKE" | "UNCERTAIN",
            "confidence": 0-100,
            "explanation": "Brief summary.",
            "sources": "List of source names"
        }}
        </VERDICT_JSON>
        """

        # --- PHASE 3: EXECUTION ---
        try:
            response = client.chat.completions.create(
                model="anthropic/claude-3-haiku", 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text}
                ],
                temperature=0.3,
            )
            
            return {
                "ai_response": response.choices[0].message.content,
                "sources": search_results
            }
            
        except Exception as e:
            return {
                "ai_response": f"Error: {e}",
                "sources": []
            }