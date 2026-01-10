# import os
# import json
# from openai import OpenAI
# from duckduckgo_search import DDGS
# from dotenv import load_dotenv

# load_dotenv()

# # Initialize Client
# client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )

# class SmartAgent:
#     def __init__(self):
#         self.search_tool = DDGS()

#     def get_live_info(self, query):
#         """
#         Searches the web with 'Freshness' enforcement.
#         - max_results=5: increased to catch breaking news that might not be #1 yet.
#         - timelimit='y': restricts results to the past year to avoid outdated stats.
#         """
#         try:
#             # FIX 1: Increased max_results to 5 to dig deeper for latest news
#             # FIX 2: Added timelimit='y' to physically block old articles
#             results = list(self.search_tool.text(query, max_results=5, timelimit='y'))
#             return results
#         except Exception as e:
#             return []

#     def process_input(self, user_text):
#         """
#         The Master Function:
#         1. Decides INTENT (Chat vs. News).
#         2. If Chat -> Responds politely.
#         3. If News -> Performs Search + Fact Check.
#         """
        
#         # --- PHASE 1: SEARCH ---
#         search_results = self.get_live_info(user_text[:300])

#         # --- PHASE 2: THE PROMPT ---
#         system_prompt = f"""
#         You are 'Veritas', an advanced AI News Analyst.
        
#         You have two modes of operation based on the user's input:

#         MODE 1: CASUAL CHAT / GENERAL QUESTIONS
#         - If the user says "Hi", "Who are you?", "Thanks", or asks general questions unrelated to specific news claims.
#         - ACTION: Reply naturally, politely, and concisely. 
#         - If they ask what you can do, say: "I am Veritas. I use AI and live web search to verify news and detect misinformation."
#         - OUTPUT FORMAT: Just the text of your reply.

#         MODE 2: NEWS VERIFICATION
#         - If the user provides a statement, headline, or article text that sounds like a claim (True or False).
#         - ACTION: Cross-reference the input with the SEARCH RESULTS provided below.
        
#         SEARCH RESULTS (Live Evidence):
#         {json.dumps(search_results)}

#         CRITICAL INSTRUCTIONS FOR VERDICT:
#         1. **Check Dates:** If search results have conflicting data (e.g., "India is 5th" vs "India becomes 4th"), ALWAYS trust the source with the most recent date or the one labeled "projection" for the current year.
#         2. **Nuance:** If the status is changing (e.g., "Projected to become 4th next month" vs "Is currently 4th"), be precise in your explanation.
#         3. **Verdict Logic:**
#            - Matches Evidence -> REAL
#            - Contradicts Evidence -> FAKE
#            - No Evidence/Ambiguous -> UNCERTAIN

#         IMPORTANT: If you detect this is MODE 2 (News), you MUST format your output strictly as specific HTML to fit the app interface:
        
#         <VERDICT_JSON>
#         {{
#             "type": "news_check",
#             "verdict": "REAL" | "FAKE" | "UNCERTAIN",
#             "confidence": 0-100,
#             "explanation": "Your brief fact-check summary here. Explicitly mention the correct current status if the user is wrong.",
#             "sources": "List of sources found in search"
#         }}
#         </VERDICT_JSON>
#         """

#         # --- PHASE 3: EXECUTION ---
#         try:
#             response = client.chat.completions.create(
#                 model="anthropic/claude-3-haiku", 
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_text}
#                 ],
#                 temperature=0.3, # Low temperature for factual consistency
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             return f"Error connecting to AI: {e}"
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
        search_results = self.get_live_info(user_text[:300])

        # --- PHASE 2: THE PROMPT ---
        system_prompt = f"""
        You are 'Veritas', an advanced AI News Analyst.
        
        MODE 1: CASUAL CHAT
        - If input is generic ("Hi", "Thanks"), reply normally.

        MODE 2: NEWS VERIFICATION
        - Analyze the claim using the SEARCH RESULTS below.
        
        SEARCH RESULTS:
        {json.dumps(search_results)}

        CRITICAL INSTRUCTIONS:
        1. Trust fresh data (compare dates).
        2. If Mode 2, output strictly as HTML JSON:
        
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
            
            # RETURN BOTH TEXT AND DATA
            return {
                "ai_response": response.choices[0].message.content,
                "sources": search_results
            }
            
        except Exception as e:
            return {
                "ai_response": f"Error: {e}",
                "sources": []
            }