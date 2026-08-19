import os
import json
import time
from tavily import TavilyClient
from dotenv import load_dotenv

import config
from knowledge_base import FactCheckKB
from query_generator import generate_multi_queries
from retrieval import execute_tavily_search, execute_multi_query_retrieval
from evidence_scorer import score_and_rank_candidates
from verifier import analyze_claim_type, evaluate_evidence_relationships, synthesize_verdict_and_confidence
from llm_client import generate_chat_completion

load_dotenv()

tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

class SmartAgent:
    def __init__(self):
        # Initialize Persistent Knowledge Base (v2.1)
        self.kb = FactCheckKB()

    def get_live_info(self, query):
        """Original Baseline Tavily Search implementation."""
        try:
            response = tavily_client.search(
                query=query,
                topic="news",
                days=365,
                max_results=config.TAVILY_MAX_RESULTS_PER_QUERY,
                include_answer=True
            )
            return response.get('results', [])
        except Exception as e:
            return []

    def is_casual_chat(self, text: str) -> bool:
        """Determines if user text is a casual greeting or meta-question."""
        greetings = ["hi", "hello", "hey", "thanks", "thank you", "who are you", "what can you do", "help"]
        norm = text.strip().lower()
        return norm in greetings or any(norm.startswith(g) for g in ["hi ", "hello ", "hey "])

    def process_input(self, user_text, retrieval_mode=None, kb_enabled=None):
        """
        Processes user text and returns structured result dictionary with sub-second optimization.
        """
        start_time = time.time()
        
        mode = retrieval_mode if retrieval_mode is not None else config.RETRIEVAL_MODE
        use_kb = kb_enabled if kb_enabled is not None else config.KNOWLEDGE_BASE_ENABLED

        # 1. CASUAL CHAT CHECK
        if self.is_casual_chat(user_text):
            try:
                chat_reply, used_prov = generate_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are Veritas, a polite AI fact-checking assistant. Answer casual greetings concisely."},
                        {"role": "user", "content": user_text}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
            except Exception as e:
                chat_reply = "Hello! I am Veritas AI, your fact-checking analyst. Paste a news headline or claim to verify."
                
            return {
                "ai_response": chat_reply,
                "verdict_data": None,
                "sources": [],
                "retrieval_details": {"is_chat": True, "latency_ms": round((time.time() - start_time) * 1000, 2)}
            }

        retrieval_details = {
            "retrieval_mode": mode,
            "kb_enabled": use_kb,
            "kb_status": "DISABLED" if not use_kb else "MISS",
            "kb_similarity": 0.0,
            "kb_reused": False,
            "kb_age_days": 0.0,
            "queries": [],
            "raw_results_count": 0,
            "deduplicated_count": 0,
            "duplicates_removed": 0,
            "selected_evidence": [],
            "conflict_detected": False,
            "domain_breakdown": {},
            "latency_ms": 0.0
        }

        # --- PHASE 1: INSTANT SEMANTIC KNOWLEDGE BASE SEARCH ---
        if use_kb:
            kb_res = self.kb.search_similar_fact_check(user_text)
            retrieval_details["kb_status"] = kb_res["status"]
            retrieval_details["kb_similarity"] = kb_res["similarity_score"]
            retrieval_details["kb_age_days"] = kb_res["age_days"]

            if kb_res["hit"] and kb_res["is_fresh"]:
                retrieval_details["kb_reused"] = True
                entry = kb_res["entry"]
                
                verdict_data = {
                    "type": "news_check",
                    "verdict": entry["verdict"],
                    "confidence": entry["confidence"],
                    "explanation": f"[Reused from Knowledge Base persistent memory (Verified {kb_res['age_days']} days ago)]: {entry['explanation']}",
                    "sources": ", ".join(entry.get("source_domains", []))
                }
                
                ai_response = f"<VERDICT_JSON>\n{json.dumps(verdict_data, indent=2)}\n</VERDICT_JSON>"
                retrieval_details["latency_ms"] = round((time.time() - start_time) * 1000, 2)
                retrieval_details["selected_evidence"] = entry.get("evidence_selected", [])
                
                return {
                    "ai_response": ai_response,
                    "verdict_data": verdict_data,
                    "sources": entry.get("evidence_selected", []),
                    "retrieval_details": retrieval_details
                }

        # --- PHASE 2: PARALLEL MULTI-QUERY RETRIEVAL ---
        selected_evidence = []
        queries_used = []

        if mode == "baseline":
            raw_results = self.get_live_info(user_text[:300])
            selected_evidence = raw_results
            queries_used = [{"query": user_text[:300], "category": "direct", "purpose": "Baseline single query"}]
            retrieval_details["queries"] = queries_used
            retrieval_details["raw_results_count"] = len(raw_results)
            retrieval_details["deduplicated_count"] = len(raw_results)
            retrieval_details["selected_evidence"] = raw_results
        else:
            queries_info = generate_multi_queries(user_text)
            queries_used = queries_info
            retrieval_details["queries"] = queries_info

            # Parallel multi-query retrieval (ThreadPoolExecutor)
            candidates, rstats = execute_multi_query_retrieval(queries_info)
            retrieval_details["raw_results_count"] = rstats["raw_results_count"]
            retrieval_details["deduplicated_count"] = rstats["deduplicated_count"]
            retrieval_details["duplicates_removed"] = rstats["duplicates_removed"]

            selected_evidence, sstats = score_and_rank_candidates(candidates, user_text)
            retrieval_details["selected_evidence"] = selected_evidence
            retrieval_details["conflict_detected"] = sstats.get("conflict_detected", False)
            retrieval_details["domain_breakdown"] = sstats.get("domain_breakdown", {})

        # --- PHASE 3: FAST CLAIM-EVIDENCE VERIFICATION & SYNTHESIS ---
        claim_info = analyze_claim_type(user_text)
        annotated_evidence = evaluate_evidence_relationships(user_text, claim_info, selected_evidence)
        verdict_synth = synthesize_verdict_and_confidence(user_text, claim_info, annotated_evidence)

        # Single-pass unified LLM completion call for fast explanation summary
        prompt = f"""
        You are 'Veritas', an advanced AI News Analyst.
        
        CLAIM: "{user_text}"
        CLAIM CATEGORY: {claim_info.get('type')}
        DETERMINED VERDICT: {verdict_synth.get('verdict')}
        CALIBRATED CONFIDENCE: {verdict_synth.get('confidence')}%

        EVIDENCE SOURCES:
        {json.dumps([{"domain": e.get("domain"), "title": e.get("title"), "stance": e.get("stance"), "content": e.get("content", "")[:250]} for e in annotated_evidence], indent=2)}

        Provide a concise 2-sentence explanation summarizing WHY the claim is {verdict_synth.get('verdict')}.
        If the claim is FAKE / REFUTED, explicitly name what the evidence actually establishes (e.g. who currently holds the role, or the actual planned date).
        Do NOT change the DETERMINED VERDICT from {verdict_synth.get('verdict')}.

        Output STRICT JSON format:
        <VERDICT_JSON>
        {{
            "type": "news_check",
            "verdict": "{verdict_synth.get('verdict')}",
            "confidence": {verdict_synth.get('confidence')},
            "explanation": "Your concise 2-sentence explanation.",
            "sources": "List of top source domains"
        }}
        </VERDICT_JSON>
        """

        try:
            ai_response, used_prov = generate_chat_completion(
                messages=[
                    {"role": "system", "content": "You are a fact-checking analyst. Follow instructions and output strict JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,
                max_tokens=300
            )
            
            verdict_data = None
            if "<VERDICT_JSON>" in ai_response:
                try:
                    json_str = ai_response.split("<VERDICT_JSON>")[1].split("</VERDICT_JSON>")[0].strip()
                    verdict_data = json.loads(json_str)
                except Exception:
                    pass

            if not verdict_data:
                verdict_data = {
                    "type": "news_check",
                    "verdict": verdict_synth["verdict"],
                    "confidence": verdict_synth["confidence"],
                    "explanation": verdict_synth["explanation"],
                    "sources": ", ".join(e.get("domain", "") for e in selected_evidence)
                }
                ai_response = f"<VERDICT_JSON>\n{json.dumps(verdict_data, indent=2)}\n</VERDICT_JSON>"

            # Store completed result in Knowledge Base for sub-second future queries
            if use_kb:
                try:
                    self.kb.store_fact_check(
                        original_claim=user_text,
                        verdict=verdict_data.get("verdict", "UNCERTAIN"),
                        confidence=int(verdict_data.get("confidence", 50)),
                        explanation=verdict_data.get("explanation", ""),
                        evidence_snippets=[s.get("content", "") for s in annotated_evidence],
                        source_urls=[s.get("url", "") for s in annotated_evidence],
                        source_titles=[s.get("title", "") for s in annotated_evidence],
                        source_domains=[s.get("domain", "") for s in annotated_evidence],
                        source_credibility_info=[{"domain": s.get("domain"), "score": s.get("credibility_score"), "category": s.get("domain_category")} for s in annotated_evidence],
                        publication_dates=[s.get("published_date", "") for s in annotated_evidence],
                        search_queries_used=[q.get("query", "") for q in queries_used],
                        evidence_selected=annotated_evidence
                    )
                except Exception as ex:
                    print(f"[SmartAgent] KB store error: {ex}")

            retrieval_details["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            retrieval_details["selected_evidence"] = annotated_evidence

            return {
                "ai_response": ai_response,
                "verdict_data": verdict_data,
                "sources": annotated_evidence,
                "retrieval_details": retrieval_details
            }
            
        except Exception as e:
            retrieval_details["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            fallback_verdict = {
                "type": "news_check",
                "verdict": verdict_synth["verdict"],
                "confidence": verdict_synth["confidence"],
                "explanation": verdict_synth["explanation"],
                "sources": ""
            }
            return {
                "ai_response": f"<VERDICT_JSON>\n{json.dumps(fallback_verdict)}\n</VERDICT_JSON>",
                "verdict_data": fallback_verdict,
                "sources": annotated_evidence,
                "retrieval_details": retrieval_details
            }