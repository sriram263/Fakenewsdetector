import os
import json
import time
import re
from tavily import TavilyClient
from dotenv import load_dotenv

import config
from knowledge_base import FactCheckKB
from intent_classifier import classify_input_intent
from query_generator import generate_multi_queries
from retrieval import execute_tavily_search, execute_multi_query_retrieval
from evidence_scorer import score_and_rank_candidates
from verifier import evaluate_evidence_relationships, analyze_claim_type
from llm_client import generate_chat_completion
from normalization import stem_word, STOP_WORDS

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

    def process_input(self, user_text, retrieval_mode=None, kb_enabled=None):
        """
        Direct LLM-Driven Fact-Checking Pipeline:
        Combines User Claim + Retrieved Web Articles directly into the LLM for high-accuracy evaluation.
        """
        start_time = time.time()
        
        mode = retrieval_mode if retrieval_mode is not None else config.RETRIEVAL_MODE
        use_kb = kb_enabled if kb_enabled is not None else config.KNOWLEDGE_BASE_ENABLED

        # --- STAGE 1: FIRST-STAGE INTENT CLASSIFICATION ---
        intent_info = classify_input_intent(user_text)
        if intent_info["intent"] == "CONVERSATIONAL":
            try:
                chat_reply, used_prov = generate_chat_completion(
                    messages=[
                        {"role": "system", "content": "You are Veritas AI, a friendly, professional AI fact-checking analyst. Respond warmly to greetings, farewells, or casual chatter."},
                        {"role": "user", "content": user_text}
                    ],
                    temperature=0.3,
                    max_tokens=150
                )
            except Exception:
                chat_reply = "Goodbye! Feel free to ask whenever you need to verify news or claims. Have a great day!"
                
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

        # --- STAGE 2: PERSISTENT SEMANTIC KNOWLEDGE BASE SEARCH ---
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

        # --- STAGE 3: PARALLEL MULTI-QUERY RETRIEVAL ---
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

            candidates, rstats = execute_multi_query_retrieval(queries_info)
            retrieval_details["raw_results_count"] = rstats["raw_results_count"]
            retrieval_details["deduplicated_count"] = rstats["deduplicated_count"]
            retrieval_details["duplicates_removed"] = rstats["duplicates_removed"]

            selected_evidence, sstats = score_and_rank_candidates(candidates, user_text)
            retrieval_details["selected_evidence"] = selected_evidence
            retrieval_details["conflict_detected"] = sstats.get("conflict_detected", False)
            retrieval_details["domain_breakdown"] = sstats.get("domain_breakdown", {})

        # --- STAGE 4: DIRECT LLM EVALUATION (USER CLAIM + RETRIEVED ARTICLES) ---
        articles_payload = [
            {
                "title": e.get("title", ""),
                "domain": e.get("domain", ""),
                "snippet": e.get("content", "")[:350],
                "url": e.get("url", "")
            }
            for e in selected_evidence
        ]

        llm_prompt = f"""
        You are Veritas AI, an expert news fact-checking analyst.

        USER CLAIM: "{user_text}"

        RETRIEVED LIVE NEWS ARTICLES:
        {json.dumps(articles_payload, indent=2)}

        CRITICAL VERDICT DECISION RULES:
        RULE 1 (CONTRADICTED OR FABRICATED CLAIMS): Check if the user claim asserts an event, acquisition, office role, or future date that is false, fabricated, or contradicted by facts (e.g. claiming Elon Musk bought WhatsApp when Meta owns it; claiming someone holds an office when another person does; or asserting a future event date like 2028). If the claim is false or fabricated, the verdict MUST BE "FAKE". You are STRICTLY FORBIDDEN from setting verdict to "UNCERTAIN" when a claim is false or fabricated!
        RULE 2 (REAL VERDICT): Set verdict to "REAL" ONLY IF the news articles explicitly confirm the EXACT person, event, or fact named in the user claim.
        RULE 3 (UNCERTAIN VERDICT): Set verdict to "UNCERTAIN" ONLY IF search results are completely empty or uninformative.

        Output STRICT JSON format:
        {{
            "type": "news_check",
            "verdict": "REAL" | "FAKE" | "UNCERTAIN",
            "confidence": 95,
            "explanation": "Concise 2-sentence explanation explicitly naming the true office-holder / facts established by the news articles.",
            "sources": "List of top domain names"
        }}
        """

        try:
            ai_response, used_prov = generate_chat_completion(
                messages=[
                    {"role": "system", "content": "You are an expert fact-checker. Follow instructions and output strict JSON."},
                    {"role": "user", "content": llm_prompt}
                ],
                temperature=0.0,
                max_tokens=350
            )

            # Robust Universal JSON Parser (handles ```json, <VERDICT_JSON>, or raw JSON)
            verdict_data = None
            start_j = ai_response.find("{")
            end_j = ai_response.rfind("}")
            if start_j != -1 and end_j != -1 and end_j > start_j:
                try:
                    candidate_json = ai_response[start_j:end_j+1]
                    verdict_data, _ = json.JSONDecoder().raw_decode(candidate_json)
                except Exception:
                    try:
                        verdict_data, _ = json.JSONDecoder().raw_decode(ai_response[start_j:])
                    except Exception:
                        pass

            if not verdict_data:
                # Rule-based fallback stance evaluation if JSON parsing fails
                annotated_evidence = evaluate_evidence_relationships(user_text, analyze_claim_type(user_text), selected_evidence)
                supports = [e for e in annotated_evidence if e.get("stance") == "SUPPORTS"]
                refutes = [e for e in annotated_evidence if e.get("stance") == "REFUTES"]
                
                if refutes and not supports:
                    v_str, conf = "FAKE", 90
                    exp = f"Flagged as FAKE: Credible evidence directly contradicts this claim. {refutes[0].get('stance_reasoning', '')}"
                elif supports and not refutes:
                    v_str, conf = "REAL", 90
                    exp = f"Verified REAL: Reliable sources explicitly confirm this claim. {supports[0].get('stance_reasoning', '')}"
                else:
                    v_str, conf = "UNCERTAIN", 40
                    exp = "Uncertain: Search results lack definitive confirmation or refutation."

                verdict_data = {
                    "type": "news_check",
                    "verdict": v_str,
                    "confidence": conf,
                    "explanation": exp,
                    "sources": ", ".join(e.get("domain", "") for e in selected_evidence[:3])
                }
                ai_response = f"<VERDICT_JSON>\n{json.dumps(verdict_data, indent=2)}\n</VERDICT_JSON>"

            # Precise Stance Annotation per Snippet for UI Display
            claim_info = analyze_claim_type(user_text)
            claimed_person = claim_info.get("claimed_entity", "").lower()
            claimed_words = [w for w in claimed_person.split() if len(w) >= 3 and stem_word(w) not in STOP_WORDS]

            final_v = verdict_data.get("verdict", "UNCERTAIN")
            for e in selected_evidence:
                text_content = (e.get("title", "") + " " + e.get("content", "")).lower()
                matches_claimed_person = any(w in text_content for w in claimed_words) if claimed_words else False

                if final_v == "FAKE":
                    e["stance"] = "REFUTES"
                    e["stance_reasoning"] = "Source provides facts contradicting or refuting the claim."
                elif final_v == "REAL":
                    if matches_claimed_person or not claimed_words:
                        e["stance"] = "SUPPORTS"
                        e["stance_reasoning"] = "Source explicitly confirms claimed facts."
                    else:
                        e["stance"] = "INSUFFICIENT"
                        e["stance_reasoning"] = "Source provides background context."
                else:
                    e["stance"] = "INSUFFICIENT"
                    e["stance_reasoning"] = "Source provides general background context."

            # Store in Knowledge Base (only REAL or FAKE verdicts get stored!)
            if use_kb and verdict_data.get("verdict") in ["REAL", "FAKE"]:
                try:
                    self.kb.store_fact_check(
                        original_claim=user_text,
                        verdict=verdict_data.get("verdict"),
                        confidence=int(verdict_data.get("confidence", 90)),
                        explanation=verdict_data.get("explanation", ""),
                        evidence_snippets=[s.get("content", "") for s in selected_evidence],
                        source_urls=[s.get("url", "") for s in selected_evidence],
                        source_titles=[s.get("title", "") for s in selected_evidence],
                        source_domains=[s.get("domain", "") for s in selected_evidence],
                        source_credibility_info=[{"domain": s.get("domain"), "score": s.get("credibility_score"), "category": s.get("domain_category")} for s in selected_evidence],
                        publication_dates=[s.get("published_date", "") for s in selected_evidence],
                        search_queries_used=[q.get("query", "") for q in queries_used],
                        evidence_selected=selected_evidence
                    )
                except Exception as ex:
                    print(f"[SmartAgent] KB store error: {ex}")

            retrieval_details["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            retrieval_details["selected_evidence"] = selected_evidence

            return {
                "ai_response": ai_response,
                "verdict_data": verdict_data,
                "sources": selected_evidence,
                "retrieval_details": retrieval_details
            }

        except Exception as e:
            retrieval_details["latency_ms"] = round((time.time() - start_time) * 1000, 2)
            fallback_verdict = {
                "type": "news_check",
                "verdict": "UNCERTAIN",
                "confidence": 40,
                "explanation": f"Unable to process claim evaluation: {e}",
                "sources": ""
            }
            return {
                "ai_response": f"<VERDICT_JSON>\n{json.dumps(fallback_verdict)}\n</VERDICT_JSON>",
                "verdict_data": fallback_verdict,
                "sources": selected_evidence,
                "retrieval_details": retrieval_details
            }