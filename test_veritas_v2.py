import os
import sys
import json
import shutil

# Ensure UTF-8 stdout encoding for Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import config
from knowledge_base import FactCheckKB
from verifier import analyze_claim_type, evaluate_evidence_relationships, synthesize_verdict_and_confidence
from agent import SmartAgent

def run_all_tests():
    print("==================================================")
    print("VERITAS AI v2.1 -- VERIFICATION & TESTING SUITE")
    print("==================================================")

    test_dir = "./test_kb_data"
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    os.makedirs(test_dir, exist_ok=True)

    agent = SmartAgent()
    agent.kb = FactCheckKB(kb_dir=test_dir)

    passed_tests = 0

    # --------------------------------------------------
    # TEST A: Direct Role Contradiction (e.g. Sriram is PM of India)
    # --------------------------------------------------
    print("\n[TEST A] Direct Role Contradiction ('The prime minister of India is Sriram')...")
    claim_a = "The prime minister of India is Sriram."
    res_a = agent.process_input(claim_a, retrieval_mode="enhanced", kb_enabled=False)
    v_a = res_a.get("verdict_data", {})
    
    print(f"  Result Verdict: {v_a.get('verdict')}, Confidence: {v_a.get('confidence')}%\n  Explanation: {v_a.get('explanation')}")
    assert v_a.get("verdict") == "FAKE", f"Expected FAKE for contradicted role, got {v_a.get('verdict')}"
    assert v_a.get("confidence") >= 80, f"Expected high confidence >= 80%, got {v_a.get('confidence')}%"
    print("  [PASSED] TEST A: Contradicted role correctly classified as FAKE / REFUTED.")
    passed_tests += 1

    # --------------------------------------------------
    # TEST B: Temporal Contradiction (Chandrayaan-4 in 2026 vs 2027)
    # --------------------------------------------------
    print("\n[TEST B] Temporal Contradiction ('India launched Chandrayaan-4 in 2026')...")
    claim_b = "India launched Chandrayaan-4 in 2026."
    res_b = agent.process_input(claim_b, retrieval_mode="enhanced", kb_enabled=False)
    v_b = res_b.get("verdict_data", {})
    
    print(f"  Result Verdict: {v_b.get('verdict')}, Confidence: {v_b.get('confidence')}%\n  Explanation: {v_b.get('explanation')}")
    assert v_b.get("verdict") in ["FAKE", "UNCERTAIN"], f"Expected FAKE or UNCERTAIN, got {v_b.get('verdict')}"
    assert v_b.get("verdict") != "REAL", "Completed launch claim in 2026 must NOT be classified REAL when mission is scheduled for 2027"
    print("  [PASSED] TEST B: Temporal mismatch correctly flagged.")
    passed_tests += 1

    # --------------------------------------------------
    # TEST C: True Current Role Verification
    # --------------------------------------------------
    print("\n[TEST C] True Current Role Verification ('Narendra Modi is the prime minister of India')...")
    claim_c = "Narendra Modi is the prime minister of India."
    res_c = agent.process_input(claim_c, retrieval_mode="enhanced", kb_enabled=False)
    v_c = res_c.get("verdict_data", {})
    
    print(f"  Result Verdict: {v_c.get('verdict')}, Confidence: {v_c.get('confidence')}%\n  Explanation: {v_c.get('explanation')}")
    assert v_c.get("verdict") == "REAL", f"Expected REAL for true current role, got {v_c.get('verdict')}"
    assert v_c.get("confidence") >= 80, f"Expected high confidence >= 80%, got {v_c.get('confidence')}%"
    print("  [PASSED] TEST C: True current role correctly classified as REAL.")
    passed_tests += 1

    # --------------------------------------------------
    # TEST D: Genuine Uncertainty Handling
    # --------------------------------------------------
    print("\n[TEST D] Genuine Uncertainty Handling (Obscure unverified rumor)...")
    claim_d = "Secret underwater city discovered beneath Lake Baikal in July 2026."
    res_d = agent.process_input(claim_d, retrieval_mode="enhanced", kb_enabled=False)
    v_d = res_d.get("verdict_data", {})
    
    print(f"  Result Verdict: {v_d.get('verdict')}, Confidence: {v_d.get('confidence')}%\n  Explanation: {v_d.get('explanation')}")
    assert v_d.get("verdict") in ["UNCERTAIN", "FAKE"], f"Expected UNCERTAIN or FAKE, got {v_d.get('verdict')}"
    print("  [PASSED] TEST D: Unverifiable rumor correctly classified.")
    passed_tests += 1

    # --------------------------------------------------
    # TEST E: Paraphrased Claim KB Hit
    # --------------------------------------------------
    print("\n[TEST E] Paraphrased Claim Semantic KB Hit...")
    agent.kb.store_fact_check(
        original_claim="Did India ban the 2000 rupee note in 2023?",
        verdict="REAL",
        confidence=95,
        explanation="RBI announced withdrawal of 2000 rupee notes in May 2023.",
        evidence_snippets=["RBI withdraws 2000 notes"],
        source_urls=["https://rbi.org.in"],
        source_titles=["RBI Notification"],
        source_domains=["rbi.org.in"],
        source_credibility_info=[{"domain": "rbi.org.in", "score": 1.0, "category": "official"}],
        publication_dates=["2023-05-19"],
        search_queries_used=["RBI 2000 rupee notes"],
        evidence_selected=[]
    )
    
    claim_e = "Is it true that India banned ₹2000 currency notes in 2023?"
    res_e = agent.process_input(claim_e, retrieval_mode="enhanced", kb_enabled=True)
    assert res_e["retrieval_details"]["kb_reused"] == True, "Expected semantic KB hit for paraphrased claim"
    print("  [PASSED] TEST E: Paraphrased claim recognized & reused from KB.")
    passed_tests += 1

    # --------------------------------------------------
    # TEST F: Entity Conflict Prevention
    # --------------------------------------------------
    print("\n[TEST F] Entity Mismatch Prevention (₹2000 vs ₹500 notes)...")
    claim_f = "India discontinued ₹500 notes in 2026."
    res_f = agent.process_input(claim_f, retrieval_mode="enhanced", kb_enabled=True)
    assert res_f["retrieval_details"]["kb_reused"] == False, "Distinct numeric entity (500 vs 2000) must NOT match KB entry"
    print("  [PASSED] TEST F: Entity mismatch correctly prevented false KB match.")
    passed_tests += 1

    # --------------------------------------------------
    # TEST G: Security & Privacy Check
    # --------------------------------------------------
    print("\n[TEST G] Security & Privacy Check...")
    kb_json_str = json.dumps(agent.kb.metadata_store)
    openrouter_key = os.getenv("OPENROUTER_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if openrouter_key:
        assert openrouter_key not in kb_json_str, "API key found in KB storage!"
    if tavily_key:
        assert tavily_key not in kb_json_str, "API key found in KB storage!"
    print("  [PASSED] TEST G: No secrets exposed in KB metadata.")
    passed_tests += 1

    print("\n==================================================")
    print(f"ALL {passed_tests} VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

if __name__ == "__main__":
    run_all_tests()
