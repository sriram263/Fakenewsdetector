# VERITAS AI v2.1 — VERIFICATION & RENDERING BUG FIX REPORT

---

## 1. Exact Cause of the Sriram / Prime Minister Misclassification
In v2.0, the LLM prompt simply asked: *"Verify the claim using the SEARCH RESULTS above. Output UNCERTAIN if evidence is insufficient."*
When presented with the claim `"Sriram is the Prime Minister of India"` alongside search results stating *"Narendra Modi is the Prime Minister of India"*, the naive LLM prompt failed to perform explicit stance analysis. Because the text snippet did not contain the exact phrase *"Sriram is not the prime minister"*, the LLM defaulted to assuming the evidence was merely "topically related but missing direct info about Sriram", resulting in `UNCERTAIN (50%)`.

---

## 2. Exact Changes Made to Claim Verification
We introduced an explicit 3-stage **Claim-Evidence Verification Layer** in [`verifier.py`](file:///c:/My_Learning/FND%20Modified/verifier.py):
1. **Claim Analysis:** Categorizes claims generically into `CURRENT_ROLE`, `TEMPORAL`, `EVENT_OCCURRENCE`, `NUMERICAL`, or `GENERAL_FACT`.
2. **Stance Evaluation (`evaluate_evidence_relationships`):** For each retrieved source, explicitly classifies the relationship as `SUPPORTS`, `REFUTES`, or `INSUFFICIENT`.
3. **Synthesis & Calibration (`synthesize_verdict_and_confidence`):** Synthesizes the verdict (`REAL`, `FAKE`, `UNCERTAIN`) and calculates confidence based on stance consensus and domain credibility scores.

---

## 3. How SUPPORTS / REFUTES / INSUFFICIENT Works
- **`SUPPORTS`**: Retrieved evidence explicitly confirms the claimed entity, role, date, or event occurrence.
- **`REFUTES`**: Retrieved evidence directly contradicts the claim:
  - *Exclusive Roles:* If evidence identifies *Person B* holding an exclusive office (e.g. Narendra Modi is PM), the claim that *Person A* (Sriram) holds it is classified as `REFUTES`.
  - *Temporal/Status Mismatches:* If a claim states an event was completed in Year X, but evidence shows it is only planned/scheduled for Year Y (or completed in Year Z), it is classified as `REFUTES`.
  - *Official Debunks:* If official sources refute or deny a claim, it is classified as `REFUTES`.
- **`INSUFFICIENT`**: Evidence is topically related (mentions the entity or general topic) but does not establish or contradict the specific claimed attribute.

---

## 4. How Current-Role Claims Are Handled
For `CURRENT_ROLE` claims (e.g., *"X is the Prime Minister of India"*, *"X is the CEO of Y"*):
1. `verifier.py` identifies the role and target organization/country generically without hard-coding names.
2. Retrieved current authoritative sources are inspected for office-holder identification.
3. If reliable sources identify a different office-holder, the relationship is marked `REFUTES`, resulting in a verdict of `FAKE` with high confidence (85%–95%).

---

## 5. How Temporal Claims Are Handled
For `TEMPORAL` claims (e.g., *"India launched Chandrayaan-4 in 2026"*):
1. The claim category is identified as `TEMPORAL`.
2. The verifier evaluates event status terms (*launched / completed / occurred*) versus (*planned / scheduled / proposed / expected / 2027*).
3. If evidence shows the mission is scheduled for a future year (2027), the completed launch claim for 2026 is marked `REFUTES` → `FAKE` (85% confidence).

---

## 6. How Confidence Is Calculated
Confidence is calibrated empirically from evidence stance consensus and domain credibility scores:
- **Strong Refutation (Refuting sources > 0, Supporting = 0):** Confidence = $80\% - 95\%$ (scaling with max refuting domain credibility).
- **Strong Support (Supporting sources > 0, Refuting = 0):** Confidence = $80\% - 95\%$ (scaling with max supporting domain credibility).
- **Conflicting Evidence (Both supporting and refuting sources present):** Confidence = $50\%$.
- **Insufficient Evidence (No direct support or refutation):** Confidence = $30\% - 40\%$.

---

## 7. How KB Records Are Protected from Old Incorrect Verdicts
- **Version Tagging:** All KB entries now include `"verification_version": "v2.1"`.
- **Automatic Outdating:** When `search_similar_fact_check()` runs in [`knowledge_base.py`](file:///c:/My_Learning/FND%20Modified/knowledge_base.py), any entry lacking `"v2.1"` is flagged as `STALE_HIT`, forcing fresh web retrieval and re-verification under the new verifier logic.
- **Manual Reset:** Added a **"🧹 Clear KB Memory"** button in the sidebar and `clear_kb()` in backend.

---

## 8. Exact Cause of the HTML Rendering Bug
In [`main.py`](file:///c:/My_Learning/FND%20Modified/main.py), the verdict box was constructed as raw HTML strings (`<div class="...">...</div>`) and passed to `st.markdown()`. Streamlit's markdown renderer often escapes multiline HTML strings or treats indented HTML lines as code blocks, displaying raw HTML tags as text on screen.

---

## 9. Exact Rendering Line / Function Changed
In [`main.py`](file:///c:/My_Learning/FND%20Modified/main.py):
- Replaced multiline raw HTML strings with `render_verdict_card()` using **native Streamlit UI components**:
  - `st.success()` for `REAL` verdict
  - `st.error()` for `FAKE` verdict
  - `st.warning()` for `UNCERTAIN` verdict
  - `st.container()` and `st.caption()` for clean, native UI rendering.

---

## 10. Files Modified
- [`main.py`](file:///c:/My_Learning/FND%20Modified/main.py)
- [`agent.py`](file:///c:/My_Learning/FND%20Modified/agent.py)
- [`knowledge_base.py`](file:///c:/My_Learning/FND%20Modified/knowledge_base.py)
- [`test_veritas_v2.py`](file:///c:/My_Learning/FND%20Modified/test_veritas_v2.py)
- [`README.md`](file:///c:/My_Learning/FND%20Modified/README.md)

## 11. Files Created
- [`verifier.py`](file:///c:/My_Learning/FND%20Modified/verifier.py)
- [`veritas_ai_v2_1_report.md`](file:///c:/My_Learning/FND%20Modified/veritas_ai_v2_1_report.md)

---

## 12. Tests Executed & Results

| Test | Claim Tested | Verdict | Confidence | Stance Evaluation | Test Result |
| :--- | :--- | :---: | :---: | :--- | :---: |
| **TEST A** | *"The prime minister of India is Sriram."* | **FAKE** | **95%** | REFUTES (Modi is PM) | **[PASSED]** |
| **TEST B** | *"India launched Chandrayaan-4 in 2026."* | **FAKE** | **85%** | REFUTES (Scheduled 2027) | **[PASSED]** |
| **TEST C** | *"Narendra Modi is the prime minister of India."* | **REAL** | **95%** | SUPPORTS (Confirmed) | **[PASSED]** |
| **TEST D** | *"Secret underwater city in Lake Baikal in 2026."* | **UNCERTAIN** | **40%** | INSUFFICIENT (No direct info) | **[PASSED]** |
| **TEST E** | Paraphrased Claim KB Hit | **REAL** | **95%** | Reused from v2.1 KB | **[PASSED]** |
| **TEST F** | Entity Mismatch (₹2000 vs ₹500 notes) | **N/A** | **N/A** | KB Miss (Entity Conflict) | **[PASSED]** |
| **TEST G** | Security Check (No secrets in KB) | **N/A** | **N/A** | No API keys logged | **[PASSED]** |

---

## 13. Remaining Limitations
1. **Unlisted Domain Fallback:** Unlisted obscure blogs fallback to `general` domain credibility (0.60); custom domain rules can be registered in [`config.py`](file:///c:/My_Learning/FND%20Modified/config.py).
2. **Date Extraction:** Web articles lacking metadata fall back to neutral freshness scores (0.60).
