# 🛡️ Technical Audit & System Architecture Documentation
## Veritas AI | Agentic Fake News Detector

> **Document Version:** 1.0  
> **Target Audience:** Researchers, AI Engineers, Peer Reviewers  
> **Inspection Scope:** Full Codebase (`main.py`, `agent.py`, `requirements.txt`, `.env`, `README.md`)

---

## 1. Executive Technical Summary

**Veritas AI** is an Agentic Retrieval-Augmented Generation (RAG) system built for real-time fake news detection and fact-checking. Unlike static machine learning classifiers (e.g., Naive Bayes, SVM, or traditional BERT models trained on fixed corpora), Veritas AI dynamically fetches real-time web context via the **Tavily AI Search API** and passes retrieved news snippets into an LLM reasoning engine (**Claude 3 Haiku** via **OpenRouter API**) to produce automated truthfulness verdicts (`REAL`, `FAKE`, `UNCERTAIN`), confidence scores, and structured explanations.

---

## 2. Complete Technical Inspection (50 Detailed Audit Points)

### 1. Complete Project Architecture
The project follows a single-node, client-server web architecture running within Python:
* **Presentation Layer:** [main.py](file:///c:/My_Learning/FND%20Modified/main.py) built on **Streamlit** for real-time web interaction, session state tracking, CSS custom card rendering, and log exportation.
* **Agent / Controller Layer:** [agent.py](file:///c:/My_Learning/FND%20Modified/agent.py) (`SmartAgent` class) managing interaction flow between search tools and LLM inference engines.
* **External Tooling & APIs:**
  1. **Tavily AI Search API:** Real-time web retrieval for live news context.
  2. **OpenRouter API Proxy:** Interfacing with Anthropic Claude 3 Haiku (`anthropic/claude-3-haiku`) using OpenAI Python SDK.

### 2. Every Folder and File and its Purpose
* **[main.py](file:///c:/My_Learning/FND%20Modified/main.py)** (254 lines): Streamlit application entry point. Defines UI layout, CSS styles (`.real-box`, `.fake-box`, `.uncertain-box`), session state (`agent`, `messages`, `stats`), sidebar metrics, chat history rendering, regex parsing of LLM outputs, source link visualization, timezone handling, and plain-text chat log export.
* **[agent.py](file:///c:/My_Learning/FND%20Modified/agent.py)** (102 lines): Core intelligence controller. Initializes `OpenAI` client pointing to OpenRouter base URL (`https://openrouter.ai/api/v1`) and `TavilyClient`. Defines `SmartAgent` class with methods `get_live_info(query)` and `process_input(user_text)`. Constructs prompt templates and manages temperature/model parameters.
* **[requirements.txt](file:///c:/My_Learning/FND%20Modified/requirements.txt)** (58 lines): Lists exact pinned dependencies including `streamlit==1.52.2`, `openai==2.15.0`, `tavily-python`, `python-dotenv==1.2.1`, `pandas==2.3.3`, `beautifulsoup4==4.14.3`, `pytz==2025.2`.
* **[.env](file:///c:/My_Learning/FND%20Modified/.env)** (6 lines): Environment configuration file storing credentials (`OPENROUTER_API_KEY`, `CLAUDE_API_URL`, `TAVILY_API_KEY`).
* **[README.md](file:///c:/My_Learning/FND%20Modified/README.md)** (69 lines): Project description, feature list, runner-and-judge architecture overview, and setup/installation instructions.
* **`Veritas_AI_Presentation.pptx`**: Slide presentation file for project demonstration.
* **[.gitignore](file:///c:/My_Learning/FND%20Modified/.gitignore)**: Git ignore rules for `venv/`, `__pycache__/`, `.env`, `*.pkl`, `.DS_Store`.
* **`venv/`**: Python virtual environment.
* **`__pycache__/`**: Cached Python bytecode.

### 3. Frontend Technology and Implementation
* **Framework:** Streamlit (v1.52.2).
* **Custom Styling:** Injected HTML/CSS via `st.markdown("<style>...</style>", unsafe_allow_html=True)` ([main.py:L16-L50](file:///c:/My_Learning/FND%20Modified/main.py#L16-L50)).
* **Dynamic Card Styling:** 
  * `REAL` verdict: Green box (`background-color: #d1fae5; border-left: 6px solid #10b981`).
  * `FAKE` verdict: Red box (`background-color: #fee2e2; border-left: 6px solid #ef4444`).
  * `UNCERTAIN` verdict: Amber box (`background-color: #fef3c7; border-left: 6px solid #f59e0b`).
* **Interactivity:** Streamlit chat input (`st.chat_input`), expandable panels (`st.expander`), metric boxes (`st.metric`), log download button (`st.download_button`), session reset button (`st.button`).

### 4. Backend Technology and Implementation
* **Language:** Python 3.10+.
* **Libraries:** `openai` SDK mapped to OpenRouter REST endpoint, `tavily-python` SDK, `python-dotenv`, `re`, `json`, `pytz`, `datetime`, `bs4` (BeautifulSoup4).
* **Design Pattern:** Runner & Judge (Tool Calling / Context Injection before Generation).

### 5. Complete Data Flow from User Input to Final Result
1. **User Query:** User enters a news headline or query into Streamlit input (`prompt`).
2. **State Appending:** Input added to `st.session_state.messages` with role `"user"`.
3. **Agent Invocation:** `st.session_state.agent.process_input(prompt)` is called synchronously ([main.py:L178](file:///c:/My_Learning/FND%20Modified/main.py#L178)).
4. **Live Search Call:** `get_live_info()` takes `user_text[:300]` and queries Tavily API for top 5 news articles ([agent.py:L49](file:///c:/My_Learning/FND%20Modified/agent.py#L49)).
5. **Prompt Synthesis:** Search results are JSON-serialized via `json.dumps()` and embedded into `system_prompt` ([agent.py:L52-L80](file:///c:/My_Learning/FND%20Modified/agent.py#L52-L80)).
6. **LLM Inference:** `client.chat.completions.create()` invokes `anthropic/claude-3-haiku` with temperature `0.3` ([agent.py:L84-L91](file:///c:/My_Learning/FND%20Modified/agent.py#L84-L91)).
7. **Regex Extraction:** [main.py:L183](file:///c:/My_Learning/FND%20Modified/main.py#L183) parses `<VERDICT_JSON>(.*?)</VERDICT_JSON>` from raw text output.
8. **UI Rendering & Metric Update:** `main.py` extracts `verdict`, `confidence`, `explanation`, increments session statistics, constructs HTML report card, and lists sources.
9. **State Storage:** Output and raw metadata saved to `st.session_state.messages`.

### 6. How Fake-News Detection Currently Works
The system evaluates truthfulness using **Zero-Shot LLM Reasoning grounded on Live Web Retrieval**. It compares the input statement against top live news sources retrieved from Tavily to check for empirical agreement or contradiction.

### 7. Exact ML/DL/LLM Models Being Used
* **LLM Engine:** `anthropic/claude-3-haiku` (accessed through OpenRouter proxy).
* **Search / Retrieval Model:** Managed ranking algorithms inside Tavily AI's Search Engine.
* *Note:* No locally trained ML/DL models (e.g. Scikit-learn classifiers, PyTorch/TensorFlow networks, or local Transformer embeddings) exist in the codebase.

### 8. Exact RAG Implementation
* **Architecture:** Ephemeral Web-RAG (Fetch-on-Demand Web Search).
* **Mechanism:** Direct prompt injection of raw JSON search payloads returned from Tavily API. No persistent vector indexing or local embedding generation is performed.

### 9. Where Knowledge Base/Documents Come From
Dynamic live web crawling via Tavily AI Search API configured for `topic="news"` and looking back `days=365`. There is no static local knowledge base.

### 10. How Documents Are Processed
Search hits from Tavily return list of dictionaries with keys (`title`, `url`, `content`, `score`). The system converts these directly to a string using `json.dumps(search_results)` without secondary cleaning or filtering.

### 11. Chunking Strategy
**Not Applicable / None.** The system relies entirely on pre-summarized `content` snippets returned by Tavily API.

### 12. Embedding Model
**Not Applicable / None.** No text embedding models (e.g. OpenAI `text-embedding-3`, HuggingFace sentence-transformers) are initialized or used.

### 13. Vector Database / Vector Store
**Not Applicable / None.** No vector storage solution (e.g., ChromaDB, FAISS, Pinecone) is integrated.

### 14. Retrieval Mechanism
REST API call via `TavilyClient.search()` with parameter `query=user_text[:300]`, `topic="news"`, `days=365`, `max_results=5`, `include_answer=True`.

### 15. Number of Retrieved Documents/Chunks (Top-K)
Hardcoded to **`max_results=5`** ([agent.py:L31](file:///c:/My_Learning/FND%20Modified/agent.py#L31)).

### 16. Reranking, if any
**None.** Relies solely on Tavily API's default search ranking order.

### 17. Prompt Templates Used
Defined in [agent.py:L52-L80](file:///c:/My_Learning/FND%20Modified/agent.py#L52-L80). Instructs the model to act as `'Veritas', an advanced AI News Analyst`, evaluates provided `SEARCH RESULTS`, and enforces two modes:
* `MODE 1: CASUAL CHAT` -> Plain text output for greetings.
* `MODE 2: NEWS VERIFICATION` -> Enforces `<VERDICT_JSON>` tags with keys `type`, `verdict`, `confidence`, `explanation`, `sources`.

### 18. LLM Used
`anthropic/claude-3-haiku` via OpenRouter.

### 19. LLM Parameters/Configuration
* `model`: `"anthropic/claude-3-haiku"`
* `temperature`: `0.3`
* `messages`: System prompt + User input text.
* `base_url`: `"https://openrouter.ai/api/v1"`

### 20. How Retrieved Evidence is Passed to the LLM
Injected directly into System Prompt formatted as raw JSON under `SEARCH RESULTS: {json.dumps(search_results)}`.

### 21. How Final Fake/Real Decision is Produced
Claude-3 Haiku generates JSON content containing `"verdict": "REAL" | "FAKE" | "UNCERTAIN"`. [main.py](file:///c:/My_Learning/FND%20Modified/main.py) parses this tag via regular expressions and assigns appropriate UI formatting and counter increments.

### 22. Explanations / Evidence / Citations
* **Explanations:** Provided via `explanation` field in LLM response JSON.
* **Citations:** Provided via expandable UI code box listing source titles and URLs ([main.py:L233-L238](file:///c:/My_Learning/FND%20Modified/main.py#L233-L238)).

### 23. How Hallucination is Handled
Grounding via injected search context and low temperature (`0.3`). *Limitation:* No automated entailment scoring or self-consistency checking is present.

### 24. How Conflicting Sources Are Handled
Delegated entirely to zero-shot LLM reasoning within Claude-3 Haiku. No local domain weighting or algorithmic conflict resolution exists.

### 25. Dataset(s) Used
**None.** No static benchmark datasets (e.g. WELFake, LIAR, ISOT) are included or queried.

### 26. Dataset Size and Classes
**Not Applicable.** Dynamically produces 3 target classes: `REAL`, `FAKE`, `UNCERTAIN`.

### 27. Train/Validation/Test Split
**Not Applicable.** No supervised training or fine-tuning is performed.

### 28. Preprocessing
* String truncation of user input: `user_text[:300]`.
* Regex parsing of response string: `re.search(r'<VERDICT_JSON>(.*?)</VERDICT_JSON>', ...)`
* HTML tag stripping using BeautifulSoup for plain-text log export ([main.py:L107](file:///c:/My_Learning/FND%20Modified/main.py#L107)).

### 29. Evaluation Metrics
**None.** No evaluation scripts or benchmark pipelines exist in codebase.

### 30. Current Accuracy / F1 / Precision / Recall
**Not determined from the available project files.** (No benchmark evaluations have been run or logged).

### 31. Deployment Architecture
Monolithic desktop/local deployment. Ran locally via command `streamlit run main.py`.

### 32. APIs / Services Used
1. OpenRouter API (LLM proxy)
2. Tavily AI API (News search engine)

### 33. Database / Storage
* **Session Storage:** In-memory `st.session_state`.
* **Export:** Text file download (`veritas_log.txt`). No SQL/NoSQL database used.

### 34. Authentication
* **User Level:** None (open web UI).
* **API Level:** Key-based authentication via `.env` file (`OPENROUTER_API_KEY`, `TAVILY_API_KEY`).

### 35. Error Handling
* `get_live_info()` catches exceptions and returns an empty list `[]` ([agent.py:L35-L36](file:///c:/My_Learning/FND%20Modified/agent.py#L35-L36)).
* `process_input()` catches LLM invocation errors and returns error message string ([agent.py:L98-L101](file:///c:/My_Learning/FND%20Modified/agent.py#L98-L101)).
* `main.py` fallback handles JSON parsing exceptions gracefully ([main.py:L222-L223](file:///c:/My_Learning/FND%20Modified/main.py#L222-L223)).

### 36. Logging
Basic string list compilation in `generate_chat_log()` for export. No formal Python `logging` or telemetry framework configured.

### 37. Caching
**None.** Every input triggers fresh network search and LLM API calls.

### 38. Latency / Performance Characteristics
* **Latency:** ~2 to 5 seconds per query (dependent on network call to Tavily + OpenRouter API response speed).
* **Execution:** Synchronous blocking on main Streamlit thread.

### 39. Hardware / Software Requirements
* **Hardware:** Standard CPU workstation (2+ GB RAM), Internet connectivity. No GPU needed.
* **Software:** Python 3.10+, modern web browser.

### 40. Exact Python/Framework/Library Versions
* Python 3.10+
* `streamlit==1.52.2`
* `openai==2.15.0`
* `tavily-python` (version unpinned in `requirements.txt`)
* `python-dotenv==1.2.1`
* `beautifulsoup4==4.14.3`
* `pytz==2025.2`
* `pandas==2.3.3`

### 41. Environment Variables / Configuration
* `OPENROUTER_API_KEY`: Auth token for OpenRouter.
* `TAVILY_API_KEY`: Auth token for Tavily Search.
* `CLAUDE_API_URL`: API URL (`https://openrouter.ai/api/v1/chat/completions`).

### 42. Code Authorship (Human vs. AI Generated)
Contains hybrid structure: standard Streamlit boilerplate and Tavily template patterns combined with custom user state handlers, sidebar logic, and regex output parsing.

### 43. Current Limitations of the Implementation
1. **Inefficient Search Execution:** Tavily search executes on line 49 *before* LLM determines if input is a greeting or news query, wasting API quota on casual chat.
2. **Ephemeral State:** All session state lost on browser reload.
3. **Hardcoded Limits:** Search query capped at `300` chars; results capped at `5`.
4. **Brittle Output Parsing:** Relies on regex matching `<VERDICT_JSON>` rather than strict schema enforcement.
5. **No Local Dataset / Offline Verification:** 100% reliant on external network connectivity and active paid API keys.

### 44. Unobvious Implemented Features
* **Dual Intent Processing in System Prompt:** Handles greetings vs. factual news checks inside a single prompt template.
* **HTML Sanitization on Log Download:** Uses BeautifulSoup to strip HTML styling from chat history when downloading `veritas_log.txt`.
* **Dynamic Timezone Selection:** Adjusts displayed local time using `pytz`.

### 45. Partially Implemented Features
* **Intent Recognition:** Formulated in prompt instructions, but executed *after* API tools have already been triggered.
* **Source Metadata Display:** Titles and URLs are shown, but search relevance scores and article publication dates are omitted from UI.

### 46. Features Planned but NOT Implemented
* Multi-model benchmark selection toggle.
* Local persistent vector store / PDF article uploader.
* Offline BERT / Fine-tuned baseline models.
* Automated accuracy/F1 evaluation suite against benchmark datasets.

### 47. Technical Weaknesses
* **API Cost Inefficiency:** Search API call fired unconditionally for every input.
* **Lack of Text Chunking / Indexing:** Dumps raw JSON array directly into system prompt.
* **Single Point of Failure:** Total dependency on third-party APIs (Tavily & OpenRouter).

### 48. Assumptions Made by System
* Assumes Tavily top 5 search hits represent objective truth.
* Assumes user input under 300 characters captures full context of headline/claim.
* Assumes LLM strictly outputs valid XML tags and valid JSON.

### 49. Realistic Replacement / Improvement Candidates
1. **Intent Gatekeeper:** Place intent classification *before* web search tool call.
2. **Hybrid RAG System:** Combine static vector database (ChromaDB/FAISS containing verified dataset facts) with web search RAG.
3. **Structured Outputs:** Replace regex string matching with Pydantic output validation.
4. **Local LLM / SLM Support:** Add fallback support for local Ollama / LLaMA-3 models.

### 50. Complete End-to-End Workflow Sequence
1. Streamlit application loads (`main.py`) and initializes `SmartAgent` instance into `st.session_state`.
2. User selects timezone in sidebar (default: `Asia/Kolkata`).
3. User enters prompt in `st.chat_input`.
4. User message rendered to chat container with current timestamp.
5. `SmartAgent.process_input()` invoked.
6. `SmartAgent` slices prompt to first 300 characters and submits synchronous API request to Tavily AI.
7. Tavily searches `topic="news"` for past 365 days and returns top 5 results.
8. `SmartAgent` formats system prompt containing JSON string of search results and two mode instructions.
9. `SmartAgent` calls OpenRouter API invoking `anthropic/claude-3-haiku` with temperature `0.3`.
10. OpenRouter streams response back to `SmartAgent`.
11. `main.py` receives response dictionary.
12. `main.py` executes Regex check for `<VERDICT_JSON>` tags.
13. If tag is found:
    * Parses JSON object (`verdict`, `confidence`, `explanation`).
    * Updates sidebar metrics counters (`checked`, `real`, `fake`).
    * Formats colored HTML card (`.real-box`, `.fake-box`, or `.uncertain-box`).
    * Renders collapsible source panel with title & URL formatting.
14. If tag is missing (Casual Chat):
    * Displays raw text response.
15. Assistant message saved to `st.session_state.messages`.
16. Sidebar metrics re-rendered.

---

## Required Summary Sections (A to Q)

### A. PROJECT OVERVIEW
Veritas AI is an agentic, web-augmented fake news detection tool designed to combat online misinformation by verifying live headlines against real-time web search results using LLM reasoning.

### B. ARCHITECTURE
Client-Server Runner & Judge Pattern. Streamlit UI frontend -> Python `SmartAgent` controller -> Tavily Search API (Runner) -> OpenRouter Claude 3 Haiku LLM (Judge).

### C. COMPLETE TECHNICAL STACK
* **Frontend:** Streamlit 1.52.2, HTML5/CSS3.
* **Backend:** Python 3.10+, OpenAI Python SDK (v2.15.0), Tavily Python SDK.
* **External APIs:** OpenRouter API, Tavily AI Search API.
* **Utilities:** BeautifulSoup4, Pytz, Python-dotenv, Pandas.

### D. END-TO-END DATA FLOW
`User Input` -> `Streamlit UI` -> `SmartAgent.process_input()` -> `Tavily Search API (top 5 news)` -> `System Prompt Construction (JSON Injection)` -> `OpenRouter Claude 3 Haiku API` -> `Regex Tag Extractor` -> `HTML Report Card & Source UI` -> `Session State`.

### E. RAG PIPELINE
Ephemeral Web RAG pipeline. Live web query via Tavily -> JSON string serialization -> System prompt context injection -> LLM evaluation. No persistent vector storage or embedding models used.

### F. FAKE NEWS DETECTION PIPELINE
Dynamic verification via LLM zero-shot factual comparison against live web news context. Output classification into `REAL`, `FAKE`, or `UNCERTAIN` accompanied by confidence rating and rationale.

### G. CURRENT MODEL(S)
* **Reasoning Model:** Anthropic Claude 3 Haiku (`anthropic/claude-3-haiku`).
* **Retrieval Model:** Tavily AI News Search Engine.

### H. DATASET AND EVALUATION
* **Dataset:** None (uses live web search data).
* **Evaluation Metrics:** Not implemented in current codebase.

### I. DEPLOYMENT
Local workstation execution (`streamlit run main.py`). Monolithic execution model.

### J. CURRENT IMPLEMENTED FEATURES
* Real-time web news retrieval (Tavily AI).
* Dual-mode processing (Casual Chat vs. News Verification).
* Dynamic HTML report cards with verdict color-coding.
* Confidence score generation.
* Related source link extraction and display.
* Session statistics dashboard (Checked, Real, Fake counters).
* Export chat log to text file (`veritas_log.txt`) with HTML sanitization.
* Timezone selector for localized timestamps.

### K. PARTIAL FEATURES
* Intent recognition (prompt-instructed, but tool execution runs prior to intent evaluation).
* Source metadata visualization (displays title and URL, omits relevance score/date).

### L. UNIMPLEMENTED FEATURES
* Local dataset evaluation suite (WELFake, LIAR benchmark tests).
* Persistent vector database / local document uploader.
* Offline fine-tuned ML models (e.g. RoBERTa/BERT).
* Multi-model selection dropdown in UI.
* User authentication and database persistence.

### M. CURRENT LIMITATIONS
* Fires search API calls unconditionally on every message (wasteful for casual greetings).
* Relying on regular expressions (`<VERDICT_JSON>`) can lead to parsing errors if LLM output varies.
* 300-character claim truncation limit.
* Complete vulnerability to third-party API downtime or rate limits.

### N. POTENTIAL IMPROVEMENT AREAS
* Implement an intent routing gatekeeper *before* triggering search APIs.
* Enforce Pydantic / OpenAI Structured Outputs.
* Build a hybrid retrieval system (Local Vector Store + Web Search RAG).
* Add automated offline evaluation scripts with standard confusion matrices.

### O. PROJECT STRENGTHS
* Zero knowledge cutoff issues (handles breaking news events from minutes ago).
* Clean, interactive visual user interface with session management.
* Concise explainability with direct evidence citations.
* Lightweight codebase requiring no local GPU resources.

### P. PROJECT WEAKNESSES
* Inefficient API utilization (pre-search before classification).
* Lack of offline operation capability or local dataset verification.
* Absence of quantitative evaluation metrics (Precision/Recall/F1).

### Q. COMPLETE FILE-BY-FILE DESCRIPTION

| File Path | Lines | Primary Purpose |
| :--- | :--- | :--- |
| [main.py](file:///c:/My_Learning/FND%20Modified/main.py) | 254 | Streamlit UI entry point, state management, CSS styling, metric dashboard, log exporter, regex output parser. |
| [agent.py](file:///c:/My_Learning/FND%20Modified/agent.py) | 102 | SmartAgent logic, Tavily search invocation, prompt template builder, OpenRouter API call wrapper. |
| [requirements.txt](file:///c:/My_Learning/FND%20Modified/requirements.txt) | 58 | Pinned Python dependency list. |
| [.env](file:///c:/My_Learning/FND%20Modified/.env) | 6 | Environment variables configuration (API keys). |
| [README.md](file:///c:/My_Learning/FND%20Modified/README.md) | 69 | High-level project documentation, architecture overview, installation instructions. |

---

## INFORMATION REQUIRED TO FIND AN IEEE ACCESS BASE PAPER

To identify a suitable **IEEE Access base paper** for your college mini-project defense and research expansion, your paper search should target publications in **Agentic RAG, Web-Augmented Fact-Checking, and Real-Time Misinformation Detection**.

### Core Technical Characteristics of Current Project:
1. **Architecture:** Agentic Web Retrieval-Augmented Generation (Web-RAG) for real-time claim verification using LLM reasoning (Claude-3 Haiku) and external search APIs (Tavily AI).
2. **Primary Advantage:** Overcomes knowledge-cutoff constraints of static fine-tuned classifiers (like BERT/RoBERTa) by retrieving live web evidence.
3. **Current Gaps / Limitations to Target in Base Paper:**
   * **Lack of Hybrid Verification:** Current project relies purely on web retrieval without local benchmark dataset indexing or fine-tuned stance detection.
   * **Unconditional Tool Invocation:** Lacks an upstream Intent Classification Router to prevent redundant web queries.
   * **Lack of Multimodal Verification:** Evaluates text claims only, ignoring fake images/deepfakes.
   * **Absence of Quantitative Benchmark Evaluation:** Lacks automated evaluation against standard datasets (WELFake, LIAR, FEVER).

### Keywords for IEEE Access Literature Search:
* `"Retrieval-Augmented Generation" AND "Fake News Detection"`
* `"Agentic Fact-Checking" AND "Large Language Models"`
* `"Real-Time Misinformation Verification" AND "Web Search"`
* `"Hybrid Fact-Checking" AND "Large Language Models"`
