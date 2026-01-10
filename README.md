# 🛡️ Veritas AI | Agentic Fake News Detector

> **A Next-Gen Fact-Checking Agent that combines LLM Reasoning with Real-Time Web Search.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![OpenRouter](https://img.shields.io/badge/AI-Claude%203%20Haiku-orange)
![Status](https://img.shields.io/badge/Status-Active-success)

Veritas AI is not just a text classifier. Unlike traditional "static" machine learning models that are limited to their training data, Veritas uses an **Agentic RAG (Retrieval-Augmented Generation)** architecture. It autonomously searches the live web for evidence and uses an LLM to "reason" about the truthfulness of a claim in real-time.

## 🚀 Key Features

* **🕵️ Agentic Workflow:** Uses **Claude-3-Haiku** (via OpenRouter) as a "Reasoning Engine" to understand user intent (Chat vs. Fact-Check).
* **🌍 Live Web Search (RAG):** Integrated **DuckDuckGo Search** to fetch real-time articles, allowing the system to debunk rumors that started *5 minutes ago*.
* **🧠 Hybrid Verification:**
    * **Context:** Checks live sources to verify *facts*.
    * **Logic:** Uses LLM reasoning to analyze *clickbait patterns* and *logical fallacies*.
* **📊 Live Session Stats:** Real-time dashboard that tracks how many articles you have verified in the current session.

## ⚙️ How It Works (The Architecture)

The system follows a **"Runner & Judge"** pattern to overcome the knowledge cutoff limits of standard AI.

1.  **Input:** User provides a headline (e.g., *"Aliens landed in New York today"*).
2.  **The Runner (DuckDuckGo):** The system autonomously scrapes the latest articles from trusted web sources matching the claim.
3.  **The Judge (LLM Agent):** The AI reads the scraped search results and compares them against the user's claim.
    * *If evidence matches:* Verdict is **REAL**.
    * *If evidence contradicts:* Verdict is **FAKE**.
4.  **Output:** Returns a color-coded verdict, a confidence score, and a summary explaining *why* it is fake, citing the missing evidence.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python)
* **Intelligence:** OpenRouter API (Claude-3 / Mistral / Llama 3)
* **Search Engine:** `duckduckgo-search` (Privacy-focused live web scraping)
* **Logic:** Custom Python Agent with Intent Recognition

## 📦 Installation & Setup

1.  **Clone the Repo:**
    ```bash
    git clone [https://github.com/sriram263/Fakenewsdetector.git](https://github.com/sriram263/Fakenewsdetector.git)
    cd Fakenewsdetector
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up API Keys:**
    Create a `.env` file in the root directory and add your OpenRouter key:
    ```env
    OPENROUTER_API_KEY=your_api_key_here
    ```

4.  **Run the App:**
    ```bash
    streamlit run main.py
    ```

## 🤝 Contributing
Feel free to fork this project and submit pull requests. Suggestions for adding more search tools (like Google Serper or Tavily) are welcome!

---
*Built with ❤️ by Sriram*