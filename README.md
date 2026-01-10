# 🛡️ Veritas AI | Agentic Fake News Detector

> **A Next-Gen Fact-Checking Agent that combines LLM Reasoning with Real-Time Web Search.**

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![OpenRouter](https://img.shields.io/badge/AI-Claude%203%20Haiku-orange)
![Tavily](https://img.shields.io/badge/Search-Tavily%20AI-green)
![Status](https://img.shields.io/badge/Status-Active-success)

Veritas AI is not just a text classifier. Unlike traditional "static" machine learning models that are limited to their training data, Veritas uses an **Agentic RAG (Retrieval-Augmented Generation)** architecture. It autonomously searches the live web for high-quality evidence and uses an LLM to "reason" about the truthfulness of a claim in real-time.

## 🚀 Key Features

* **🕵️ Agentic Workflow:** Uses **Claude-3-Haiku** (via OpenRouter) as a "Reasoning Engine" to understand user intent (Chat vs. Fact-Check).
* **🌍 Professional RAG Search:** Integrated **Tavily AI** (built specifically for LLMs) to fetch clean, real-time news, allowing the system to verify breaking news events that happened *minutes ago*.
* **🧠 Hybrid Verification:**
    * **Context:** Checks live sources to verify *facts*.
    * **Logic:** Uses LLM reasoning to analyze *clickbait patterns* and *logical fallacies*.
* **📊 Live Session Stats:** Real-time dashboard that tracks how many articles you have verified in the current session.

## ⚙️ How It Works (The Architecture)

The system follows a **"Runner & Judge"** pattern to overcome the knowledge cutoff limits of standard AI.

1.  **Input:** User provides a headline (e.g., *"India becomes the 4th largest economy"*).
2.  **The Runner (Tavily AI):** The system autonomously hunts for the latest news articles, filtering out low-quality blogs to find trusted sources.
3.  **The Judge (LLM Agent):** The AI reads the scraped search results and compares them against the user's claim.
    * *If evidence matches:* Verdict is **REAL**.
    * *If evidence contradicts:* Verdict is **FAKE**.
4.  **Output:** Returns a color-coded verdict, a confidence score, and a summary explaining *why* it is fake, citing the missing evidence.

## 🛠️ Tech Stack

* **Frontend:** Streamlit (Python)
* **Intelligence:** OpenRouter API (Claude-3 / Mistral / Llama 3)
* **Search Engine:** `tavily-python` (Optimized for RAG & Live News)
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
    Create a `.env` file in the root directory and add your keys:
    ```env
    OPENROUTER_API_KEY=your_claude_api_key
    TAVILY_API_KEY=your_tavily_api_key
    ```

4.  **Run the App:**
    ```bash
    streamlit run main.py
    ```

## 🤝 Contributing
Feel free to fork this project and submit pull requests. Suggestions for adding more agentic tools are welcome!

---
*Built with ❤️ by Sriram*