import streamlit as st
import re
import joblib
import numpy as np
import requests
from bs4 import BeautifulSoup
import os
from dotenv import load_dotenv
from datetime import datetime
from custom_responses import custom_responses
from db import insert_chat

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")

# Load ML model and vectorizer  
model = joblib.load("models/fake_news_model.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")
threshold = joblib.load("models/optimal_threshold.pkl")

CLAUDE_API_URL = os.getenv("CLAUDE_API_URL")
# Claude API
def fetch_claude_fact_check(user_text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "anthropic/claude-3-haiku",
        "messages": [
            {"role": "system", "content": "You are a helpful fact-checking assistant. When a news statement is likely fake, provide what might be true instead."},
            {"role": "user", "content": f"Fact-check this news: {user_text}. What might be true instead?"}
        ]
    }
    try:
        response = requests.post(CLAUDE_API_URL, headers=headers, json=data, timeout=10)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return "⚠️ Claude API failed to respond."
    except Exception as e:
        return f"⚠️ Claude API error: {str(e)}"


# Predefined response helper
theme_emoji = "🧠"
def get_custom_response(user_input):
    user_input = user_input.lower()
    best_match = None
    highest_score = 0

    for key, response in custom_responses.items():
        # Use regex to match whole words (case-insensitive)
        pattern = r'\b' + re.escape(key) + r'\b'
        if re.search(pattern, user_input):
            score = len(key)
            if score > highest_score:
                highest_score = score
                best_match = response

    return best_match

def is_valid_news_input(text):
    # Remove extra whitespace
    text = text.strip()
    # Check if input is empty or less than 5 characters
    if len(text) < 5:
        return False
    # Ensure input has at least 2 alphabetic words (to avoid gibberish)
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
    if len(words) < 2:
        return False
    return True



def extract_title_content(text):
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    meaningful = [l for l in lines if not re.search(r'\b(fake|real|check|news|bot|tell me|is this)\b', l, re.IGNORECASE)]
    title = meaningful[0] if meaningful else ""
    content = " ".join(meaningful[1:]) if len(meaningful) > 1 else ""
    return title, content

# Page config
st.set_page_config(page_title="Fake News Detector", page_icon="🧠")
st.title("📰 Fake News Detector Chatbot")
st.markdown("""
    <div style='background-color:red; padding: 10px; border-left: 5px solid #ffa500; border-radius: 5px; color:white;'>
        ⚠️ <strong>Disclaimer:</strong> This tool is for educational use only. Always verify important news independently.
    </div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
# Sidebar

with st.sidebar:
    st.header("📁 Downloads")

    if "chat_ready" not in st.session_state:
        st.session_state.chat_ready = True

    def generate_chat_log():
        from bs4 import BeautifulSoup
        log = []
        for msg in st.session_state.get("messages", []):
            role = "You" if msg["role"] == "user" else "Bot"
            text = BeautifulSoup(msg["content"], "html.parser").get_text().strip()
            log.append(f"{role}:\n{text}\n" + "-"*50)
        return "\n\n".join(log)

    # ✅ This always runs
    chat_log_text = generate_chat_log()
    st.download_button(
        label="📥 Download Chat Log",
        data=chat_log_text,
        file_name="chat_history.txt",
        mime="text/plain"
    )

    # ✅ Moved this OUTSIDE the "if st.session_state.chat_ready"
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("""---""")

    st.markdown("""
    💡 **Tip:** To chat with the model or greetings and things like accuracy, algorithm, etc., prefix your question with **`faq:`**  
    Example:  
    `faq: what is the accuracy?`
    """)

    st.markdown("""---""")

# Display existing messages
# Display existing messages
# --- Display Previous Messages ---
for i, msg in enumerate(st.session_state.messages):
    if "timestamp" not in msg or not msg["timestamp"]:
        st.session_state.messages[i]["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html="<div" in msg["content"])
        try:
            t = datetime.strptime(msg["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
        except:
            t = msg["timestamp"]
        st.caption(f"🕒 {t}")



# Chat inputt
user_input = st.chat_input("Paste your news article (max 1000 chars) or ask a question...")

if user_input:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if len(user_input) > 1000:
        st.error("⚠ Input exceeds 1000 characters! Please shorten your content.")
        st.stop()


    # Log user message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input,
        "timestamp": timestamp
    })

    with st.chat_message("user"):
        st.markdown(user_input)
        t = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
        st.caption(f"🕒 {t}")

    # Handle FAQ
    if user_input.lower().startswith("faq:"):
        q = user_input[4:].strip()
        answer = get_custom_response(q) or "❓ Sorry, I don't understand. Try pasting a news headline or use 'faq:' for help."

        assistant_msg = {
            "role": "assistant",
            "content": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        st.session_state.messages.append(assistant_msg)
        # Save FAQ interaction to DB
        with st.spinner("🧠 Reading your Message...."):    
            insert_chat("user", user_input)
            insert_chat("bot", answer)  # defaults: prediction='UNKNOWN', confidence=0.00


        with st.chat_message("assistant"):
            st.markdown(answer)
            t = datetime.strptime(assistant_msg['timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
            st.caption(f"🕒 {t}")


    else:
        faq_keywords = list(custom_responses.keys())
        lowered = user_input.lower().strip()

        if not is_valid_news_input(user_input):
            if any(re.search(r'\b' + re.escape(k.lower()) + r'\b', lowered) for k in faq_keywords):
                friendly_reminder = "🤖 It looks like you're asking a general question. Please prefix it with `faq:` so I know it's not news."
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": friendly_reminder,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                with st.chat_message("assistant"):
                    st.markdown(friendly_reminder)
                    st.caption(f"🕒 " + datetime.now().strftime("%b %d %Y, %H:%M"))
                st.stop()


        if not is_valid_news_input(user_input):
            error_msg = "❌ Your input doesn't seem like meaningful news. Please try again with a proper sentence."
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            with st.chat_message("assistant"):
                st.markdown(error_msg)
                st.stop()

        title, content = extract_title_content(user_input)
        combined = f"{title} {content}".strip()
        vec = vectorizer.transform([combined])
        prob = model.predict_proba(vec)[0][1]
        prediction = "FAKE" if prob >= threshold else "REAL"
        confidence = float(round(prob if prediction == "FAKE" else 1 - prob, 2))

        with st.spinner("🧠 Thinking about the news..."):
            fact = fetch_claude_fact_check(user_input)

        if prediction == "REAL":
            html_response = f"""
                <div style='background-color:#d4edda; padding: 10px; border-left: 5px solid #28a745; border-radius: 5px;'>
                    📢 <strong style='color:#155724;'>This news appears to be real</strong><br><br>
                    <span style='color:#155724;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span>
                </div>
            """
        else:
            html_response = f"""
                <div style='background-color:#f8d7da; padding: 10px; border-left: 5px solid #dc3545; border-radius: 5px;'>
                    🚨 <strong style='color:#721c24;'>This news appears to be fake</strong><br><br>
                    <span style='color:#4f1216;'>🔍 <strong>Confidence:</strong> {round(confidence * 100, 2)}%</span><br><br>
                    <div style='background-color:#1e1e1e; color:#f1f1f1; padding:10px; border-radius:10px; font-family:monospace;'>
                        <strong>What might be true:</strong><br>
                        {fact}
                    </div>
                </div>
            """
        timestamp_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with st.spinner("🧠 Searching for relatable sources...."):
            insert_chat("user", user_input)
            insert_chat(speaker="bot", message=prediction, prediction=prediction, confidence_score=round(confidence * 100, 2))

        assistant_msg = {
            "role": "assistant",
            "content": html_response,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        st.session_state.messages.append(assistant_msg)

        with st.chat_message("assistant"):
            st.markdown(html_response, unsafe_allow_html=True)
            t = datetime.strptime(assistant_msg['timestamp'], "%Y-%m-%d %H:%M:%S").strftime("%b %d %Y, %H:%M")
            st.caption(f"🕒 {t}")