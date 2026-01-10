import streamlit as st
import re
import json
import os
from datetime import datetime
from dotenv import load_dotenv
from agent import SmartAgent

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Veritas | AI News Analyst", page_icon="🛡️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .report-box { padding: 20px; border-radius: 12px; margin-top: 15px; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .real-box { background-color: #d1fae5; border-left: 6px solid #10b981; color: #064e3b; }
    .fake-box { background-color: #fee2e2; border-left: 6px solid #ef4444; color: #7f1d1d; }
    .uncertain-box { background-color: #fef3c7; border-left: 6px solid #f59e0b; color: #78350f; }
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "agent" not in st.session_state:
    st.session_state.agent = SmartAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"checked": 0, "real": 0, "fake": 0}

# --- SIDEBAR DASHBOARD ---
with st.sidebar:
    st.title("🛡️ Veritas Dashboard")
    
    # Session Stats Widget
    st.markdown("### 📊 Session Stats")
    col1, col2, col3 = st.columns(3)
    col1.metric("Checked", st.session_state.stats["checked"])
    col2.metric("Real", st.session_state.stats["real"])
    col3.metric("Fake", st.session_state.stats["fake"])
    
    st.divider()

    # Action Buttons
    st.markdown("### ⚙️ Controls")
    
    def generate_chat_log():
        from bs4 import BeautifulSoup
        log = []
        for msg in st.session_state.messages:
            role = "User" if msg["role"] == "user" else "Veritas"
            text = BeautifulSoup(msg["content"], "html.parser").get_text().strip()
            log.append(f"[{msg.get('timestamp', '')}] {role}: {text}")
        return "\n\n".join(log)

    # Buttons in columns for better layout
    b_col1, b_col2 = st.columns(2)
    
    with b_col1:
        if st.button("🧹 Clear", type="secondary"):
            st.session_state.messages = []
            st.session_state.stats = {"checked": 0, "real": 0, "fake": 0}
            st.rerun()
            
    with b_col2:
        chat_log = generate_chat_log()
        st.download_button("📥 Save", chat_log, "veritas_log.txt", "text/plain", type="primary")

    st.divider()
    
    # "About" Expander
    with st.expander("ℹ️ How it works"):
        st.caption("""
        1. **Paste a Headline:** Or a short article snippet.
        2. **AI Analysis:** Veritas searches the live web + checks patterns.
        3. **Verdict:** You get a Real/Fake rating with confidence score.
        """)
    
    st.markdown("Made with ❤️ by Sriram")

# --- MAIN CHAT INTERFACE ---
st.title("🛡️ Veritas AI")
st.markdown("#### *The Truth is Just a Search Away*")
st.caption("Paste any news headline below. I'll cross-reference it with live web results instantly.")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🛡️"):
        st.markdown(msg["content"], unsafe_allow_html=True)
        if "timestamp" in msg:
            st.caption(f"🕒 {msg['timestamp']}")

# Input Handler
if prompt := st.chat_input("Paste news headline or ask a question..."):
    timestamp = datetime.now().strftime("%H:%M")
    
    # User Message
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.caption(f"🕒 {timestamp}")

    # AI Response
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("🔍 Scanning global news sources..."):
            raw_response = st.session_state.agent.process_input(prompt)
            
            # JSON Parsing Logic
            json_match = re.search(r'<VERDICT_JSON>(.*?)</VERDICT_JSON>', raw_response, re.DOTALL)
            final_html = ""
            
            if json_match:
                # Update Stats
                st.session_state.stats["checked"] += 1
                try:
                    data = json.loads(json_match.group(1))
                    verdict = data.get("verdict", "UNCERTAIN")
                    confidence = data.get("confidence", 0)
                    explanation = data.get("explanation", "Analysis complete.")
                    
                    if verdict == "REAL":
                        st.session_state.stats["real"] += 1
                        css = "real-box"
                        icon = "✅"
                        head = "Verified Real"
                    elif verdict == "FAKE":
                        st.session_state.stats["fake"] += 1
                        css = "fake-box"
                        icon = "🚨"
                        head = "Flagged as Fake"
                    else:
                        css = "uncertain-box"
                        icon = "⚠️"
                        head = "Unverified / Context Missing"

                    final_html = f"""
                    <div class="report-box {css}">
                        <h3 style="margin:0;">{icon} {head}</h3>
                        <p style="font-size:0.9em; opacity:0.8;">Confidence: {confidence}%</p>
                        <hr style="opacity:0.2;">
                        <p>{explanation}</p>
                    </div>
                    """
                except:
                    final_html = raw_response
            else:
                final_html = raw_response

            st.markdown(final_html, unsafe_allow_html=True)
            st.caption(f"🕒 {timestamp}")
            st.session_state.messages.append({"role": "assistant", "content": final_html, "timestamp": timestamp})