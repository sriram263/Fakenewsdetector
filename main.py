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

# --- SIDEBAR PLACEHOLDER ---
with st.sidebar:
    st.title("🛡️ Veritas Dashboard")
    # 1. Create a placeholder that we can WIPE and REFILL
    sidebar_placeholder = st.empty()
    
    st.divider()
    
    with st.expander("ℹ️ How it works"):
        st.caption("""
        1. **Paste a Headline:** Or a short article snippet.
        2. **AI Analysis:** Veritas searches the live web + checks patterns.
        3. **Verdict:** You get a Real/Fake rating with confidence score.
        """)
    st.markdown("Made with ❤️ by Sriram")

# --- HELPER: RENDER SIDEBAR UI ---
# We verify 'unique_id' to prevent the "Duplicate Widget ID" error
def render_sidebar_ui(unique_id):
    """Renders the stats and buttons into the placeholder."""
    # .container() allows us to group elements inside the placeholder
    with sidebar_placeholder.container():
        # A. Stats Section
        st.markdown("### 📊 Session Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Checked", st.session_state.stats["checked"])
        col2.metric("Real", st.session_state.stats["real"])
        col3.metric("Fake", st.session_state.stats["fake"])
        
        st.markdown("---")
        
        # B. Controls Section
        st.markdown("### ⚙️ Controls")
        
        def generate_chat_log():
            from bs4 import BeautifulSoup
            log = []
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Veritas"
                content = msg["content"]
                if "<div" in content: 
                    soup = BeautifulSoup(content, "html.parser")
                    text = soup.get_text(separator=" ", strip=True)
                else:
                    text = content
                log.append(f"[{msg.get('timestamp', '')}] {role}: {text}")
            return "\n\n".join(log)

        chat_log = generate_chat_log()

        b_col1, b_col2 = st.columns(2)
        
        # KEY FIX: We pass f"{unique_id}" to the key so Streamlit sees them as different buttons
        with b_col1:
            if st.button("🧹 Clear", key=f"clear_{unique_id}", type="secondary"):
                st.session_state.messages = []
                st.session_state.stats = {"checked": 0, "real": 0, "fake": 0}
                st.rerun()
                
        with b_col2:
            st.download_button(
                label="📥 Save", 
                data=chat_log, 
                file_name="veritas_log.txt", 
                mime="text/plain", 
                type="primary",
                key=f"download_{unique_id}" # Unique key prevents error
            )

# --- INITIAL RENDER (STARTUP) ---
# We render this IMMEDIATELY with ID "startup" so the user sees the buttons
render_sidebar_ui("startup")

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
    
    st.session_state.messages.append({"role": "user", "content": prompt, "timestamp": timestamp})
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.caption(f"🕒 {timestamp}")

    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("🔍 Scanning global news sources..."):
            raw_response = st.session_state.agent.process_input(prompt)
            
            json_match = re.search(r'<VERDICT_JSON>(.*?)</VERDICT_JSON>', raw_response, re.DOTALL)
            final_html = ""
            
            if json_match:
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

            # --- LATE UPDATE (FIX) ---
            # 1. Clear the old "Startup" sidebar
            sidebar_placeholder.empty()
            # 2. Render the new "Update" sidebar with Fresh Data and NEW IDs
            render_sidebar_ui("update")