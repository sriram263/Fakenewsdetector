import streamlit as st
import re
import json
import pytz
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
    /* 1. VERDICT BOX STYLES */
    .report-box { 
        padding: 20px; 
        border-radius: 12px; 
        margin-top: 10px; 
        margin-bottom: 5px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
    }
    .real-box { background-color: #d1fae5; border-left: 6px solid #10b981; color: #064e3b; }
    .fake-box { background-color: #fee2e2; border-left: 6px solid #ef4444; color: #7f1d1d; }
    .uncertain-box { background-color: #fef3c7; border-left: 6px solid #f59e0b; color: #78350f; }
    
    /* 2. CHAT INPUT BORDER -> BLUE */
    div[data-testid="stChatInput"] > div {
        border-color: #3b82f6 !important; 
        border-width: 2px !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #2563eb !important; 
        box-shadow: 0 0 0 1px #2563eb !important;
    }

    /* 3. GENERAL TWEAKS */
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    
    /* TIGHTEN SPACING: Pull expanders up closer to the box */
    div[data-testid="stExpander"] {
        margin-top: 0px !important;
        border: none !important;
        box-shadow: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- STATE MANAGEMENT ---
if "agent" not in st.session_state:
    st.session_state.agent = SmartAgent()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stats" not in st.session_state:
    st.session_state.stats = {"checked": 0, "real": 0, "fake": 0}

# --- SIDEBAR UI ---
with st.sidebar:
    st.title("🛡️ Veritas Dashboard")
    sidebar_placeholder = st.empty()
    
    st.divider()
    
    # Regional Settings (Default: India)
    with st.expander("⚙️ Regional Settings"):
        selected_timezone = st.selectbox(
            "Timezone:",
            ["Asia/Kolkata", "US/Pacific", "US/Eastern", "UTC", "Europe/London"],
            index=0 
        )
    
    with st.expander("ℹ️ How it works"):
        st.caption("""
        1. **Paste a Headline:** Veritas searches the live web.
        2. **AI Analysis:** Checks facts against 5+ sources.
        3. **Verdict:** Real/Fake rating with evidence.
        """)
    st.markdown("Made with ❤️ by Sriram")

# --- HELPER: GET TIME ---
def get_current_time(tz_name):
    tz = pytz.timezone(tz_name)
    return datetime.now(tz).strftime("%H:%M")

# --- HELPER: RENDER SIDEBAR ---
def render_sidebar_ui(unique_id):
    with sidebar_placeholder.container():
        st.markdown("### 📊 Session Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Checked", st.session_state.stats["checked"])
        col2.metric("Real", st.session_state.stats["real"])
        col3.metric("Fake", st.session_state.stats["fake"])
        
        st.markdown("---")
        st.markdown("### ⚙️ Controls")
        
        def generate_chat_log():
            from bs4 import BeautifulSoup
            log = []
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Veritas"
                content = msg.get("raw_text", "")
                if not content: 
                    soup = BeautifulSoup(msg["content"], "html.parser")
                    content = soup.get_text(separator=" ", strip=True)
                log.append(f"[{msg.get('timestamp', '')}] {role}: {content}")
            return "\n\n".join(log)

        chat_log = generate_chat_log()

        b_col1, b_col2 = st.columns(2)
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
                key=f"download_{unique_id}"
            )

# Initial Render
render_sidebar_ui("startup")

# --- MAIN CHAT INTERFACE ---
st.title("🛡️ Veritas AI")
st.markdown("#### *The Truth is Just a Search Away*")
st.caption(f"📍 Region: {selected_timezone} | 🕒 Local Time: {get_current_time(selected_timezone)}")

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🛡️"):
        # 1. Render HTML Content (If it's a Verdict)
        if "html_content" in msg:
            st.markdown(msg["html_content"], unsafe_allow_html=True)

            # 2. Render Sources (ONLY if they exist in the saved message)
            if "sources_data" in msg and msg["sources_data"]:
                with st.expander("📚 Related News Sources"):
                    formatted_sources = ""
                    for idx, s in enumerate(msg["sources_data"], 1):
                        formatted_sources += f"{idx}. {s.get('title', 'Unknown')}\n   URL: {s.get('url', 'N/A')}\n\n"
                    st.code(formatted_sources, language="text")
        else:
            # 3. Fallback for Chat (Normal Text)
            st.markdown(msg["content"], unsafe_allow_html=True)
        
        if "timestamp" in msg:
            st.caption(f"🕒 {msg['timestamp']}")

# Input Handler
if prompt := st.chat_input("Paste news headline or ask a question..."):
    current_time = get_current_time(selected_timezone)
    
    # 1. User Message
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "raw_text": prompt,
        "timestamp": current_time
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.caption(f"🕒 {current_time}")

    # 2. AI Processing
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("🔍 Scanning global news sources..."):
            
            agent_output = st.session_state.agent.process_input(prompt)
            raw_text_response = agent_output["ai_response"]
            sources_list = agent_output["sources"]
            
            # Extract Verdict
            json_match = re.search(r'<VERDICT_JSON>(.*?)</VERDICT_JSON>', raw_text_response, re.DOTALL)
            
            html_content = ""
            is_news = False
            
            if json_match:
                # --- IT IS NEWS ---
                is_news = True
                st.session_state.stats["checked"] += 1
                try:
                    data = json.loads(json_match.group(1))
                    verdict = data.get("verdict", "UNCERTAIN")
                    confidence = data.get("confidence", 0)
                    explanation_text = data.get("explanation", "Analysis complete.")
                    
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
                        head = "Unverified"

                    html_content = f"""
                    <div class="report-box {css}">
                        <h3 style="margin:0; padding-bottom: 5px;">{icon} {head}</h3>
                        <p style="font-size:0.9em; opacity:0.8; margin: 0 0 10px 0;">Confidence: {confidence}%</p>
                        <div style="background:rgba(0,0,0,0.1); height:1px; margin-bottom:12px;"></div>
                        <p style="margin:0; line-height:1.5;">{explanation_text}</p>
                    </div>
                    """
                except:
                    html_content = raw_text_response
            else:
                # --- IT IS CHAT ---
                # We use the raw text response directly (e.g., "Hello! How can I help?")
                is_news = False
                html_content = raw_text_response

            # 3. Render Output
            st.markdown(html_content, unsafe_allow_html=True)

            # 4. Render Sources (Only if it's ACTUALLY news)
            if sources_list and is_news:
                with st.expander("📚 Related News Sources"):
                    formatted_sources = ""
                    for idx, s in enumerate(sources_list, 1):
                        formatted_sources += f"{idx}. {s.get('title', 'Unknown')}\n   URL: {s.get('url', 'N/A')}\n\n"
                    st.code(formatted_sources, language="text")

            st.caption(f"🕒 {current_time}")
            
            # 5. Save History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": html_content, 
                "html_content": html_content if is_news else None, # Only save HTML for news
                "raw_text": raw_text_response,
                "timestamp": current_time,
                "sources_data": sources_list if is_news else None
            })

            # Force Sidebar Update
            sidebar_placeholder.empty()
            render_sidebar_ui("update")