import streamlit as st
import pytz
from datetime import datetime
from dotenv import load_dotenv

import config
from agent import SmartAgent

# Load environment variables
load_dotenv()

# Page config
st.set_page_config(page_title="Veritas | AI News Analyst v2.1", page_icon="🛡️", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* CHAT INPUT BORDER -> BLUE */
    div[data-testid="stChatInput"] > div {
        border-color: #3b82f6 !important; 
        border-width: 2px !important;
    }
    div[data-testid="stChatInput"] > div:focus-within {
        border-color: #2563eb !important; 
        box-shadow: 0 0 0 1px #2563eb !important;
    }

    /* GENERAL TWEAKS */
    .stButton>button { width: 100%; border-radius: 8px; font-weight: 600; }
    
    div[data-testid="stExpander"] {
        margin-top: 5px !important;
        border-radius: 8px !important;
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

    # Feature 1 & 2 Controls (Configurable Modes & KB Toggle)
    with st.expander("⚙️ Engine & Knowledge Base Settings", expanded=True):
        selected_mode = st.selectbox(
            "Retrieval Engine Mode:",
            ["enhanced", "baseline"],
            index=0,
            help="Enhanced: Multi-Query + Evidence Quality Ranking | Baseline: Single Tavily Query"
        )
        kb_enabled_toggle = st.toggle(
            "Enable Semantic Knowledge Base",
            value=True,
            help="Store and reuse completed fact-checks via persistent local vector DB"
        )
        if st.button("🧹 Clear KB Memory", key="clear_kb_btn", type="secondary"):
            st.session_state.agent.kb.clear_kb()
            st.toast("Knowledge Base memory cleared!", icon="🧹")
    
    # Regional Settings (Default: India)
    with st.expander("🌐 Regional Settings"):
        selected_timezone = st.selectbox(
            "Timezone:",
            ["Asia/Kolkata", "US/Pacific", "US/Eastern", "UTC", "Europe/London"],
            index=0 
        )
    
    with st.expander("ℹ️ How it works"):
        st.caption("""
        1. **Semantic KB Check:** Reuses verified fact-checks if fresh & similar.
        2. **Multi-Query Expansion:** Generates 4 complementary queries.
        3. **Claim-Evidence Verification:** Evaluates source stances (Supports/Refutes/Insufficient).
        4. **AI Verdict:** Synthesizes calibrated verdict & confidence.
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
        col1_val = st.session_state.stats["real"]
        col2.metric("Real", col1_val)
        col3.metric("Fake", st.session_state.stats["fake"])
        
        st.markdown("---")
        st.markdown("### ⚙️ Controls")
        
        def generate_chat_log():
            log = []
            for msg in st.session_state.messages:
                role = "User" if msg["role"] == "user" else "Veritas"
                content = msg.get("explanation", msg.get("content", ""))
                log.append(f"[{msg.get('timestamp', '')}] {role}: {content}")
            return "\n\n".join(log)

        chat_log = generate_chat_log()

        b_col1, b_col2 = st.columns(2)
        with b_col1:
            if st.button("🧹 Clear Chat", key=f"clear_{unique_id}", type="secondary"):
                st.session_state.messages = []
                st.session_state.stats = {"checked": 0, "real": 0, "fake": 0}
                st.rerun()
        with b_col2:
            st.download_button(
                label="📥 Save Log", 
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

# Helper: Render Verdict Card matching Mockup UI
def render_verdict_card(verdict_data, retrieval_details):
    if not verdict_data:
        return
    
    verdict = verdict_data.get("verdict", "UNCERTAIN").upper()
    confidence = verdict_data.get("confidence", 0)
    explanation = verdict_data.get("explanation", "")
    
    kb_badge = ""
    if retrieval_details and retrieval_details.get("kb_reused"):
        sim_pct = retrieval_details.get("kb_similarity", 0) * 100
        kb_badge = f'<span style="background-color: #3b82f6; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; float: right;">⚡ KB Hit ({sim_pct:.1f}% Match)</span>'

    if verdict == "REAL":
        icon = "✅"
        title = "Verified Real"
        text_color = "#064e3b"
        bg_color = "#d1fae5"
        border_color = "#10b981"
    elif verdict == "FAKE":
        icon = "🚨"
        title = "Flagged as Fake"
        text_color = "#7f1d1d"
        bg_color = "#fee2e2"
        border_color = "#ef4444"
    else:
        icon = "⚠️"
        title = "Unverified / Uncertain"
        text_color = "#78350f"
        bg_color = "#fef3c7"
        border_color = "#f59e0b"

    card_html = f"""<div style="background-color: {bg_color}; border-left: 6px solid {border_color}; color: {text_color}; padding: 22px 26px; border-radius: 16px; margin: 10px 0 15px 0; box-shadow: 0 4px 12px rgba(0,0,0,0.06); font-family: system-ui, -apple-system, sans-serif;">{kb_badge}<div style="display: flex; align-items: center; gap: 10px; margin-bottom: 4px;"><span style="font-size: 1.8rem; line-height: 1;">{icon}</span><h2 style="margin: 0; padding: 0; font-size: 1.45rem; font-weight: 700; color: {text_color}; border: none; background: transparent; display: inline;">{title}</h2></div><div style="font-size: 0.92rem; opacity: 0.85; font-weight: 500; margin-bottom: 12px; margin-left: 2px;">Confidence: {confidence}%</div><div style="background: {border_color}; opacity: 0.25; height: 1px; margin-bottom: 14px; border-radius: 1px;"></div><div style="font-size: 1.05rem; line-height: 1.6; font-weight: 450; color: {text_color};">{explanation}</div></div>"""
    
    st.markdown(card_html, unsafe_allow_html=True)

# Helper: Render Sources Expander
def render_sources_expander(sources_data):
    if not sources_data:
        return
    with st.expander("📚 Related News Sources & Evidence Stances"):
        for idx, s in enumerate(sources_data, 1):
            title = s.get('title', 'Unknown Title')
            url = s.get('url', 'N/A')
            category = s.get('domain_category', 'general').upper()
            score = s.get('final_evidence_score', None)
            stance = s.get('stance', 'INSUFFICIENT').upper()
            
            stance_icon = "✅" if stance == "SUPPORTS" else ("🚨" if stance == "REFUTES" else "ℹ️")
            score_str = f" | Quality Score: {score}" if score is not None else ""
            
            st.markdown(f"**{idx}. {stance_icon} [{stance}]** [{category}] [{title}]({url}){score_str}")
            if s.get("stance_reasoning"):
                st.caption(f"_{s.get('stance_reasoning')}_")
            elif s.get("content"):
                snippet = s.get("content")[:200] + "..." if len(s.get("content", "")) > 200 else s.get("content")
                st.caption(f"_{snippet}_")

# Helper: Render Retrieval Details Expander
def render_retrieval_details_expander(details):
    if not details or details.get("is_chat"):
        return
    with st.expander("🔍 Evidence Retrieval & Fact-Check Details"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Retrieval Mode", details.get("retrieval_mode", "N/A").upper())
        c2.metric("KB Status", details.get("kb_status", "N/A"))
        c3.metric("KB Similarity", f"{details.get('kb_similarity', 0.0)*100:.1f}%")
        c4.metric("Latency", f"{details.get('latency_ms', 0)} ms")

        st.markdown("---")
        st.markdown("##### 🎯 Multi-Query Expansion & Deduplication")
        q_list = details.get("queries", [])
        if q_list:
            for q in q_list:
                st.text(f"• [{q.get('category', 'query').upper()}] {q.get('query')} ({q.get('purpose', '')})")
        else:
            st.text("Single query executed.")

        st.markdown(f"""
        - **Raw Results Retrieved:** {details.get('raw_results_count', 0)}
        - **After Deduplication:** {details.get('deduplicated_count', 0)} ({details.get('duplicates_removed', 0)} duplicates removed)
        - **Final Top Sources Selected:** {len(details.get('selected_evidence', []))}
        - **Cross-Source Conflict Detected:** {'⚠️ YES' if details.get('conflict_detected') else '✅ NO'}
        """)

# Display Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🛡️"):
        if msg.get("verdict_data"):
            render_verdict_card(msg.get("verdict_data"), msg.get("retrieval_details"))
            render_sources_expander(msg.get("sources_data"))
            render_retrieval_details_expander(msg.get("retrieval_details"))
        else:
            st.markdown(msg["content"])
        
        if "timestamp" in msg:
            st.caption(f"🕒 {msg['timestamp']}")

# Input Handler
if prompt := st.chat_input("Paste news headline or ask a question..."):
    current_time = get_current_time(selected_timezone)
    
    # 1. User Message
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt, 
        "timestamp": current_time
    })
    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)
        st.caption(f"🕒 {current_time}")

    # 2. AI Processing
    with st.chat_message("assistant", avatar="🛡️"):
        with st.spinner("🔍 Executing multi-query expansion & claim-evidence verification..."):
            
            agent_output = st.session_state.agent.process_input(
                prompt,
                retrieval_mode=selected_mode,
                kb_enabled=kb_enabled_toggle
            )
            
            chat_response = agent_output.get("ai_response", "")
            verdict_data = agent_output.get("verdict_data", None)
            sources_list = agent_output.get("sources", [])
            retrieval_details = agent_output.get("retrieval_details", {})
            
            if verdict_data:
                # Update Session Stats
                st.session_state.stats["checked"] += 1
                v_str = verdict_data.get("verdict")
                if v_str == "REAL":
                    st.session_state.stats["real"] += 1
                elif v_str == "FAKE":
                    st.session_state.stats["fake"] += 1

                # Render Verdict Card & Expanders
                render_verdict_card(verdict_data, retrieval_details)
                render_sources_expander(sources_list)
                render_retrieval_details_expander(retrieval_details)
            else:
                # Render Casual Chat Reply
                st.markdown(chat_response)

            st.caption(f"🕒 {current_time}")
            
            # Save History
            st.session_state.messages.append({
                "role": "assistant", 
                "content": chat_response, 
                "verdict_data": verdict_data,
                "explanation": verdict_data.get("explanation") if verdict_data else chat_response,
                "timestamp": current_time,
                "sources_data": sources_list if verdict_data else None,
                "retrieval_details": retrieval_details if verdict_data else None
            })

            # Force Sidebar Update
            sidebar_placeholder.empty()
            render_sidebar_ui("update")