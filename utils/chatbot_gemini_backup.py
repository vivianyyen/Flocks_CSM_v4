"""
utils/chatbot.py
AI analyst chatbot powered by Google Gemini (gemini-2.0-flash).
Uses google-genai SDK (NOT google-generativeai).

secrets.toml:
    [gemini]
    api_key = "AIzaSy-..."
"""

import streamlit as st
import pandas as pd
import json

try:
    from google.genai import Client, types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(df: pd.DataFrame) -> str:
    lines = [
        "You are an expert cybersecurity and incident intelligence analyst.",
        "You are embedded in a live incident monitoring dashboard.",
        "Answer questions concisely and accurately using the data context below.",
        "If asked to analyse trends or make predictions, reason step by step.",
        "",
        "=== CURRENT DASHBOARD DATA SUMMARY ===",
        f"Total incidents: {len(df)}",
    ]

    def top5(col):
        if col in df.columns:
            return df[col].value_counts().head(5).to_dict()
        return {}

    for label, col in [
        ("Top categories",        "category"),
        ("Top incident types",    "incident_type"),
        ("Top countries",         "country"),
        ("Impact breakdown",      "impact"),
        ("Top sources",           "source"),
        ("Top entities affected", "entity_affected"),
    ]:
        d = top5(col)
        if d:
            lines.append(f"{label}: {json.dumps(d)}")

    if "incident_date" in df.columns:
        dated = df.dropna(subset=["incident_date"])
        if not dated.empty:
            lines.append(
                f"Date range: {dated['incident_date'].min().date()} "
                f"to {dated['incident_date'].max().date()} (GMT+8)"
            )

    lines += [
        "===",
        "",
        "Respond in clear English. Use bullet points for lists.",
        "If asked about predictions, clarify they are analytical estimates.",
        "Keep responses under 300 words unless more detail is specifically requested.",
    ]
    return "\n".join(lines)


# ── Gemini client (new SDK) ───────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def _get_client():
    if not GEMINI_AVAILABLE:
        return None, "package_missing"
    if "gemini" not in st.secrets or "api_key" not in st.secrets["gemini"]:
        return None, "missing_key"
    try:
        client = Client(api_key=st.secrets["gemini"]["api_key"])
        return client, "ok"
    except Exception as e:
        return None, str(e)


# ── Chat UI ───────────────────────────────────────────────────────────────────

def chatbot_ui(df: pd.DataFrame):
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    suggestions = [
        "What are the top 3 most critical incidents?",
        "Which country has the most incidents?",
        "Summarise the main threat categories.",
        "Are there any emerging trends?",
        "Which incident type is most frequent?",
    ]

    st.markdown("""
    <style>
    .chat-container {
        background: #161b22; border: 1px solid #21262d;
        border-radius: 12px; padding: 16px 20px;
        max-height: 420px; overflow-y: auto; margin-bottom: 12px;
    }
    .chat-msg-user {
        background: #1f3349; border-left: 3px solid #388bfd;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin: 8px 0; font-size: 14px; color: #c9d1d9;
    }
    .chat-msg-bot {
        background: #1a1f2e; border-left: 3px solid #3fb950;
        padding: 10px 14px; border-radius: 0 8px 8px 0;
        margin: 8px 0; font-size: 14px; color: #c9d1d9; line-height: 1.6;
    }
    .chat-role { font-size:11px; font-weight:600; letter-spacing:0.08em; margin-bottom:4px; text-transform:uppercase; }
    .user-role { color: #388bfd; }
    .bot-role  { color: #3fb950; }
    </style>
    """, unsafe_allow_html=True)

    # Suggestion chips
    st.markdown("<p style='font-size:12px;color:#484f58;margin-bottom:6px'>Suggested questions:</p>", unsafe_allow_html=True)
    chip_cols = st.columns(len(suggestions))
    for i, (col, s) in enumerate(zip(chip_cols, suggestions)):
        if col.button(s, key=f"suggest_{i}", use_container_width=True):
            st.session_state.chat_history.append({"role": "user", "content": s})
            st.session_state["_pending_q"] = s

    # Chat history display
    if st.session_state.chat_history:
        chat_html = "<div class='chat-container'>"
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                chat_html += (f"<div class='chat-msg-user'>"
                              f"<div class='chat-role user-role'>You</div>{msg['content']}</div>")
            else:
                content = msg["content"].replace("\n", "<br>")
                chat_html += (f"<div class='chat-msg-bot'>"
                              f"<div class='chat-role bot-role'>Gemini Analyst</div>{content}</div>")
        chat_html += "</div>"
        st.markdown(chat_html, unsafe_allow_html=True)

    # Input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input(
            "Ask the AI analyst…",
            placeholder="e.g. What are the most affected entities this month?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Send →")

    question = None
    if submitted and user_input.strip():
        question = user_input.strip()
        st.session_state.chat_history.append({"role": "user", "content": question})
    elif "_pending_q" in st.session_state:
        question = st.session_state.pop("_pending_q")

    # Generate response
    if question:
        client, status = _get_client()

        if status == "package_missing":
            st.error("❌ `google-genai` not installed. Ensure `requirements.txt` has `google-genai>=1.0.0`.")
        elif status == "missing_key":
            st.error("❌ Gemini API key missing. Add `[gemini]` with `api_key` to Streamlit secrets.")
        elif client is None:
            st.error(f"❌ Gemini init failed: {status}")
        else:
            with st.spinner("Gemini analyst is thinking…"):
                try:
                    # Build conversation history for the new SDK
                    contents = []
                    for m in st.session_state.chat_history[:-1]:
                        role = "model" if m["role"] == "assistant" else "user"
                        contents.append(
                            types.Content(role=role, parts=[types.Part(text=m["content"])])
                        )
                    # Add current question
                    contents.append(
                        types.Content(role="user", parts=[types.Part(text=question)])
                    )

                    response = client.models.generate_content(
                        model="gemini-2.0-flash",
                        contents=contents,
                        config=types.GenerateContentConfig(
                            system_instruction=_build_system_prompt(df),
                            max_output_tokens=1024,
                            temperature=0.4,
                        ),
                    )
                    answer = response.text
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Gemini error: {e}")

    if st.session_state.chat_history:
        if st.button("🗑️ Clear conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
