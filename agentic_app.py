# app.py
# pip install streamlit pandas
import json
import pandas as pd
import streamlit as st

from tools_sql_agent import (
    SYSTEM_MESSAGE,
    run_agent_with_history,
)

st.set_page_config(page_title="Expense Q&A (Conversational)", page_icon="💬", layout="wide")
st.title("💬 Expense Q&A — Conversational (Tools + Memory)")

# --- Session state: initialize chat history ---
if "messages" not in st.session_state:
    # start with the system instruction so the model always sees it
    st.session_state.messages = [SYSTEM_MESSAGE]

# --- Sidebar: paging controls & utilities ---
with st.sidebar:
    st.header("Query Options")
    page = st.number_input("Page", min_value=1, value=1, step=1)
    page_size = st.number_input("Page Size", min_value=1, max_value=1000, value=50, step=25)

    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = [SYSTEM_MESSAGE]
        st.experimental_rerun()

# --- Display conversation so far (only human/assistant) ---
def render_chat(msgs):
    for m in msgs:
        t = type(m).__name__
        if t == "HumanMessage":
            st.chat_message("user").write(m.content)
        elif t == "AIMessage":
            st.chat_message("assistant").write(m.content)
        # ToolMessage/SystemMessage are hidden from the UI to reduce clutter

render_chat(st.session_state.messages)

st.markdown("### Ask (multi‑line)")
user_text = st.text_area(
    "Type your question(s) here. The agent will remember prior context. Type 'exit' or 'quit' to end.",
    height=120,
)

col_run, col_exit = st.columns([1,1])
run_clicked = col_run.button("Run", type="primary")
exit_clicked = col_exit.button("Exit")

if exit_clicked or (user_text and user_text.strip().lower() in {"exit", "quit"}):
    st.info("Session ended. Clear Conversation to start over.")
    st.stop()

if run_clicked and user_text.strip():
    with st.spinner("Thinking..."):
        updated, answer = run_agent_with_history(
            history=st.session_state.messages,
            user_text=user_text.strip(),
            page=int(page),
            page_size=int(page_size),
            max_steps=8
        )
        st.session_state.messages = updated

    # Show the latest exchange
    render_chat([st.session_state.messages[-2], st.session_state.messages[-1]])

    # Optional: parse any JSON result blocks for quick charts
    # (If the assistant includes result JSON inline, you can extract & chart here.)
