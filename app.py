# save this as app.py
# pip install streamlit pandas

import streamlit as st
import pandas as pd
from excel_preprocess import ask
from tools_sql_agent import run_agent
# ---- Import your ask() function here ----
# from your_existing_module import ask

# Dummy implementation for demo:
def ask_question(question: str) -> pd.DataFrame:
    # Replace with your real query executor
    answer = ask(question)
    data = {
        "label": [f"This is the answer for: {question}"],
        "question": [question],
        "answer": [answer]
    }
    return pd.DataFrame(data)

st.set_page_config(page_title="Expense Q&A", page_icon="💬")

st.title("💬 Expense Data Q&A")

# Text input for question
user_q = st.text_area("Enter your question (type 'exit' or 'quit' to stop):", "")

if st.button("Submit") and user_q:
    if user_q.strip().lower() in {"exit", "quit"}:
        st.warning("Exiting app. Refresh to restart.")
        st.stop()  # stops execution for this session

    try:
        # df_ans = ask(user_q)
        df_ans = run_agent(user_q)
        st.subheader("Answer")
        st.dataframe(df_ans)
    except Exception as e:
        st.error(f"Error: {e}")
