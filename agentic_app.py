# app.py
# pip install streamlit pandas altair

import json
import pandas as pd
import streamlit as st

# ---- import your tool-driven agent bits ----
from tools_sql_agent import (
    SYSTEM_MESSAGE,
    run_agent_with_history,
    sql_query,  # tool to run raw SQL (returns JSON)
)

# -------------------------
# auto-charting helper (Altair)
# -------------------------
import numpy as np
import altair as alt

def _coerce_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == object:
            try:
                parsed = pd.to_datetime(out[c], errors="raise", utc=False)
                if parsed.notna().mean() >= 0.7:
                    out[c] = parsed
            except Exception:
                pass
    return out

def _roles(df: pd.DataFrame):
    dt_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.datetime64)]
    num_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    cat_cols = []
    for c in df.columns:
        if c in dt_cols or c in num_cols:
            continue
        if df[c].dtype == object:
            cat_cols.append(c)
        else:
            if df[c].nunique(dropna=True) <= 50:
                cat_cols.append(c)
    for c in num_cols:
        if df[c].nunique(dropna=True) <= 12:
            cat_cols.append(c)
    cat_cols = list(dict.fromkeys(cat_cols))
    return dt_cols, num_cols, cat_cols

def auto_chart(df_in: pd.DataFrame):
    """
    Returns (chart, description) where chart is an Altair chart or None.
    """
    if df_in is None or df_in.empty:
        return None, "No data."

    df = _coerce_datetimes(df_in)
    dt_cols, num_cols, cat_cols = _roles(df)

    # 1) time series
    if dt_cols and num_cols:
        x = dt_cols[0]; y = num_cols[0]
        agg = df.groupby(x, as_index=False)[y].sum()
        chart = (
            alt.Chart(agg)
            .mark_line(point=True)
            .encode(x=alt.X(x, title=str(x)),
                    y=alt.Y(y, title=str(y)),
                    tooltip=[x, y])
            .properties(height=360)
        )
        return chart, f"Line chart of {y} over {x}"

    # 2) categorical + numeric(s)
    if cat_cols and num_cols:
        cat = cat_cols[0]
        if len(num_cols) == 1:
            y = num_cols[0]
            agg = df.groupby(cat, as_index=False)[y].sum()
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(x=alt.X(cat, sort='-y', title=str(cat)),
                        y=alt.Y(y, title=str(y)),
                        tooltip=[cat, y])
                .properties(height=360)
            )
            return chart, f"Bar chart of {y} by {cat}"
        else:
            long = df.melt(id_vars=[cat], value_vars=num_cols, var_name="metric", value_name="value")
            agg = long.groupby([cat, "metric"], as_index=False)["value"].sum()
            chart = (
                alt.Chart(agg)
                .mark_bar()
                .encode(x=alt.X(cat, sort='-y', title=str(cat)),
                        y=alt.Y("value", title="value"),
                        color=alt.Color("metric", legend=alt.Legend(title="Metric")),
                        tooltip=[cat, "metric", "value"])
                .properties(height=360)
            )
            return chart, f"Grouped bar of metrics by {cat}"

    # 3) scatter
    if len(num_cols) >= 2:
        x, y = num_cols[:2]
        chart = (
            alt.Chart(df)
            .mark_circle(size=60)
            .encode(x=alt.X(x, title=str(x)),
                    y=alt.Y(y, title=str(y)),
                    tooltip=[x, y] + [c for c in df.columns if c not in (x, y)][:3])
            .properties(height=360)
        )
        return chart, f"Scatter plot of {y} vs {x}"

    # 4) histogram
    if len(num_cols) == 1:
        v = num_cols[0]
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(alt.X(v, bin=alt.Bin(maxbins=30), title=str(v)),
                    alt.Y('count()', title='count'))
            .properties(height=360)
        )
        return chart, f"Histogram of {v}"

    return None, "Table view (no obvious chart type)."

# -------------------------
# Streamlit layout
# -------------------------
st.set_page_config(page_title="Expense Q&A (Tools + Memory + Charts)", page_icon="📊", layout="wide")
st.title("📊 Expense Q&A — Tools + Conversational Memory + Auto‑Charts")

# Sidebar controls
with st.sidebar:
    st.header("Options")
    page = st.number_input("Page", min_value=1, value=1, step=1)
    page_size = st.number_input("Page Size", min_value=1, max_value=1000, value=50, step=25)
    st.markdown("---")
    if st.button("Clear Conversation"):
        st.session_state.messages = [SYSTEM_MESSAGE]
        st.experimental_rerun()

# Session: initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [SYSTEM_MESSAGE]

# Render chat (hide tool/system in UI)
def render_chat(msgs):
    from langchain_core.messages import HumanMessage, AIMessage
    for m in msgs:
        if isinstance(m, HumanMessage):
            st.chat_message("user").write(m.content)
        elif isinstance(m, AIMessage):
            st.chat_message("assistant").write(m.content)

tab1, tab2 = st.tabs(["Ask in English (Agent)", "Run Raw SQL"])

# --- Tab 1: Conversational agent ---
with tab1:
    render_chat(st.session_state.messages)

    st.markdown("### Ask (multi‑line)")
    user_q = st.text_area(
        "The agent will remember prior turns and tool results. Type 'exit' or 'quit' to end.",
        height=120,
        key="agent_q",
    )
    c1, c2 = st.columns([1,1])
    run_clicked = c1.button("Run Agent", type="primary")
    exit_clicked = c2.button("Exit")

    if exit_clicked or (user_q and user_q.strip().lower() in {"exit", "quit"}):
        st.info("Session ended. Clear Conversation to start over.")
        st.stop()

    if run_clicked and user_q.strip():
        with st.spinner("Thinking..."):
            updated, answer = run_agent_with_history(
                history=st.session_state.messages,
                user_text=user_q.strip(),
                page=int(page),
                page_size=int(page_size),
                max_steps=4,
            )
            st.session_state.messages = updated

        # Show the last exchange immediately
        render_chat([st.session_state.messages[-2], st.session_state.messages[-1]])

# --- Tab 2: Raw SQL runner with auto‑chart ---
with tab2:
    sql = st.text_area("SQL (SELECT only)", "SELECT * FROM expense LIMIT 10", height=120, key="sql_text")
    c3, c4 = st.columns([1,1])
    page2 = c3.number_input("Page", min_value=1, value=int(page), step=1, key="page2")
    page_size2 = c4.number_input("Page Size", min_value=1, max_value=1000, value=int(page_size), step=25, key="ps2")

    if st.button("Run SQL Tool"):
        try:
            payload = json.loads(sql_query.invoke({"sql": sql, "page": int(page2), "page_size": int(page_size2)}))
        except Exception as e:
            st.error(f"SQL tool error: {e}")
        else:
            df = pd.DataFrame(payload["rows"], columns=payload["columns"])

            st.markdown("#### Data")
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("#### Auto Chart")
            chart, desc = auto_chart(df)
            if chart is None:
                st.info(f"No obvious chart to show ({desc}).")
            else:
                st.caption(desc)
                st.altair_chart(chart, use_container_width=True)

            st.markdown(f"Page {payload['page']} / {payload['total_pages']}  •  Rows: {len(df)} / {payload['total_rows']}")
            st.markdown("**Effective SQL**")
            st.code(payload["sql_effective"], language="sql")

st.caption("Tip: Use the Agent tab for natural‑language questions; use Raw SQL for power queries.")
