# tools_sql_agent.py
# pip install duckdb pandas langchain langchain-openai sqlglot

import io
import json
import re
from typing import Dict, Any, List, Optional, Tuple

import duckdb
import pandas as pd
import sqlglot
import datetime as dt
import decimal
import numpy as np


from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage


def _json_default(obj):
    # Datetimes / dates / times
    if isinstance(obj, (pd.Timestamp, dt.datetime, dt.date, dt.time)):
        return obj.isoformat()
    # Decimals
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    # numpy scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    # NaNs
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    # Fallback
    return str(obj)

def df_to_chart_json(df: pd.DataFrame) -> Dict[str, Any]:
    # Build rows with Python-native, JSON-safe values
    rows = []
    for row in df.itertuples(index=False, name=None):
        rows.append([_json_default(v) for v in row])
    return {"columns": list(df.columns), "rows": rows}

# -------------------------------
# DB connection (DuckDB local)
# -------------------------------
CON = duckdb.connect("expenses.duckdb")  # your DB with table(s)/view(s)

# -------------------------------
# Utils
# -------------------------------
FENCED_SQL_RE = re.compile(r"```(?:\s*sql)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)

def prune_history(messages: list, max_turns: int = 6):
    """
    Keep the System message + the last `max_turns` user/assistant/tool turns.
    """
    sys = [m for m in messages if isinstance(m, SystemMessage)]
    rest = [m for m in messages if not isinstance(m, SystemMessage)]
    pruned = sys[:1] + rest[-(max_turns*3):]  # rough cap (user/assistant/tool per turn)
    return pruned

def clean_sql(text: str) -> str:
    s = text.strip()
    m = FENCED_SQL_RE.search(s)
    if m:
        s = m.group(1).strip()
    lines = s.splitlines()
    if lines and lines[0].strip().lower() == "sql":
        s = "\n".join(lines[1:]).strip()
    s = re.sub(r"^\s*sql\s*:\s*", "", s, flags=re.IGNORECASE).strip()
    return s.rstrip(";").strip()

def assert_readonly_select(sql: str) -> None:
    low = f" {sql.lower()} "
    if " select " not in low and not low.strip().startswith("with "):
        raise ValueError("Only SELECT queries are allowed.")
    for bad in (" insert ", " update ", " delete ", " drop ", " alter ", " create ", " grant ", " revoke "):
        if bad in low:
            raise ValueError("Write/DDL statements are forbidden.")

def parse_duckdb(sql: str) -> None:
    # Validate SQL syntax in DuckDB dialect
    sqlglot.parse_one(sql, read="duckdb")

def run_df(sql: str) -> pd.DataFrame:
    return CON.execute(sql).fetch_df()

def count_total_rows(source_sql: str) -> int:
    return int(CON.execute(f"SELECT COUNT(*) AS n FROM ({source_sql}) AS sub").fetchone()[0])

def paginate_sql(source_sql: str, page: int, page_size: int) -> Tuple[str, int, int]:
    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 50), 1000))  # cap to 1000 per page
    offset = (page - 1) * page_size
    effective = f"SELECT * FROM ({source_sql}) AS sub LIMIT {page_size} OFFSET {offset}"
    return effective, page, page_size

def df_to_chart_json(df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "columns": list(df.columns),
        "rows": df.values.tolist(),
    }

def table_info(name: str) -> Optional[str]:
    try:
        info = CON.execute(f"PRAGMA table_info('{name}')").fetchall()
        cols = ", ".join(f"{c[1]} {c[2]}" for c in info)
        return f"{name}({cols})"
    except Exception:
        return None

# -------------------------------
# Tools
# -------------------------------
EXPOSED_OBJECTS = [
    "expense",
    "mv_expense_by_month",
    "mv_expense_by_month_category",
    "mv_expense_by_month_merchant",
]

@tool("schema_describe", return_direct=False)
def schema_describe(_: str = "") -> str:
    """
    Return a JSON schema summary: {"objects":[{"name":"...", "signature":"table(col type, ...)"},...]}
    """
    objs = []
    for name in EXPOSED_OBJECTS:
        sig = table_info(name)
        if sig:
            objs.append({"name": name, "signature": sig})
    return json.dumps({"objects": objs}, ensure_ascii=False)

@tool("sql_query", return_direct=False)
def sql_query(sql: str, page: int = 1, page_size: int = 50) -> str:
    """
    Execute a READ-ONLY SELECT with pagination. Returns JSON:
    {
      "columns": [...],
      "rows": [[...], ...],
      "page": 1,
      "page_size": 50,
      "total_rows": 123,
      "total_pages": 3,
      "sql_source": "...",     # cleaned user SQL
      "sql_effective": "..."   # wrapped with LIMIT/OFFSET
    }
    """
    if not sql:
        raise ValueError("Missing 'sql' parameter.")

    source = clean_sql(sql)
    assert_readonly_select(source)
    parse_duckdb(source)

    # total count BEFORE pagination
    total = count_total_rows(source)

    effective, page, page_size = paginate_sql(source, page, page_size)
    parse_duckdb(effective)
    df = run_df(effective)

    total_pages = (total + page_size - 1) // page_size if page_size else 1
    payload = df_to_chart_json(df)
    payload.update({
        "page": page,
        "page_size": page_size,
        "total_rows": total,
        "total_pages": total_pages,
        "sql_source": source,
        "sql_effective": effective,
    })
    return json.dumps(payload, ensure_ascii=False, default=_json_default)

# -------------------------------
# Minimal agent loop (tool-calling)
# -------------------------------
TOOLS = [schema_describe, sql_query]
AGENT_SYSTEM = (
    "You are a cautious analytics assistant. "
    "First call schema_describe to see available objects. "
    "Then plan a SELECT and call sql_query(sql, page, page_size). "
    "Aggregate in SQL (SUM/AVG/COUNT), use WHERE for date ranges, and rely on LIMIT/OFFSET via the tool. "
    "Return a concise answer; include the final SQL in an appendix."
)
PROMPT = ChatPromptTemplate.from_messages([
    ("system", AGENT_SYSTEM),
    ("user", "{question}")
])
LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)


# --- add these imports ---
def run_agent(question: str, page: int = 1, page_size: int = 50, max_steps: int = 4) -> str:
    """
    Let the model call tools; if it plans SQL, we pass page/page_size.
    """
    messages = PROMPT.format_messages(question=question)

    for _ in range(max_steps):
        ai = LLM.invoke(messages)

        # If the model did not call tools, we're done.
        tool_calls = getattr(ai, "tool_calls", None)
        if not tool_calls:
            return ai.content

        # Append the assistant message that requested tools
        messages.append(ai)

        # Execute each tool call and append a ToolMessage with tool_call_id
        for call in tool_calls:
            # Handle both dict- and attr-style tool call objects
            tc_id = call.get("id") if isinstance(call, dict) else getattr(call, "id", None)
            tc_name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            tc_args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {}) or {}

            if tc_name == "schema_describe":
                result = schema_describe.invoke({})
            elif tc_name == "sql_query":
                tc_args.setdefault("page", page)
                tc_args.setdefault("page_size", page_size)
                result = sql_query.invoke(tc_args)
            else:
                # Unknown tool safeguard
                result = json.dumps({"error": f"Unknown tool: {tc_name}"})

            # Append the tool result with the REQUIRED tool_call_id
            messages.append(ToolMessage(content=result, tool_call_id=tc_id, name=tc_name))

    return "Stopped after max tool steps. Refine the question or adjust parameters."

SYSTEM_MESSAGE = SystemMessage(content=AGENT_SYSTEM)

def run_agent_with_history(
    history: list,                 # list[BaseMessage]: System/Human/AI/Tool
    user_text: str,
    page: int = 1,
    page_size: int = 50,
    max_steps: int = 4,
):
    """
    Continue a conversation given `history` and a new user message.
    Returns (updated_history, final_text).
    """
    # Start from prior messages; append the new Human turn
    messages = list(history) + [HumanMessage(content=user_text)]

    for _ in range(max_steps):
        ai = LLM.invoke(messages)
        tool_calls = getattr(ai, "tool_calls", None)

        if not tool_calls:
            # Final assistant turn; append and return
            messages.append(ai)
            return messages, ai.content

        # Append the assistant message with tool_calls
        messages.append(ai)

        # Execute tools and append ToolMessage per call (must include tool_call_id)
        for call in tool_calls:
            tc_id  = call.get("id")   if isinstance(call, dict) else getattr(call, "id", None)
            tc_name= call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
            tc_args= call.get("args") if isinstance(call, dict) else getattr(call, "args", {}) or {}

            if tc_name == "schema_describe":
                result = schema_describe.invoke({})
            elif tc_name == "sql_query":
                tc_args.setdefault("page", page)
                tc_args.setdefault("page_size", page_size)
                result = sql_query.invoke(tc_args)
            else:
                result = json.dumps({"error": f"Unknown tool: {tc_name}"})

            messages.append(ToolMessage(content=result, tool_call_id=tc_id, name=tc_name))

    # If we hit max_steps without a final answer:
    messages.append(AIMessage(content="Stopped after max tool steps."))
    messages = prune_history(messages, max_turns=6)
    return messages, "Stopped after max tool steps."