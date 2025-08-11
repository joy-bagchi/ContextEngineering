# pip install duckdb pandas langchain langchain-openai sqlglot tiktoken python-dateutil openpyxl
import os
import re
import duckdb
import pandas as pd
from dateutil import parser as dtparser
from typing import List, Tuple
import sqlglot

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# -----------------------
# Config
# -----------------------
OPENAI_MODEL = "gpt-4o-mini"
FILE_PATH = r"C:\Users\jayba\OneDrive - jaybagchi.com\Personal\Expense Data.xlsx"  # adjust if needed, e.g. r"C:\path\Expense Data.xlsx"
DB_PATH = "expenses.duckdb"
TABLE_NAME = "expense"  # canonical table name

# -----------------------
# Helpers: column cleaning
# -----------------------
def snake(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"__+", "_", s)
    return s.strip("_").lower()

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize headers
    df = df.copy()
    df.columns = [snake(c) for c in df.columns]

    # Try to infer likely date/amount columns (best-effort; tweak to your schema)
    likely_date_cols = [c for c in df.columns if re.search(r"(date|txn|posted|timestamp)", c)]
    likely_amount_cols = [c for c in df.columns if re.search(r"(amount|total|value|debit|credit)", c)]

    # Parse dates
    for c in likely_date_cols:
        try:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        except Exception:
            pass

    # Coerce numerics
    for c in likely_amount_cols:
        try:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        except Exception:
            pass

    return df

# -----------------------
# Load Excel → DataFrame
# -----------------------
# If your workbook has multiple sheets, you can choose one by name/index.
df = pd.read_excel(FILE_PATH, sheet_name=0)
df = normalize_df(df)

# If your file has specific expected columns, you can assert them here, e.g.:
# required = {"txn_date", "merchant", "category", "amount"}
# missing = required - set(df.columns)
# if missing: raise ValueError(f"Missing expected columns: {missing}")

# -----------------------
# DuckDB: create / replace
# -----------------------
con = duckdb.connect(DB_PATH)

# Register the DataFrame as a DuckDB view for efficient ingest
con.register("df_src", df)

# Create/replace canonical table with typed columns (DuckDB infers types here)
con.execute(f"CREATE OR REPLACE TABLE {TABLE_NAME} AS SELECT * FROM df_src")

# Optional: add convenient materialized views (tweak col names to your data)
# We try to find best-guess columns:
cols = set(df.columns)
date_col = next((c for c in ["txn_date","date","posted_date","transaction_date"] if c in cols), None)
amount_col = next((c for c in ["amount","total_amount","value","debit","credit"] if c in cols), None)
merchant_col = next((c for c in ["merchant","vendor","payee","counterparty"] if c in cols), None)
category_col = next((c for c in ["category","sub_category","type"] if c in cols), None)
state_col = next((c for c in ["state","region"] if c in cols), None)

def safe_ident(x):
    return x if x is not None else None

# Create month truncation view if date/amount present
if date_col and amount_col:
    con.execute(f"""
    CREATE OR REPLACE VIEW mv_expense_by_month AS
    SELECT date_trunc('month', {date_col}) AS year_month,
           SUM({amount_col}) AS total_amount
    FROM {TABLE_NAME}
    WHERE {date_col} IS NOT NULL
    GROUP BY 1;
    """)
    if category_col:
        con.execute(f"""
        CREATE OR REPLACE VIEW mv_expense_by_month_category AS
        SELECT date_trunc('month', {date_col}) AS year_month,
               {category_col} AS category,
               SUM({amount_col}) AS total_amount
        FROM {TABLE_NAME}
        WHERE {date_col} IS NOT NULL
        GROUP BY 1,2;
        """)
    if merchant_col:
        con.execute(f"""
        CREATE OR REPLACE VIEW mv_expense_by_month_merchant AS
        SELECT date_trunc('month', {date_col}) AS year_month,
               {merchant_col} AS merchant,
               SUM({amount_col}) AS total_amount
        FROM {TABLE_NAME}
        WHERE {date_col} IS NOT NULL
        GROUP BY 1,2;
        """)

# -----------------------
# Build schema string for the planner prompt (compact)
# -----------------------
def duckdb_table_info(con, table: str) -> List[Tuple[int,str,str]]:
    return con.execute(f"PRAGMA table_info('{table}')").fetchall()  # (cid, name, type, ...)

schema_lines = []
ti = duckdb_table_info(con, TABLE_NAME)
cols_desc = ", ".join(f"{r[1]} {r[2]}" for r in ti)
schema_lines.append(f"- {TABLE_NAME}({cols_desc})")

def view_schema(view_name: str):
    try:
        ti = duckdb_table_info(con, view_name)
        cols_desc = ", ".join(f"{r[1]} {r[2]}" for r in ti)
        schema_lines.append(f"- {view_name}({cols_desc})")
    except Exception:
        pass

for v in ["mv_expense_by_month", "mv_expense_by_month_category", "mv_expense_by_month_merchant"]:
    view_schema(v)

SCHEMA_TEXT = "\n".join(schema_lines)

# -----------------------
# LLM SQL Planner
# -----------------------
PLANNER = ChatPromptTemplate.from_messages([
    ("system",
     "You write SAFE SQL for DuckDB using ONLY the provided schema. "
     "Rules:\n"
     "- Prefer aggregations in SQL (SUM/AVG/COUNT), not raw rows.\n"
     "- Use WHERE for date ranges and categories/merchants when asked.\n"
     "- Always include LIMIT for result safety.\n"
     "- Return ONLY the SQL (no prose)."),
    ("system", "SCHEMA:\n{schema}"),
    ("human", "Question:\n{question}")
])

llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0, max_tokens=1024)
import re
import sqlglot

def clean_sql(text: str) -> str:
    s = text.strip()
    # If fenced: ```sql\n ... \n```  or ```\n...\n```
    m = re.search(r"```(?:\s*sql)?\s*\n(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        s = m.group(1).strip()
    # Remove lone leading 'sql' line if present
    lines = s.splitlines()
    if lines and lines[0].strip().lower() == "sql":
        s = "\n".join(lines[1:]).strip()
    # Remove a leading "SQL:" label if present
    s = re.sub(r"^\s*sql\s*:\s*", "", s, flags=re.IGNORECASE)
    # Collapse stray backticks just in case
    s = s.replace("`", "").strip()
    return s

def plan_sql(question: str) -> str:
    raw = llm.invoke(PLANNER.format_messages(schema=SCHEMA_TEXT, question=question)).content
    sql = clean_sql(raw)
    print("\n--- Planner ---\n", sql)
    print("\n--- Schema ---\n", SCHEMA_TEXT)  #
    print("\n--- DuckDB columns ---\n", duckdb.COLUMNS)
    # Guardrails
    lower = sql.lower()
    if not lower.startswith("select") and " with " not in f" {lower} ":
        raise ValueError(f"Planner did not return a SELECT:\n{sql}")
    if any(bad in lower for bad in ("insert", "update", "delete", "drop", "alter")):
        raise ValueError("Write statements forbidden.")
    if "limit" not in lower:
        sql += "\nLIMIT 200"

    # Parse as DuckDB SQL (sets dialect explicitly)
    sqlglot.parse_one(sql, read="duckdb")
    return sql


def run_sql(sql: str) -> pd.DataFrame:
    return con.execute(sql).fetch_df()

def ask(question: str) -> pd.DataFrame:
    sql = plan_sql(question)
    print("\n--- Planned SQL ---\n", sql)
    df_ans = run_sql(sql)
    print("\n--- Rows:", len(df_ans))
    return df_ans

# -----------------------
# Examples (uncomment to run)
# -----------------------
# ex1 = ask("Total expense by category for July 2025.")
# ex2 = ask("Which merchant had the highest total amount in July 2025?")
# ex3 = ask("Show categories with more than 50% month-over-month increase from June to July 2025.")
# print(ex1.head())
# print(ex2.head())
# print(ex3.head())

print("Ready. Try calling ask('Which merchant had the highest total amount in July 2025?').")
print("\nSchema seen by the planner:\n", SCHEMA_TEXT)

