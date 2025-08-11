# pip install duckdb pandas langchain langchain-openai sqlglot
import duckdb, pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import sqlglot

# 1) Load Excel to DuckDB
df = pd.read_excel(r"C:\Users\jayba\OneDrive - jaybagchi.com\personal\Expense Data.xlsx")  # columns like: txn_date, store_id, state, sales_amount, tax_amount
con = duckdb.connect("expense.duckdb")
con.execute("""
CREATE TABLE IF NOT EXISTS fact_sales AS SELECT * FROM df
""")  # for prod: create schema with types; then INSERT/REPLACE

# Optional: helpful materialized views
con.execute("""
CREATE OR REPLACE VIEW mv_tax_by_store_month AS
SELECT date_trunc('month', txn_date) AS year_month,
       store_id, state, sum(sales_amount) AS sales, sum(tax_amount) AS tax
FROM fact_sales
GROUP BY 1,2,3;
""")

# 2) Tiny SQL planner prompt
PLANNER = ChatPromptTemplate.from_messages([
    ("system",
     "You write safe SQL for DuckDB using only tables: fact_sales, mv_tax_by_store_month. "
     "Always aggregate in SQL, filter by date/state when requested, and include LIMIT. "
     "Return ONLY SQL, no prose."),
    ("human", "Question:\n{question}\nSchema:\n- fact_sales(txn_date DATE, store_id INT, state TEXT, sales_amount DOUBLE, tax_amount DOUBLE)\n- mv_tax_by_store_month(year_month DATE, store_id INT, state TEXT, sales DOUBLE, tax DOUBLE)")
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def plan_sql(question: str) -> str:
    sql = llm.invoke(PLANNER.format_messages(question=question)).content.strip().strip("`")
    # Guardrails
    parsed = sqlglot.parse_one(sql)
    if parsed is None or "select" not in sql.lower():
        raise ValueError("Planner did not produce SELECT.")
    if "limit" not in sql.lower():
        sql += "\nLIMIT 200"
    for bad in ("insert","update","delete","drop","alter"):
        if bad in sql.lower():
            raise ValueError("Write statements forbidden.")
    return sql

def run_sql(sql: str):
    return con.execute(sql).fetch_df()  # dataframe

# Example Q: largest CA store in July 2025 (planner should produce the right GROUP BY)
q = "Which store in California has the largest Sales Tax collected for July 2025?"
sql = plan_sql(q)
df_ans = run_sql(sql)
print(sql)
print(df_ans.head())
