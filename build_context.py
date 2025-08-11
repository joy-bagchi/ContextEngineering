# -*- coding: utf-8 -*-
"""
Context Engineering demo with LangChain + LangGraph
- Pruning: MMR retrieval + (optional) contextual compression
- Budgeting: hard token caps per section
- Summarizing: LLM-based compression for long chunks
- Router: decides when to retrieve
- Prompt logging & token accounting for debugging
"""

from typing import List, TypedDict, Literal, Optional
import math

from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import tiktoken

# -----------------------------
# Config: models & token budget
# -----------------------------
ANSWER_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

# Token budget strategy (tune these)
CTX_WINDOW = 128000           # pick the real window of your model
SYSTEM_BUDGET = 1200          # system instructions cap
HISTORY_BUDGET = 1800         # chat history cap (not used heavily here)
RETRIEVED_BUDGET = 6000       # cap for retrieved context (core lever)
SAFETY_MARGIN = 1500          # headroom for the model’s own output, tool chatter, etc.

# -------------------------------------------------
# Utility: approximate tokens with tiktoken encoder
# -------------------------------------------------
def count_tokens(text: str, model: str = ANSWER_MODEL) -> int:
    try:
        enc = tiktoken.encoding_for_model(model)
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def trim_to_token_budget(snippets: List[str], budget: int, model: str = ANSWER_MODEL) -> List[str]:
    kept, total = [], 0
    for s in snippets:
        t = count_tokens(s, model)
        if total + t > budget:
            break
        kept.append(s)
        total += t
    return kept

# -----------------
# Demo source data
# -----------------
RAW_DOCS = [
    Document(page_content="""
Llamas (Lama glama) are domesticated South American camelids widely used as pack animals.
They are native to the Andes: Peru, Bolivia, Ecuador, Chile, and Argentina.
""", metadata={"id": "doc_llama_intro", "region": "andes", "year": 2020}),
    Document(page_content="""
The natural habitats include high-altitude grasslands and plateaus.
Llamas can carry loads over long distances and are well adapted to thin air.
""", metadata={"id": "doc_habitat", "region": "andes", "year": 2018}),
    Document(page_content="""
Alpacas differ from llamas in fiber quality and size. Alpacas are primarily bred for fleece.
""", metadata={"id": "doc_alpaca_compare", "region": "andes", "year": 2019}),
    Document(page_content="""
Some care guides mistakenly list North American ranches as natural llama habitats.
Those are managed environments rather than native ranges.
""", metadata={"id": "doc_misinfo_guard", "region": "na", "year": 2021}),
]

# --------------------------------------
# Build chunks → embeddings → retriever
# --------------------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)
CHUNKS = splitter.split_documents(RAW_DOCS)

emb = OpenAIEmbeddings(model=EMBED_MODEL)
vs = FAISS.from_documents(CHUNKS, emb)

# **Pruning #1: MMR retriever** (diversity & relevance)
retriever = vs.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 8,        # final results
        "fetch_k": 24, # candidate pool before MMR selection
        "lambda_mult": 0.5
    },
)

# **Summarizing**: query-focused compression of chunks
compressor_llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)
compressor = LLMChainExtractor.from_llm(compressor_llm)
compressed_retriever = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor,
)

# -----------------
# Prompt templates
# -----------------
ROLE_INSTR = (
    "You are a precise, source-citing assistant for technical users.\n"
    "Use the supplied CONTEXT if relevant. If information is missing, say so.\n"
    "Cite sources like [id] where id is from each chunk's metadata."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", ROLE_INSTR),
    ("system", "CONTEXT:\n{context}"),
    ("human", "QUESTION:\n{question}")
])

# --------------------------------
# Graph state and helper functions
# --------------------------------
class CEState(TypedDict, total=False):
    question: str
    # routing
    need_retrieval: bool
    # artifacts
    raw_docs: List[Document]
    compressed_docs: List[Document]
    context_snippets: List[str]
    context: str
    answer: str
    debug: dict

# Router schema (very small, deterministic)
from pydantic import BaseModel, Field
class Route(BaseModel):
    need_retrieval: bool = Field(..., description="True if external knowledge is needed.")

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "Decide if the user's question needs external retrieval. Answer with a boolean."),
    ("human", "{question}")
])

def router_node(state: CEState) -> CEState:
    llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0).with_structured_output(Route)
    route = llm.invoke(router_prompt.format_messages(question=state["question"]))
    return {**state, "need_retrieval": bool(route.need_retrieval)}

def route_edge(state: CEState) -> Literal["retrieve", "answer"]:
    # If retrieval isn't needed, go straight to answering with empty context.
    return "retrieve" if state.get("need_retrieval") else "answer"

# ---------
# Retrieval
# ---------
def retrieve_node(state: CEState) -> CEState:
    q = state["question"]
    # **Pruning #2 (optional): metadata filtering example**
    # Example (commented): retriever.search_kwargs["filter"] = {"region": "andes", "year": {"$gte": 2018}}
    raw = retriever.invoke(q)
    # **Summarizing** (contextual compression)
    compressed = compressed_retriever.invoke(q)
    dbg = {
        "raw_count": len(raw),
        "compressed_count": len(compressed),
        "raw_ids": [d.metadata.get("id") for d in raw],
        "compressed_ids": [d.metadata.get("id") for d in compressed],
    }
    return {**state, "raw_docs": raw, "compressed_docs": compressed, "debug": dbg}

# -------------------------
# Compose budgeted context
# -------------------------
def compose_node(state: CEState) -> CEState:
    # Render each doc as a compact, citeable snippet
    def render(doc: Document) -> str:
        sid = doc.metadata.get("id", "unknown")
        txt = " ".join(doc.page_content.split())
        return f"[{sid}] {txt}"

    # Prefer compressed docs; fall back to raw if compression failed
    docs = state.get("compressed_docs") or state.get("raw_docs") or []
    snippets = [render(d) for d in docs]

    # **Budgeting**: enforce token caps
    system_toks = count_tokens(ROLE_INSTR)
    # history not included in this small demo; reserve anyway
    remaining = CTX_WINDOW - SAFETY_MARGIN - system_toks - HISTORY_BUDGET
    cap = min(RETRIEVED_BUDGET, max(0, remaining))

    kept = trim_to_token_budget(snippets, budget=cap)
    context = "\n\n".join(kept)

    dbg = state.get("debug", {})
    dbg.update({
        "system_tokens": system_toks,
        "retrieved_tokens_total": sum(count_tokens(s) for s in snippets),
        "retrieved_tokens_kept": sum(count_tokens(s) for s in kept),
        "retrieved_snippets_kept": len(kept),
        "cap_used": cap,
    })
    return {**state, "context_snippets": kept, "context": context, "debug": dbg}

# -------------
# Answer node
# -------------
def answer_node(state: CEState) -> CEState:
    llm = ChatOpenAI(model=ANSWER_MODEL, temperature=0)
    # Build messages (even if context is empty)
    msgs = ANSWER_PROMPT.format_messages(
        question=state["question"],
        context=state.get("context", "")
    )
    # **Prompt logging** (for debugging)
    print("\n--- PROMPT SENT TO MODEL ---")
    for m in msgs:
        print(f"{m.type.upper()}:\n{m.content}\n")
    print("--- END PROMPT ---\n")

    resp = llm.invoke(msgs)
    ans = resp.content

    # Debug: counts
    prompt_tokens = sum(count_tokens(m.content) for m in msgs)
    dbg = state.get("debug", {})
    dbg.update({"prompt_tokens_total": prompt_tokens})
    return {**state, "answer": ans, "debug": dbg}

# ----------------
# Wire the graph
# ----------------
graph = StateGraph(CEState)
graph.add_node("router", router_node)
graph.add_node("retrieve", retrieve_node)
graph.add_node("compose", compose_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("router")
graph.add_conditional_edges("router", route_edge)
graph.add_edge("retrieve", "compose")
graph.add_edge("compose", "answer")
graph.add_edge("answer", END)

app = graph.compile()

# ---------------
# Demo questions
# ---------------
if __name__ == "__main__":
    qs = [
        "Where do llamas live?",
        "Compare llamas and alpacas briefly and cite sources.",
        "Say hello without using external knowledge."
    ]
    for q in qs:
        print("="*80)
        print("Q:", q)
        result = app.invoke({"question": q})
        print("\nANSWER:\n", result["answer"])
        print("\nDEBUG:\n", result["debug"])
