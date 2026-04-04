import os
import sys
import time
import json
import uuid
import random
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from retriever import get_strategy1_retriever, get_strategy2_retriever, get_strategy3_retriever

load_dotenv()

# Real OpenAI pricing
INPUT_COST_PER_1K  = 0.0005
OUTPUT_COST_PER_1K = 0.0015

app = FastAPI(
    title="Financial RAG System",
    description="Production RAG API with smart routing for financial documents",
    version="2.0.0"
)

# ── Request / Response Models ──────────────────────────────
class QueryRequest(BaseModel):
    question: str
    strategy: str = "auto"

class QueryResponse(BaseModel):
    query_id: str
    question: str
    answer: str
    strategy_used: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str

class ABTestResult(BaseModel):
    query_id: str
    strategy: str
    correct: bool
    latency_ms: float

# ── In-memory logging ──────────────────────────────────────
query_logs = []
ab_results = []

# ── Build chains at startup ────────────────────────────────
print("Loading retrievers...")
s1_retriever = get_strategy1_retriever(k=6)
s2_retriever = get_strategy2_retriever(year_filter="2024", k=6)
s3_retriever = get_strategy3_retriever(k=6)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context:
{context}
Question: {question}
""")

s1_chain = (
    {"context": s1_retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
s2_chain = (
    {"context": s2_retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)
s3_chain = (
    {"context": s3_retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ── Smart Router ───────────────────────────────────────────
def smart_router(question: str) -> str:
    """
    Route based on query type:
    - Year-specific queries → Strategy 1 (pure vector, no year confusion)
    - General queries → Strategy 3 (hybrid BM25, best general accuracy)
    Strategy 2 available via explicit request only.
    """
    years = ["2020", "2021", "2022", "2023", "2024"]
    if any(year in question for year in years):
        return "strategy1"
    return "strategy3"

# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Financial RAG System",
        "version": "2.0.0",
        "status": "healthy",
        "routing": "smart — year-specific → strategy1, general → strategy3",
        "endpoints": ["/query", "/ab/results", "/logs", "/metrics"]
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    query_id = str(uuid.uuid4())[:8]

    # Smart routing
    if request.strategy == "auto":
        strategy = smart_router(request.question)
    else:
        strategy = request.strategy

    # Select chain
    if strategy == "strategy1":
        chain = s1_chain
    elif strategy == "strategy2":
        chain = s2_chain
    else:
        chain = s3_chain

    # Execute with real token tracking
    start = time.time()
    try:
        with get_openai_callback() as cb:
            answer = chain.invoke(request.question)
            input_tokens  = cb.prompt_tokens
            output_tokens = cb.completion_tokens
            real_cost = (input_tokens / 1000) * INPUT_COST_PER_1K + \
                        (output_tokens / 1000) * OUTPUT_COST_PER_1K
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.time() - start) * 1000

    # Log everything
    log_entry = {
        "query_id": query_id,
        "question": request.question,
        "answer": answer,
        "strategy": strategy,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(real_cost, 6),
        "timestamp": datetime.now().isoformat()
    }
    query_logs.append(log_entry)

    return QueryResponse(
        query_id=query_id,
        question=request.question,
        answer=answer,
        strategy_used=strategy,
        latency_ms=round(latency_ms, 2),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cost_usd=round(real_cost, 6),
        timestamp=datetime.now().isoformat()
    )

@app.post("/ab/feedback")
def ab_feedback(result: ABTestResult):
    ab_results.append(result.dict())
    return {"status": "logged", "query_id": result.query_id}

@app.get("/ab/results")
def get_ab_results():
    if not ab_results:
        return {"message": "No A/B results yet — send feedback via /ab/feedback"}
    s1 = [r for r in ab_results if r["strategy"] == "strategy1"]
    s2 = [r for r in ab_results if r["strategy"] == "strategy2"]
    s3 = [r for r in ab_results if r["strategy"] == "strategy3"]
    def stats(results):
        if not results:
            return {}
        correct = sum(1 for r in results if r["correct"])
        avg_latency = sum(r["latency_ms"] for r in results) / len(results)
        return {
            "total": len(results),
            "correct": correct,
            "accuracy": round(correct / len(results) * 100, 1),
            "avg_latency_ms": round(avg_latency, 2)
        }
    return {
        "strategy1": stats(s1),
        "strategy2": stats(s2),
        "strategy3": stats(s3),
        "total_queries": len(ab_results)
    }

@app.get("/logs")
def get_logs(limit: int = 10):
    return {
        "total_queries": len(query_logs),
        "recent": query_logs[-limit:]
    }

@app.get("/metrics")
def get_metrics():
    if not query_logs:
        return {"message": "No queries yet"}
    latencies  = [q["latency_ms"] for q in query_logs]
    costs      = [q["cost_usd"] for q in query_logs]
    in_tokens  = [q["input_tokens"] for q in query_logs]
    out_tokens = [q["output_tokens"] for q in query_logs]
    s1 = [q for q in query_logs if q["strategy"] == "strategy1"]
    s2 = [q for q in query_logs if q["strategy"] == "strategy2"]
    s3 = [q for q in query_logs if q["strategy"] == "strategy3"]
    return {
        "total_queries": len(query_logs),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "avg_input_tokens": round(sum(in_tokens) / len(in_tokens), 0),
        "avg_output_tokens": round(sum(out_tokens) / len(out_tokens), 0),
        "avg_cost_usd": round(sum(costs) / len(costs), 6),
        "total_cost_usd": round(sum(costs), 6),
        "strategy1_queries": len(s1),
        "strategy2_queries": len(s2),
        "strategy3_queries": len(s3),
        "routing": "year-specific→strategy1, general→strategy3"
    }