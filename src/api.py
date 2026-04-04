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
from retriever import get_strategy1_retriever, get_strategy2_retriever

load_dotenv()

app = FastAPI(
    title="Financial RAG System",
    description="Production RAG API with A/B testing for financial documents",
    version="1.0.0"
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

# ── A/B Router ─────────────────────────────────────────────
def ab_router():
    return "strategy1" if random.random() < 0.5 else "strategy2"

# ── Routes ─────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "service": "Financial RAG System",
        "version": "1.0.0",
        "status": "healthy",
        "endpoints": ["/query", "/ab/results", "/logs", "/metrics"]
    }

@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    query_id = str(uuid.uuid4())[:8]
    strategy = ab_router() if request.strategy == "auto" else request.strategy
    chain = s1_chain if strategy == "strategy1" else s2_chain
    start = time.time()
    try:
        answer = chain.invoke(request.question)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    latency_ms = (time.time() - start) * 1000
    log_entry = {
        "query_id": query_id,
        "question": request.question,
        "answer": answer,
        "strategy": strategy,
        "latency_ms": latency_ms,
        "timestamp": datetime.now().isoformat()
    }
    query_logs.append(log_entry)
    return QueryResponse(
        query_id=query_id,
        question=request.question,
        answer=answer,
        strategy_used=strategy,
        latency_ms=round(latency_ms, 2),
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
    latencies = [q["latency_ms"] for q in query_logs]
    avg_latency = sum(latencies) / len(latencies)
    s1 = [q for q in query_logs if q["strategy"] == "strategy1"]
    s2 = [q for q in query_logs if q["strategy"] == "strategy2"]
    total_cost = len(query_logs) * 0.002
    return {
        "total_queries": len(query_logs),
        "avg_latency_ms": round(avg_latency, 2),
        "min_latency_ms": round(min(latencies), 2),
        "max_latency_ms": round(max(latencies), 2),
        "strategy1_queries": len(s1),
        "strategy2_queries": len(s2),
        "ab_split": f"{len(s1)}/{len(s2)}",
        "estimated_cost_usd": round(total_cost, 4),
        "cost_per_query_usd": 0.002
    }