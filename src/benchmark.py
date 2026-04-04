import json
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever import get_strategy1_retriever, get_strategy2_retriever, get_strategy3_retriever
from golden_set import GOLDEN_SET

load_dotenv()

def run_strategy(strategy_name, retriever, questions):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    Question: {question}
    """)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    results = []
    correct = 0
    total_latency = 0
    print(f"\n{'='*60}")
    print(f"Running: {strategy_name}")
    print(f"{'='*60}")
    for item in questions:
        start = time.time()
        answer = chain.invoke(item["question"])
        latency = time.time() - start
        total_latency += latency
        expected_keywords = [k.strip().lower() for k in item["expected"].split(",")]
        passed = any(kw in answer.lower() for kw in expected_keywords)
        if passed:
            correct += 1
            status = "✅"
        else:
            status = "❌"
        print(f"{status} Q{item['id']}: {item['question'][:50]}... ({latency:.2f}s)")
        results.append({
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": answer,
            "passed": passed,
            "latency": latency
        })
    accuracy = (correct / len(questions)) * 100
    avg_latency = total_latency / len(questions)
    print(f"\nAccuracy: {accuracy:.1f}%")
    print(f"Avg Latency: {avg_latency:.2f}s")
    return {
        "strategy": strategy_name,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "correct": correct,
        "total": len(questions),
        "results": results
    }

def run_benchmark():
    questions = GOLDEN_SET
    s1_retriever = get_strategy1_retriever(k=6)
    s1_results = run_strategy("Strategy 1 — Pure Vector Search", s1_retriever, questions)
    s2_retriever = get_strategy2_retriever(year_filter="2024", k=6)
    s2_results = run_strategy("Strategy 2 — Metadata Filter 2024", s2_retriever, questions)
    s3_retriever = get_strategy3_retriever(k=6)
    s3_results = run_strategy("Strategy 3 — Hybrid BM25 + Vector", s3_retriever, questions)
    print(f"\n{'='*60}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"{'Strategy':<45} {'Accuracy':>10} {'Latency':>10}")
    print(f"{'-'*60}")
    for r in [s1_results, s2_results, s3_results]:
        print(f"{r['strategy']:<45} {r['accuracy']:>9.1f}% {r['avg_latency']:>9.2f}s")
    with open("benchmark_results.json", "w") as f:
        json.dump({
            "strategy1": s1_results,
            "strategy2": s2_results,
            "strategy3": s3_results
        }, f, indent=2)
    print(f"\nResults saved to benchmark_results.json")

if __name__ == "__main__":
    run_benchmark()