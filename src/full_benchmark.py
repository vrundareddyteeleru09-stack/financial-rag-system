import os
import sys
import time
import json
import math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from retriever import get_strategy1_retriever, get_strategy2_retriever, get_strategy3_retriever

load_dotenv()

COST_PER_1K_TOKENS = 0.002
AVG_TOKENS_PER_QUERY = 800

def load_ground_truth():
    with open("expanded_golden_set.json") as f:
        data = json.load(f)
    # Only use questions with expected answers
    questions = [q for q in data if q.get("expected") and not q.get("needs_review")]
    print(f"Loaded {len(questions)} ground truth questions")
    return questions

def run_strategy(name, retriever, questions):
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    Question: {question}
    """)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )

    results = []
    correct = 0
    total_latency = 0
    total_cost = 0

    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"Questions: {len(questions)}")
    print(f"{'='*60}")

    for q in questions:
        start = time.time()
        answer = chain.invoke(q["question"])
        latency = time.time() - start
        total_latency += latency

        # Cost estimate
        cost = (AVG_TOKENS_PER_QUERY / 1000) * COST_PER_1K_TOKENS
        total_cost += cost

        # Check correctness
        expected_keywords = [k.strip().lower() for k in q["expected"].split(",")]
        passed = any(kw in answer.lower() for kw in expected_keywords)
        if passed:
            correct += 1

        status = "✅" if passed else "❌"
        print(f"{status} Q{q['id']} [{q['category']}]: {q['question'][:45]}...")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer,
            "expected": q["expected"],
            "passed": passed,
            "latency": latency,
            "cost": cost
        })

    total = len(questions)
    accuracy = (correct / total) * 100
    avg_latency = total_latency / total
    avg_cost = total_cost / total

    # Per category breakdown
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["correct"] += 1

    print(f"\nAccuracy: {accuracy:.1f}%")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Avg Cost/Query: ${avg_cost:.4f}")
    print(f"Total Cost: ${total_cost:.4f}")

    return {
        "strategy": name,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "total_cost": total_cost,
        "correct": correct,
        "total": total,
        "categories": categories,
        "results": results
    }

def print_results_table(all_results):
    print(f"\n{'='*70}")
    print("CORE RESULTS TABLE")
    print(f"{'='*70}")
    print(f"{'Strategy':<35} {'Accuracy':>10} {'Latency':>10} {'Cost/Q':>10} {'Total':>10}")
    print(f"{'-'*70}")
    for r in all_results:
        print(f"{r['strategy']:<35} {r['accuracy']:>9.1f}% {r['avg_latency']:>9.2f}s ${r['avg_cost']:>8.4f} ${r['total_cost']:>8.4f}")

    print(f"\n{'='*70}")
    print("ACCURACY BY CATEGORY")
    print(f"{'='*70}")

    # Get all categories
    all_cats = set()
    for r in all_results:
        all_cats.update(r["categories"].keys())

    print(f"{'Category':<25}", end="")
    for r in all_results:
        short_name = r["strategy"].split("—")[0].strip()[:12]
        print(f" {short_name:>12}", end="")
    print()
    print("-"*70)

    for cat in sorted(all_cats):
        print(f"{cat:<25}", end="")
        for r in all_results:
            if cat in r["categories"]:
                c = r["categories"][cat]
                pct = (c["correct"] / c["total"]) * 100 if c["total"] > 0 else 0
                print(f" {pct:>11.0f}%", end="")
            else:
                print(f" {'N/A':>12}", end="")
        print()

def run_full_benchmark():
    print("="*60)
    print("FULL BENCHMARK — 3 STRATEGIES vs 50 GROUND TRUTH QUESTIONS")
    print("="*60)

    questions = load_ground_truth()

    if not questions:
        print("No ground truth questions found. Run expand_golden_set.py first.")
        return

    # Run all 3 strategies
    s1 = run_strategy("Strategy 1 — Pure Vector", get_strategy1_retriever(k=6), questions)
    s2 = run_strategy("Strategy 2 — Metadata Filter", get_strategy2_retriever(year_filter="2024", k=6), questions)
    s3 = run_strategy("Strategy 3 — Hybrid BM25", get_strategy3_retriever(k=6), questions)

    all_results = [s1, s2, s3]

    # Print core results table
    print_results_table(all_results)

    # Save
    with open("full_benchmark_results.json", "w") as f:
        json.dump({
            "strategy1": s1,
            "strategy2": s2,
            "strategy3": s3
        }, f, indent=2)

    print(f"\n✅ Results saved to full_benchmark_results.json")
    print("\nThis is your paper's core result table.")

if __name__ == "__main__":
    run_full_benchmark()