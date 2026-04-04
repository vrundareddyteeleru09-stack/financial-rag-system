import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.callbacks import get_openai_callback
from retriever import get_strategy1_retriever, get_strategy2_retriever, get_strategy3_retriever

load_dotenv()

# Real OpenAI pricing gpt-3.5-turbo
INPUT_COST_PER_1K  = 0.0005
OUTPUT_COST_PER_1K = 0.0015

def load_ground_truth():
    with open("expanded_golden_set.json") as f:
        data = json.load(f)
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
    total_input_tokens = 0
    total_output_tokens = 0

    print(f"\n{'='*60}")
    print(f"Strategy: {name}")
    print(f"Questions: {len(questions)}")
    print(f"{'='*60}")

    for q in questions:
        start = time.time()
        with get_openai_callback() as cb:
            answer = chain.invoke(q["question"])
            input_tokens  = cb.prompt_tokens
            output_tokens = cb.completion_tokens
        latency = time.time() - start
        total_latency += latency

        # Real cost
        cost = (input_tokens / 1000) * INPUT_COST_PER_1K + \
               (output_tokens / 1000) * OUTPUT_COST_PER_1K
        total_cost += cost
        total_input_tokens  += input_tokens
        total_output_tokens += output_tokens

        # Correctness
        expected_keywords = [k.strip().lower() for k in q["expected"].split(",")]
        passed = any(kw in answer.lower() for kw in expected_keywords)
        if passed:
            correct += 1

        status = "✅" if passed else "❌"
        print(f"{status} Q{q['id']} [{q['category']}]: "
              f"{q['question'][:45]}... "
              f"[in={input_tokens} out={output_tokens} ${cost:.4f}]")

        results.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": answer,
            "expected": q["expected"],
            "passed": passed,
            "latency": latency,
            "cost": cost,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens
        })

    total = len(questions)
    accuracy   = (correct / total) * 100
    avg_latency = total_latency / total
    avg_cost    = total_cost / total
    avg_input   = total_input_tokens / total
    avg_output  = total_output_tokens / total

    # Per category
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["correct"] += 1

    print(f"\nAccuracy:          {accuracy:.1f}%")
    print(f"Avg Latency:       {avg_latency:.2f}s")
    print(f"Avg Input Tokens:  {avg_input:.0f}")
    print(f"Avg Output Tokens: {avg_output:.0f}")
    print(f"Avg Cost/Query:    ${avg_cost:.5f}")
    print(f"Total Cost:        ${total_cost:.4f}")

    return {
        "strategy": name,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "avg_cost": avg_cost,
        "total_cost": total_cost,
        "avg_input_tokens": avg_input,
        "avg_output_tokens": avg_output,
        "correct": correct,
        "total": total,
        "categories": categories,
        "results": results
    }

def print_results_table(all_results):
    print(f"\n{'='*75}")
    print("CORE RESULTS TABLE")
    print(f"{'='*75}")
    print(f"{'Strategy':<30} {'Accuracy':>10} {'Latency':>10} "
          f"{'In Tok':>8} {'Out Tok':>8} {'Cost/Q':>10}")
    print(f"{'-'*75}")
    for r in all_results:
        print(f"{r['strategy']:<30} "
              f"{r['accuracy']:>9.1f}% "
              f"{r['avg_latency']:>9.2f}s "
              f"{r['avg_input_tokens']:>8.0f} "
              f"{r['avg_output_tokens']:>8.0f} "
              f"${r['avg_cost']:>9.5f}")

    print(f"\n{'='*75}")
    print("ACCURACY BY CATEGORY")
    print(f"{'='*75}")
    all_cats = set()
    for r in all_results:
        all_cats.update(r["categories"].keys())

    print(f"{'Category':<25}", end="")
    for r in all_results:
        short = r["strategy"][:12]
        print(f" {short:>12}", end="")
    print()
    print("-"*75)

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
    print("FULL BENCHMARK — 3 STRATEGIES vs GROUND TRUTH")
    print("="*60)

    questions = load_ground_truth()
    if not questions:
        print("No ground truth questions found.")
        return

    s1 = run_strategy("Strategy 1 — Pure Vector",
                      get_strategy1_retriever(k=6), questions)
    s2 = run_strategy("Strategy 2 — Metadata Filter",
                      get_strategy2_retriever(year_filter="2024", k=6), questions)
    s3 = run_strategy("Strategy 3 — Hybrid BM25",
                      get_strategy3_retriever(k=6), questions)

    all_results = [s1, s2, s3]
    print_results_table(all_results)

    with open("full_benchmark_results.json", "w") as f:
        json.dump({"strategy1": s1, "strategy2": s2, "strategy3": s3}, f, indent=2)

    print(f"\n✅ Results saved to full_benchmark_results.json")
    

if __name__ == "__main__":
    run_full_benchmark()