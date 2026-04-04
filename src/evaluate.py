import json
from golden_set import GOLDEN_SET
from rag import ask_question

def evaluate_rag_system():
    results = []
    correct = 0
    total = len(GOLDEN_SET)
    
    print(f"\n{'='*60}")
    print(f"RUNNING EVALUATION — {total} questions")
    print(f"{'='*60}\n")
    
    for item in GOLDEN_SET:
        answer = ask_question(item["question"])
        
        # Check if any expected keyword appears in answer
        expected_keywords = [k.strip().lower() for k in item["expected"].split(",")]
        answer_lower = answer.lower()
        passed = any(keyword in answer_lower for keyword in expected_keywords)
        
        if passed:
            correct += 1
            status = "✅ PASS"
        else:
            status = "❌ FAIL"
        
        result = {
            "id": item["id"],
            "category": item["category"],
            "question": item["question"],
            "answer": answer,
            "expected_keywords": item["expected"],
            "passed": passed
        }
        results.append(result)
        print(f"{status} | Q{item['id']} [{item['category']}]: {item['question'][:60]}...")
    
    accuracy = (correct / total) * 100
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {total}")
    print(f"Correct: {correct}")
    print(f"Failed: {total - correct}")
    print(f"Accuracy: {accuracy:.1f}%")
    print(f"\nBy Category:")
    
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"correct": 0, "total": 0}
        categories[cat]["total"] += 1
        if r["passed"]:
            categories[cat]["correct"] += 1
    
    for cat, stats in categories.items():
        cat_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"  {cat}: {stats['correct']}/{stats['total']} ({cat_accuracy:.0f}%)")
    
    # Save results
    with open("evaluation_results.json", "w") as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }, f, indent=2)
    
    print(f"\nResults saved to evaluation_results.json")
    return accuracy, results

if __name__ == "__main__":
    evaluate_rag_system()