import os
import sys
import time
import json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The failing question from Stage 2
FLOAT_QUESTION = "How does Berkshire Hathaway make money from insurance?"
FLOAT_EXPECTED = "float"

# All Stage 1 golden set questions
TEST_QUESTIONS = [
    ("What was Berkshire Hathaway's total revenue in 2024?", "321,643"),
    ("Who is the CEO of Berkshire Hathaway?", "Buffett"),
    ("What are Berkshire's main business segments?", "insurance"),
    ("How does Berkshire Hathaway make money from insurance?", "float"),
    ("What cybersecurity risks does Berkshire mention?", "cyber"),
    ("What is BNSF?", "rail"),
    ("What are the main risks Berkshire faces?", "regulatory"),
]

def load_and_chunk():
    data_folder = os.path.join(BASE_DIR, "data")
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_folder, filename))
            documents.extend(loader.load())
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(documents)

def test_embeddings(embedding_name, embeddings, chunks, questions):
    print(f"\n{'='*60}")
    print(f"Testing: {embedding_name}")
    print(f"{'='*60}")

    start = time.time()
    vs = FAISS.from_documents(chunks, embeddings)
    index_time = time.time() - start
    print(f"Index time: {index_time:.2f}s")

    retriever = vs.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    Answer based only on this context:
    {context}
    Question: {question}
    """)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    correct = 0
    total_latency = 0
    float_fixed = False

    for question, expected in questions:
        start = time.time()
        answer = chain.invoke(question)
        latency = time.time() - start
        total_latency += latency
        passed = expected.lower() in answer.lower()
        if passed:
            correct += 1
        if question == FLOAT_QUESTION and passed:
            float_fixed = True
        status = "✅" if passed else "❌"
        print(f"{status} {question[:55]}...")

    accuracy = (correct / len(questions)) * 100
    avg_latency = total_latency / len(questions)

    print(f"\nAccuracy: {accuracy:.1f}%")
    print(f"Avg Latency: {avg_latency:.2f}s")
    print(f"Float problem fixed: {'✅ YES!' if float_fixed else '❌ No'}")

    return {
        "embedding": embedding_name,
        "accuracy": accuracy,
        "avg_latency": avg_latency,
        "index_time": index_time,
        "float_fixed": float_fixed
    }

def run_domain_embedding_test():
    print("="*60)
    print("DOMAIN EMBEDDINGS EXPERIMENT")
    print("Hypothesis: Financial domain embeddings fix the float problem")
    print("="*60)

    chunks = load_and_chunk()
    print(f"Loaded {len(chunks)} chunks")

    results = []

    # Model 1 — FinBERT (financial domain)
    print("\nLoading FinBERT (financial domain model)...")
    finbert = HuggingFaceEmbeddings(
        model_name="ProsusAI/finbert",
        model_kwargs={"device": "cpu"}
    )
    r1 = test_embeddings("FinBERT (Financial Domain)", finbert, chunks, TEST_QUESTIONS)
    results.append(r1)

    # Model 2 — General BERT for comparison
    print("\nLoading General BERT (baseline)...")
    general = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    r2 = test_embeddings("General BERT (Baseline)", general, chunks, TEST_QUESTIONS)
    results.append(r2)

    # Summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Accuracy':>10} {'Latency':>10} {'Float Fixed':>12}")
    print("-"*70)
    for r in results:
        fixed = "✅ YES" if r["float_fixed"] else "❌ No"
        print(f"{r['embedding']:<35} {r['accuracy']:>9.1f}% {r['avg_latency']:>9.2f}s {fixed:>12}")

    # Save
    with open("domain_embedding_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to domain_embedding_results.json")

    # Key finding
    print(f"\n{'='*60}")
    print("KEY FINDING")
    print(f"{'='*60}")
    if any(r["float_fixed"] for r in results):
        print("✅ Domain-specific embeddings FIXED the float problem!")
        print("Conclusion: General embeddings insufficient for financial domain concepts.")
        print("Recommendation: Use FinBERT embeddings for financial RAG systems.")
    else:
        print("❌ Even domain-specific embeddings could not fix the float problem.")
        print("Conclusion: The float failure requires LLM fine-tuning, not just better embeddings.")
        print("Recommendation: Fine-tune GPT on financial corpora or use financial LLM.")

if __name__ == "__main__":
    run_domain_embedding_test()