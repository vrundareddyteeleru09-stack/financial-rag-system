# Financial RAG System — Failure Analysis & Evaluation

A systematic study of why RAG systems fail on financial documents —
and how to measure and fix those failures.

## The Problem
RAG systems confidently return wrong answers on financial and compliance
documents. This project diagnoses exactly why and measures improvements
rigorously — with a golden set evaluation framework and statistical tracking.

## What This Project Does
- Ingests real financial documents (SEC filings, annual reports)
- Answers questions using Retrieval-Augmented Generation (RAG)
- Measures answer accuracy against a golden set of 20 known answers
- Identifies and documents failure modes systematically
- Benchmarks retrieval strategies and proves improvements statistically

## Results Summary

| Version | Change | Accuracy |
|---------|--------|----------|
| v1 — Baseline | k=3 retrieval, basic chunking | 75% (apparent) |
| v2 — Eval fix | Fixed keyword matching bugs | 90% (true baseline) |
| v3 — k=6 | Increased chunks retrieved | 95% |

## Key Findings

**Finding 1 — PDF Quality Kills Accuracy Before RAG Even Starts**
Browser-saved PDFs created 3 chunks. Properly downloaded PDFs created
1278 chunks from the same document. Data quality is the first failure
mode — garbage in, garbage out.

**Finding 2 — Evaluation Frameworks Have Bugs Too**
3 of our initial 5 "failures" were evaluation bugs — keyword matching
too strict for formatted numbers and name variations. True baseline
accuracy was 90%, not 75%. Measuring your measurement system matters.

**Finding 3 — Retrieval Depth Matters For Spread-Out Information**
Increasing retrieved chunks from k=3 to k=6 improved risk_factors
category from 75% to 100%. Concentrated information retrieves well.
Spread-out information needs more context.

**Finding 4 — Domain-Specific Concepts Require Domain-Specific Retrieval**
The insurance "float" concept — Berkshire's core money-making mechanism
— was never retrieved correctly. The system retrieved general insurance
descriptions instead. Domain-specific concepts need specialized
embeddings or hybrid retrieval. This is Stage 2.

## Accuracy By Category (Final)

| Category | Accuracy |
|----------|----------|
| financial_figures | 100% |
| company_info | 100% |
| risk_factors | 100% |
| investments | 100% |
| strategy | 100% |
| business_overview | 80% |
| **Overall** | **95%** |

## Tech Stack
- Python 3.11
- LangChain — RAG orchestration
- OpenAI GPT-3.5-turbo — answer generation
- FAISS — vector similarity search
- PyPDF — document loading

## Project Structure
financial-rag-system/
├── data/               # Financial documents
├── src/
│   ├── ingest.py       # Document loading and chunking
│   ├── rag.py          # RAG query pipeline
│   ├── golden_set.py   # 20 test questions with known answers
│   ├── evaluate.py     # Evaluation framework
│   └── retriever.py    # Retrieval strategies
├── evaluation_results.json  # Latest evaluation results
└── tests/
## Running The Project
```bash
pip install -r requirements.txt
python3 src/ingest.py
python3 src/rag.py
python3 src/evaluate.py
```

## What's Next — Stage 2
- Benchmark 3 retrieval strategies: pure vector, metadata filtering,
  hybrid BM25+vector
- Prove statistically which strategy wins on financial documents
- Target: fix the insurance float failure mode
- Add LLM-as-a-judge evaluation layer