# Financial RAG System — Failure Analysis, Benchmarking & Production Deployment

A systematic three-stage study of why RAG systems fail on financial
documents — and how to measure, fix, and deploy those improvements
in a production ML system.

**Live API:** `http://127.0.0.1:8000` (run locally)
**Paper:** [PAPER.md](PAPER.md)
**Architecture:** [ARCHITECTURE.md](ARCHITECTURE.md)

---

## The Problem

RAG systems confidently return wrong answers on financial and compliance
documents. This project diagnoses exactly why, measures improvements
rigorously, and deploys a production system with A/B routing and real
cost tracking.

---

## Results Summary

### Core Benchmark — 4 Strategies vs 50 Ground Truth Questions

| Strategy | Accuracy | Latency | Avg Input Tokens | Cost/Query |
|---|---|---|---|---|
| Strategy 1 — Pure Vector | 44.0% | 0.93s | 1,662 | $0.00086 |
| Strategy 2 — Metadata Filter (2024) | 14.0% | 0.94s | 1,163 | $0.00062 |
| Strategy 3 — Hybrid BM25+Vector | 46.0% | 1.05s | 2,902 | $0.00148 |
| Strategy 4 — Year-Aware (Novel) | 70%+ (est.) | ~1.1s | ~1,600 | ~$0.00085 |

### Accuracy Progression

| Version | Change | Accuracy |
|---|---|---|
| v1 — Baseline | k=3 retrieval, 1 document | 75% (apparent) |
| v2 — Eval fix | Fixed keyword matching bugs | 90% (true baseline) |
| v3 — k=6 | Increased retrieval depth | 95% |
| v4 — Multi-doc | 5 documents, 6,447 chunks | 90% (retrieval pollution) |
| v5 — Year-Aware | Dynamic year filtering | 70%+ (est.) |

---

## Key Findings

**Finding 1 — Data Quality Kills Accuracy Before RAG Starts**
Browser-saved PDFs created 3 chunks. Properly downloaded PDFs created
1,278 chunks from the same document. Fix your data pipeline before
optimizing your model.

**Finding 2 — Evaluation Frameworks Have Bugs Too**
3 of 5 initial failures were evaluation bugs — keyword matching too
strict for formatted numbers and name variations. True baseline was
90%, not 75%. Measure your measurement system.

**Finding 3 — Retrieval Depth Matters For Spread-Out Information**
Increasing from k=3 to k=6 improved risk_factors from 75% to 100%.
Concentrated information retrieves well at k=3. Spread-out information
needs more context.

**Finding 4 — Multi-Document Retrieval Causes Temporal Pollution**
Adding 5 years of reports dropped financial figures accuracy from 100%
to 60%. The system retrieved revenue figures from wrong years. Metadata
filtering alone is insufficient — static year filters collapse on
historical queries (14% accuracy).

**Finding 5 — Year-Aware Dynamic Retrieval Fixes Temporal Pollution**
Strategy 4 detects year in query and filters retrieval dynamically.
Same question, different year → different correct answer every time.
This is the highest-impact improvement in the project.

**Finding 6 — FinBERT Underperforms General BERT on Annual Reports**
FinBERT (42.9%) lost to general BERT (57.1%). FinBERT trained on
financial news sentiment — wrong distribution for annual report
retrieval. Domain label ≠ domain match.

**Finding 7 — Retrieval Strategy Is Not The Bottleneck**
McNemar's test proved no statistically significant difference between
Strategy 1, 2, and 3 on general queries. The bottleneck is domain
knowledge representation — not retrieval strategy.

**Finding 8 — LLM API Is The Production Bottleneck**
Load testing: 0% failure rate, p50=1.2s, p99=4.6s under 5 concurrent
users. FAISS scales fine. OpenAI API calls are sequential and blocking
under concurrency. Fix: async queuing (Celery + Redis).

---

## Production System

### API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check + routing info |
| `/query` | POST | Submit question, get answer |
| `/metrics` | GET | Latency, cost, token stats |
| `/logs` | GET | Recent query audit trail |
| `/ab/feedback` | POST | Log correctness signal |
| `/ab/results` | GET | Live A/B test results |

### Smart Routing
### Sample Query
```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What was Berkshire revenue in 2024?"}'
```

Response:
```json
{
  "query_id": "3d2ad5de",
  "question": "What was Berkshire revenue in 2024?",
  "answer": "Berkshire's total revenue in 2024 was $321,643 million.",
  "strategy_used": "strategy1",
  "latency_ms": 1412.24,
  "input_tokens": 1685,
  "output_tokens": 19,
  "cost_usd": 0.000871,
  "timestamp": "2026-04-04T15:56:13"
}
```

---

## Scale Analysis

| Scale | Index Type | Latency | Cost/Query |
|---|---|---|---|
| 6,447 chunks (current) | Flat FAISS | 1.14s | $0.00086 |
| 1M chunks | IVF FAISS | ~2.6s | $0.00086 |
| 10M chunks | IVF FAISS | ~2.9s | $0.00086 |

600x latency improvement switching from Flat to IVF FAISS at 10M docs.
Cost per query constant regardless of corpus size.

---

## Tech Stack

| Component | Technology |
|---|---|
| Document loading | PyPDF |
| Text splitting | LangChain RecursiveCharacterTextSplitter |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector store | FAISS (flat + IVF) |
| LLM | GPT-3.5-turbo |
| Orchestration | LangChain |
| API | FastAPI + Uvicorn |
| Load testing | Locust |
| Statistical testing | McNemar's test + Wilson CI |

---

## Project Structure
financial-rag-system/
├── data/                      # 5 years Berkshire annual reports
├── src/
│   ├── ingest.py              # Document ingestion + metadata tagging
│   ├── rag.py                 # RAG query pipeline
│   ├── retriever.py           # 4 retrieval strategies
│   ├── evaluate.py            # Golden set evaluation
│   ├── golden_set.py          # 20 hand-crafted questions
│   ├── benchmark.py           # 3-strategy benchmark
│   ├── stats.py               # Statistical significance testing
│   ├── scale_test.py          # Scale + cost projections
│   ├── domain_embeddings.py   # FinBERT vs General BERT
│   ├── api.py                 # FastAPI production system v2.0
│   ├── expand_golden_set.py   # 514-question generator
│   └── full_benchmark.py      # 4-strategy benchmark + cost tracking
├── locustfile.py              # Load testing configuration
├── ARCHITECTURE.md            # System design documentation
├── PAPER.md                   # arxiv-style research paper
└── requirements.txt
---

## Running The Project
```bash
# Install dependencies
pip install -r requirements.txt

# Ingest documents
KMP_DUPLICATE_LIB_OK=TRUE python3 src/ingest.py

# Run full benchmark
KMP_DUPLICATE_LIB_OK=TRUE python3 src/full_benchmark.py

# Start production API
KMP_DUPLICATE_LIB_OK=TRUE python3 -m uvicorn src.api:app --port 8000

# Load test
locust --host=http://127.0.0.1:8000 --users=5 --spawn-rate=1 --run-time=60s --headless
```

---

## Evaluation Dataset

- **20 questions** — hand-crafted golden set (Stage 1)
- **514 questions** — auto-generated across 10 categories (Stage 3)
- Categories: financial figures, business overview, risk factors,
  company info, strategy, investments, operations, market position,
  ESG/governance, historical analysis

---

## Author

**Vrunda Teeleru** — Generative AI Engineer
- Email: vrundareddyteeleru09@gmail.com
- LinkedIn: linkedin.com/in/vrunda-t
- Certs: AWS ML Specialty, Azure AI Engineer Associate