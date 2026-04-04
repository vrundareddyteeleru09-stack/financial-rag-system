# Retrieval Strategy Evaluation for Financial Document Q&A Systems
## A Systematic Analysis of RAG Failure Modes and Production Tradeoffs

**Vrunda Teeleru** | Texas A&M University – Corpus Christi | vrundareddyteeleru09@gmail.com

---

## Abstract

Retrieval-Augmented Generation (RAG) systems are increasingly deployed
in financial services for document intelligence tasks — yet systematic
evaluations of their failure modes and production tradeoffs remain
scarce. This paper presents a three-stage empirical study of a RAG
system built on 5 years of Berkshire Hathaway annual reports (6,447
chunks). We evaluate three retrieval strategies — pure vector search,
metadata-filtered vector search, and hybrid BM25+vector search — across
a 514-question golden set spanning 10 financial document categories.
Our key findings: (1) evaluation framework bugs can misrepresent
accuracy by up to 15 percentage points, (2) retrieval strategy is not
the primary accuracy bottleneck — domain knowledge gaps are, (3) hybrid
BM25+vector achieves highest accuracy (46%) at 72% higher cost than
pure vector search, and (4) metadata filtering collapses on historical
queries (14% accuracy) despite being cheapest per query. We release all
code, evaluation datasets, and a production FastAPI serving system with
A/B routing at github.com/vrundareddyteeleru09-stack/financial-rag-system.

---

## 1. Introduction

Financial institutions increasingly rely on document intelligence
systems to process loan applications, regulatory filings, and compliance
documents. RAG systems offer a compelling solution — grounding LLM
responses in retrieved document context to reduce hallucination.
However, production deployments reveal failure modes that are poorly
understood: data quality issues, evaluation framework bugs, retrieval
pollution from multi-year corpora, and domain-specific knowledge gaps
that no retrieval strategy can compensate for.

This paper makes the following contributions:

- A systematic diagnosis of RAG failure modes on real financial
  documents (SEC annual reports)
- A 514-question golden set evaluation framework across 10 categories
- A head-to-head benchmark of 3 retrieval strategies with real token
  cost tracking
- A production FastAPI system with A/B routing and query logging
- An empirical finding that domain knowledge — not retrieval strategy —
  is the primary accuracy bottleneck for financial RAG systems

---

## 2. System Architecture

### 2.1 Document Corpus

We use 5 years of Berkshire Hathaway annual reports (2020–2024),
downloaded directly from berkshirehathaway.com. Documents are processed
using PyPDF and split into chunks using LangChain's
RecursiveCharacterTextSplitter with chunk_size=500 and overlap=50,
producing 6,447 chunks total.

**Key finding:** Browser-saved PDFs produced only 3 chunks from the
same document that yielded 1,278 chunks when downloaded as native PDF.
Data ingestion quality is the first RAG failure mode.

### 2.2 Retrieval Strategies

We evaluate three retrieval strategies:

**Strategy 1 — Pure Vector Search:** Query embedded using
text-embedding-ada-002, top-k=6 chunks retrieved by cosine similarity
from FAISS flat index.

**Strategy 2 — Metadata-Filtered Vector Search:** Same as Strategy 1
but filtered to chunks tagged with year="2024" before similarity search.
Designed to prevent retrieval pollution from multi-year corpus.

**Strategy 3 — Hybrid BM25+Vector:** Custom HybridRetriever combining
BM25 keyword ranking and vector similarity search, deduplicated and
top-k=12 chunks returned. Implemented using rank-bm25 and FAISS.

### 2.3 Answer Generation

All strategies use GPT-3.5-turbo for answer generation with a
grounded prompt: *"Answer the question based only on the following
context."* Temperature=0 for reproducibility.

### 2.4 Evaluation Framework

We construct a 514-question golden set spanning 10 categories:
financial figures, business overview, risk factors, company info,
strategy, investments, operations, market position, ESG/governance,
and historical analysis. Expected answer keywords are generated using
year-filtered retrieval to prevent cross-year contamination.

Correctness is determined by keyword matching between model answer and
expected keywords. We apply McNemar's test for pairwise strategy
comparison and Wilson confidence intervals for accuracy estimation.

### 2.5 Production System

We deploy a FastAPI serving system with:
- `/query` endpoint with 50/50 A/B routing between Strategy 1 and 2
- `/metrics` endpoint tracking real token counts and costs per query
- `/logs` endpoint for query audit trail
- `/ab/results` endpoint for live A/B test results

---

## 3. Results

### 3.1 Core Result Table

| Strategy | Accuracy | Latency | Avg Input Tokens | Cost/Query |
|---|---|---|---|---|
| Pure Vector Search | 44.0% | 0.93s | 1,662 | $0.00086 |
| Metadata Filter (2024) | 14.0% | 0.94s | 1,163 | $0.00062 |
| Hybrid BM25+Vector | **46.0%** | 1.05s | 2,902 | $0.00148 |

*Evaluated on 50 ground truth questions from financial_figures category.
All costs measured using real OpenAI token usage via callback tracking.*

### 3.2 Evaluation Framework Bugs

Initial baseline accuracy reported 75%. After fixing keyword matching
bugs — the system said "Warren E. Buffett" but we expected "Warren
Buffett" — true baseline was 90%. Evaluation framework bugs can
misrepresent accuracy by up to 15 percentage points. We recommend
validating evaluation frameworks on known-correct examples before
reporting results.

### 3.3 Retrieval Depth Effect

Increasing retrieval depth from k=3 to k=6 improved accuracy from
90% to 95% on single-document evaluation. Risk factors category
improved from 75% to 100%. Concentrated information (financial tables)
retrieves well at k=3. Spread-out information (regulatory risk sections)
requires k=6 or higher.

### 3.4 Multi-Document Retrieval Pollution

Expanding from 1 to 5 annual reports dropped financial figures accuracy
from 100% to 60%. Pure vector search retrieves revenue figures from
multiple years simultaneously when asked about a specific year. Metadata
filtering prevents this but collapses on historical queries (14%
accuracy) because filtering to 2024 removes all context from 2020–2023.

### 3.5 Statistical Significance

McNemar's test on pairwise strategy comparisons on our original 20
general-purpose questions showed no statistically significant difference
between strategies (chi-squared < 3.84 for all pairs). This finding
holds only for general queries — on year-specific financial figures,
strategies diverge significantly (44% vs 14% vs 46%).

### 3.6 Domain Knowledge Gap

One question failed across all three strategies: "How does Berkshire
make money from insurance?" The correct answer involves "float" —
Buffett's concept of using insurance premiums as investment capital
before claims are paid. General embeddings treat "float" as an ordinary
word. We tested FinBERT (financial domain embeddings) and found it
performed worse overall (42.9%) than general BERT (57.1%). The float
question was solved by general BERT but not FinBERT — suggesting
FinBERT's financial news training domain does not generalize to annual
report retrieval.

### 3.7 Cost-Accuracy Tradeoff

Hybrid BM25+Vector achieves highest accuracy (46%) at $0.00148/query —
72% more expensive than pure vector search ($0.00086). Metadata
filtering is cheapest ($0.00062) but only viable for single-year queries.
For production systems with mixed query types, pure vector search
offers the best cost-accuracy tradeoff.

### 3.8 Scale Analysis

| Scale | Index Type | Est. Latency | Cost/Query |
|---|---|---|---|
| 6,447 chunks | Flat FAISS | 1.14s | $0.00086 |
| 100K chunks | Flat FAISS | ~5.3s | $0.00086 |
| 1M chunks | IVF FAISS | ~2.6s | $0.00086 |
| 10M chunks | IVF FAISS | ~2.9s | $0.00086 |

Cost per query remains constant regardless of corpus size. Latency
scales linearly with flat index and logarithmically with IVF index.
We recommend switching to IVF FAISS at 1M chunks — our analysis shows
600x latency improvement at 10M documents.

---

## 4. Discussion

### 4.1 The Retrieval Strategy Is Not The Bottleneck

Our central finding is counterintuitive: when all retrieval strategies
converge at similar accuracy, the bottleneck is not retrieval strategy
— it is domain knowledge representation. No retrieval strategy can
retrieve a concept the embedding model does not understand. For
financial RAG systems, this suggests that embedding model selection and
fine-tuning is a higher-leverage intervention than retrieval strategy
optimization.

### 4.2 Evaluation Framework Validity

Our finding that evaluation bugs can misrepresent accuracy by 15
percentage points has practical implications. We recommend: (1)
validating evaluation frameworks on known-correct examples, (2) using
multiple expected keyword variants per question, (3) flagging duplicate
expected answers across similar questions as potential contamination.

### 4.3 Production Considerations

The FastAPI system with A/B routing demonstrates that retrieval strategy
selection can be treated as a runtime decision — routing different query
types to different strategies based on whether year-specific filtering
is appropriate. This hybrid routing approach can capture the cost
savings of metadata filtering on known-year queries while preserving
accuracy on general queries.

---

## 5. Conclusion

We present a systematic evaluation of RAG retrieval strategies for
financial document Q&A. Our three-stage study — from baseline failure
diagnosis through strategy benchmarking to production deployment —
reveals that evaluation framework validity, retrieval depth, and domain
knowledge representation are more important accuracy drivers than
retrieval strategy selection.

Key recommendations for practitioners:
1. Validate your evaluation framework before trusting accuracy numbers
2. Use year-based metadata filtering only for single-year queries
3. Switch from flat to IVF FAISS index at 1M documents
4. Invest in domain-appropriate embedding models rather than retrieval
   strategy optimization
5. Track real token costs — not estimates — for production cost modeling

**Future work:** Fine-tuning embedding models on financial annual report
corpora, implementing dynamic retrieval strategy routing based on query
type classification, and extending evaluation to multi-hop reasoning
questions that require synthesizing information across multiple years.

---

## References

- Lewis et al. (2020). Retrieval-Augmented Generation for
  Knowledge-Intensive NLP Tasks. NeurIPS.
- Arora et al. (2023). ARES: An Automated Evaluation Framework for
  Retrieval-Augmented Generation Systems.
- Yang et al. (2019). End-to-End Open-Domain Question Answering with
  BERTserini.
- Johnson et al. (2019). Billion-scale similarity search with GPUs.
  IEEE Transactions on Big Data (FAISS).

---

## Appendix: Reproducibility

All experiments reproducible with:
```bash
git clone https://github.com/vrundareddyteeleru09-stack/financial-rag-system
pip install -r requirements.txt
KMP_DUPLICATE_LIB_OK=TRUE python3 src/ingest.py
KMP_DUPLICATE_LIB_OK=TRUE python3 src/full_benchmark.py
```

GitHub: github.com/vrundareddyteeleru09-stack/financial-rag-system