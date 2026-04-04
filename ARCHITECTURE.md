# Financial RAG System — Architecture Documentation

## System Overview

A production-grade RAG evaluation system for financial documents,
designed to systematically identify and measure failure modes at scale.

## Architecture Diagram

┌─────────────────────────────────────────────────────────────┐
│                     INGESTION PIPELINE                       │
├─────────────────────────────────────────────────────────────┤
│  PDF Documents → PyPDF Loader → Text Splitter → Embeddings  │
│                                      ↓                       │
│                              FAISS Vector Store              │
│                         (Flat Index < 1M docs)               │
│                         (IVF Index > 1M docs)                │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│                      QUERY PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│  User Query → Embedding → FAISS Search → Top-K Chunks       │
│                                              ↓               │
│                                    LLM (GPT-3.5-turbo)      │
│                                              ↓               │
│                                         Answer               │
└─────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────────────────────────────┐
│                   EVALUATION PIPELINE                        │
├─────────────────────────────────────────────────────────────┤
│  Answer → Golden Set Comparison → Accuracy Score            │
│        → Statistical Significance Testing                    │
│        → Latency Measurement                                 │
│        → Cost Analysis                                       │
└─────────────────────────────────────────────────────────────┘

## Key Design Decisions

### 1. Chunk Size: 500 tokens with 50-token overlap
**Decision:** 500 tokens balances context preservation with retrieval precision.
**Tradeoff:** Smaller chunks = better precision but less context per chunk.
**At scale:** Switch to 1024 tokens above 100K documents for better coverage.
**Evidence:** Chunk size analysis showed 1024 tokens optimal at reduced corpus size.

### 2. Retrieval Depth: k=6
**Decision:** Retrieve 6 chunks per query instead of standard k=3.
**Why:** Proved through evaluation that k=3 missed spread-out information (risk factors).
**Evidence:** k=6 improved accuracy from 90% to 95% on single-document corpus.
**Tradeoff:** More tokens sent to LLM = higher cost per query (~$0.002 vs ~$0.001).

### 3. FAISS Index Type: Flat → IVF at Scale
**Decision:** Use Flat index below 1M documents, IVF index above.
**Why:** At 10M documents, Flat index latency = 1,768s. IVF index latency = 2.89s.
**Evidence:** Scale analysis shows 600x latency improvement with IVF at 10M docs.
**Implementation:** Switch index type at 1M document threshold.

### 4. Metadata Tagging: Year-based filtering
**Decision:** Tag every chunk with source year during ingestion.
**Why:** Multi-year corpus causes retrieval pollution on time-specific queries.
**Evidence:** Adding 5 years of reports dropped financial figures accuracy from 100% to 60%.
**Tradeoff:** Metadata filtering adds 0.19s latency but prevents year confusion.

### 5. Retrieval Strategy: Pure Vector for General Queries
**Decision:** Use pure vector search as default strategy.
**Why:** Statistical analysis (McNemar's test) showed no significant difference between
pure vector, metadata filtering, and hybrid BM25+vector at 90% accuracy.
**Evidence:** All 3 strategies converged at exactly 90% — bottleneck is domain knowledge.
**Exception:** Use metadata filtering for year-specific financial queries.

## Scale Analysis Results

| Scale | Index Type | Retrieval Latency | Total Latency | Cost/Query |
|-------|-----------|-------------------|---------------|------------|
| 6,447 chunks | Flat | 0.34s | 1.14s | $0.0018 |
| 100K chunks | Flat | 5.27s | 6.07s | $0.0018 |
| 1M chunks | IVF | 1.80s | 2.60s | $0.0018 |
| 10M chunks | IVF | 2.09s | 2.89s | $0.0018 |

**Key insight:** Cost per query is constant regardless of corpus size.
Latency grows logarithmically with IVF index — production-viable at any scale.

## Failure Mode Analysis

| Failure Mode | Root Cause | Fix Applied | Result |
|-------------|-----------|-------------|--------|
| 3 chunks from browser PDF | Image-based PDF | Download text PDF | 1,278 chunks |
| 75% apparent accuracy | Evaluation keyword bugs | Fix keyword matching | 90% true accuracy |
| 60% on financial figures (multi-doc) | Retrieval pollution | Year metadata filter | Prevented |
| Insurance float not retrieved | Domain knowledge gap | Requires fine-tuning | Open problem |

## Production Recommendations

**For < 100K documents:**
- Flat FAISS index
- chunk_size=500, overlap=50
- k=6 retrieval
- Pure vector search

**For 100K - 1M documents:**
- Flat FAISS index with metadata filtering
- chunk_size=1024, overlap=100
- k=8 retrieval
- Hybrid BM25 + vector for domain-specific queries

**For > 1M documents:**
- IVF FAISS index (nlist=1024)
- chunk_size=1024, overlap=100
- k=10 retrieval
- Dedicated embedding model fine-tuned on domain

## Open Problems

1. **Insurance float failure** — domain-specific concepts require
   domain-specific embeddings. General OpenAI embeddings insufficient.

2. **Cross-year temporal reasoning** — system cannot compare metrics
   across years without explicit year filtering.

3. **Table extraction** — financial tables in PDFs lose structure
   during text extraction, reducing numerical accuracy.

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Document loading | PyPDF | Reliable text extraction |
| Text splitting | LangChain RecursiveCharacterTextSplitter | Respects sentence boundaries |
| Embeddings | OpenAI text-embedding-ada-002 | Best general-purpose embeddings |
| Vector store | FAISS | Production-grade, scales to billions |
| LLM | GPT-3.5-turbo | Best cost/performance ratio |
| Orchestration | LangChain | Production-ready RAG framework |
| Evaluation | Custom golden set + McNemar's test | Statistically rigorous |