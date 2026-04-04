# Financial RAG System — Failure Analysis & Evaluation

A systematic study of why RAG systems fail on financial documents — 
and how to measure and fix those failures.

## The Problem
RAG systems confidently return wrong answers on financial and compliance 
documents. This project diagnoses exactly why and measures improvements rigorously.

## What This Project Does
- Ingests real financial documents (SEC filings, annual reports)
- Answers questions using Retrieval-Augmented Generation (RAG)
- Measures answer accuracy against a golden set of known answers
- Identifies and documents failure modes systematically
- Benchmarks multiple retrieval strategies statistically

## Tech Stack
- Python 3.11
- LangChain — RAG orchestration
- OpenAI GPT-3.5-turbo — answer generation
- FAISS — vector similarity search
- PyPDF — document loading

## Project Structure
financial-rag-system/
├── data/          # Financial documents
├── src/
│   ├── ingest.py     # Document loading and chunking
│   ├── rag.py        # RAG query pipeline
│   ├── retriever.py  # Retrieval strategies
│   └── evaluate.py   # Evaluation framework
└── tests/
## Current Findings
- Finding 1: Browser-saved PDFs create 3 chunks vs 1278 for proper PDFs
- Finding 2: Spread-out information (risk sections) returns vague answers
- Finding 3: Concentrated information (revenue tables) returns accurate answers

## Running The Project
```bash
pip install -r requirements.txt
python3 src/ingest.py
python3 src/rag.py
```