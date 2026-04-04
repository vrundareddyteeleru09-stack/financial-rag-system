import os
from typing import List
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "vector_store")

def get_strategy1_retriever(k=6):
    """Strategy 1 — Pure vector search (baseline)"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_strategy2_retriever(year_filter=None, k=6):
    """Strategy 2 — Metadata filtering + vector search"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    if year_filter:
        search_kwargs = {"k": k, "filter": {"year": year_filter}}
    else:
        search_kwargs = {"k": k}
    return vector_store.as_retriever(search_kwargs=search_kwargs)

class HybridRetriever(BaseRetriever):
    """Strategy 3 — Hybrid BM25 + Vector search"""
    bm25_retriever: BM25Retriever
    vector_retriever: any
    k: int = 6

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        bm25_docs = self.bm25_retriever.invoke(query)
        vector_docs = self.vector_retriever.invoke(query)
        seen = set()
        combined = []
        for doc in bm25_docs + vector_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                combined.append(doc)
        return combined[:self.k * 2]

def get_strategy3_retriever(k=6):
    """Strategy 3 — Hybrid BM25 + Vector search"""
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    data_folder = os.path.join(BASE_DIR, "data")
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_folder, filename))
            documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        VECTOR_STORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )
    vector = vector_store.as_retriever(search_kwargs={"k": k})

    return HybridRetriever(
        bm25_retriever=bm25,
        vector_retriever=vector,
        k=k
    )