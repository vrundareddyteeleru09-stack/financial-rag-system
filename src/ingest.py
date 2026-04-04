import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

def load_and_chunk_documents(data_folder: str):
    documents = []
    for filename in os.listdir(data_folder):
        if filename.endswith(".pdf"):
            print(f"Loading: {filename}")
            loader = PyPDFLoader(os.path.join(data_folder, filename))
            docs = loader.load()
            
            # Add year metadata from filename
            year = None
            for y in ["2020", "2021", "2022", "2023", "2024"]:
                if y in filename:
                    year = y
                    break
            
            for doc in docs:
                doc.metadata["year"] = year
                doc.metadata["source_file"] = filename
            
            documents.extend(docs)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def create_vector_store(chunks):
    print("Creating embeddings and vector store...")
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("vector_store")
    print("Vector store saved!")
    return vector_store

if __name__ == "__main__":
    chunks = load_and_chunk_documents("data")
    create_vector_store(chunks)