import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

def load_vector_store():
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vector_store

def ask_question(question: str):
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": 6})
    
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the question based only on the following context:
    {context}
    
    Question: {question}
    """)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke(question)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    print("-" * 50)
    return answer

if __name__ == "__main__":

    questions = [
    "What was Berkshire Hathaway's total revenue in 2024?",
    "What are Berkshire Hathaway's main business segments?",
    "What risks does Berkshire Hathaway mention in this report?"
]
    
    for q in questions:
        ask_question(q)