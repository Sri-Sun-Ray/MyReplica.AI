from langchain_openai import OpenAIEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore

def create_vectorstore(chunks):

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db

def save_vectorstore(db, path = "vectorstore/faiss_index"):
    db.save_local(path)

def load_vectorstore(path="vectorstore/faiss_index"):

    embeddings = OpenAIEmbeddings()
    return FAISS.load_local(path,embeddings)
