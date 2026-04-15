from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
import shutil
import os

def create_vectorstore(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )
    db = FAISS.from_documents(chunks, embeddings)
    return db

def save_vectorstore(db, path = "vectorstore/faiss_index"):
    
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path) # deleting the file
        else:
            shutil.rmtree(path) # delete folder
    db.save_local(path)

def load_vectorstore(path="vectorstore/faiss_index"):

    embeddings = HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2"
    )
    return FAISS.load_local(path,embeddings)
