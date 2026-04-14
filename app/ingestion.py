from langchain_community.document_loaders import PyPDFLoader, TextLoader # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter # type: ignore
import os

def load_and_split(file_path):
    ext = os.path.splitext(file_path)[1]

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported file type")

    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)
    return chunks