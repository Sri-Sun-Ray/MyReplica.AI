from fastapi import FastAPI, UploadFile

import shutil

from app.ingestion import load_and_split
from app.embeddings import create_vectorstore, save_vectorstore, load_vectorstore
from app.retrieval import create_qa_chain

app = FastAPI()

db = None
qa_chain = None

@app.post("/upload")
async def upload_pdf(file: UploadFile):

    file_path = f"data/raw/{file.filename}"

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file,buffer)
    chunks = load_and_split(buffer)
    global db, qa_chain
    db = create_vectorstore(chunks)
    save_vectorstore(db)
    qa_chain = create_qa_chain(db)

    return {"message": "File processed successfully"}

@app.get("/ask")
def ask_question(query:str):
    global qa_chain

    if not qa_chain:
        return {"error": "Upload a document First"}
    result = qa_chain.run(query)
    return {"answer": result}
