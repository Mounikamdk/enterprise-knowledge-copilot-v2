from fastapi import FastAPI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

app = FastAPI()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

@app.get("/ask")
def ask(query: str):
    docs = db.similarity_search(query, k=2)
    answer = "\n".join([doc.page_content for doc in docs])
    return {"answer": answer}