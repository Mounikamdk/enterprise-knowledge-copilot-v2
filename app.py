import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📄 Enterprise Knowledge Copilot")
st.write("Ask questions from your PDF")

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB dynamically
db = FAISS.from_documents(docs, embeddings)

query = st.text_input("Ask your question")

if st.button("Get Answer"):
    docs = db.similarity_search(query, k=2)
    answer = "\n".join([doc.page_content for doc in docs])
    st.write("### Answer:")
    st.write(answer)