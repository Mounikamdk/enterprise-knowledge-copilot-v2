import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("📄 Enterprise Knowledge Copilot")
st.write("Ask questions from your PDF")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

query = st.text_input("Ask your question")

if st.button("Get Answer"):
    docs = db.similarity_search(query, k=2)
    answer = "\n".join([doc.page_content for doc in docs])
    st.write("### Answer:")
    st.write(answer)