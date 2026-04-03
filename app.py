import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

st.title("📄 Enterprise Knowledge Copilot")
st.write("Ask questions from your PDF")

loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.from_documents(docs, embeddings)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]
)

chain = load_qa_chain(llm, chain_type="stuff")

query = st.text_input("Ask your question")

if st.button("Get Answer"):
    relevant_docs = db.similarity_search(query, k=2)
    answer = chain.run(
        input_documents=relevant_docs,
        question=query
    )
    st.write("### Answer:")
    st.write(answer)