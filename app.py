import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain

st.title("📄 Enterprise Knowledge Copilot")
st.write("Ask questions from your PDF")

# Load PDF
loader = PyPDFLoader("data/sample.pdf")
docs = loader.load()

# Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create vector DB
db = FAISS.from_documents(docs, embeddings)

# OpenAI LLM
llm = ChatOpenAI(
    temperature=0,
    model="gpt-3.5-turbo",
    api_key=st.secrets["OPENAI_API_KEY"]
)

chain = load_qa_chain(llm, chain_type="stuff")

query = st.text_input("Ask your question")

if st.button("Get Answer"):
    docs = db.similarity_search(query, k=2)
    answer = chain.run(input_documents=docs, question=query)
    st.write("### Answer:")
    st.write(answer)