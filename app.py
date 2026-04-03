import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

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

# OpenAI model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=st.secrets["OPENAI_API_KEY"]
)

query = st.text_input("Ask your question")

if st.button("Get Answer"):
    relevant_docs = db.similarity_search(query, k=2)

    context = "\n".join([doc.page_content for doc in relevant_docs])

    prompt = f"""
    Answer the question based only on the resume context below.

    Resume Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    st.write("### Answer:")
    st.write(response.content)