# app.py
import os
import pickle
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from PyPDF2 import PdfReader
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

# --------------------------
# Sidebar settings
# --------------------------
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose model",
    ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]
)

index_file = "vectorstore.pkl"

st.title("📄 RAG Chatbot")
st.write("Upload documents and ask questions in a conversational way!")

# --------------------------
# Upload documents
# --------------------------
uploaded_files = st.file_uploader(
    "Upload TXT or PDF files", type=["txt", "pdf"], accept_multiple_files=True
)

docs = []

if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            pdf_reader = PdfReader(file)
            text = "".join([page.extract_text() for page in pdf_reader.pages])
            docs.append(text)
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            docs.append(text)

# --------------------------
# Process / Load Vectorstore
# --------------------------
if docs:
    with st.spinner("🔍 Processing documents..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, length_function=len
        )
        docs_chunks = text_splitter.split_text(" ".join(docs))

        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        vectorstore = FAISS.from_texts(docs_chunks, embeddings)

        with open(index_file, "wb") as f:
            pickle.dump(vectorstore, f)

        st.success("✅ Documents processed and saved!")
elif os.path.exists(index_file):
    with open(index_file, "rb") as f:
        vectorstore = pickle.load(f)
    st.sidebar.info("📂 Loaded previous document index")
else:
    vectorstore = None

# --------------------------
# Initialize LLM
# --------------------------
@st.cache_resource
def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_length=512)
    return HuggingFacePipeline(pipeline=pipe)

if vectorstore:
    llm = load_model(model_choice)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
    )

    # --------------------------
    # Chat Interface
    # --------------------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask a question about your documents...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            st.write("🤔 Thinking...")
            answer = qa_chain.run(user_input)
            st.markdown("**Answer:** " + answer)

        # Save chat history
        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", answer))

    # Display past chat
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)
