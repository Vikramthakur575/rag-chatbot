import os
import pickle
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")

st.sidebar.title("⚙️ Settings")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ["google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large"]
)

index_file = "vectorstore.pkl"

st.title("🤖 Your Personal Document Assistant")
st.write("Upload your PDFs or text files and chat with them just like talking to a human assistant!")

uploaded_files = st.file_uploader(
    "Drop your files here", type=["txt", "pdf"], accept_multiple_files=True
)

docs = []
if uploaded_files:
    for file in uploaded_files:
        if file.type == "application/pdf":
            reader = PdfReader(file)
            text = "".join([page.extract_text() for page in reader.pages])
            docs.append(text)
        elif file.type == "text/plain":
            text = file.read().decode("utf-8")
            docs.append(text)

if docs:
    with st.spinner("✨ Reading and understanding your documents..."):
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(" ".join(docs))
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_texts(chunks, embeddings)
        with open(index_file, "wb") as f:
            pickle.dump(vectorstore, f)
        st.success("✅ Your documents are ready to chat with!")
elif os.path.exists(index_file):
    with open(index_file, "rb") as f:
        vectorstore = pickle.load(f)
    st.sidebar.info("📂 Using previously saved documents")
else:
    vectorstore = None

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

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.chat_input("Ask me anything from your documents...")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            st.write("🤔 Let me think...")
            answer = qa_chain.run(user_input)
            st.markdown("💡 " + answer)

        st.session_state.chat_history.append(("user", user_input))
        st.session_state.chat_history.append(("assistant", answer))

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.markdown(msg)

