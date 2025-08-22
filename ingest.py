import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".csv"}

def load_documents(data_dir: str):
    data_path = Path(data_dir)
    docs = []
    for path in data_path.rglob("*"):
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
        if ext == ".pdf":
            loader = PyPDFLoader(str(path))
            docs.extend(loader.load())
        elif ext in {".txt", ".md"}:
            loader = TextLoader(str(path), encoding="utf-8")
            docs.extend(loader.load())
        elif ext == ".csv":
            loader = CSVLoader(str(path))
            docs.extend(loader.load())
    return docs

def chunk_documents(documents, chunk_size:int=1000, chunk_overlap:int=150):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

def build_and_save_vectorstore(chunks, persist_dir, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
    print(f"🔎 Using embeddings: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vs = FAISS.from_documents(chunks, embeddings)
    os.makedirs(persist_dir, exist_ok=True)
    vs.save_local(persist_dir)
    print(f"✅ Vectorstore saved to {persist_dir}")
    return persist_dir

if __name__ == "__main__":
    print("📂 Loading documents from data/ ...")
    docs = load_documents("data")
    print(f"Loaded {len(docs)} documents.")

    print("✂️ Splitting documents ...")
    chunks = chunk_documents(docs)
    print(f"Split into {len(chunks)} chunks.")

    print("⚡ Building vectorstore ...")
    build_and_save_vectorstore(chunks, "vectorstore")

