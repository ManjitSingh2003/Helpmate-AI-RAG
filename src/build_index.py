# src/build_index.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# Configuration from environment
DOCS_PATH = os.environ.get("POLICY_DOCS_PATH", "./Policy-Documents")
PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 1000))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", 200))
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

def load_documents(path):
    loader = DirectoryLoader(path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} documents.")
    return docs

def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    print(f"✅ Created {len(splits)} text chunks.")
    return splits

def create_vectorstore(splits):
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vect = Chroma.from_documents(documents=splits, embedding=embedding_model, persist_directory=PERSIST_DIR)
    vect.persist()
    print(f"✅ Vector store persisted at: {PERSIST_DIR}")
    return vect

if __name__ == "__main__":
    docs = load_documents(DOCS_PATH)
    splits = split_documents(docs)
    create_vectorstore(splits)
