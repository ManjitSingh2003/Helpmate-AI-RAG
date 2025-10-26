# src/utils/retrieval.py
import os
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.environ.get("CHROMA_PERSIST_DIR", "./chroma_db")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")

def load_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vect = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    return vect

def get_retriever(k=6):
    vect = load_vectorstore()
    retriever = vect.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever

