# src/utils/embeddings.py
from langchain.embeddings import HuggingFaceEmbeddings

def get_embedding_model(model_name="sentence-transformers/all-mpnet-base-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)
