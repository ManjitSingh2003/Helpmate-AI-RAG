# src/utils/retrieval.py
import os
from langchain_community.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def get_vectorstore():
    """
    Loads PDF documents from the policy-documents folder in repo root,
    splits them into chunks, creates embeddings, and returns a Chroma vectorstore.
    """
    # 'utils' folder -> go up 2 levels to repo root
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    docs_path = os.path.join(base_path, "Policy-Documents")  # folder in repo root

    if not os.path.exists(docs_path):
        raise FileNotFoundError(f"Directory not found: '{docs_path}'")

    # Load PDFs
    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No documents found in {docs_path}")

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    vectorstore = Chroma.from_documents(splits, embedding=embedding_model)

    return vectorstore
