# src/utils/retrieval.py
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_vectorstore():
    """
    Loads PDF documents, splits them into chunks, creates embeddings,
    and returns a Chroma vectorstore.
    This function can be cached safely.
    """
    docs_path = os.getenv("DOCUMENT_PATH", "/content/drive/MyDrive/Policy-Documents")
    
    # Load PDFs from the directory
    loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    if not documents:
        raise ValueError(f"No documents found in {docs_path}")

    # Split documents into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(documents)

    # Create embeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Initialize Chroma vectorstore
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

    return vectorstore

def get_retriever(k=6):
    """
    Returns a retriever object from a cached vectorstore with
    top-k similarity search.
    """
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever
