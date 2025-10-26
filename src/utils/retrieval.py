# src/utils/retrieval.py
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv

load_dotenv()

def get_retriever(k=6):
    """
    Builds and returns a FAISS-based retriever for policy documents.
    Allows dynamic control of 'k' (number of retrieved chunks).
    """
    # Load all policy PDFs from the folder
    docs_path = os.getenv("POLICY_DOCS_PATH", "./Policy-Documents")

    documents = []
    for file in os.listdir(docs_path):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(docs_path, file))
            documents.extend(loader.load())

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Embed using SBERT model
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(splits, embeddings)

    # ✅ Return retriever with dynamic k
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return retriever


