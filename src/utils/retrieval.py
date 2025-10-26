from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def get_retriever():
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
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

    # Create FAISS vector database
    vectorstore = FAISS.from_documents(splits, embeddings)

    # ✅ Return retriever for LangChain 0.2+ compatibility
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    return retriever
