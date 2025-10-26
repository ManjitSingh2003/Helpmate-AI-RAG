# src/helpmate_app.py
import sys
import os

# Add the src folder to Python path
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

from dotenv import load_dotenv 
load_dotenv()

import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.retrieval import get_retriever
from src.prompts import custom_rag_prompt

st.set_page_config(page_title="HelpMate AI", layout="wide")
st.title("🧭 HelpMate AI — Insurance Policy Q&A")

st.markdown("""
**Ask questions about insurance policy documents and get concise, context-aware answers.**
""")

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
k = st.sidebar.slider("Number of retrieved chunks", 1, 10, 6)
model_name = st.sidebar.text_input("GROQ model", value=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))

@st.cache_resource
def init_llm():
    if "GROQ_API_KEY" not in os.environ:
        st.error("GROQ_API_KEY not found. Please set it in the environment or .env file.")
        return None
    return ChatGroq(model=model_name)

@st.cache_resource
def init_retriever(k):
    return get_retriever(k=k)

llm = init_llm()
retriever = init_retriever(k)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

query = st.text_input("💬 Ask your question:", "")

if st.button("🔍 Get Answer") and query.strip():
    with st.spinner("Generating answer..."):
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | custom_rag_prompt
            | llm
            | StrOutputParser()
        )
        answer = rag_chain.invoke(query)
        st.markdown("### ✅ Answer")
        st.write(answer)

        docs = retriever.get_relevant_documents(query)
        st.markdown("### 📄 Retrieved Sources")
        for i, d in enumerate(docs[:k], start=1):
            st.write(f"**Source {i}:**")
            st.caption(d.metadata.get('source', 'Unknown Document'))
            st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))

