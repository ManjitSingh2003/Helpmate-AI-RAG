# src/helpmate_app.py
import sys
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.utils.retrieval import get_retriever
from src.prompts import custom_rag_prompt

# --- Path Setup ---
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), "utils"))

# --- Load environment variables ---
load_dotenv()

# --- Streamlit UI ---
st.set_page_config(page_title="HelpMate AI", layout="wide")
st.title("🧭 HelpMate AI — Insurance Policy Q&A")
st.markdown("""
**Ask questions about insurance policy documents and get concise, context-aware answers.**
""")

# --- Sidebar controls ---
st.sidebar.header("⚙️ Configuration")
k = st.sidebar.slider("Number of retrieved chunks", 1, 10, 6)
model_name = st.sidebar.text_input(
    "GROQ model", 
    value=os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
)

# --- Initialize LLM ---
@st.cache_resource
def init_llm(model_name):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("⚠️ GROQ_API_KEY not found. Please set it in your .env file.")
        return None
    return ChatGroq(model=model_name, groq_api_key=api_key)

# --- Initialize retriever ---
def init_retriever(k):
    return get_retriever(k=k)

llm = init_llm(model_name)
retriever = init_retriever(k)

# --- Helper function ---
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- User input ---
query = st.text_input("💬 Ask your question:", "")

if st.button("🔍 Get Answer") and query.strip():
    with st.spinner("Generating answer..."):
        try:
            # --- Define RAG chain ---
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | custom_rag_prompt
                | llm
                | StrOutputParser()
            )

            # --- Generate answer ---
            answer = rag_chain.invoke(query)
            st.markdown("### ✅ Answer")
            st.write(answer)

            # --- Retrieve supporting documents ---
            docs = retriever.invoke(query)  # ✅ Replaced get_relevant_documents()
            st.markdown("### 📄 Retrieved Sources")
            for i, d in enumerate(docs[:k], start=1):
                st.write(f"**Source {i}:**")
                st.caption(d.metadata.get("source", "Unknown Document"))
                st.write(d.page_content[:600] + ("..." if len(d.page_content) > 600 else ""))

        except Exception as e:
            st.error(f"⚠️ An error occurred: {str(e)}")

