# HelpMate AI — Generative Search System for Insurance Policy Documents

> **HelpMate AI** is an advanced Retrieval-Augmented Generation (RAG) system designed to revolutionize how insurance policy documents are searched and understood. The system allows users to ask natural language questions and receive precise, context-aware answers derived directly from **200+ real insurance policy documents**.

🔗 **Live App:** [Access the Streamlit App](https://manjitsingh2003-helpmate-ai-rag-helpmate-app-ec8big.streamlit.app/)

---

## 🧠 Project Overview

Navigating through large, unstructured insurance documents can be time-consuming and confusing for users. HelpMate AI simplifies this process by combining **semantic retrieval** and **generative response synthesis** to deliver accurate, interpretable answers. Built using **SBERT embeddings**, **ChromaDB**, **LangChain**, and **GROQ AI**, this open-source system is optimized for cost-efficiency, scalability, and high accuracy.

The project was developed as part of the **Postgraduate Program in Data Science and AI at IIIT Bangalore** and has now been **deployed on Streamlit** for public access.

---

## 🧩 Key Highlights

* Built a **semantic retrieval and generation pipeline** integrating **SBERT**, **ChromaDB**, **LangChain**, and **GROQ AI**.
* Preprocessed and indexed **200+ policy documents** with optimized chunking and recursive text splitting.
* Developed a **contextual generative search interface** for querying unstructured insurance policy text.
* Implemented **embedding workflows and metadata handling** to ensure coherent contextual responses.
* Focused on **accuracy, scalability, and cost-efficiency** using open-source models with **zero paid APIs**.
* Delivered a **domain-specific generative search system** adaptable to other sectors like law, finance, and healthcare.

✅ **Outcome:** Delivered a scalable, interpretable, and domain-tuned RAG system that helps users quickly interpret, compare, and understand complex insurance policies through intelligent, human-like conversations.

---

## ⚙️ Tech Stack

| Component                | Technology                                    |
| ------------------------ | --------------------------------------------- |
| **Programming Language** | Python                                        |
| **Framework**            | LangChain                                     |
| **Embedding Model**      | SBERT (Sentence-BERT)                         |
| **Vector Store**         | ChromaDB / FAISS                              |
| **LLM Backend**          | GROQ AI                                       |
| **Frontend**             | Streamlit                                     |
| **Development Tools**    | Jupyter Notebook, dotenv, PyPDF, NumPy, FAISS |

---

## 📂 Repository Structure

```
Helpmate-AI-RAG/
├─ Policy-Documents/               # Insurance policy PDFs for ingestion
├─ src/                            # Core modules of the RAG pipeline
│  ├─ __init__.py
│  ├─ build_index.py               # Builds and saves Chroma/FAISS vector index
│  ├─ prompts.py                   # Prompt templates for contextual generation
│  ├─ utils/
│     ├─ __init__.py
│     ├─ embeddings.py             # SBERT embedding generation via Groq API
│     ├─ loader.py                 # PDF ingestion and recursive text splitting
│     ├─ retrieval.py              # Context retrieval logic
│
├─ helpmate_app.py                 # Streamlit application for live querying
├─ Project_HelpMateAI_Code_Groq.ipynb  # End-to-end demonstration notebook
├─ HelpMate_AI_RAG_Documentation.pdf   # Technical documentation and architecture
├─ requirements.txt                # Python dependencies
└─ .gitignore
```

---

## 🧰 Setup Instructions

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/ManjitSingh2003/Helpmate-AI-RAG.git
cd Helpmate-AI-RAG
```

### 2️⃣ Create and Activate Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.\.venv\Scripts\activate    # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add API Keys

Create a `.env` file in the root directory:

```
GROQ_API_KEY=your_groq_api_key_here
LANGCHAIN_API_KEY=your_langchain_api_key_here
```

> ⚠️ Never commit `.env` or any API keys to GitHub.

### 5️⃣ Build the Index

```bash
python src/build_index.py
```

This processes policy documents, generates embeddings, and stores them in a Chroma or FAISS vector store.

### 6️⃣ Launch the Application

```bash
streamlit run helpmate_app.py
```

Then open your browser at `http://localhost:8501` or use the deployed app link above.

---

## 🧬 How It Works

1. **Document Ingestion:** PDFs are parsed and split into optimized chunks using recursive text splitting.
2. **Embedding Generation:** Text chunks are vectorized using **SBERT embeddings** via the Groq API.
3. **Indexing:** Embeddings are stored in **ChromaDB** (or FAISS) for efficient semantic search.
4. **Query Retrieval:** LangChain retrieves the top-k most relevant chunks for a given query.
5. **Generative Response:** GROQ AI constructs a coherent, contextually grounded answer using retrieved passages.
6. **User Interaction:** The Streamlit app provides a seamless chat-like interface for end users.

---

## 🧪 Example Queries

| User Query                                                  | Example Response                                                                                       |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| *“What is the waiting period for pre-existing conditions?”* | Returns the clause specifying the waiting period (e.g., 48 months) extracted directly from the policy. |
| *“What’s covered under accidental death benefits?”*         | Summarizes relevant sections of the policy document describing eligibility and coverage terms.         |

---

## 🌟 Impact

* Enhanced accessibility and interpretability of complex insurance policies.
* Delivered instant, contextually relevant answers from massive document sets.
* Built a fully open-source, **zero-cost RAG system** leveraging SBERT, LangChain, and Groq AI.
* Designed for scalability and domain adaptation across diverse enterprise use cases.

---

## 🔍 Future Enhancements

* Add document **summarization**, **comparison**, and **sentiment analysis** modules.
* Implement **feedback-based re-ranking** to improve answer quality.
* Integrate **multilingual policy support** for broader applicability.

---

## 📘 References

* [LangChain Documentation](https://python.langchain.com)
* [Groq API Documentation](https://groq.com)
* [Streamlit Documentation](https://docs.streamlit.io)

---

## 👨‍💻 Author

**Manjit Singh**

PGP in Data Science & AI - *IIIT Bangalore*

---
