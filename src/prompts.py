# src/prompts.py
from langchain_core.prompts import PromptTemplate

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know—don't try to make one up.
Use three sentences maximum and keep the answer as concise as possible.
Always end the answer with "thanks for asking!".

{context}

Question: {question}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)
