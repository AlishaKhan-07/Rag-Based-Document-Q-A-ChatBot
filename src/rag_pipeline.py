"""
rag_pipeline.py
----------------
Role: The "R" + "A" + "G" glue. Given a user question:
  1. Retrieve the top-k most relevant chunks from FAISS (Retrieval)
  2. Stuff them into a prompt as context (Augmentation)
  3. Ask Groq's Llama 3.1 to answer using ONLY that context (Generation)

This is the piece you'll extend for LectureLens: swap the retriever's
source (transcript chunks + slide-OCR chunks instead of PDF chunks) and
the rest of this file barely changes.
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

LLM_MODEL_NAME = "openai/gpt-oss-20b"  # fast + free-tier friendly on Groq
# Note: llama-3.1-8b-instant was Groq's official pick for this role, but Groq
# deprecated and fully decommissioned it (free/developer tier) on 16 Aug 2026.
# openai/gpt-oss-20b is Groq's recommended 1:1 replacement. If Groq deprecates
# this one too in future, check console.groq.com/docs/deprecations for the
# current replacement and just update this one line.

RAG_PROMPT_TEMPLATE = """You are a helpful assistant that answers questions
strictly using the provided context from a document. If the answer is not
present in the context, say "I couldn't find that in the document" instead
of guessing.

Context:
{context}

Question: {question}

Answer clearly and concisely:"""


def get_llm() -> ChatGroq:
    """
    Returns the Groq-hosted Llama 3.1 chat model.
    Requires GROQ_API_KEY to be set in .env
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Add it to your .env file. "
            "Get a free key at https://console.groq.com/keys"
        )

    return ChatGroq(
        model=LLM_MODEL_NAME,
        api_key=api_key,
        temperature=0.2,  # low temperature -> factual, less creative
    )


def format_docs(docs) -> str:
    """Join retrieved chunks into a single context string for the prompt."""
    return "\n\n".join(doc.page_content for doc in docs)


def build_rag_chain(vector_store, k: int = 4):
    """
    Builds a runnable RAG chain using LangChain Expression Language (LCEL).
    `k` = how many chunks to retrieve per question.
    """
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain, retriever


def ask_question(vector_store, question: str, k: int = 4):
    """
    Convenience function: builds the chain and answers one question.
    Also returns the source chunks used, for showing citations in the UI.
    """
    chain, retriever = build_rag_chain(vector_store, k=k)
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources


if __name__ == "__main__":
    # Quick manual test: python -m src.rag_pipeline
    from src.vector_store import load_vector_store

    vs = load_vector_store()
    question = "Summarize the main topic of this document."
    answer, sources = ask_question(vs, question)

    print("Q:", question)
    print("A:", answer)
    print(f"\nUsed {len(sources)} source chunk(s).")
