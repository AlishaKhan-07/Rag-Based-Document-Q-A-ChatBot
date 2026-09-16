"""
app.py
------
Role: The user-facing layer. A simple Streamlit chat UI that:
  1. Lets the user upload a PDF (or use data/sample.pdf)
  2. Runs it through document_loader -> text_splitter -> vector_store
  3. Lets the user ask questions, answered via rag_pipeline
  4. Shows which source chunks the answer came from (basic "explainability" -
     this is exactly the hook LectureLens will build on, showing
     video timestamps instead of page numbers)

Run with:  streamlit run src/app.py
"""

import os
import sys
import streamlit as st

# Allow "python -m" style imports (src.xxx) when run directly by Streamlit
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.document_loader import load_document
from src.text_splitter import split_documents
from src.vector_store import build_vector_store, load_vector_store, vector_store_exists, VECTORSTORE_DIR
from src.rag_pipeline import ask_question

st.set_page_config(page_title="RAG Document Q&A Chatbot", page_icon="📄")
st.title("📄 RAG-Based Document Q&A Chatbot")
st.caption("Upload a PDF and ask questions about it — powered by LangChain + FAISS + Groq")

DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- Sidebar: document setup ----------
with st.sidebar:
    st.header("1. Load a document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    use_sample = st.checkbox("Use data/sample.pdf instead", value=not uploaded_file)

    process_btn = st.button("Process document", type="primary")

    if process_btn:
        if uploaded_file is not None and not use_sample:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            file_path = os.path.join(DATA_DIR, "sample.pdf")

        if not os.path.exists(file_path):
            st.error(f"No file found at {file_path}. Upload a PDF or add one to data/.")
        else:
            with st.spinner("Loading document..."):
                docs = load_document(file_path)
            with st.spinner("Splitting into chunks..."):
                chunks = split_documents(docs)
            with st.spinner(f"Embedding {len(chunks)} chunks & building FAISS index..."):
                vector_store = build_vector_store(chunks)

            st.session_state["vector_store"] = vector_store
            st.session_state["doc_name"] = os.path.basename(file_path)
            st.success(f"Processed '{os.path.basename(file_path)}' — {len(chunks)} chunks indexed.")

    # Auto-load an existing index on first run, if present
    if "vector_store" not in st.session_state and vector_store_exists(VECTORSTORE_DIR):
        st.session_state["vector_store"] = load_vector_store()
        st.session_state["doc_name"] = "(previously indexed document)"

    if "vector_store" in st.session_state:
        st.info(f"Active document: **{st.session_state.get('doc_name')}**")

# ---------- Main: chat interface ----------
if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about the document...")

if question:
    if "vector_store" not in st.session_state:
        st.warning("Please process a document first (see sidebar).")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = ask_question(st.session_state["vector_store"], question)
            st.markdown(answer)

            with st.expander("📚 Sources used"):
                for i, src in enumerate(sources, 1):
                    page = src.metadata.get("page", "N/A")
                    st.markdown(f"**Chunk {i} (page {page})**")
                    st.caption(src.page_content[:300] + "...")

        st.session_state["messages"].append({"role": "assistant", "content": answer})
