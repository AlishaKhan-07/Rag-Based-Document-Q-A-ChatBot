"""
vector_store.py
----------------
Role: Build a FAISS vector index from text chunks, save it to disk, and
reload it later so we don't have to re-embed every time the app starts.

Flow:
    chunks (Documents) -> embed each chunk -> store vectors in FAISS
    -> save index to vectorstore/ folder -> load it back for retrieval
"""

import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from src.embeddings import get_embedding_model

VECTORSTORE_DIR = "vectorstore"
INDEX_FILE_NAME = "index.faiss"  # the actual file FAISS.save_local() writes


def vector_store_exists(path: str = VECTORSTORE_DIR) -> bool:
    """
    True only if a real, loadable FAISS index is present at `path`.
    Checking os.path.exists(path) alone is NOT enough — an empty
    'vectorstore/' folder (e.g. the placeholder shipped in this repo,
    or a folder left over from a failed build) exists but has nothing
    to load, which crashes FAISS.load_local() with a confusing
    "could not open index.faiss" error.
    """
    return os.path.exists(os.path.join(path, INDEX_FILE_NAME))


def build_vector_store(chunks: List[Document], save_path: str = VECTORSTORE_DIR) -> FAISS:
    """
    Embed all chunks and build a FAISS index from scratch, then persist
    it to disk so build_vector_store doesn't need to run every time.
    """
    embedding_model = get_embedding_model()
    vector_store = FAISS.from_documents(chunks, embedding_model)

    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"Vector store built and saved to '{save_path}' ({len(chunks)} chunks).")

    return vector_store


def load_vector_store(load_path: str = VECTORSTORE_DIR) -> FAISS:
    """
    Load a previously saved FAISS index from disk.
    allow_dangerous_deserialization=True is required by FAISS/LangChain
    because it unpickles metadata — safe here since WE created this file.
    """
    if not vector_store_exists(load_path):
        raise FileNotFoundError(
            f"No vector store found at '{load_path}'. Run build_vector_store() first "
            f"(or, in the app, process a document from the sidebar)."
        )

    embedding_model = get_embedding_model()
    vector_store = FAISS.load_local(
        load_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    return vector_store


def get_or_build_vector_store(chunks: List[Document] = None, path: str = VECTORSTORE_DIR) -> FAISS:
    """
    Convenience wrapper: load if it already exists on disk, otherwise
    build it fresh from the given chunks. Used by app.py so the index
    isn't rebuilt on every Streamlit rerun.
    """
    if vector_store_exists(path):
        print("Existing vector store found, loading from disk...")
        return load_vector_store(path)

    if chunks is None:
        raise ValueError("No existing vector store found and no chunks provided to build one.")

    print("No existing vector store found, building a new one...")
    return build_vector_store(chunks, path)


if __name__ == "__main__":
    # Quick manual test: python -m src.vector_store
    from src.document_loader import load_document
    from src.text_splitter import split_documents

    docs = load_document("data/sample.pdf")
    chunks = split_documents(docs)
    vs = build_vector_store(chunks)

    results = vs.similarity_search("What is this document about?", k=2)
    for r in results:
        print("---")
        print(r.page_content[:300])
