"""
embeddings.py
-------------
Role: Provide a single, reusable embedding model object for the whole
project. Both vector_store.py (when building the FAISS index) and
rag_pipeline.py (when embedding the user's query) must use the SAME
embedding model — otherwise similarity search breaks. Centralizing it
here guarantees that.

Model choice:
    "sentence-transformers/all-MiniLM-L6-v2" — small (~80MB), fast, runs
    fully on CPU (no GPU needed), and gives solid retrieval quality for
    a student project. Good enough for LectureLens later too; can be
    swapped for a bigger model (e.g. bge-base) with one line change.
"""

from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns a LangChain-compatible HuggingFace embedding model.
    `normalize_embeddings=True` so cosine similarity works cleanly with FAISS.
    """
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


if __name__ == "__main__":
    # Quick manual test: python src/embeddings.py
    model = get_embedding_model()
    vector = model.embed_query("What is a RAG pipeline?")
    print(f"Embedding model loaded: {EMBEDDING_MODEL_NAME}")
    print(f"Embedding dimension: {len(vector)}")
