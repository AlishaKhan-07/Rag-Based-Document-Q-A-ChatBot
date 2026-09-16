"""
rag-ai-chatbot / src
Core package for the RAG pipeline.

Modules:
    document_loader -> loads raw documents (PDF now, video/audio transcripts later for LectureLens)
    text_splitter    -> chunks documents into smaller pieces
    embeddings       -> wraps the embedding model (HuggingFace sentence-transformers)
    vector_store     -> builds / saves / loads the FAISS index
    rag_pipeline     -> ties retriever + LLM (Groq Llama 3.1) together
"""
