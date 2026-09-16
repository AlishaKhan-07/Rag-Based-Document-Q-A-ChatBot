"""
text_splitter.py
-----------------
Role: Break large Document objects (full PDF pages) into smaller overlapping
chunks. Embedding models and LLM context windows work far better on small,
semantically coherent chunks than on whole pages/documents.

Why these numbers:
    chunk_size=1000, chunk_overlap=150 is a solid default for dense text
    (like lecture notes / technical PDFs). Overlap ensures a sentence that
    got cut at a chunk boundary still appears fully in at least one chunk.
"""

from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


def split_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    """
    Split a list of Documents into smaller chunked Documents.
    Metadata (source_file, page number) is preserved automatically
    by LangChain's splitter for each resulting chunk.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],  # tries paragraph -> sentence -> word
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add a simple chunk_id — useful later for LectureLens citations
    # (e.g. "which chunk / timestamp did this answer come from?")
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    return chunks


if __name__ == "__main__":
    # Quick manual test: python -m src.text_splitter
    from src.document_loader import load_document

    docs = load_document("data/sample.pdf")
    chunks = split_documents(docs)
    print(f"Original pages: {len(docs)} -> Chunks: {len(chunks)}")
    print("--- Preview of chunk 0 ---")
    print(chunks[0].page_content[:400])
    print("Metadata:", chunks[0].metadata)
