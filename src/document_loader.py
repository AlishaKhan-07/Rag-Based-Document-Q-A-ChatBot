"""
document_loader.py
-------------------
Role: Load raw source documents from disk and convert them into LangChain
`Document` objects (text + metadata) that the rest of the pipeline can use.

Why it's written this way (for future LectureLens extension):
    Right now we only load PDFs. Later, LectureLens will need to load
    lecture video transcripts (from Whisper) and slide text (from OCR).
    Instead of hard-coding "load a PDF" everywhere, we use a small
    loader-registry pattern keyed by file extension. To add video support
    later, you'll just write `load_transcript()` and register it under
    ".txt" / ".vtt" — nothing else in the project has to change.
"""

import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document


def load_pdf(file_path: str) -> List[Document]:
    """
    Load a single PDF file and return a list of LangChain Document objects
    (one Document per page, by default from PyPDFLoader).
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # Tag each page with clean metadata (useful later for citations in UI)
    for doc in documents:
        doc.metadata["source_file"] = os.path.basename(file_path)

    return documents


# Registry: extension -> loader function.
# This is the extension point for LectureLens (add ".vtt": load_transcript, etc.)
LOADER_REGISTRY = {
    ".pdf": load_pdf,
}


def load_document(file_path: str) -> List[Document]:
    """
    Generic entry point. Picks the right loader based on file extension.
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in LOADER_REGISTRY:
        raise ValueError(
            f"No loader registered for '{ext}' files. "
            f"Supported: {list(LOADER_REGISTRY.keys())}"
        )
    return LOADER_REGISTRY[ext](file_path)


def load_all_documents(data_dir: str) -> List[Document]:
    """
    Load every supported file inside a directory (e.g. data/).
    Used when you have multiple PDFs, not just one sample.pdf.
    """
    all_docs: List[Document] = []
    for fname in os.listdir(data_dir):
        ext = os.path.splitext(fname)[1].lower()
        if ext in LOADER_REGISTRY:
            fpath = os.path.join(data_dir, fname)
            all_docs.extend(load_document(fpath))
    return all_docs


if __name__ == "__main__":
    # Quick manual test: python src/document_loader.py
    docs = load_document("data/sample.pdf")
    print(f"Loaded {len(docs)} page(s).")
    if docs:
        print("--- Preview of page 1 ---")
        print(docs[0].page_content[:500])
        print("Metadata:", docs[0].metadata)
