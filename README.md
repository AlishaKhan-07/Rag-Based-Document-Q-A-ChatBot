# RAG-Based AI Chatbot for Document Question Answering

A simple, modular Retrieval-Augmented Generation (RAG) chatbot that answers
questions about an uploaded PDF, built with **LangChain + FAISS +
HuggingFace sentence-transformers + Groq (Llama 3.1)**.

Designed to later extend into **LectureLens: An Intelligent Explainable
Multimodal RAG System for Lecture Video Question Answering**.

## Architecture

```
PDF ──► document_loader.py ──► text_splitter.py ──► embeddings.py
                                                          │
                                                          ▼
                                                  vector_store.py (FAISS)
                                                          │
User question ─────────────────────────────────────────► │
                                                          ▼
                                              rag_pipeline.py (retrieve + Groq LLM)
                                                          │
                                                          ▼
                                                    app.py (Streamlit UI)
```

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. Get a free Groq API key: https://console.groq.com/keys
   Add it to `.env`:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```

   > The LLM model used is `openai/gpt-oss-20b` (set in `src/rag_pipeline.py`).
   > Groq periodically retires older models — if you ever get a "model not
   > found / decommissioned" error, check
   > [console.groq.com/docs/deprecations](https://console.groq.com/docs/deprecations)
   > for the current replacement and update `LLM_MODEL_NAME` in that one file.

3. Put a PDF at `data/sample.pdf` (or upload one from the app UI).

4. Run the app:
   ```bash
   streamlit run src/app.py
   ```

## Testing modules individually (phase by phase)

```bash
python -m src.document_loader     # Phase 2: check PDF loads correctly
python -m src.text_splitter       # Phase 3: check chunking
python -m src.embeddings          # Phase 4: check embedding model loads
python -m src.vector_store        # Phase 5: build FAISS index + test search
python -m src.rag_pipeline        # Phase 6: ask a test question end-to-end
```

## File Roles

| File | Role |
|---|---|
| `src/document_loader.py` | Loads PDF(s) into LangChain `Document` objects. Uses a loader-registry so new file types (video transcripts) can be added later. |
| `src/text_splitter.py` | Splits documents into overlapping ~1000-char chunks for better retrieval. |
| `src/embeddings.py` | Single shared embedding model (`all-MiniLM-L6-v2`) used everywhere for consistency. |
| `src/vector_store.py` | Builds/saves/loads the FAISS vector index. |
| `src/rag_pipeline.py` | Retrieves relevant chunks + calls Groq Llama 3.1 to generate the answer. |
| `src/app.py` | Streamlit chat UI tying everything together, with source citations shown. |
| `.env` | Stores `GROQ_API_KEY` (never commit this). |
| `requirements.txt` | All Python dependencies. |
| `vectorstore/` | Auto-generated FAISS index files (not committed to git). |

## Roadmap to LectureLens

| This project | LectureLens extension |
|---|---|
| `document_loader.py` loads PDF | Add loaders for video transcripts (Whisper) + slide OCR |
| `text_splitter.py` chunks by characters | Chunk by transcript timestamp segments |
| Single FAISS index | Multimodal index (text + slide-image embeddings) |
| Text-only answers | Answers with **video timestamp citations** ("this is explained at 12:34") |
| Basic prompt | Explainability layer: show *why* a chunk was retrieved (similarity scores, highlighted evidence) |

Because each phase here is already its own module with a clean interface,
extending to LectureLens mostly means writing new loaders/chunkers and
plugging them into the same `vector_store.py` / `rag_pipeline.py` — the
core RAG logic doesn't need to change.
