# PolicyBot RAG API

This project provides a **Retrieval-Augmented Generation (RAG) API** built with **FastAPI**.  
It is designed to answer questions based on the content of a provided PDF or DOCX document.  
The system uses a **hybrid retrieval approach**, combining sparse and dense search methods to find the most relevant passages before a language model generates a concise answer.

---

## ✨ Features

- **PDF/DOCX Processing**: Extracts text from both PDF and DOCX file formats.  
- **Hybrid Retrieval**: Combines **BM25 (sparse)** and **Sentence-Transformers (dense)** search.  
- **Reciprocal Rank Fusion (RRF)**: Merges results from both retrieval methods.  
- **Cross-Encoder Re-ranking**: Re-ranks passages using a cross-encoder model.  
- **Extractive & Generative QA**: Supports extractive Q&A and LLM-based generative answers.  
- **FastAPI Backend**: Production-ready API with a clean request/response structure.  

---

## 🧠 How It Works

1. **Document Ingestion**: The API fetches a PDF or DOCX file from a given URL.  
2. **Text Extraction**: Extracts and cleans text from the document.  
3. **Chunking**: Splits text into smaller passages.  
4. **Indexing**:  
   - A sparse index (**Whoosh**) for BM25 keyword search.  
   - A dense index (**Sentence-Transformers**) for semantic search.  
5. **Hybrid Search**:  
   - Sparse index retrieves keyword-relevant passages.  
   - Dense index retrieves semantically relevant passages.  
6. **Fusion & Re-ranking**:  
   - Combines results using **RRF**.  
   - Applies cross-encoder re-ranking to find the most relevant passages.  
7. **Answer Generation**:  
   - If `USE_LLM_READER=True` and `OPENAI_API_KEY` is available → uses LLM (e.g., GPT-4o-mini).  
   - Otherwise → falls back to extractive QA.  
8. **API Response**: Returns answers in JSON format.  

---

## 🛠️ Configuration

You can configure the API by editing constants in `main.py` or via environment variables.

| Constant             | Description                                      | Default                                     |
|-----------------------|--------------------------------------------------|---------------------------------------------|
| `EMBED_MODEL_NAME`    | Model for dense embeddings                       | `intfloat/e5-base-v2`                       |
| `RERANK_MODEL_NAME`   | Cross-encoder model for re-ranking               | `cross-encoder/ms-marco-MiniLM-L-6-v2`      |
| `READER_MODEL`        | LLM for generative answers                       | `gpt-4o-mini`                               |
| `REQUIRE_BEARER`      | If true, requires bearer token for requests      | `True`                                      |
| `EXPECTED_BEARER`     | The expected bearer token                        | `Bearer ...`                                |
| `USE_LLM_READER`      | Toggle between LLM or extractive reader          | `True`                                      |
| `OPENAI_API_KEY`      | OpenAI API key (needed if `USE_LLM_READER=True`) | *(required)*                                |

---

## 💾 Installation

Clone the repository:

```bash
git clone https://github.com/Sneh9903/Policy_bot.git
cd Policy_bot

```
## 🚀 Usage

```bash
uvicorn main:app --reload
```

