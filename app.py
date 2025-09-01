# import os
# import io
# import re
# import json
# import httpx
# from typing import List, Dict, Optional
# from fastapi import FastAPI, Header, HTTPException
# from pydantic import BaseModel, AnyHttpUrl
# from whoosh import index
# from whoosh.fields import Schema, TEXT, ID
# from whoosh.analysis import StemmingAnalyzer
# from whoosh.qparser import MultifieldParser
# from whoosh.filedb.filestore import RamStorage
# from sentence_transformers import SentenceTransformer
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch
# from pypdf import PdfReader
# import tiktoken

# # --------------- Config ---------------
# EMBED_MODEL_NAME = "intfloat/e5-base-v2"  # 768-dim, faster than large; good quality
# RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # efficient, strong reranker
# TOPK_SPARSE = 40
# TOPK_DENSE = 40
# FUSION_KEEP = 60
# RERANK_KEEP = 10
# CONTEXT_PASSAGES = 8
# MAX_PASSAGE_TOKENS = 220
# MAX_TOKENS_PER_EMBED_BATCH = 280_000
# REQUIRE_BEARER = True
# EXPECTED_BEARER = "Bearer b7a83688229ee229beb214ec38a7d7c9de760fc2a8c6fe7c3ef7d68c89ea7eac"

# USE_LLM_READER = True
# READER_MODEL = os.environ.get("READER_MODEL", "gpt-4o-mini")

# # --------------- Utilities ---------------
# def tokenizer():
#     try:
#         return tiktoken.get_encoding("cl100k_base")
#     except Exception:
#         return None

# TOK = tokenizer()

# def count_tokens(text: str) -> int:
#     if TOK:
#         return len(TOK.encode(text))
#     return int(len(text.split()) * 1.3)

# def safe_trim(text: str, max_tokens: int) -> str:
#     if count_tokens(text) <= max_tokens:
#         return text
#     words = text.split()
#     low, high = 0, len(words)
#     while low < high:
#         mid = (low + high) // 2
#         s = " ".join(words[:mid])
#         if count_tokens(s) <= max_tokens:
#             low = mid + 1
#         else:
#             high = mid
#     return " ".join(words[:high - 1])

# def normalize_ws(s: str) -> str:
#     return re.sub(r"\s+", " ", s).strip()

# # --------------- PDF handling ---------------
# async def fetch_pdf(url: str) -> bytes:
#     async with httpx.AsyncClient(timeout=60) as client:
#         r = await client.get(url)
#     if r.status_code != 200:
#         raise HTTPException(status_code=400, detail=f"Failed to fetch PDF: HTTP {r.status_code}")
#     ctype = r.headers.get("content-type", "")
#     if "pdf" not in ctype.lower():
#         pass
#     return r.content

# def extract_pdf_text(pdf_bytes: bytes) -> str:
#     reader = PdfReader(io.BytesIO(pdf_bytes))
#     pages = []
#     for p in reader.pages:
#         try:
#             pages.append(p.extract_text() or "")
#         except Exception:
#             pages.append("")
#     return "\n".join(pages)

# # --------------- Chunking ---------------
# def paragraph_split(text: str) -> List[str]:
#     parts = [normalize_ws(p) for p in re.split(r"\n{2,}", text) if p.strip()]
#     if not parts:
#         parts = [normalize_ws(text)]
#     return parts

# def sentenceish_split(paragraph: str) -> List[str]:
#     sents = re.split(r"(?<=[.!?])\s+", paragraph)
#     return [normalize_ws(s) for s in sents if s.strip()]

# def make_passages(text: str, target_tok: int = 220, overlap_tok: int = 40) -> List[str]:
#     paragraphs = paragraph_split(text)
#     passages = []
#     for para in paragraphs:
#         sents = sentenceish_split(para)
#         buf, t = [], 0
#         for s in sents:
#             ts = count_tokens(s)
#             if t + ts > target_tok and buf:
#                 passages.append(" ".join(buf))
#                 buf = [buf[-1]] if buf else []
#                 t = count_tokens(" ".join(buf))
#             buf.append(s)
#             t += ts
#         if buf:
#             passages.append(" ".join(buf))
#     passages = [safe_trim(p, MAX_PASSAGE_TOKENS) for p in passages if p.strip()]
#     return passages

# # --------------- Sparse index (Whoosh in-memory) ---------------
# def build_sparse_index(passages: List[str]) -> tuple:
#     schema = Schema(id=ID(stored=True, unique=True),
#                     text=TEXT(stored=True, analyzer=StemmingAnalyzer()))
#     storage = RamStorage()
#     idx = storage.create_index(schema)
#     writer = idx.writer()
#     for i, p in enumerate(passages):
#         writer.add_document(id=str(i), text=p)
#     writer.commit()
#     return idx, storage

# def bm25_search(idx, query: str, topk: int) -> List[tuple]:
#     qp = MultifieldParser(["text"], schema=idx.schema)
#     q = qp.parse(query)
#     with idx.searcher() as s:
#         res = s.search(q, limit=topk)
#         return [(hit["id"], float(hit.score)) for hit in res]

# # --------------- Dense embedding search (in-memory) ---------------
# class DenseIndexer:
#     def __init__(self, model_name: str):
#         self.model = SentenceTransformer(model_name)
#         self.embeddings = None
#         self.passages = None

#     def index(self, passages: List[str]):
#         self.passages = passages
#         embs = self.model.encode(passages, normalize_embeddings=True, convert_to_tensor=True, batch_size=64)
#         self.embeddings = embs

#     def search(self, query: str, topk: int) -> List[tuple]:
#         qvec = self.model.encode([query], normalize_embeddings=True, convert_to_tensor=True)
#         sims = (self.embeddings @ qvec.T).squeeze(1)
#         vals, idxs = torch.topk(sims, k=min(topk, sims.shape[0]))
#         out = []
#         for v, i in zip(vals.tolist(), idxs.tolist()):
#             out.append((str(i), float(v)))
#         return out

# # --------------- Reciprocal Rank Fusion ---------------
# def rrf(lists: List[List[tuple]], k: int = 60) -> List[str]:
#     scores = {}
#     for lst in lists:
#         for rank, (pid, _score) in enumerate(lst):
#             scores[pid] = scores.get(pid, 0.0) + 1.0 / (60 + rank)
#     ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
#     return [pid for pid, _ in ranked[:k]]

# # --------------- Cross-encoder reranker ---------------
# class CrossEncoder:
#     def __init__(self, model_name: str):
#         self.tok = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
#         self.model.eval()

#     @torch.no_grad()
#     def rerank(self, query: str, id2text: Dict[str, str], cand_ids: List[str], topk: int) -> List[str]:
#         pairs = [(query, id2text[c]) for c in cand_ids if c in id2text]
#         if not pairs:
#             return []
#         enc = self.tok(
#             [p[0] for p in pairs],
#             [p[1] for p in pairs],
#             truncation=True, padding=True, max_length=512, return_tensors="pt"
#         )
#         logits = self.model(**enc).logits.squeeze(-1)
#         order = torch.argsort(logits, descending=True).tolist()
#         ranked = [cand_ids[i] for i in order if i < len(cand_ids)]
#         return ranked[:topk]

# # --------------- LLM reader ---------------
# async def llm_answer(query: str, contexts: List[str]) -> str:
#     from openai import AsyncOpenAI
#     client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
#     blocks = []
#     for i, c in enumerate(contexts):
#         blocks.append(f"[{i+1}] {c}")
#     evidence = "\n\n".join(blocks)
#     system = "You are a precise assistant. Answer ONLY using the provided evidence and cite as [n]. If insufficient, say you cannot find it."
#     user = f"Question: {query}\n\nEvidence:\n{evidence}\n\nWrite a concise answer with citations."
#     resp = await client.chat.completions.create(
#         model=READER_MODEL,
#         messages=[{"role": "system", "content": system},
#                   {"role": "user", "content": user}],
#         temperature=0.2,
#         max_tokens=400
#     )
#     return resp.choices[0].message.content.strip()

# def extractive_answer(query: str, contexts: List[str]) -> str:
#     return "\n\n".join(contexts[:2]) if contexts else "No relevant context found."

# # --------------- FastAPI models ---------------
# class RunRequest(BaseModel):
#     documents: AnyHttpUrl
#     questions: List[str]

# class RunResponse(BaseModel):
#     answers: List[str]

# # --------------- App ---------------
# app = FastAPI(title="HackRX RAG API")

# @app.post("/hackrx/run", response_model=RunResponse)
# @app.post("/api/v1/hackrx/run", response_model=RunResponse)
# async def run_endpoint(payload: RunRequest, authorization: Optional[str] = Header(None)):
#     # Auth
#     if REQUIRE_BEARER:
#         if not authorization or authorization.strip() != EXPECTED_BEARER:
#             raise HTTPException(status_code=401, detail="Unauthorized")

#     # Fetch and parse PDF
#     pdf_bytes = await fetch_pdf(str(payload.documents))
#     raw_text = extract_pdf_text(pdf_bytes)
#     if not raw_text.strip():
#         raise HTTPException(status_code=400, detail="Could not extract text from PDF")

#     # Build passages
#     passages = make_passages(raw_text, target_tok=MAX_PASSAGE_TOKENS, overlap_tok=40)

#     # Build sparse index
#     whoosh_idx, _storage = build_sparse_index(passages)

#     # Dense index
#     dense = DenseIndexer(EMBED_MODEL_NAME)
#     dense.index(passages)

#     # Map ids to text
#     id2text = {str(i): passages[i] for i in range(len(passages))}

#     # Prepare reranker
#     reranker = CrossEncoder(RERANK_MODEL_NAME)

#     answers: List[str] = []
#     for q in payload.questions:
#         sparse = bm25_search(whoosh_idx, q, TOPK_SPARSE)
#         dense_res = dense.search(q, TOPK_DENSE)

#         candidates = rrf([sparse, dense_res], k=FUSION_KEEP)
#         ranked = reranker.rerank(q, id2text, candidates, topk=RERANK_KEEP)

#         final_ids = ranked[:CONTEXT_PASSAGES]
#         contexts = [id2text[i] for i in final_ids]

#         if USE_LLM_READER and os.environ.get("OPENAI_API_KEY"):
#             ans = await llm_answer(q, contexts)
#         else:
#             ans = extractive_answer(q, contexts)
#         answers.append(ans)

#     return RunResponse(answers=answers)

import os
import io
import re
import json
import httpx
from typing import List, Dict, Optional
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, AnyHttpUrl
from whoosh import index
from whoosh.fields import Schema, TEXT, ID
from whoosh.analysis import StemmingAnalyzer
from whoosh.qparser import MultifieldParser
from whoosh.filedb.filestore import RamStorage
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pypdf import PdfReader
from docx import Document
import tiktoken
from pipe_training_layerwise_base_original_qa import get_ans 

# --------------- Config ---------------
EMBED_MODEL_NAME = "intfloat/e5-base-v2"
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOPK_SPARSE = 40
TOPK_DENSE = 40
FUSION_KEEP = 60
RERANK_KEEP = 10
CONTEXT_PASSAGES = 8
MAX_PASSAGE_TOKENS = 220
MAX_TOKENS_PER_EMBED_BATCH = 280_000
REQUIRE_BEARER = True
EXPECTED_BEARER = "Bearer b7a83688229ee229beb214ec38a7d7c9de760fc2a8c6fe7c3ef7d68c89ea7eac"

USE_LLM_READER = True
READER_MODEL = os.environ.get("READER_MODEL", "gpt-4o-mini")

# --------------- Utilities ---------------
def tokenizer():
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None

TOK = tokenizer()

def count_tokens(text: str) -> int:
    if TOK:
        return len(TOK.encode(text))
    return int(len(text.split()) * 1.3)

def safe_trim(text: str, max_tokens: int) -> str:
    if count_tokens(text) <= max_tokens:
        return text
    words = text.split()
    low, high = 0, len(words)
    while low < high:
        mid = (low + high) // 2
        s = " ".join(words[:mid])
        if count_tokens(s) <= max_tokens:
            low = mid + 1
        else:
            high = mid
    return " ".join(words[:high - 1])

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

# --------------- File handling ---------------
async def fetch_document(url: str) -> tuple:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(url)
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail=f"Failed to fetch file: HTTP {r.status_code}")

    ctype = r.headers.get("content-type", "").lower()
    filename = url.split("/")[-1].lower()

    if "pdf" in ctype or filename.endswith(".pdf"):
        return "pdf", r.content
    elif "word" in ctype or filename.endswith(".docx"):
        return "docx", r.content
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

def extract_pdf_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for p in reader.pages:
        try:
            pages.append(p.extract_text() or "")
        except Exception:
            pages.append("")
    return "\n".join(pages)

def extract_docx_text(docx_bytes: bytes) -> str:
    file_stream = io.BytesIO(docx_bytes)
    doc = Document(file_stream)
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)

# --------------- Chunking ---------------
def paragraph_split(text: str) -> List[str]:
    parts = [normalize_ws(p) for p in re.split(r"\n{2,}", text) if p.strip()]
    if not parts:
        parts = [normalize_ws(text)]
    return parts

def sentenceish_split(paragraph: str) -> List[str]:
    sents = re.split(r"(?<=[.!?])\s+", paragraph)
    return [normalize_ws(s) for s in sents if s.strip()]

def make_passages(text: str, target_tok: int = 220, overlap_tok: int = 40) -> List[str]:
    paragraphs = paragraph_split(text)
    passages = []
    for para in paragraphs:
        sents = sentenceish_split(para)
        buf, t = [], 0
        for s in sents:
            ts = count_tokens(s)
            if t + ts > target_tok and buf:
                passages.append(" ".join(buf))
                buf = [buf[-1]] if buf else []
                t = count_tokens(" ".join(buf))
            buf.append(s)
            t += ts
        if buf:
            passages.append(" ".join(buf))
    passages = [safe_trim(p, MAX_PASSAGE_TOKENS) for p in passages if p.strip()]
    return passages

# --------------- Sparse index (Whoosh in-memory) ---------------
def build_sparse_index(passages: List[str]) -> tuple:
    schema = Schema(id=ID(stored=True, unique=True),
                    text=TEXT(stored=True, analyzer=StemmingAnalyzer()))
    storage = RamStorage()
    idx = storage.create_index(schema)
    writer = idx.writer()
    for i, p in enumerate(passages):
        writer.add_document(id=str(i), text=p)
    writer.commit()
    return idx, storage

def bm25_search(idx, query: str, topk: int) -> List[tuple]:
    qp = MultifieldParser(["text"], schema=idx.schema)
    q = qp.parse(query)
    with idx.searcher() as s:
        res = s.search(q, limit=topk)
        return [(hit["id"], float(hit.score)) for hit in res]

# --------------- Dense embedding search (in-memory) ---------------
class DenseIndexer:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.passages = None

    def index(self, passages: List[str]):
        self.passages = passages
        embs = self.model.encode(passages, normalize_embeddings=True, convert_to_tensor=True, batch_size=64)
        self.embeddings = embs

    def search(self, query: str, topk: int) -> List[tuple]:
        qvec = self.model.encode([query], normalize_embeddings=True, convert_to_tensor=True)
        sims = (self.embeddings @ qvec.T).squeeze(1)
        vals, idxs = torch.topk(sims, k=min(topk, sims.shape[0]))
        return [(str(i), float(v)) for v, i in zip(vals.tolist(), idxs.tolist())]

# --------------- Reciprocal Rank Fusion ---------------
def rrf(lists: List[List[tuple]], k: int = 60) -> List[str]:
    scores = {}
    for lst in lists:
        for rank, (pid, _score) in enumerate(lst):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (60 + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [pid for pid, _ in ranked[:k]]

# --------------- Cross-encoder reranker ---------------
class CrossEncoder:
    def __init__(self, model_name: str):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

    @torch.no_grad()
    def rerank(self, query: str, id2text: Dict[str, str], cand_ids: List[str], topk: int) -> List[str]:
        pairs = [(query, id2text[c]) for c in cand_ids if c in id2text]
        if not pairs:
            return []
        enc = self.tok(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            truncation=True, padding=True, max_length=512, return_tensors="pt"
        )
        logits = self.model(**enc).logits.squeeze(-1)
        order = torch.argsort(logits, descending=True).tolist()
        return [cand_ids[i] for i in order if i < len(cand_ids)][:topk]

# --------------- LLM reader ---------------
async def llm_answer(query: str, contexts: List[str]) -> str:
    # from openai import AsyncOpenAI
    # client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    blocks = [f"[{i+1}] {c}" for i, c in enumerate(contexts)]
    evidence = "\n\n".join(blocks)
    # system = "You are a precise assistant. Answer ONLY using the provided evidence and cite as [n]. If insufficient, say you cannot find it."
    # user = f"Question: {query}\n\nEvidence:\n{evidence}\n\nWrite a concise answer with citations."
    # resp = await client.chat.completions.create(
    #     model=READER_MODEL,
    #     messages=[{"role": "system", "content": system},
    #               {"role": "user", "content": user}],
    #     temperature=0.2,
    #     max_tokens=400
    # )
    
    ans = get_ans(query, evidence)
    return ans


def extractive_answer(query: str, contexts: List[str]) -> str:
    return "\n\n".join(contexts[:2]) if contexts else "No relevant context found."

# --------------- FastAPI models ---------------
class RunRequest(BaseModel):
    documents: AnyHttpUrl
    questions: List[str]

class RunResponse(BaseModel):
    answers: List[str]

# --------------- App ---------------
app = FastAPI(title="HackRX RAG API")

@app.post("/hackrx/run", response_model=RunResponse)
@app.post("/api/v1/hackrx/run", response_model=RunResponse)
async def run_endpoint(payload: RunRequest, authorization: Optional[str] = Header(None)):
    # Auth
    if REQUIRE_BEARER:
        if not authorization or authorization.strip() != EXPECTED_BEARER:
            raise HTTPException(status_code=401, detail="Unauthorized")

    # Fetch and parse file
    file_type, file_bytes = await fetch_document(str(payload.documents))
    if file_type == "pdf":
        raw_text = extract_pdf_text(file_bytes)
    elif file_type == "docx":
        raw_text = extract_docx_text(file_bytes)
    else:
        raise HTTPException(status_code=400, detail="Unsupported document format")

    if not raw_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from document")

    # Build passages
    passages = make_passages(raw_text, target_tok=MAX_PASSAGE_TOKENS, overlap_tok=40)

    # Build sparse index
    whoosh_idx, _storage = build_sparse_index(passages)

    # Dense index
    dense = DenseIndexer(EMBED_MODEL_NAME)
    dense.index(passages)

    # Map ids to text
    id2text = {str(i): passages[i] for i in range(len(passages))}

    # Prepare reranker
    reranker = CrossEncoder(RERANK_MODEL_NAME)

    answers: List[str] = []
    for q in payload.questions:
        sparse = bm25_search(whoosh_idx, q, TOPK_SPARSE)
        dense_res = dense.search(q, TOPK_DENSE)
        candidates = rrf([sparse, dense_res], k=FUSION_KEEP)
        ranked = reranker.rerank(q, id2text, candidates, topk=RERANK_KEEP)
        final_ids = ranked[:CONTEXT_PASSAGES]
        contexts = [id2text[i] for i in final_ids]
        if USE_LLM_READER and os.environ.get("OPENAI_API_KEY"):
            ans = await llm_answer(q, contexts)
        else:
            ans = extractive_answer(q, contexts)
        answers.append(ans)

    return RunResponse(answers=answers)

