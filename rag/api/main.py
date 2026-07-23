from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag.parsing import parse_docs, DOCS_DIR
from rag.chunking import chunk_sections
from rag.retrieval.dense import load_model, embed_chunks, search as dense_search
from rag.retrieval.lexical import build_index, search as lexical_search
from rag.retrieval.fusion import fuse_rank_based
from rag.retrieval.rerank import load_reranker, rerank
from rag.retrieval.filtering import filter_indices

app = FastAPI()

STATE = {}  # model/index/reranker loaded once at startup, not per-request


@app.on_event("startup")
def startup():
    manual_chunks = chunk_sections(parse_docs(DOCS_DIR / "manual"))
    wiki_chunks = chunk_sections(parse_docs(DOCS_DIR / "wiki"))
    chunks = manual_chunks + wiki_chunks

    model = load_model()
    embeddings = embed_chunks(model, chunks)
    bm25 = build_index(chunks)
    reranker = load_reranker()

    STATE["chunks"] = chunks
    STATE["model"] = model
    STATE["embeddings"] = embeddings
    STATE["bm25"] = bm25
    STATE["reranker"] = reranker


class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid_rerank"
    k: int = 5
    n: int = 20
    source: Optional[str] = None


class SearchResult(BaseModel):
    text: str
    score: float
    source: str
    breadcrumb: list[str]
    file_path: str


@app.post("/search", response_model=list[SearchResult])
def search(req: SearchRequest):
    if req.mode not in ("dense", "hybrid", "hybrid_rerank"):
        raise HTTPException(400, f"unknown mode: {req.mode}")

    chunks = STATE["chunks"]
    allowed = filter_indices(chunks, source=req.source) if req.source else None

    dense_results = dense_search(
        STATE["model"], req.query, chunks, STATE["embeddings"], req.n, allowed_indices=allowed
    )

    if req.mode == "dense":
        results = dense_results[: req.k]
    else:
        lexical_results = lexical_search(
            req.query, chunks, STATE["bm25"], req.n, allowed_indices=allowed
        )
        fused = fuse_rank_based(dense_results, lexical_results)

        if req.mode == "hybrid":
            results = fused[: req.k]
        else:
            results = rerank(STATE["reranker"], req.query, fused[: req.n], req.k)

    return [
        SearchResult(
            text=chunk.text,
            score=float(score),
            source=chunk.source,
            breadcrumb=chunk.breadcrumb,
            file_path=str(chunk.file_path),
        )
        for chunk, score in results
    ]
