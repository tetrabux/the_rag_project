from sentence_transformers import CrossEncoder
from rag.chunking import chunk_sections
from rag.parsing import parse_docs, DOCS_DIR
from rag.retrieval.fusion import fuse_min_max
from rag.retrieval.dense import search as dense_search
from rag.retrieval.lexical import build_index, search as lexical_search
from rag.retrieval.dense import embed_chunks, load_model

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_reranker():
    return CrossEncoder(MODEL_NAME)


def rerank(reranker, query, candidates, k):
    # cross-encoder scores query+doc jointly, so it's slower but more precise than the bi-encoder dense pass
    pairs = [[query, chunk.text] for chunk, _ in candidates]
    scores = reranker.predict(pairs)
    chunks_only = [chunk for chunk, _ in candidates]
    
    rated_candidates = sorted(zip(chunks_only, scores), key=lambda x: x[1], reverse=True)
    
    return rated_candidates[:k]
    

if __name__ == "__main__":
    print("Loading Reranker ...")
    reranker = load_reranker()
    dense_model = load_model()

    print("Embedding the chunks ...")
    chunks = chunk_sections(parse_docs(DOCS_DIR / "wiki"))
    embeddings = embed_chunks(dense_model, chunks)

    print("Building the index ...")
    bm25 = build_index(chunks)

    print("Searching ...")
    query = "What is the MAX_NETWORK_RETRY_BACKOFF?"
    
    
    dense_results = dense_search(dense_model, query, chunks, embeddings, 20)
    lexical_results = lexical_search(query, chunks, bm25, 20)
    fused_results = fuse_min_max(dense_results, lexical_results)

    print("Reranking ...")
    reranked = rerank(reranker, query, fused_results, 10)

    for chunk, score in reranked:
        print(f"Score: {score}")
        print(f"Text: {chunk.text}")
        print()