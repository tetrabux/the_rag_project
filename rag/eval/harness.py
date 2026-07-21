from rag.parsing import parse_docs, DOCS_DIR
from rag.chunking import chunk_sections
from rag.retrieval.dense import load_model, embed_chunks, search as dense_search
from rag.retrieval.lexical import build_index, search as lexical_search
from rag.retrieval.fusion import fuse_rank_based
from rag.retrieval.rerank import load_reranker, rerank
from rag.retrieval.filtering import filter_indices
from rag.eval.queries import EVAL_QUERIES

N_CANDIDATES = 20
K_FINAL = 5


def build_corpus():
    manual_chunks = chunk_sections(parse_docs(DOCS_DIR / "manual"))
    wiki_chunks = chunk_sections(parse_docs(DOCS_DIR / "wiki"))
    return manual_chunks + wiki_chunks


def is_hit(chunk, expected_files):
    return chunk.file_path.name in expected_files


def reciprocal_rank(results, expected_files):
    for rank, (chunk, _) in enumerate(results, start=1):
        if is_hit(chunk, expected_files):
            return 1.0 / rank
    return 0.0


def recall_at_k(results, expected_files, k):
    top_k = results[:k]
    return 1.0 if any(is_hit(chunk, expected_files) for chunk, _ in top_k) else 0.0


def run_query(query_item, mode, model, embeddings, bm25, chunks, reranker, n_candidates=N_CANDIDATES):
    query = query_item["query"]
    allowed = None
    if query_item.get("filter"):
        allowed = filter_indices(chunks, **query_item["filter"])

    dense_results = dense_search(model, query, chunks, embeddings, n_candidates, allowed_indices=allowed)

    if mode == "dense":
        return dense_results[:K_FINAL]

    lexical_results = lexical_search(query, chunks, bm25, n_candidates, allowed_indices=allowed)
    fused = fuse_rank_based(dense_results, lexical_results)

    if mode == "hybrid":
        return fused[:K_FINAL]

    if mode == "hybrid_rerank":
        return rerank(reranker, query, fused[:n_candidates], K_FINAL)

    raise ValueError(f"unknown mode: {mode}")


def evaluate(modes=("dense", "hybrid", "hybrid_rerank"), n_candidates=N_CANDIDATES,
             chunks=None, model=None, embeddings=None, bm25=None, reranker=None):
    if chunks is None:
        chunks = build_corpus()
        model = load_model()
        embeddings = embed_chunks(model, chunks)
        bm25 = build_index(chunks)
        reranker = load_reranker()

    rows = []
    unanswerable_notes = []

    for query_item in EVAL_QUERIES:
        expected_files = query_item["expected_files"]
        for mode in modes:
            results = run_query(query_item, mode, model, embeddings, bm25, chunks, reranker, n_candidates)

            if not expected_files:
                top_chunk, top_score = results[0]
                unanswerable_notes.append({
                    "query": query_item["query"], "mode": mode,
                    "top_score": top_score, "top_file": top_chunk.file_path.name,
                })
                continue

            rows.append({
                "query": query_item["query"],
                "tag": query_item["tag"],
                "mode": mode,
                "recall@5": recall_at_k(results, expected_files, K_FINAL),
                "mrr": reciprocal_rank(results, expected_files),
            })

    return rows, unanswerable_notes


def summarize(rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(row["tag"], row["mode"])].append(row)

    print(f"{'tag':<15}{'mode':<15}{'n':<5}{'recall@5':<12}{'mrr':<8}")
    for (tag, mode), items in sorted(groups.items()):
        n = len(items)
        avg_recall = sum(r["recall@5"] for r in items) / n
        avg_mrr = sum(r["mrr"] for r in items) / n
        print(f"{tag:<15}{mode:<15}{n:<5}{avg_recall:<12.3f}{avg_mrr:<8.3f}")


if __name__ == "__main__":
    rows, unanswerable_notes = evaluate()
    summarize(rows)

    print()
    print("=== unanswerable queries: what the system returns anyway ===")
    for note in unanswerable_notes:
        print(note)
