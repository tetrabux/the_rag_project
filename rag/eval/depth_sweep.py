from rag.retrieval.dense import load_model, embed_chunks
from rag.retrieval.lexical import build_index
from rag.retrieval.rerank import load_reranker
from rag.eval.harness import build_corpus, evaluate

DEPTHS = [10, 25, 50, 100, 200]  # widen candidate pool to see where reranker depth stops helping (see README)


def sweep():
    chunks = build_corpus()
    model = load_model()
    embeddings = embed_chunks(model, chunks)
    bm25 = build_index(chunks)
    reranker = load_reranker()

    per_depth_rows = {}

    print(f"{'n':<8}{'recall@5':<12}{'mrr':<8}")
    for n in DEPTHS:
        rows, _ = evaluate(
            modes=("hybrid_rerank",), n_candidates=n,
            chunks=chunks, model=model, embeddings=embeddings, bm25=bm25, reranker=reranker,
        )
        per_depth_rows[n] = {row["query"]: row for row in rows}

        avg_recall = sum(r["recall@5"] for r in rows) / len(rows)
        avg_mrr = sum(r["mrr"] for r in rows) / len(rows)
        print(f"{n:<8}{avg_recall:<12.3f}{avg_mrr:<8.3f}")

    return per_depth_rows


def find_regressions(per_depth_rows, from_n, to_n):
    before = per_depth_rows[from_n]
    after = per_depth_rows[to_n]
    regressions = []
    for query, row_before in before.items():
        row_after = after[query]
        if row_after["mrr"] < row_before["mrr"]:
            regressions.append((query, row_before["mrr"], row_after["mrr"]))
    return regressions


if __name__ == "__main__":
    per_depth_rows = sweep()

    print()
    print("=== queries where n=50 -> n=200 hurt (mrr dropped) ===")
    for query, mrr_before, mrr_after in find_regressions(per_depth_rows, 50, 200):
        print(f"{query!r}: mrr {mrr_before:.3f} -> {mrr_after:.3f}")
