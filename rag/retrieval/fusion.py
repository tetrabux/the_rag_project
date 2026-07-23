import numpy as np


def min_max_normalize(results):
    if not results:
        return []

    scores = [r[1] for r in results]
    min_score = min(scores)
    max_score = max(scores)

    if max_score == min_score:
        # Avoid division by zero; if all scores identical, return 0.5 for all
        return [(r[0], 0.5) for r in results]

    return [
        (chunk, (score - min_score) / (max_score - min_score))
        for chunk, score in results
    ]


def fuse_min_max(dense_results, lexical_results):
    # kept as the "naive" baseline — see README for why this is unstable across candidate depths
    # Normalize both lists
    dense_norm = min_max_normalize(dense_results)
    lexical_norm = min_max_normalize(lexical_results)

    # Build a lookup dict: chunk -> its normalized scores
    scores_map = {}
    for chunk, score in dense_norm:
        scores_map[chunk] = {"dense": score, "lexical": 0.0}
    for chunk, score in lexical_norm:
        if chunk not in scores_map:
            scores_map[chunk] = {"dense": 0.0, "lexical": score}
        else:
            scores_map[chunk]["lexical"] = score

    # Combine: add normalized scores
    fused = [(chunk, data["dense"] + data["lexical"]) for chunk, data in scores_map.items()]

    # Sort by combined score, descending
    fused.sort(key=lambda x: x[1], reverse=True)

    return fused


def fuse_rank_based(dense_results, lexical_results, k=60, dense_weight=1.0, lexical_weight=1.0):
    # k=60 is the standard RRF constant — dampens how much rank 1 dominates the fused score
    scores_map = {}

    for rank, (chunk, _) in enumerate(dense_results, start=1):
        scores_map[chunk] = scores_map.get(chunk, 0.0) + dense_weight / (k + rank)

    for rank, (chunk, _) in enumerate(lexical_results, start=1):
        scores_map[chunk] = scores_map.get(chunk, 0.0) + lexical_weight / (k + rank)

    fused = list(scores_map.items())
    fused.sort(key=lambda x: x[1], reverse=True)

    return fused


if __name__ == "__main__":
    from rag.parsing import parse_docs, DOCS_DIR
    from rag.chunking import chunk_sections
    from rag.retrieval.dense import load_model, embed_chunks, search as dense_search
    from rag.retrieval.lexical import build_index, search as lexical_search

    manual_chunks = chunk_sections(parse_docs(DOCS_DIR / "manual"))
    wiki_chunks = chunk_sections(parse_docs(DOCS_DIR / "wiki"))
    all_chunks = manual_chunks + wiki_chunks

    model = load_model()
    embeddings = embed_chunks(model, all_chunks)
    bm25 = build_index(all_chunks)

    queries = {
        "identifier": "MAX_NETWORK_RETRY_BACKOFF",
        "conceptual": "How to install the package?",
    }

    for label, query in queries.items():
        dense_results = dense_search(model, query, all_chunks, embeddings, 10)
        lexical_results = lexical_search(query, all_chunks, bm25, 10)

        print(f"=== {label}: {query} — min-max fusion ===")
        for chunk, score in fuse_min_max(dense_results, lexical_results)[:5]:
            print(f"Score: {score:.4f}")
            print(f"Text: {chunk.text[:150]}")
            print()

        print(f"=== {label}: {query} — rank-based fusion ===")
        for chunk, score in fuse_rank_based(dense_results, lexical_results)[:5]:
            print(f"Score: {score:.4f}")
            print(f"Text: {chunk.text[:150]}")
            print()
