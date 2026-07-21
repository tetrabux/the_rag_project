from rank_bm25 import BM25Okapi
from rag.chunking import chunk_sections
from rag.parsing import parse_docs, DOCS_DIR
import re

def tokenize(text):
    return re.findall(r"[a-z0-9_]+", text.lower())


def build_index(chunks):
    tokenized_chunks = [tokenize(chunk.text) for chunk in chunks]
    return BM25Okapi(tokenized_chunks)


def search(query, chunks, bm25, n, allowed_indices=None):
    tokenized_query = tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    indexed_scores = list(enumerate(scores))

    if allowed_indices is not None:
        indexed_scores = [
            (i, score) for i, score in indexed_scores
            if i in allowed_indices
        ]

    indexed_scores = sorted(indexed_scores, key=lambda x: x[1], reverse=True)
    top_n = indexed_scores[:n]
    return [(chunks[i], score) for i, score in top_n]


if __name__ == "__main__":
    manual_chunks = chunk_sections(parse_docs(DOCS_DIR / "manual"))
    wiki_chunks = chunk_sections(parse_docs(DOCS_DIR / "wiki"))

    all_chunks = manual_chunks + wiki_chunks

    bm25 = build_index(all_chunks)

    query = "MAX_NETWORK_RETRY_BACKOFF"
    results = search(query, all_chunks, bm25, 5)

    for chunk, score in results:
        print(f"Score: {score}")
        print(f"Text: {chunk.text}")
        print()
    