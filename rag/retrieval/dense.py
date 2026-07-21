import numpy as np
from sentence_transformers import SentenceTransformer
from rag.chunking import chunk_sections
from rag.parsing import parse_docs, DOCS_DIR
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_model():
    print("Downloading the model ...")
    return SentenceTransformer(MODEL_NAME)


def embed_chunks(model, chunks):
    print("Embedding the chunks ...")
    embeddings = model.encode([chunk.text for chunk in chunks])
    return embeddings


def search(model, query, chunks, embeddings, n, allowed_indices=None):
    query_embedding = model.encode(query).reshape(1,-1)
    scores = cosine_similarity(query_embedding, embeddings)[0]

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
    print("Parsing the documents ...")
    manual_chunks = chunk_sections(parse_docs(DOCS_DIR / "manual"))
    wiki_chunks = chunk_sections(parse_docs(DOCS_DIR / "wiki"))
    
    all_chunks = manual_chunks + wiki_chunks

    model = load_model()
    embeddings = embed_chunks(model, all_chunks)
    
    query = "MAX_NETWORK_RETRY_BACKOFF"
    
    results = search(model, query, all_chunks, embeddings, 5)
    
    for chunk, score in results:
        print(f"Score: {score}")
        print(f"Text: {chunk.text}")
        # print(f"Breadcrumb: {chunk.breadcrumb}")
        # print(f"Source: {chunk.source}")
        # print(f"File path: {chunk.file_path}")
        print()
    

