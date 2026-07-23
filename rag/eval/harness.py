from rag.parsing import parse_docs, DOCS_DIR

from rag.chunking import chunk_sections

from rag.retrieval.dense import load_model, embed_chunks, search as dense_search
from rag.retrieval.lexical import build_index, search as lexical_search
from rag.retrieval.fusion import fuse_rank_based
from rag.retrieval.rerank import load_reranker, rerank
from rag.retrieval.filtering import filter_indices

from rag.eval.queries import EVAL_QUERIES
from rag.eval.generation import generate_answer, make_client, load_cache

from rag.eval.judge import judge_answer, load_judge_cache

import json
from pathlib import Path

N_CANDIDATES = 20
K_FINAL = 5


def build_corpus():
    print("Building Corpus ...")
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


RELEVANCE_TIERS = {"evades": 0, "does_not_address": 1, "partially_addresses": 2, "fully_addresses": 3}


def get_oracle_context(query_item, chunks):
    # the "known-correct" context, pulled by file rather than retrieved — this is what the oracle test compares against
    context = []

    for chunk in chunks:
        if chunk.file_path.name in query_item['expected_files']:
            context.append((chunk, 1.0))
    return context


def run_oracle(query_item, chunks, cache, client, judge_cache):
    oracle_context = get_oracle_context(query_item, chunks)
    answer = generate_answer(query_item['query'], oracle_context, cache, client)['answer']
    if answer is None:
        # judge_answer crashes on a None answer (its cache key does string concat), so bail out early here
        return None, {'faithfulness': None, 'relevance': None}
    judgement = judge_answer(query_item['query'], answer, oracle_context, judge_cache, client)
    return answer, judgement



def attribute_failure(retrieved_judgement, oracle_judgement):
    if (retrieved_judgement['relevance'] == None or retrieved_judgement['faithfulness']== None or
            oracle_judgement['relevance']==None or oracle_judgement['faithfulness']==None):
        return None
    if retrieved_judgement['faithfulness'] >= 0.8 and retrieved_judgement['relevance'] == 'fully_addresses':
        return 'ok'
    # gap-based, not a fixed bar: did the correct context actually rescue the answer?
    if (oracle_judgement['faithfulness'] - retrieved_judgement['faithfulness'] >= 0.3) or (RELEVANCE_TIERS[oracle_judgement['relevance']] > RELEVANCE_TIERS[retrieved_judgement['relevance']]):
        return 'retrieval_bug'
    return 'generation_bug'


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
             chunks=None, model=None, embeddings=None, bm25=None, reranker=None, cache=None, client=None,
             judge_cache=None):
    print("Evaluating ...")
    if chunks is None:
        chunks = build_corpus()
        model = load_model()
        embeddings = embed_chunks(model, chunks)
        bm25 = build_index(chunks)
        reranker = load_reranker()

    if cache is None:
        cache = load_cache()

    if judge_cache is None:
        judge_cache = load_judge_cache()
    
    if client is None:
        client = make_client()


    rows = []
    unanswerable_notes = []

    for query_item in EVAL_QUERIES:
        expected_files = query_item["expected_files"]
        for mode in modes:
            results = run_query(query_item, mode, model, embeddings, bm25, chunks, reranker, n_candidates)

            if not expected_files:
                # nothing in the corpus should answer this — log what the system returns anyway, don't score it
                top_chunk, top_score = results[0]
                unanswerable_notes.append({
                    "query": query_item["query"], "mode": mode,
                    "top_score": top_score, "top_file": top_chunk.file_path.name,
                })
                continue

            answer = generate_answer(query_item["query"], results, cache, client)['answer']
            
            if answer is not None:
                judgement = judge_answer(query_item['query'], answer, results, judge_cache, client)
                oracle_answer, oracle_judgement = run_oracle(query_item, chunks, cache, client, judge_cache)
                attribution = attribute_failure(judgement, oracle_judgement)
            else:
                judgement = {
                    "faithfulness": None,
                    "relevance": None,
                    "relevance_reasoning": None,
                    "answer_text": None
                }
                oracle_answer = ""
                oracle_judgement = {"faithfulness": None, "relevance": None}
                attribution = None

            rows.append({
                "query": query_item["query"],
                "tag": query_item["tag"],
                "mode": mode,
                "recall@5": recall_at_k(results, expected_files, K_FINAL),
                "mrr": reciprocal_rank(results, expected_files),
                "answer":answer,
                "faithfulness": judgement["faithfulness"],
                "relevance": judgement["relevance"],
                "relevance_reasoning": judgement["relevance_reasoning"],
                "context":[chunk.text for chunk, _ in results],
                "attribution": attribution,
                'oracle_faithfulness': oracle_judgement['faithfulness'],
                'oracle_relevance': oracle_judgement['relevance'],
            })


    return rows, unanswerable_notes


def save_results(rows,path):
    print("Saving Results ...")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    
    



def summarize(rows):
    from collections import defaultdict
    groups = defaultdict(list)
    for row in rows:
        groups[(row["tag"], row["mode"])].append(row)

    print(f"{'tag':<26}{'mode':<15}{'n':<5}{'recall@5':<12}{'mrr':<8}")
    for (tag, mode), items in sorted(groups.items()):
        n = len(items)
        avg_recall = sum(r["recall@5"] for r in items) / n
        avg_mrr = sum(r["mrr"] for r in items) / n
        print(f"{tag:<26}{mode:<15}{n:<5}{avg_recall:<12.3f}{avg_mrr:<8.3f}")


if __name__ == "__main__":
    rows, unanswerable_notes = evaluate()
    summarize(rows)
    save_results(rows, Path(__file__).parent / "results.json")

    print()
    print("=== unanswerable queries: what the system returns anyway ===")
    for note in unanswerable_notes:
        print(note)
