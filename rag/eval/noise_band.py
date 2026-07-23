import statistics
import time
from collections import namedtuple, Counter
from pathlib import Path

from rag.eval.generation import make_client
from rag.eval.judge import judge_answer
from rag.eval.calibrate import load_results, sample_rows

N_ROWS = 5
N_REPEATS = 5
SEED = 7  # different from calibrate.py's SEED — an independent sample of rows

FakeChunk = namedtuple("FakeChunk", ["file_path", "text"])


def to_fake_results(context):
    # judge_answer expects (chunk, score) pairs, but here we're re-judging an already-generated answer
    # from results.json, not doing fresh retrieval — so a lightweight stand-in chunk is enough
    results = []
    for idx, con in enumerate(context):
        results.append((FakeChunk(file_path=f"chunk_{idx}", text=con), 0.0))
    return results


def repeated_judgments(query, answer, results, client, n_repeats=N_REPEATS):
    faithfulness_scores = []
    relevance_verdicts = []
    for i in range(n_repeats):
        result = judge_answer(query, answer, results, {}, client)
        faithfulness_scores.append(result['faithfulness'])
        relevance_verdicts.append(result['relevance'])
        time.sleep(2)  # firing 5 repeats back to back trips the free-tier rate limit otherwise
    return faithfulness_scores,relevance_verdicts

        


def relevance_consistency(relevance_verdicts):
    # relevance is categorical, so stdev doesn't apply — consistency is how often the mode verdict recurs instead
    top_verdict, top_count = Counter(relevance_verdicts).most_common(1)[0]
    return top_count / len(relevance_verdicts)


def measure_noise_band(n_rows=N_ROWS, n_repeats=N_REPEATS, seed=SEED):
    rows = load_results()
    sample = sample_rows(rows, n=n_rows, seed=seed)
    client = make_client()

    per_row_stdevs = []

    for row in sample:
        results = to_fake_results(row["context"])
        faithfulness_scores, relevance_verdicts = repeated_judgments(
            row["query"], row["answer"], results, client, n_repeats
        )

        valid_scores = [s for s in faithfulness_scores if s is not None]
        valid_verdicts = [v for v in relevance_verdicts if v is not None]
        if len(valid_scores) < 2:
            print(f"{row['query'][:60]!r:<64} skipped (only {len(valid_scores)} successful repeats)")
            continue

        stdev = statistics.stdev(valid_scores)
        consistency = relevance_consistency(valid_verdicts)

        print(f"{row['query'][:60]!r:<64} stdev={stdev:.3f} relevance_consistency={consistency:.2f}")
        per_row_stdevs.append(stdev)

    noise_band = sum(per_row_stdevs) / len(per_row_stdevs)
    print(f"\nOverall faithfulness noise band (avg stdev): {noise_band:.3f}")

    return noise_band


if __name__ == "__main__":
    measure_noise_band()
