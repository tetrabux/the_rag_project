import json
from pathlib import Path

from rag.eval.regression import check_soft_gate, NOISE_BAND

RESULTS_PATH = Path(__file__).parent / "results.json"


def load_results(path=RESULTS_PATH):
    with open(path, "r") as f:
        return json.load(f)


def split_by_mode(rows):
    # "hybrid" = reranker off, "hybrid_rerank" = reranker on — both already exist in the same results.json run
    return [r for r in rows if r["mode"] == "hybrid"], [r for r in rows if r["mode"] == "hybrid_rerank"]


def avg_by_tag(rows, metric):
    by_tag = {}
    for row in rows:
        if row[metric] is None:
            continue
        by_tag.setdefault(row["tag"], []).append(row[metric])

    averages = {}
    for tag, values in by_tag.items():
        averages[tag] = sum(values) / len(values)

    return averages


def report(off_rows, on_rows):
    soft_verdicts = check_soft_gate(off_rows, on_rows)

    off_recall = avg_by_tag(off_rows, "recall@5")
    on_recall = avg_by_tag(on_rows, "recall@5")
    off_mrr = avg_by_tag(off_rows, "mrr")
    on_mrr = avg_by_tag(on_rows, "mrr")
    off_faith = avg_by_tag(off_rows, "faithfulness")
    on_faith = avg_by_tag(on_rows, "faithfulness")

    tags = sorted(set(off_recall) & set(on_recall))

    for tag in tags:
        recall_before = off_recall[tag]
        recall_after = on_recall[tag]
        mrr_before = off_mrr[tag]
        mrr_after = on_mrr[tag]

        line = f"{tag:<26} recall@5 {recall_before:.3f}->{recall_after:.3f}  mrr {mrr_before:.3f}->{mrr_after:.3f}"

        if tag in off_faith and tag in on_faith:
            faith_before = off_faith[tag]
            faith_after = on_faith[tag]
            delta = faith_after - faith_before
            multiple = abs(delta) / NOISE_BAND  # how many noise bands the move is — the "is this real" number from the spec
            verdict = soft_verdicts.get(tag, "n/a")
            line += f"  faithfulness {faith_before:.3f}->{faith_after:.3f} ({delta:+.3f}, {multiple:.1f}x noise band, {verdict})"
        else:
            line += "  faithfulness n/a"

        print(line)


def run_ev2(results_path=RESULTS_PATH):
    rows = load_results(results_path)
    off_rows, on_rows = split_by_mode(rows)
    report(off_rows, on_rows)


if __name__ == "__main__":
    run_ev2()
