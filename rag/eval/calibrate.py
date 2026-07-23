"""Judge calibration (G4).

Trust verdict: against 15 hand-labels, the judge scores 87% agreement on
faithfulness (13/15) and 80% on relevance (12/15). All 5 disagreements run
in the same direction — the judge rated the answer more favorably than the
human did (calling a hallucinated claim "faithful" in 2/15 cases, and
calling a partially-addressing answer "fully_addresses" in 3/15 cases) —
it never scored an answer worse than a human would have. That makes the
judge trustworthy enough to scale to the full set for *relative* signal:
its leniency bias is systematic rather than random, so a drop between two
runs is still a real drop. Its *absolute* numbers should be read as an
upper bound rather than a precise score, though — a judge-reported 1.0
faithfulness does not guarantee zero hallucination. Any regression gate
built on this judge (G5) should weight run-over-run movement over absolute
thresholds, since the consistent leniency bias mostly cancels out when
comparing two runs to each other.
"""

import json
import random
from pathlib import Path

RESULTS_PATH = Path(__file__).parent / "results.json"
LABELS_PATH = Path(__file__).parent / "human_labels.json"
SAMPLE_SIZE = 15
SEED = 42  # fixed so the same 15 rows get sampled every time you run this

RELEVANCE_OPTIONS = ["fully_addresses", "partially_addresses", "does_not_address", "evades"]


def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)


def row_key(row):
    return f"{row['query']}||{row['mode']}"


def sample_rows(rows, n=SAMPLE_SIZE, seed=SEED):
   answerable = [r for r in rows if r['answer'] is not None and r['faithfulness'] is not None]
   return random.Random(seed).sample(answerable,min(n,len(answerable)))


def load_labels():
    if not LABELS_PATH.exists():
        return {}
    with open(LABELS_PATH, "r") as f:
        return json.load(f)


def save_labels(labels):
    LABELS_PATH.parent.mkdir(exist_ok=True, parents=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def prompt_faithfulness():
    while True:
        answer = input("Does the answer contain any unsupported/hallucinated claim? (y/n): ").strip().lower()
        if answer == "y":
            return False   # unsupported claim present -> NOT fully faithful
        if answer == "n":
            return True    # no unsupported claims -> fully faithful
        print("Please enter y or n.")


def prompt_relevance():
   while True:
        for i, option in enumerate(RELEVANCE_OPTIONS, start=1):
            print(f"{i}. {option}")
        choice = input("Pick a number (1-4): ").strip()
        if choice in ("1", "2", "3", "4"):
            return RELEVANCE_OPTIONS[int(choice) - 1]
        print("Please enter a number 1-4.")


def label_row(row):
    print("=" * 80)
    print("QUERY:", row["query"])
    print("-" * 80)
    print("CONTEXT:")
    for chunk_text in row["context"]:
        print(chunk_text)
        print()
    print("-" * 80)
    print("ANSWER:", row["answer"])
    print("=" * 80)

    human_faithful = prompt_faithfulness()
    human_relevance = prompt_relevance()

    return {
        "human_faithful": human_faithful,
        "human_relevance": human_relevance,
        "judge_faithfulness": row["faithfulness"],
        "judge_relevance": row["relevance"],
    }


def run_labeling():
    rows = load_results()
    sample = sample_rows(rows)
    labels = load_labels()

    for row in sample:
        key = row_key(row)
        if key in labels:
            continue

        labels[key] = label_row(row)
        save_labels(labels)

    print(f"Labeled {len(labels)} / {len(sample)} sampled rows.")


def compute_agreement():
    labels = load_labels()

    faithfulness_matches = 0
    relevance_matches = 0
    faithfulness_disagreements = []
    relevance_disagreements = []

    for key, label in labels.items():
        judge_faithful_bucketed = label["judge_faithfulness"] == 1.0
        if judge_faithful_bucketed == label["human_faithful"]:
            faithfulness_matches += 1
        else:
            faithfulness_disagreements.append({
                "key": key,
                "human_faithful": label["human_faithful"],
                "judge_faithfulness": label["judge_faithfulness"],
            })

        if label["judge_relevance"] == label["human_relevance"]:
            relevance_matches += 1
        else:
            relevance_disagreements.append({
                "key": key,
                "human_relevance": label["human_relevance"],
                "judge_relevance": label["judge_relevance"],
            })

    n = len(labels)
    faithfulness_accuracy = faithfulness_matches / n
    relevance_accuracy = relevance_matches / n

    print(f"Faithfulness accuracy: {faithfulness_accuracy:.2f} ({faithfulness_matches}/{n})")
    print(f"Relevance accuracy:    {relevance_accuracy:.2f} ({relevance_matches}/{n})")

    print("\n--- Faithfulness disagreements ---")
    for d in faithfulness_disagreements:
        print(d)

    print("\n--- Relevance disagreements ---")
    for d in relevance_disagreements:
        print(d)

    return faithfulness_accuracy, relevance_accuracy


if __name__ == "__main__":
    run_labeling()
