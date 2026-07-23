import json
from pathlib import Path

BASELINE_PATH = Path(__file__).parent / "baseline.json"
RESULTS_PATH = Path(__file__).parent / "results.json"
NOISE_BAND = 0.020  # from EV1 (rag/eval/noise_band.py) — avg faithfulness stdev across repeats


def save_baseline(rows, path=BASELINE_PATH):
    with open(path,'w') as f:
        json.dump(rows,f)


def load_baseline(path=BASELINE_PATH):
    json_file = open(path,'r')
    return json.load(json_file)


def index_rows(rows):
    return {(row['query'],row['mode']): row for row in rows}


def check_hard_gate(baseline_rows, current_rows):
    # recall@5 and mrr are deterministic (same query + corpus + code -> same result), so any drop is real, not noise
    baseline_dict = index_rows(baseline_rows)
    current_dict = index_rows(current_rows)
    
    regressions = []

    for (query, mode) in baseline_dict:
        if (query,mode) not in current_dict:
            continue
        
        baseline_row = baseline_dict[(query,mode)]
        current_row = current_dict[(query,mode)]

        for metric in ["recall@5", "mrr"]:
            baseline_val = baseline_row[metric]
            current_val = current_row[metric]
            if current_val < baseline_val:
                regressions.append((query,mode,metric,baseline_val,current_val))

    
    return regressions
        
        

def check_soft_gate(baseline_rows, current_rows, noise_band=NOISE_BAND):
    # faithfulness is judge-scored, so it's noisy — only flag drift that clears the measured noise band
    deltas = {}
    for tag in set(r['tag'] for r in baseline_rows+current_rows):
        baseline_faith = [row['faithfulness'] for row in baseline_rows if row['tag'] == tag and row['faithfulness'] is not None]
        current_faith = [row['faithfulness'] for row in current_rows if row['tag'] == tag and row['faithfulness'] is not None]

        if not baseline_faith or not current_faith:
            continue
        
        baseline_avg = sum(baseline_faith) / len(baseline_faith)
        current_avg = sum(current_faith) / len(current_faith)
        delta = current_avg - baseline_avg

        if delta > noise_band:
            deltas[tag] = "improved"
        elif delta < -noise_band:
            deltas[tag] = "regressed"
        else:
            deltas[tag] = "noise"

    return deltas


def run_regression(baseline_path=BASELINE_PATH, results_path=RESULTS_PATH):
    baseline_rows = load_baseline(baseline_path)
    current_rows = load_baseline(results_path)

    regressions = check_hard_gate(baseline_rows, current_rows)
    soft_verdicts = check_soft_gate(baseline_rows, current_rows)

    print("=== Hard gate (recall@5 / mrr) ===")
    if regressions:
        print(f"FAIL - {len(regressions)} regression(s):")
        for query, mode, metric, baseline_val, current_val in regressions:
            print(f"  [{mode}] {metric} dropped {baseline_val:.3f} -> {current_val:.3f}: {query}")
    else:
        print("PASS - no regressions")

    print(f"\n=== Soft gate (faithfulness, per tag, noise band +/-{NOISE_BAND:.3f}) ===")
    for tag, verdict in sorted(soft_verdicts.items()):
        print(f"  {tag:<26} {verdict}")

    return regressions, soft_verdicts


if __name__ == "__main__":
    run_regression()
