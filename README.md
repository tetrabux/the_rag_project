# Hybrid RAG Retrieval Engine + Eval Suite

A retrieval system combining dense (embedding) and lexical (BM25) search, fused with
a scale-invariant rank-based method, with optional cross-encoder reranking, metadata
filtering, and a FastAPI service on top — plus a from-scratch eval suite that scores
the generation half (faithfulness, answer relevance) with a calibrated LLM judge, and
gates regressions on *measured* judge noise instead of a guessed threshold.

Two parts to this README: **Part 1** is the retrieval engine (chunking → dense +
lexical search → fusion → reranking → API). **Part 2** is the eval suite on top of
it (golden set → generation → judge → calibration → regression gate → attribution →
the reranker before/after number).

---

# Part 1 — Retrieval Engine

## Ablation table — dense-only vs hybrid vs hybrid+rerank

From `rag/eval/harness.py`, run against all 22 eval queries (`rag/eval/queries.py`),
sliced by tag. `n=20` candidates, `k=5` final results.

| tag            | mode          | n | recall@5 | mrr   |
|----------------|---------------|---|----------|-------|
| conceptual     | dense         | 5 | 1.000    | 1.000 |
| conceptual     | hybrid        | 5 | 1.000    | 1.000 |
| conceptual     | hybrid_rerank | 5 | 1.000    | 1.000 |
| filtered       | dense         | 2 | 1.000    | 1.000 |
| filtered       | hybrid        | 2 | 1.000    | 1.000 |
| filtered       | hybrid_rerank | 2 | 1.000    | 1.000 |
| identifier     | dense         | 3 | 1.000    | 0.778 |
| identifier     | hybrid        | 3 | 1.000    | 1.000 |
| identifier     | hybrid_rerank | 3 | 1.000    | 1.000 |
| misspelled     | dense         | 2 | 1.000    | 1.000 |
| misspelled     | hybrid        | 2 | 1.000    | 1.000 |
| misspelled     | hybrid_rerank | 2 | 1.000    | 1.000 |
| multi_section  | dense         | 3 | 1.000    | 1.000 |
| multi_section  | hybrid        | 3 | 1.000    | 1.000 |
| multi_section  | hybrid_rerank | 3 | 1.000    | 1.000 |
| near_duplicate | dense         | 4 | 1.000    | 1.000 |
| near_duplicate | hybrid        | 4 | 1.000    | 0.875 |
| near_duplicate | hybrid_rerank | 4 | 1.000    | 1.000 |

Two real signals, both consistent with the corpus's design premise (identifier-dense
wiki docs vs conceptual prose manual docs):

- **Identifier queries: dense-only mrr=0.778, hybrid/hybrid_rerank=1.000.** Dense
  embeddings confuse near-identical identifiers (e.g. `MAX_AUTH_RETRY_BACKOFF` vs
  `MAX_NETWORK_RETRY_BACKOFF` score similarly since they sit in near-identical
  sentence templates), so the correct chunk sometimes ranks below a same-template
  wrong one. Lexical retrieval's exact token matching fixes this once fused in.
- **Near-duplicate queries: hybrid alone dips to mrr=0.875, dense-only and
  hybrid_rerank both stay at 1.000.** With several equally-correct chunks sharing an
  identifier, rank-based fusion's ordering among those near-duplicates isn't always
  optimal — but the cross-encoder reranker fixes the ordering back up.

### Unanswerable queries

Four eval queries have no correct answer in the corpus. The system never refuses —
it always returns `k` results regardless. But the *scores* differ meaningfully by
mode:

- Dense cosine similarity stays high (0.60–0.75) even for genuinely unrelated
  queries — not useful for telling "no answer" apart from "weak answer."
- The cross-encoder reranker's score is a much better confidence signal: genuine
  matches scored ~4–7 elsewhere in this eval set, but all four unanswerable queries
  scored **negative** (-7.5 to -11.0) after reranking.

Finding: if this system needed to detect "no answer in corpus" and refuse or say so,
the reranker score — not the retriever score — is the signal to threshold on.

## Fusion strategy: why naive score normalization is unstable

We first tried the naive approach: min-max normalize both retrievers' scores to 0-1
per query, then average. This turns out to be unstable — not because it's sensitive
to multiplying a retriever's raw scores by some constant (min-max normalization
already cancels that out), but because it's sensitive to **whatever min and max
happen to be present in the current candidate set**, which shifts depending on
candidate depth `n`.

Concrete example: the identifier query `MAX_NETWORK_RETRY_BACKOFF` has four equally
correct answers in the corpus (`err_4000.md`, `err_4161.md`, `err_4336.md`,
`err_4364.md`). With min-max fusion:

- At `n=10`, the #1 result is `err_4336.md` (score 1.9293).
- At `n=50`, the #1 result flips to `err_4161.md` (score 1.9343).

Nothing about the query or the documents changed — only the size of the candidate
pool did. A wider pool shifted the min/max reference points used for normalization,
which rescaled every score and flipped which of two equally-correct documents came
out on top.

Rank-based fusion (reciprocal rank fusion) does not have this problem. For the same
query at both `n=10` and `n=50`, the #1 result stayed `err_4161.md` with an
identical score (0.0325) — because it depends only on each chunk's rank position
within its own retriever's list, never on absolute score values or the min/max of
an arbitrary candidate window.

## Reranker candidate depth: finding the sweet spot

With `k` fixed at 5, we swept candidate depth `n` over `[10, 25, 50, 100, 200]` for
the hybrid+rerank mode:

| n   | recall@5 | mrr   |
|-----|----------|-------|
| 10  | 1.000    | 1.000 |
| 25  | 1.000    | 1.000 |
| 50  | 1.000    | 1.000 |
| 100 | 1.000    | 1.000 |
| 200 | 1.000    | 1.000 |

The curve does not fall. That means either the eval set is too easy or candidate
depth isn't doing what we think — we checked precisely (not just rounded aggregates)
and confirmed every single answerable query stays at rank 1 (mrr=1.0) at *both*
n=10 and n=200, with zero exceptions across all 20 answerable queries.

Investigation: this corpus's identifier and near-duplicate queries are built almost
entirely from the literal identifier string itself (e.g. querying
`MAX_NETWORK_RETRY_BACKOFF`, where that exact token also appears verbatim in the
correct chunk's text). That's an extremely strong, unambiguous signal for both BM25
(exact token match) and the cross-encoder reranker — at this corpus size (~330
chunks), widening the candidate pool to 200 (well over half the entire corpus) never
actually introduces a distractor confusing enough to displace the correct chunk from
rank 1. The reranker is comfortably strong enough, and the corpus small/clean enough,
that this particular failure mode doesn't surface here.

This is an eval-set/corpus-scale limitation, not evidence that reranker depth is
irrelevant in general — with a larger or more adversarial corpus (near-duplicate
documents that don't share the literal query tokens, or many more plausible
distractors), we'd expect the usual rise-then-fall curve to appear. That's a
legitimate follow-up once this baseline exists — deliberately not done yet, since
hardening the corpus before you have something to compare against would leave
nothing to measure improvement against.

## Known limitation: multi-tenant security

`rag/api/main.py`'s `/search` endpoint takes an optional `source` field to scope
results (wiki vs manual), using the same filtering mechanism proven correct by the
filtering test suite. That's fine as a *convenience* filter — but it is not a
security boundary, and in a real multi-tenant deployment it would need to be one.

The structural problem: **the filter is optional and client-supplied, not mandatory
and identity-derived.** There is no authentication on the request, and nothing ties
a caller's identity to a required, server-enforced scope. Any client can simply omit
`source` and see the entire corpus — including data that, in a real multi-tenant
system, would belong to other tenants. The filtering logic itself is correct as a
mechanism (it does guarantee correctness + recall preservation once a scope is
given), but at the API layer that scope is never actually forced — it's a knob the
caller can choose to turn, not a rule the server enforces. A real deployment needs
the allowed-subset filter derived from an authenticated tenant ID on every request,
never from a client-supplied field.

---

# Part 2 — Eval Suite

Module 1's eval was saturated — nearly every cell scored 1.000, so it couldn't tell
a real improvement from a real regression. This module (1) de-saturates the golden
set, (2) adds the generation half (an actual LLM answer, not just retrieved chunks),
(3) scores that answer on two axes with an LLM judge, (4) calibrates that judge
against hand labels instead of trusting it blindly, (5) measures the judge's own
noise so a regression gate isn't set on a guess, and (6) can tell you, for any failed
query, whether retrieval or generation is at fault.

## Golden set (G1)

`rag/eval/queries.py` — 29 queries across 9 tags, deliberately including the cases
that break a naive system: a semantically confusable distractor shared by 19 docs, a
paraphrase whose correct chunk shares no literal words with the query (forces real
reasoning, not string match), a near-duplicate-precision case where 10 docs share an
identical description and only one is actually correct, and unanswerable queries with
no correct chunk at all. Dense-only baseline has real headroom on these (see the
ablation table above) — the set earns its keep.

## Generation + judge (G2, G3)

`rag/eval/generation.py` takes the top-k retrieved chunks + query and produces an
answer via an LLM API, instructed to say "I don't know" rather than guess when the
context doesn't cover it. `rag/eval/judge.py` then scores that answer on two axes,
both via a structured JSON rubric (not a raw 1-10 score, which tends to cluster):

- **Faithfulness** — the answer is decomposed into individual claims, and each claim
  is checked against the retrieved context. Score = fraction of claims supported.
- **Answer relevance** — one of four verdicts (`fully_addresses`, `partially_addresses`,
  `does_not_address`, `evades`) for whether the answer actually engages the question.

Both the generation and judge calls are cached keyed on their exact inputs
(`rag/eval/.cache/`), so re-running the eval doesn't re-bill or re-hit rate limits for
anything already scored — only genuinely new (or previously-failed) calls go out.

## Judge calibration — is this judge trustworthy? (G4)

An unvalidated judge is just a second opinion, not ground truth. `rag/eval/calibrate.py`
hand-labels 15 generated answers and compares against the judge's scores on the same
15 (`rag/eval/human_labels.json`):

- **Faithfulness agreement: 87% (13/15)**
- **Relevance agreement: 80% (12/15)**
- **All 5 disagreements run in the same direction** — the judge rated the answer more
  favorably than the human did, never the reverse. It called a hallucinated claim
  "faithful" in 2/15 cases, and called a partially-addressing answer
  "fully_addresses" in 3/15 cases.

Trust verdict: the judge is trustworthy enough to scale to the full set for
**relative** signal — its leniency bias is systematic, not random, so a real drop
between two runs still shows up as a drop. Its **absolute** numbers should be read as
an upper bound, not a precise score: a judge-reported 1.0 faithfulness does not
guarantee zero hallucination. This is exactly why the regression gate below is built
around *movement between runs*, not fixed absolute thresholds.

## Judge noise band (EV1)

Faithfulness is judged by an LLM, so re-scoring the *same* answer twice won't
necessarily give the same number. `rag/eval/noise_band.py` re-judges a sample of
outputs 5 times each with nothing changed, and measures the run-to-run spread:

**Noise band: 0.020 avg faithfulness stdev.**

This number is load-bearing: it's the only thing separating "the reranker actually
helped" from "the judge is just being the judge again." Any regression gate set
tighter than 0.020 would trip on noise alone.

## Regression gate: hard vs soft (G5)

`rag/eval/regression.py` freezes a baseline run (`save_baseline`) and diffs a new run
against it, per query and per tag — not just an aggregate:

- **Hard gate** (`check_hard_gate`) — `recall@5` and `mrr` are **deterministic**: same
  query, same corpus, same code → same retrieval result, always. There is no noise to
  account for, so *any* drop is a real regression and blocks. It reports every
  `(query, mode, metric)` that dropped.
- **Soft gate** (`check_soft_gate`) — faithfulness is judge-scored and therefore
  noisy, so movement is classified per tag against the 0.020 EV1 band: `improved` if
  the delta clears +0.020, `regressed` if it clears -0.020, otherwise `noise`
  (tracked, not blocking).

This is the acceptance test the spec asks for: change one thing (a prompt word, `k`,
the reranker toggle), rerun, and the gate tells you — beyond noise — whether it
helped, hurt, or is indistinguishable, and on which slice.

## Failure attribution: retrieval bug or generation bug? (G6)

For any query, `run_oracle` in `rag/eval/harness.py` re-answers it using the *known-
correct* context directly (pulled by file, not retrieved) instead of whatever the
retriever found, then re-judges that answer. `attribute_failure` compares the two
judgements:

- If the retrieved-context answer was already good (faithfulness ≥ 0.8 and fully
  relevant) → `ok`.
- Else, if the oracle context meaningfully improved things (faithfulness jumped ≥0.3,
  or relevance moved up a tier) → `retrieval_bug` — the right context existed but
  wasn't found.
- Else → `generation_bug` — even the correct context didn't fix the answer, so the
  fault is in generation/prompting, not retrieval.

This directly separates two failure modes that a single faithfulness number cannot
tell apart on its own — which is also why faithfulness alone doesn't prove
correctness: a low score could mean "retrieval handed the model garbage" or "the
model hallucinated despite good context," and only the oracle-context test tells you
which.

## The reranker before/after number (EV2)

`rag/eval/ev2_reranker.py` compares `hybrid` (reranker off) against `hybrid_rerank`
(reranker on) from the same run, sliced by tag, faithfulness delta shown against the
0.020 noise band:

| tag                       | recall@5      | mrr           | faithfulness            | verdict     |
|---------------------------|---------------|---------------|--------------------------|-------------|
| identifier                | 1.000 → 1.000 | 1.000 → 1.000 | 0.933 → 1.000 (+0.067)   | **improved** (3.3× band) |
| multi_section              | 1.000 → 1.000 | 1.000 → 1.000 | 1.000 → 0.944 (-0.056)   | **regressed** (2.8× band) |
| paraphrase                 | 0.500 → 1.000 | 0.250 → 0.625 | 1.000 → 1.000 (+0.000)  | noise |
| distractor                 | 1.000 → 1.000 | 0.500 → 1.000 | 1.000 → 1.000 (+0.000)  | noise |
| near_duplicate              | 1.000 → 1.000 | 0.875 → 1.000 | 1.000 → 1.000 (+0.000)  | noise |
| conceptual / filtered / misspelled | unchanged | unchanged | unchanged | noise |
| near_duplicate_precision   | 0.000 → 0.000 | 0.000 → 0.000 | 1.000 → 1.000 (+0.000)  | noise |

**The one honest claim this earns:** the reranker lifted `identifier`-tag answer
faithfulness by +0.067 — 3.3× the measured judge noise band — a real, defensible
gain. It is not a free lunch: `multi_section` faithfulness dropped -0.056, 2.8× the
band, a genuine regression in the other direction. Everywhere else, faithfulness
moved within noise and cannot be claimed as a real change. Separately, and with no
noise caveat needed since these metrics are deterministic, the reranker produced real
retrieval gains on `paraphrase` (mrr 0.25→0.625), `distractor` (mrr 0.5→1.0), and
`near_duplicate` (mrr 0.875→1.0).

**Known gap:** `near_duplicate_precision` — the single query where 10 docs share an
identical description and only one is correct — stays at `recall@5 = 0.000` with the
reranker both on and off. The reranker fixes *ordering* among retrieved candidates;
it can't fix a case where the correct chunk never made it into the candidate set in
the first place. That's a retrieval-depth/embedding problem, not something reranking
solves.

## How to run it

```bash
uv run python -m rag.eval.harness        # full eval run, writes rag/eval/results.json
uv run python -m rag.eval.calibrate       # hand-label a fresh sample (interactive)
uv run python -m rag.eval.noise_band      # re-measure the EV1 noise band
uv run python -c "from rag.eval.regression import save_baseline; import json; \
  save_baseline(json.load(open('rag/eval/results.json')))"   # freeze a baseline
uv run python -m rag.eval.regression      # gate the current results.json against baseline.json
uv run python -m rag.eval.ev2_reranker    # reranker on/off report, sliced by tag
```

Generation + judging call an LLM API (OpenRouter free-tier models) — everything else
(retrieval, chunking, metrics, gating) runs on CPU.
