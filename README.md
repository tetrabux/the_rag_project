# Hybrid RAG Retrieval Engine

A retrieval system combining dense (embedding) and lexical (BM25) search, fused with
a scale-invariant rank-based method, with optional cross-encoder reranking, metadata
filtering, and a FastAPI service on top.

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
