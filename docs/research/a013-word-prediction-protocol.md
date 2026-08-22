# Cell-LM v1: first preregistered word-prediction protocol

Status: DESIGN — no corpus download, no runs. One-shot discipline throughout;
run hold until frozen with exact hashes.
Provenance: researched by subagent (repo language history, ladder schema,
prior retired gates), written by orchestrator from the agent's transcript
after an iteration-limit truncation. Requires independent verification pass
before any merge to master.

## Position in the program

Rung R5: word prediction that beats preregistered baselines at matched budget.
Depends on: representation choice (a013-representation-choice.md, in flight)
and local three-factor readout (dormant machinery). Inherits prompt-last
framing and the a012 sealed lessons.

## Prior language-gate history (why v1 must differ)

Code-language protocols v1-v4 exist in benchmarks/ history; all were retired
before producing capability claims (architecture.md:115-118 records that
throughput results were never accepted as language capability without
responsive-cell, firing, state-rank, and held-out prediction measurements —
v1 finally delivers exactly those measurements).

## Corpus

Primary choice: Project Gutenberg excerpt (~5-10 MB raw text), checksummed
and downloaded ONCE at freeze time; recorded by SHA-256 in the protocol so
every rerun-of-record uses identical bytes. Deterministic seed-derived
validation split (last 10% by position, family-disjoint books where possible:
train on K complete works, validate on 2 unseen works — book-level split is
the transfer axis). Fallback if network-free execution is mandated: seeded
synthetic text from a fixed PCG with Zipf-like token distribution — declared
weaker (no real linguistic structure) and only usable for mechanics gates,
never for the LM claim.

## Tokenization

Byte-level BPE with vocabulary fixed at freeze time (candidate sizes {4k,
8k}) chosen by ONE sweep on train-only data before the cell run — the sweep
is part of freeze, not tuning. No vocab changes after any cell run starts.

## Task framing

Next-token prediction over the token stream with a012's framing discipline:
context block, then the prediction target last (prompt-last). Sliding window
contexts of W tokens (W parameterized per arm, bounded by trace horizons:
eligibility ~60 s biological => effective context is tokens-per-second x 60).

## Architecture assumptions (per sibling docs)

- Distributed codebook mandatory (representation doc picks random-sparse vs
  similarity-projected; both admissible arms).
- Local sparse three-factor readout ONLY (ridge retired: O(N^2)/O(N^3)).
- Error-gated modulation per biology survey #1 (dopamine-RPE analogue):
  update only on mispredictions above margin.
- N parameterized {128 bridge, 1e3, 1e4} for v1 (Python circuit to ~4k,
  CUDA full-touch beyond); larger N belongs to later rungs.

## Evaluation (preregistered baselines BEFORE any cell run)

Both baselines trained on the identical train split, evaluated on identical
held-out books:

- B1: modified Kneser-Ney 5-gram (reference implementation, fixed
  discounting), reporting perplexity + top-1 accuracy.
- B2: tiny neural LM — 2-layer transformer or LSTM, <=2M parameters, <=30
  minutes single-GPU training budget, fixed hyperparameters declared at
  freeze; same metrics.

Cell system passes toward the LM rung only if it beats BOTH baselines on
held-out perplexity AND top-1 accuracy, with capacity gates (train-fit) and
generalization gates (unseen-books) reported separately, all-or-nothing per
family, mirroring the a013 ladder structure. Beating neither = fail; beating
one = partial (recorded, not celebrated).

## Accounting skeleton (mirrors typed_composition_protocol_v1.json style)

- protocol_id: humanbrain-cell-lm-v1
- base_commit / corpus_sha256 / tokenizer_sha256 / baseline_config hashes:
  pinned at freeze
- expected_accounting: observations_per_pass, strengthen_calls,
  race_count bounds — computed once the arm matrix freezes ({PARAM})
- eligibility_gates: completion sealing before scoring; decision-collision =
  reject; margins finite-positive; one-shot claim marker per arm
- execution: plan-by-default; --execute requires --confirm-protocol-sha256 +
  --confirm-corpus-sha256 + metadata --run-id; evidence under
  artifacts/cell_lm_v1/ with the same durability contract; run hold until
  separately authorized
- performance_measurements recorded, never gating

## Red tests (test-first)

1. Corpus loader rejects byte drift (hash pin enforced).
2. Book-level split disjointness (no train/validation author overlap).
3. Tokenizer determinism + round-trip; vocab size pinned.
4. Framing order test: prediction target is LAST token position (a012 lesson
   encoded as a test).
5. Baseline harness reproducibility: KN counts deterministic; neural LM seed
   pinned, budget-bounded (training wall-clock asserted <= ceiling).
6. Metric implementation tests vs hand-computed small examples
   (perplexity, accuracy).
7. Gate arithmetic: all-or-nothing family logic unit-tested with synthetic
   scorecards.
8. Anti-contamination: validation books absent from train manifest.
9. Evidence-envelope integrity: file-set equality + digest recheck on load
   (reuse runner's publication primitives).
10. Run-hold: --execute without confirmations refuses BEFORE cohort spend.

## Open items

- Representation doc must land before freeze (codebook arm choice).
- Neural-LM framework selection (pure-numpy vs torch install) — decide at
  freeze; torch not currently a dependency (pyproject deps = numpy only),
  so default is pure-numpy LSTM to keep the dependency surface unchanged.
- Exact accounting constants after arm matrix freeze.
