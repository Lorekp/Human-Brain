# Cell-LM v1: first preregistered word-prediction protocol

Status: DESIGN — no corpus download, no runs. One-shot discipline throughout;
run hold until frozen with exact hashes.
Provenance: researched by subagent (repo language history, ladder schema,
prior retired gates), written by orchestrator from the agent's transcript
after an iteration-limit truncation. Requires independent verification pass
before any merge to master.

## Position in the program

Rung R5: word prediction that beats preregistered baselines at matched budget
(the R-numbering is defined by this doc; no in-repo ladder uses R-labels yet —
see open items).
Depends on: representation choice (a013-representation-choice.md, in flight)
and local three-factor readout (dormant machinery). Inherits prompt-last
framing and the a012 sealed lessons.

## Prior language-gate history (why v1 must differ)

Code-language protocols v1-v4 exist in benchmarks/ (verified: all four JSONs
present, added in ba25083). Earlier draft text said all were retired before
producing capability claims — FALSE for v3 (correction below).
docs/architecture.md:115-118 records the discipline that throughput results
were never accepted as language capability without responsive-cell, firing,
state-rank, and held-out prediction measurements — v1 finally delivers
exactly those measurements, at word level.

CROSS-REVIEW CORRECTION (2026-08-23): "all retired without capability
claims" is FALSE for v3. It ran its preregistered one-shot BYTE-level gate
and produced a held-out capability result: test split 86.42% vs 86.40%
same-order Witten-Bell (+1 prediction, passing every file;
docs/benchmark-status.md:127-168; README capability row), plus a
domain-shift cohort at 37.97% vs 37.85% WB. v4 then made a speed-only
change (5.4x inference) without re-running the consumed test. What genuinely
never existed: a WORD-level or free-running capability claim — the
free-running development gate FAILS today (17.51%, tied with unigram;
benchmark-status.md:170-177). Two consequences carried into v1: (a) prior
gates measured teacher-forced bytes, so word-level perplexity against real
baselines is genuinely new evidence; (b) v3 passed by ONE prediction, which
is why v1 preregisters minimum margins with bootstrap CIs.

## Corpus

Primary choice: Project Gutenberg excerpt (~5-10 MB raw text), checksummed
and downloaded ONCE at freeze time; recorded by SHA-256 in the protocol so
every rerun-of-record uses identical bytes; exact book IDs are listed in the
frozen protocol itself. Feasibility (verified 2026-08-23):
www.gutenberg.org serves HTTP 503 to scripted clients, so the freeze script
pins the official-mirror form
`https://gutenberg.pglaf.org/cache/epub/{ID}/pg{ID}.txt` (tested: HTTP 200,
174 KB for ID 11) with gutenberg.org/cache/epub as best-effort primary; a
fetch failure at freeze BLOCKS freeze rather than silently switching
corpora. GATING SPLIT is whole-book holdout: train on K complete works,
validate on 2 unseen works, author-disjoint ENFORCED via red tests 2/8 —
book identity is the transfer axis. A positional last-10%-of-stream cut may
additionally be reported as a NON-gating diagnostic; it never substitutes
for the book-level split (the earlier "last 10% by position" wording
contradicted red tests 2/8 and is demoted accordingly). Fallback if
network-free execution is mandated: seeded
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

All systems train on the identical train split AND consume the identical
frozen token stream (same BPE vocabulary as the cell system): perplexities
are comparable only across shared tokenization, and byte-level BPE removes
UNK-policy divergence. All evaluated on identical held-out books:

- B0 (floor guard): unigram-frequency predictor over the train stream. Any
  system at or below B0 is running the degenerate modal-token strategy —
  automatic fail regardless of other metrics.
- B1: modified Kneser-Ney 5-gram — implementation and version PINNED at
  freeze (default: pure-Python/numpy KN interpolation, keeping the dependency
  surface unchanged; KenLM cross-check optional and non-gating), reporting
  perplexity + top-1 accuracy.
- B2: tiny neural LM — 2-layer transformer or LSTM as a preregistered
  trainable-parameter SWEEP {0.25M, 1M, 2M}; each arm <=30 minutes
  single-GPU (or declared CPU-equivalent wall-clock), fixed hyperparameters
  per arm declared at freeze; same metrics. The sweep supplies the baseline
  scaling curve for the parameter-matched comparison below.

Pass bar: beats BOTH B1 and the largest B2 arm on held-out perplexity AND
top-1 accuracy BY PREREGISTERED MINIMUM MARGINS with bootstrap CIs (1000
resamples over held-out segments; the CI of each difference must exclude
zero) — v3 passed its gate by ONE prediction, so margin-free passes are
unacceptable. Capacity gates (train-fit) and generalization gates
(unseen-books) reported separately, all-or-nothing per family, mirroring the
a013 ladder structure. Beating neither = fail; beating some but not all =
partial (recorded, not celebrated).

Parameter matching (the honest axis): RAW SYNAPSE COUNTS ARE NOT THE MATCHING
AXIS — an N=1e4 arm carries orders of magnitude more weights than B2's 2M
cap, and winning while larger proves nothing about efficiency. At freeze
every cell arm declares P_plastic = the number of parameters the learning
rule can modify from data during the run (three-factor readout weights +
eligible plastic recurrent synapses) and P_frozen = everything else
(codebooks, fixed projections, structural synapses); an independent recount
audits both (red test 13). The parameter-matched comparator is the B2 sweep
arm whose trainable-parameter count is nearest P_plastic, and its outcome is
REPORTED beside the pass decision. If P_plastic exceeds the 2M sweep ceiling,
the capability claim may stand but NO parameter-efficiency claim is
permitted.

Generalization measurements:
- Context length: evaluate held-out at >=2 context lengths inside the ~60 s
  eligibility horizon (W and 2W tokens). Hard gate: top-1 accuracy at 2W must
  not fall below accuracy-at-W minus a preregistered tolerance — the
  trace-horizon advantage is the architecture's premise, so it must survive a
  longer window rather than degrade.
- Frequency-stratified accuracy and prediction type diversity
  (distinct-token ratio) reported per system, exposing repeated-token
  degenerate strategies beyond the B0 floor.
- Cell-side training compute budget declared at freeze (wall-clock and
  strengthen_calls bounds), symmetric with B2's declared budget.

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
11. Margin/bootstrap arithmetic: synthetic scorecards that beat baselines
    inside the minimum margin score FAIL.
12. Unigram-floor guard: harness flags any system at or below B0.
13. P_plastic/P_frozen audit: independent traversal of the arm's
    modifiable-parameter registry equals the declared counts.
14. Shared-token-stream assertion: all systems scored on byte-identical
    token id sequences from the one frozen tokenizer.

## Open items

- Representation doc must land before freeze (codebook arm choice).
- Neural-LM framework selection (pure-numpy vs torch install) — VERIFIED
  2026-08-23: pyproject dependencies are ["numpy>=2.0"] exactly (pytest /
  mini-swe-agent live in optional extras), so the default pure-numpy LSTM
  keeps the dependency surface unchanged; torch would need an explicit
  optional extra declared before freeze.
- B1 pin under the same constraint: pinned pure-Python/numpy Kneser-Ney is
  the default; KenLM only as an optional cross-check.
- Rung label: "R5" numbering appears nowhere else in-repo (docs/, README,
  benchmarks checked); either define the ladder before freeze or drop the
  numeric label.
- Exact accounting constants after arm matrix freeze.

## Verification log

Cross-review 2026-08-23 (reviewer #2 of 2; worktree at commit 7ffb856,
target file committed 5b0fe45):

- VERIFIED: code_language_protocol_v1.json through v4.json exist under
  benchmarks/ (added in ba25083, 2026-08-21).
- CORRECTED: prior text claimed all four protocols were retired before any
  capability claim. False for v3 — it produced preregistered held-out
  byte-prediction results (test 86.42% vs 86.40% Witten-Bell, +1 prediction;
  domain-shift cohort 37.97% vs 37.85%; docs/benchmark-status.md:127-168;
  README capability row). True: no word-level or free-running capability
  claim exists anywhere; the free-running gate fails (17.51%, tied with
  unigram; benchmark-status.md:170-177); v4 was speed-only. History section
  rewritten; margin requirement motivated directly by the v3 +1 pass.
- VERIFIED: docs/architecture.md:115-118 carries the
  throughput-is-not-capability discipline exactly as cited.
- VERIFIED: pyproject core dependencies = ["numpy>=2.0"] exactly (pytest and
  mini-swe-agent are optional extras); pure-numpy baseline defaults are
  consistent with the dependency surface.
- VERIFIED: no "~15 s" horizon figure appears anywhere in this document;
  "eligibility ~60 s biological" matches the corrected table in
  a013-cognitive-properties.md (voltage R·C ≈78.6 ms -> ≈59 s horizon;
  eligibility 80 ms -> ≈60 s; :144-148).
- VERIFIED empirically: www.gutenberg.org returned HTTP 503 to scripted
  clients; official mirror gutenberg.pglaf.org/cache/epub/{ID}/pg{ID}.txt
  returned HTTP 200 (174 KB, ID 11) on 2026-08-23. Corpus section pins book
  IDs + mirror form; fetch failure blocks freeze.
- CORRECTED: split definition was self-contradictory ("last 10% by position"
  vs "book-level split is the transfer axis" vs red tests 2/8 requiring
  whole-book author disjointness). Gating split is now the author-disjoint
  whole-book holdout; positional cut demoted to non-gating diagnostic.
- SCIENCE EDITS: added B0 unigram floor guard; pinned B1 implementation;
  converted B2 into a {0.25M, 1M, 2M} preregistered sweep; added
  minimum-margin + bootstrap-CI gating; context-length generalization gate
  within the ~60 s horizon; frequency-stratified/diversity reporting against
  repeated-token degenerate strategies; cell-side compute budget; shared
  frozen token stream requirement; and the P_plastic/P_frozen honest
  parameter-matching axis (match what the learning rule can move, not raw
  synapse count). Red tests 11-14 added. "Beat BOTH at fixed budgets"
  retained as the capability bar; nearest-P_plastic B2 arm recorded as the
  matched-parameter disclosure alongside it.
- FLAGGED: "Rung R5" numbering appears nowhere else in-repo (checked docs/,
  README, benchmarks); open item added to define the ladder or drop the
  numeric label before freeze.
