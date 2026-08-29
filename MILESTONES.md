# MILESTONES — capability gate ladder (M0–M3)

Status: ACTIVE LADDER INDEX. Single source of truth for what counts as M0–M3,
reconciled 2026-08-24 from `docs/research/gemma4-comparison-design.md`
(§Milestone ladder, decision rules) and
`docs/research/simple-swe-opening-preregistration.md` (baselines, freeze/seed
ceremony, stack verification). Those documents remain the design rationale;
this file is the operational gate list. M4–M6 (HumanEval+ ≥10%, MBPP+ ≥20% +
unseen-family transfer, SWE-bench Verified vs Gemma-4-12B-QAT) stay defined in
gemma4-comparison-design.md and are out of scope here.

## Shared measurement contract (applies to every milestone)

- **Task suite**: self-hosted simple-SWE cohorts (`benchmarks/simple_swe_protocol_v1.json`),
  preregistered before any cell run: a **capacity cohort** from visible
  families and a **transfer cohort** from held-out families; family-disjoint,
  excluding any instance overlapping prior checkpoint growth trajectories.
- **Metric**: pass@1 under hidden-case oracle execution in the Job-Object
  sandbox, all-or-nothing per task, reported per cohort.
- **Replication**: k = 3 independent seeds per system. **Gate rule**:
  win = non-overlapping Wilson 95% CIs across seeds (decision rule fixed in
  gemma4-comparison-design.md); tie = overlapping CIs; no single-run claims.
- **Fairness**: identical task formatter and tool budget (none for v1);
  majority-or-best-of-k declared in advance; cross-tokenizer budgets compared
  in bytes, never raw tokens; compute accounting (wall-clock + parameters)
  reported for both sides; contamination statement shipped with every table.
- **Ordering discipline**: every baseline below runs through the IDENTICAL
  freeze → externally anchored seed → reveal → evaluate → ledger path as the
  cell system, and its results are locked into the evidence envelope BEFORE
  cell completions are unsealed (enforced by
  `test_baseline_controls_cannot_precede_or_change_the_cell_summary` and
  `runner.record_baseline_control`).

## Stack

| Component | What | Status (2026-08-24) |
|---|---|---|
| Evaluation stack | `src/humanbrain/simple_swe/` freeze/reveal/adapter/compiler/sandbox/runner + hash-chained ledger; protocol pins python-3.11 | Verified end-to-end 7/7 on fixtures on this host; venv-trampoline × Job-Object bug FIXED (sandbox now probe-and-redirects launcher executables to the base interpreter) |
| B-random baseline | Uniform program sampler over the grammar space (preregistration §Preregistered baselines) | Defined, not yet run |
| B-template baseline | Heuristic template-filler keyed on task signature | Defined, not yet run |
| N-gram prior art | `benchmarks/simple_swe_public_baseline.py` ran prompt-only CellLanguageModel n-gram candidates (order 8 / 64), noncanonical public dev bench (`artifacts/simple_swe_public_baseline_v1.json`) | Reusable implementation seed for M1's same-corpus smoothed n-gram baseline; the public bench itself is NOT an M-gate result |
| External anchor | HumanEval+ via EvalPlus (secondary anchor only; contamination acknowledged — "unseen-generator performance", never sole basis of a claim) | Not yet run |

## Milestones

### M0 — beat random completion
- **Claim**: the grown cell system completes simple-SWE tasks better than
  uniform random program sampling.
- **Metric**: pass@1 on the preregistered capacity + transfer cohorts, k=3.
- **Gate**: cell pass@1 > B-random pass@1 on the same cohort, any margin,
  with non-overlapping seed CIs per the shared decision rule.
- **Stack**: both systems through the identical simple-SWE oracle path;
  B-random registered as a baseline control before cell unseal.

### M1 — beat smoothed n-gram code baseline
- **Claim**: the cell system beats a smoothed n-gram language model over code,
  trained on the same corpus used for cell growth.
- **Metric**: pass@1, same cohorts, k=3.
- **Gate**: CI-separated pass@1 win over the smoothed n-gram baseline
  (same-corpus training declared and frozen before evaluation).
- **Stack**: identical simple-SWE oracle path; baseline seeded from the
  existing n-gram CellLanguageModel machinery (public-bench candidates),
  retrained same-corpus and rerun as a canonical baseline control.

### M2 — beat templated-heuristic agent
- **Claim**: the cell system beats signature-keyed template filling
  (pattern-match on the task signature + template emit).
- **Metric**: pass@1, same cohorts, k=3.
- **Gate**: CI-separated pass@1 win over B-template under the shared rules.
- **Stack**: identical simple-SWE oracle path; B-template locked in before
  cell unseal.

### M3 — beat tiny neural LM (fixed param/compute budget)
- **Claim**: the cell system beats a small neural LM at a parameter/compute
  budget fixed and declared BEFORE the comparison, on BOTH simple-SWE hidden
  tests AND HumanEval+ (>0 with CI excluding zero).
- **Metric**: pass@1 on simple-SWE cohorts + HumanEval+ pass@1, k=3.
- **Gate**: CI-separated wins on both suites; HumanEval+ score >0 with CI
  excluding zero; budget axis stated (fairness rule 6 byte accounting).
- **Stack**: simple-SWE oracle path plus EvalPlus harness for HumanEval+;
  tiny-LM completions graded by the same oracle/formatter path; result
  reported as unseen-generator performance.

## Dependencies and sequencing

The old a013/L1-readout prerequisite has been superseded by the typed semantic
mainline. Before M0 opens, the generic semantic engine must cover the declared
TRAIN types/families, the prompt-only semantic bridge must pass its frozen
TRAIN capacity split with its controls, and the paired visible-I/O/prompt-only
family-disjoint public DEV gate must pass without tuning on DEV. The final
prompt-semantic checkpoint, typed compiler identity, source bundle, and
python-3.11 sandbox then become the objects frozen for M0.

Opening order remains: freeze ceremony → externally anchored seed commitment
→ task generation → baselines first → cell completions unsealed last. The
30/30 visible-I/O TRAIN bridge is prerequisite mechanism evidence only and does
not satisfy or partially score M0.
