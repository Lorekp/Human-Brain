# swe_m01_runs — M0/M1 preregistered battery runner

Runner: `run_m0m1.py`. Executes the full preregistered M0/M1 milestone
battery on the python_qa channel the moment an architecture passes, and
evaluates both milestone gates against the LOCKED n-gram comparators in
`experiments/python_qa_baselines_v1/`.

## What one invocation does

```
python experiments/swe_m01_runs/run_m0m1.py
```

1. Scores 3 seeds {17, 18, 19} x growth tokenizers {concept, byte} = 6 cells:
   `qa_experiments.run()` with `--corpus python_qa` (frozen artifact: grow on
   all 75 TRAIN, evaluate all 25 VAL), CUDA full-touch reservoir at N=131072,
   sparse_three_factor readout (`--readout-rule auto` resolves it), 2 growth
   epochs, slack 8; concept arm uses the frozen pilot mining gates
   (freq >= 2, MI >= 1.0 bits) per `experiments/python_qa_scale/BridgeFindings.md`.
2. Every scored cell executes through `humanbrain.contract_audit.audit()`
   (mandatory, not optional): the decorator fails fast if torch/autograd
   machinery enters `sys.modules` around a scored run — proof-of-rollout of
   the frozen-evaluation contract. Weights are grown encapsulated inside
   `run()`, so the import tripwire is the enforceable seam at this boundary.
3. Per-cell records land in `records/` (harness schema verbatim,
   `corpus_sha256` re-verified against the pin on every load).
4. Gates (seed-pooled proportions per tokenizer arm):
   - **M0** — cell span accuracy beats the random/unigram floor
     (`unigram_smoothed`, 0.0 = 0/104), Newcombe method-10 95% CI excluding
     zero on the positive side AND degeneracy guard passing on every record.
   - **M1** — cell span accuracy beats the smoothed trigram
     (`trigram_smoothed`, 5.769% = 6/104), same CI criterion.
   - CI convention matches `experiments/codebook_compare/_analyze.py` and
     `experiments/replay_compare/`: Wilson score intervals at
     z = 1.959963984540054 combined by Newcombe's square-and-add (Method 10,
     Newcombe 1998); the MILESTONES.md non-overlapping-Wilson view is also
     reported as a secondary column.
5. Writes `battery_summary.json` (cells, gates, verdicts, claim grammar,
   provenance, declared deviations). Exit 0 only when BOTH gates PASS.

Pricing (measured, BridgeFindings.md §4): ~4 min concept row + ~8 min byte
row of exclusive GPU at N=131072; compute is not binding on this channel.

## Modes

| command | effect |
|---|---|
| `... --dry-run` | Validate corpus-pin integrity, locked comparators, harness seam, audit wiring (incl. negative probe: injected fake `torch` module must trip `ContractViolation`) — no circuits built, zero GPU minutes. |
| `... --dry-run --strict` | Same, but exit 1 while any check is BLOCKED. |
| `... --device cpu --cells 96` | Fixture-scale smoke (deviation from preregistration; recorded as such). |
| `... --determinism-precheck` | AUDITED double-run determinism probe on a short slice — run once BEFORE unsealing the battery (BridgeFindings §4). |
| `... --force` | Re-execute cells whose records exist (default: resume). |

## Readiness state (2026-08-25, branch codex/path3-repetition @ 5981159)

- `--dry-run`: corpus/comparators/audit/CUDA checks PASS; the single BLOCKED
  item is the `--corpus python_qa` harness seam, which lives on
  `codex/swe-m01-harness` (@1d4d351: `humanbrain.python_qa_adapter` +
  `ExperimentConfig.corpus`). Merge that branch into this line and the same
  dry-run reports READY (verified against the merged tree without touching
  tracked files).
- Full-battery path proven end-to-end at CPU fixture scale under the
  simulated merge: scored cells run through the audit, records + summary +
  gate math verified (Newcombe deltas, guard gating, verdicts).

## Guard rails baked in

- Wrong-tree detection: refuses to score if `humanbrain` imports from another
  checkout (the dev venv's editable install points at the main repo).
- Digest chain: artifact bytes vs `manifest.json`, canonical entries digest
  vs the pinned `789e3ac…dd446ec`, every record re-checked before gates.
- Partial-slice smoke runs get deviation-suffixed record filenames so they
  can never be resumed as preregistered full-cohort cells.
- Prediction counts are NOT comparable across tokenizer conventions (byte/
  concept races vs word-level n-gram predictions over the same 25 VAL tasks)
  — compare accuracies and CIs, never raw counts.
