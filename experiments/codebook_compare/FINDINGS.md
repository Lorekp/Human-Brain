# Codebook comparison (decisive representation experiment): semantic structure in the afferent encoding does NOT unlock calibration transfer

Experiment ID: `codebook_compare` · run date 2026-08-23 · device `HUMANBRAIN_DEVICE=cpu`
· all 27 runs at harness commit `081a492` (`--codebook` knob) · outputs in
`experiments/codebook_compare/` · analysis: `_analyze.py` → `_analysis_output.txt`,
degeneracy probe: `_probe_degeneracy.py` → `_probe_degeneracy_output.txt`

## Question

Single-axis decisive test from the preregistration in
`docs/research/a013-representation-choice.md`: the status-quo encoding gives
related tokens nearly disjoint presynaptic substrate (expected cross-token
afferent overlap 2/16 ≈ 12.5% by construction), which mechanically explains the
sealed transfer failure. If we make shared structure PHYSICAL — token
assemblies that overlap monotonically in embedding similarity — does ridge-readout
calibration transfer appear? Hypothesis **H-code**: calibration-cases fraction
(similarity − random) > 0 with 95% CI excluding zero; secondary (random −
explicit) > 0. Preregistered fallback if similarity ≯ random: record and pivot
to concept-mined chunk tokens.

## Design and execution integrity

- **Confirmatory cells (preregistered)**: codebook {explicit, random,
  similarity} × N ∈ {128, 512} × seeds {7, 8, 9} = 18 runs.
- **Extension cells (tasked capacity point, exploratory)**: same arms at
  N = 1024 = 9 more runs, for a total of **27**. Verdict gates are computed on
  the confirmatory cells only; N=1024 is reported separately.
- Every run: readout `ridge`, schedule `massed`, data_scale 42 TRAIN tasks,
  supervision `all_transition`, k=16 assemblies (harness default =
  input_fanout; identical k across arms per preregistration), 3 corpus passes.
- **Determinism**: every arm ran twice in-process (`--check-determinism`) and
  produced byte-identical records minus `wall_s` (harness-enforced; exit 0 on
  all 27).
- **Red tests** (`tests/test_ladder_codebook.py`, 12 passed in this worktree):
  codebook determinism under seed with domain-separated streams; frozen
  similarity-projection pins (similarity codebook identical across RUN seeds —
  seed must not perturb it; confirmed stable in the metrics' `codebook_sha256`);
  assembly size/collision bounds; grow==completion single shared mapping;
  framing-order preservation under re-encoding; CLI gate rejections. Note: the
  random arm's codebook hash varies across seeds BY DESIGN (per-seed
  deterministic draw); the similarity arm's hash is run-seed-invariant.
- **Fairness**: `curriculum_sha256` is identical across all three arms within
  every (seed, N) cell — every codebook saw byte-identical task cohorts.
- Explicit is the untouched status-quo afferent table; explicit rows reproduce
  the pilot ridge rows of `experiments/readout_compare/FINDINGS.md` exactly on
  all integer metrics (e.g. N512 s7 = 106·34·5, N1024 s8 = 126·126 cases),
  anchoring this matrix to the pilot.
- Shared-box caveat: this machine runs other agents' workloads concurrently;
  `wall_s` numbers are informational only (serialized execution, no timing
  verdicts drawn). Case/replay/margin metrics are deterministic and
  noise-immune; no anomalous slowdown (>3× vs pilot scaling) was observed.

## Capacity table (TRAIN; mean ± sd over seeds {7,8,9})

| codebook | N | train_cases frac (/126) | train_replay frac (/42) | margin_median_pa |
| --- | --- | --- | --- | --- |
| explicit | 128 | 0.1455 ± 0.0229 | 0.0635 ± 0.0364 | 3.17 ± 15.07 |
| explicit | 512 | 0.5635 ± 0.2409 | 0.5238 ± 0.2474 | 100.73 ± 4.98 |
| explicit | 1024 | 0.9815 ± 0.0165 | 0.9762 ± 0.0238 | 223.22 ± 4.00 |
| random | 128 | 0.0979 ± 0.0183 | 0.0159 ± 0.0137 | −5.41 ± 7.57 |
| random | 512 | 0.5873 ± 0.2205 | 0.5317 ± 0.2546 | 85.59 ± 27.69 |
| random | 1024 | 0.9444 ± 0.0599 | 0.9444 ± 0.0599 | 215.20 ± 3.89 |
| similarity | 128 | 0.1138 ± 0.0516 | 0.0317 ± 0.0364 | 7.37 ± 4.64 |
| similarity | 512 | **0.7354 ± 0.0121** | **0.6905 ± 0.0238** | 101.16 ± 5.83 |
| similarity | 1024 | 0.9656 ± 0.0465 | 0.9603 ± 0.0496 | 215.71 ± 2.40 |

Per-seed raw (train_cases/126 · replay/42 · calib/54):

- explicit 128: 15·1·7 / 20·3·10 / 20·4·7 — 512: 106·34·5 / 55·16·7 / 52·16·8 — 1024: 123·41·8 / 126·42·10 / 122·40·7
- random 128: 15·1·10 / 11·0·7 / 11·1·10 — 512: 42·10·10 / 92·29·8 / 88·28·10 — 1024: 120·40·10 / 126·42·5 / 111·37·8
- similarity 128: 21·3·7 / 14·1·10 / 8·0·10 — 512: 93·30·8 / 91·28·7 / 94·29·10 — 1024: 115·38·5 / 124·41·8 / 126·42·10

**Capacity finding**: at the N=512 bridge size the similarity codebook raises
mean train capacity by +0.15 over both controls (0.735 vs 0.563 explicit /
0.587 random) AND collapses the across-seed variance (±0.012 vs ±0.24):
explicit and random show unstable emergence at N=512 (106 vs 55/52; 42 vs
92/88) while similarity fits uniformly (93/91/94). At N=1024 all arms approach
the capacity ceiling and the differences wash out.

## Transfer table (CALIBRATION; held-out 18 tasks / 54 cases)

| codebook | N | calib_cases frac (/54) | per-seed passes (/54) |
| --- | --- | --- | --- |
| explicit | 128 | 0.1481 ± 0.0321 | 7/10/7 |
| explicit | 512 | 0.1235 ± 0.0283 | 5/7/8 |
| explicit | 1024 | 0.1543 ± 0.0283 | 8/10/7 |
| random | 128 | 0.1667 ± 0.0321 | 10/7/10 |
| random | 512 | 0.1728 ± 0.0214 | 10/8/10 |
| random | 1024 | 0.1420 ± 0.0466 | 10/5/8 |
| similarity | 128 | 0.1667 ± 0.0321 | 7/10/10 |
| similarity | 512 | 0.1543 ± 0.0283 | 8/7/10 |
| similarity | 1024 | 0.1420 ± 0.0466 | 5/8/10 |

Every arm sits in the same 0.12–0.17 band at every N — the band the pilot
established as dominated by the degenerate constant-program floor (~6/54
identity-prompt floor plus lucky single-case coincidences).

## Delta CIs on calibration-cases fraction (Newcombe hybrid-score intervals built from Wilson score intervals, z=1.96; pooled over seeds; confirmatory cells)

| comparison | scope | delta | 95% CI | excludes zero |
| --- | --- | --- | --- | --- |
| similarity − random | N=128 | +0.0000 | [−0.0817, +0.0817] | no |
| similarity − random | N=512 | −0.0185 | [−0.0996, +0.0626] | no |
| **similarity − random (H-code primary)** | **pooled {128,512}** | **−0.0093** | **[−0.0666, +0.0481]** | **no — FAIL** |
| random − explicit | N=128 | +0.0185 | [−0.0615, +0.0984] | no |
| random − explicit | N=512 | +0.0494 | [−0.0287, +0.1273] | no |
| random − explicit (secondary) | pooled {128,512} | +0.0340 | [−0.0217, +0.0895] | no |
| similarity − random (extension) | N=1024 | +0.0000 | [−0.0769, +0.0769] | no |
| random − explicit (extension) | N=1024 | −0.0123 | [−0.0905, +0.0658] | no |

Per-seed paired deltas (robustness; seeds 7,8,9): similarity−random @128
[−3,+3,0], @512 [−2,−1,0], @1024 [−5,+3,+2]; random−explicit @128 [+3,−3,+3],
@512 [+5,+1,+2], @1024 [+2,−5,+1]. No sign consistency anywhere for the
primary contrast; the secondary contrast is 3/3 positive only at N=512 but its
pooled CI still includes zero.

## VERDICTS

- **H-code primary (similarity − random > 0, CI excluding 0): FAIL.**
  Δ = −0.0093, 95% CI [−0.0666, +0.0481] — the interval includes zero and the
  point estimate is nominally negative. Semantic projection adds nothing to
  calibration transfer beyond distributional overlap.
- **Secondary (random − explicit > 0): NOT CONFIRMED.** Δ = +0.0340,
  95% CI [−0.0217, +0.0895]; nominal positive, CI includes zero. Distributed
  codes alone do not beat explicit identity under ridge readout.
- **Degeneracy probe** (`_probe_degeneracy.py`; regrows each circuit from the
  recorded arguments and completes all 18 calibration tasks teacher-free):
  the modal emitted program is the 1-token constant `variable-0` in EVERY
  arm × N × seed, including similarity at scale. Mean non-modal (genuine)
  case passes per arm-cell: 0.0–3.33 of 54 — all inside the floor band the
  pilot documented. No codebook produces genuine program-level transfer.

### Fit-vs-generalize dissociation (reportable capacity effect)

The N=512 similarity advantage is real but confined to FIT: +0.15 mean train
cases with variance collapsing from ±0.24 to ±0.01 (93/91/94 across seeds vs
explicit's unstable 106/55/52). Overlap-structured afferent geometry makes the
TRAIN corpus more linearly separable and emergence reliable, yet moves
calibration transfer by literally nothing (Δ ≈ −0.01). Conclusion: where the
shared structure acts (input-side feature overlap feeding an eligibility
substrate) is upstream of whatever binds programs to generalizable behavior;
distributed overlap helps the readout fit without helping the circuit
generalize. Transfer in this stack is not limited by codebook geometry alone.

## Honest fallback (per the preregistration's own clause)

Since similarity ≯ random, the preregistered fallback applies: semantic
projection of the EXISTING token vocabulary is a dead end for transfer.
Recommended pivot, in order:

1. **Concept-mined chunk tokens** — replace/augment the vocabulary so that the
   repeated latent operators across families are single tokens whose assemblies
   are shared by construction (attacks family-to-family binding directly rather
   than token-similarity cosmetics). The probe's constant-program signature
   suggests the binding problem sits at operator/argument identity, which chunk
   tokens address and embedding overlap cannot.
2. Alternatively, accept the sealed-result framing: under a closed-form ridge
   readout, transfer requires fixing feature QUALITY (local-rule constants or
   concept tokens), not feature DISTRIBUTION — see also the schedule_compare
   null (presentation order inert) as further evidence that input-side
   rearrangements do not move calibration.

## Files

- `explicit_N{128,512,1024}_s{7,8,9}.json`, `random_…`, `similarity_…` — 27 fixed-schema metrics records (git_head 081a492, `config_dump.codebook` + `codebook_sha256` provenance per record)
- `_run_matrix.sh`, `_run_log.txt` — serialized launcher + full run log (every arm `--check-determinism`, exit 0 ×27)
- `_analyze.py`, `_analysis_output.txt` — tables, Newcombe/Wilson CIs, verdict inputs
- `_probe_degeneracy.py`, `_probe_degeneracy_output.txt`, `_probe_degeneracy_results.jsonl` — post-hoc degeneracy probe
