# FINDINGS addendum — dense resample: archive sparsity did NOT limit the emptiness conclusion

Date: 2026-08-25. Branch `codex/path3-repetition` @ `5981159`. Follow-up to
`experiments/windowed_probes/FINDINGS.md` §4 ("Archive saturation caveat").
Scope: ONE arm (`lam097_concept`, mid-sweep), re-logged at **stride 1** (every
teacher-forced token gets an observation-sink snapshot; 16x denser than the
stride-16 archive), same freeze corpus / seed 7 / `\nA:` decision rule /
unchanged week-1 harness (`experiments/ustate_probes/log_states_union.py`,
sweep narrowed by `log_dense_arm.py`). Scoring: UNCHANGED
`humanbrain.windowed_probe`, week-1 protocol (logistic 250 iters decay@125,
dim-cap 1024 on `union`; exact full-dim kernel ridge everywhere; VAL-selected
lambda; shuffled-label seed 11; majority-prior + random-ranking anchors).

## Verdict

**NEGATIVE CONFIRMED DENSELY. With ~14x more recoverable history per item,
zero configurations beat majority-prior — the state-empty conclusion stands
unqualified.**

## Anchors (why the comparison is clean)

- Curriculum digest of the dense replay equals the committed concept digest
  `f0bc99f6…` (same corpus, tokenization, boundaries); all 13 arrays
  sha256-verified against the fresh manifest before scoring.
- **k=1 reproduces week-1 bit-for-bit**: union 1.72% and count_union 1.25%
  at k=1 equal the sparse-archive rows exactly (+0.00 pp). k=1 windows ARE
  the archived boundary states, so every deeper-window delta below is
  attributable ONLY to added history sampling density.
- Scorer bookkeeping anchored: majority-prior 26/639 ✓, random-ranking
  seed-11 1/639 ✓ (frozen values).

## Saturation removed

| | sparse (stride 16) | dense (stride 1) |
|---|---|---|
| stride rows (train+val items) | 2,041 | 32,606 |
| eligible history snapshots / item (mean, max) | ~8.8, 25 | **125.3, 328** |
| window fill k=16 | 35.2% | **94.1%** |
| window fill k=64 | 8.4% | **72.0%** |

(fill <100% at k=64 because short items still cannot supply 63 snapshots;
the archive ceiling itself moved from ~9 to 328 snapshots.)

## Result: accuracy does NOT escape the noise band

lam097_concept, best decoder per cell (639 VAL span predictions; bars:
majority-prior 4.069% = 26/639, unigram 4.538%):

| family | k | sparse acc | dense acc | delta |
|---|---|---|---|---|
| union | 1 | 1.72% | 1.72% | +0.00 |
| union | 4 | 1.88% | **2.50%** | +0.63 |
| union | 16 | 1.72% | 2.19% | +0.47 |
| union | 64 | 1.41% | 2.03% | +0.63 |
| count_union | 1 | 1.25% | 1.25% | +0.00 |
| count_union | 4 | 2.19% | 0.94% | −1.25 |
| count_union | 16 | 1.56% | 0.94% | −0.63 |
| count_union | 64 | 1.56% | 1.72% | +0.16 |

- Best dense configuration overall: `union / k=4 / logistic` at **2.50%
  (16/639), Wilson 95% CI [1.55%, 4.03%]** — the CI upper bound sits BELOW
  both the unigram bar (4.54%) and the majority-prior point estimate
  (4.07%). Shuffled-label controls at that node: 1.41% (logistic), 0.31%
  (ridge) — real fits again edge their controls, i.e. decoders recover only
  the global answer-frequency profile available without any state.
- **0 of 8 dense configurations beat majority-prior; 0 beat unigram**
  (week-1 sparse: 0 of 8 for this arm likewise).
- Max movement is +0.63 pp (k=4 and k=64, union) — far inside binomial
  noise at n=639 (±1.55 pp Wilson half-width at this level) and an order of
  magnitude below the >4 pp materiality bar set for qualifying the negative.
  count_union moves slightly DOWN at k=4/16. No depth-scaling emerges:
  the dense k-curve stays non-monotone around 1–2.5%.

## Reading

The week-1 caveat is resolved AGAINST the qualification: the emptiness was
not an under-sampling artifact. When every prompt token's union state is
available (mean 125 usable snapshots/item vs ~9), jointly, at full window
dimensionality, with VAL-tuned regularization, still no readout extracts
answer identity above a no-input frequency prior. The stride was hiding
nothing material; the information simply is not in the state stream.
Direction B's premise (conjunctive context-token features written online)
remains the correct target, and the earlier recommendation to log finer
strides in future diagnostics is hereby discharged — finer strides were
probed here and change nothing.

## Disclosures

1. npz binaries remain local-only (week-1 convention):
   `%LOCALAPPDATA%/Temp/hb_dense_probe/concept/logged_states.npz`
   (90.5 MB); the committed `dense_manifest_lam097_concept_s1.json` pins all
   array hashes. Regenerate via `run_dense_probe.sh` (deterministic, seeded).
2. Only the lam097_concept arm was dense-resampled (task scope); week-1's
   other arms share the encoder family and corpus, so their sparse curves
   are expected to behave identically — noted as inference, not measurement.
3. Wall: ~90 s logging + 1228 s scoring (~22 min CPU), py3.14.0/numpy 2.2.6
   (same interpreter stack as week 1).
4. A first driver attempt wrote the results JSON to a stray MSYS-mangled
   path (`C:\c\Users\...`); the file was recovered verbatim into this
   directory and the script patched (`cygpath -w` on all native-python
   args). Content verified identical via manifest-anchored rerun logic.

## Files (this directory)

- `log_dense_arm.py` — single-arm stride-1 wrapper around the unchanged
  week-1 logging harness.
- `run_dense_probe.sh` — end-to-end driver (re-log → hash self-check +
  curriculum-digest check → windowed scoring).
- `dense_manifest_lam097_concept_s1.json` — committed hash anchor of the
  dense artifact.
- `windowed_probes_dense_lam097_concept.json` — raw results (parent-probe
  schema; anchors, controls, CIs, fill fractions).
