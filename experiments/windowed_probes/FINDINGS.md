# Windowed-decoder retro-probe of the U-STATE archives — NEGATIVE CONFIRMED WITH SHARPER INSTRUMENT

Date: 2026-08-25. Branch `codex/path3-repetition`. Tier: srlm deep-dive adopt
row A4 (windowed-decoder diagnostic). Code:
`src/humanbrain/windowed_probe.py` (+18 unit tests in
`tests/test_windowed_probe.py`, all green); driver
`run_retro_probes.sh`; raw results `windowed_probes_byte.json` /
`windowed_probes_concept.json`.

## Question

Week-1 U-STATE decoding failed at a SINGLE timestep (best raw-union decode
2.19% vs unigram 4.54%). Two hypotheses were live:

- **H1 state-empty** — the union-state stream carries no answer-relevant
  information ANYWHERE in its recent history;
- **H2 decoder-limited** — information IS present in recent states but the
  single-timestep readout could not express it.

Instrument: per decision point, concatenate the last-k logged states leading
up to the boundary (k in {1, 4, 16, 64}; stride-16 snapshots from THIS item's
prompt region only + the exact boundary snapshot; hold-first padding; windows
never cross the per-item encoder reset) and score the preregistered G-U2
answer-span task (639 VAL positions, ranked-consumption scoring identical to
r0c) plus a who-said-X speaker readout, under two decoders:

1. **logistic** — the UNCHANGED r0c softmax trainer (dimension-capped at 1024,
   reduced budget 250 iters / decay@125 on the primary `union` family);
2. **ridge** — EXACT linear-kernel kernel regression (closed form through the
   Gram matrix): consumes the FULL concatenated window (up to 131,072 dims at
   k=64) with no dimension cap and VAL-selected lambda — the capacity-
   unlimited decoder arm.

Controls: shuffled TRAIN labels (seed 11) per node, majority-prior and
random-ranking span controls, Wilson 95% CIs throughout.

## Provenance (the archive was re-materialized, provably bit-identical)

The week-1 npz binaries were local-only and absent from disk. Both archives
were regenerated with the committed deterministic harness (`log_states_union.py`,
seed 7, same interpreter/numpy as week 1: py3.14.0/np2.2.6) and scoring
REFUSES to start unless every array sha256 matches the COMMITTED manifests
(39/39 arrays each) and the rebuilt replay segmentation reproduces the
committed curriculum digests (`c2cc7357…` byte, `f0bc99f6…` concept). Scorer
bookkeeping is anchored by exact reproduction of the frozen controls
(majority-prior 26/639, random-ranking-seed11 1/639). One logger quirk was
found and documented: archived `*_boundary_idx` records t-1 (the harness
appends `token_offset - 1` before incrementing); segmentation here uses the
true capture index, digest-verified.

## Result: NO window depth recovers answer-span decode — H1 confirmed

**48 configurations scored (6 arms x 2 state families x 4 depths); ZERO beat
unigram 4.538%; zero even beat majority-prior 4.069%.**

| encoding | best config | acc | Wilson 95% CI | vs bars |
|---|---|---|---|---|
| concept | lam099_concept, union, **k=4**, logistic | **2.82%** (18/639) | [1.79%, 4.41%] | CI upper < unigram 4.54% |
| byte | lam099_n2048, union, **k=1**, logistic | 2.50% (16/639) | [1.55%, 4.03%] | CI upper < unigram 4.54% |

Depth-scaling curves (accuracy by k, best decoder per node):

| arm | family | k=1 | k=4 | k=16 | k=64 |
|---|---|---|---|---|---|
| lam090 | union | 1.41 | 1.56 | 1.88 | 0.94 |
| lam097 | union | 1.56 | 2.19 | 2.19 | 2.35 |
| lam099 | union | 2.50 | 2.35 | 2.19 | 2.19 |
| lam090_concept | union | 0.78 | 2.03 | 1.41 | 1.10 |
| lam097_concept | union | 1.72 | 1.88 | 1.72 | 1.41 |
| lam099_concept | union | 1.88 | 2.82 | 1.88 | 1.41 |

(count_union ridge-only rows are flat at 0.63–2.35%, see result JSONs.)

Readings:

1. **No depth scaling.** Curves are non-monotone noise around 1–3%; several
   arms DEGRADE at k=64. The capacity-unlimited ridge — seeing every one of up
   to 131k window dims exactly — never escapes the 1–2.4% band. This is what
   H1 predicts and H2 forbids.
2. **The small real-vs-shuffled margins confirm absence, not leakage.** Real
   fits edge their shuffled-label controls at 20/24 nodes per archive, but the
   winners sit AT or BELOW majority-prior: decoders recover only the global
   answer-word frequency profile, which is available without any state.
3. **Best-over-depth adds nothing over week-1's single step**: 2.82% vs 2.19%
   (+0.6pp, well inside binomial noise at n=639).
4. **Archive saturation caveat (favors the negative).** Mean window fill:
   k=4 -> 88%, k=16 -> 56%, k=64 -> 14% (items average ~8.8 eligible stride
   snapshots; max 25). The archive's recoverable history saturates near
   k~10-16, so these sweeps already probe essentially ALL state history that
   exists in this artifact — and it contains no answer signal.

## Speaker-identity readout: also dead at every depth

Who-said-X top-1 on the labeled subset (TRAIN 60 / VAL 20, 30 train speakers):
every decoder x depth lands at 0–5%, i.e. at or below the trivial
majority-speaker baseline (1/20 = 5%); shuffled controls indistinguishable.
Only 3/20 VAL items even have an in-vocabulary speaker, so this subset is
near-chance by construction — consistent with (and weaker than) week-1 P3,
which itself showed self-store 38% -> transfer 0%.

## Verdict

NEGATIVE_WITH_LOCALIZATION_V2 is **CONFIRMED with a sharper instrument**. The
U-STATE failure localizes to the STATES, not the decoder: given every logged
state in each item's prompt history — jointly, at full dimensionality, with
regularization selected on VAL — no linear or kernel readout recovers answer
identity above a no-input frequency prior. Direction B's premise (conjunctive
context-token features written online) remains the correct target; additionally,
any future windowed diagnostic should log FINER strides (stride 1-4) if it
wants to probe within-prompt state dynamics beyond this archive's ~9-snapshot
ceiling.

## Disclosures (deliberate, none silent)

1. Logistic ran at a REDUCED budget (250 iters, decay@125, d<=1024, union
   family only) because the host measures ~0.7 GFLOPS (week-1 disclosed ~3
   GFLOPS/core); full r0c budget would be ~90 min PER FIT. Ridge carries the
   full-depth curve everywhere at full dimensionality, and scorer fidelity is
   anchored by exact control reproduction rather than a refit anchor.
2. Archives regenerated deterministically (see Provenance); hashes verified
   against the COMMITTED manifests before any scoring; npz binaries remain
   local-only and are NOT committed.
3. Archived `boundary_idx` off-by-one convention discovered and documented in
   code; does not affect any week-1 result (that harness used its own indices
   consistently).
4. Speaker subset is degenerate (3/20 val-in-vocab); treated as corroborating,
   not decisive, evidence.

## Files

- `src/humanbrain/windowed_probe.py` — loader (ustate-canonical + generic
  observation-sink fallback), window extraction, r0c-mirror scoring, both
  decoders, controls, Wilson CIs, CLI (`python -m humanbrain.windowed_probe`).
- `tests/test_windowed_probe.py` — 18 tests green (window order/padding/
  isolation, primal-dual ridge equivalence, scorer semantics incl.
  ranked-consumption, hash verification, control determinism).
- `experiments/windowed_probes/`: README.md, run_retro_probes.sh,
  windowed_probes_byte.json, windowed_probes_concept.json, FINDINGS.md.
