# Windowed-decoder diagnostic tier (A4) — retro-probe of U-STATE archives

Question: week-1 U-STATE decoding failed at a SINGLE timestep (boundary state).
Does ANY bounded window of recent states (last-k concatenated, k in
{1,4,16,64}) recover answer-span decode above the unigram 4.54% bar — i.e. was
the negative "state-empty" or "decoder-limited"?

## Contents

- `run_retro_probes.sh` — full reproducibility driver: deterministic archive
  regeneration + sha256 verification against the COMMITTED manifests +
  curriculum-digest segmentation check + both sweeps.
- `windowed_probes_byte.json` / `windowed_probes_concept.json` — raw results
  (per arm x family x depth x decoder x task, shuffled-label controls, Wilson
  CIs, window-fill stats, anchors).

## Instrument

- Windows: last min(k-1, available) in-item pre-boundary stride snapshots
  (stride-16 logged states) in chronological order + the boundary snapshot;
  hold-first padding; never crosses the per-item encoder reset.
- Decoders: r0c softmax logistic (dim-capped; reduced budget disclosed in
  `protocol.logistic_budget_note`) AND exact linear-kernel ridge (closed form,
  FULL window dimensionality, VAL-selected lambda) as the capacity-unlimited arm.
- Tasks: G-U2 answer-span decode over the 639 VAL positions (primary; frozen
  comparators unigram 29/639 = 4.538%, majority prior 26/639, trigram 104/639,
  two-sigma threshold 6.185%) and who-said-X speaker identity on the labeled
  subset (TRAIN 60 / VAL 20).
- Controls: shuffled TRAIN labels (seed 11), majority-prior, random-ranking.

## Integrity chain

npz binaries were local-only and absent; regeneration is seeded/deterministic,
and scoring aborts unless all 39 arrays per archive match the committed
manifest hashes and the rebuilt replay reproduces the committed curriculum
digest (`c2cc7357...` byte / `f0bc99f6...` concept). Scorer bookkeeping is
anchored by exact reproduction of the frozen controls (majority 26/639,
random-seed11 1/639). See FINDINGS.md for results.
