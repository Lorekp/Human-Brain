# Change Note 3 (T-ROUTE) — E2 probe amendment: raw unnormalized W + top-k row
# sparsification. PRE-SPEND for E2: committed BEFORE any E2 episode is encoded,
# any E2 winner set is routed, or any E2 gate is evaluated.

Parent protocol: `docs/research/trained-inter-space-preregistration.md` (FROZEN
2026-08-25; E1 SPENT with verdict `NEGATIVE_CORE_WOUND` recorded verbatim in
`experiments/t_route/e1_results.json`). This note amends NOTHING about E1 and
edits nothing in the parent document — it is a standalone, cited amendment that
defines a new probe (E2) scoped as follows. After the first E2 pairing event is
encoded, nothing below may be changed. Failure is a publishable result.

## C1. Motivation (context, not a gate)

E1 localized its core wound to arm A's post-training transform: the IDF-gain
homeostat collapsed winners onto a fixed rare-byte pool disjoint from gold
regions (who_said base_med exactly 0.0 on both splits) while the SAME trained
substrate's unnormalized dense diagnostic (arm B) reached 79.5/80.0 median
overlap (~7x randomized-W). Hypothesis under probe: **the homeostat, not the
Hebbian accumulation, destroys routing signal**; if so, dropping normalization
and optionally adding top-k row sparsity should recover G-T1-style enrichment
on the token-granularity sparse substrate itself.

## C2. Arm family (frozen)

For each k in the sweep {8, 16, 32}:

- Substrate: the UNCHANGED §1/§2 trained accumulator — sparse W (W_SEED=7,
  fan-in 32, init U(0.5,1.5)) plus ONE-SHOT aggregated token-pair delta via
  `apply_dense_delta` over TRAIN episodes only (O_MAX=8, H_WIN=12). This is
  byte-for-byte the matrix whose homeostat transform produced E1 arm A.
- Post-training transform (REPLACES `apply_homeostasis`; NO homeostat, NO
  normalization of any kind): per row b, keep the k entries with LARGEST value
  among the row's structural nonzeros and zero the rest; ties broken by
  ASCENDING column index; rows with fewer than k nonzeros keep all of them.
  Deterministic selection; NO random source is introduced anywhere in E2.
- Routing readout: UNCHANGED §4 machinery — `drive = M_k @ q_pop`, winners =
  `kwinner_indices(drive, K=128)` (ties ascending index), q_pop = undecayed
  question union through the UNCHANGED projector (n_bits=2048, w_ones=32,
  seed=7). k=32 is the identity transform: it isolates "homeostat dropped,
  nothing else" exactly.
- Config labels: `arm_k8`, `arm_k16`, `arm_k32`.

## C3. Controls (identical to E1, frozen)

- `shuffled_question`: within each split, item i routed with the question of
  item pi(i), `pi = default_rng(11).permutation(len(split))`, THROUGH THE SAME
  CONFIG's sparsified W; scored against item i's own gold region.
- `randomized_W`: untrained seed-13 sparse matrix (never sees an episode),
  shared across configs; routed once.
- No wrong-passage / shuffled-W legs: E1's G-T3a denominator was empty by
  corpus construction and E2 changes only the post-transform; E2 gates below do
  not include a G-T3 re-evaluation.

## C4. Reproducibility precondition (must hold before any E2 logging)

The corpus pins, tokenizer fit recipe, projector, accumulation, and control
seeds are inherited verbatim from the parent §0-§3 (all digests re-verified).
Before routing ANY E2 config, the harness MUST deterministically rebuild every
E1 substrate and demonstrate BIT-EXACT agreement of all six E1 routing sets
(`armA`, `armB`, `shuffled_question`, `randomized_W`, `wrong_passage`,
`shuffled_W`) against `experiments/t_route/logged_winner_sets.npz` after that
file passes its committed `e1_manifest.json` array-hash verification. Any
mismatch => STOP; nothing E2 runs. Gold masks, matched flags, families, sigma,
and permutations for scoring are taken FROM the verified E1 log (same pairing
data guaranteed), never recomputed independently.

## C5. Gates (frozen)

Same G-T1 structure per config, evaluated by a scorer that reads ONLY the
logged E2 sets (+ the hash-verified E1 log for controls/baseline):

- Eligibility: GOLD_UNMATCHED exclusion positional per parent §4 (matched flag
  from the E1 log). Strata: `who_said` GATING; `address_lookup`,
  `verse_continuation` REPORTED (expected empty denominators).
- Per config c, per split s in {TRAIN-self, VAL}: medians of
  |winners ∩ gold_bits| over eligible items for base = c, shuf =
  shuffled_question(c), rand = randomized_W.
- Config c PASSES G-T1 iff on BOTH splits:
  `base_med > 0 AND base_med >= 2*shuf_med AND base_med >= 1.5*rand_med`.
- E2-level verdict (mechanical): ANY config passing =>
  `ROUTING_RECOVERED_WITHOUT_HOMEOSTAT`; no config passing =>
  `RECOVERY_FAILS_SPARSITY_INSUFFICIENT`. The disjunction across the frozen
  3-point sweep is declared NOW (multiplicity acknowledged; no post-hoc
  selection of a primary k).
- Descriptive contrast (frozen, not gated): overlap medians of the LOGGED E1
  `armB` winner sets recomputed from the verified log on the same eligible
  items, reported beside each config; also whether max_k base_med beats arm B
  per split. If raw dense (k=32 / arm-B-scale signal) is already maximal,
  sparsity is concluded NOT to help routing — only memory footprint — and that
  conclusion is stated verbatim in FINDINGS.
- Headroom context reported: median/quantiles of gold region sizes per stratum
  per split (interpretation aid only).

## C6. One-shot discipline

`run_e2.py` executes the whole frozen pipeline ONCE: integrity pins ->
deterministic rebuild -> C4 reproduction check -> route/log ALL configs and
controls into `e2_logged_winner_sets.npz` + array-hash manifest (refuses to
overwrite; binary npz stays local). `score_e2.py` then evaluates gates FROM THE
LOGGED SETS ONLY into `e2_results.json`. No reruns, no threshold edits, no seed
changes, no additional k values, no alternate decoders after seeing results.

## C7. Artifacts

All under `experiments/t_route/e2/`: this note, `run_e2.py`, `score_e2.py`
(committed before spend), then `e2_manifest.json`, `e2_results.json`,
`FINDINGS.md` (committed after the single execution). Commit discipline per
parent §8: incremental commits on `codex/a013-cognition-design`, pull before
push, only own files touched.
