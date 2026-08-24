# U-STATE week-1 preregistration (Direction A) — FROZEN BEFORE IMPLEMENTATION

Status: frozen 2026-08-24, before any `sdr_union` implementation or probe
execution. Branch `codex/a013-cognition-design`. Sources adopted verbatim:
`docs/research/custom-architecture-directions.md` @ 31bb830,
`docs/research/race-pivot-gate-verdict.md`, comparator anchors in
`experiments/r0c_probes/probes.py` (UNCHANGED — it scores this experiment too).

Nothing below may be edited after the first probe token is observed. Failure is
a publishable result. All thresholds literal; all controls seeded; no re-sweep,
no threshold edits, no seed changes, no alternate decoders after seeing results.

## 1. Representation spec (pinned)

- Tokenizer: `BytePhraseTokenizer.fit(train_texts, TokenizerConfig(max_phrases=0,
  min_frequency=3))` — identical to r0c; ids 0..255 are bytes; VAL encodings are
  byte-prefixes of full serializations (asserted per item).
- Projection: for token id `t`, `rng = numpy.random.default_rng((seed, n, w, t))`;
  active set = `rng.choice(n, size=w, replace=False)` (sorted); `s_t` is the
  w-hot indicator vector over `n` bits. Defaults `n=2048`, `w=32`, `seed=7`.
- Union state: `U_t = max(lam * U_{t-1}, s_t)` elementwise, `U_0 = 0`, float64
  internally. This IS the network state; there is no reservoir in week-1 probes.
- Count-union variant (diagnostic only): `C_t = lam * C_{t-1} + s_t` (recency-
  weighted multiplicities; analog of the eligibility-trace bank).
- Decode helper (diagnostics): score(t) = mean of `U[b]` over bits `b` of sdr(t);
  ranking by (-score, token_id).
- Determinism: same (seed, n, w, tokens, lam) => bit-identical states, always.

## 2. Empirical capacity precondition (measured BEFORE this freeze)

Script: `experiments/ustate_probes/capacity_precondition.py`; raw JSON beside it.
Bible TRAIN-200 prompts only (mean 147.6 / median 158.5 / range 30–393 byte
tokens). Verbatim:

| metric | analytic | measured |
|---|---|---|
| pairwise token-SDR overlap | w^2/n = 0.5 bit | mean 0.4885912698412698, median 0.0 |
| fresh-union support, L=200 distinct | n(1-(1-w/n)^200) = 1960.21 bits | (bound only; see below) |
| support (any touch), all lambdas | decay never reaches exact 0 | 786.52 bits mean |
| effective support (>=1e-2 floor) | ~w/(1-lam) window | 609.125 (lam .90) / 752.74 (.97) / 786.52 (.99) |
| pooled median 1-cosine, TRAIN pairs | G-U1 anchor | 0.18041589038669315 (.90) / 0.18983825453666142 (.97) / 0.17330033581913007 (.99) |
| max cross-item cosine | — | 0.9999106408477216 (.90) / 0.9932868856522437 (.97) / 0.9816014133780478 (.99) |
| share of pairs cosine > 0.90 | — | 0.20045226130653265 (.90) / 0.10592964824120603 (.97) / 0.14467336683417084 (.99) |
| universal bits (active >=95% items) / their L2 mass share | — | 389 bits; mass share 0.8201137726921257 (.90) / 0.6840894975655409 (.97) / 0.6043453140932328 (.99) |

Reading frozen here: the MEDIAN pair clears the 0.10 separation anchor at every
lambda, but a collapse tail exists (common-byte universal mass 0.60–0.82 of L2).
G-U1 therefore keys on the pooled MEDIAN exactly like the r0b G1 precedent
(C_rand 0.1019); tail statistics are reported verbatim in FINDINGS either way.

Reproduction check (gate 0): the D3 logged-state run must reproduce this table
within 2x per metric (same split, same projection). Any metric off > 2x =>
STOP before scoring anything: implementation/spec bug, not a scientific result.

## 3. Arms, sweep, and npz slot mapping (frozen before code)

Lambda sweep {0.90, 0.97, 0.99} becomes three arms at n=2048, w=32, seed 7:
`arm_lam090_n2048`, `arm_lam097_n2048`, `arm_lam099_n2048`.

Logged via the observation_sink replay pattern (mirror of
`experiments/r0c_probes/log_states.py`; same corpus manifests, same split_prompt
"\nA:" rule, same teacher-forced boundary capture, same curriculum-digest
construction — digest MUST equal the r0c value
`c2cc7357ab073d0c8447c203e4482eedb751a464dce6df3d0aa134b66c65ee5c` because
corpus+tokenizer are identical; mismatch => STOP).

Slot mapping into the UNCHANGED `probes.py` layout:

| npz array | content |
|---|---|
| `*_boundary_voltage` (f64) | U_T float64 union state (n=2048) |
| `*_boundary_spikes` (u32) | current-token SDR indicator s_T in {0,1}^2048 |
| `*_boundary_feature` (f32) | U_T as float32 |
| `*_boundary_traces` (f32) | C_T count-union (diagnostic feature set) |
| `*_tokfeat__union_*` / `*_tokfeat__count_union_*` | per-token stride rows f16 (stride 16; boundary rows f32) |

Consequence (mechanical): probes.py P1/P2 "raw_voltage_spikes" feature set =
[U_T || s_T] standardized, dim 4096 (= max_d cap, no subsampling); "traces_bank"
= C_T. P2 SDM keys = train-median binarization thereof. Corpus pins inherited
from r0c records: FREEZE-CANDIDATE-v3-BIBLE.json sha256
`361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c`, TRAIN
manifest `d256c27dacec0cb2a169005611338c3408bbb252ba619dcde86fb01e3f8c9a7b`,
VAL manifest `2771de48a90d9e694cc4260e9392d83c8e39f2aca7d57fe06dd51c7ce9a55bec`
(logger recomputes all three into its own manifest; any mismatch => STOP).
Probes invoked exactly once: `python experiments/r0c_probes/probes.py
--probes-dir experiments/ustate_probes` against TRAIN 200 / VAL 60 (639
answer-word positions enforced by SpanScorer).

## 4. Frozen comparators (verbatim, unchanged)

- unigram span-acc 0.04538341158059468 (29/639)
- trigram span-acc 0.162754303599374 (104/639)
- two-sigma threshold p0 + 2*sqrt(p0(1-p0)/639) = 0.061851478566709396
  (sigma = 0.008234...; z >= 2 vs unigram is the operative test)
- majority-prior-within-answer-vocab and random-ranking-seed-11 controls come
  from probes.py itself; shuffled-label (P1) and shuffled-key (P2, CONTROL_SEED
  11) controls likewise. None may be modified.

## 5. Gates

- **G-U1 separation** (per arm): pooled median (1 - cosine) across ALL distinct
  item pairs of boundary unions U_T >= 0.10 on TRAIN AND on VAL (states only;
  no labels involved). Equivalently pooled median cosine <= 0.90. Tail stats
  (max cosine, share > 0.90, universal-mass share) reported but NOT gating.
- **G-U2 decode** (per arm): best of {logistic, MLP} span-acc on the
  `raw_voltage_spikes` feature set >= 0.061851478566709396 (z >= 2 vs unigram
  4.54%). Only `raw_voltage_spikes` gates; `traces_bank` rows are diagnostics.
- **G-U3 anti-leakage** (per arm, at the arm's best G-U2 config):
  (a) P2 shuffled-key control accuracy <= 0.061851478566709396, AND
  (b) P1 mlp_shuffled_label_control accuracy <= majority-prior accuracy +
      2*sigma (i.e. <= 0.057151... on the identical scorer).
  "Shuffled-key control ~= majority-prior": neither control may clear the gate.
- **G-U4 fixture replay** (pass path only): typed-program capacity replay leg
  (incumbent machinery, 42 tasks) through the integrated circuit with
  `feature_family="sdr_union"` installed must reach >= 39/42 (G-P3 precedent).
- ESCALATE-(d) supersession: if best raw-union MLP span-acc >= trigram
  0.162754303599374, escalate option (d) as a constraint decision regardless of
  other gates (probes.py d5_verdict computes this mechanically).

An arm QUALIFIES iff G-U1 ∧ G-U2 ∧ G-U3. Week-1 proceeds iff >= 1 arm qualifies.

## 6. Gate order and stop-on-failure

0. Reproduction of §2 precondition within 2x (else STOP, fix nothing silently).
1. Log states (TRAIN+VAL), assert curriculum digest == r0c value (else STOP).
2. G-U1 per arm. Arms failing G-U1 are excluded from qualification (their probe
   rows are still emitted once, mechanically, as localization diagnostics).
3. UNCHANGED probes.py P1/P2/P3 once on all arms; G-U2/G-U3 evaluated per arm
   from `probes_results.json` verbatim numbers.
4. Verdict (§7). NEGATIVE stops everything: no integration, no VAL cohort spend.
5. Pass path only: G-U4 replay leg, then the ONE-SHOT Bible-QA VAL race leg at
   N=96; N=1024 runs ONLY if N=96 passes its gates. No reruns, no second seeds.

Local CPU diagnostics (P1-P3) always run once even when the verdict is already
determined — localization is mandatory; what stop-on-failure protects is
downstream cohorts and integration compute, not the written record.

## 7. Mechanical verdict (D3 checkpoint)

Computed from `experiments/ustate_probes/probes_results.json` + the logger
manifest's G-U1 block, no judgment calls:

- `ESCALATE_(d)`: qualifies exist AND best qualifying-arm MLP >= trigram.
- `PROCEED_TO_INTEGRATION`: some arm qualifies AND best MLP < trigram.
- `NEGATIVE_WITH_LOCALIZATION_V2`: otherwise. FINDINGS must localize: which
  gate died (separation / decode / leakage-control), at which lambda, with the
  verbatim table; Direction B (apical coincidence encoder) is elevated in the
  report. If the pass path later fails G-U4, the final verdict downgrades to
  NEGATIVE_WITH_LOCALIZATION_V2 (capacity-regressive-at-integration), cohort
  spent, no retry.

## 8. Commit discipline

Incremental commits after each day-milestone on
`codex/a013-cognition-design`: D1 this document (+precondition script/json),
D2 `src/humanbrain/sdr_union.py` + red tests, D3 logger + npz manifest +
`probes_results.json` + `experiments/ustate_probes/FINDINGS.md` (npz binary
stays local; array-level sha256s committed in the manifest), D4-5 pass/fail
path artifacts. Push after each milestone.
