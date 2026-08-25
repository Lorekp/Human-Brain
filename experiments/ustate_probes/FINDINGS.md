# U-STATE week-1 D1-D5 findings: SDR union-state probes — NEGATIVE-WITH-LOCALIZATION-V2

Date: 2026-08-25 (probe window opened 2026-08-24 20:05 UTC). Branch
`codex/a013-cognition-design`. Protocol:
`docs/research/ustate-preregistration.md` @ d2b97d7 + `ustate-change-note-1.md`
@ eec20ac (concept-arm amendment per b6fb803, filed pre-verdict) + erratum.
Corpus: Bible v3-freeze TRAIN 200 / VAL 60 (639 answer-word positions),
byte tokenizer identical to r0c; curriculum digest byte arms =
`c2cc7357ab073d0c8447c203e4482eedb751a464dce6df3d0aa134b66c65ee5c` (= r0c
anchor), concept arms = `f0bc99f69ef3caca1a3efe54b421a565e12f957e081d8ec29ab8f3c4c29998be`.

## D5 DECISION CHECKPOINT VERDICT: NEGATIVE_WITH_LOCALIZATION_V2

Mechanical rule evaluation (`verdict.json`, preregistration §7):

| arm | G-U1 sep | G-U2 decode >= 0.061851 | G-U3 anti-leak | qualifies |
|---|---|---|---|---|
| lam090_n2048 (byte) | PASS (0.1804/0.1825) | **FAIL** (logistic 0.015649) | PASS | no |
| lam097_n2048 (byte) | PASS (0.1898/0.1878) | **FAIL** (logistic 0.015649) | PASS | no |
| lam099_n2048 (byte) | PASS (0.1733/0.1657) | **FAIL** (mlp 0.021909) | PASS | no |
| lam090_concept_n2048 | PASS (0.5637/0.6074) | **FAIL** (mlp 0.014085) | PASS | no |
| lam097_concept_n2048 | PASS (0.4322/0.4669) | **FAIL** (logistic 0.018779) | PASS | no |
| lam099_concept_n2048 | PASS (0.4023/0.4562) | **FAIL** (logistic 0.021909) | PASS | no |

No arm qualifies => NEGATIVE_WITH_LOCALIZATION_V2. Stop-on-failure fired at
G-U2: G-U4 (fixture replay leg) was NEVER reached and NO downstream cohort was
spent. Per the fail path: Direction B (apical coincidence encoder, G-A gates)
is ELEVATED as the primary custom-architecture direction.

## What was run

- D1 freeze committed BEFORE implementation (d2b97d7): gates, gate order,
  stop-on-failure, lambda sweep {0.90,0.97,0.99}, capacity precondition
  measured pre-freeze (`capacity_precondition.py/.json`; pairwise token-SDR
  overlap 0.48859 vs analytic w^2/n=0.5).
- Gate 0 reproduction: logged-state stats within [0.85x, 1.00x] of the frozen
  table on every metric (manifest `gate0_reproduction.pass=true`).
- D2 `src/humanbrain/sdr_union.py` (280 LoC): seeded projection
  `default_rng((seed,n,w,t)).choice`, union `U=max(lam*U,s)`, count union,
  decode, cosine diagnostics; 16 red->green tests (determinism / capacity /
  decay identities), all green.
- D3 logging via the observation_sink replay pattern
  (`log_states_union.py`): teacher-forced replay of all 260 items x 3 lambdas
  x {byte, concept}; boundary = prompt-through-"A:" state; determinism
  reset-and-replay spot-checks PASS on all six arms; artifacts
  `experiments/{ustate_probes,ustate_probes_concept}/logged_states.npz`
  (31.8 MB / 23.4 MB local-only) + hash manifests (committed).
- UNCHANGED `experiments/r0c_probes/probes.py` P1/P2/P3 executed once per arm
  on bit-identical per-arm view artifacts (`make_arm_views.py`; sha256-verified
  against canonical manifests), merged verbatim into
  `probes_results_merged.json` (`merge_results.py`). Controls verified
  identical across views: majority-prior 0.0406885758998435 (26/639),
  random-ranking-seed11 0.001564945226917058 (1/639).

## Frozen comparators (verbatim)

unigram 0.04538341158059468 (29/639); trigram 0.162754303599374 (104/639);
G-U2 two-sigma threshold 0.061851478566709396 (sigma=0.008234).

## Probe results (verbatim)

### P1 — logistic + MLP decode on RAW unions ([U_T || s_T], dim 4096)

| arm | feature set | logistic | MLP | mlp_shuffled-label |
|---|---|---|---|---|
| lam090_n2048 | raw_voltage_spikes | 0.015649 (10) | 0.010955 (7) | 0.009390 |
| lam090_n2048 | traces_bank | 0.018779 (12) | 0.007825 (5) | 0.018779 |
| lam097_n2048 | raw_voltage_spikes | 0.015649 (10) | 0.009390 (6) | 0.012520 |
| lam097_n2048 | traces_bank | 0.014085 (9) | 0.014085 (9) | 0.018779 |
| lam099_n2048 | raw_voltage_spikes | 0.015649 (10) | 0.021909 (14) | 0.021909 |
| lam099_n2048 | traces_bank | 0.017214 (11) | 0.014085 (9) | 0.020344 |
| lam090_concept | raw_voltage_spikes | 0.012520 (8) | 0.014085 (9) | 0.007825 |
| lam090_concept | traces_bank | 0.020344 (13) | 0.018779 (12) | 0.015649 |
| lam097_concept | raw_voltage_spikes | 0.018779 (12) | 0.014085 (9) | 0.017214 |
| lam097_concept | traces_bank | 0.012520 (8) | 0.014085 (9) | 0.017214 |
| lam099_concept | raw_voltage_spikes | 0.021909 (14) | 0.018779 (12) | 0.015649 |
| lam099_concept | traces_bank | 0.021909 (14) | 0.018779 (12) | 0.012520 |

**Best raw-union decode anywhere: 0.021909 (2.19%)** — below unigram 4.54%,
below majority-prior 4.07%, one third of the G-U2 bar 6.19%. Every learned
decoder loses to ignoring-the-state (majority prior). Shuffled-label controls
(0.8-2.2%) sit below their winners: the failure is honest signal absence, not
leakage or training pathology.

### P2 — SDM store/query sweep (train-median binarized keys)

| arm | best span-acc | correct | distinct | shuffled-key ctrl | z vs unigram | best config | gate |
|---|---|---|---|---|---|---|---|
| lam090_n2048 | 0.031299 | 20 | 30 | 0.032864 | -1.711 | r=None k=None | FAIL |
| lam097_n2048 | 0.031299 | 20 | 30 | 0.032864 | -1.711 | r=None k=None | FAIL |
| lam099_n2048 | 0.031299 | 20 | 30 | 0.032864 | -1.711 | r=None k=None | FAIL |
| lam090_concept | 0.029734 | 19 | 119 | 0.015649 | -1.901 | r=None k=8 | FAIL |
| lam097_concept | 0.029734 | 19 | 119 | 0.015649 | -1.901 | r=None k=8 | FAIL |
| lam099_concept | 0.029734 | 19 | 119 | 0.015649 | -1.901 | r=None k=8 | FAIL |

Byte optimum is again the FULL store (r0c's "no neighborhood structure"
signature); concept keys yield far more diverse retrievals (119 vs 30 distinct
answers) and cleaner anti-leakage separation (control 1.56% vs query 2.97%),
but accuracy stays BELOW the unigram comparator in every arm.

### P3 — who-said-X speaker recovery (best P2 config per arm)

| arm | VAL pooled | TRAIN pooled (self-store) |
|---|---|---|
| byte arms (all lambdas) | 0.10 (2/20) | 0.0333 (2/60) |
| concept arms (all lambdas) | **0.00 (0/20)** | **0.3833 (23/60)** |

The concept self-store number is the sharpest localization datum of the week:
the SDM over concept-key unions replays its OWN items' speaker tokens at 38%
while transferring ZERO to held-out items. Capacity and retrieval machinery
work; the stored content simply does not generalize because the mapping
context-identity -> answer words is absent from these states.

## Localization (v2)

1. **Separation is NOT the bottleneck — twice confirmed.** G-U1 passes with
   huge margins everywhere (concept arms 3-5x the 0.10 bar; collapse tail
   0.06-0.11% of VAL pairs vs bytes' 12.6-20.7%), replicating b6fb803's
   geometry result on QA items. Yet G-U2 fails HARDER than the falsified
   reservoir states did (best 2.19% here vs r0c's 3.29%). The r0-b lesson
   ("separation does not imply decodability") now replicates for INPUT-SIDE
   sparse codes: geometric context identity does not encode ANSWER IDENTITY.
2. **The composition-study advantage did not transfer.** Concept unions beat
   byte unions on every geometry metric (median 1-cosine up to 3.4x higher;
   tails ~300x smaller) exactly as b6fb803 predicted, but book-ID probe
   separability (+54% relative at lambda=.99) bought NOTHING on answer-span
   decoding (best 2.19% both encodings). Task-relevant information, not
   token diversity, is what the decoder lacks.
3. **Store/replay capacity intact, generalization zero** (P3 self-store 38.33%
   vs held-out 0.00% for concepts). The associative route fails at content,
   not at mechanics.
4. **Scale-of-decay irrelevant**: results are flat across the entire lambda
   sweep for both encodings — this is not a recency-window tuning problem.

Consequence per the frozen falsifier clause: the pure input-side SDR
union-state route is DEAD for QA reading on this evidence base. The week's
original organizing insight ("write context into retrievable form at encode
time") is insufficient: what must be WRITTEN is task-conditioned conjunctions,
not raw token identities.

## Direction B elevated

Per preregistration §7 fail path, **Direction B (apical dendritic coincidence
encoder)** becomes the primary custom-architecture direction; its week-1
preregistration (05fb3ef) and cell implementation (f57dc20) are already landed
by the parallel workstream, and its premise directly targets the localized
deficit: conjunctive (context AND token) features created online, rather than
superposed token sets. The P3 self-store/generalization split also sanctions a
Direction-B follow-up diagnostic: score apical arms on TRAIN-self-store vs
VAL generalization separately, since that contrast cleanly detects the same
failure mode early.

## Disclosures and deviations (all deliberate, none silent)

1. **Change Note 1** added three concept arms post-freeze but PRE-verdict,
   citing b6fb803 (pre-spend composition study); byte arms untouched; verdict
   rule unchanged (arm-agnostic). Erratum: boundary cut refined to
   prefix-consistency because 18 mined concepts contain the literal `\nA:`
   marker; concept arms uniformly omit the final 4-8 cue bytes
   (`boundary_missed_prompt_bytes` recorded per row) — observing the
   straddling token would have leaked answer-region bytes into decision
   states. Byte arms keep the exact r0c boundary semantics.
2. **Execution topology**: probes ran once per bit-identical per-arm view
   artifact (not once per encoding artifact) because the CI host measures
   ~3 GFLOPS/core and serial execution exceeded feasibility (~20h). Arrays are
   sha256-identical to the canonical manifests; controls verified identical
   across views; results merged verbatim. The first (killed) serial byte-leg
   invocation emitted no results before termination.
3. **BLAS threading**: single-threaded OpenBLAS pinned via env var for all
   probe invocations (thread-count affects only float summation order at
   1e-16 scale; seeds and code paths unchanged).
4. **Decoder note**: ranked-consumption decoding and 639-position scoring are
   probes.py's own, unchanged; heads restricted to the 673-type TRAIN answer
   vocabulary (same disclosed advantage as r0c).
5. npz binaries stay local; array-level sha256s for all arrays are in the
   committed manifests; per-view `probes_results.json` files are committed
   alongside the merge.

## Files

- `docs/research/ustate-preregistration.md`, `docs/research/ustate-change-note-1.md`
- `src/humanbrain/sdr_union.py`, `tests/test_sdr_union.py` (16 tests green)
- `experiments/ustate_probes/`: capacity_precondition.{py,json},
  log_states_union.py, make_arm_views.py, merge_results.py, verdict.py,
  logged_states_manifest.json, probes_results_merged.json, verdict.json,
  view_*/probes_results.json (+ concept twins under
  `experiments/ustate_probes_concept/`)
- Local only: `logged_states.npz` (both artifacts), per-view npz copies

## Bottom line for the roadmap

NEGATIVE_WITH_LOCALIZATION_V2, mechanically decided, cohort-clean. Input-side
SDR unions are maximally separable and minimally decodable: the QA-relevant
information is not present at the boundary state under either encoding, so no
readout or store built on top can recover it. Direction A dies cleanly;
Direction B (apical coincidence encoding of context-token conjunctions)
inherits the mainline with a sharpened early-warning diagnostic.
