# R0-c Thread-1 D1-D5 findings: state-logging harness + P1/P2/P3 offline probes

Date: 2026-08-24. Branch: `codex/a013-cognition-design` (worktree
human-brain-a013-cognition-design). Strategy source:
`docs/research/r0c-strategy-decision.md` @ c7d8c0a. Corpus:
FREEZE-CANDIDATE-v3-BIBLE.json (`candidate_sha256`
`361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c`),
TRAIN manifest `d256c27dacec0cb2a169005611338c3408bbb252ba619dcde86fb01e3f8c9a7b`,
VAL manifest `2771de48a90d9e694cc4260e9392d83c8e39f2aca7d57fe06dd51c7ce9a55bec`.

## D5 DECISION CHECKPOINT VERDICT: NEGATIVE-WITH-LOCALIZATION

Mechanical rule evaluation (strategy doc, Thread 1):

1. **Escalate-(d)? NO** - best learned-decoder span-acc 3.29% << trigram 16.28%
   (0.162754303599374). Nowhere near a constraint decision for the user.
2. **Integrate? NO** - P2 memory route fails its gate at every arm/scale (below).
3. **Negative-with-localization? YES** - best MLP span-acc 0.03286384976525822
   < unigram 0.04538341158059468. Per the preregistered rule: *the
   reservoir-state route is dead for QA; keys fall back to input-assembly
   space.*

Thread-1 D6-9 integration work does NOT start. Runner-up thread (broadcast +
HOT-2 margins, shared injection seam) and input-assembly-space keys become the
live options; localization evidence below says the failure is in the reservoir
STATE, not the decode chain - so reusing the injection seam with
input-derived keys (the banked assembly-overlap-random space) is the
consistent next probe target.

## D1-2 harness (what was run)

- Growth mirrors `RecurrentCellLanguageModel.grow`: byte-level tokenizer
  (`max_phrases=0`, vocab 256), `grow_sequence(epochs=1)` over the 200 TRAIN
  serialized texts (prompt-through-`A:` + answer framing from the v3
  manifests), ridge readout, seed 7, ladder config
  (`input_fanout=16`, `recurrent_fanout=8`, inhibitory 0.2). N=1024 uses the
  ladder scipy/OpenBLAS ridge seam (`install_scipy_ridge`; F+1 = 8193);
  N=96 uses the numpy path (F+1 = 769).
- Two arms per N (they cannot share a circuit - see Deviations):
  - `explicit_spike` (`feature_family="spike_counts"`): explicit ladder
    afferent wiring; logs per-token spike-count windows AND the trace bank,
    i.e. families {spike_counts, traces}.
  - `asm_rand` (`feature_family="assembly_overlap"`): banked random codebook
    installed over afferents (k=16, seed-streams `[seed, 0xC0DEB005, token]`);
    logs family {assembly_overlap-rand}.
- Replay: teacher-forced observe over every TRAIN/VAL serialized text;
  **decision point** = state right after observing the prompt through `A:`
  inclusive (byte-token index `len(prompt_ids)-1`). Logged there: raw
  `voltage_relative_mv` (f64), `_token_spike_counts` (u32),
  `feature_state_vector()` (f32), flat trace bank (f32). Per-token feature
  streams: f32 at every decision point, float16 elsewhere at stride 16
  (N=1024) / stride 1 (N=96).
- Determinism: replay reset-and-redo spot-check reproduces boundary voltage
  AND feature vector bit-for-bit - PASS at all four arm x N configs. Curriculum
  digest identical across arms and scales:
  `c2cc7357ab073d0c8447c203e4482eedb751a464dce6df3d0aa134b66c65ee5c`.
- Artifacts: `experiments/r0c_probes/logged_states.npz` (292.5 MB, local only -
  array-level sha256s recorded in `logged_states_manifest.json`), plus
  `logged_states_manifest.json`, `probes_results.json` (committed).

## Frozen comparators (experiments/qa_baselines_bible_v1, verbatim)

| comparator | VAL span-acc | correct/639 |
|---|---|---|
| unigram_smoothed | 0.04538341158059468 | 29 |
| trigram_smoothed | 0.162754303599374 | 104 |

P2 gate threshold, computed exactly: p0 + 2 sigma with p0 = 29/639 gives
sigma = 0.008234 -> **two-sigma accuracy threshold = 0.061851478566709396**
(the task brief's "~7%" approximation was pessimistic; the z>=2 test is the
operative gate and it is reported here verbatim).

## Probe results

### P1 - decode ceiling on raw decision-point states (logistic + 2-layer MLP)

Head restricted to the TRAIN answer vocabulary (673 word types) - an
intentional decoder advantage over the full-vocab comparators; see
Disclosures. Training targets are answer-bag soft labels; decoding by
ranked-consumption (see Disclosures). Controls on the identical scorer:

| control | span-acc | correct/639 |
|---|---|---|
| majority prior within answer vocab (no state at all) | 0.0406885758998435 | 26 |
| random ranking (seed 11) | 0.001564945226917058 | 1 |

Learned decoders (all 10 configs; `mlp_shuf` = shuffled-label control):

| arm x N | feature set | logistic | MLP | MLP shuffled-label |
|---|---|---|---|---|
| asm_rand N=1024 | asm_overlap_rand_features (4096 dims) | 0.026604 (17) | 0.025039 (16) | 0.017214 |
| asm_rand N=1024 | raw_voltage_spikes (2048) | 0.025039 (16) | 0.025039 (16) | 0.025039 |
| asm_rand N=1024 | traces_bank (4096 of 8192) | 0.006260 (4) | **0.032864 (21)** | 0.007825 |
| asm_rand N=96 | asm_overlap_rand_features (768) | 0.025039 (16) | **0.032864 (21)** | 0.014085 |
| asm_rand N=96 | raw_voltage_spikes (192) | 0.021909 (14) | 0.026604 (17) | 0.012520 |
| asm_rand N=96 | traces_bank (768) | 0.014085 (9) | 0.014085 (9) | 0.021909 |
| explicit_spike N=1024 | raw_voltage_spikes (2048) | 0.015649 (10) | 0.029734 (19) | 0.025039 |
| explicit_spike N=1024 | traces_bank (4096 of 8192) | 0.017214 (11) | 0.025039 (16) | 0.015649 |
| explicit_spike N=96 | raw_voltage_spikes (192) | 0.018779 (12) | 0.020344 (13) | 0.018779 |
| explicit_spike N=96 | traces_bank (768) | 0.017214 (11) | 0.015649 (10) | 0.020344 |

Readings:

- **Best MLP 0.03286384976525822 < unigram 0.04538341158059468** - the
  preregistered "route dead" branch fires. Best logistic 0.026604068857589983.
- Every learned decoder loses to the majority-prior control (4.07%) computed
  on the SAME restricted head: conditioning on the boundary state is worth
  LESS than ignoring the state entirely. The states carry prior-like
  frequency information, not item-specific answer identity.
- Scale does not rescue it: N=1024 does not dominate N=96 on matched feature
  sets (raw-state MLP 2.97% vs 2.03%, but asm-rand-feature MLP 2.50% vs
  3.29%) - consistent with a representation deficit, not a capacity deficit.

### P2 - SDM-style associative memory route (store/query sweep)

Keys: sign-vs-train-median binarization of the answer:-boundary state
(`asm_rand` arm keys = assembly-overlap-rand feature vector; `explicit_spike`
arm keys = raw voltage||spike snapshots). Store: TRAIN key -> answer-token
set. Query: weighted vote of stored sets by bit-agreement overlap. Sweep:
radius {None, 0.50..0.90 step 0.05} x top-k {None,1,2,4,8,16}, selected on VAL
per the strategy's explicit sweep instruction (full table in
`probes_results.json`). Gate: span-acc >= 0.061851478566709396 AND distinct
predicted answers >= 3 AND shuffled-key control <= gate threshold.

| arm x N | key source | bits | best span-acc | correct/639 | z vs unigram | distinct | shuffled-key ctrl | gate |
|---|---|---|---|---|---|---|---|---|
| explicit_spike N=1024 | raw_voltage_spikes | 2048 | **0.0406885758998435** | 26 | -0.570 | 30 | 0.025039 | FAIL |
| asm_rand N=96 | asm_overlap_rand feats | 768 | 0.035994 | 23 | -1.140 | 30 | 0.029734 | FAIL |
| explicit_spike N=96 | raw_voltage_spikes | 192 | 0.035994 | 23 | -1.140 | 30 | 0.035994 | FAIL |
| asm_rand N=1024 | asm_overlap_rand feats | 8192 | 0.026604 | 17 | -2.281 | 30 | 0.021909 | FAIL |

Readings:

- **No configuration passes.** Best (explicit-wiring raw states, N=1024)
  reaches only 0.0406885758998435 - below the unigram comparator itself and
  far below the two-sigma threshold 0.061851478566709396.
- In EVERY arm x N the sweep optimum is the FULL store (`radius=None`,
  `top_k=None`) - any thresholding or sparsification of the neighborhood only
  removes signal. Binarized key space has no item-selective neighborhood
  structure to exploit: retrieval degrades gracefully toward a global
  answer-frequency prior (which lands near unigram-level performance, as it
  should).
- Anti-leakage controls behave: shuffled-value controls score at or below
  their query accuracies (no key leakage inflating the store), and distinct
  predicted answers = 30 >= 3 (only the accuracy condition fails).

### P3 - who-said-X speaker recovery by chapter-distance bucket

Best P2 config per arm/N (full store), consumption-decoded answers scored by
speaker-token overlap. VAL who_said n=20 (distances: 17x <=1, 1x d=3, 2x d=6);
TRAIN who_said n=60 (58x <=1, 1x d=3, 1x d=8):

| arm x N | VAL pooled | VAL <=1 | VAL 2-4 | VAL 5-8 | TRAIN pooled |
|---|---|---|---|---|---|
| asm_rand N=96 | 0.10 (2/20) | 2/17 | 0/1 | 0/2 | 0.0333 (2/60) |
| explicit_spike N=96 | 0.10 (2/20) | 2/17 | 0/1 | 0/2 | 0.0333 (2/60) |
| explicit_spike N=1024 | 0.05 (1/20) | 1/17 | 0/1 | 0/2 | 0.0333 (2/60) |
| asm_rand N=1024 | 0.00 (0/20) | 0/17 | 0/1 | 0/2 | 0.0333 (2/60) |

Localization reading: the VAL distance distribution is concentrated at <=1
chapter, and EVEN THERE speaker recovery is <= 11.8% (with a 3.3% TRAIN
self-store floor). Identity is not dying across long context ranges in this
corpus slice - it is ALREADY ABSENT at the answer:-boundary state. This
sharpens the R0-b conclusion: the banked C_rand family passed G1 separation
(0.1019) but separation does not imply decodability; the failure localizes to
the reservoir STATE REPRESENTATION (what the dynamics retain), not to the
decode chain (linear/nonlinear/memorized all fail alike) and not to
positional carry over chapter distances.

## Disclosures and deviations (all deliberate, none silent)

1. **Decoder advantage, disclosed:** P1/P2 heads operate on the 673-type TRAIN
   answer vocabulary (the memory route stores answer-token sets by
   construction), while the frozen comparators are full-vocab (11740 types).
   The gates were still failed - the negative is conservative.
2. **Ranked-consumption decoding rule** (pinned in `probes.py` docstring
   before scoring): one ranked candidate list per item from a single boundary
   state; position j consumes the highest-ranked unused candidate; scoring
   bookkeeping (wordization regex `[a-z0-9']+`, casefold, 639-position
   denominator) mirrors `tools/qa_baselines.py::answer_span_eval`.
3. **Two circuits per N** instead of one: `feature_state_vector()` exposes
   only the ACTIVE family, `traces` cannot enter combos, and the asm-rand
   codebook rewires afferents (different dynamics). Arm `explicit_spike`
   (`family="spike_counts"`) keeps exact ladder wiring while making both
   traces and spike-count banks host-visible per token.
4. **Byte-level tokenizer** (`max_phrases=0`, vocab 256) - keeps VAL encoding
   a pure byte-prefix of the full serialization (asserted per item) and avoids
   OOV paths; "ladder config" fidelity applies to the circuit config.
5. **Per-token features stride-sampled float16** away from decision points
   (stride 16 at N=1024, stride 1 at N=96) to keep the artifact at 292.5 MB;
   every decision point is captured at float32 in all three families.
   `growth_epochs=1`.
6. **P2 selection on VAL** is prescribed by the strategy text ("query VAL by
   thresholded overlap, sweep radius/top-k"); the full sweep table is
   committed so post-hoc optimism is auditable. Shuffled-key and
   majority-prior/random-ranking controls bound what a leak would look like.
7. **npz not committed** (binary, 292.5 MB); `logged_states_manifest.json`
   records shape/dtype/sha256 for all 52 arrays plus curriculum digests, so
   the artifact is reconstructible and verifiable locally.

## Files

- `src/humanbrain/recurrent.py` - `observation_sink` hook (opt-in callback
  fired after each token's feature-boundary update; instance-level, default
  None, zero behavior change when unset; not part of checkpoint schemas).
- `tests/test_recurrent_observation_sink.py` - 4 tests (default/validation,
  ordering, read-only bit-identity, traces-family path). Full recurrent test
  surface green (18/18 incl. these).
- `experiments/r0c_probes/log_states.py` - D1-2 harness (rebuilds the npz).
- `experiments/r0c_probes/probes.py` - P1/P2/P3 + D5 checkpoint.
- `experiments/r0c_probes/logged_states_manifest.json`,
  `experiments/r0c_probes/probes_results.json` - committed records.
- `experiments/r0c_probes/logged_states.npz` - local artifact (not committed).

## Bottom line for the roadmap

The mechanical D5 outcome is NEGATIVE-WITH-LOCALIZATION: reservoir-state keys
are dead for Bible-QA under linear, nonlinear, and SDM-style readout alike at
N=96 and N=1024. The localized next move sanctioned by this data: keep the
race-layer injection seam but derive keys from INPUT-side structure (the
banked assembly-overlap-random input-assembly space or explicit token-identity
features), and treat winner-feedback broadcast (runner-up thread) as the live
mainline alternative. Escalation of modern learned readouts (option d) is NOT
warranted by any number above (best 3.29% vs trigram 16.28%).
