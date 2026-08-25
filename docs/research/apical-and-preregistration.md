# APICAL-AND week-1 preregistration (Direction B) — FROZEN 2026-08-24

Status: PREREGISTERED before implementation. This file is committed BEFORE any
TwoCompartmentGLIF5 code exists in the tree (D1 discipline; race-pivot-gate
precedent). Base commit: `31bb830` on `codex/a013-cognition-design`.
Direction source: `custom-architecture-directions.md` §Direction B @ `31bb830`.
Prior terminal context: `race-pivot-gate-verdict.md` (reservoir family
falsified at small N under preregistered gates).

Everything below (equations, grid values, gates, thresholds, seeds, scoring,
verdict rules, stop rules) is frozen now. Post-freeze changes require a v2
document; silent deviation is fabrication.

## 1. Model frozen: TwoCompartmentGLIF5

Subclass of `AdaptiveGlifPopulation` (`src/humanbrain/cells.py`), CPU path only
(pure NumPy step; never touches the native/CUDA kernels). Soma equations are a
bit-exact replication of `glif_step_collect`
(`src/humanbrain/_native_src/src/lib.rs`) over the fitted Allen GLIF5 constants
— verified by the flag-off parity tests below.

Apical compartment (new state per cell i):

- `v_a[i] += (dt/tau_a_ms) * (u_a[i] - v_a[i])` every fitted 0.05 ms step;
  `v_a` starts at 0 (relative units), `u_a` is the apical drive vector.
- Plateau rule: when `v_a[i] > theta_p` AND `gate_remaining[i] == 0`, open the
  gate: `gate_remaining[i] = round(gate_duration_ms / dt)` (= 2000 steps at
  100 ms). While `v_a` stays above `theta_p`, an expired gate re-opens on the
  same test (sustained depolarization sustains plateaus). No refractory on the
  apical compartment beyond this rule.
- Multiplicative soma-threshold gate: while `gate_remaining[i] > 0`, the soma
  crossing test uses `total_threshold * (1 - g_gate)` instead of
  `total_threshold`; each executed step decrements `gate_remaining`. Nothing
  else about the soma changes (no current injection from the gate).
- Flag-off parity: with `u_a == 0` everywhere, `v_a` stays identically 0, no
  gate ever opens, and every somatic array update must be BIT-IDENTICAL to the
  native `AdaptiveGlifPopulation.step()` / native token loop. RED TEST FIRST:
  parity tests are written and observed failing (class absent) before the
  implementation lands.
- Two-compartment state (`v_a`, `gate_remaining`) is experiment-scoped; it is
  NOT part of any checkpoint schema.

## 2. Frozen arm grid (18 arms)

| dimension | values |
|---|---|
| tau_a | {50, 100, 200} ms |
| theta_p | {0.25, 0.50, 0.75} normalized drive units |
| g_gate | {0.30, 0.60} |
| gate_duration_ms | 100 (fixed all arms) |

Arm id `A<tau>_<theta>_<g>` (e.g. A100_0.50_0.60). All 18 arms are simulated;
every arm is scored against every gate; no arm selection occurs.

## 3. Context drive (apical input), frozen

U-STATE coordination: D5 checks `experiments/ustate_probes/FINDINGS.md`. If
that agent's D3 passed (union states pass G-U1 separation), the apical drive
uses the U-STATE union vector through the SAME projection/scaling below
(replacing the byte histogram as the summary source). Otherwise — and that is
the operative branch today, no FINDINGS file existing at freeze time — the

**fallback context summary** applies:

- `h_t` = L1-normalized cumulative histogram of prompt bytes observed so far
  (causal; updated once per token from the serialized prompt text only — never
  answer bytes).
- `u_raw_t = W_ctx @ h_t`, `W_ctx` shape (cells x 256), entries iid N(0,1)/16,
  seed 20260824, generated once per population size.
- `u_a(t) = zscore(u_raw_t) * A_ctx` with `A_ctx = 1.0` (z-score across cells).
- After the decision point ('A:' inclusive), `u_a` freezes at its decision-point
  value while answer tokens stream somatically.

## 4. Gates (verbatim from the task brief, operationalized)

- **G-A1 conjunction index**: somatic response to context+token >= 1.5x
  max(single inputs) on held-out pairs, noise-controlled. Operationalization:
  per VAL item i, context drive from its full prompt; token burst = byte token
  ids of its answer span (<= 32 tokens); protocol: reset -> apical-only phase
  300 ms -> burst window 50 ms (ladder pulse config: 5000 pA, 1.5 ms pulses,
  5 ms interval) -> 100 ms tail; response R = total somatic spikes in
  [burst start, burst start + 150 ms]. Conditions CT (drive + burst),
  C (drive, empty burst), T (no drive, same burst).
  Noise control: every condition repeated with 8 iid noise seeds (9001..9008),
  per-cell per-substep soma current noise ~ N(0, 50 pA); CI_pair =
  mean_noise(R_CT) / max(mean_noise(R_C), mean_noise(R_T)).
  Arm CI = MEDIAN of CI_pair over all 60 VAL items.
  **PASS iff arm CI >= 1.5.** Diagnostics (non-gating): fraction of pairs with
  CI > 1; mismatched-context selectivity meanNoise R(c_i,b_i)/R(c_j,b_i).
- **G-A2 separation >= 0.10 at decision points via R0-G1 definition**:
  replay VAL teacher-forced; decision point = after observing the serialized
  prompt through 'A:' inclusive (r0-c rule); snapshot the eligibility-trace
  bank (f32, bias excluded); pooled MEDIAN 1 - cosine over unordered VAL pairs
  within the same QA family stratum. **PASS iff median > 0.10**
  (= median cosine < 0.90, identical threshold to R0/R0-b G1).
- **G-A3 decode z >= 2 vs unigram**: span accuracy of decoded answers using the
  r0-c probes.py machinery VERBATIM (logistic + 2-layer MLP, head restricted to
  the 673-type TRAIN answer vocabulary, soft answer-bag targets,
  ranked-consumption decoding, wordized 639-position denominator mirroring
  tools/qa_baselines.py). Feature sets: trace bank and raw voltage||spike
  counts at decision points (TRAIN trains decoders, VAL scores).
  **PASS iff best decoder span-acc >= 0.061851478566709396**
  (= p0 + 2 sigma with p0 = unigram_smoothed 29/639, sigma = 0.008234; the
  exact two-sigma threshold computed in r0c FINDINGS and reused unchanged).
  Controls recomputed identically: majority-prior-within-answer-vocab
  (0.0406885758998435 expected), random ranking (seed 11), shuffled-label MLP.
  Frozen comparators (qa_baselines_bible_v1): unigram 0.04538341158059468,
  trigram 0.162754303599374.

## 5. Verdict rules (fixed now)

- Per-arm PASS iff G-A1 AND G-A2 AND G-A3.
- Program verdict **PASS** iff at least one of the 18 arms passes all three
  gates (family-wise multiplicity over 18 preregistered arms disclosed rather
  than corrected; the grid itself is the preregistered experiment).
- Otherwise **NEGATIVE-FALSIFIED**, reported with per-arm numbers verbatim.
- Pre-declared falsifier: any arm passing G-A1 (AND-index) whose best decode is
  BELOW the majority-prior-within-vocab control (0.0406885758998435) means
  "conjunctive features do not recover QA identity" — the apical route dies and
  effort redirects to explicit-store hybrids, regardless of other arms.
- Infrastructure failure (crash, parity failure, corpus mismatch): fix infra,
  re-run the fixture leg; a partially-scored cohort is never rescored or
  partially reported as a verdict. Stop-on-failure: if G-A1 fails in ALL arms
  the verdict is NEGATIVE-FALSIFIED at G-A1 and no further legs are owed.

## 6. Execution order

1. Fixture leg (N=96 population, first 3 VAL items): validates wiring copy,
   parity harness, logging, and probe plumbing end to end. Fixture numbers are
   diagnostics only and gate nothing except infrastructure readiness.
2. Parity gate: flag-off numpy token loop vs native `RecurrentCellCircuit`
   observe() must match bit-for-bit (all reservoir arrays after >= 200 tokens of
   real corpus text) before ANY flag-on run counts.
3. Cohort spent exactly once: G-A1 conjunction battery (18 arms, N=1024) then
   full TRAIN+VAL decision-point replay (18 arms batched, N=1024), then offline
   probes. All results committed, including negatives.

## 7. Frozen seeds and constants

- Circuit/wiring seed 7 (identical construction to the r0-c explicit_spike
  N=1024 arm: input_fanout=16, recurrent_fanout=8, inhibitory 0.2, ridge rule;
  wiring copied from an actual RecurrentCellCircuit instance so RNG consumption
  order is bit-faithful).
- W_ctx seed 20260824; noise seeds 9001..9008; random-ranking control seed 11;
  tokenizer BytePhraseTokenizer(max_phrases=0, min_frequency=3); growth_epochs
  irrelevant (no readout training here; dynamics do not consume readout state).
- Corpus: FREEZE-CANDIDATE-v3-BIBLE manifests under
  `$LOCALAPPDATA/Temp/hb_bible_freeze/evidence` (TRAIN 200 / VAL 60; digests
  recorded in r0c FINDINGS).

## 8. Artifacts

`experiments/apical_probes/`: `conjunction_results.json`,
`logged_states_apical_manifest.json` (+ local npz, not committed),
`probes_results.json`, `FINDINGS.md` (numbers verbatim), harness scripts.
Source: `src/humanbrain/cells.py` subclass + `tests/test_two_compartment_glif5.py`.
