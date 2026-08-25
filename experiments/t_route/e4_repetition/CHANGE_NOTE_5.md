# Change Note 5 — E4 repetition-regime probe (T-ROUTE × Assembly-Calculus mechanisms)

**Status: FROZEN BEFORE EXECUTION.** This note is committed before `run_e4.py` is ever
run. Nothing below may change after the first successful execution; deviations found
during mechanics validation get amended here *pre-spend* (same discipline as Change
Notes 1–3). Parent docs:
`docs/research/path-decision-synthesis.md` @ e16d22c §Path 3 + execution-order step 3,
and `docs/research/deep-dive-assembly-calculus.md` §Cheap falsification (quoted there
verbatim; gates restated in §7 below).

## 1. Question

Does multiplicative `*=(1+β)` potentiation plus k-cap winner selection **active during
encoding** recover query-dependent routing in the repetition regime, where E2's one-shot
additive accumulation demonstrably did not (verdict
`RECOVERY_FAILS_SPARSITY_INSUFFICIENT`: high absolute overlap, static ~query-independent
winners)?

Falsifier branches (verbatim intent from synthesis §Path 3):
1. Prediction: enrichment recovers **only on the repeated-pair stratum** and collapses on
   single-exposure items ⇒ pivot viable exclusively inside repetition-regime task shapes.
2. If even the repeated stratum fails to enrich ⇒ multiplicative+k-cap adds nothing over
   E2; the assembly-flavored pivot closes entirely.
3. If single-exposure items route query-dependently ⇒ the E3 conclusion is undermined;
   NEMO deserves an installation spike.

## 2. Data (pinned)

`experiments/t_route/synthetic_episodes/episodes_train.jsonl` — 1,656 TRAIN-side
synthetic who-said episodes (49 train books; frozen QA addresses excluded).

- LF-canonical sha256: `a94a8c22cedace1b533351b3a811829b68f62a374daaceff683b052b1b359e5a`
  (equals the synthesizer manifest pin `scopes.train.sha256`). The working tree checks
  out CRLF (`fb759b86d7c796967e4bdb74558f835ffc92af3762add434ba68d4f264b6b293` raw) via
  git autocrlf; the harness verifies BOTH digests (raw bytes and `\r\n`→`\n`
  normalized) and STOPs on mismatch of either.
- Pair key (exact strings): `(meta.quote, answer)`. Measured census (frozen):
  **1 pair with ≥10 repetitions** — `('the God of Israel', 'the LORD')`, 27 episodes at
  27 distinct addresses (all `thus_saith` tier); 1,593 singleton pairs; 17 mid-band pairs
  (2–9 reps, 36 episodes). This degeneracy is itself a measured property of the task
  universe (natural KJV attribution has essentially no verbatim quote/speaker repetition
  outside liturgical formulas) and is reported as such.

### Strata (frozen)

- **REPEATED**: episodes whose pair count ≥ 10 → n = 27.
- **SINGLE**: episodes whose pair count == 1 → n = 1,593.
- **MID** (2–9 reps): n = 36 — encoded in the stream but gate-ineligible; descriptive
  output only.

Interpretation note (deviation from parent-doc letter, frozen here): the synthesis says
"single-exposure VAL items". The natural-corpus VAL cohort cannot be scored here because
the E1/E2 freeze directory and logged npz artifacts do not exist in this worktree, so
gold/matched arrays for those items are unreconstructable at their pinned digests.
"Single-exposure VAL" is therefore operationalized as the SINGLE stratum above: items
experienced exactly once by the encoder inside the same pinned distribution. Both
strata are scored on encoded material — mirroring E1/E2, whose train-split items were
likewise encoded-and-evaluated, and matching the assembly-calculus claim under test
(pattern completion of stored bindings), which is about retrieval of what was learned,
not generalization to unseen pairings. The shuffled-question and randomized-W controls
carry the "is it routing or a static pool?" burden exactly as in G-T1.

## 3. Substrate & encoding

E2 configuration throughout (`experiments/t_route/e2/run_e2.py` constants):

| constant | value | meaning |
|---|---|---|
| N_BITS / W_ONES / PROJECTOR_SEED | 2048 / 32 / 7 | SDRProjector |
| FAN_IN / W_SEED | 32 / 7 | SparseProjection support+init U(0.5,1.5) |
| K_WINNERS (= k-cap K, encode AND retrieve) | 128 | kwinner_indices |
| H_WIN / O_MAX | 12 / 8 | pairing window / anchor cap |
| RAND_W_SEED | 13 | randomized-W control |
| SHUFFLED_Q_SEED | 11 | question permutation π |

Tokenizer: `BytePhraseTokenizer.fit(serialized_prompt_answer of all 1,656 episodes,
max_phrases=0, min_frequency=3)`; assert vocab_size == 256. (Pure function of pinned
bytes; the E2 recipe applied to this corpus since the original fit corpus is unpinned
here.)

**Encoding stream:** all 1,656 episodes in file order = 1,656 discrete rounds. Learning
is the running dynamics (DD-ASP): each round,

1. `q_pop = population_union(question tokens)`; passage tokens from `context`.
2. Event stream per episode = E1/E2 pairing recipe restricted to THIS episode:
   for each unique question token tq, anchors = first O_MAX occurrences in the passage,
   every passage token within ±H_WIN of an anchor contributes token-pair (tq, tp) with
   within-episode multiplicity m.
3. **k-cap during encoding**: `drive = W_current @ q_pop` over all 2,048 post bits;
   `winners = kwinner_indices(drive, 128)` (whole-area competition).
4. Potentiation: expand events to bit pairs (a ∈ sdr(tq) × b ∈ sdr(tp)); keep only
   b ∈ winners with (a,b) present on the fixed fan-in support; per (b,a) aggregate
   multiplicity m_round and apply the arm's update.

Arm updates on synapse values:

- **mult_b{0.05,0.25,0.5}**: `w[b,a] *= (1+β)^m` implemented as integer fire-count
  matrix `C[b,a] += m`; materialized values `init·(1+β)^min(C, CAP_EXP(β))`.
- **add_gated**: `w[b,a] += m` (same winner gating — isolates rule vs gating).
- **add_free**: E2-style batch additive accumulation over the same 1,656 episodes, no
  gating (isolates mechanism vs data regime — "did repetition alone do it?").

Controls routed like arms but never trained: `shufq_<arm>` (π =
`default_rng(11).permutation(1656)`, item i routed with q_pop[π[i]]), `randomized_W`
(untrained seed-13 projection).

No homeostat, no normalization, no decay, no support growth (fixed fan-in-32 topology is
part of the E2 config; connectivity growth is explicitly out of scope for this probe).

### Numerical-safety clamp (frozen pre-run)

Unbounded `(1+β)^t` can overflow float64 (wor case exponent ≤ 200×1656 ≈ 3.3e5 fires).
Freeze `CAP_EXP(β) = floor(log_{1+β}(1e250 / 1.5))`: **β=0.05 → 11790, β=0.25 → 2577,
β=0.5 → 1418** (max weight ≤ 1e250, far above any discriminative scale; ordering within
a row preserved). This is float64 arithmetic protection, NOT a plasticity rule (NEMO has
no cap); clip counts are reported. β=1.0 was considered and excluded pre-run: growth
`2^t` provably exceeds float64 range within the stream (2^1656 > 1e308) even ignoring
within-episode multiplicities.

## 4. Gold regions & eligibility (E1 recipe verbatim)

Answer-token sequence matched contiguously in passage tokens; each match region extended
by ADJ=2 on both sides; gold bits = union of sdr indices of region tokens;
`matched = ≥1 region`. Eligibility = matched ∧ stratum membership.

## 5. Scoring (score_e4.py reads the log only)

Per arm × stratum on eligible items: overlap = |route winners ∩ gold bits|;
`base_med`, `shuf_med`, `rand_med` medians; enrichment ratios vs shuf/rand. Descriptive
(not gated): gold-size headroom, mean pairwise cross-question winner Jaccard within
stratum (≤200-item sample), MID-band table, max |weight| and clip counts per arm.

## 6. Gates (frozen)

Per arm × stratum, evaluated only if eligible n ≥ 20:

```
PASS(stratum) := base_med > 0 AND base_med >= 2*shuf_med AND base_med >= 1.5*rand_med
```

(the G-T1 conjuncts applied verbatim per stratum; both controls must be beaten).

## 7. Verdict logic (mechanical, frozen)

Let R = any arm PASSes REPEATED; S = any arm PASSes SINGLE.

| R | S | verdict |
|---|---|---|
| no | any | `MULT_KCAP_ADDS_NOTHING_OVER_E2__ASSEMBLY_PIVOT_CLOSES` |
| yes | no | `REPETITION_REGIME_CONFIRMED__PIVOT_VIABLE_ONLY_THERE` |
| yes | yes | `SINGLE_EXPOSURE_ROUTES_QUERY_DEPENDENTLY__E3_UNDERMINED_NEMO_INSTALLATION_SPIKE` |
| no | yes | same E3-undermined verdict, flagged inconsistent (repeated failed where single passed) |

Branch 2 of §1 maps to row 1; branch 1 to row 2; branch 3 to rows 3–4.

## 8. One-shot discipline

`run_e4.py` refuses to overwrite existing outputs; npz stays local (array-level sha256
in `e4_manifest.json`); scorer reads logs only; no tuning hooks; deterministic seeds
throughout. Committed files:
`CHANGE_NOTE_5.md, run_e4.py, score_e4.py, toy_check.py, e4_manifest.json,
e4_results.json, FINDINGS.md`.

Pre-spend mechanics validation (`toy_check.py`, throwaway temp files only): unit math of
the count/clamp machinery; gated-additive ≡ batch-additive when every candidate post bit
wins; end-to-end run+score on synthetic full-size logs reproduces hand-computed
overlaps and both verdict scenarios.
