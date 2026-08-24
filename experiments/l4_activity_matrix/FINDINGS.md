# L4 activity-matrix rerun — post-seam validation harness

**Date:** 2026-08-24
**Branch:** `codex/scale-opt-cpu` (worktree `human-brain-scale-opt-cpu`)
**Status:** harness built and dry-run against the Seam-A baseline (`fa9070e`, Seam B not yet
merged). Artifact: `l4_activity_matrix_seam-a-baseline.json` (baseline rows below).
Script: `../procedural_adapter_spike/l4_activity_matrix.py`.
**Purpose:** make the seam-landing-to-verdict latency zero for the CPU-EP upper-band
question. When Seam B (indexed staged membership, `docs/research/cpu-seams-spec.md`)
merges, one command re-runs the extended L3 activity matrix and §5's pre-registered
expectations are checked against measured rows — no re-derivation, no new harness work.

---

## 1. Question this answers on merge day

FINDINGS.md §3.3(b) mapped the regime boundary at N_cfg = 1e6: at ≥1% uniform activity,
both stepping modes are pinned under the shared injection-path wall (`apply_event`'s
linear `.position()` staged scan, O((f·A)²)/token), and EP/FT lands at 1.1× / 0.7× /
0.5× for 1% / 2% / 5%. Seam B removes that wall (O(f·A·log f·A)). What remains:

1. Where do EP and FT actually land at 1–5% once the wall is gone?
2. Does the new 20% row stay feasible, and who wins there?
3. Does event-proportional clear the 160 tok/s band anywhere at ≥1% *uniform*
   activity — or does that gate stay sparse-regime-only?
4. Does the Seam-A `read_cells` mirror-read stay negligible inside EP tokens
   (Phase-B host-mirror feasibility at these activity volumes)?

## 2. Harness protocol

Matrix: N_cfg = 1,000,000 × activity {1%, 2%, 5%, 20%} × stepping_mode
{full_touch, event_proportional}. Stress GLIF configuration, fanout 4, 100-step token
windows, raw C-ABI drive — L3-identical except where stated.

- **Per-config timing isolation:** every (mode × activity) cell runs in a FRESH runtime
  with its own chunked fixture build (fresh-gid ratchet in 2,048-wide advance windows),
  settles 400 steps, and only then enters the timed stream. Cells cannot contaminate
  each other's timings; build walls are recorded separately.
- **Fixture auto-sizing (new, documented):** only injected cells spike (downstream weight
  0.25 pA is subthreshold), so one token generates f·A edges delivering inside the next
  call, staging up to `cfg·(1−exp(−f·A/cfg))` fresh recipients on top of the fixture.
  Rows at 1–5% keep the exact L3 fixture (2^18); the 20% row's staged-peak model
  (~806k fresh + injected non-resident bound) exceeds what 2^18 leaves under the
  resident-capacity fail-stop at cfg=1e6, so its fixture auto-shrinks via the sizing
  rule to the largest power of two that fits: **2^17 = 131,072**. Within-row EP-vs-FT
  ratios remain apples-to-apples (identical schedules); cross-row absolute walls carry
  the fixture-size caveat recorded per row.
- **Circuit breaker:** a single token exceeding `MAX_TOKEN_SECONDS` (900 s) truncates
  that cell's stream (`truncated_by_breaker` in the row) instead of hanging the run —
  relevant only for pre-Seam-B runs, where the quadratic wall makes 20% expensive.
- **Mirror-read diagnostic:** after each cell's throughput stream, a few schedule tokens
  are re-driven; after each advance, the previous token's first K=16 targets are read via
  `read_cells` (the Phase-B mirror pattern: read what you just touched). Reports median
  read µs and overhead % of the token. Without Seam A the diagnostic records the
  snapshot-fallback reference instead of a per-token number.

Run commands:

```bash
# full merge-day rerun (12 tokens/cell, L3 protocol):
python experiments/procedural_adapter_spike/l4_activity_matrix.py --label post-seam-b

# quick smoke (any subset):
python experiments/procedural_adapter_spike/l4_activity_matrix.py --tokens 4 \
    --activities 1 --modes ep --label smoke
```

Artifacts land in `experiments/l4_activity_matrix/l4_activity_matrix_<label>.json`.

## 3. Seam detection (built, validated)

The harness probes the LOADED library before measuring and stamps every artifact:

- **Seam A `read_cells`:** `hasattr(lib, 'humanbrain_procedural_runtime_read_cells')`
  FFI probe + Python binding presence. Present → optimized targeted read path for the
  mirror diagnostic; absent → snapshot-fallback reference.
- **Seam B indexed staged membership:** internal Rust field, deliberately no ABI change,
  so detection is behavioral — a **single decisive-scale probe**. One advance() call
  delivers R=16 repeats for each of M=24,576 fresh gids ("scan-heavy": first delivery
  stages each gid; repeats pay the staged-membership lookup while |staged| grows to M
  inside the call); an identical-volume call into just-absorbed resident gids
  ("resident-reference") is reported as context. A quadratic `.position()` scan costs
  ~R·M²/4 comparisons (seconds at this scale); an index is O(M log M),
  marshal-dominated (tens of ms). Verdict: ABSOLUTE scan-arm median against a 400 ms
  floor. Source grep for `staged_index: BTreeMap<u64, u32>` / `.get(`/`.insert(`
  corroborates (bare `staged_index` false-positives on pre-existing local counters).
- **Provenance:** the native loader is source-addressed, so uncommitted `_native_src`
  edits change the loaded binary silently. Every artifact records HEAD, whether
  `procedural.rs` is dirty, and the attested source digest + binary hash.

Validated discrimination (final probe design, 2026-08-24):

| library | scan-arm median | reference (context) | verdict |
| --- | ---: | ---: | --- |
| fa9070e (Seam A, linear scan) | 7,428 ms | 6,087 ms | `linear_scan` → seam_a |
| working tree (uncommitted Seam B WIP) | 35 ms | 40 ms | `indexed` → seam_a+seam_b |

Margins around the 400 ms floor: >18× above (linear) and ~11× below (indexed).

Three probe-design failures were caught and fixed on the way (recorded so nobody re-steps
on them): (a) 100 pA probe current is NOT subthreshold under the stress GLIF — 163,842
stray spikes invalidated a first attempt (now 0.05 pA: worst-case prime+measure
accumulation 17×0.05 = 0.85 < threshold floor 1.0, plus a stray-spike guard); (b) native
validation rejects `edge_slot >= fanout`, so repeat deliveries must vary the SOURCE gid
for canonical-key uniqueness and keep slots at 0 (status-1 failures otherwise); (c)
ratio/growth-style classifiers are load-fragile on this desktop — subtracting or dividing
ms-scale terms flipped verdicts on an unchanged library across runs (1513× → 21× growth
on the same binary), because both arms pay unrelated resident-count-dependent costs on
the linear build (the documented quadratic-class public-API terms). The absolute-wall
classifier at a scale where the quadratic term dominates by orders of magnitude is
stable in both directions.

## 4. Baseline rows @ Seam A (`fa9070e`)

Dry-run profile: `--tokens 4 --max-token-seconds 300` (quadratic-wall containment on a
shared desktop running ~2.7x slower than the L3-day machine — see consistency anchors
below; the merge-day rerun uses the full 12 tokens). Fixture per §2 sizing. The 20%
rows truncated after their first over-breaker token; their medians are SINGLE-token
walls and lower bounds on steady-state cost.

| Activity | Fixture | EP ms/tok | EP tok/s | FT ms/tok | FT tok/s | EP/FT | EP cell-steps/tok | FT cell-steps/tok |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1% (10k ev/tok) | 262,144 | 991.9 | 1.008 | 1588.1 | 0.63 | 1.60 | 1,540,340 | 27,120,508 |
| 2% (20k ev/tok) | 262,144 | 3054.4 | 0.327 | 3584.2 | 0.279 | 1.17 | 2,994,809 | 27,980,444 |
| 5% (50k ev/tok) | 262,144 | 26122.2 | 0.038 | 15649.5 | 0.064 | 0.59 | 6,902,947 | 30,322,089 |
| 20% (200k ev/tok) | 131,072 | 373551.3† | 0.003 | 538099.0† | 0.002 | 1.50‡ | 12,897,602 | 25,195,644 |

† single over-breaker token (FT: 1071 s; EP: 729 s), stream truncated by design.
‡ single-token comparison, indicative only.

Mirror-read diagnostic (Seam A present, K=16 `read_cells`): 18.3–29.7 µs/token across
all non-truncated cells = **≤0.003 % of any token wall** — the Phase-B host mirror is
free even before Seam B.

Consistency anchors vs L3 (12 tokens, quieter machine): FT 1% 507→1588 ms (3.1x),
EP 1% 441→992 ms (2.2x), FT 5% 5970→15649 ms (2.6x), EP 5% 8981→26122 ms (2.9x).
Same ordering and regime shape; absolute levels carry the desktop-load factor. Note
EP/FT at 20% flips positive (1.5x) even pre-Seam-B on single tokens — FT's full-touch
stepping floor stops helping once residency is small relative to traffic.

## 5. Pre-registered expectations for the post-Seam-B rerun

Written BEFORE the Seam-B merge, from the spec §B.4 cost model, the §B.8(B-G2)
acceptance bounds, and the L3 decomposition — so the merge-day comparison is
hypothesis-testing, not curve-fitting.

**E1 — Both modes accelerate; the wall is mode-independent.** Injection cost drops from
O((f·A)²) to O(f·A·log f·A) for FT and EP alike. Spec acceptance bounds (conservative
ceilings): EP ≤ 3.0 s/token and FT ≤ 1.7 s/token at 5%/fanout-4 (vs 9.01/4.99 measured).
Prediction: 5% rows land WELL inside those bounds; 2% improves similarly; 1% improves
least in absolute terms (it was least wall-bound).

**E2 — FT flattens toward its stepping floor.** Full-touch must step every resident
every step regardless of activity: ~2^18 residents × 100 steps ≈ 26M cell-steps/token ≈
0.5 s at L3's implied ~19 ns/cell-step. Post-Seam-B, FT ms/token should compress toward
a roughly activity-independent floor (~500–900 ms) + linear-ish queue marshalling of A
events, i.e. FT stops growing steeply between 1% and 5%.

**E3 — EP wins everywhere ≥1%, and the margin GROWS with activity.** With the shared
wall gone, the residual difference is stepping economics (EP touched 3.6–14.5× less
kernel work at every L3 activity level) minus dormancy-replay costs (bounded here:
alternating-half pools keep replay gaps ≤ ~2 windows). Prediction: EP/FT goes from
1.1/0.7/0.5 (L3) to > 1 at all of 1–5%, increasing with activity; at 20% EP/FT ≥ 1
(first measurement ever at that level).

**E4 — The 160 tok/s band does NOT clear at ≥1% uniform activity — host-marshal-bound.**
Pre-registering the negative result with its mechanism: Python-side event construction
costs ~0.96 µs/event (L1 caveat). At 1% that is ~9.6 ms/token of pure host marshal —
already above the entire 6.25 ms/token budget of 160 tok/s, before any kernel work.
Predicted EP medians post-Seam-B: 1% ≈ 30–150 ms/token (≈ 7–35 tok/s), 20% ≈ 0.3–1 s/token.
The 160+ gate therefore remains reachable ONLY in the sparse text-like regime
(FINDINGS §3.3(a): 363 tok/s at 640 events/token), and the operator decision rule
(§3.3(d)) survives Seam B unchanged in form. Falsification condition: EP median < 6.25 ms
at ANY ≥1% row would overturn E4 and reopen the band question.

**E5 — Mirror reads stay free.** `read_cells` K=16 measures ~37 µs end-to-end at R=1e6
(acceptance A-G1). Against post-Seam-B token walls of ≥30 ms, overhead stays < 0.2%
at every activity level. Phase-B host mirroring is not the binding constraint.

**E6 — 20% row feasibility.** Post-Seam-B the 20% cell completes without the breaker;
pre-Seam-B it may truncate or take many minutes per token (quadratic wall at ~632k
staged/token) — expected and documented, not a harness failure.

## 6. Merge-day procedure

1. Land Seam B; rebuild/pull; confirm `git status` clean for `_native_src`.
2. `python experiments/procedural_adapter_spike/l4_activity_matrix.py --label post-seam-b`
   (12 tokens/cell; add `--activities 1,2,5,20` implicitly default).
3. Check the artifact's `seam_detection.seam_level == "seam_a+seam_b"` and that
   `source_corroboration.symbol_present` is true — if the probe says `linear_scan`,
   STOP: the loaded binary predates the merge (stale cache/build).
4. Fill §4's table column-by-column next to the baseline; check E1–E6 explicitly,
   especially E4's falsification condition.
5. Record the verdict paragraph in FINDINGS §3.3(b) follow-up + update
   `docs/benchmark-status.md`; cite both JSON artifacts (baseline + post-seam-b).

## 7. Deviations & caveats

- Baseline dry-run used `--tokens 4` (vs L3's 12) for wall-clock containment under the
  quadratic wall; merge-day rerun uses 12. Medians over 4 tokens carry wider error;
  consistency anchors vs L3's 12-token rows are within expected desktop-load variance.
- 20%-row fixture is 2^16, not the L3 2^18 (capacity math in §2). Cross-row absolute
  comparisons at 20% vs lower rows are NOT fixture-controlled; the row exists for the
  EP/FT ratio and the feasibility answer.
- Shared-desktop ambient load: acceptance history shows ±40% run-to-run spread on
  µs-scale seams; ms-scale token medians are far steadier, but treat < ±10% deltas
  between single runs as noise.
- The dry-run executed from a detached temp worktree at fa9070e (clean Seam-A source)
  because the shared worktree carried uncommitted Seam B WIP that the source-addressed
  loader would have built silently. Provenance fields in the artifact pin this.
