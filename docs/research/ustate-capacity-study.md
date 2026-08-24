# U-STATE capacity study — empirical validation of the w^2/n union-capacity claims

Status: measured 2026-08-24, BEFORE any D3 probe compute was spent on scoring
(the §2-declared capacity precondition study, extended to the full lambda x n x w
grid). Branch `codex/a013-cognition-design`. Script:
`experiments/ustate_capacity/run_study.py`; raw data beside it
(`ustate_capacity_results.json`, `overlap_vs_length_curves.csv`). Runtime 16 s,
pure numpy, CPU-only, bit-identical across reruns.

## What was measured

Grid: lambda in {0.90, 0.97, 0.99} x {(n=2048, w=32), (n=4096, w=64)}, seed 7.
Projection imported from `src/humanbrain/sdr_union.py` (module sha256
`1de7f6edd914...`, working tree at measurement time; its only uncommitted delta
vs D2 is a semantically-neutral `UnionEncoder.encode` caching refactor — this
study calls `SDRProjector` directly).

Corpus: frozen FREEZE-CANDIDATE-v3-BIBLE evidence dir (`pg10.trimmed.bin` +
`pg10_book_spans_v3.json`, freeze-json sha256
`361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c` as pinned in
the preregistration). TRAIN-role books only. All 25,266 TRAIN verses parsed with
the builder's own inline-marker rule (asserted per-book counts); 200 sampled
without replacement via `default_rng(7)`; byte ids 0..255 (the section 1
tokenizer with max_phrases=0).

Item statistics: verse byte-lengths mean 129.6 / median 119 / range 29-441.
Two series per config: verse-prefix unions at checkpoints L (only verses with
length >= L contribute; n_items recorded per point) and full-verse boundary
unions (every verse at its own length — the direct G-U1 analog); plus 200
disjoint contiguous 800-token windows of the canonical TRAIN growth stream.
Active-bit conventions follow the D1 precondition: any-touch (> 0) and
effective (>= 1e-2 floor).

## 1. The analytic w^2/n token-overlap claim: VALIDATED EXACTLY

All 32,640 byte-token pairs per dimension:

| dims | analytic w^2/n | measured mean | measured median | p95 | max |
|---|---|---|---|---|---|
| n=2048, w=32 | 0.5 bit | 0.49933 | 0.0 | 2 | 5 |
| n=4096, w=64 | 1.0 bit | 0.99252 | 1.0 | 3 | 7 |

Mean matches the hypergeometric expectation to <1%; the distribution is heavily
right-skewed (most pairs share zero bits, a tail shares several) — the mean, not
the median, is the analytic quantity.

The companion fresh-support claim `n(1-(1-w/n)^L)` does NOT transfer to real
text streams: an 800-token KJV window contains only ~48 distinct byte tokens,
so union support saturates at alphabet coverage |U SDR(v)| = 1080/2048 bits
(measured any-touch = covered bits exactly), not at the idealized 1960@L=200.
Real capacity for distinctness is bounded by occurring-alphabet coverage
~ n(1-(1-w/n)^V), V ~= 50-80 for KJV text, well before length effects.

## 2. Overlap-vs-length curves (pooled median cross-item cosine)

Full curves: `overlap_vs_length_curves.csv`. Shape at every lambda: cosine
rises over the first ~50-100 tokens and then PLATEAUS — it does not grow with
length. The plateau level is set by lambda's recency weighting of the shared
universal bits (space, 'e', ...), exactly the mass D1 identified.

Verse prefixes, median cosine (effective active bits):

| config | L=16 | L=32 | L=64 | L=128 | boundary |
|---|---|---|---|---|---|
| n2048/w32 lam .90 | 0.578 | 0.612 | 0.621 | 0.611 | 0.603 |
| n2048/w32 lam .97 | 0.612 | 0.729 | 0.790 | 0.802 | 0.786 |
| n2048/w32 lam .99 | 0.603 | 0.727 | 0.813 | 0.867 | 0.850 |
| n4096/w64 lam .90 | 0.582 | 0.612 | 0.622 | 0.610 | 0.601 |
| n4096/w64 lam .97 | 0.614 | 0.728 | 0.791 | 0.802 | 0.785 |
| n4096/w64 lam .99 | 0.606 | 0.727 | 0.815 | 0.868 | 0.848 |

Boundary detail (full-verse states, 19,900 pairs each): max cross-item cosine
0.999 (lam .90) / 0.981 (.97) / 0.979 (.99) at n2048; share of pairs above 0.90
cosine: 0.40% / 2.43% / 13.28%. The collapse tail is present at every lambda
and grows steeply with lambda.

800-token windows, median cosine (share of pairs > 0.90):

| position | lam .90 | lam .97 | lam .99 |
|---|---|---|---|
| 100 | 0.570 (0.1%) | 0.770 (0.4%) | 0.831 (3.5%) |
| 400 | 0.582 (0.1%) | 0.780 (0.7%) | 0.874 (19.5%) |
| 800 | 0.582 (0.1%) | 0.780 (0.5%) | 0.871 (16.6%) |

n4096/w64 tracks n2048/w32 within 0.003 median cosine everywhere — the two
dimension settings are separation-equivalent because they share p = w/n = 1/64;
the larger setting doubles absolute collision mass (w^2/n: 1.0 vs 0.5 bit),
state size, and compute for no measured separation gain.

## 3. Answers to the three deliverable questions

**(a) Config maximizing distinctness at 800-token windows subject to >=1%
retained sparsity:** all six configs retain >= 51.7% untouched-bit fraction
under the 1e-2 floor at window end (effective support 550-1976 bits of
2048/4096), so sparsity is non-binding everywhere; the mechanical winner is
`n4096/w64, lambda=0.90` (median cosine 0.5796), with `n2048/w32, lambda=0.90`
statistically indistinguishable (0.5825). Lambda is the only knob that moves
distinctness; dimension choice does not.

**(b) Does ANY config keep cross-item cosine <= 0.90 at realistic verse
lengths (G-U1 precondition)?** YES — all six pass the pooled-MEDIAN criterion
on full-verse boundary states: median cosine 0.603 / 0.786 / 0.850 (n2048) and
0.601 / 0.785 / 0.848 (n4096) for lambda = .90 / .97 / .99. Margins shrink
monotonically with lambda; at lambda=0.99 the margin is thin (0.85 vs the 0.90
gate) and 13.3% of verse pairs plus 16.6% of window-end pairs exceed 0.90, so
that arm survives G-U1 only because the gate keys on the median at verse-scale
prompts. No arm would survive a long-context (800-token) version of the gate at
lambda=0.99.

**(c) Recommendation for the preregistered default:** proceed with the frozen
sweep at n=2048, w=32, seed 7 — it passes G-U1 at realistic lengths at every
lambda and nothing in the grid justifies switching dimensions. Expected G-U1
margin ordering if the D3 reproduction needs a tiebreak: lambda=0.90 >> 0.97 >
0.99, with 0.99 the fragile arm (watch its tail stats). Note for the §6 gate-0
reproduction check: these verse-based numbers are intentionally NOT comparable
one-to-one with the §2 precondition table, which used QA serialized prompts;
the "Q: ... A:" framing boilerplate adds large shared universal mass (D1: pooled
median 1-cosine ~0.18 even at lambda=.90 vs 0.397 for bare verses here). Same
projection, different item class — a >2x discrepancy there is expected, not an
implementation bug.

## Reproduction

```
python experiments/ustate_capacity/run_study.py
```

Reads `$LOCALAPPDATA/Temp/hb_bible_freeze/evidence/{pg10.trimmed.bin,
pg10_book_spans_v3.json}` and the freeze JSON; writes
`ustate_capacity_results.json` + `overlap_vs_length_curves.csv` beside the
script. Determinism verified by rerun (identical floats).
