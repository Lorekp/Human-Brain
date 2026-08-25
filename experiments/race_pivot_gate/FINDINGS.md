# race_pivot_gate — N=96 fixture leg: TERMINAL INELIGIBILITY (verbatim)

Date: 2026-08-24 · HEAD lineage: codex/a013-cognition-design @ 63c213b (+tooling)
Runner: `benchmarks/ladder_experiments.py --experiment race_pivot_gate --arm <A> --N 96 --seed 7 --check-determinism`
(one invocation per arm; `--check-determinism` enforced byte-identical duplicate
records modulo wall_s BEFORE each write). Aggregated by `tools/race_pivot_gate.py
--leg 96 --skip-missing` (eligibility recomputed from raw metrics).

## Terminal summary (verbatim)

```
P-S1: gates=FAIL distinct=6 exact=0 replay=42/42 cases=126/126 modal_share=0.1667 margin_median_pa=224.5972 stable_frac=0.4617 flip_rate=0.0000 wall_s=64.1
P-S2: gates=FAIL distinct=6 exact=0 replay=42/42 cases=126/126 modal_share=0.1667 margin_median_pa=184.3015 stable_frac=0.5227 flip_rate=0.0000 wall_s=50.8
P-S4: gates=FAIL distinct=6 exact=0 replay=42/42 cases=126/126 modal_share=0.1667 margin_median_pa=122.6936 stable_frac=0.4123 flip_rate=0.0000 wall_s=50.5
P-R4: gates=FAIL distinct=6 exact=0 replay=42/42 cases=126/126 modal_share=0.1667 margin_median_pa=205.0639 stable_frac=0.8610 flip_rate=0.5080 wall_s=48.8
P-M1: gates=FAIL distinct=6 exact=0 replay=42/42 cases=126/126 modal_share=0.1667 margin_median_pa=224.5972 stable_frac=0.4617 flip_rate=0.0000 selective=0.080808 coverage=1.000000 wall_s=49.2
LEG VERDICT: INELIGIBLE (gates fail)
```

Frozen thresholds: G-P1 distinct>=3 (PASS everywhere), G-P2 exact>=1 (FAIL
everywhere — zero exact matches in every arm), G-P3 replay>=39/42 (PASS
everywhere; replay leg pinned byte-identical to the recorded baseline anchor
`readout_compare/ridge_N1024_s7.json` config, 41/42 baseline — this run 42/42,
126/126 cases). P-M1 fixture diagnostics: selective_accuracy 0.080808 with
coverage 1.000000 (tau_low = -442.01 pA never fired an abstention; tau_high =
1093.41 pA); the frozen P-M1 success gate is N=1024-only and was not evaluated.
τ freeze blocks (pool_n=161 TRAIN decisions, hash-sealed pre-scoring) are in
each record's `preregistration.tau_block`.

## Consequences (protocol §5/§8, applied without negotiation)

- G-P2 failure at the fixture leg is TERMINAL for every arm: no arm proceeds to
  the N=1024 leg. The N=1024 cohort is NOT spent. Exit code 2 = valid terminal
  INELIGIBILITY — preserved, never retried silently.
- Readout: broadcast de-collapsed the answers (distinct 6/6 arms, modal_share
  0.167 — vs R0-b's degenerate priors) and capacity is intact (42/42), but the
  loop produced ZERO exactly-correct free-run completions: diverse-but-wrong,
  the deliberate G-P2 tightening doing its job. Settle dynamics differ by arm
  as designed (P-R4 flips 50.8% of positions under chase and stabilizes 86.1%;
  P-S family never flips by construction).
- Nothing was tuned, re-seeded, or re-executed after the first token.
