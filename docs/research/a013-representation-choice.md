# A013 representation choice

Status: DESIGN. Provenance: researched by subagent (9h; transcript-mined after
final-answer provider failure), authored by orchestrator from the recovered
research; verify pass REQUIRED before merge.

## The status quo, quantified (recovered computation)

Current encoding: each token owns a fixed random 16-of-N afferent subset.
Hypergeometric overlap of two tokens' subsets at N=128, k=16:

- distribution over overlap x∈{0..8}: mode at 2; **expected overlap = 2.0
  cells** (~12.5% of k);
- P(overlap ≥ 6) = 0.56%, P(overlap ≥ 8) ≈ 0.007%.

So unrelated-token codes are nearly orthogonal BY CONSTRUCTION — which also
means related tokens (tasks sharing structure) share almost nothing. Mean
afferent channels per reservoir cell: 34.75. Eligibility retention windows:
≥0.5 within ~11.1 token intervals; ≥0.1 within ~36.8 intervals — the useful
binding window is roughly a few dozen tokens around an answer.

This arithmetic explains the sealed transfer failure mechanically: with
expected cross-task feature overlap this low, readout weights trained on one
family find no shared presynaptic substrate on held-out families.

## Options assessment (summary)

| Representation | Binding window fit | Shared structure | Cost/token | Verdict |
|---|---|---|---|---|
| Byte-level UTF-8 | sequences too long for ~36-interval window | low unless n-gram hashed | lowest per symbol | reject as primary |
| BPE/unigram subword (4k–8k) | moderate | moderate (shared subwords) | medium | acceptable fallback |
| Small alphabet + positional redundancy | good | artificial | high redundancy | reject |
| Typed program tokens (status quo) | good, but explicit identity | **near-zero (2/16)** | low | current failure mode |
| Chunk/concept tokens | best window fit | highest if concepts shared | medium | aspirational; needs concept miner |
| Continuous embeddings → top-k projection | tunable k | **by construction (similar→overlap)** | low extra | **recommended** |

## Recommendation (ranked)

1. **Similarity-projected sparse codebook**: embed each token once (frozen,
   any small embedding model or deterministic hash-trick over characters),
   project to R^N_E, take top-k cells as the token's assembly. Overlap becomes
   monotone in embedding similarity — shared structure becomes physical
   overlap, exactly what local three-factor learning needs to transfer.
2. Random sparse codebook (same k) as the CONTROL arm — isolates "distributed"
   from "semantically structured".
3. Explicit identities retained only as the sealed baseline arm.

## Decisive experiment (preregistration-ready)

Arms: {explicit, random-distributed, similarity-projected} × N ∈ {128 bridge,
512} × seeds {7,8,9} on the harness (`benchmarks/ladder_experiments.py`),
readout = winner of the readout_compare follow-up (see pilot FINDINGS: local
rules need constant derivation first; ridge remains the capacity reference).

Hypothesis H-code: calibration-cases fraction (similarity − random) > 0 with
Wilson CI excluding zero; secondary: (random − explicit) > 0. If similarity ≯
random, semantic projection adds nothing beyond distribution — record and
pivot to concept-mined chunk tokens.

Red tests: codebook determinism under seed; collision bound test (max tokens
per assembly ≤ ceiling); frozen-projection-hash pinning; framing-order
preservation under re-encoding.
