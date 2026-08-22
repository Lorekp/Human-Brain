# Gemma-4-12B comparison protocol design

Status: DESIGN — no bench runs, no serving changes. This document fixes the
comparison target BEFORE the cell system can tune toward it.
Provenance: researched by subagent (licenses verified from primary sources,
Wilson CIs computed), written by orchestrator from the agent's transcript
after an iteration-limit truncation. Requires independent verification pass
before any merge to master.

## Discovered prior art in this repo (verified during research)

- `benchmarks/mini_swe_gemma.yaml` already exists: Gemma-4-12B-it-QAT-Q4_0
  served via llama.cpp on `http://127.0.0.1:8081/v1` through litellm/
  mini-swe-agent; temperature 1.0, top_p 0.95, top_k 64, seed 42,
  max_tokens 4096.
- A companion `benchmarks/mini_swe_cell.yaml` defines the cell system's
  adapter under model name `human-cell-dendritic`.
- `docs/benchmark-status.md` records a Gemma QAT Q4_0 GGUF identity hash and
  at least one executed SWE trajectory artifact (`sqlfluff__sqlfluff-1625`).
=> The harness is not hypothetical: it exists and has run once. The
comparison protocol formalizes it rather than builds it.

## Candidate bench assessment (license + execution facts)

| Bench | License (verified) | n tasks | Contamination risk | Host executability | Grading objectivity | Low-end sensitivity |
|---|---|---|---|---|---|---|
| SWE-bench Verified | MIT (princeton-nlp/SWE-bench) | 500 | HIGH — in every major LLM's training corpus | Windows-hostile (docker-centric); WSL2 required | high (human-validated patches, fail-to-pass tests) | poor below ~5% |
| SWE-bench Lite | MIT | 300 | HIGH | same docker issue | medium-high | poor |
| HumanEval+ (EvalPlus) | MIT code / Apache-2.0 harness | 164 | EXTREME — canonical, memorized | excellent (pure python sandbox) | very high (exec tests, plus extra tests) | good (fine-grained pass@k) |
| MBPP+ (EvalPlus) | CC-BY-4.0 data | 378 | HIGH but weaker than HumanEval | excellent | very high | good |
| Self-hosted simple-SWE suite | project-internal | parameterizable | ZERO (fresh cohort, family-disjoint) | native | high (oracle-checked hidden cases) | best (cohort sized to resolution need) |

## Recommendation

- **Primary: self-hosted simple-SWE suite** (`simple_swe` machinery,
  preregistered cohorts). Only contamination-free option; only option whose
  difficulty can be tuned to resolve differences between a nascent cell
  system and a 12B LLM; already integrated with the cell completion adapter.
- **Secondary: HumanEval+ (EvalPlus)** — external anchor with objective exec
  grading, trivial host executability, and published Gemma-class reference
  numbers. Contamination is acknowledged: this measures the cell system
  against a benchmark Gemma has memorized, so it is reported as
  "unseen-generator performance", never used alone for the win/lose claim.
- **Tertiary/reference: SWE-bench Verified** for the eventual headline claim,
  run inside WSL2, ONLY when the cell system clears the secondary rung;
  contamination disclosed symmetrically (Gemma trained on it, cell system
  did not — which makes a cell win there STRONGER, not weaker).

## Wilson 95% CI half-widths (computed, percentage points)

| Bench (n) | p=1% | p=5% | p=10% | p=20% | p=40% |
|---|---|---|---|---|---|
| Verified (500) | ±0.9 | ±1.9 | ±2.6 | ±3.5 | ±4.3 |
| Lite (300) | ±1.3 | ±2.5 | ±3.4 | ±4.5 | ±5.5 |
| HumanEval+ (164) | ±1.9 | ±3.5 | ±4.6 | ±6.1 | ±7.4 |
| MBPP+ (378) | ±1.1 | ±2.2 | ±2.9 | ±4.0 | ±5.0 |

Decision rule: win requires non-overlapping CIs across k>=3 seeds per system
(paired where tasks allow). Tie = overlapping CIs. No single-run claims ever.

## Fairness rules (fixed now)

1. Same task formatter, same tool budget (none for v1 comparison; identical
   mini-swe tool loop if/when agentic mode is compared), identical retry
   policy: k independent samples, majority-or-best-of-k declared in advance.
2. Decoding fixed: Gemma temp 1.0 / top_p 0.95 / top_k 64 / seed 42 (already
   recorded in repo config); cell system uses its own deterministic race —
   noted as an asymmetry inherent to substrates, disclosed, not hidden.
3. Compute accounting: wall-clock AND parameter count reported for both;
   "matched compute" claims require stating the axis matched.
4. Contamination statement shipped with every result table.
5. Claim grammar: "cell system >= Gemma-4-12B-QAT on <bench> at k=3, p<0.05
   by Wilson interval" — nothing stronger.

## Milestone ladder (below-Gemma targets, quarterly-resolvable)

M0: beat random completion on simple-SWE hidden tests (any margin, k=3).
M1: beat smoothed n-gram code baseline (trained on same corpus).
M2: beat templated-heuristic agent (pattern-match + template emit).
M3: beat tiny neural LM (fixed param/compute budget) on simple-SWE AND
    HumanEval+ > 0 with CI excluding zero.
M4: HumanEval+ >= 10% (CI-bound ±4.6pp at that point).
M5: MBPP+ >= 20%; simple-SWE unseen-family transfer demonstrated.
M6: SWE-bench Verified attempted; headline comparison vs Gemma-4-12B-QAT.

Honest power analysis: M0-M2 feasible this quarter contingent on the a013
representation + local-learning results (they gate everything). M3 plausible
within two quarters post-a013. M4-M6 are year-scale and depend on the sparse
runtime demonstrating real network dynamics at >=1e7 cells (activity-density
caveat from prior campaigns applies: array-throughput != useful computation).

## Red tests (test-first, when implemented)

- Formatter equivalence test: byte-identical task rendering to both systems.
- Determinism test: Gemma seed pinning reproduces; cell race reproducibility.
- CI math unit tests against known Wilson tables.
- Contamination audit test: bench task IDs checked against training-corpus
  manifests where available.
