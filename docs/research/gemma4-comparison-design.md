# Gemma-4-12B comparison protocol design

Status: DESIGN — no bench runs, no serving changes. This document fixes the
comparison target BEFORE the cell system can tune toward it.
Provenance: researched by subagent (licenses verified from primary sources,
Wilson CIs computed), written by orchestrator from the agent's transcript
after an iteration-limit truncation. Requires independent verification pass
before any merge to master. Independent verification pass applied 2026-08-23
(see Verification log below).

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
| SWE-bench Verified | MIT (code repo; HF dataset cards state no license) | 500 | HIGH — in every major LLM's training corpus | Windows-hostile (docker-centric); WSL2 required | high (human-validated patches, fail-to-pass tests) | poor below ~5% |
| SWE-bench Lite | MIT (code repo; cards silent) | 300 | HIGH | same docker issue | medium-high | poor |
| HumanEval+ (EvalPlus) | MIT base code / Apache-2.0 (EvalPlus set + harness) | 164 | EXTREME — canonical, memorized | excellent (pure python sandbox) | very high (exec tests, plus extra tests) | good (fine-grained pass@k) |
| MBPP+ (EvalPlus) | Apache-2.0 (EvalPlus card; upstream Google MBPP CC-BY-4.0) | 378 | HIGH but weaker than HumanEval | excellent | very high | good |
| Self-hosted simple-SWE suite | project-internal | parameterizable | ZERO (fresh cohort, family-disjoint) | native | high (oracle-checked hidden cases) | best (cohort sized to resolution need) |

## Recommendation

- **Primary: self-hosted simple-SWE suite** (`simple_swe` machinery,
  preregistered cohorts). Only contamination-free option; only option whose
  difficulty can be tuned to resolve differences between a nascent cell
  system and a 12B LLM; already integrated with the cell completion adapter.
  Cohort construction must exclude the sqlfluff/L031 rule-patch family and
  any instance overlapping prior checkpoint growth trajectories (the shipped
  checkpoint grew on `sqlfluff__sqlfluff-1625`).
- **Secondary: HumanEval+ (EvalPlus)** — external anchor with objective exec
  grading, trivial host executability, and published Gemma-class reference
  numbers. Contamination is acknowledged: this measures the cell system
  against a benchmark Gemma has memorized, so it is reported as
  "unseen-generator performance", never used alone for the win/lose claim.
- **Tertiary/reference: SWE-bench Verified** for the eventual headline claim,
  run inside WSL2, ONLY when the cell system clears the secondary rung;
  contamination disclosed asymmetrically: Gemma saw SWE-bench-scale content
  in pretraining, and the shipped cell checkpoint additionally grew on one
  SWE-bench Lite dev instance (`sqlfluff__sqlfluff-1625` — verified outside
  both the Lite test-300 and the Verified test-500). A cell win over a
  memorization-inflated Gemma baseline is stronger evidence than a win over
  a clean baseline ONLY IF no graded instance overlaps any task family the
  checkpoint has experienced; both exposures are stated with every result.

## Wilson 95% CI half-widths (computed, percentage points)

| Bench (n) | p=1% | p=5% | p=10% | p=20% | p=40% |
|---|---|---|---|---|---|
| Verified (500) | ±0.9 | ±1.9 | ±2.6 | ±3.5 | ±4.3 |
| Lite (300) | ±1.3 | ±2.5 | ±3.4 | ±4.5 | ±5.5 |
| HumanEval+ (164) | ±1.9 | ±3.5 | ±4.6 | ±6.1 | ±7.4 |
| MBPP+ (378) | ±1.1 | ±2.2 | ±3.0 | ±4.0 | ±4.9 |

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
6. Cross-tokenizer budget axis: budgets are compared in characters/bytes,
   never raw tokens — Gemma is capped at max_tokens 4096 (BPE) per turn while
   the cell adapter allows max_new_tokens 512 (cell tokens, a different unit);
   report both caps and their byte-equivalents with every result.

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

## Verification log

Cross-review pass, 2026-08-23, against primary sources: raw LICENSE files via
github.com, HuggingFace dataset API + README frontmatter, datasets-server row
APIs (exact probes: hit at offset n-1, empty at offset n), and an independent
Wilson recomputation (z = 1.959964). Commit under review: 5b0fe45.

1. **Licenses**
   - SWE-bench repo `LICENSE` = MIT (raw.githubusercontent.com/SWE-bench/
     SWE-bench/main/LICENSE) — VERIFIED.
   - princeton-nlp/SWE-bench, ..._Lite, ..._Verified HF dataset cards declare
     NO license (frontmatter and body both silent) — CORRECTED: table now
     scopes MIT to the code repo instead of implying the data is MIT.
   - openai/human-eval `LICENSE` = MIT — VERIFIED.
   - evalplus/evalplus `LICENSE` = Apache-2.0 — VERIFIED.
   - evalplus/humanevalplus card license = apache-2.0; evalplus/mbppplus card
     license = apache-2.0; google-research-datasets/mbpp = cc-by-4.0 —
     CORRECTED: MBPP+ row previously claimed "CC-BY-4.0 data"; CC-BY-4.0 is
     upstream Google MBPP only.
2. **Task counts** — Verified test = 500, Lite test = 300 (dev split is a
   separate 23), HumanEval+ test = 164, MBPP+ test = 378. ALL VERIFIED.
3. **Wilson 95% half-widths recomputed** (pp): Verified 0.946/1.934/2.637/
   3.500/4.278; Lite 1.279/2.516/3.411/4.514/5.510; HumanEval+ 1.877/3.454/
   4.630/6.090/7.415; MBPP+ 1.113/2.232/**3.036**/4.023/**4.915**.
   CORRECTED MBPP+ p=10% (±2.9 → ±3.0) and p=40% (±5.0 → ±4.9); all other 18
   cells confirmed. M4's "±4.6pp" remains correct.
4. **Prior-art repo facts** — benchmarks/mini_swe_gemma.yaml matches every
   cited value: model_name openai/gemma-4-12B-it-qat-q4_0, api_base
   http://127.0.0.1:8081/v1, temperature 1.0, top_p 0.95, extra_body top_k 64,
   seed 42, max_tokens 4096 (also parallel_tool_calls=true, not cited).
   benchmarks/mini_swe_cell.yaml exists with model_name human-cell-dendritic
   (checkpoint artifacts/swe_sqlfluff_1625_cells.npz, max_new_tokens 512).
   docs/benchmark-status.md records the Gemma QAT Q4_0 GGUF SHA-256
   (93567e57…) plus executed sqlfluff__sqlfluff-1625 trajectories (expert,
   dendritic-v4, gemma) and eval artifact hashes. ALL VERIFIED. simple_swe
   machinery exists in-tree (src/humanbrain/simple_swe/sandbox.py,
   benchmarks/simple_swe_protocol_v1.json) and the cell completion adapter is
   documented in docs/research/cell-program-decoding.md.
5. **Contamination-symmetry claim** — ORIGINAL TEXT WRONG IN PART: the cell
   system is not uncontaminated on SWE-bench. Its shipped checkpoint grew from
   teacher interactions ON sqlfluff__sqlfluff-1625 (growth recorded in
   docs/benchmark-status.md). Membership scan over full instance-ID
   enumerations: sqlfluff__sqlfluff-1625 ∈ SWE-bench Lite dev(23),
   ∉ Lite test(300), ∉ Verified test(500). Also noted: only 93/300 Lite-test
   instances appear in Verified's 500. Tertiary-rung disclosure rewritten to
   state both exposures and condition the "stronger win" argument on zero
   graded-instance overlap with experienced task families.
6. **Fairness review** — found one undisclosed asymmetry: per-turn output
   budget is 4096 BPE tokens (Gemma yaml) vs 512 cell tokens (cell yaml);
   cell tokens are a different unit, so raw-token budgets are not comparable.
   Added fairness rule 6 (byte/char normalization + disclosure). v1 tool
   budget (none for both systems) is symmetric as claimed; note that
   parallel_tool_calls=true exists only on the Gemma side and must be
   equalized or disclosed if/when agentic mode is compared.
7. **Primary=self-hosted justification** — ACCEPTED: construction-level
   contamination freedom and difficulty tunability are decisive for a nascent
   system vs a fixed 12B baseline, and external-validity risk is mitigated by
   the preregistered protocol and the secondary HumanEval+ anchor. Strengthened
   the cohort-exclusion requirement (see Recommendation edit).

Residual risks not fixable by edits here: (a) SWE-bench dataset licensing is
genuinely unstated at card level — seek explicit permission before any
redistribution of instances; (b) "published Gemma-class reference numbers"
for HumanEval+ were not re-verified in this pass — cross-check the Gemma
technical report / EvalPlus leaderboard before citing any number; (c) the
decision rule (non-overlap of two independent 95% Wilson intervals) is more
conservative than the stated "p<0.05" for a difference — acceptable but the
claim grammar should not imply a two-sided paired test was run.
