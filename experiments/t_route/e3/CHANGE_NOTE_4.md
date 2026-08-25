# Change Note 4 (T-ROUTE) — E3 probe amendment: QUOTE-SPAN-GATED third factor.
# PRE-SPEND: committed BEFORE any E3 gated pairing event is encoded, any E3
# winner set is routed, or any E3 gate is evaluated.

Parent protocol: `docs/research/trained-inter-space-preregistration.md`
(FROZEN 2026-08-25; E1 SPENT `NEGATIVE_CORE_WOUND`, E2 SPENT
`RECOVERY_FAILS_SPARSITY_INSUFFICIENT`). This note amends NOTHING about E1/E2
and edits nothing in the parent documents — it is a standalone, cited amendment
defining probe E3, the LAST untested variant inside local-learning constraints
per `docs/research/t-route-e1e2-verdict.md` (@25c869f). After the first E3
gated pairing event is encoded, nothing below may be changed. Failure is a
publishable result and, per that verdict document's Program-state section,
would be program-grade for Bible-QA.

## C1. Motivation (context, not a gate)

E2 proved the raw unnormalized Hebbian accumulator is a QUERY-INDEPENDENT
static prior: shuffled-question enrichment 1.000–1.057, winners collapsed to a
~170-bit pool (cross-question winner Jaccard 0.78–0.81), sparsity flat across
k ∈ {8,16,32}. The verdict document localizes the missing ingredient: the third
factor must be QUOTE-SPECIFIC, not merely episode-scoped. Hypothesis under
probe (declared before any E3 data exists): gating dW by token-position overlap
between the question's verbatim quote and each passage position turns W from a
corpus-prior accumulator into a quote-conditioned routing table whose winners
depend on WHICH quote is asked about (breaking the ≥0.78 cross-question
Jaccard signature) while preserving gold-region drive.

## C2. Arm family (frozen)

Single config; NO sweep (E2 showed k flat ⇒ multiplicity control).

- Substrate: the UNCHANGED §1/§2 sparse accumulator (n_bits=2048, w_ones=32,
  fan_in=32, W_SEED=7 init U(0.5,1.5), projector seed 7) receiving ONE-SHOT
  aggregated token-pair delta over TRAIN episodes ONLY, computed with the
  QUOTE-SPAN GATE:
  - Quote text of episode e: the substring of item["question"] between the
    FIRST '"' and the LAST '"'. Episodes with fewer than two quotes, or an
    empty span, have an EMPTY mask.
  - Span localization: q_bytes = utf-8(quote); p_bytes =
    utf-8("\n".join(item["context"])); occurrences = ALL contiguous
    byte-subsequence matches of q_bytes in p_bytes (overlapping allowed);
    mask(e) = union of half-open token-position ranges [k, k+len(q_bytes)).
    The tokenizer recipe is verbatim E1/E2 (`max_phrases=0, min_frequency=3`,
    fit on TRAIN serialized_prompt_answer); vocab_size==256 is asserted at run
    time, i.e., tokens ARE bytes, so subsequence matching is exact and
    boundary-effect-free.
  - Pairing rule VERBATIM E1/E2 §2 (unique question tokens in first-occurrence
    order; anchors = ascending positions of tq in the passage, first O_MAX=8;
    pairs to every passage token within ±H_WIN=12), then multiplied by the
    gate: a pair (tq, p_ids[jp]) contributes iff the POSTSYNAPTIC passage
    position jp ∈ mask(e) — algebraically dW_ij *= 1[token_j ∈ quote_span(e)],
    implemented as masked counting (identical mass on identical synapses).
  - Episodes with an empty mask contribute ZERO plasticity. Design-time census
    (run pre-commit, disclosed here; saw corpus text only, no routing/gold
    data): exactly the 60 TRAIN who_said episodes carry a quoted substring
    (60/60 matched in-passage, all single-match; byte lengths mean 36.9,
    median 36.5, range [12, 64]); address_lookup (70) and verse_continuation
    (70) questions contain no quotes and therefore train the gated arms NOT
    AT ALL. VAL episodes never touch W (unchanged parent rule).
- Post-training transform (REPLACES nothing else; NO homeostat, NO
  normalization): per-row top-k sparsification with k=16 FIXED — keep the k=16
  largest-valued entries per row among structural nonzeros, zero the rest;
  ties by ASCENDING column index; rows with <16 nonzeros keep all (E2's exact
  `topk_row_sparsify` code, k frozen at the E2-best config; the E2 sweep was
  flat, so this choice carries no selected-outcome information).
- Config labels: `arm_q16` (quote-span-gated base), `rs16` (random-span
  control, C3b).
- Routing readout UNCHANGED §4: full-question undecayed population union
  through the unchanged projector → drive = M @ q_pop → `kwinner_indices`,
  K=128, ties ascending index. The gate is a PLASTICITY factor only; no
  query-time change of any kind.

## C3. Controls (frozen)

a. Identical to E2: `shuffled_question` — within each split, item i routed with
   question pi(i), pi = default_rng(11).permutation(len(split)), THROUGH THE
   SAME config's matrix, scored against item i's own gold region (labels
   `shufq_q16`, `shufq_rs16`). `randomized_W` — untrained seed-13 sparse
   projection, shared, routed once (reused from the hash-verified E2 log).
b. NEW random-span control `rs16`: IDENTICAL machinery, identical inclusion
   set (exactly the episodes gated in arm_q16), identical per-episode masked
   token count — except placement: for each included episode e in canonical
   TRAIN manifest order, one draw start_e ~ default_rng(17).integers(0,
   n_p − l_e + 1) with l_e = len(q_bytes(e)) (all matches share it); mask =
   contiguous window [start_e, start_e + l_e). One draw per episode; the RNG
   stream is consumed only by included episodes, in canonical order. This arm
   isolates QUOTE-SPECIFIC PLACEMENT from mere gating/sparsification of the
   training mass.
c. No wrong-passage / shuffled-W legs: E1 denominators were empty by corpus
   construction; unchanged from E2's C3.

## C4. Reproducibility preconditions (ALL must hold before ANY E3 logging)

Corpus pins, tokenizer recipe, projector, accumulation and control seeds are
inherited verbatim (all digests re-verified at run time). Before building any
E3 matrix, the harness MUST:

1. hash-verify `experiments/t_route/logged_winner_sets.npz` against its
   committed `e1_manifest.json`, deterministically rebuild every E1 substrate
   and demonstrate BIT-EXACT agreement of ALL SIX E1 routing sets (as E2 C4);
2. hash-verify `experiments/t_route/e2/e2_logged_winner_sets.npz` against
   `e2_manifest.json` and demonstrate BIT-EXACT agreement of ALL SEVEN E2
   logged routing sets (`arm_k8`, `arm_k16`, `arm_k32`, `shufq_k8`,
   `shufq_k16`, `shufq_k32`, `randomized_W`) rebuilt with the verbatim E2
   sparsifier from the rebuilt ungated accumulator;
3. assert the independently recomputed ungated pair Counter equals the E1/E2
   `aggregate_pair_counts` output exactly (same pairing data guarantee).

Gold masks, matched flags, families, sigma and permutations for scoring are
taken FROM the verified logs, never recomputed.

## C5. Gates (frozen; declared NOW, before any E3 data exists)

Eligibility: GOLD_UNMATCHED exclusion positional (matched flags from the
verified E1 log). Strata: `who_said` GATING; `address_lookup`,
`verse_continuation` REPORTED (expected empty denominators, as in E1/E2).

Per split s ∈ {TRAIN-self, VAL}, on eligible who_said items, medians of
|winners ∩ gold_bits|:
base_med = median(`arm_q16`), shuf_med = median(`shufq_q16`),
rand_med = median(`randomized_W`, from verified E2 log),
rsp_med = median(`rs16`), rshuf_med = median(`shufq_rs16`).

G-T1' conjuncts (both splits required): base_med > 0 AND
base_med >= 2*shuf_med AND base_med >= 1.5*rand_med.

Specificity ratios (guard shuf/rshuf = 0 ⇒ ratio = +inf when the numerator is
positive): spec_q = base_med / shuf_med; spec_rs = rsp_med / rshuf_med.

E3-level mechanical three-way verdict:

- `QUOTE_SPAN_ROUTING_RECOVERED` iff G-T1' holds on BOTH splits AND
  spec_q > spec_rs on BOTH splits ⇒ the quote-specific third factor works
  where episode-scoped factors failed ⇒ install behind a flag and schedule a
  dedicated VAL leg (recorded implication; installation itself out of scope
  for E3 artifacts).
- `GATING_ARTIFACT_NO_QUOTE_SPECIFICITY` iff G-T1' holds on both splits but
  spec_q > spec_rs fails on either split ⇒ raw gating/mass-reduction effects,
  not quote-specificity; no install.
- `QUOTE_SPAN_GATE_FAILS` otherwise ⇒ program-grade conclusion recorded per
  `docs/research/t-route-e1e2-verdict.md`: within local-learning constraints
  for this representation+rule class, Bible-QA needs non-local optimization or
  trained-by-construction connectivity (the R3 episodic-store hybrid).

Descriptive (not gated): per-family/split median tables including `randomized_W`,
plus logged E2 `arm_k16` and logged E1 `armB` baselines recomputed from their
verified logs on the same eligible items; gold-size headroom quantiles;
diagnostics: distinct winner bits ever selected per split, median cross-question
winner Jaccard(winners_i, winners_shufq_i) for `arm_q16` and `rs16` (any pass
claim must break the measured 0.78–0.81 static-prior signature), gated-mass
fraction vs ungated pairing events, span census echo.

## C6. One-shot discipline

`run_e3.py` executes the whole frozen pipeline ONCE: integrity pins →
deterministic rebuild → C4 checks (1)(2)(3) → build gated accumulators →
route/log ONLY the four new routing sets {`arm_q16`, `shufq_q16`, `rs16`,
`shufq_rs16`} × {train, val} into `e3_logged_winner_sets.npz` + array-hash
manifest (refuses overwrite; binary npz stays local, untracked).
`score_e3.py` then evaluates gates FROM THE LOGGED SETS ONLY into
`e3_results.json`. No reruns, no threshold edits, no seed changes, no added
configs after seeing results.

## C7. Artifacts

All under `experiments/t_route/e3/`: this note, then `toy_check.py`,
`run_e3.py`, `score_e3.py` (committed BEFORE spend; toys must be ALL GREEN),
then post-spend `e3_manifest.json`, `e3_results.json`, `FINDINGS.md`. Commit
discipline per parent §8: incremental commits on `codex/a013-cognition-design`,
pull before push, only own files touched. CPU-only, pure numpy, deterministic.
