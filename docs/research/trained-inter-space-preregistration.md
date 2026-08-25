# Trained inter-space mapping week-1 preregistration (T-ROUTE) — FROZEN BEFORE IMPLEMENTATION

Status: frozen 2026-08-25, before any `hebbian_projection` implementation or E1
execution. Branch `codex/a013-cognition-design` @ 884c987. Sources adopted
verbatim: `docs/research/t-route-strategy.md` @ 884c987 (gates G-T1/G-T2/G-T3
and decision semantics quoted from it unchanged),
`docs/research/ustate-capacity-study.md` (P-pop lambda choice),
`docs/research/custom-architecture-directions.md`,
comparator anchors in `experiments/r0c_probes/probes.py` and
`experiments/qa_baselines_bible_v1/*.json` (UNCHANGED — they score this
experiment too), corpus pins inherited from the U-STATE preregistration.

Nothing below may be edited after the first episode is encoded. Failure is a
publishable result. All thresholds literal; all controls seeded; no re-sweep,
no threshold edits, no seed changes, no alternate decoders after seeing
results. Amendments before spend follow the U-STATE Change Note 1 discipline
(explicit, cited, committed before execution).

## 1. Representation spec (pinned)

- Tokenizer: `BytePhraseTokenizer.fit([it["serialized_prompt_answer"] for it in
  train_items], TokenizerConfig(max_phrases=0, min_frequency=3))` — identical
  fit recipe to r0c/U-STATE loggers; `tokenizer.vocab_size == 256` asserted
  (byte ids 0..255). Question stream = ids of `item["question"]`; passage
  stream = ids of `"\n".join(item["context"])`; answer stream = ids of
  `item["answer"]`.
- Projection: the UNCHANGED `sdr_union.SDRProjector(n_bits=2048, w_ones=32,
  seed=7)` — same seeded token->SDR convention (`default_rng((seed, n, w,
  token_id)).choice`, sorted). Both spaces share this one codebook; weight
  sharing across items is keyed by word identity through it.
- **Q-pop**: undecayed union of question-stream token SDRs (union with
  `lam_Q = 1.0`: `Q = OR_i s_{q_i}`). Rationale frozen: a question is a query,
  not a growing context; no recency weighting.
- **P-pop**: passage union state over the passage stream with `lam_P = 0.90`
  (the capacity study's mechanical winner for distinctness at verse scale;
  n2048/w32/lam.90 median cosine 0.5796–0.5825). P-pop is used only for arm-B
  correlation binding; routing readout uses per-token SDRs + position
  bookkeeping (below).
- **W** (arm A substrate): sparse `n x n` float64 matrix, fan-in `k_W = 32`
  pre-synaptic partners per post bit drawn WITHOUT replacement:
  `rng = numpy.random.default_rng((W_SEED, n_bits, k_W, post_bit))`;
  `partners = rng.choice(n_bits, size=k_W, replace=False)` (unsorted, as
  drawn); initial values `vals = rng.uniform(0.5, 1.5, size=k_W)` from the
  SAME generator continuation (seeded-random init, same projector-keyed
  convention as sdr_union.py). All other entries structurally zero.
  `W_SEED = 7`.
- Accumulation is ONE-SHOT: a single pass over TRAIN episodes, no epochs, no
  decay between episodes, no learning-rate schedule. Synapse-local: each
  synapse accumulates only its own `pre x post x M` products.

## 2. Episode builder and pairing rule (pinned)

Episodes consume the build_bible_qa_corpus.py manifests verbatim:
`$LOCALAPPDATA/Temp/hb_bible_freeze/evidence/qa_manifest_train.json` (200
items) / `qa_manifest_val.json` (60 items); logger verifies ALL of:
freeze candidate RAW bytes sha256
`361c83907d9b353c8128a55ea2b4ca49a01c2aa8c8d1f6224db4661c4c9e166c`;
TRAIN manifest raw-file sha256
`d82fe615dced145d4a6d10d19e90583d1eca4bb6be79e638a96bebc511b0a0e8` with its
embedded `sha256_canonical` field equal to the U-STATE-inherited pin
`d256c27dacec0cb2a169005611338c3408bbb252ba619dcde86fb01e3f8c9a7b`; VAL
manifest raw-file sha256
`4f9be9b725c630cbc0e94f8cea4a96413bc026895a3d1b5312882e1980e37dec` with
embedded `sha256_canonical` field equal to
`2771de48a90d9e694cc4260e9392d83c8e39f2aca7d57fe06dd51c7ce9a55bec`
(any mismatch => STOP; nothing runs). Families: who_said PRIMARY (gating),
address_lookup / verse_continuation reported strata.

Pairing events per TRAIN episode (the three-factor estimator of the strategy's
"quote-subassembly -> same-subassembly-at-passage-position + local adjacency"
target; M is a binary third factor = question-onset eligibility):

- For each question token id `tq` at any question position: let `J` = positions
  `j` in the passage stream with `t_p[j] == tq` (token-identity anchor match;
  capped at the FIRST `O_MAX = 8` occurrences by ascending position to bound
  high-frequency tokens like "the"/",").
- For each anchored `j`: every passage position `j'` with
  `|j' - j| <= H_WIN = 12` contributes event `(tq, t_p[j'])` with `M = 1`.
- Accumulation identity: `dW[np.ix_(sdr(tq), sdr(t_p[j']))] += M` — outer
  product of the two w-hot token SDRs scaled by M, i.e. a 32x32 block += 1
  per event.
- No other source of dW exists. VAL episodes NEVER touch W.

## 3. Arms (frozen)

- **Arm A** `sparse_trained_homeostat`: sparse trained W of §1/§2 plus
  homeostatic IDF-like row scaling applied ONCE after accumulation:
  `df[b]` = number of TRAIN passage streams whose token-union contains bit b;
  `scale[b] = ln((N_pass + 1) / (df[b] + 1))`; row `b` of W multiplied by
  `scale[b]`, then rows with nonzero norm L2-normalized
  (`W[b,:] /= ||W[b,:]||_2` iff norm > 0; zero rows stay zero). This is the
  install candidate.
- **Arm B** `dense_unnormalized_correlation`: `C += outer(Q_pop_e, P_pop_e)`
  accumulated over TRAIN episodes in canonical manifest order (undecayed
  unions both sides), stored dense float64, NO normalization of any kind —
  the correlational soft-binding signature measured twice already
  (strategy ranking #2). Differs from A in binding granularity, support, and
  normalization exactly as the strategy frames the comparison.
- **Controls** (all seeded, none tuned): 
  - `shuffled-question`: item i routed with the question of item
    `pi(i)`, `pi = default_rng(11).permutation(len(split))` computed within
    each split; scored against item i's OWN gold region.
  - `randomized-W`: untrained sparse matrix, identical topology/value recipe
    with `W_SEED = 13` (never sees any episode).
  - `wrong-passage`: item i routed with its real question but scored against
    the gold region of item `sigma(i)`, `sigma(i) = (i + len(split)//2) % len`
    split (guaranteed mismatch for len >= 2).

## 4. Routing readout and gold regions (identical machinery for every arm/control)

- Route: `drive = M_arm @ q_pop` (M_arm ∈ {A, B, randomized-W}); winners =
  top `K = 128` drive indices; ties broken by ascending bit index; exact-k,
  unique, deterministic.
- Gold answer region: every exact contiguous match of the answer token
  sequence in the passage stream, extended by `ADJ = 2` positions each side
  (clamped); `gold_bits` = undecayed union of those tokens' SDRs. Items with
  ZERO matches are `GOLD_UNMATCHED` and mechanically excluded from every
  G-T1/G-T3 denominator (exclusion counts reported; exclusion is positional,
  never judged).
- Overlap metric: absolute intersection size `|winners ∩ gold_bits|` (bits).

## 5. Gates (verbatim semantics from t-route-strategy.md, mechanical operationalization)

- **G-T1 routing enrichment**: median overlap(retrieved winners, gold
  answer-region SDRs) >= 2x shuffled-question AND >= 1.5x randomized-W, per
  family stratum, TRAIN-self vs VAL split reported (early-warning contrast).
  Operationalization: medians computed WITHIN (stratum, split) over eligible
  items. GATING stratum = who_said (declared primary in §2; the strategy's
  decisive fact is who-said structure; address/continuation families lack
  quote anchors by construction and are REPORTED, not gated). G-T1 PASSES iff
  on BOTH splits (TRAIN-self AND VAL): `base_med > 0 AND base_med >= 2 *
  shuf_med AND base_med >= 1.5 * rand_med` (arm A numbers; zero-denominator
  case handled by the explicit `base_med > 0` conjunct). The TRAIN-self/VAL
  enrichment contrast is reported verbatim in FINDINGS either way.
- **G-T2 full pipeline**: full pipeline VAL exact >= 3/639. Pipeline (two-hop
  routed retrieval, arm A): hop 1 route q_pop -> winner bits S1 (K=128);
  hop 2 temporal contiguity — backproject S1 to passage positions
  (`pos_score(j) = #\{j' : |j-j'| <= H2\}` with `H2 = 8` counting positions
  whose token writes >= 1 bit of S1), take argmax center (ties: lowest
  index), retrieved region `R` = undecayed union of token SDRs in
  `[center-H2, center+H2]`. Extraction: candidates = distinct TRAIN answer
  strings; candidate SDR `B_c` = undecayed union of encode(c);
  `score(c) = |B_c ∩ R| / |B_c|`; ranking by (-score, majority_rank_position,
  string) where majority_rank_position is probes.py `majority_ranking` order
  over TRAIN answers. Rankings fed to the UNCHANGED `probes.py SpanScorer`
  (639 VAL answer-word positions enforced). Arm-B pipeline reported as
  diagnostic; TRAIN-self pipeline reported as early-warning diagnostic.
- **G-T3 anti-leak**: wrong-passage routing <= chance; shuffled-W inside
  random band. Operationalization: chance per comparison = analytic
  hypergeometric expectation `K * |gold_sigma(i)| / n`; G-T3a: MEDIAN wrong-
  passage overlap <= MEDIAN of per-item expectations (who_said stratum, pooled
  splits, arm-A routing). `shuffled-W` = W with its nonzero VALUES permuted
  among the existing support via `default_rng(11)` (topology fixed);
  G-T3b: `rand_med/1.5 <= shufw_med <= rand_med * 1.5` where `rand_med` is
  the randomized-W median (who_said, pooled splits).

An arm's week advances iff G-T1 ∧ G-T3 (+ G-T2 for the pass path).

## 6. Gate order and stop-on-failure

0. Integrity: corpus/manifest digests, vocab==256, counts (200/60), GOLD_
   UNMATCHED census — any mismatch => STOP, fix nothing silently.
1. Log routed winner sets for ALL arms/controls (observation_sink pattern)
   and evaluate G-T3 then G-T1 from the logged sets only.
2. G-T1 FAIL => verdict NEGATIVE_CORE_WOUND (falsifier fired: locally-trained
   projections carry no more cross-space structure than raw fixed-random
   overlap). The pipeline leg (G-T2) is NOT spent — directions die together
   at the shared root. Diagnostics still recorded verbatim.
3. G-T1 PASS ∧ G-T3 FAIL => verdict NEGATIVE_WITH_LOCALIZATION_V3_LEAK;
   pipeline run once as DIAGNOSTIC ONLY (cannot produce a pass path).
4. G-T1 ∧ G-T3 PASS => run G-T2 pipeline leg ONCE.
5. Verdict per §7. No reruns, no second seeds, no threshold edits.

## 7. Mechanical verdict (no judgment calls)

Computed from `experiments/t_route/e1_results.json` + logged-winner-set
manifest, numbers verbatim:

- `T1 fail` => **NEGATIVE_CORE_WOUND** — trained-inter-space hypothesis takes
  its core wound; T-ROUTE and correlational directions die together at the
  shared root; escalation ladder (synthetic episodes, then global-scalar RPE
  third factor) noted as the only remaining options within the class.
- `T1 pass ∧ T3 pass ∧ T2 pass` (>= 3/639) => **PROCEED_TO_INSTALL** —
  install behind config flag; one-shot official VAL leg week 2.
- `T1 pass ∧ T3 pass ∧ T2 fail` => **NEGATIVE_WITH_LOCALIZATION_V3** —
  routing works, extraction broken; escalate extraction ONLY.
- `T1 pass ∧ T3 fail` => **NEGATIVE_WITH_LOCALIZATION_V3_LEAK** (per §6.3).

Comparators reported verbatim beside every pipeline number, UNCHANGED:
unigram span-acc 0.04538341158059468 (29/639 = 4.54%), majority-prior-within-
answer-vocab 0.0406885758998435 (26/639 = 4.07%), random-ranking-seed11
0.001564945226917058 (1/639).

## 8. Commit discipline

Incremental commits on `codex/a013-cognition-design`, pull before push, only
own files touched: D1 this document; D2 red tests; D2-3 module green +
tests; D4 logging harness + npz/manifest + e1_results.json (binary npz stays
local; array-level sha256s committed in the manifest); D5 FINDINGS.md.
