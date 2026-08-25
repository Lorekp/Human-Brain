# T-ROUTE E4 — repetition-regime probe: FINDINGS

**Verdict (frozen logic, Change Note 5 §7): `MULT_KCAP_ADDS_NOTHING_OVER_E2__ASSEMBLY_PIVOT_CLOSES`.**

Multiplicative `(1+β)` potentiation with k-cap winner selection active during encoding
does **not** recover query-dependent routing on the synthetic-episode corpus — not even
on the ≥10-repetition stratum. Per the preregistered falsifier (path-decision-synthesis
§Path 3 / DD-ASP §Cheap falsification), the assembly-flavored task pivot **closes**; the
E3 conclusion stands and NEMO deserves no installation spike from this probe.

## Execution record

- Design frozen pre-spend: `CHANGE_NOTE_5.md` (commit `a38aba7`); harness `run_e4.py`,
  scorer `score_e4.py`, mechanics validation `toy_check.py` (T1/T2/T3 all green before
  any corpus encoding). Data pins verified both raw-CRLF and LF-canonical
  (`a94a8c22…` == synthesizer manifest pin); pair census matched frozen values.
- One-shot log: `e4_logged_winner_sets.npz` (+ `e4_manifest.json`, array-level sha256),
  1656 rounds × 5 arms, 189.9 s CPU. Scorer output: `e4_results.json`.
- Incident (transparency): the first execution crashed writing the manifest
  (variable-scoping bug *after* all computation and npz write; partial preserved as
  `e4_logged_winner_sets.partial-crash1.npz`). Harness repaired, rerun once; the two
  independent full encodings produced **bit-identical arrays (11/11)** — a free
  determinism receipt; the repair touched no scientific semantics.
- git head at run: `9a93ab9`.

## Gates (frozen rule per arm×stratum: base_med>0 ∧ base≥2×shuf ∧ base≥1.5×rand)

| arm | REPEATED (n=27) base/shuf/rand | PASS | SINGLE (n=1566) base/shuf/rand | PASS |
|---|---|---|---|---|
| mult_b0.05 | 39 / 39 / 12 | no | 34 / 34 / 11 | no |
| mult_b0.25 | 34 / 34 / 12 | no | 31 / 31 / 11 | no |
| mult_b0.5  | 34 / 34 / 12 | no | 30 / 30 / 11 | no |
| add_gated  | 34 / 34 / 12 | no | 30 / 30 / 11 | no |
| add_free   | 88 / 91 / 12 | no | 80 / 80 / 11 | no |

MID band (n=36, descriptive): same shape — base≈shuf everywhere. Enrichment vs
shuffled-question is **1.00 in every gated cell** (add_free REPEATED: 0.97 — shuffled
slightly *higher*). No arm passes any stratum ⇒ verdict row 1 of Change Note 5 §7.

## What the failure looks like mechanically

1. **Winner sets are literally question-independent.** Cross-question winner Jaccard =
   **1.000 exactly** for every gated arm in both strata; per-item base-vs-shuffled
   routing Jaccard median = 1.000 (IQR 1.000–1.000). The routed top-128 is a constant
   function of nothing but W. Multiplicative growth under repeated winner-gated
   potentiation creates a small "aristocracy" of astronomically heavy synapses
   (398–1138 synapses pinned at the pre-frozen CAP_EXP ceiling by arm, β-ordered) whose
   rows dominate drive for *any* question population. The attractor locks in — but it
   locks in a **frequency/frame prior** (question-frame tokens like `Who`/`said` anchor
   the potentiation), not a query-addressable memory. Rich-get-richer compounds the
   static-pool pathology; it does not carve query-conditioned structure out of it.
2. **Ungated additive replication of E2 on synthetic data** (`add_free`, same 1656
   episodes): absolute overlap rises (88/272 gold bits on REPEATED vs ~3× randomized-W)
   yet shuffled-question matches or beats it — precisely E2's
   `RECOVERY_FAILS_SPARSITY_INSUFFICIENT` signature. Repetition-bearing data alone,
   without the new mechanisms, recovers nothing either.
3. **Winner-gating costs absolute coverage**: gated arms sit at ~30–39/272 gold bits vs
   add_free's 80–91 — competition during encoding concentrates updates on early random
   winners instead of letting all anchored pairs accumulate.

## Corpus fact the probe quantifies

Exact-string `(quote, speaker)` repetition in the 1,656 TRAIN-side synthetic episodes:
**1 pair reaches ≥10 repetitions** — `("the God of Israel", "the LORD")`, 27 episodes,
all `thus_saith` tier, 27 distinct addresses; 1,593 pairs occur exactly once; 17 pairs
in 2–9. Natural KJV attribution has essentially **no** verbatim quote/speaker repetition
outside closed liturgical formulas. This is direct, measured support for the DD-ASP
regime thesis from the data side: the repetition regime that assembly calculus needs
barely exists in the who-said task universe, and where it does exist, the mechanisms
still fail to route query-dependently.

## Falsifier accounting (verbatim branches)

- *"If even the repeated stratum fails to enrich ⇒ multiplicative+k-cap adds nothing
  over E2 and the assembly-flavored pivot closes entirely."* — **FIRED.** Every arm
  fails REPEATED (enrichment vs shuf = 1.00).
- Branch 3 (single-exposure routes query-dependently ⇒ E3 undermined / NEMO spike):
  **not fired** — SINGLE enrichment vs shuf is likewise 1.00 everywhere.
- Prediction 1 (recovery only on repeated stratum): **not observed** — there is no
  recovery anywhere to stratify.

## Caveats & scope

- The REPEATED stratum is a single (quote, speaker) pair ×27 episodes — thin, though it
  is the entire population of qualifying pairs in the pinned corpus (not a sample).
- Gold regions follow the E1 recipe (contiguous answer-token match, ADJ=2); 27
  singleton episodes were unmatched and excluded by the frozen eligibility rule.
- Strata operationalization: "single-exposure VAL items" realized as the SINGLE stratum
  (pair count == 1, encoded exactly once) because the E1/E2 freeze directory and logged
  npz do not exist in this worktree, making natural-corpus VAL gold unreconstructable
  at pinned digests (Change Note 5 §2, frozen pre-run). Both strata are scored on
  encoded material, mirroring E1/E2 train-split scoring; the controls carry the
  static-vs-routing discrimination exactly as G-T1 did.
- Fixed fan-in-32 support retained from the E2 config: the probe tests update rule +
  selection timing, not connectivity growth (frozen out-of-scope).
- Numerical clamp (CAP_EXP) was genuinely load-bearing: 398–1138 synapses per mult arm
  reached the ceiling; unfrozen growth would have overflowed float64 mid-run.

## Bottom line for the campaign

Path 3's cheap falsification is spent and returned the closing branch: **neither
multiplicative potentiation nor encoding-time k-cap competition changes the E1/E2
outcome class** — high-overlap static pools, zero query dependence, at 10× the episode
count and up to 27× exposure per item. Task-pivot hopes riding on assembly-calculus
plasticity are closed on this substrate family; remaining path-decision weight shifts
to paths 1–2 per the synthesis execution order.
