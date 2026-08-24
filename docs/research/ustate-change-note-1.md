# U-STATE week-1 preregistration — CHANGE NOTE 1 (concept-arm amendment)

Status: AMENDMENT to `ustate-preregistration.md`, filed 2026-08-24 BEFORE any
D3 probe verdict was emitted (the frozen byte-leg probe invocation is
mid-flight; no P1/P2/P3 number had been read when this note was committed).
Sanctioned per the composition study's pre-spend recommendation.

## Justification (external, not post hoc)

`experiments/concept_sdr_composition/` @ b6fb803 measured THIS design space on
the SAME frozen corpus, projector (n=2048/w=32/seed=7), and lambda sweep
BEFORE any U-STATE probe spend:

- median 1-cosine separation: concept beats bytes +26% / +64% / +119% at
  lambda 0.90 / 0.97 / 0.99;
- collapse tail (pairs cos > 0.90): byte 34.92% vs concept **0.12%** at
  lambda = 0.99;
- universal-bit L2 mass (the pathology preregistration §2 froze): 0.700 →
  0.447 at lambda = 0.99; distinct written ids 74 → 439;
- book-ID decode proxy: 14.0% → 21.5% (chance 2.04%) at lambda = 0.99, with a
  `byte_matched` control showing the effect comes from WHICH SDRs get
  written, not fewer writes.

The effect was predictable from the universal-bit pathology already documented
in the frozen §2 table; no probe result influenced this amendment.

## What changes

1. THREE ADDITIONAL ARMS: concept-tokenized unions at the same lambdas,
   arm ids `lam090_concept_n2048`, `lam097_concept_n2048`,
   `lam099_concept_n2048`. Tokenizer: `ConceptTokenizer.fit(TRAIN texts,
   base=BytePhraseTokenizer(max_phrases=0,min_frequency=3),
   config=ConceptConfig())` — identical inventory to b6fb803 (freq>=50,
   MI>=5 bits, cap 512, band [1024,1536)); residual bytes stay base-band.
   SDR projection and union rule UNCHANGED (ids simply extend into the band).
2. PRIMARY designation: the concept arms are the primary encoding for week-1
   reporting per the recommendation. The three byte arms remain in the record
   exactly as originally frozen (no re-run, no edit); the mechanical §7
   verdict rule stays arm-agnostic (ANY arm qualifying triggers its branch).
3. ARTIFACT SEPARATION: concept arms log to
   `experiments/ustate_probes_concept/logged_states.npz` (+manifest). The
   frozen byte artifact is not mutated. The UNCHANGED probes.py is invoked
   once per logged artifact; "exactly once" (§3) now reads "once per
   artifact". `verdict.py` merges both results files mechanically.
4. GATE SCOPE: G-U1/G-U2/G-U3 operationalizations apply verbatim per arm for
   BOTH encodings. Gate 0 (2x reproduction of the byte precondition table)
   applies to BYTE arms only; concept arms record their capacity statistics
   with b6fb803 Result-1 values as the expectation reference (not gated).
5. BOUNDARY RULE under multi-byte tokens: prompt_ids := longest prefix of
   full_ids whose cumulative `token_bytes` length fits inside the prompt's
   UTF-8 bytes; asserted lossless round-trip (`decode(prompt_ids) ==
   prompt_text`) per item. Curriculum digests: byte groups must still equal
   the r0c anchor `c2cc7357...`; concept arms necessarily differ (different
   tokenization) and record their own digest.

## What does NOT change

Gates, thresholds, comparators, controls, seeds, projector, union rule,
lambda sweep, stop-on-failure order, mechanical verdict mapping, one-shot VAL
discipline, commit discipline.

Committed: <this file>, logger extension, then concept logging -> probes ->
merged verdict. Any deviation from this note gets recorded in FINDINGS.md as
a numbered disclosure.
