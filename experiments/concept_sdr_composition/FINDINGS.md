# FINDINGS: do concept tokens yield more separable SDR unions than bytes?

**Study**: `concept-sdr-composition-v1` — composition of two landed wins
(concept tokenizer + U-STATE SDR union memory).
**Runner**: `run_composition.py` · **Raw numbers**: `results_v1.json`
**Head**: `2b0dd14` (sdr_union.py D2) · Corpus sha256 verified against
`FREEZE-CANDIDATE-v3-BIBLE.json` before any encoding.

## Answer

**Yes.** Under an identical seeded projector (n=2048, w=32, seed=7) and the
frozen union rule `U_t = max(λ·U_{t-1}, s_t)`, encoding Bible TRAIN verses
with concept tokens produces measurably more separable union states than byte
tokens — on pairwise cosine geometry, on the collapse-tail diagnostics that
ustate-preregistration.md §2 flagged, and on a 49-way book-ID decode probe.
The advantage **grows with λ**, exactly where byte unions are known to
collapse.

## Setup (one paragraph)

49 TRAIN books from the frozen split → first 4 verses each + 4 top-up = **200
verses** (46,722 bytes). Three arms, same projector, λ ∈ {0.90, 0.97, 0.99}
(prereg sweep):

| arm | tokenization | writes/200 verses |
|---|---|---|
| `byte_full` | BytePhraseTokenizer(max_phrases=0) — pure bytes, ids [0,256) — the week-1 U-STATE tokenizer | 46,722 |
| `concept` | same base + 512 mined concepts (freq≥50, MI≥5 bits) mapped to their own SDRs in band [1024,1536); residual bytes stay base-band | 28,410 (0.61×) |
| `byte_matched` | byte stream truncated to the concept stream's length (controls for fewer SDR writes / less decay dilution) | 28,410 |

Concept streams used **439 distinct ids (367 concept-band)** vs **74 distinct
ids** for bytes — the diversity delta is the hypothesized mechanism.

## Result 1 — Pairwise union cosine (200 verses ⇒ 19,900 pairs)

Pooled median cosine (**lower = more separable**) and collapse tail:

| λ | byte med cos | concept med cos | Δ | byte frac>0.90 | concept frac>0.90 |
|---|---|---|---|---|---|
| 0.90 | 0.6940 | **0.6136** | −0.080 | 1.56% | 1.06% |
| 0.97 | 0.8251 | **0.7123** | −0.113 | 4.85% | 0.32% |
| 0.99 | 0.8847 | **0.7475** | −0.137 | 34.92% | **0.12%** |

In G-U1 units (median 1−cosine): byte 0.3060/0.1749/0.1153 vs concept
**0.3864/0.2877/0.2525** — concept separability is +26% / +64% / **+119%**
over bytes. At λ=0.99 the byte collapse tail is catastrophic (35% of verse
pairs above 0.90 cosine, max 0.9967) while concepts hold max cross-item
cosine at 0.9535 and reduce the tail ~290×. Sanity: byte_full at λ=0.97
(median 1−cos 0.1749) reproduces the frozen capacity-precondition table
(0.1898 on TRAIN-200 prompts) well inside the 2× reproduction gate.

## Result 2 — Mechanism: the common-byte collapse tail shrinks

Prereg §2 identified universal bits (active in ≥95% of items) carrying
0.60–0.82 of L2 mass as byte-union pathology. Replicated and treated:

| λ | univ bits byte→concept | univ L2 mass byte→concept |
|---|---|---|
| 0.90 | 462 → 350 | 0.877 → 0.705 |
| 0.97 | 462 → 350 | 0.793 → 0.562 |
| 0.99 | 462 → 350 | 0.700 → **0.447** |

Mean union support (>0) rises 836.9 → **1079.3 bits** (+29%). Active-bit
trajectories (λ=0.97, 20 normalized-position bins) show why: byte unions
plateau early (first/mid/last bin = 215/794/887 active bits) because skewed
bytes (' ', 'e', 't') re-set the same bits; concept unions keep acquiring
fresh bits through the verse (170/996/**1244**). The `byte_matched` control
plateaus identically to `byte_full` (155/713/819) — **fewer writes alone does
not produce the effect**; the effect comes from *which* SDRs get written
(phrases → fresh near-orthogonal 32-bit sets instead of re-hammering 74
recycled byte SDRs).

## Result 3 — Decode-ability proxy: 49-way book-ID logistic probe

Logistic regression (standardized, C=1) on the 2048-dim union, stratified
4-fold CV, chance = 1/49 = 2.04%; shuffled-label control run at λ=0.97:

| λ | byte_full | concept | byte_matched |
|---|---|---|---|
| 0.90 | 11.5% | **12.0%** | 4.5% |
| 0.97 | 12.0% | **16.5%** | 8.0% |
| 0.99 | 14.0% | **21.5%** | 8.5% |

Concept unions are more decodable at every λ (+0.5 / +4.5 / +7.5 pts;
+54% relative at λ=0.99), and concept fold accuracies are stable
(20–22% everywhere at λ=0.99) while byte folds swing 8–20%. Shuffled-label
control: 1.5% ≈ chance ✓. The matched-steps control being *worse* than
byte_full rules out "concepts win because their unions decay less" — at equal
write budget, prefix-byte unions are strictly less decodable.

## Verdict and caveats

Concept tokens improve SDR union separability on **all three metric families,
at all three frozen lambdas**, monotonically more at higher λ. For the
U-STATE agent: swapping the week-1 byte tokenizer for the concept stack is a
representation upgrade that composes cleanly with the existing projector —
no changes needed, ids just extend into [1024,1536).

Caveats (honest):
- Probe sample is small by design (~4 verses/book; 150 train rows, 2048
  features). Absolute accuracies are low; the operative readout is the
  **paired comparison on identical folds**, which favors concepts everywhere.
- Concept streams are 39% shorter; `byte_matched` controls write-count for
  trajectory shape and probe, though not for content coverage (a concept
  union sees the whole verse where its control sees a prefix).
- Top mined concepts are function-word chunks (' th', 'the', ' the '); even
  these help by diverting mass away from universal byte bits — richer
  concept inventories should only widen the gap.
- Single corpus (KJV TRAIN books), single seed (prereg-pinned). No λ or seed
  was tuned post hoc.
