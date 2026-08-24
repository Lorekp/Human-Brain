# Custom-architecture research — post-reservoir-falsification directions

Status: research report adopted as program decision input, 2026-08-24.
Full report in delegation transcript deleg_8d162a6c; operative content here.

## Ranking

| # | Direction | P(exact>=1 / 4wk) | Cost | Verdict |
|---|---|---|---|---|
| 1 | **A. SDR union-state memory** (HTM/TM-derived) | **~55%** | $0-7 | RECOMMEND |
| 2 | **B. Apical dendritic coincidence encoder** | ~30% | $0-14 | RECOMMEND (parallel) |
| 3 | C. Astrocyte slow-gate episode tags | ~20% | low-med | Hold |
| 4 | D. STDP assemblies + workspace broadcast | ~15% | med-high | Fold into A/B later |
| 5 | E. Predictive-coding hierarchy | ~12% | high | Defer |

Dropped: oscillator/gamma phase-binding (no substrate), NCA/manifolds
(no text-computational grounding).

## The organizing insight

Input-side structure separates contexts (C_rand G1 = 0.1019 PASS) while
reservoir dynamics destroy identity before the decision point (cosine
0.9989; MLP ceiling below ignoring-the-state control). Architectures that
WRITE context into retrievable/conjunctive form at encode time outrank
architectures hoping trained dynamics spontaneously retain it.

## Direction A: U-STATE (recommended primary)

Token bytes -> seeded random SDRs (n=2048-8192, w=32-64) -> union state
U_t = max(lambda*U_{t-1}, s_t) IS the network state; race() queries U_t
directly. Different contexts yield near-disjoint unions BY CONSTRUCTION
(~w^2/n bits expected overlap) — cosine collapse is structurally impossible
until decay saturates, and decay is a preregistered knob. Reuses everything:
QA harness, probes.py unchanged against new states, injection seam,
observation_sink. New code: ~300 LoC numpy. R0-c's SDM failure doesn't
apply: those keys were reservoir-filtered coarse vectors; direct input-side
unions were never tested.

Gates: G-U1 separation >=0.10 + cosine <=0.90; G-U2 decode >=6.19% (z>=2 vs
unigram); G-U3 shuffled-key control; G-U4 replay >=39/42. Verdict inside
5 days, $0 until the optional <$7 VAL leg.

Falsifier: decode-on-raw-unions below majority-prior control => entire
input-representation branch dies cleanly.

## Direction B: APICAL-AND (recommended parallel)

Two-compartment GLIF5 (apical tau~100ms integrator + plateau threshold
multiplicatively lowering soma threshold): per-cell nonlinear AND between
context signal and token stream — conjunctive features created online, the
exact operation fixed random mixing cannot perform. Composes with A (apical
driven by A's union vector). Gates: conjunction index >=1.5x, separation
>=0.10, decode z>=2. ~400 LoC CPU-only. P(fail like others) ~55%; its
falsifier redirects to explicit-store hybrids if AND features pass but
decode fails.

## Week-1 plan (both run in parallel, ~1 day shared overhead)

A: D1 preregistration freeze -> D2 sdr_union.py (~300 LoC) -> D3 rerun the
UNCHANGED r0c probe suite on union states -> D4-5 verdict: pass => install
feature_family="sdr_union" behind config flag, one-shot Bible-QA VAL leg;
fail => negative-with-localization-v2 and elevate B.

B: D1 preregistration freeze -> D2-4 two-compartment cells.py subclass ->
D5 drive apical from A's union at N=1024 through probes.py.

Friday deliverable: first above-majority-prior decode signal in program
history, or a clean kill of the sparse-code route. Either is decision-grade.
