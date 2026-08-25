"""B' APICAL-LMS Stage-0 harness (P-A execution-order step 2).

Garg-2022-style synthetic in-context regression driven through the existing
``TwoCompartmentGLIF5`` substrate with a hand-implemented local apical update
``du_A = alpha*u_A + gamma*e_t*x_t`` riding ``observe()``.

Preregistration: docs/research/path-decision-synthesis.md @ e16d22c,
section "Probe P-A -- B' = APICAL-LMS Stage-0". Constants frozen in
``freeze.py`` BEFORE implementation (see README.md). The corpus cohort leg
is BLOCKED until the sibling retro-probe (step -1) licenses it; only the
fixture-scale smoke leg has ever been run from this directory.
"""
