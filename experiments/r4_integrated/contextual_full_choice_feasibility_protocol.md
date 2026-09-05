# Saved-TRAIN full-choice ranking feasibility

Numerical diagnostic only, 2026-09-05. Entire preparation, one bounded solve and
verification fit **08:35–09:05 UTC**. Freeze this protocol and reviewed script
before any solve using the actual saved TRAIN basis. Synthetic arithmetic/input
checks are allowed. No learner, encoder, native simulation, teacher execution,
new sources/questions or learned-parameter publication occurs.

## Question and authenticated inputs

The [full-choice acquisition protocol](contextual_full_choice_protocol.md)
used complete supervised competition but failed its acquisition thresholds.
Ask whether one shared linear score vector can simultaneously rank the exact
actual-TRAIN targets above every other available choice. This separates a
possible fixed-feature ranking witness from the observed optimizer trajectory;
it does not establish reachable learning, data adequacy or generalization.

Use only completed `benchmarks/contextual_full_choice_acquisition_v1.json/.npz`,
source `dd871c936893628f2deb37e79fa3c25da0f2be50`, and their fixed input fixture.
Authenticate JSON SHA256
`c25cfd867a203cc8ce6ba5f5b2af11cf880fd18ccab055fd5243c452b4671c08`
and NPZ SHA256
`ff7f61a8b6134a78d578316faf24df0863128aab90891f18205c10374de1983b`.
Use the 64 actual questions in fixture order, not the 64 alternate forms, growth
sources, final learned weights, official reading material or any fresh labels.

Reconstruct unchanged gain16 BINDING features through the pure saved-data basis
helpers; no project runtime imports. The frozen standalone script contains the
reconstruction arithmetic and binds its source hash; the independent auditor
uses the separately hashed pure helpers from the acquisition audit. Validate sealed TRAIN source hash, candidate source/byte
custody, query membership and normalized saved components. Match all 64 rebuilt
actual-query feature hashes against authenticated acquisition offers. Verify
source, fixture and both immutable input artifact hashes before and after work.
There are 1,458 candidate occurrences plus STOP. Assert exactly one authorized
choice per actual question: 48 unique literal targets and 16 explicit STOPs.
A duplicate or missing gold is an input-contract stop, not an arbitrary choice
of provenance. Thus there are exactly **64 × 1,458 = 93,312** comparisons.

## Fixed mathematical problem

Let w=(v,s) contain 1,621 effective GO-minus-NO coordinates and one learned STOP
coordinate. Persistent bias remains zero. For candidate c, x_c=(feature_c,0);
for STOP, x_STOP=(0,...,0,1). Include the pure-query feature coordinates: unlike
old pair-only feasibility, they affect candidate-versus-STOP comparisons.

```text
minimize t
subject to (x_gold - x_wrong) dot w >= 1.000001, for every wrong choice
           -t <= w_i <= t, i=0..1621
           t >= 0
```

All 1,622 w coordinates have unrestricted signs and no fixed magnitude box;
t is optimized, not prescribed. SciPy's default nonnegative variable bounds
must be overridden explicitly for w. No regularization/feature/target scaling,
post-solve witness rescaling or learned weight initialization is implicit.
Minimizing the L-infinity norm is a conditioning proxy, not a capacity measure.
A feasible witness need not certify the globally minimum norm of the full LP.

Use installed SciPy **1.17.1** `linprog(method="highs-ds")`, `presolve=True`,
`primal_feasibility_tolerance=1e-9`, `dual_feasibility_tolerance=1e-9`; other solver
options remain documented defaults. The method-specific
[SciPy documentation](https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs-ds.html)
and local `scipy/optimize/_linprog_highs.py` describe free bounds, dual simplex,
feasibility tolerances and time limits. Record actual SciPy/NumPy/HiGHS identity.

Use deterministic constraint generation, at most **32 LP solves**. Start from
zero w and select one most violated wrong choice per query; order by fixture
query, then candidate index (STOP last) for equal violations. Add only new rows,
retain all previously admitted rows, and run a complete 93,312-comparison scan
after each solution. This is computational separation of a fixed problem,
not teacher exposure selection or a hyperparameter search. If the same violated
row persists without a new cut, terminate inconclusive rather than loop/repair.
The entire run has a **900-second** allowance, including input reconstruction,
separation, solver calls and verification, capped by the hard 09:05 UTC deadline.
Check the deadline during reconstruction and scans; each solver receives only
the remaining allowance. This is stricter than a solver-only limit.
No second algorithm, tolerance/rank/scale sweep or restart is authorized.

## Verification and scientific outcomes

A successful solver status is not the certificate. Save the finite float64 w
and verify **all 93,312 original, unscaled margins with math.fsum**, requiring
minimum margin **>=1**, with no relaxed 1-minus-tolerance acceptance. The LP's
1.000001 target provides a predeclared numerical buffer. Independently compare
ordinary matrix scoring against the fsum reference within
`1e-10 + 1e-10*abs(reference)`. Record minimum margin, maximum disagreement,
nonfinite/violated counts, weight norms/ranges and cancellation magnitude.
This is an independently checked floating-point witness, not an exact rational
proof. Do not replace the saved witness with tolerance-adjusted coordinates.

A fully verified witness answers only that these fixed actual-TRAIN labels can
be strictly ranked by this representation. Unit margins exclude exact raw
choice ties; they do not prove the finite-precision affine/GLIF race would
execute the same choices at potentially large witness magnitudes. Do not deploy
w or report host fitting as neuronal learning, an acquisition pass or fresh fit.

Solver infeasibility, numerical trouble, duplicate-cut stall, round/time budget
exhaustion or failed verification is **INCONCLUSIVE**, unless a separate exact
contradiction is already available (for example an exactly zero comparison row).
No new certificate solver is required. In particular, approximately zero Farkas
residuals cannot certify infeasibility with unbounded w: their residual dot w
can be arbitrarily large. Do not infer representation impossibility from status2.

## Artifacts, cost and exit

Save immutable JSON plus NPZ witness/active-row IDs: input/source/helper hashes,
fixed options, cut order, per-round status/objective/timing/full-scan minima,
final status and witness hash. An independent reader verifies the saved witness
without calling the solver or its verifier; failures retain partial evidence.
Document the final full-scan ordering and row-to-query/candidate mapping.

A 1,622-coordinate float64 witness occupies 12,976 bytes. One 1,458×1,621 feature
matrix uses 18,907,344 bytes; retaining all 64 uses 1,210,070,016 bytes. Materializing
all comparison rows densely would use 1,210,816,512 bytes before auxiliary t,
solver copies or workspaces. The implementation materializes the 64-query feature
array as a disk-backed .npy memmap under ignored `.pytest-tmp/`, not a 1.21GB heap
cache or published benchmark artifact. Record its exact local path/bytes and
retain it only as diagnostic scratch; mapped resident pages still count in parent
memory. Publish only the compact witness/active-constraint NPZ plus JSON, using
sparse active cuts/norm inequalities where implemented. At most 2,048 new query cuts can be admitted in
32 rounds; report actual rows, cached arrays, solver/process memory and time.
The source/query basis, gradients and existing acquisition artifacts remain
unchanged. Close with feasible-witness or inconclusive evidence by the deadline;
this diagnostic does not authorize another learning rule or extra teaching.
