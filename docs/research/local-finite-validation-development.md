# Exact finite validation: bounded development

2026-09-06, after v2 closed at 1,468.472 seconds projected / 792.051 MiB.
The original v1/v2 contracts, cost margins and results remain immutable.

Observed workload failure: the optimized five-arm configuration still exceeds
900 seconds. A separate single-roster complete-lesson profile attributes
162.527 ms of 448.818 ms (about 36%) to `torch.isfinite` and Boolean `all`.
The saved profile is `.pytest-tmp/local-complete-lesson-profile.result.json`,
SHA-256 `796543ea50a5316c83a6d8b3b3f10b511be3c6d77eb6adfb0771b7dff7bdb038`.
These are instrumented attribution figures, not an alternate cost admission.

Hypothesis: use the existing NumPy dependency to classify finite values and
reduce the resulting Boolean mask through a detached zero-copy CPU view for
float32/float64 strided tensors. This preserves every validation, mask, error,
mutation boundary and neuron operation. Unsupported dtype/device/layout or
conversion uses the existing Torch expression. NumPy data is read only by this
helper and no view is retained. Boolean AND has no floating-point reduction
order issue; temporary Boolean mask bytes still count as runtime scratch.

One helper candidate only, shared by memory/cell and decoder finite checks.
Before timing, compare Boolean values/failure behavior on finite extremes,
NaN/positive/negative infinity, empty arrays, requires-grad tensors, masked
padding, noncontiguous/overlapping/zero-stride storage and unsupported conversion
or dtype/layout. Confirm no mutation, detached local state, exact existing
overflow failures and full focused component/composition QA.

One serial five-arm complete-lesson comparison against frozen v2 source, with
matched models/saved development episode 29, two warmups and seven paired
samples per arm. Check exact lesson records, states, weights and predictions
apart from elapsed time. Include existing snapshot/hash/finite validation cost.
Budget: at most 20 minutes implementation/QA and one 60-second timing watchdog;
no fit, seed/configuration sweep, scientific evaluation or v3 admission. A pass
requires useful complete-lesson improvement and exact parity. Return results to
the orchestrator for review before considering a successor cost run.

## Result

The helper and eighteen existing predicates (eleven memory/cell and seven
decoder) now use the shared finite check. Every existing call site, surrounding
shape/device/dtype/history check, mask and error string remains. The temporary
NumPy view is explicitly read-only; its flag does not alter tensor mutability.
Negative/conjugate view conversion failures use Torch instead of resolving or
copying the view. The helper has no cache, retained array or neural update.

The single paired comparison completed in 10.105 seconds. It used the exact
existing runner lesson, including custody hashes and snapshots, 32-byte output
cap and 0.01 teacher rate. Baseline source was frozen v2 commit
`ea2a2954e700c0bff9095a1ee6269bdfb3031be5`; candidate was the working source.

| Arm | Baseline median | Candidate median | Speedup |
| --- | ---: | ---: | ---: |
| Tagged | 199.695 ms | 151.527 ms | 1.318x |
| Aggregate | 79.206 ms | 65.490 ms | 1.209x |
| Writer/maps frozen | 83.070 ms | 53.527 ms | 1.552x |
| Fully frozen | 76.664 ms | 48.576 ms | 1.578x |
| Writes disabled | 101.575 ms | 55.858 ms | 1.818x |

The sum of five medians falls from 540.210 ms to 374.978 ms (1.441x). All
90 disposable lessons matched every returned record except elapsed time,
including prediction bytes/EOS, losses, update flags, before/after and memory
hashes, and complete resulting parameter tensors. Exposure was 5,760 document
byte/terminator steps and 900 teacher byte/EOS events; none of these models is
reused for learning evidence.

Combined memory/decoder/composition/admission QA passed all 62 tests in 4.55
seconds. An additional six affected tests passed after extending fallback
coverage to conjugate views and the existing masked-input fixture to include
both NaN and infinity. Ruff and diff checks passed. Covered boundaries include
read-only NumPy zero-copy addresses, requires-grad input without gradient
mutation, finite extremes, empty/strided/overlapping/expanded tensors, negative
and conjugate views, unsupported dtype/layout/meta device behavior, and exact
conversion-failure fallback. Existing overflow atomicity and restored-state
tests pass. An initial fallback-test failure was an unescaped regex in the
test's comparison of Torch's sparse error text; exact error-string comparison
corrected it without changing the helper.

Receipts:

- Doctor: `.pytest-tmp/dev/doctor-yrs65is6/receipt.json`.
- Combined tests: `.pytest-tmp/local-finite-focused-v2.xml` (62 passed).
- Final affected tests: `.pytest-tmp/local-finite-focused-final.xml` (6 passed).
- Timing source: `.pytest-tmp/local-finite-validation-compare.py`.
- Timing result: `.pytest-tmp/local-finite-validation-compare.result.json`, SHA-256
  `4ec428186f7ba320690a9f559f5aca9fa01af33f868a7ad040987cb42b4c862b`.
- Measured memory source SHA-256:
  `d2df1e57d3fda87a70fc662d536cc83472e56d3576f9604914a9a87d0f075d30`.
- Measured decoder source SHA-256:
  `2c7ce9876828a9f253bc017c29200e63d26224378c57a65a28bc55b3e684b967`.

The bounded improvement is useful and exact on these checks. It does not
establish v3 admission: twice 1,024 times the summed candidate medians is already
about 768 seconds before extra capped-prediction/process/finalization charges,
and admission uses maximum sampled lessons rather than medians. A successor
must measure its complete workload under the unchanged ceiling and margin.
