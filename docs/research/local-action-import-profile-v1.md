# One authored child import profile

Selected inner diagnostic, 2026-09-06, after the
[two-child parent profile](local-action-executor-profile-v1.md) localized about
94-95% of executor time inside subprocess invocation. No bootstrap optimization
or altered acquisition is selected yet.

Run exactly one fresh corrected authored Case (`qa1023`, `qa_config1023`, value
`ab`) using the qualified base executable, copied environment with the original
venv launcher override, exact existing flags and bootstrap, plus only the
diagnostic `-X importtime` option. Materialize the ordinary three supplied files;
there is no model, teacher, motor or generated answer. Preserve raw stdout and
all raw stderr, exact argv, fixed source/qualification identity and file hashes.
Require return code zero, stdout `ab\n` and unchanged materialized bytes.

Use CPython's [documented import timing](https://docs.python.org/3.11/using/cmdline.html#cmdoption-X),
which distinguishes each import's self time from cumulative nested time. Parse
the exact timing lines while retaining their originals and indentation. Do not
sum cumulative rows or interpret absence as zero cost: modules already loaded,
non-import work, startup before timing and output overhead are outside such a
simple sum. The fixture creates no threads; the documented multithreaded-output
limitation remains noted. Timing output itself perturbs this execution.

The additional option changes `sys._xoptions` and stderr. This run is explicitly
a diagnostic child command, not an unchanged-executor qualification or a
performance admission. Read the site/startup and later runpy/pkgutil subtrees
to determine whether regular-source bootstrap specialization is worth pursuing.
Keep site processing, fresh-process lifetime, verification and isolation intact.

Use an ignored one-shot script, fresh exclusive output directory, source review,
doctor, a 60-second/2,048-MiB parent watchdog and the ordinary ten-second child
timeout. Count one subprocess invocation and returned receipt separately;
preserve any partial failure and stop without retries. Authenticate qualification,
launch and source hashes before and after. Existing full-workload ceilings,
neuronal rule, exposure and accepted-action probe are unchanged.

## Completed diagnostic and selected return

The single child passed on source revision
`9d359280e872c07708cde9b25b1f9fa645f5c172`: one invocation, one returned execution
receipt, expected `ab\n` and unchanged files. There were no neuronal, teacher
or motor calls. The complete monitored diagnostic took 1.8597486 seconds with
a 79.12109375-MiB observed tree peak. The diagnostic subprocess interval was
0.068990999993 seconds; it is not an uninstrumented performance estimate.
Doctor: `.pytest-tmp/dev/doctor-qhxm1vw2/receipt.json`. Profiler SHA256:
`7d26f211b5444babbf44b6b7c41cbf242e0d3123a057a06d2e65097a82a37930`.

Three separate top-level import rows report cumulative times of 16.009 ms for
`site`, 5.684 ms for `runpy` and 17.000 ms for `pkgutil`. The latter includes
`typing` at 14.769 ms and its `re` subtree at 9.803 ms; these nested values must
not be added to the 17 ms. `_virtualenv` at 9.591 ms is inside site. Ordinary
task module `qa_config1023` took 0.280 ms. The retained 62 import rows and raw
stderr support prioritizing the general `pkgutil` loading path. They do not
predict an exact 17-ms saving or prove a full-workload speedup.

Select the [v4 regular-source bootstrap](../../experiments/r4_integrated/local_joint_address_online_v4.md):
read source bytes with `io.open_code`, compile them, then reuse this qualified
CPython's `runpy._run_module_code` for temporary module/argv lifecycle. Keep site,
runtime, fresh-process lifetime, neuronal rules and the full workload fixed.
The alternative public-only implementation would need to reproduce that
lifecycle manually. The selected private helper requires explicit source/runtime
binding and authored semantic qualification; it is not a general public API.

Saved directory `.pytest-tmp/action-import-profile-v1`; report SHA256
`c572341d64ad359c48941ed2083247692b43344926fec676d5c96c5dcc39d077`;
completion `64c677abe09317e6adf6a5285361fb217c56a4561a2be05997c2f4812553e469`;
finished monitor `b46076b3ded6138f160a77cc2fdffa2740ac64ff620f92be2a52cc53a6cc05e2`.
Raw stdout/stderr, parsed rows and exact files are bound in the report. No
additional child or repeated profiling sample is selected.
