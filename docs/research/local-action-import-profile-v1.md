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
