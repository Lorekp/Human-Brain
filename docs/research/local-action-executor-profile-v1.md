# Localize the qualified executor cost

Selected two-child authored diagnostic, 2026-09-06. The
[completed v3 cost](../../experiments/r4_integrated/local_joint_address_online_v3_result.md)
projects 996.224 seconds against 900. Saved execution intervals cover 99.539%
of acquisition action time, but combine filesystem, subprocess and validation
work. The three ranked explanations are subprocess startup/handling, parent
path resolution/materialization, then parent verification/receipt construction.
No source optimization is selected until this diagnostic resolves that boundary.

Use the existing failed complete cost admissions as the measured baseline. Under
the diagnosing-bugs performance branch, do not repeatedly rerun the costly
learning workload to reproduce it or pretend a two-child sample is a stable
time-admission test. The bounded profile addresses component localization only.

After source review and checkout doctor, execute the actual unchanged
`direct_python_action_bridge_v1.execute_utf8` exactly twice under separate
`cProfile.Profile(timer=time.perf_counter)` instances: first an authored pending
workspace, then its authored corrected counterpart, both using identifier width
1023 and expected value `ab`. Files are supplied by the ordinary Case utility;
no neuronal model, teacher, motor, generated answer or checkpoint load is involved.
The pending child must print `pending`, and the corrected child must print `ab`.
Both use fresh workspaces and fresh qualified child interpreters. Keep exact
flags, bootstrap, environment preservation, raw capture and verification.

Authenticate the existing qualification, launch and unchanged bridge before and
after the pair. Use an ignored one-shot diagnostic script and new exclusive output
directory. Save each actual receipt, raw profile and scalar/caller statistics,
bound source/qualification/profile hashes and exact execution count. Preserve
partial output on failure, stop rather than retry, and require byte-identical
materialized files plus expected raw stdout/stderr and return codes.

Read the single cumulative `subprocess.run` boundary, total executor interval,
per-function exclusive times and caller edges. Never sum nested cumulative
durations. Parent cProfile cannot distinguish child startup/site/bootstrap/
task execution/pipe waiting, and repeated file methods may span materialization
and verification. State those limits rather than invent disjoint phase totals.
Instrumentation changes timing, so this pair provides no uninstrumented speedup,
whole-workload admission or acquired-capability evidence.

This diagnostic does not alter neuronal rules, frozen runner/checkers, exposure,
the accepted-action probe or resource ceilings. A selected optimization must
later pass its affected semantic QA and new complete cost/audit; an identical
cost retry, output caching or persistent interpreter is not selected.

## Completed parent-side diagnostic

Both authored executions passed, with exactly two subprocess calls and two
completed execution receipts. Source revision is
`6bd77f745fcb6cde08e30233366a6e0fee08e56e`; the ignored profiler SHA256 is
`eed61b48d7bc9e9df78482c9dbe1dfc90364388842e958d3246f6bdcb6c2043a`.
The actual executor/qualification stayed unchanged. Doctor receipt is
`.pytest-tmp/dev/doctor-wax1_i90/receipt.json`. The monitored complete diagnostic
took 2.997574 seconds with a 79.61328125-MiB observed tree peak. No neural model,
teacher or motor ran, and no cost/full-learning weights were loaded.

| Profile boundary | Pending | Corrected |
| --- | ---: | ---: |
| Executor cumulative | 79.3027 ms | 72.0005 ms |
| Subprocess cumulative | 75.7120 ms | 67.7562 ms |
| Difference, parent work outside subprocess | 3.5907 ms | 4.2443 ms |
| CreateProcess exclusive, inside subprocess | 9.6408 ms | 4.4032 ms |
| Parent lock acquisition exclusive | 64.6406 ms | 61.7294 ms |

Subprocess accounts for about 94-95% of the instrumented executor. Parent file
path work is too small in these fixtures to be the main next target. Holding
every other v3 projection term fixed, its 96.224-second shortfall corresponds
to about 11.746 ms per replicated action; this comparison is only prioritization,
not a new projection from instrumented timings. Lock waiting does not establish
lock contention or a Python startup cause: the parent waits for reader threads
and the child. Child startup, imports, task execution and communication remain
combined. Select a bounded child import/source review before modifying transport.

Saved directory: `.pytest-tmp/direct-action-profile-v1`. Report SHA256
`f12228b2b69f595d1976d92aa15a1ff48174d7de7e332f12f30c940f7b8de483`;
completion `7d7415fb53e604ec92ca669242d039792d2ee857049d1ffe27ea174a55094897`;
finished monitor `9270bb00f15eba5dc879ee33e591a45648dc34eaf243a641cf1e3c57c7d6d648`.
Raw profiles, per-function/caller JSON, stdout/stderr and complete execution
receipts are bound by the report. No independent workload audit or uninstrumented
performance comparison was executed for these two fixtures.
