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
