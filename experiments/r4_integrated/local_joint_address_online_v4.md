# Joint-address online v4: regular Python source execution

Selected engineering return, 2026-09-06. Inherit the complete
[v3 contract](local_joint_address_online_v3.md) at
`7521d52949a5dfc3335813e3c5b5423e7ec43fd5`. Its audited cost rejected the
996.224-second final projection; no full fit ran. The
[parent](../../docs/research/local-action-executor-profile-v1.md) and
[child import](../../docs/research/local-action-import-profile-v1.md) diagnostics
select a smaller regular-source bootstrap, not a new learning architecture.

## Selected execution boundary

Use a new `source_python_action_bridge_v1.py` with the unchanged direct-runtime
launch identity, environment preservation, flags, UTF-8 stream setup, working
directory, timeout, output capture and complete file verification. Read the
ordinary materialized `main.py` through `io.open_code` at its absolute path,
compile its bytes with filename `main.py`, then call the qualified installed
`runpy._run_module_code` with `mod_name='__main__'`, `pkg_name=''` and
`script_name='main.py'`. This retains the existing temporary main module, argv
replacement/restoration and script-global initialization while skipping general
path/zip/bytecode dispatch and its `pkgutil` import. Site processing stays enabled.

The supported entry point is a regular Python source file. Do not claim parity
for directories, zip entries or sourceless bytecode. Byte compilation preserves
source encoding cookies and BOM handling. Internal bootstrap imports, audit
events and traceback wrapper frames can differ; preserve complete raw error
output and compare the meaningful error/exit/file behavior declared below.
There is no universal observational-equivalence claim for introspection of
bootstrap internals. Every action still starts a fresh interpreter.

Reuse the authenticated runtime qualification
`03a33ae6af0488727f7b165346646e89aed801f6f61be93dd0e8b60ca6534dc5` and
launch identity `17777a200f0a9f98fa4bcaf14402ef4d5381451dbcc9dec86290bd7f0ec7b7e6`.
Add separate bootstrap policy/hash and a seven-pair semantic qualification
manifest, bound to the new bridge, bootstrap, existing runtime and exact saved
QA evidence. Do not repeat the earlier runtime identity/output qualification.
The immutable launch object and execute/actuate signatures remain unchanged.
Bind the bootstrap policy/hash in each action receipt, config and admission;
bind the separate qualification path/hash in `action_execution`. The source-free
request adds the bootstrap hash before its sole exclusive publication and the
worker echoes it; these metadata additions do not enter neuronal computations.
Authenticate fixed artifacts at preparation and completion, counting that work.

## Authored semantic qualification

After source review, Ruff and doctor, run seven fixed old/new execution pairs
in the existing test home: fourteen real child executions, zero neuronal models,
teachers or motor calls. Fix expected criteria before running. Cover temporary
main namespace and argv, registered main-module identity and exit cleanup,
sibling imports, source BOM/cookie handling, syntax/runtime errors, explicit
SystemExit and invalid UTF-8 output, and the pending path. Cases may combine
compatible observations. Keep each pair's supplied source files identical and
compare raw files/output/return behavior except prospectively declared wrapper
traceback differences. Save all raw pair receipts before assertions and preserve
failures; do not change expected semantics based on an observed mismatch.

No backpropagating reference, neuronal warm start, motor demonstration or cost
workload is needed to qualify this bootstrap. Reuse earlier local arithmetic and
same-instance/control QA only for unchanged source. Version the independent
auditors, retaining all scientific/resource clauses; add the new bootstrap
qualification/command/metadata checks and only their narrowly affected pure QA.

## Unchanged scientific and resource workload

Keep the actual X-before-B local neuron rule, marginal recall, seed 733, rate
.01, four arms, fresh weights, 1,024 lessons per arm, all documents/queries,
teacher schedule, same-instance action-before-supervised-feedback order and
twenty final cases per arm. Main source remains
`4a4aa24f14f30705cb8a83a0954291ed86e7e4473744920775d3a7961d770ccc`.
There is no architecture or accuracy gain predicted by this transport change.
Preserve all older bridges/runners/results and their original evidence.

After reviewed source freeze and fresh preparation, run one new complete cost:
sixteen first attempts/actions, 48 teacher byte/EOS offers, eighty final actions,
two supplied motor demonstrations, one supplied accepted-action probe at ordinal
1023 and four cap predictions. Count 99 execution receipts and discard cost
weights. Keep all disjoint M/A/J timings, accepted-probe guarded maxima, cap
allowance, finished independent-audit row timing, residual terms and the complete
33,801,664-byte endpoint payload. Bind exact commands, source/qualification hashes
and completed owned-process monitors before final admission.

Only a final independently checked projection at most 900 seconds and 2,048 MiB
admits the fixed fresh 4,096-lesson run (12,288 teacher offers, 4,178 executions)
and saved-only audit. Their actual aggregate must fit those same ceilings.
No altered rate, exposure, accepted-probe omission, extra epochs or identical
timing retry follows a failed admission. The supplied host motor and explicit
teacher supervision remain scientific limitations, separate from local neuronal
answer generation and the long-term exclusive-neuron software goal.

## Completed pre-cost verification

All seven authored pairs passed on their first run: fourteen real child
executions, zero failures/skips and 6.12 pytest seconds. No neuronal model,
teacher or motor ran. Doctor receipt:
`.pytest-tmp/dev/doctor-qi7kg81_/receipt.json`. The exact tested source is
preserved at `.pytest-tmp/source-python-bootstrap-v1-tested-source.py`, SHA256
`64f0992d1fca709d57d48c2522efc1d644b6c2751642d50606c6133cb374ceff`.
JUnit `.pytest-tmp/source-python-bootstrap-v1-semantic-qa.xml` has SHA256
`6091f10514114248d1f1df5b60d4038df1e5a3ab45f78705dadaa0a953e3b57e`.
Ruff passed before and after execution with no subsequent test-source edits.

The separate qualified bootstrap manifest is
`.pytest-tmp/source-python-bootstrap-v1-qualification.json`, SHA256
`063550945a3c426225d9cc2d7284eb7849343bc42353d16762b7d3d2da7d82e1`.
It binds seven passed pair receipts, seven raw pair receipts, JUnit and the
archived tested source. The reviewed saved-only combiner completed once in
1.5932371 process seconds, with no new children or neuronal calls. Its resulting
manifest passed the actual loader, preserving the older runtime qualification.
Bootstrap SHA256 is `9d4239f54f38a520d3da269be86a3f3f58c901be92b9f05d0a66b6ad4bd11c7e`;
bridge SHA256 is `e3c807404dc48526650284b72378b305476118fb000a7edcb0c0c06d3676a93b`;
runner SHA256 is `af83519be79a9031fb4e3db6dfcac92e6f0d41a45b3ec91e78e3da37a7341705`.

Eleven source/AST checks passed without project execution. Receipt
`.pytest-tmp/joint-online-v4-source-parity.json`, SHA256
`61a0fbee4a98d70972583cb201c8116c6f5914fc26716429afbe3ca7a7f0174b`,
compares unchanged lesson, probe, final-action, actuation and main bodies;
preparation changes only the selected bridge import, export/worker add only
bootstrap metadata, and run changes only loader/custody sections. Neuronal
arithmetic and the complete resource formula remain unchanged.

The independent checkers passed all sixty narrowly affected authored helper
checks on their first attempt, with zero failures and 0.1928718 seconds inside
the helper. They performed no model, child, real-qualification or checkpoint
execution/read. Existing unchanged 134-case coverage was not repeated. Helper
receipt SHA256 is `dbb479bae1499190ed15c56df49f2c9d79897c0955e800730cdebfd6c1dacf0c`;
readiness `.pytest-tmp/local-joint-address-online-v4-readiness.json` is
`9043f3326b7e92f233c5ff557b6d37b92d755bfb5059ca5972362962d15b1782`.
Frozen cost checker SHA256 is
`03dcabbc233c2b66ba4e767b468e47505b23fccc257be92a3fd4123f6d9a518e`;
run checker is `4ef87dd7e77de41a488738fb3ea6c77c332c38fd74bb4125152a75e626c31306`.
Only the DRAFT guard changes after helper execution; exact tested raw files are
retained. The additional audit-body source review is SHA256
`c7cd405c96a1be1f536abbbc329cf579a60286851d02b0f93c7c9d02dac4c02d` and limits
changes to version/dependencies/bootstrap custody/request metadata and scope.
