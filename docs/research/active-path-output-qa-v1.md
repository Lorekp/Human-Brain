# Active-path output ordinary QA

The two active-path nodes passed on their first launch at
`7be6269409e1fc60ba74f72109564401579afbe6`. This saved-only review inspected their
raw/pass receipts, three execution records, JUnit and watchdog, plus the already
passed incremental-prefix test. No test, model, checkpoint or action was replayed.

The active-path doctor receipt is
`.pytest-tmp/dev/doctor-otvm07c7/receipt.json`; it binds the stated revision and an
empty tracked diff. JUnit records two tests, zero failures/errors/skips, 6.261 s.
The whole monitored launch took 7.6734305 s with sampled owned-tree peak
324.58984375 MiB, within its 60 s / 2048 MiB bound. Brief child peaks can escape
the 500 ms sampler; the figure is not an exact instantaneous memory maximum.

## Actual exposure and verified behavior

| Scope | Readers / ingests | Free calls / decisions | Entered reader teacher loops / symbols | Real children |
| --- | ---: | ---: | ---: | ---: |
| Standalone clamp/custody node | 1 / 1 | 4 / 8 | 4 / 9 | 0 |
| Combined execution/STOP/ack node | 1 / 1 | 4 / 7 | 1 / 2 | 3 |
| Total | 2 / 2 | 8 / 15 | 5 / 11 | 3 |

The two ingests contain four raw bytes and two EOS symbols total. There are two
fixed output-cell modules, one combined model and one action circuit; no prior
or file-checkpoint loads. Three standalone snapshot restores and one combined
in-memory fixture reload are disclosed QA controls. Of the 11 teacher symbols,
two were fully processed before an intentionally injected final rejection; nine
belong to successful reader corrections. Four active output clamps were offered
and two published. There are four actual action-head labels: three for SPLICE
and one for corrective STOP.

The standalone raw journal has 22 records and matches the pass receipt's raw
hash. All four actual responses emit `a`, then EOS through addresses 0 and 2.
For target `ab`, the first error is the recorded EOS at position 1 with its own
prefix `a`. The passed independent scalar oracle checks all 75,044 U entries
against the local outer-product update and checks the resulting currents at
that same retained feature. All teacher conditional calls use the original held
U. Exact target `a` offers no U clamp; disabled target `b` computes the offer but
publishes nothing. The final injected rejection follows two B/C offers and
preserves every live group and the same pending response, which is then
acknowledged without a forward or teacher.

Ten validation-only rejection cases cover substitution, request/feature/current/
step/tape changes and pending snapshot/ingest/load. One additional over-cap
`teach` invocation rejects before any neuronal forward or teacher-symbol offer.
All 11 raw rejection records preserve before/after model and tape hashes and the
same pending object. They are additional API calls, not additional entered
teacher loops. In this empty-query fixture B has zero effective update despite
its enabled teaching group; C changes in the successful enabled cases. Do not
confuse an enabled group with a changed tensor.

The combined node makes three action proposals and one standalone cap-zero
reading call. Two invalid combined teaching calls reject before head/reader
publication. Its first completed action is followed by a SPLICE correction
with two reader offers and one U clamp. Corrective STOP offers only its operation
label: zero reader symbols, zero U/B/C offers, one nested acknowledgement. The
third action is acknowledged with no teaching. Both nested and combined pending
tickets are resolved.

The three distinct actual child PIDs are 44372, 41300 and 33244. Each executes the
authored unchanged `main.py`, exits 0, writes bytes `610a` (`a` plus newline), has
empty stderr and complete capture/files. The free proposals are authored STOPs;
the receipt named `splice-execution.json` means the subsequent correction is
SPLICE, not that a learned splice was executed. Every command uses the qualified
base Python flags `-I -S -X utf8 -B -c` and temporary-module bootstrap SHA256
`7213881ba88871e024789913aa742bba520cc79520b8a7b87292c0d3b2b745b9`.

## Incremental-prefix check

The separate `test_incremental_prefix_full_capacity_exact_activity` passed:
512 authored candidates, 17 prefix positions, one cache construction, 16 valid
advances, three rejected advances and 18 feature calls. Its comparison makes
68 frozen full-kernel calls (four blocks at each position). It checks bit-exact
features and identical held-weight scores, including key-copy and invalid-input
behavior. This test constructs no neuronal model and offers no teacher/action.

Actual incremental work records 786,432 lag pairs, 14,942,208 thresholds and
42,467,328 fixed input applications. Held storage is 294,912 key bytes, 524,288
Boolean-match bytes and 2,048 count bytes. JUnit records one passed node in
3.997 s; the monitored launch is 5.0998744 s and 263.70703125 MiB. These checks
establish the selected arithmetic and lifecycle mechanics, not acquired language
or coding skill, sequence generalization, or a measured end-to-end speedup.

## Evidence hashes

All paths below are under `.pytest-tmp`; test receipts are inside their named
`*-temp` directory. Reader source recorded in the raw journal is
`75c3584127577ebca6099e3f24d8b53fc855e9377a8732499a7cbce643706aee`.

| Artifact | SHA256 |
| --- | --- |
| active-path reader raw | `a4487e540ebe61f8c34363699a8261a5ab564d0501c33047055df437c87ef8f3` |
| active-path reader passed | `c6aabb0fcc004226ad9f24ba64f104c4a1222abb3018b086033aa08ebef25243` |
| combined lifecycle | `1877a89a2a911e532e587d51bc6813c1d1d6aea07f2af80f21b4461d4409fd4e` |
| splice execution | `e5518eb7c09c8d5175795d2b97516e96ec8b68ba3ddaaad865c6d1bc05d5e1cd` |
| STOP execution | `ef2b32378d983524d306cd2d8a7d885a3ef2c702aa3dc11919f8cd1bb0c2b05f` |
| acknowledgement execution | `5a030b4f0760a3c091f3c2f90248484751b64ca93de905ef0120668f33545c35` |
| active-path JUnit | `03a475748836c5ace2490895f7fad9d212caf4bd66c590177bddabfcef102a2a` |
| active-path watchdog | `87c275e8f8bc5f6af0bb881a7305953a85b00cf70e5f1b2f51431fbe1dd3119e` |
| active-path doctor | `b345605bb7527ca6946b5b5391e8d1887f22b8135d47fd1b445ee08f0b22044d` |
| incremental-prefix parity | `7936e7c24e15c2db1c409bd066c13bdca70f066419f93a772253da27c1991acb` |
| incremental-prefix JUnit | `f00bf880b9aaf225925ab84e9e53ef921662e636971f467950af76290b2048b2` |
| incremental-prefix watchdog | `de059438b42a43b50c06aec7bb55ae47f9847e932e36c57b862e2affa7692a37` |
