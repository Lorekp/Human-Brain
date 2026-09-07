# Assembly validation within an owned response

Source `3f504e3e320ad5b7138f03eadb7c73669ead2405` validates the assembly bank
once on entry to a synchronous free-response call. Its loop uses those same
validated tensor references through a private read kernel. Public `evaluate`
still checks inputs and the bank, and teacher-stage handling is unchanged.
Final full model/tape hashes and new object/buffer identity guards run before
response publication. No cached validity survives the call.

The entire agreement/competition/current-read body is unchanged. Work counters
drop the removed per-event validation scans, preserving actual neural work.
On the completed v4 trajectory, the source arithmetic removes 10,484 scans and
328,828,531,090 validation values: 78.28% of the counted validation population.
This is an operation-count projection on saved inputs, not a measured speedup
or a rerun of the frozen v4 experiment. Neither hashes nor neuron gates are
removed. Arbitrary concurrent mutation is not supported; persistent injected
changes are rejected before publication, with potentially later error timing.

## Ordinary affected QA

Three existing nodes in `tests/test_byte_sequence_decoder.py` pass on their
first monitored launch: local assembly staging, reader lifecycle, and combined
reader/action custody. Ruff passes for the two touched modules and test file.

The reader fixture compares forty actual empty/occupied-bank events against
the fully checked public evaluation path. Currents, agreement counts, selection
and all neural work counters match exactly. Each completed response records
one full entry validation; the omitted event scans total 1,362,917,100 values
in this fixture. Two additional three-event attempted responses inject a
persistent value mutation and equal-valued buffer replacement. Both fail before
pending state or experience publication. The deliberately changed fixture is
restored after each rejection.

This launch constructs one standalone bank and two readers, one inside the
combined actor, using authored small tapes and no prior checkpoint. The reader
has three completed responses/forty events, forty checked reference evaluations,
two rejected free calls/six events, and the existing seven entered teacher
events/six completed events. The combined fixture retains its three real STOP
executions and late-publication failure/retry checks. No broader task, extra
training round or capability threshold is introduced.

Pytest reports **3 passed in 12.35 seconds**. The external guard reports
**13.6605048 seconds and 639.046875 MiB**. A nonfatal pytest cache-directory
permission warning occurs; JUnit and retained fixture evidence are written.
These ordinary QA timings are not compared with earlier fixtures as a speed
experiment. No native source changed.

| Receipt | SHA256 |
| --- | --- |
| `.pytest-tmp/dev/doctor-hzoc_qw9/receipt.json` | `1b29b9b37d5ece5169e53573a380c5a51c886c42e9f5080ac22e7231134ac732` |
| `.pytest-tmp/sequence-assembly-owned-response-qa-v1.launch.json` | `71464d35fa031ef264c79d05a75fcfa01b98e75f2cc09195d6b73e5ddcbdacd1` |
| `.pytest-tmp/sequence-assembly-owned-response-qa-v1.junit.xml` | `45a1c7061feb93f6a311b5fa08b8d11531747d2639dff09ffd93b1ed40dcf527` |

Fixture records are under `.pytest-tmp/sequence-assembly-owned-response-qa-v1-tmp`.
The source proof is `.pytest-tmp/sequence-assembly-custody-optimization-options.md`,
SHA256 `f65a22ddc213c761028dea31f7e11015aec7d761cb3a53a1fd49ee48b10ad1fb`.
The v4 bank and its original source/reviewer remain unchanged. A successor
reviewer must account for the new validation scope in its own released source.
