# Owned-payload ordinary QA

All three selected nodes passed on their first launch at commit **`768e15c4a5a460c9bc57b15842f8dc1110d53edf`**. Pytest reported **3 passed in 17.77 seconds**; JUnit records 17.767 seconds. The 120-second/2048-MiB watchdog passed at **18.9013638 seconds**, with **1238.98046875 MiB** maximum sampled owned-process peak sum.

These are ordinary file-ownership, durability and actor-continuation checks for the [selected optimization](persistent-checkpoint-owned-payload-v1.md). They do not restore V7, rerun a broader workload or establish a measured full-workload speedup. This report reads the completed receipts and hashes the held source files; no test, model, action or checkpoint decoder was invoked again.

## Source and environment

The selected checkout's doctor passed in 0.297 seconds. It binds the commit above, the checkout-local Python 3.11.15 environment, CPU device and an empty tracked diff. It also reports two untracked paths: `docs/research/public-observation-sequence-context-v1.md` and `src/humanbrain/direct_request_splice_learner.py`. Thus the receipt establishes clean tracked source, not a completely empty Git status. Repository-configured Ruff passes on both implementation modules, with no findings on changed test lines; unrelated historical test-file findings remain outside this affected review.

| Held source | Verified SHA256 |
|---|---|
| `experiments/r4_integrated/persistent_checkpoint_payload_lease.py` | `512ab80cd643a498a89b2050542c062fd4d92303eb910d6d4eef3dcd87a16dfb` |
| `experiments/r4_integrated/persistent_session_checkpoint.py` | `8901072276c3849c528dd237c5b3e17187fe427e6789488ea6677b101893577e` |
| `tests/test_byte_sequence_decoder.py` | `4e7b816e801e508c5c6dff40329c9333c5b2ed5c31d3232fbbf244046110cbab` |

The only warning was pytest's inability to write its `.pytest_cache` node-ID cache because access was denied. It did not fail a selected node. Stderr is empty. The watchdog samples every 500 ms and explicitly warns that brief children can be missed; the two action receipts independently establish their actual child executions.

## Exact selected exposure

| Existing test-file node | Recorded scope |
|---|---|
| `test_persistent_checkpoint_payload_lease` | Three authored byte files, 13 expected refusals, no models, children, teacher symbols or tensor loads |
| `test_persistent_checkpoint_interrupted_frontiers` | Ten constructed opaque publication/frontier cases, no models, children, teacher symbols or tensor loads |
| `test_persistent_checkpoint_actual_actor_continuation` | Four actors, two tiny ingests, six free calls, two actual STOP children, two teachers/four symbols/six action labels; four commit calls including one injected failure, three publications, one restore with two fixture tensor loads |

The tiny ingests contain two raw bytes and two EOS positions total. The enabled and disabled teacher calls each receive two symbols. The two actual children, PIDs 3028 and 5256, return zero with stdout `300a` (`0\n`) and complete capture. Their saved child durations are 0.0402590 and 0.0344414 seconds. These calls belong to the original ordinary fixture, not this documentation review.

The continuation compares the existing actors with the restored actors: online emits byte `31` then EOS; disabled emits EOS immediately. These controlled outputs check continuation parity, not acquired language ability. Both restricted loads consume newly authored fixture checkpoints; **experimental checkpoint loads are zero**.

## Ownership and failure evidence

The byte fixture records these 13 refusals: an existing write handle; an existing writable mapping after its originating file handle closes; a same-length incorrect publication handoff; in-place write; truncation; deletion; rename; replacement from an unleased file; a new writable mapping; a different path identity; reuse after close; a fresh digest mismatch after post-close mutation; and corruption under the portable full-hash fallback. Windows sharing/access errors reject the prohibited OS operations; explicit integrity mismatches raise `ValueError`.

It records seven lease-open attempts, five acquired handles, five releases, three successfully authenticated versions, a peak of two retained handles, and zero live handles at receipt time. Five real hash passes read 175 bytes. Attempt counters therefore include failed opens rather than silently treating them as absent work. The forced unsupported-backend fixture records full hashing and detects changed bytes.

The ten opaque cases are clean state, outstanding intent, partial journal, attempt awaiting teacher, unpublished payload, unpublished manifest, published HEAD, changed source, corrupt payload and empty execution directory. They exercise parent publication/frontier logic without decoding their opaque payloads or importing a neuronal continuation.

The actor fixture injects a lease-verification failure during the prior-boundary check, **before a new snapshot/payload write**. It preserves the old HEAD and unresolved intent, with no fallback and no additional prediction or teaching. Removing only that authored fixture fault permits the already prepared transaction to commit. This is not automatic production retry or overwrite of an orphaned payload. Three actual publications remain, despite four commit invocations.

Same-store inspection reuses leased identities; explicit fresh inspection streams bytes. An independently reopened store begins without trusted leases. Restore records four fresh payload checks across its initial/final inspections and authenticates the exact fixture bytes passed to its two restricted loads. Source bindings, journal checks, pending-state ownership, model/tape seals and strict numerical restoration remain exercised.

## Traffic accounting, not a speed measurement

The continuation receipt records three protected payload versions, 14 reused payload checks, six prepublication metadata checks and **1,217,749,624 counterfactual avoided hash bytes**. Its first repeated inspection avoids rereading **173,964,232 bytes**, exactly the two fixture payload sizes. These are bytes that the old repeated-hash path would read; they are not actual I/O and are not subtracted from observed wall time.

Actual work remains charged: continuation commit/inspection work reports 98 file-hash passes and **869,875,162 bytes read for hashing**, including source/evidence reads and explicit fresh checks. The separately reopened restoration records 25 hash passes and **521,917,564 hashed bytes**. Initial temporary-file authentication and the protected final handoff remain distinct real reads. The commit-work snapshot is taken while its three owned handles are still live; it is not a post-cleanup handle count.

No comparable pre-change complete workload was run in this QA. The 18.90-second result proves the selected checks fit their ordinary envelope; it does not establish a percentage speed improvement or a counterfactual V7 completion time. The next broader session must measure complete persistence costs with its actual trajectory and retained state.

## Receipt identities

Paths below are relative to `.pytest-tmp/`. Fixture paths share prefix `persistent-checkpoint-owned-payload-v1-fixtures/`.

| Receipt | SHA256 |
|---|---|
| `persistent-checkpoint-owned-payload-v1.launch.json` | `8ac3b43d019def144020848cdf2b3b361fbfab752220196a9cf467840db1b44e` |
| `persistent-checkpoint-owned-payload-v1.stdout` | `404dfaa303627232b69d717a1c0df182358aa47a2793a3769ed0c62424b782a5` |
| `persistent-checkpoint-owned-payload-v1.stderr` | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| `persistent-checkpoint-owned-payload-v1.junit.xml` | `98b3f84fb750f7d3cffae3439780b5d1174c7b11f76c262d87291e542a57bc16` |
| `dev/doctor-8auxx0dl/receipt.json` | `4dbc5d2aa742455ff68442e157866a942c0dbcb782765f5ca12e93d4ed16c41d` |
| `test_persistent_checkpoint_pay0/payload-lease-passed.json` | `3547da86f0fb76589279b14b4e390b63ec88a2fd3b3e21febf2092a69b9363ca` |
| `test_persistent_checkpoint_int0/persistent-checkpoint-frontiers-passed.json` | `eb1ece75d050d923cde2cbf9b8eb2aedddf4c5b3dba6e03b28944792247bac33` |
| `test_persistent_checkpoint_act0/persistent-checkpoint-continuation-passed.json` | `02749c91451b25f788ecb6f5908dbe2237c8d69f62e83270cee1c4fb6d9b700b` |

The monitor additionally binds watchdog source SHA256 `4f26b669442a596cf3023c8d066ef49ff6231732baa575b8c503ad4ae3d198dd`. The saved command selects exactly the three nodes above with `-q -x`, empty pytest `addopts`, a dedicated basetemp and JUnit output. There was no second QA invocation.
