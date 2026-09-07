# Lossless 1024-slot capacity support

Source `53d71890d685f2c9af417ed3580d1b54b90a24f1` adds
`PersistentCoincidenceMemory1024`, `compete1024` and
`RollingPrefixCoincidence1024` through shared private implementations. Existing
512-slot entry points retain their fixed limits and equations. This is capacity
support for the shared successor; no experiment checkpoint has been promoted
and no broader run has occurred with it.

`promote_512_state` validates the complete source, copies all 512 rows and the
write cursor into independent 1024-slot storage, checks exact prefix equality
and validates the zero/-1 new tail. It copies 825,352 bytes and initializes
1,650,696 bytes, without ingestion or new learning. The owning initializer must
authenticate the source and exclude pending responses. Appending uses the
existing local association writes, preserves old occupied rows and leaves each
old document EOS absorbing. Every occupied address remains eligible.

Full competition retains the two threshold stages and global lower-index tie
rule. At 1024 candidates it evaluates 523,776 pair comparators and retains
4,727,296 tensor bytes in its receipt. This is four times the approximate pair
work of 512 candidates; it is not a runtime forecast. Prefix activity uses the
same 32-lag algorithm over all rows. Fact capacity and assembly capacity remain
distinct: growing the former does not enlarge the latter's 512 slots.

Both future controls may receive the same new local factual association writes
while one disables only correction/routing plasticity. That must be labeled
explicitly; such a control does not disable every form of plasticity. No
task-selected tape, eviction, old-state reset or hidden factual exposure is
introduced by promotion.

## Ordinary affected QA

Four existing-file nodes pass on their first monitored launch:

- `test_persistent_coincidence_1024_promotion_and_append` promotes one authored
  505-slot source, appends six documents to 1023, rejects an overflowing append,
  fills the final EOS slot and rejects further growth. All old occupied values,
  empty promotion rows, document boundaries and 1024 recall cursors are checked.
  Three successful ingests contain 1,016 raw bytes and eight EOS events; two
  append attempts reject atomically. No reader, teacher or checkpoint is used.
- `test_persistent_1024_competition_and_prefix` checks five full-score vectors
  from one to 1024 candidates, including all ties, cross-512 winners/ties and
  finite float32 extremes. It compares the old 512 competition exactly and
  checks rejection of over-capacity old inputs and nonfinite scores. Two prefix
  caches make six feature calls and four advances; all 1024 rows agree with an
  independent scalar sparse-key oracle at three causal prefixes.
- `test_persistent_coincidence_512_append_and_reader_regression` preserves the
  existing small-reader parity, full-512 capacity, append and disabled-teacher
  regression. Its authored fixture creates two readers; it loads no checkpoint.
- `test_sequence_assembly_rolling_prefix_exact_boundaries` preserves exact
  fixed-kernel parity at output lengths through 128 and the original 512 bound.

Ruff passes on the touched source and existing test file. Pytest reports
**4 passed in 4.14 seconds**; the external guard reports **5.2339619 seconds
and 289.859375 MiB**. A nonfatal pytest cache permission warning is preserved;
JUnit and retained fixtures are complete. These timings are ordinary QA costs,
not a matched speed comparison or learned-capability measurement.

| Receipt | SHA256 |
| --- | --- |
| `.pytest-tmp/dev/doctor-p2uj8pzf/receipt.json` | `71460b8bfac0ced4b44653bc183b9662fe23793b24645263e0b52917d3e52fbc` |
| `.pytest-tmp/persistent-capacity-1024-qa-v1.launch.json` | `d92095f37ce6c71f9d5b10527dd67590d051efe5c054d7b5a6805e832359bcbd` |
| `.pytest-tmp/persistent-capacity-1024-qa-v1.junit.xml` | `0dcec3d0a416d23fb7b187970e29c57a3190743ea79a67615af84d79a8333a1a` |

Fixtures are under `.pytest-tmp/persistent-capacity-1024-qa-v1-tmp`. Doctor's
tracked diff contains only the evolving roadmap/active-plan documents; the
routing draft and paused direct-request source are untracked and unexercised.
The source-only capacity proposal and exact accounting are preserved in
`.pytest-tmp/persistent-language-action-v5-memory-growth-contract.md`, SHA256
`325de42e57ad9c166628da64a0bfc6b3871a8de337cee088713caa627423172d`.
