# Fixed raw roster specification: neuronal splice v1

Prospective source specification only, for root's single selected integrated comparison. No roster implementation, model, checkpoint load, forward, action or acquisition has run. Freeze the generated raw roster and its hash before cost. Do not choose examples or add rounds after seeing outputs.

## Counts, task meaning and common state

Four arms: learned, action-frozen/reader-plastic, all-slow-frozen, retained-memory-disabled. Each starts with the exact completed v6 learned reader's slow tensors and the same fresh action arrays. Mode metadata is explicit; identical numerical initialization does not mean identical publication gates. No old endpoint tape is imported. Each arm constructs its own mode-appropriate raw-document states once and reuses those immutable states; disabled memory has zero keys/content in every state, including reading-retention states.

Per arm: 32 TRAIN tasks in eight rounds =256 actual first attempts; then 32 fresh tasks and16 TRAIN retention tasks =48 action endpoints without teaching. Exactly one real workspace verification follows every attempt, including explicit STOP or a rejected splice. STOP/rejection verifies the unchanged workspace, never a substituted correct edit. Total full-run verifications:4*(256+48)=1216. No semantic motor or motor demonstrations.

SET means replace the content of the explicitly named variable's single-quoted literal in the explicitly named file with the raw fragment supplied by the queried retained fact. KEEP explicitly means leave the workspace unchanged. These are narrow authored instructions, not general language or inferred module roles. Every head and the reader receive the same bytes `SET <path> <var> <key>` or `KEEP <path> <var> <key>` with single ASCII spaces. No host extracts a memory question, path, key or variable for a neuronal input. All paths still compete and every raw boundary remains eligible.

The ordinary fixture generator knows its source construction and corrective tuple. This teacher/evaluator data stays outside free encoding and the executor. The correct path/start/end/payload becomes available to teaching only after the real action receipt is published. The executor receives the actual proposal and workspace only; expected files/stdout are used by the external grader after execution.

## Exact byte constructors and payload tables

All following text templates are ASCII bytes, with LF only. Relative paths have no directory component. For each task, two data files plus `main.py` are supplied. Given variable v and raw interior s, a line is `v + b" = '" + s + b"'\n"`. A data file contains exactly the x and y lines, in the specified order, with the specified prefix. Both files have identical initial variable interiors and layout; the path request must distinguish them.

Layout A: prefix empty, lines x then y. Layout B: prefix `b"# pad\n"`, lines y then x. Fresh layout C: prefix `b"# fresh 0123456789\n\n"`, lines x then y. Fresh layout D: the same prefix, lines y then x. The teacher interval is precisely the requested variable's interior: start = prefix length + lengths of preceding complete lines +5; end = start + length of its original interior. These construction offsets belong to labels, never candidate masks.

For data paths `<a>.py` and `<b>.py`, main bytes are exactly `import <a>, <b>\nprint(<a>.x, <a>.y, <b>.x, <b>.y, sep='|')\n`. Its verification observes all four variables; collateral changes cannot hide in an unused file. File observation order may vary as specified; canonical ordering is the existing raw-path ABI. No fixture code uses `Case.files`, occurrence selection, syntax repair or automatic escaping.

Payload arrays, given as raw hex to make backslashes unambiguous:

| r | R[r] | I[r] |
|---|---|---|
|0|6162 (`ab`)|6161 (`aa`)|
|1|6261 (`ba`)|6262 (`bb`)|
|2|6263 (`bc`)|6363 (`cc`)|
|3|6362 (`cb`)|6464 (`dd`)|
|4|615c2762|6162 (`ab`)|
|5|635c2764|6261 (`ba`)|
|6|6364 (`cd`)|6364 (`cd`)|
|7|6463 (`dc`)|6463 (`dc`)|

R[4]/R[5] are the four raw bytes `a`, backslash, apostrophe, `b` and `c`, backslash, apostrophe, `d`. Their exact bytes occur in the document and must be emitted unchanged; Python prints the three-byte values `a'b`/`c'd`. This is source-fragment copying, not learned escaping. Define P[0:8]=R, P[8:16]=I, P[16:24]=empty bytes, P[24:32]=I.

Define `D(key,payload) = b"Record " + key + b" has value " + payload + b"."`. No source string normalization or decoding occurs. An empty-payload fact is the literal document ending `has value .`; deletion teaching offers EOS alone. Empty payload is not a learned answerability claim and need not be memory-dependent.

## Exact TRAIN allocation and schedule

For t=0..31, task ID is T followed by two decimal digits, key is `k` followed by the same two digits, r=t modulo8. The two data paths are `a0.py`,`b0.py`. Choose target file `a0.py` when r is even, otherwise `b0.py`; choose x when floor(r/2) is even, otherwise y. Choose layout A for r<4, B otherwise. Thus every group of eight crosses both files, both variables and both layouts once.

| IDs | Required operation | Before interior in ALL four variables | Correct raw payload |
|---|---|---|---|
|T00–T07|SET / SPLICE replacement|`old`|R[r]|
|T08–T15|SET / SPLICE insertion|empty|I[r]|
|T16–T23|SET / SPLICE deletion|`old`|empty|
|T24–T31|KEEP / STOP|`old`|no payload teacher|

Each task owns one two-document state M-Ttt. Its documents are D(ktt,P[t]) and D(k((t+9) modulo32),P[(t+9) modulo32]); reverse their order when t is odd. Decoy keys use the SAME k-prefix and real target-value distribution; no special decoy role code or always-irrelevant value is provided. Source identity and document order are frozen. Each state is ingested once before acquisition, never again during eight rounds. The same task's raw request/files/state recur; teachers do not repair the next task's starting workspace or alter its retained tape.

For round j=0..7, traverse t values `(i+4*j) modulo32` for i=0..31. This is a fixed cyclic rotation, not outcome-driven scheduling. Every task is seen eight times, with all four groups in each round. Arms execute the same order. Count first attempts separately from their corrections. Per arm there are192 SPLICE corrections and64 STOP corrections. The payload teacher offers per32-task round are sum(len(R))+sum(len(I))+24 EOS =20+16+24=60; eight rounds give480, all arms1920. Direct action labels are3 per SPLICE and1 per STOP:640/arm,2560 total.

The four fixed ordinary cost lesson IDs are T00,T08,T16,T24 in that order. They cover replacement length2, insertion length2, deletion length0 and explicit STOP. These offer7 payload symbols and10 action labels per arm. Separate maximum-feature/payload and transport category probes remain the cost protocol's declared engineering fixtures, not additional acquisition examples.

## Thirty-two fresh tasks: sixteen mandatory contrast pairs

Order final fresh endpoints by category P,L,V,O, then r=0..3, then member0,member1. No fresh endpoint receives feedback. Their keys are disjoint from all TRAIN target/decoy keys. No fresh path occurs in TRAIN. Every pair must pass both members; changing an ID while preserving the same required action is insufficient.

**P: path binding, eight tasks.** Pair r uses paths `a{r+1}.py`,`b{r+1}.py`, target variable x for even r/y for odd r, layout C for even r/D otherwise, all interiors `old`. Key k(32+r), payload from `[ac,ca,bd,db]`, decoy key k(52+r) with reversed two-byte payload. One shared two-document state M-Pr, target first for even r and second for odd r. Both members share workspace bytes and fact state; member0 requests the a-file, member1 the b-file. Supply file observations in reversed order for member1, although canonical indexing must erase enumeration order. Both raw paths must be selected correctly.

**L: boundary/variable binding, eight tasks.** Pair r uses `m{r}.py`,`n{r}.py`; target path m-file for even r/n-file for odd r. Key k(36+r), payload `[ad,da,bc,cb]`, decoy k(56+r) with reversed payload; one shared two-document state M-Lr with parity-based document order. All interiors `old`. For r=0,1, request x in both members; member0 uses layout C and member1 layout D, so the correct byte offset moves. For r=2,3, both members use layout C; member0 requests x and member1 requests y, so identical literal interiors require different occurrences. These are explicit-variable selection tasks, not inference of which duplicate assignment is semantically active.

**V: retained fact assignment swaps, eight tasks.** Pair r uses `p{r}.py`,`q{r}.py`, target p-file, variable x for even r/y for odd r, layout C for even r/D otherwise, all interiors `old`. Both members have the identical request `SET p{r}.py <var> k{40+r}` and identical files. Their distinct states M-Vr-0/M-Vr-1 contain the same two keys k(40+r),k(60+r) in that fixed order, but swap their payload assignments. Payload pairs are `(ac,ca)`, `(bd,db)`, `(625c2761,645c2763)`, `(615c2764,635c2761)`; the latter two pairs are raw hex fragments, not literal hex text. Correct emitted payload must reverse with the assignment. States for r=0,1 are ingested BEFORE acquisition and untouched through all256 lessons; this supplies real intervening learning delay, without uncounted distractor queries. For r=2,3, append three documents with keys k(68+3*(r-2)+j), j=0..2, and payloads `[cc,cd,dd]` in that order. These five-document states are ingested after acquisition and test interference. At most124 tape slots including EOS are used. No WTA policy change or host selection of the target document is permitted.

**O: explicit operation contrasts, eight tasks.** Pair r uses `u{r}.py`,`v{r}.py`, target u-file for even r/v-file for odd r, variable x for even r/y otherwise, layout C for even r/D otherwise. State M-Or contains target key k(44+r) with payload `[ad,da,empty,625c2763]` and decoy k(64+r) with payload `[da,ad,ab,645c2761]`, parity-based document order. All interiors are empty only for r=1; otherwise `old`. Member0 says SET and must respectively replace, insert, delete or copy the quoted fragment. Member1 says KEEP on the identical workspace/state and must STOP. The empty deletion and nonempty KEEP cases explicitly separate operation from payload EOS/length. KEEP does not mean “infer that the current workspace already satisfies an unstated goal.”

These fresh categories give28 required SPLICE and4 required STOP tasks. The reader's changed request syntax, same-class path/key matching and empty/EOS or quoting outputs remain genuine learning risks. The roster does not claim its feature rankings are universally realizable.

## Retention, old-reader panel and lifecycle

The16 action-retention endpoints, after the32 fresh endpoints, reuse T00–03,T08–11,T16–19,T24–27 in that order, with their original raw requests, before-files and original M-T states. There is no new ingestion or teacher. Twelve require SPLICE and four STOP. These are early-task retention, not fresh transfer.

Root additionally selected the exact20 old-v6 raw-question reader-only cases before AND after acquisition in every arm:160 actual `predict_with_receipt` calls in total, max_bytes=2, zero actions/teachers. Bind `.pytest-tmp/local-joint-address-online-v6-run/offline_teacher_roster.json`, SHA256 `009b42e22976733bba207e46337ec23d5f6de5ab15ab061bc6fd08539e44a2b1`, `checks` IDs0..19. Preserve each row's raw query/expected bytes and full ordered `document_hex`. Construct one new state per distinct memory_id: fit-0..3, novel-0..3, interference-0..3. For interference states ingest their complete five-document lists directly once; do not both ingest complete lists and append `additional_document_hex`. No original checkpoint tape is loaded. The same12 states are reused pre/post without writes.

The panel contains exactly20 predictions per pass; original delay_reads fields do NOT trigger extra reads here. IDs12–15 repeat the original raw questions/states as fixed retention observations; their original delayed labels do not establish a newly reproduced32-read delay protocol. Preserve case-wise correctness on the18 original successes (all IDs except16,18), and report all20 including the two known failures. Memory-disabled constructs zero-key/content versions rather than importing populated reference tapes. Report that control separately rather than demanding an impossible common populated initialization.

Per-arm lifecycle: copy common reader tensors and construct fresh action weights; ingest32 TRAIN states, four delayed V states and12 old-panel states; run old-panel pre pass; execute256 lessons; ingest the remaining16 fresh states; run48 action endpoints; run old-panel post pass. The20 fresh states comprise P4+L4+V8+O4. Therefore64 retained states/arm,256 across arms. They are separate episode-associated literal tapes, not one compressed global knowledge store. There are152 source documents/arm: TRAIN64, fresh52, old-panel36. By the byte constructors above the expected ingestion ledger is3490 raw document bytes+152 EOS/arm (13960 raw+608 EOS total); implementation must independently verify these arithmetic counts before freeze.

## Grading and custody boundaries

After every actual free proposal, apply exactly its selected raw splice or record a structural rejection. Execute resulting/unchanged files exactly once. Save raw stdout/stderr/status, before/after file hashes and rejection before external grading. A SET task succeeds only with a nonrejected SPLICE, byte-exact expected final workspace including every untouched byte/file, and the expected successful executable output. A KEEP task succeeds only with explicit STOP, byte-exact unchanged workspace and successful unchanged verification. Thus a successful process printing the stale value is not a solved edit. Failed actions receive correction only in acquisition.

Canonical teacher file/span/payload correctness is a separately reported stricter diagnostic: an alternative valid splice that produces precisely the expected files is a legitimate task success. No task failure is repaired by replacing free output with the canonical teacher. For STOP correction, the ABI's unused tuple fields are inactive placeholders, never start/end/payload labels; operation teaching reconstructs the actual free proposal as selected by the source contract. Raw-fragment versus displayed-string pairs for quoted payloads are explicit evaluator labels; no generic escape routine supplies free bytes.

Report per-category paired success,12/4 retention edit/STOP counts, operation/file/start/end/payload/EOS errors, invalid/rejected/limit outputs, current and late-block first-attempt acquisition curves, and old-panel case-wise retention. Strict useful transfer requires both members of each pair, while correct STOP is reported separately from edited tasks. The absolute old20/20 reader gate remains historically unmet. This narrow tool-learning return does not establish planning, general coding, learned escaping or autonomous multistep recovery. Root owns final acceptance thresholds, full cost/probe schema and execution admission.
