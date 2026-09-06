# Typed byte-splice topology v1: concrete source/math proposal

This is the single topology selected under `docs/research/neuronal-byte-splice-action-v1.md`. Its source/math review ran no learner, checkpoint, forward pass, action or training. It is accepted for ordinary authored implementation/QA within the limits below, not for a fit or general coding claim. The operation bias augments its feature energy bound to 33; the bias-free selector bound remains 32.

## Inputs, bounds and representations

One persistent model owns the existing v6 reader/codec and three fresh action output populations. Reuse its fixed 258-by-32 float32 byte-code array, with the existing EOS=256 code; no lexical dictionary, semantic parser or answer-specific code is added. The same raw request bytes feed reader and action paths, with no host extraction of a key or reformatted memory question. Changed request distribution is genuine new learning, not guaranteed transfer of v6 recall.

Fix these bounds: request at most 31 raw bytes; each relative UTF-8 path at most 31 encoded bytes; one to four files, each at most 128 raw bytes; generated replacement at most 16 raw bytes plus its actual EOS decision. Reject out-of-bounds observations before activity; do not truncate. File contents may be arbitrary bytes. Paths retain generic workspace confinement validation. Sort file observations by their raw encoded path for deterministic indexing; expose every raw byte boundary including EOF. At maximum there are B=4*(128+1)=516 file/start candidates. No token, AST, quote, identifier, plausible-value or correct-span filter exists.

The first action slice explicitly names its target raw path in the request. The learner must compare that request against every path; the host must not resolve the mentioned name. Unnamed import/dependency roles across files are outside this slice. This is an honest restriction on a small tool-learning task, not language generalization.

All arrays below use the fixed byte codes:

- Q and P_f are 32-by-32 request/path populations. Row0 is EOS, then bytes in reverse order, then zero rows. Bounds ensure every byte survives; row positions preserve order.
- F_f is a 129-by-32 whole-file population: bytes in forward order, one EOS row and trailing zero rows.
- C_fi is a 32-by-32 boundary population. Rows0..15 are bytes immediately to the left, nearest first, zero outside the file. Rows16..31 are bytes to the right, nearest first; index equal to file length carries EOS and later positions carry zeros.
- H_span is 129-by-32 for the selected raw span in forward order, followed by EOS and zero rows. G_payload is 17-by-32 for actual replacement bytes followed by EOS only if generation actually emitted EOS; remaining rows are zero.

A zero padding row is a declared absence signal. Neither code preparation nor candidate indexing inspects teacher data. Whole-file and path populations remain separate from local boundary context so long-file suffixes do not erase the path.

## Forward circuit and physical parameter counts

For every feature vector below, concatenate the stated raw blocks first, then apply the fixed transformation `phi = sqrt(32) * raw / max(||raw||_2, 1e-8)` in float32. Do not normalize blocks separately. Use finite checks; energy is 32 up to ordinary rounding. Matrix products are actual forward dendritic activity computed before any trained score weight.

**File/start:** for every boundary c=(f,i), form

`raw_start(c) = [vec(Q P_f.T), vec(Q F_f.T), vec(Q C_fi.T), i/128]`.

Widths are 1024+4128+1024+1 = **6177**. One shared 6177-synapse scalar output circuit computes each candidate score. Softmax over all B candidates followed by the fixed lowest-index argmax convention selects one joint file/start address. A fixed relay maps that neuronal address to file and byte offset. There is no candidate-specific bias or learned address slot table; a shared scalar bias would cancel and is omitted.

**End:** enumerate every j in the selected file with j>=i, including EOF. This is format validity, not correct-span filtering. Form

`raw_end(j | f,i) = [vec(Q C_fj.T), vec(C_fi C_fj.T), (j-i)/128]`.

Width is 1024+1024+1 = **2049**, with one shared scalar score circuit and the same softmax/argmax convention. At most129 candidates exist. The actual free start conditions the actual free end. No teacher start enters free activity.

**Payload:** call the current same-instance reader's `predict_with_receipt(request, retained_fact_state, max_bytes=16)` once with unchanged source-adjacent branch/codec semantics. Its raw bytes and actual EOS/limit outcome belong to this proposal. The host does not escape or quote them. The declared transferred reader prior is identical across arms; action synapses are fresh. No separate executor model is trained.

**Operation, last:** tentative span and payload are computed before STOP/SPLICE. Form

`raw_op = [vec(Q C_fi.T), vec(Q P_f.T), vec(H_span G_payload.T), (j-i)/128, len(payload)/16, payload_EOS_activity]`.

Width is1024+1024+2193+3 = **4244**. Two output neurons have weight[2,4244] and bias[2]; softmax/argmax chooses STOP or SPLICE, with STOP winning ties. Payload EOS is an observed neuronal stopping decision, not the host's operation choice. Operation activity runs even after payload hits its cap, so STOP is always reachable. If SPLICE is selected without payload EOS, the validator rejects that proposal. Empty EOS payload is valid deletion; equal boundaries are insertion.

Total fresh action parameters are6177+2049+(2*4244+2)=**16716 float32 values, 66864 bytes**. Initialize each population with independent private generators at seed+5/+6/+7 and ordinary fan-in/fan-out Xavier uniform weights, zero operation bias. These generators must not alter reader initialization or process RNG state. All learned tensors have requires_grad=False. Physical score circuits are1+1+2; candidate evaluations are B, E<=129 and2, not thousands of separately parameterized neurons.

The proposal record is explicit STOP or `(SPLICE,file,start,end,raw_payload,workspace_digest)` plus compact choices/work. The host validates and applies exactly those bytes to the authenticated pre-action workspace. It never chooses a useful location, substitutes pending, corrects malformed output or repairs syntax. Running a verifier after an edit/STOP is a declared harness action, not a learned RUN choice.

## Same-instance local teaching

All model parameters remain held across one free proposal and its real action. Release free feature banks before external execution; retain bounded raw proposal/state identities, not feature matrices or a hidden answer. Receive external execution evidence and only then a corrective action. Reconstruct teaching observations from the authenticated immutable PRE-action workspace, not its possibly wrong edited version.

For a SPLICE teacher, make the start bank with the teacher file/start label; build the end bank under that teacher start; use the teacher end and raw teacher payload for the operation's SPLICE example. These are explicit teacher-driven contexts. The reader receives the same raw request and supervised raw payload bytes/EOS through its existing local teacher rule. For a STOP teacher, offer only the STOP operation target using the actual free tentative span/payload and its EOS status. Do not fabricate file/end/payload labels or teach a fictitious empty payload. Thus STOP teaching needs the bounded actual proposal receipt even though its feature bank was discarded.

For either candidate selector, form its actual phi_c before scores and use direct output error `e_c = p_c - onehot(teacher)_c`. Its local weight change is `-0.01 * sum_c e_c * phi_c`. Compute the sum in float64, check/cast once to float32, then multiply/subtract at float32 for publication, matching the established local arithmetic convention. Operation uses its two direct class errors times its own phi and the corresponding bias errors. No error travels through argmax, fixed feature matrices, start conditioning or payload generation. Reader credit remains its existing direct B/codec rule; action loss does not backpropagate into it.

Hold weights through all teacher phases, stage proposals, validate every enabled proposal, then publish together. An invalid late proposal leaves all groups unchanged. This extends the existing transactional local-teaching boundary rather than fitting another model. Execution grades are recorded but are not numerical updates; corrective supervision is not reward learning or autonomous recovery. With fixed features of squared norm about32, direct selector covariance curvature is bounded by32; eta=.01 gives a conditional gain bound about.32. This is a held-feature objective bound, not convergence or end-to-end task monotonicity.

## Exact reuse, cost and unresolved representability

For start construction, QP and QF are invariant within a file and may be formed once per file, then copied as actual presynaptic blocks into each candidate before WHOLE-vector normalization. QC is formed per boundary. At four full files this is17,567,744 scalar feature products, rather than repeatedly recomputing101,978,112. It preserves numerical product order within each block; ordinary scalar/vector parity must verify it. End and operation banks are formed only after their selected prefix. Free banks are released phase by phase; teaching reconstructs its own banks and holds only the current phase plus staged weight deltas. No uncounted replay or feature/result cache spans actions.

The maximum start phi bank is12,749,328 bytes; a float64 cast and float64 weighted-product bank are25,498,656 bytes each. Their core overlap is63,746,640 bytes before code/context, normalization scratch, scores, deltas, reader activity and serialization buffers. Hubble's resource review fixes the full arithmetic/lifetime ledger. O(BD) is an asymptotic description, not a speed estimate. All raw scans, candidate products/reductions, normalizations, selector passes, actual payload decisions, corrective passes, teacher targets, byte storage and executions must enter new cost admission; v6 timing does not price this circuit.

Before any fit, ordinary authored QA must expose these limits:

- Requests/paths outside31bytes and files outside128bytes reject; four files do not constitute a repository-scale encoder. Request text is shared unchanged with the reader; distribution shift can break recalled values and must be measured.
- Joint dot-product features and linear output scores are not injective universal representations. Exact equal phi vectors with different correct labels are unlearnable for that population. Repeated local contexts can leave only a linear positional preference; that cannot select every arbitrary interior duplicate. Check fresh moved/repeated-key/path-swap fixtures at the feature level before proposing acquisition, without adding semantic candidate masks.
- Each file's own whole-file term provides some distant evidence but does not model cross-file imports. Explicit-path scope prevents claiming that unsupported relation. No path parser may supply the correct file.
- The payload remains a contiguous-source/codec prior. It cannot be assumed to insert new escaping bytes, assemble separated code fragments or synthesize arbitrary Python. First payloads must be literal raw source-copy cases; escaped code is supported only when the actual emitted bytes satisfy syntax. Include a visible quoting case to reject host escaping, and retain unsupported composition failures rather than claiming them solved.
- Operation sees the provisional span/payload, not an oracle-correct location. Correct idempotent STOP therefore depends on correct earlier location/payload choices. A wrong provisional span can hide evidence elsewhere in the file; STOP-on-failure is not guaranteed general answerability. Its direct features also do not prove arbitrary equality classification across all byte strings.

QA should first test raw encoding/bounds, every boundary, scalar local updates, actual-free versus teacher-start end contexts, STOP after capped payload, no teacher labels on STOP, late atomic failure, and one same-object actual-action-before-correction trace. Feature collision checks on authored same-input-context/opposite-location cases precede fits; mechanics QA is not acquired skill. The later comparison uses fresh action weights with identical declared reader prior in learned, action-frozen/reader-plastic, all-slow-frozen and retained-fact-disabled arms. Root freezes fresh layout/key/value pairs, STOP/error counts, retention checks and cost only after this topology's ordinary QA. No interference-rule change or competing architecture queue is selected here.
