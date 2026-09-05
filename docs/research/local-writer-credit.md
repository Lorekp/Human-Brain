# Delayed local writer credit: candidate contract
2026-09-06; independent derivation only. No implementation, fit or numerical experiment.
Sources: plastic_memory.py local_step/apply_learning_signal; byte_sequence_decoder.py teach_local/context_signal.
[Bellec et al., Results/Methods](https://doi.org/10.1038/s41467-020-17236-y) motivates eligibility × learning signals;
this memory extension is our approximation, not an e-prop theorem or biological-fidelity claim.
## Forward computation and missing credit
Write: h←cell(x,h), k=normalize(Kh), v=tanh(Vh), β=sigmoid(b·h), M←M+β(v−Mk)kᵀ.
Read: u=query_cell(query), q=normalize(Qu), r=Mq; decoder context=[u,r] produces byte/EOS probabilities p.
The fast rule does not train h/K/V/b/Q. A random writer with a trained readout is a control, not the goal.
At an authorized teacher step δ=p−onehot(byte); existing decoder broadcasts g=B_context δ through fixed connections.
g is NOT ∂loss/∂context. Module signals L^m=C_m g use fixed declared matrices, never per-synapse external errors.
## Compare the two economical trace choices
Aggregate local E_ji with Δθ_ji=−η L_j E_ji costs O(P), and may heuristically learn representations.
Its feedback depends on current query/prediction/value, but it loses explicit attribution to particular earlier writes.
Counterexample: eligibilities +1 and−1 at orthogonal keys aggregate to0; the two queries require +1 versus−1.
This proves information loss, not impossibility of aggregate-feedback learning. Keep it as a matched credit control.
SELECT one K-vector T_ji per writer/map synapse, tagged by actual write-key neuronal activity.
Hebbian accumulation T←λT+β e_ji k would ignore overwrite. Instead use T←T−β(T·k)k+β e_ji k, λ=1.
All right-hand terms use old T. At feedback form E_ji(q)=Σ_k q_k T_ji,k; apply Δθ_ji=−η L_j E_ji(q).
This transports tags through the fixed-key overwrite operator, but omits key/β derivatives, cross-neuron recurrent
paths and slow-weight changes. It is not the full derivative of M, a recurrent gradient or a guarantee of descent.
A synapse receives its own state/eligibility, pre/post activity, β and declared key k/query q broadcasts plus L_j.
Those broadcasts are neuron activities/errors, not hidden reverse-mode results or externally computed update tensors.
No per-document/per-write external trace cache is selected; T has fixed size per active memory stream.
## The writer and addressing maps must actually learn
Writer-cell e_ji uses the implemented self-path eligibility recurrence; omitted off-diagonal recurrent credit is explicit.
For a map z_j=f(a_j), a_j=Σ_i W_ji h_i, local e_ji=ψ_j h_i and bias eligibility=ψ_j.
Use ψ=1−v_j² for values, β(1−β) for strength, and diagonal normalization gain for key/query maps:
ψ_j=(1−z_j²)/||a|| above ε, otherwise1/ε. Omitted off-diagonal normalization terms must be disclosed.
Key/value/strength maps enter the same tagged update using their own fixed neuron-level C_m g broadcasts.
At read time Q uses current e^Q_ki=ψ_k u_i and ΔQ_ki=−η L^Q_k e^Q_ki; it needs no stored source derivative.
Query cells use current query eligibility and the query part of context_signal or a declared fixed projection.
Frozen byte embeddings may supply inputs; writer/query transformations and these maps remain plastic.
## Event clocks and custody
Start a teaching episode with zero M/activity/local eligibilities/T and retained slow weights.
Advance write traces once per actual byte including terminator; no target enters ingestion and no wall-time decay occurs.
Purge raw/token inputs and transient writer activity before the genuine read-only first query/byte attempt.
Retain only M, T, slow weights and needed local traces; count T as source-derived plasticity state, never decoder input.
Hold writer/address/query slow weights fixed through that lesson. Decoder may update locally per explicitly taught byte/EOS.
Average its BEFORE-update fixed broadcasts over teacher steps; apply upstream updates once using that lesson's query activity.
Then retire T/query traces. Do not recompute old eligibility or rebuild M after a slow update; later writes can interfere.
Read-only delays change neither M nor T. Teacher-forced output is additional declared exposure, never a first-attempt success.
## Cost, control and one bounded return
Writer P=64·32+64²+64=6,208: float32 O(P) eligibility24,832B; K64 tags1,589,248B/stream.
Controller+key/value/strength maps P=14,593: tags3,735,808B/stream; query traces/M/live state are extra.
This is O(PK), NOT O(P). Per-write snapshots would cost O(RP); one end-document tag does not represent all byte writes exactly.
Use varying key/value assignments and complete two-document ingestion; evaluate both keys from each shared memory.
Compare tagged credit, aggregate-only credit, writer/maps-frozen with trained query/decoder, fully frozen, and writes-disabled.
Match examples, teacher bytes, first attempts and execution; freeze the increased control/teaching totals separately.
Require fit and exact new-value byte/action improvement over writer/maps-frozen, plus same-query/different-memory and
same-memory/different-query success. Moving weights alone does not establish useful learned writer representations.
One fixed configuration: ≤15-minute disposable cost admission, then one ≤15-minute integrated development run.
Return to the existing20 checks/arm action geometry; no inherited BPTT admission or undisclosed additional practice.
Failure to fit or beat frozen-writer closes this candidate. Synthetic locality/fit is not prose understanding or a general gate.
