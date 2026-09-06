# Raw discrimination and local address learning: primary-source review

Status: bounded source/literature review, 2026-09-06. No implementation,
checkpoint load, prediction, alternative decoding, experiment or QA execution.
The [roadmap](roadmap-reading-to-coding.md) and
[completed learning review](neuronal-splice-online-v2-learning-review.md) own the
shared priority. This note selects no architecture or acquisition.

The strongest next question is **whether generic forward identity features make
the required address distinctions representable and learnable with the existing
local teacher**. A winner-aware update is mathematically possible at the same
output synapses, but literature on supervised perceptrons does not supply the
missing correct-address target. Changing both features and credit together
would make the next result harder to interpret. This recommendation agrees with
the shared learning review; it is not a second implementation queue.

## What the three primary sources support

| Primary source | Relevant evidence | Boundary for this project |
|---|---|---|
| [Cover, 1965, *Geometrical and Statistical Properties of Systems of Linear Inequalities*, §§II–IV, VI](https://isl.stanford.edu/~cover/papers/paper2.pdf) | Treats fixed measurement functions followed by a linear threshold. Product features change the class of separating surfaces. Capacity results explicitly require geometric assumptions; the paper also treats ambiguous generalization. | Supports studying the feature map before blaming the learning rule. It does not prove that this byte-coincidence map separates our requests or that more features improve transfer. Binary coincidence patterns are not automatically in general position. |
| [Plate, 1995, *Holographic Reduced Representations*, abstract/introduction](https://www2.fiit.stuba.sk/~kvasnicka/CognitiveScience/6.prednaska/plate.ieee95.pdf) | Demonstrates a distributed binding approach with fixed-width vectors and noisy reconstruction, with associative cleanup as a distinct component. | Evidence that compressed distributed representations can support useful structure, not evidence that all random-code schemes preserve exact categorical identity. Its convolution/reconstruction machinery differs from this learner's lagged dot products; no numerical noise bound is transferred. We do not adopt a cleanup memory or external answer filter. |
| [Collins, 2002, *Discriminative Training Methods for Hidden Markov Models*, Fig. 2 and §3](https://aclanthology.org/W02-1001.pdf) | Scores candidates linearly and updates by the supplied correct candidate's features minus the predicted candidate's features. Its margin-separable mistake bound depends on bounded feature differences and a separating parameter vector. | Establishes a direct additive output-learning precedent without differentiating the decoder. The supervised output in the theorem is known. Our address is latent: receiving payload bytes is not receiving the correct memory location. The theorem cannot justify substituting an inferred address for that label. |

These are three original papers, not secondary summaries. Cover and Collins'
equations were read in their primary PDFs. Plate is used only for the readable
abstract/introduction; the mirror's full PDF text extraction was corrupt, so no
unread detailed theorem or bound is attributed to it.

## Exact current source and what it leaves unresolved

In [LocalMemoryLearner](../../src/humanbrain/local_memory_learner.py), fixed
Gaussian byte codes have 32 channels. Query lags retain the last 31 raw bytes
plus EOS. For each stored candidate, the actual forward map is
`X_s[i,j] = sum_d Z[i,d] K_s[j,d]`; one shared learned output layer gives
`a_s = sum_ij B[i,j] X_s[i,j]`. The
[competitive reader](../../src/humanbrain/competitive_local_memory_learner.py)
commits to the actual winning score; it inherits corrective teaching unchanged.
With `p = softmax(a)` and the final forward teacher-prefix posterior `b`,
the local publication is `delta B = eta * sum_s X_s (b_s - p_s)`.
No error is transported backward through the query/key feature map or winner.

The saved v2 review finds unique rank-two intended starts in its nine fresh
payload failures. That establishes wrong ordering in those examples, not its
cause. Distinct random vectors need not collide exactly. Nevertheless, by
elementary linear algebra, a 257-symbol Gram matrix formed from 32-dimensional
vectors has rank at most 32 and cannot equal the 257-dimensional identity.
Thus this dot-product representation cannot make every different symbol pair
universally orthogonal. This is a source-based mathematical observation, not a
measurement of how much random overlap caused the saved errors.

Conversely, if any proposed feature map gives `X_t = X_w` for two candidates,
then every possible `B` gives equal scores to them. No direct output update can
repair that collision. Different features are necessary for discrimination,
but do not guarantee a suitable learned margin, teacher evidence or transfer.

## Generic categorical/coincidence features

A categorical encoder can make same-symbol evidence exact before the learned
score synapses. For example, a byte/EOS population with distinct category
activities can feed explicit coincidence neurons. One-hot dot products are
an elementary equality construction; a compact fixed Boolean circuit is another
possible representation. Such wiring is a disclosed categorical prior, not a
learned discovery. The algorithm must apply to all byte values, EOS, lag pairs
and candidate positions; it must not parse a key, select a document role,
mask a correct span or exclude inconvenient candidates.

The parallel source/math proposal uses nine-bit dual-rail symbol codes, with
18 nonnegative rails and nine active rails per occupied symbol. Pairwise AND
activities followed by a nine-match threshold can recognize identical symbols.
This observation assumes the raw rails and explicit occupied/empty convention;
zero-filled empty lags must never become an empty/empty match. It is not a
literature-validated solution or a selected implementation.

Exact single-symbol equality still leaves temporal binding and nuisance request
context to be resolved. A linear sum over equality cells can rank aligned match
counts and may separate simple conjunctions; it does not explicitly provide
every nonlinear relation between multiple matches. Equality-only features are
also invariant under a consistent relabeling of symbols. That is useful for
generic identity matching but cannot alone express arbitrary absolute-symbol
meaning. Repeated sequences, identical relevant context, lost older context and
roles requiring unrepresented information remain possible ambiguities. These
are representability questions to settle before fitting, not reasons to insert
a host role parser or add benchmark-specific features.

Changing the map changes the meaning and scale of `B` and stored key activity.
Old address weights cannot be declared a meaningful matched prior merely
because tensor dimensions coincide. Any selected design must state how address
weights and tapes start, what codec knowledge is preserved, and the physical
feature, storage and local-update costs. No speed or accuracy gain follows from
the papers or this source review.

## Winner-aware learning: local, but not yet justified here

For a known correct address `t` and free winner `w`, an additive rule
`delta B = eta * (X_t - X_w)` on mistakes resembles Collins' supervised
candidate update. Each changed synapse uses its own presynaptic activity and
direct output-phase signal; the winner circuit need not be differentiated.
But obtaining `t` by matching authored keys or payload substrings would create
the forbidden target-address oracle. The cited supervised guarantee therefore
does not apply to that proposed use here. [Collins, Fig. 2 and §3](https://aclanthology.org/W02-1001.pdf).

An oracle-free variant could retain the actual free one-hot activity `w` and
use the same forward teacher posterior: `delta B = eta * sum_s X_s (b_s-w_s)`.
This is a new local contrast, not the current `b-p` update, not a derivative of
the committed policy and not automatically a perceptron mistake correction.
The teacher posterior can favor a wrong address because the codec is wrong,
because several starts emit the same payload, or because EOS alone is ambiguous.
Sharpening that posterior to a teacher winner does not remove those assumptions.
Clamping teaching to the free winner instead leaves its posterior on that same
support and supplies no alternative address credit.

For fixed features, the proposed change to a target-versus-winner score margin
would be `eta * (X_t-X_w) dot sum_s X_s(b_s-w_s)`. Its sign is not guaranteed
positive without assumptions about both the posterior and geometry. The actual
execution grade could say that an action failed, but it does not identify which
address, codec output, file/span or operation caused that failure. Turning that
grade into a new numerical teaching signal would require another explicit rule.

Keep one shared source/math priority: generic discrimination before the same
locally trained output layer, with collision limits and teacher assumptions
made explicit. The working competition, real proposal-before-feedback loop,
independent STOP/location failures and no-backpropagation constraint remain.
Only fresh controlled acquisition under a separately selected complete resource
admission could test a successor; this literature review predicts no repaired
v2 output and establishes no general coding capability.

Discovery note: graph symbol search returned no current reader nodes, so exact
source was read directly. A task-specific Mem0 search returned unrelated older
development context; none was used as scientific evidence.
