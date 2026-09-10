# Acquisition decision input: observe the selected coupled rule once

Decision input updated to root's selected direction, 2026-09-10; the separately
frozen protocol owns execution. Keep the preserved `17a879c` learner at .01 for
one finite integrated acquisition measurement before calibrating its
coefficients. The [roadmap](roadmap-reading-to-coding.md),
[active plan](persistent-language-action-development-v1.md) and
[budget review](integrated-local-learning-v1-budget-review.md) remain authoritative.
This note executes no model, teacher or recurrence and changes no implementation.

## A tighter contact timescale

For the selected proposal's idealization—one repeated unit input, one target,
uniform 257-cell softmax, direct CE alone—write `a=.01*257/256` and
`m_next=m+a/(1+exp(m)/256)`. Define `F(m)=m+exp(m)/256`.
Its derivative increases with m. Over one step, the integral of F' is at least
`F'(m)*delta=a` and at most `exp(a)*F'(m)*delta=a*exp(a)`, since `delta<=a`.
Thus, for target margin `m_q=log(256*q/(1-q))`, the first contact count reaching q
is bounded by

`ceil([F(m_q)-F(0)]/[a*exp(a)]) <= n_q <= ceil([F(m_q)-F(0)]/a)`.

The upper bound is sufficient **only within this idealized recurrence**. The
central closed-form quantity `[log(256*q/(1-q))+q/(1-q)-1/256]/a` also supplies
an approximation accurate to about 1% in these bounds. Direct evaluation of
these formulas, without iterating the recurrence, gives:

| Literal target q | Necessary contacts | Sufficient contacts |
| --- | ---: | ---: |
| .5 | 646 | 652 |
| .9 | 1,651 | 1,668 |
| .99 | 10,763 | 10,872 |

This is substantially more informative than the previous maximum-margin-step
lower bound. At historical 24.375% inclusion, .9 corresponds to roughly 6,800
corrections at each position, and .99 to roughly 44,600. Those conversions are
planning sensitivities, not predictions. Inclusion is now learned differently;
shared repeated characters can add to a cell's learning, while conflicting
features and changed context can oppose it. Scan/public terms are absent from
the derivation. Neither these bounds nor a teacher-prefix mixture product is a
bound on free-answer acquisition in the actual recurrent actor.

## Why observe before calibrating

The calculation establishes an expensive direct-contact timescale. It does not
identify the best coefficient for the coupled rule. Brier utility can change
mode inclusion, context/gate prefix credit can change task separation, and scan
centering can change competing currents. None has been observed during enabled
integrated learning yet. Raising a common coefficient now would alter all three
alongside literal learning; raising literal alone could leave the mode cap and
wrong context intact. The finite update norm is not a useful-step or convergence
guarantee. There is no scientifically determined scalar multiplier in the saved
evidence, so a coefficient chosen merely to fit a desired runtime would be an
unsupported calibration.

Root selects **512 corrections per skill** as one fixed integrated comparison:
R64, E64, then 448 alternating pairs; panels at main indices 0, 64, 128, 256,
512, 768 and 1,024. This is sixteen times the previous correction opportunity,
with the already selected per-contact teacher coefficient 24/69 times its
predecessor reading/editing value. Those are exposure/coefficient comparisons,
not predicted gains. At historical inclusion it supplies only about 125
literal contacts per position. It is therefore **not a budget justified to
produce confident complete responses from direct CE alone**. Its justification
is a finite opportunity to observe the coupled learning computation, including
whether its assumed inclusion, selectivity and routing improvements occur.

Use the same retained ONLINE numerical state, matched predecessor and selected
enabled/disabled arms, and continued simple reference, in two source-isolated
triads, each starting at actor/reader experience 401/406. Keep actual free prediction
before every correction, actual edit
execution, fixed difficulty, RNG continuation and the existing four panel cases.
The preserved 20:24 motor span and both versions' initial 0/4 complete outcomes
are the starting evidence. Live actor state is never reset by a panel. Complete
acquisition requires both taught free outcomes at the final two panels; report
all live outcomes as well, without hiding successes or failures between panels.

The historical complete-run scaling is about 13.5 hours for both triads at this
ceiling; it is not a throughput measurement for v3. Root selects at most 12 hours
per worker (24 hours paired), plus a separately bounded audit. The new protocol
sets a 768 MiB artifact cap per worker, with a 256 MiB pending reserve inside
that cap, and a 6 GiB free-space floor. Two capped workers retain at most 1.5 GiB
before separately declared shared/audit artifacts. At the reported approximately
9.14 GB free, this leaves room above that new floor; exact admission must include
all shared/audit writes. This does not relax any historical frozen 8 GiB contract.
Unmodified main journals alone project about 13.72 GB, so use compact practice
receipts and full scheduled diagnostics without claiming a CPU speedup.
Keep enabled teacher time, free length and actual disk growth visible. There is
no success- or contact-triggered stop or extension. Any resource stop is an
incomplete declared comparison, not a smaller completed cohort or permission to
continue automatically.

## Terminal diagnosis, not another automatic gate

Report actual positionwise contact counts, direct CE and target-versus-competitor
margin at matching activities; mode probabilities and mixture components;
teacher/free prefix and activity alignment; block interference; exact free
bytes/EOS/limits; executed grade; and preserved motor behavior. Count actual
observed offers separately from selected literal publications. Capture sufficient
fixed-panel vectors for target/margin comparisons rather than inferring margins
from p alone.

If complete skills remain wrong with low contacts, the result establishes
failure under this finite exposure and whether the coupled dynamics help; it
does not falsify the actor's eventual ability or make an architecture addition
necessary. If mode inclusion improves and competing margins improve at held
activities while literal confidence remains low, that supplies actual grounds
to review contact-unit calibration against measured teacher cost. If inclusion
or task separation fails, or scan/inter-skill margins deteriorate, more literal
gain is not identified as the correction. These are diagnostic interpretations,
not automatic subsequent experiments. One terminal review selects any next
action; no repetition ladder, 4,096-repeat continuation or rate/seed sweep is
released by this decision input.
