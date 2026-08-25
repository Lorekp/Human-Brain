#!/usr/bin/env python3
"""T-ROUTE E3 pre-spend unit checks (Change Note 4 C7: ALL GREEN before spend).

Covers:
  T1  quote extraction + byte-subsequence span localization (single match,
      overlapping multi-match union, zero-match, missing quotes, empty quotes);
  T2  positional pairing rule: ungated Counter == verbatim aggregate recipe;
      quote-gated counter == independent brute force; random-span control
      properties (fixed length, valid range, reproducibility, RNG stream
      consumed only by included episodes in canonical order);
  T3  topk_row_sparsify: tie-break ASCENDING column, k-clamp, identity at
      k = fan_in;
  T4  frozen three-way verdict function incl. the +inf specificity guard;
  T5  scorer overlap function vs independent set recomputation.

Run: python toy_check.py   (exits nonzero on any failure)
"""

from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from run_e3 import (  # noqa: E402
    RANDOM_SPAN_SEED,
    aggregate_pair_counts,
    contiguous_subsequence_matches,
    episode_pair_occurrences,
    quote_span_mask,
    quote_token_ids,
    random_span_start,
    topk_row_sparsify,
)
from score_e3 import evaluate_verdict, overlaps_own_gold, specificity  # noqa: E402
from humanbrain.tokenizer import BytePhraseTokenizer  # noqa: E402

FAILURES: list[str] = []


def check(name: str, cond: bool) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}")
    if not cond:
        FAILURES.append(name)


def brute_force_counts(question: str, context: str, span: set[int] | None = None) -> Counter:
    """Independent (deliberately naive) reimplementation of the §2 rule.

    With span=None this is the ungated rule; otherwise pairs whose PASSAGE
    POSITION jp is not in span are dropped (the E3 gate).
    """
    p = list(context.encode("utf-8"))
    q = list(dict.fromkeys(question.encode("utf-8")))
    counts: Counter = Counter()
    for tq in q:
        anchors = [j for j, t in enumerate(p) if t == tq]
        for a in anchors[:8]:
            for jp in range(max(0, a - 12), min(len(p), a + 12 + 1)):
                if span is not None and jp not in span:
                    continue
                counts[(tq, p[jp])] += 1
    return counts


def main() -> int:
    tok = BytePhraseTokenizer()  # no phrases -> pure byte identity tokens

    print("T1: quote extraction + span localization")
    ctx = 'Mark 7:10 For Moses said, Honour thy father; and more'
    q1 = 'Who said, "Honour thy father"?'
    p_ids = tok.encode(ctx)
    mask, n_m, s_len = quote_span_mask(q1, p_ids, tok)
    exp_start = ctx.find("Honour thy father")
    check(
        "single match at expected offset",
        n_m == 1 and s_len == len(b"Honour thy father") and bool(mask[exp_start : exp_start + s_len].all())
        and int(mask.sum()) == s_len,
    )
    check(
        "overlapping multi-match union",
        contiguous_subsequence_matches(list(map(ord, "ababa")), list(map(ord, "aba"))) == [0, 2],
    )
    m2, n2, l2 = quote_span_mask('x "ab" y', tok.encode("abab"), tok)
    check("multi-match mask union", n_m == 1 and n2 == 2 and l2 == 2 and int(m2.sum()) == 4)
    m3, n3, _ = quote_span_mask('Who said, "zzz not here"?', p_ids, tok)
    check("zero match -> all-false mask with n=0", n3 == 0 and not m3.any())
    m4, n4, _ = quote_span_mask("Who said, nothing quoted?", p_ids, tok)
    check("missing quotes -> empty-mask episode", m4 is None and n4 == 0)
    m5, n5, _ = quote_span_mask('Who said, ""?', p_ids, tok)
    check("empty quoted span -> zero plasticity mask", n5 == 0 and not m5.any())
    check("quote_token_ids content", quote_token_ids(q1, tok) == tok.encode("Honour thy father"))

    print("T2: pairing rule + gated counters + random-span control")
    context = "abcabd"
    question = 'Who said, "abc"?'
    item = {"context": [context], "question": question}
    p_ids_t = tok.encode(context)
    occ = episode_pair_occurrences(tok.encode(question), p_ids_t)
    ungated = Counter((tq, p_ids_t[jp]) for tq, jp in occ)
    bf = brute_force_counts(question, context)
    check("ungated positional counter == verbatim aggregate recipe", ungated == aggregate_pair_counts([item], tok))
    check("ungated counter == independent brute force", ungated == bf)
    # 'a' anchors at 0,3; 'd' at 5; 'b' at 1,4; 'c' at 2 -> 4 anchors x 6 = 36 events
    check("hand total (36 events)", sum(ungated.values()) == 36)
    gmask, g_nmatch, g_len = quote_span_mask(question, p_ids_t, tok)
    check("toy span is {0,1,2}, single match, len 3", g_nmatch == 1 and g_len == 3 and list(np.flatnonzero(gmask)) == [0, 1, 2])
    gated = Counter((tq, p_ids_t[jp]) for tq, jp in occ if gmask[jp])
    gated_bf = brute_force_counts(question, context, span={0, 1, 2})
    check(
        "gated counter == independent brute force on span {0,1,2}",
        gated == gated_bf and sum(gated.values()) == 18,
    )
    rng_a = np.random.default_rng(RANDOM_SPAN_SEED)
    starts = [random_span_start(rng_a, 6, 3) for _ in range(4)]
    rng_b = np.random.default_rng(RANDOM_SPAN_SEED)
    starts_b = [random_span_start(rng_b, 6, 3) for _ in range(4)]
    check("random-span draws reproducible", starts == starts_b and all(0 <= s <= 3 for s in starts))
    # stream consumed ONLY by included episodes, canonical order:
    rng_c = np.random.default_rng(RANDOM_SPAN_SEED)
    sim = []
    for included in (True, False, True):
        if included:
            sim.append(random_span_start(rng_c, 10, 4))
    rng_d = np.random.default_rng(RANDOM_SPAN_SEED)
    manual = [random_span_start(rng_d, 10, 4), random_span_start(rng_d, 10, 4)]
    check("excluded episodes do NOT consume RNG stream", sim == manual)

    print("T3: topk_row_sparsify")
    vals = np.array([[1.0, 3.0, 3.0, 2.0], [0.0, 0.0, 0.0, 9.0]])
    sup = np.array([[0, 1, 2, 3], [0, 1, 2, 3]])
    out = topk_row_sparsify(sup, vals, 2)
    check("ties kept by ascending column index", out[0].tolist() == [0.0, 3.0, 3.0, 0.0])
    check("k clamp keeps row nonzeros", out[1].tolist() == [0.0, 0.0, 0.0, 9.0])
    out_id = topk_row_sparsify(sup, vals, 4)
    check("k=fan_in is identity", np.array_equal(out_id, vals))

    print("T4: verdict function")
    v1, d1 = evaluate_verdict(True, 5.0, 4.0, 1.5, 1.2)
    check("RECOVERED branch", v1 == "QUOTE_SPAN_ROUTING_RECOVERED" and d1["spec_q_gt_spec_rs_both_splits"])
    v2, _ = evaluate_verdict(True, 5.0, 4.0, 6.0, 7.0)
    check("GATING_ARTIFACT branch", v2 == "GATING_ARTIFACT_NO_QUOTE_SPECIFICITY")
    v3, d3 = evaluate_verdict(False, 9.0, 9.0, 0.1, 0.1)
    check("FAILS branch dominates when G-T1' fails", v3 == "QUOTE_SPAN_GATE_FAILS" and not d3["gt1_both_splits"])
    v4, _ = evaluate_verdict(True, math.inf, math.inf, 3.0, 2.0)
    check("+inf guard (shuf_med==0)", v4 == "QUOTE_SPAN_ROUTING_RECOVERED")
    check("specificity helper inf/zero semantics",
          specificity(4.0, 0.0) == math.inf and specificity(0.0, 0.0) == 0.0 and abs(specificity(3.0, 2.0) - 1.5) < 1e-12)

    print("T5: overlap scorer")
    rng = np.random.default_rng(1234)
    Wm = rng.integers(0, 64, size=(7, 16)).astype(np.int64)
    gm = np.zeros((7, 64), dtype=bool)
    for i in range(7):
        gm[i, rng.choice(64, size=20, replace=False)] = True
    got = overlaps_own_gold(Wm, gm)
    want = np.array([len(set(Wm[i]) & set(np.flatnonzero(gm[i]))) for i in range(7)])
    check("overlaps_own_gold == independent set intersection", np.array_equal(got, want))

    print()
    if FAILURES:
        print(f"TOY CHECK FAILED ({len(FAILURES)}): {FAILURES}")
        return 1
    print("TOY CHECK: ALL GREEN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
