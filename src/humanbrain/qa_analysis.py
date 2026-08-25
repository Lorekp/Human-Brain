"""Post-verdict analysis for the R2 Bible-QA pilot (FINDINGS-style, mechanical).

Consumes the R2 pilot record (``humanbrain/qa-r2-pilot-bible/v1``,
``experiments/bible_qa_r2_pilot/pilot_record.json``) plus the preregistered
baseline records (``humanbrain/qa-baselines/v1``,
``experiments/qa_baselines_bible_v1/*.json``) and produces the decision-grade
analysis the R2 verdict needs:

- Wilson score intervals on every proportion (same z as ``run_pilot``);
- Newcombe hybrid-score (method 10) 95% intervals on pilot-vs-baseline
  span-accuracy DELTAS — the correct interval for comparing two independent
  proportions, never a naive difference of interval bounds;
- degeneracy-guard verdicts (distinct >= 3, modal share <= 0.9) for the pilot
  and every baseline, with machine-checkable reasons;
- per-family breakdown (who_said / address_lookup / verse_continuation) with
  recomputed intervals;
- a mechanical, preregistered-style rule readout (beat trigram? beat unigram?
  guard clean?) rendered as FINDINGS markdown.

No wall-clock enters any digest; everything here is a pure function of the
input records except the explicitly deterministic fixture synthesizer
(``synthesize_fixture_record``), which exists so the tool can be exercised
BEFORE the pilot finishes (one-shot discipline: the real record is written
once, by the pilot run, and never edited).
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json

ANALYSIS_SCHEMA_VERSION = "humanbrain/qa-r2-analysis/v1"
PILOT_RECORD_SCHEMA = "humanbrain/qa-r2-pilot-bible/v1"
BASELINE_RECORD_SCHEMA = "humanbrain/qa-baselines/v1"

Z_95 = 1.959963984540054  # identical constant to run_pilot.wilson

FAMILIES = ("address_lookup", "verse_continuation", "who_said")

GUARD_DISTINCT_MIN_DEFAULT = 3
GUARD_MODAL_SHARE_MAX_DEFAULT = 0.9


# ---------------------------------------------------------------------------
# Intervals
# ---------------------------------------------------------------------------


def wilson(correct: int, n: int, z: float = Z_95) -> dict[str, float]:
    """Wilson score interval on a binomial proportion (matches run_pilot)."""
    if type(correct) is not int or type(n) is not int:
        raise ValueError("wilson expects integer counts")
    if n < 0 or correct < 0 or correct > n:
        raise ValueError(f"invalid counts correct={correct} n={n}")
    if n == 0:
        return {"low": 0.0, "high": 0.0}
    p = correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    return {"low": max(0.0, centre - half), "high": min(1.0, centre + half)}


def newcombe_delta_ci(
    correct_a: int,
    n_a: int,
    correct_b: int,
    n_b: int,
    z: float = Z_95,
) -> dict[str, float]:
    """Newcombe (1998) method 10 hybrid-score interval for ``p_a - p_b``.

    Square-and-add construction over the two independent Wilson intervals:
    for d = p_a - p_b,

        low  = d - sqrt((p_a - l_a)^2 + (u_b - p_b)^2)
        high = d + sqrt((u_a - p_a)^2 + (p_b - l_b)^2)

    clamped to [-1, 1]. This is the recommended non-Pooling interval for
    comparing two proportions; unlike the plain Wald difference it behaves
    correctly near 0/1, and it reduces to the Wilson width in the degenerate
    one-sample limit.
    """
    if min(n_a, n_b) == 0:
        raise ValueError("newcombe_delta_ci needs nonzero sample sizes")
    p_a = correct_a / n_a
    p_b = correct_b / n_b
    w_a = wilson(correct_a, n_a, z)
    w_b = wilson(correct_b, n_b, z)
    point = p_a - p_b
    low = point - ((p_a - w_a["low"]) ** 2 + (w_b["high"] - p_b) ** 2) ** 0.5
    high = point + ((w_a["high"] - p_a) ** 2 + (p_b - w_b["low"]) ** 2) ** 0.5
    return {"low": max(-1.0, low), "high": min(1.0, high), "point": point}


# ---------------------------------------------------------------------------
# Record access helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpanBlock:
    """Teacher-forced answer-span counts extracted from a record."""

    label: str
    correct: int
    predictions: int
    accuracy: float | None  # None when predictions == 0 (e.g. majority floor)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_span_block(record: dict, label: str) -> SpanBlock:
    metrics = record.get("metrics")
    _require(isinstance(metrics, dict), f"{label}: missing metrics object")
    span = metrics.get("answer_span")  # type: ignore[union-attr]
    _require(isinstance(span, dict), f"{label}: missing metrics.answer_span")
    correct = span.get("correct", 0)
    predictions = span.get("predictions", 0)
    _require(
        isinstance(correct, int) and isinstance(predictions, int) and correct >= 0 and predictions >= 0,
        f"{label}: answer_span counts must be non-negative integers",
    )
    accuracy = span.get("accuracy")
    if predictions == 0:
        return SpanBlock(label, correct, 0, None)
    _require(
        isinstance(accuracy, (int, float)),
        f"{label}: answer_span.accuracy missing with nonzero predictions",
    )
    return SpanBlock(label, correct, predictions, float(accuracy))


def guard_verdict(degeneracy: dict, label: str) -> dict[str, object]:
    """Machine-checkable degeneracy-guard verdict with explicit reasons."""
    _require(isinstance(degeneracy, dict), f"{label}: missing degeneracy metrics")
    distinct = degeneracy.get("distinct_answers")
    modal_share = degeneracy.get("modal_share")
    guard_cfg = degeneracy.get("guard", {})
    distinct_min = int(guard_cfg.get("distinct_answers_min", GUARD_DISTINCT_MIN_DEFAULT))
    modal_max = float(guard_cfg.get("modal_share_max", GUARD_MODAL_SHARE_MAX_DEFAULT))
    recorded_pass = degeneracy.get("guard_pass")
    _require(
        isinstance(distinct, int) and isinstance(modal_share, (int, float)),
        f"{label}: degeneracy metrics need integer distinct_answers and numeric modal_share",
    )
    reasons: list[str] = []
    passed = True
    if distinct < distinct_min:
        passed = False
        reasons.append(
            f"distinct_answers={distinct} < required minimum {distinct_min}"
        )
    if float(modal_share) > modal_max:
        passed = False
        reasons.append(
            f"modal_share={float(modal_share):.4f} > allowed maximum {modal_max}"
        )
    if recorded_pass is not None and bool(recorded_pass) != passed:
        reasons.insert(
            0,
            (
                "WARNING: recorded guard_pass="
                f"{recorded_pass} disagrees with recomputation ({passed})"
            ),
        )
    return {
        "label": label,
        "verdict": "PASS" if passed else "FAIL",
        "distinct_answers": distinct,
        "modal_share": float(modal_share),
        "modal_answer": degeneracy.get("modal_answer"),
        "thresholds": {
            "distinct_answers_min": distinct_min,
            "modal_share_max": modal_max,
        },
        "reasons": reasons,
    }


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------


def compare_to_baseline(pilot: SpanBlock, baseline_record: dict) -> dict[str, object]:
    """Pilot-vs-baseline span-accuracy delta with a Newcombe 95% interval."""
    arm = str(baseline_record.get("arm", "unknown"))
    base = parse_span_block(baseline_record, f"baseline:{arm}")
    result: dict[str, object] = {
        "arm": arm,
        "pilot_accuracy": pilot.accuracy,
        "baseline_accuracy": base.accuracy,
        "applicable": pilot.accuracy is not None and base.accuracy is not None,
    }
    if pilot.predictions and pilot.correct is not None:
        result["pilot_wilson95"] = wilson(pilot.correct, pilot.predictions)
    else:
        result["pilot_wilson95"] = None
    if base.predictions and base.correct is not None:
        result["baseline_wilson95"] = wilson(base.correct, base.predictions)
    else:
        result["baseline_wilson95"] = None
    if not result["applicable"]:
        result.update(
            {
                "delta_point": None,
                "delta_ci95": None,
                "ci_excludes_zero": None,
                "verdict": "NOT_APPLICABLE",
                "note": (
                    "span comparison undefined (a side has zero teacher-forced "
                    "predictions; majority_floor is scored on free-run only)"
                ),
            }
        )
        return result
    assert pilot.accuracy is not None and base.accuracy is not None
    ci = newcombe_delta_ci(pilot.correct, pilot.predictions, base.correct, base.predictions)
    excludes_zero = (ci["low"] > 0.0) or (ci["high"] < 0.0)  # type: ignore[operator]
    if ci["low"] > 0.0:  # type: ignore[operator]
        verdict = "PILOT_BEATS"
    elif ci["high"] < 0.0:  # type: ignore[operator]
        verdict = "PILOT_LOSES"
    else:
        verdict = "STATISTICAL_TIE"
    result.update(
        {
            "delta_point": ci["point"],
            "delta_ci95": {"low": ci["low"], "high": ci["high"]},
            "ci_excludes_zero": excludes_zero,
            "decision_grade": excludes_zero,
            "verdict": verdict,
        }
    )
    return result


def per_family_analysis(record: dict) -> dict[str, dict[str, object]]:
    """Recompute intervals on the pilot record's per-family breakdown."""
    per_family = record.get("metrics", {}).get("per_family", {})
    _require(isinstance(per_family, dict) and per_family, "record lacks metrics.per_family")
    unknown = sorted(set(per_family) - set(FAMILIES))
    _require(not unknown, f"unexpected QA families in record: {unknown}")
    out: dict[str, dict[str, object]] = {}
    for family in FAMILIES:
        row = per_family.get(family)
        _require(isinstance(row, dict), f"per_family missing {family!r}")
        preds = int(row.get("span_predictions", 0))
        corr = int(row.get("span_correct", 0))
        ems = int(row.get("exact_matches", 0))
        tasks = int(row.get("tasks", 0))
        entry: dict[str, object] = {
            "tasks": tasks,
            "span_predictions": preds,
            "span_correct": corr,
            "span_accuracy": corr / preds if preds else 0.0,
            "span_accuracy_wilson95": wilson(corr, preds),
            "exact_matches": ems,
            "exact_match_rate": ems / tasks if tasks else 0.0,
            "exact_match_wilson95": wilson(ems, tasks),
        }
        out[family] = entry
    return out


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------


def analyze_pilot_record(
    record: dict,
    baselines: dict[str, dict],
) -> dict[str, object]:
    """Produce the full FINDINGS-ready analysis dict for the R2 pilot."""
    schema = record.get("schema_version")
    _require(
        schema == PILOT_RECORD_SCHEMA,
        f"pilot record schema mismatch: expected {PILOT_RECORD_SCHEMA!r}, got {schema!r}",
    )
    for name, baseline in baselines.items():
        bschema = baseline.get("schema_version")
        _require(
            bschema == BASELINE_RECORD_SCHEMA,
            f"baseline {name!r} schema mismatch: expected "
            f"{BASELINE_RECORD_SCHEMA!r}, got {bschema!r}",
        )

    pilot = parse_span_block(record, "pilot")
    _require(pilot.accuracy is not None, "pilot record has zero span predictions")

    comparisons = [
        compare_to_baseline(pilot, baseline)
        for _, baseline in sorted(baselines.items())
    ]

    degeneracy = record.get("metrics", {}).get("degeneracy", {})
    guard = guard_verdict(degeneracy, "pilot")

    free_run = record.get("metrics", {}).get("free_run", {})
    em_matches = int(free_run.get("exact_matches", 0))
    em_tasks = int(free_run.get("tasks", 0))

    beat_trigram = next(
        (c for c in comparisons if c["arm"] == "trigram_smoothed"), None
    )
    beat_unigram = next(
        (c for c in comparisons if c["arm"] == "unigram_smoothed"), None
    )
    mechanical_rules = [
        {
            "rule": "beat_trigram_smoothed",
            "question": "Does the pilot beat the trigram comparator (16.2754%)?",
            "verdict": beat_trigram["verdict"] if beat_trigram else "MISSING_BASELINE",
            "decision_grade": beat_trigram.get("decision_grade") if beat_trigram else False,
        },
        {
            "rule": "beat_unigram_smoothed",
            "question": "Does the pilot beat the unigram comparator (4.5383%)?",
            "verdict": beat_unigram["verdict"] if beat_unigram else "MISSING_BASELINE",
            "decision_grade": beat_unigram.get("decision_grade") if beat_unigram else False,
        },
        {
            "rule": "degeneracy_guard_clean",
            "question": "Do free-run completions pass the degeneracy guard?",
            "verdict": guard["verdict"],
            "decision_grade": guard["verdict"] == "PASS",
        },
    ]

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "inputs": {
            "pilot_schema": PILOT_RECORD_SCHEMA,
            "pilot_git_head": record.get("git_head"),
            "pilot_seed": record.get("seed"),
            "corpus_source": record.get("corpus_source"),
            "baseline_arms": sorted(baselines),
        },
        "headline": {
            "span_correct": pilot.correct,
            "span_predictions": pilot.predictions,
            "span_accuracy": pilot.accuracy,
            "span_accuracy_wilson95": wilson(pilot.correct, pilot.predictions),
            "free_run_exact_matches": em_matches,
            "free_run_tasks": em_tasks,
            "free_run_exact_match_rate": em_matches / em_tasks if em_tasks else 0.0,
            "free_run_exact_match_wilson95": wilson(em_matches, em_tasks),
        },
        "comparisons": comparisons,
        "degeneracy_guard": guard,
        "per_family": per_family_analysis(record),
        "mechanical_rules": mechanical_rules,
    }


# ---------------------------------------------------------------------------
# FINDINGS markdown rendering
# ---------------------------------------------------------------------------


def _fmt(x: object) -> str:
    return f"{float(x):.4f}"


def render_findings_markdown(analysis: dict) -> str:
    """Render the analysis dict as FINDINGS-style markdown."""
    head = analysis["headline"]
    inputs = analysis["inputs"]
    guard = analysis["degeneracy_guard"]

    lines: list[str] = []
    lines.append("# R2 Bible-QA pilot: post-verdict analysis (mechanical)")
    lines.append("")
    lines.append(
        f"Corpus: `{inputs['corpus_source']}` · pilot git head: "
        f"`{inputs['pilot_git_head']}` · seed: {inputs['pilot_seed']} · "
        f"baselines: {', '.join(inputs['baseline_arms'])}"
    )
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lo = head["span_accuracy_wilson95"]["low"]
    hi = head["span_accuracy_wilson95"]["high"]
    lines.append(
        f"- Teacher-forced span-acc: **{head['span_correct']}/{head['span_predictions']}"
        f" = {_fmt(head['span_accuracy'])}** (Wilson95 {_fmt(lo)}–{_fmt(hi)})"
    )
    lines.append(
        f"- Free-run exact match: {head['free_run_exact_matches']}/{head['free_run_tasks']}"
        f" = {_fmt(head['free_run_exact_match_rate'])} "
        f"(Wilson95 {_fmt(head['free_run_exact_match_wilson95']['low'])}–"
        f"{_fmt(head['free_run_exact_match_wilson95']['high'])})"
    )
    lines.append("")

    lines.append("## Pilot vs preregistered comparators (Newcombe 95% on the delta)")
    lines.append("")
    lines.append("| baseline arm | baseline acc | pilot acc | Δ (pilot − baseline) | 95% CI on Δ | verdict |")
    lines.append("|---|---|---|---|---|---|")
    for comp in analysis["comparisons"]:
        if comp["applicable"]:
            delta = comp["delta_ci95"]
            row = (
                f"| {comp['arm']} | {_fmt(comp['baseline_accuracy'])} "
                f"| {_fmt(comp['pilot_accuracy'])} "
                f"| {_fmt(comp['delta_point'])} "
                f"| [{_fmt(delta['low'])}, {_fmt(delta['high'])}]"
                f"{' *decisive*' if comp['decision_grade'] else ' (CI covers 0)'} "
                f"| {comp['verdict']} |"
            )
        else:
            row = f"| {comp['arm']} | n/a | {_fmt(comp['pilot_accuracy'])} | n/a | n/a | NOT_APPLICABLE |"
        lines.append(row)
    lines.append("")

    lines.append("## Degeneracy guard")
    lines.append("")
    reasons = "; ".join(guard["reasons"]) if guard["reasons"] else "clean"
    lines.append(
        f"- **{guard['verdict']}** ({reasons}) — distinct answers "
        f"{guard['distinct_answers']}, modal share {_fmt(guard['modal_share'])}, "
        f"modal answer: `{guard['modal_answer']}`"
    )
    for baseline_guard in analysis.get("baseline_guards", []):
        b_reasons = (
            "; ".join(baseline_guard["reasons"])
            if baseline_guard["reasons"]
            else "clean"
        )
        lines.append(
            f"- baseline `{baseline_guard['label']}`: {baseline_guard['verdict']} "
            f"({b_reasons})"
        )
    lines.append("")

    lines.append("## Per-family breakdown")
    lines.append("")
    lines.append("| family | tasks | span acc | Wilson95 | free-run EM |")
    lines.append("|---|---|---|---|---|")
    for family in FAMILIES:
        row = analysis["per_family"][family]
        lines.append(
            f"| {family} | {row['tasks']} "
            f"| {_fmt(row['span_accuracy'])} "
            f"| [{_fmt(row['span_accuracy_wilson95']['low'])}, "
            f"{_fmt(row['span_accuracy_wilson95']['high'])}] "
            f"| {row['exact_matches']}/{row['tasks']} |"
        )
    lines.append("")

    lines.append("## Mechanical rule evaluation (preregistered)")
    lines.append("")
    for index, rule in enumerate(analysis["mechanical_rules"], start=1):
        grade = "decision-grade" if rule["decision_grade"] else "not decision-grade"
        lines.append(f"{index}. **{rule['rule']}**: {rule['question']} → {rule['verdict']} ({grade})")
    lines.append("")
    lines.append(
        "*Generated by tools/analyze_bible_qa.py; pure function of the pilot "
        "and baseline records — no tuning, no post-hoc edits.*"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Deterministic fixture synthesis (pre-verdict smoke testing ONLY)
# ---------------------------------------------------------------------------


def _seed_stream(seed: int, label: str) -> bytes:
    return sha256(b"qa-r2-analysis-fixture\x00" + str(seed).encode() + b"\x00" + label.encode()).digest()


def _seed_int(seed: int, label: str, lo: int, hi: int) -> int:
    value = int.from_bytes(_seed_stream(seed, label)[:8], "big")
    return lo + value % (hi - lo + 1)


def _fake_digest(seed: int, label: str) -> str:
    return sha256(_seed_stream(seed, label)).hexdigest()


def synthesize_fixture_record(seed: int = 0) -> dict[str, object]:
    """Synthesize a schema-complete pilot record for smoke-testing the analyzer.

    Deterministic in ``seed``; structurally identical to
    ``run_pilot.main``'s record (same keys, same VAL shape: 60 tasks, three
    families × 20, 639 span predictions). Numbers are plausible but clearly
    flagged as synthetic via the ``synthetic_fixture`` block.
    """
    if type(seed) is not int:
        raise ValueError("seed must be an integer")

    total_preds = 639
    total_correct = _seed_int(seed, "span_correct", 118, 142)  # ~18.5–22%: above trigram
    # Split predictions/correct across families with realistic skew:
    # verse_continuation is the easiest family, who_said the hardest.
    fam_preds = {
        "address_lookup": _seed_int(seed, "preds_address_lookup", 208, 220),
        "verse_continuation": _seed_int(seed, "preds_verse_continuation", 200, 212),
        "who_said": _seed_int(seed, "preds_who_said", 205, 215),
    }
    fam_correct_frac = {
        "address_lookup": 0.20,
        "verse_continuation": 0.28,
        "who_said": 0.11,
    }
    allocated = 0
    per_family: dict[str, dict[str, object]] = {}
    fam_order = sorted(fam_preds)
    for index, family in enumerate(fam_order):
        if index == len(fam_order) - 1:
            preds = total_preds - allocated
        else:
            allocated += fam_preds[family]
            preds = fam_preds[family]
        correct = round(total_correct * fam_correct_frac[family])
        per_family[family] = {
            "tasks": 20,
            "span_predictions": preds,
            "span_correct": correct,
            "span_accuracy": correct / preds,
            "span_accuracy_wilson95": wilson(correct, preds),
            "exact_matches": _seed_int(seed, f"em_{family}", 0, 2),
            "exact_match_rate": 0.0,
        }
        per_family[family]["exact_match_rate"] = (  # type: ignore[index]
            per_family[family]["exact_matches"] / 20  # type: ignore[operator]
        )

    em_matches = sum(int(row["exact_matches"]) for row in per_family.values())  # type: ignore[arg-type]

    distinct = _seed_int(seed, "distinct", 34, 46)
    modal_share = _seed_int(seed, "modal_share_ppm", 110_000, 190_000) / 1_000_000

    record: dict[str, object] = {
        "schema_version": PILOT_RECORD_SCHEMA,
        "generated_utc": "1970-01-01T00:00:00+00:00",
        "git_head": "fixture" + _fake_digest(seed, "git_head")[:7],
        "protocol": "SYNTHETIC FIXTURE — mirrors run_pilot record shape for analyzer smoke tests",
        "corpus_source": "bible-kjv-pg10:v3-freeze-candidate",
        "corpus_digests": {
            "trimmed_sha256": _fake_digest(seed, "trimmed"),
            "val_manifest_sha256": _fake_digest(seed, "val_manifest"),
        },
        "seed": seed,
        "execution": {"requested_device": "cuda", "device": "cuda"},
        "model": {"vocabulary_size": 4096, "max_phrases": 4096},
        "train_stream": {"books": 49, "epochs": 2},
        "metrics": {
            "answer_span": {
                "tasks": 60,
                "predictions": total_preds,
                "correct": total_correct,
                "accuracy": total_correct / total_preds,
                "accuracy_wilson95": wilson(total_correct, total_preds),
            },
            "free_run": {
                "tasks": 60,
                "exact_matches": em_matches,
                "exact_match_rate": em_matches / 60,
            },
            "degeneracy": {
                "evaluated": 60,
                "distinct_answers": distinct,
                "modal_answer": "and the lord said unto moses",
                "modal_share": modal_share,
                "guard": {
                    "modal_share_max": GUARD_MODAL_SHARE_MAX_DEFAULT,
                    "distinct_answers_min": GUARD_DISTINCT_MIN_DEFAULT,
                },
                "guard_pass": distinct >= GUARD_DISTINCT_MIN_DEFAULT
                and modal_share <= GUARD_MODAL_SHARE_MAX_DEFAULT,
            },
            "per_family": per_family,
        },
        "synthetic_fixture": {
            "notice": "Not a real run: generated by qa_analysis.synthesize_fixture_record",
            "seed": seed,
        },
    }
    return record


def canonical_json_bytes(value: object) -> bytes:
    """Canonical JSON encoding shared by the analyzer's outputs/tests."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def load_baseline_records(baselines_dir) -> dict[str, dict]:
    """Load every ``*.json`` baseline record from a directory, keyed by arm."""
    from pathlib import Path

    directory = Path(baselines_dir)
    _require(directory.is_dir(), f"baselines directory not found: {directory}")
    baselines: dict[str, dict] = {}
    for path in sorted(directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != BASELINE_RECORD_SCHEMA:
            raise ValueError(f"{path.name}: unexpected schema {payload.get('schema_version')!r}")
        arm = str(payload["arm"])
        _require(arm not in baselines, f"duplicate baseline arm {arm!r}")
        baselines[arm] = payload
    _require(baselines, f"no baseline records found under {directory}")
    return baselines
