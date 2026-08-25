"""M0/M1 preregistered battery runner — python_qa channel, CUDA cell system.

Executes the full preregistered M0/M1 milestone battery from
``docs/research/gemma4-comparison-design.md`` (§Milestone ladder), MILESTONES.md
(§Decision rule) and ``experiments/python_qa_scale/BridgeFindings.md`` (§4,
gate framing) the moment an architecture passes:

    k=3 seeds {17, 18, 19} x tokenizers {concept, byte} on the FROZEN
    python_qa corpus (75 TRAIN grow / 25 VAL eval), N=131072 CUDA cells,
    sparse_three_factor readout, 2 growth epochs — then evaluates both
    milestone gates against the LOCKED n-gram comparators in
    ``experiments/python_qa_baselines_v1/``:

        M0: pooled cell answer-span accuracy beats the random/unigram floor
            (unigram_smoothed, span-acc 0.0 = 0/104) with Newcombe CI
            separation AND every scored record passing the degeneracy guard.
        M1: pooled cell answer-span accuracy beats the smoothed trigram
            (trigram_smoothed, span-acc 5.769% = 6/104) with Newcombe CI
            separation.

    CI convention (repo standard): Newcombe hybrid-score method-10 intervals
    built from Wilson score intervals at z = 1.959963984540054, computed on
    seed-POOLED proportions per arm (same construction as
    experiments/codebook_compare/_analyze.py and
    experiments/replay_compare/_analyze_gates.py); win = interval excluding
    zero on the positive side. Secondary view reported: non-overlap of the
    two pooled Wilson intervals (the MILESTONES.md phrasing).

Contract tripwires: every scored cell runs through
``humanbrain.contract_audit.audit()`` — before and after each scored run the
decorator fails fast if torch/autograd machinery entered ``sys.modules``
(the frozen-evaluation contract; weights are grown encapsulated inside
``qa_experiments.run``, so the import tripwire is the enforceable seam here).
The audit is mandatory on scored cells — there is no flag that disables it.

Modes:
    default      execute the full battery + gate evaluation (GPU minutes).
    --dry-run    validate corpus integrity, baseline comparators, harness
                 wiring and audit wiring WITHOUT constructing any circuit —
                 no GPU minutes spent. Reports BLOCKED items (e.g. the
                 ``--corpus python_qa`` seam landing via codex/swe-m01-harness)
                 without failing unless --strict is given.

Usage:
    python experiments/swe_m01_runs/run_m0m1.py --dry-run
    python experiments/swe_m01_runs/run_m0m1.py                    # the battery
    python experiments/swe_m01_runs/run_m0m1.py --device cpu --cells 96   # smoke

Pricing (measured, BridgeFindings.md §4): ~75 s/seed concept arm @131k =>
~4 min for the concept row + ~8 min byte row of exclusive RTX-4070-class GPU.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Repo bootstrap: bind THIS worktree's src/humanbrain before anything imports
# the package (the dev venv's editable install points at the main checkout).
REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = str(REPO_ROOT / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

SCHEMA_VERSION = "humanbrain/swe-m01-runs/v1"

# Pinned in experiments/python_qa_baselines_v1/FINDINGS.md BEFORE any cell run;
# identical to humanbrain.python_qa_adapter.PINNED_CORPUS_SHA256 on the harness
# branch. Every scored record must carry this corpus digest or the gates refuse
# to evaluate (comparability with the locked n-gram comparators).
PINNED_CORPUS_SHA256 = "789e3ac592fd8aa1d4f68af8704a4a9b19d64db7277efea59ebc50ff8dd446ec"
BASELINES_DIR = REPO_ROOT / "experiments" / "python_qa_baselines_v1"
CORPUS_DIR = REPO_ROOT / "experiments" / "python_qa_corpus"

DEFAULT_SEEDS = (17, 18, 19)
DEFAULT_TOKENIZERS = ("concept", "byte")
DEFAULT_CELLS = 131_072
DEFAULT_GROWTH_EPOCHS = 2
DEFAULT_CONCEPT_MIN_FREQUENCY = 2          # lowered pilot gates, BridgeFindings §3/§4
DEFAULT_CONCEPT_MIN_MI_BITS = 1.0

# Newcombe/Wilson convention shared with codebook_compare + replay_compare.
Z = 1.959963984540054


def _load_benchmarks_module():
    """Import benchmarks/qa_experiments.py by path (benchmarks is not a package)."""
    path = REPO_ROOT / "benchmarks" / "qa_experiments.py"
    spec = importlib.util.spec_from_file_location("swe_m01_qa_experiments", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load harness module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_QA = _load_benchmarks_module()

from humanbrain import contract_audit  # noqa: E402  (after src bootstrap)
from humanbrain.python_qa_corpus import canonical_json  # noqa: E402


def _require_worktree_binding() -> None:
    """Fail fast if humanbrain resolved to another checkout."""
    import humanbrain

    resolved = Path(humanbrain.__file__).resolve()
    if not str(resolved).startswith(str(REPO_ROOT)):
        raise RuntimeError(
            f"humanbrain imported from {resolved}, not this worktree ({REPO_ROOT}); "
            "refusing to score against the wrong tree"
        )


# ---------------------------------------------------------------------------
# Contract-audit wiring (proof-of-rollout). The decorator form is deliberate:
# scored cells may only ever execute through this audited entrypoint.

def _scored_cell(config: Any) -> dict[str, Any]:
    """One scored grow+evaluate pass through the QA harness (uncached)."""
    return _QA.run(config)


@contract_audit.audit
def scored_cell(config: Any) -> dict[str, Any]:
    """AUDITED scored cell: contract_audit.audit brackets the run with the
    no-forbidden-trainable-machinery tripwire (torch/autograd must never be
    loaded while numbers are produced)."""
    return _scored_cell(config)


@contract_audit.audit
def scored_determinism_precheck(config: Any) -> dict[str, Any]:
    """AUDITED double-run determinism probe on a short slice (BridgeFindings
    §4: run once on a short slice BEFORE unsealing the full battery)."""
    return _QA.check_determinism(config)


# ---------------------------------------------------------------------------
# CI math — same conventions as experiments/codebook_compare/_analyze.py.

def wilson(x: int, n: int, z: float = Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = x / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = z * ((p * (1.0 - p) / n + z * z / (4.0 * n * n)) ** 0.5) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def newcombe_diff(
    x1: int, n1: int, x2: int, n2: int, z: float = Z
) -> tuple[float, float, float] | None:
    """Newcombe hybrid-score CI for p1 - p2 (independent proportions).

    Method 10 of Newcombe (1998): square-and-add from the two Wilson
    intervals. Returns None when either denominator is zero.
    """
    if n1 == 0 or n2 == 0:
        return None
    l1, u1 = wilson(x1, n1, z)
    l2, u2 = wilson(x2, n2, z)
    p1, p2 = x1 / n1, x2 / n2
    d = p1 - p2
    lower = d - ((p1 - l1) ** 2 + (u2 - p2) ** 2) ** 0.5
    upper = d + ((u1 - p1) ** 2 + (p2 - l2) ** 2) ** 0.5
    return (d, lower, upper)


# ---------------------------------------------------------------------------
# Frozen-corpus verification (works pre-merge: uses humanbrain.python_qa_corpus,
# which exists on this branch; the adapter re-checks the same invariants).

def verify_frozen_corpus() -> dict[str, Any]:
    """Verify artifact bytes vs manifest, recompute the entries digest and
    compare against the pinned M1 value. Raises on any mismatch."""
    artifact_path = CORPUS_DIR / "python_qa_corpus_v1.json"
    manifest_path = CORPUS_DIR / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(raw).hexdigest()
    if artifact_sha256 != str(manifest["artifact_sha256"]):
        raise ValueError(
            f"corpus artifact digest mismatch: {artifact_path} != manifest "
            f"artifact_sha256 {manifest['artifact_sha256']}"
        )
    artifact = json.loads(raw.decode("utf-8"))
    entries = artifact["entries"]
    recomputed = hashlib.sha256(canonical_json({"entries": entries})).hexdigest()
    if recomputed != PINNED_CORPUS_SHA256:
        raise ValueError(
            f"pinned python-QA corpus changed: {recomputed!r} != "
            f"M1 comparator pin {PINNED_CORPUS_SHA256!r}"
        )
    counts = artifact["counts"]
    for cohort in ("TRAIN", "VAL"):
        if int(counts[cohort]["total"]) != len(entries[cohort]):
            raise ValueError(f"manifest count mismatch for cohort {cohort}")
    train_n = len(entries["TRAIN"])
    val_n = len(entries["VAL"])
    if (train_n, val_n) != (75, 25):
        raise ValueError(f"expected frozen cohorts 75/25, got {train_n}/{val_n}")
    return {
        "artifact_sha256": artifact_sha256,
        "corpus_sha256": recomputed,
        "train_tasks": train_n,
        "val_tasks": val_n,
        "schema_version": artifact["schema_version"],
    }


# ---------------------------------------------------------------------------
# Locked baseline comparators.

def load_comparators() -> dict[str, dict[str, Any]]:
    """Load + verify the locked n-gram comparator records.

    M0 comparator = unigram_smoothed (the random/unigram floor on this
    channel — its argmax is always the modal training token `self`, which
    never occurs inside a VAL answer span, hence exactly 0/104).
    M1 comparator = trigram_smoothed (5.769% = 6/104).
    majority_floor is NOT CI-comparable (guard-fails by construction, 0
    predictions) and is deliberately not a gate input.
    """
    comparators: dict[str, dict[str, Any]] = {}
    for milestone, arm in (("M0", "unigram_smoothed"), ("M1", "trigram_smoothed")):
        record = json.loads((BASELINES_DIR / f"{arm}.json").read_text(encoding="utf-8"))
        if record.get("corpus_sha256") != PINNED_CORPUS_SHA256:
            raise ValueError(
                f"{arm}.json corpus_sha256 {record.get('corpus_sha256')!r} != pin — "
                "comparators are keyed to the frozen corpus; refusing mixed-digest gates"
            )
        span = record["metrics"]["answer_span"]
        free_run = record["metrics"]["free_run"]
        comparators[milestone] = {
            "milestone": milestone,
            "arm": arm,
            "record": f"experiments/python_qa_baselines_v1/{arm}.json",
            "span_correct": int(span["correct"]),
            "span_predictions": int(span["predictions"]),
            "span_accuracy": float(span["accuracy"]),
            "free_run_matches": int(free_run["exact_matches"]),
            "free_run_tasks": int(free_run["tasks"]),
            "degeneracy_guard_pass": bool(record["metrics"]["degeneracy"]["guard_pass"]),
        }
    expected_m1_acc = 6 / 104
    if abs(comparators["M1"]["span_accuracy"] - expected_m1_acc) > 1e-12:
        raise ValueError("trigram comparator drifted from the locked 5.77% (6/104)")
    if comparators["M0"]["span_correct"] != 0:
        raise ValueError("unigram floor comparator drifted from the locked 0/104")
    return comparators


# ---------------------------------------------------------------------------
# Harness-seam detection (what dry-run reports as BLOCKED before the merge).

def detect_harness_seam() -> dict[str, Any]:
    """Is the ``--corpus python_qa`` seam available in this tree?"""
    config_fields = {f.name for f in dataclasses.fields(_QA.ExperimentConfig)}
    has_corpus_field = "corpus" in config_fields
    try:
        importlib.import_module("humanbrain.python_qa_adapter")
        adapter_ok = True
        adapter_error = None
    except ImportError as exc:
        adapter_ok = False
        adapter_error = str(exc)
    ready = has_corpus_field and adapter_ok
    return {
        "ready": ready,
        "experiment_config_has_corpus_field": has_corpus_field,
        "python_qa_adapter_importable": adapter_ok,
        "adapter_error": adapter_error,
        "instruction": (
            "merge codex/swe-m01-harness (1d4d351: python_qa_adapter + "
            "--corpus seam) into this line, then rerun"
            if not ready
            else None
        ),
    }


def _build_cell_config(args: argparse.Namespace, seed: int, tokenizer: str) -> Any:
    """ExperimentConfig for one preregistered cell (BridgeFindings §4 pilot)."""
    return _QA.ExperimentConfig(
        seed=seed,
        max_train_tasks=args.max_train_tasks,
        growth_epochs=args.growth_epochs,
        max_phrases=512,
        min_phrase_frequency=2,
        circuit=_QA.RecurrentCircuitConfig(
            reservoir_cells=args.cells,
            input_fanout=24,
            recurrent_fanout=8,
        ),
        eval_limit=args.eval_limit,
        completion_token_slack=args.completion_token_slack,
        corpus="python_qa",
        device=args.device,
        readout_rule=args.readout_rule,
        tokenizer=tokenizer,
        # Concept-arm mining gates are part of the frozen pilot registration
        # (lowered from the study protocol; see BridgeFindings §3).
        concept_min_frequency=DEFAULT_CONCEPT_MIN_FREQUENCY,
        concept_min_mutual_information_bits=DEFAULT_CONCEPT_MIN_MI_BITS,
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False).encode("utf-8"))
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Battery execution.

def run_battery(args: argparse.Namespace) -> int:
    _require_worktree_binding()
    corpus_info = verify_frozen_corpus()
    comparators = load_comparators()
    seam = detect_harness_seam()
    if not seam["ready"]:
        print(json.dumps({
            "error": "harness seam unavailable",
            "detail": seam,
            "remedy": "dry-run mode validates everything else: run with --dry-run",
        }, indent=2))
        return 2

    seeds = list(args.seeds)
    tokenizers = list(args.tokenizers)
    records_dir = args.out_dir / "records"
    planned = [(t, s) for t in tokenizers for s in seeds]
    # Partial-slice smoke configs get distinct record names so they can never
    # be resumed later as full-cohort preregistered cells.
    slice_suffix = ""
    if args.max_train_tasks is not None:
        slice_suffix += f"__tr{args.max_train_tasks}"
    if args.eval_limit is not None:
        slice_suffix += f"__ev{args.eval_limit}"

    deviations = []
    if args.max_train_tasks is not None or args.eval_limit is not None:
        deviations.append(
            "partial-slice smoke configuration (--max-train-tasks/--eval-limit): "
            "NOT the preregistered full-cohort battery; do not read gates as M0/M1 claims"
        )
    if (args.cells, args.device) != (DEFAULT_CELLS, "cuda"):
        deviations.append(
            f"non-pilot execution point: cells={args.cells}, device={args.device}"
        )

    cell_entries: list[dict[str, Any]] = []
    arms: dict[str, dict[str, Any]] = {
        t: {"records": []} for t in tokenizers
    }

    for tokenizer, seed in planned:
        record_path = records_dir / (
            f"python_qa__{tokenizer}__n{args.cells}__seed{seed}{slice_suffix}.json"
        )
        if record_path.exists() and not args.force:
            record = json.loads(record_path.read_text(encoding="utf-8"))
            source = "resumed-from-disk"
        else:
            config = _build_cell_config(args, seed, tokenizer)
            started = time.perf_counter()
            try:
                record = scored_cell(config)  # AUDITED (contract tripwires)
            except contract_audit.ContractViolation as exc:
                print(json.dumps({
                    "error": "contract violation during scored cell",
                    "cell": {"tokenizer": tokenizer, "seed": seed},
                    "detail": str(exc),
                }, indent=2))
                return 3
            wall_s = time.perf_counter() - started
            record["wall_s"] = wall_s
            _atomic_write_json(record_path, record)
            source = "executed"

        if record.get("corpus_sha256") != PINNED_CORPUS_SHA256:
            raise ValueError(
                f"{record_path}: record corpus_sha256 {record.get('corpus_sha256')!r} "
                f"!= pinned {PINNED_CORPUS_SHA256!r} — not comparable to M1 comparators"
            )
        if record.get("corpus_arm", "python_qa") != "python_qa":
            raise ValueError(f"{record_path}: unexpected corpus_arm")

        guard_pass = bool(record["metrics"]["degeneracy"]["guard_pass"])
        span = record["metrics"]["answer_span"]
        free_run = record["metrics"]["free_run"]
        try:
            record_ref = str(record_path.relative_to(REPO_ROOT))
        except ValueError:  # out-dir outside the repo (e.g. scratch validation)
            record_ref = str(record_path)
        entry = {
            "tokenizer": tokenizer,
            "seed": seed,
            "record": record_ref,
            "source": source,
            "reservoir_cells": int(record["execution"]["reservoir_cells"]),
            "kernel_backend": str(record["execution"]["kernel_backend"]),
            "span_predictions": int(span["predictions"]),
            "span_correct": int(span["correct"]),
            "span_accuracy": float(span["accuracy"]),
            "free_run_matches": int(free_run["exact_matches"]),
            "free_run_tasks": int(free_run["tasks"]),
            "degeneracy_guard_pass": guard_pass,
        }
        cell_entries.append(entry)
        arms[tokenizer]["records"].append(entry)
        print(
            f"[cell] tokenizer={tokenizer} seed={seed} span_acc={entry['span_accuracy']:.5f} "
            f"({entry['span_correct']}/{entry['span_predictions']}) "
            f"EM={entry['free_run_matches']}/{entry['free_run_tasks']} "
            f"guard={'PASS' if guard_pass else 'FAIL'} [{source}]",
            flush=True,
        )

    summary = build_summary(
        args=args,
        corpus_info=corpus_info,
        comparators=comparators,
        cell_entries=cell_entries,
        arms=arms,
        deviations=deviations,
    )
    _atomic_write_json(args.out_dir / "battery_summary.json", summary)
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))
    print(json.dumps(summary["verdicts"], indent=2, sort_keys=True))
    print(f"[summary] written: {(args.out_dir / 'battery_summary.json').resolve()}")
    verdict_values = [v for v in summary["verdicts"].values()]
    if all(v == "PASS" for v in verdict_values):
        return 0
    print(
        "[gates] not all milestone gates PASS "
        f"({summary['verdicts']}) — battery executed; verdicts above are final"
    )
    return 1


# ---------------------------------------------------------------------------
# Gate evaluation.

def evaluate_gates(
    arms: dict[str, dict[str, Any]],
    comparators: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Pooled-proportion Newcombe gates per milestone, per tokenizer arm.

    Primary criterion (repo standard, cf. replay_compare/_analyze_gates.py):
    win = Newcombe method-10 95% CI of (cell − baseline) excluding zero on
    the positive side. Secondary reported criterion (MILESTONES.md phrasing):
    non-overlap of the two pooled Wilson 95% intervals.
    """
    gates: dict[str, Any] = {}
    verdicts: dict[str, str] = {}
    for milestone in ("M0", "M1"):
        comp = comparators[milestone]
        cx, cn = comp["span_correct"], comp["span_predictions"]
        per_arm: dict[str, Any] = {}
        arm_pass_flags: list[bool] = []
        for arm_name, arm in sorted(arms.items()):
            records = arm["records"]
            if not records:
                per_arm[arm_name] = {"verdict": "PENDING", "reason": "no records"}
                continue
            px = sum(r["span_correct"] for r in records)
            pn = sum(r["span_predictions"] for r in records)
            guard_all = all(r["degeneracy_guard_pass"] for r in records)
            diff = newcombe_diff(px, pn, cx, cn)
            wl, wu = wilson(px, pn)
            bl, bu = wilson(cx, cn)
            primary_win = diff is not None and diff[1] > 0.0
            secondary_win = wl > bu  # pooled-Wilson non-overlap view
            gate_pass = primary_win and (guard_all if milestone == "M0" else True)
            arm_verdict = (
                "PASS" if gate_pass else ("FAIL" if diff is not None else "PENDING")
            )
            arm_pass_flags.append(gate_pass)
            per_arm[arm_name] = {
                "verdict": arm_verdict,
                "cell_pooled_span": {"correct": px, "predictions": pn, "accuracy": px / pn},
                "baseline_pooled_span": {
                    "correct": cx, "predictions": cn, "accuracy": cx / cn,
                },
                "newcombe_diff": {
                    "delta": diff[0], "ci95_low": diff[1], "ci95_high": diff[2],
                    "excludes_zero_positive": primary_win,
                } if diff is not None else None,
                "wilson_pooled_cell_95": [wl, wu],
                "wilson_pooled_baseline_95": [bl, bu],
                "wilson_intervals_nonoverlap": secondary_win,
                "degeneracy_guard_all_pass": guard_all,
                "degeneracy_guard_is_gate": milestone == "M0",
                "per_seed_span_accuracy": [
                    {"seed": r["seed"], "accuracy": r["span_accuracy"]}
                    for r in records
                ],
            }
        gates[milestone] = {
            "comparator": comp,
            "criterion": (
                "pooled Newcombe method-10 95% CI of (cell - baseline) excludes zero "
                "on the positive side" + (
                    "; degeneracy guard must pass on every scored record"
                    if milestone == "M0" else ""
                )
            ),
            "per_tokenizer_arm": per_arm,
        }
        if not arm_pass_flags:
            verdicts[milestone] = "PENDING"
        elif all(arm_pass_flags):
            verdicts[milestone] = "PASS"
        else:
            verdicts[milestone] = "PARTIAL" if any(arm_pass_flags) else "FAIL"
    return gates, verdicts


def build_summary(
    args: argparse.Namespace,
    corpus_info: dict[str, Any],
    comparators: dict[str, dict[str, Any]],
    cell_entries: list[dict[str, Any]],
    arms: dict[str, dict[str, Any]],
    deviations: list[str],
) -> dict[str, Any]:
    gates, verdicts = evaluate_gates(arms, comparators)

    def git(field: str) -> str | None:
        try:
            return subprocess.run(
                ["git", field], cwd=str(REPO_ROOT), capture_output=True, text=True,
                timeout=10, check=True,
            ).stdout.strip() or None
        except Exception:
            return None

    claim_grammar = [
        "cell system beats unigram_smoothed (random floor) on python_qa VAL "
        "answer-span accuracy at k=3 seeds, Newcombe method-10 95% CI excluding "
        "zero — M0, degeneracy guard passing on every scored record",
        "cell system beats trigram_smoothed (5.77%) on python_qa VAL answer-span "
        "accuracy at k=3 seeds, Newcombe method-10 95% CI excluding zero — M1",
        "nothing stronger than the above grammar is claimed (fairness rule 5, "
        "gemma4-comparison-design.md)",
    ]
    notes = [
        "Prediction COUNTS are not comparable across tokenizer conventions "
        "(cell arms count byte/concept-token races; n-gram comparators count "
        "word-level predictions over the same 25 VAL tasks) — compare "
        "accuracies and CI separation, never raw counts.",
        "Free-run exact match is reported for transparency; both locked n-gram "
        "comparators sit at 0/25 EM, and the preregistered M0/M1 gates are "
        "answer-span accuracy gates.",
        "majority_floor is not a gate input: it fails the degeneracy guard by "
        "construction (1 distinct completion, share 1.0) and makes 0 span "
        "predictions, so no CI can be formed.",
        "Weights are grown encapsulated inside qa_experiments.run(); the "
        "enforceable frozen-evaluation seam at this boundary is the "
        "contract_audit forbidden-import tripwire bracketing every scored cell.",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "runner": "experiments/swe_m01_runs/run_m0m1.py",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
        "git": {"head": git("rev-parse HEAD"), "branch": git("branch --show-current")},
        "protocol": (
            "preregistered M0/M1 battery: k=3 seeds x {concept, byte} growth "
            "tokenizers on the FROZEN python_qa corpus (grow 75 TRAIN / evaluate "
            "25 VAL), CUDA full-touch reservoir, sparse_three_factor readout; "
            "gates evaluated against the LOCKED n-gram comparators in "
            "experiments/python_qa_baselines_v1/ under the shared Newcombe/"
            "Wilson decision rule (MILESTONES.md)"
        ),
        "config": {
            "seeds": list(args.seeds),
            "tokenizers": list(args.tokenizers),
            "reservoir_cells": args.cells,
            "device": args.device,
            "readout_rule": args.readout_rule,
            "growth_epochs": args.growth_epochs,
            "completion_token_slack": args.completion_token_slack,
            "concept_min_frequency": DEFAULT_CONCEPT_MIN_FREQUENCY,
            "concept_min_mutual_information_bits": DEFAULT_CONCEPT_MIN_MI_BITS,
            "max_train_tasks": args.max_train_tasks,
            "eval_limit": args.eval_limit,
        },
        "corpus": {
            "source": "repo-src:humanbrain:python-qa-corpus-v1",
            **corpus_info,
        },
        "comparators": comparators,
        "cells": cell_entries,
        "gates": gates,
        "verdicts": verdicts,
        "claim_grammar": claim_grammar,
        "notes": notes,
        "deviations": deviations,
    }


# ---------------------------------------------------------------------------
# Dry-run: validate corpus loading + harness + audit wiring. No circuits, no
# GPU minutes.

def _check(name: str, status: str, detail: Any = None) -> dict[str, Any]:
    check = {"name": name, "status": status}
    if detail is not None:
        check["detail"] = detail
    marker = {"PASS": "+", "FAIL": "!", "BLOCKED": "~", "WARN": "?"}[status]
    print(f"[dry-run] {marker} {name}: {status}" + (f" — {detail}" if isinstance(detail, str) else ""))
    return check


def run_dry_run(strict: bool) -> int:
    checks: list[dict[str, Any]] = []
    print(f"[dry-run] repo root: {REPO_ROOT}")

    # 1. Worktree binding.
    try:
        _require_worktree_binding()
        checks.append(_check("humanbrain_binds_this_worktree", "PASS", _SRC))
    except Exception as exc:  # pragma: no cover - defensive
        checks.append(_check("humanbrain_binds_this_worktree", "FAIL", str(exc)))

    # 2. Frozen corpus integrity (digest chain + cohorts) — no GPU.
    try:
        info = verify_frozen_corpus()
        checks.append(_check("frozen_corpus_pin_integrity", "PASS", info))
    except Exception as exc:
        checks.append(_check("frozen_corpus_pin_integrity", "FAIL", str(exc)))

    # 3. Locked comparators load + match their locked values.
    try:
        comparators = load_comparators()
        checks.append(_check(
            "locked_comparators",
            "PASS",
            {m: f"{c['arm']} span_acc={c['span_accuracy']:.5f} "
                 f"({c['span_correct']}/{c['span_predictions']})"
             for m, c in comparators.items()},
        ))
    except Exception as exc:
        checks.append(_check("locked_comparators", "FAIL", str(exc)))

    # 4. Harness seam (--corpus python_qa) — lands via codex/swe-m01-harness.
    seam = detect_harness_seam()
    checks.append(_check(
        "harness_python_qa_seam",
        "PASS" if seam["ready"] else "BLOCKED",
        seam["instruction"] if not seam["ready"] else {
            "experiment_config_has_corpus_field": seam["experiment_config_has_corpus_field"],
            "python_qa_adapter_importable": seam["python_qa_adapter_importable"],
        },
    ))

    # 5. Contract-audit wiring: decorator applied + tripwire actually fires.
    if getattr(scored_cell, "__wrapped__", None) is None:
        checks.append(_check("contract_audit_wiring", "FAIL", "audit decorator not applied"))
    else:
        fake = type(sys)("torch")  # negative probe: inject, expect ContractViolation
        inserted = "torch" not in sys.modules
        if inserted:
            sys.modules["torch"] = fake
        try:
            try:
                scored_cell(None)  # must fail BEFORE touching the harness
                status, detail = "FAIL", "tripwire did not fire on injected torch module"
            except contract_audit.ContractViolation:
                status, detail = "PASS", "audit tripwire fires on forbidden-module injection"
            except Exception as exc:  # noqa: BLE001 - probe reports any other outcome
                status, detail = "FAIL", f"unexpected error from probe: {exc!r}"
        finally:
            if inserted:
                del sys.modules["torch"]
        checks.append(_check("contract_audit_wiring", status, detail))

    # 6. CUDA presence — advisory only in dry-run.
    try:
        probe = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            checks.append(_check("cuda_gpu_present", "PASS", probe.stdout.strip()))
        else:
            checks.append(_check(
                "cuda_gpu_present", "WARN",
                "nvidia-smi found no GPU — the 131k pilot needs CUDA (dry-run unaffected)",
            ))
    except FileNotFoundError:
        checks.append(_check(
            "cuda_gpu_present", "WARN",
            "nvidia-smi not on PATH — cannot confirm GPU (dry-run unaffected)",
        ))

    # 7. Output dir writable.
    try:
        probe = args_out_probe()
        checks.append(probe if probe else _check("output_dir_writable", "PASS"))
    except Exception as exc:
        checks.append(_check("output_dir_writable", "FAIL", str(exc)))

    hard_failures = [c for c in checks if c["status"] == "FAIL"]
    blocked = [c for c in checks if c["status"] == "BLOCKED"]
    print(
        f"[dry-run] checks: {len(checks)} total, {len(hard_failures)} FAIL, "
        f"{len(blocked)} BLOCKED, "
        f"{sum(1 for c in checks if c['status'] == 'WARN')} WARN"
    )
    if hard_failures:
        print("[dry-run] RESULT: BROKEN — fix FAIL items before any scored run")
        return 1
    if blocked:
        print(
            "[dry-run] RESULT: READY-PENDING-MERGE — corpus/comparators/audit "
            "validated; scoring unlocks when the harness seam merges "
            "(codex/swe-m01-harness)" + (" [--strict would exit 1]" if not strict else "")
        )
        return 1 if strict else 0
    print("[dry-run] RESULT: READY — full battery can execute unchanged")
    return 0


def args_out_probe() -> dict[str, Any] | None:
    out_dir = REPO_ROOT / "experiments" / "swe_m01_runs"
    probe = out_dir / ".write_probe"
    probe.write_bytes(b"ok")
    probe.unlink()
    return None


# ---------------------------------------------------------------------------

def _parse_seed_list(text: str) -> list[int]:
    seeds = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("--seeds must name at least one integer seed")
    return seeds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true",
                        help="validate corpus loading + harness/audit wiring; no GPU minutes")
    parser.add_argument("--strict", action="store_true",
                        help="with --dry-run: exit 1 when items are BLOCKED (pending merge), not just FAIL")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda",
                        help="execution backend forwarded to the harness (default cuda)")
    parser.add_argument("--cells", type=int, default=DEFAULT_CELLS,
                        help=f"reservoir cell count (default {DEFAULT_CELLS})")
    parser.add_argument("--readout-rule",
                        choices=("auto", "ridge", "three_factor", "sparse_three_factor"),
                        default="auto",
                        help="auto selects sparse_three_factor once the dense ridge breaches 2 GiB")
    parser.add_argument("--seeds", type=_parse_seed_list, default=list(DEFAULT_SEEDS),
                        help="comma-separated k seeds (default 17,18,19)")
    parser.add_argument("--tokenizers", nargs="+", choices=DEFAULT_TOKENIZERS,
                        default=list(DEFAULT_TOKENIZERS),
                        help="growth-tokenizer arms to run (default both)")
    parser.add_argument("--growth-epochs", type=int, default=DEFAULT_GROWTH_EPOCHS)
    parser.add_argument("--completion-token-slack", type=int, default=8)
    parser.add_argument("--max-train-tasks", type=int, default=None,
                        help="DEVIATION from preregistration: slice TRAIN (smoke only)")
    parser.add_argument("--eval-limit", type=int, default=None,
                        help="DEVIATION from preregistration: slice VAL (smoke only)")
    parser.add_argument("--force", action="store_true",
                        help="re-execute cells whose records already exist")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="output directory (default: experiments/swe_m01_runs)")
    args = parser.parse_args(argv)
    args.out_dir = args.out_dir or (REPO_ROOT / "experiments" / "swe_m01_runs")

    if args.dry_run:
        return run_dry_run(args.strict)
    if sys.version_info[:2] != (3, 11):
        print(
            f"[warn] python {'.'.join(map(str, sys.version_info[:2]))}: the SWE stack "
            "pins python-3.11; proceeding but noting the deviation",
            flush=True,
        )
    return run_battery(args)


if __name__ == "__main__":
    raise SystemExit(main())
