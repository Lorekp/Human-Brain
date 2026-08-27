"""Search minimal integer-pA repairs for rejected electrical-v3 boundary rows.

This is a measured Change Note candidate, not a protocol refreeze. It reuses the
current electrical-v3 extrema selection and physical coupled-race probe, reads no
corpus or held-out data, and never executes q4 or C1.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import calibrate_minicolumn_electrical as electrical  # noqa: E402

ARTIFACT_PATH = Path(__file__).with_name("electrical_v3_current_search.json")
ARTIFACT_ID = "r3-electrical-v3-current-search-v1"
FAILED_ROWS = ("active_vs_matching", "novel_rank_boundary")
UNCHANGED_ROWS = ("matching_vs_novel",)


def _sha256_file(path: Path) -> str:
    raw = electrical._canonical_source_bytes(path.read_bytes())
    return hashlib.sha256(raw).hexdigest()


def _fixtures() -> tuple[tuple[int, Any, tuple[int, ...], int], ...]:
    rows = []
    for seed in electrical.CONSTRUCTION_SEEDS:
        cortex = electrical._new_cortex(seed)
        candidate_gids, basket_gid, _evidence = electrical._boundary_fixture(cortex)
        rows.append((seed, cortex, candidate_gids, basket_gid))
    return tuple(rows)


def _strictly_ranked(row: Sequence[int]) -> bool:
    return len(row) == electrical.CELLS_PER_COLUMN and all(
        int(row[index]) > int(row[index + 1]) for index in range(len(row) - 1)
    )


def _evaluate(
    row: Sequence[int],
    fixtures: Sequence[tuple[int, Any, tuple[int, ...], int]],
) -> dict[str, Any]:
    currents = tuple(int(value) for value in row)
    electrical._require(_strictly_ranked(currents), "candidate rank ordering drift")
    per_seed: dict[str, dict[str, Any]] = {}
    for seed, cortex, candidate_gids, basket_gid in fixtures:
        observed = electrical._probe_boundary(
            cortex,
            basket_gid=basket_gid,
            candidate_gids=candidate_gids,
            currents=currents,
        )
        per_seed[str(seed)] = {
            "basket_to_gaba_ticks": observed["basket_to_gaba_ticks"],
            "control_spike_count": len(observed["control_spike_gids"]),
            "control_spike_ticks": observed["control_spike_ticks"],
            "control_winner_is_intended": (
                observed["control_winner"] == observed["expected_winner"]
            ),
            "inhibited_spike_count": len(observed["inhibited_spike_gids"]),
            "inhibited_winner_is_intended": (
                observed["inhibited_winner"] == observed["expected_winner"]
            ),
            "natural_latency_ticks": observed["natural_latency_ticks"],
            "passed": observed["passed"],
            "winner_to_basket_ticks": observed["winner_to_basket_ticks"],
        }
    return {
        "currents_pa": list(currents),
        "maximum_natural_latency_ticks": max(
            row["natural_latency_ticks"] for row in per_seed.values()
        ),
        "minimum_natural_latency_ticks": min(
            row["natural_latency_ticks"] for row in per_seed.values()
        ),
        "passed": all(row["passed"] for row in per_seed.values()),
        "passed_seeds": sum(int(row["passed"]) for row in per_seed.values()),
        "per_seed": per_seed,
    }


def _minimal_single_literal_solutions(
    baseline: Sequence[int],
    fixtures: Sequence[tuple[int, Any, tuple[int, ...], int]],
    *,
    window_pa: int,
) -> list[dict[str, Any]]:
    base = tuple(int(value) for value in baseline)
    for magnitude in range(1, window_pa + 1):
        solutions: list[dict[str, Any]] = []
        for index in range(len(base)):
            for delta_pa in (-magnitude, magnitude):
                candidate = list(base)
                candidate[index] += delta_pa
                if candidate[index] <= 0 or not _strictly_ranked(candidate):
                    continue
                evidence = _evaluate(candidate, fixtures)
                if evidence["passed"]:
                    solutions.append(
                        {
                            "change": {
                                "delta_pa": delta_pa,
                                "from_pa": base[index],
                                "literal_index": index,
                                "to_pa": candidate[index],
                            },
                            "evidence": evidence,
                        }
                    )
        if solutions:
            return solutions
    return []


def _recommend_combination(
    solutions: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    combinations = itertools.product(*(solutions[name] for name in FAILED_ROWS))

    def score(combo: tuple[dict[str, Any], ...]) -> tuple[int, int, tuple[Any, ...]]:
        changes = tuple(row["change"] for row in combo)
        signatures = {(row["literal_index"], row["delta_pa"] < 0) for row in changes}
        return (
            sum(abs(row["delta_pa"]) for row in changes),
            len(signatures),
            tuple((row["literal_index"], row["delta_pa"]) for row in changes),
        )

    selected = min(combinations, key=score)
    return {name: row for name, row in zip(FAILED_ROWS, selected, strict=True)}


def run_search(*, window_pa: int = 100) -> dict[str, Any]:
    electrical._require(window_pa >= 1, "search window must include at least one pA")
    baseline = electrical._scenario_currents()
    fixtures = _fixtures()
    baseline_evidence = {name: _evaluate(row, fixtures) for name, row in sorted(baseline.items())}
    for name in FAILED_ROWS:
        electrical._require(
            not baseline_evidence[name]["passed"],
            f"{name} no longer fails; this search is stale",
        )
    for name in UNCHANGED_ROWS:
        electrical._require(
            baseline_evidence[name]["passed"],
            f"{name} baseline no longer passes",
        )

    solutions = {
        name: _minimal_single_literal_solutions(
            baseline[name],
            fixtures,
            window_pa=window_pa,
        )
        for name in FAILED_ROWS
    }
    no_solution = [name for name, rows in solutions.items() if not rows]
    recommendation: dict[str, Any] | None = None
    if not no_solution:
        selected = _recommend_combination(solutions)
        changes = [{"row": name, **selected[name]["change"]} for name in FAILED_ROWS]
        candidate_rows = {
            **{name: list(baseline[name]) for name in UNCHANGED_ROWS},
            **{name: selected[name]["evidence"]["currents_pa"] for name in FAILED_ROWS},
        }
        recommendation = {
            "candidate_rows_pa": dict(sorted(candidate_rows.items())),
            "changed_literals": len(changes),
            "changes": changes,
            "selection_tiebreak": (
                "minimum total |delta_pa|, then common literal-index/direction pattern"
            ),
            "status": "MEASURED_CHANGE_NOTE_CANDIDATE_NOT_FROZEN",
            "total_absolute_delta_pa": sum(abs(row["delta_pa"]) for row in changes),
            "verification": {
                **{name: selected[name]["evidence"] for name in FAILED_ROWS},
                **{name: baseline_evidence[name] for name in UNCHANGED_ROWS},
            },
        }

    return {
        "artifact_id": ARTIFACT_ID,
        "baseline": baseline_evidence,
        "claim_boundary": {
            "c1_executed": False,
            "held_out_accessed": False,
            "protocol_modified": False,
            "synthetic_electrical_search": True,
            "temporal_fixture_executed": False,
        },
        "minimal_single_literal_solutions": solutions,
        "no_solution_rows": no_solution,
        "recommendation": recommendation,
        "schema_version": 1,
        "search": {
            "integer_step_pa": 1,
            "objective": (
                "one changed literal per failed row; minimize |delta_pa|; preserve "
                "strict descending rank order; require all physical gates on all seeds"
            ),
            "window_pa": window_pa,
        },
        "source_sha256": {
            "calibrator": _sha256_file(Path(electrical.__file__)),
            "rejected_v3_artifact": _sha256_file(electrical.ARTIFACT_PATH),
        },
        "verdict": (
            "PASS_MINIMAL_CURRENT_CHANGE_CANDIDATE"
            if recommendation is not None
            else "NO_SOLUTION_WITHIN_WINDOW"
        ),
    }


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--write", action="store_true")
    actions.add_argument("--check", action="store_true")
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    parser.add_argument("--window-pa", type=int, default=100)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    observed = run_search(window_pa=args.window_pa)
    if args.write:
        args.artifact.write_text(
            json.dumps(observed, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"electrical-v3 current search written: {args.artifact} ({observed['verdict']})")
        return 0

    expected = json.loads(args.artifact.read_text(encoding="utf-8"))
    electrical._require(
        _canonical_bytes(observed) == _canonical_bytes(expected),
        "current search differs from artifact",
    )
    print(f"electrical-v3 current search check: PASS ({observed['verdict']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
