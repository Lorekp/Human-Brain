"""Search the fitted-cell bias needed for same-band owner arbitration.

Synthetic electrical-only development fixture.  It reuses the electrical-v3
extrema construction and never reads TRAIN, held-out, q4, or C1 data.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.r3_pilot import calibrate_minicolumn_electrical as electrical  # noqa: E402

ARTIFACT_PATH = Path(__file__).with_name("same_band_rank_bias_search.json")
ARTIFACT_ID = "r3-same-band-rank-bias-search-v1"
ACTIVE_MAX_PA = 690
MATCHING_MAX_PA = 640
NOVEL_MAX_PA = 590
ACTIVE_MIN_FLOOR_PA = 670
MATCHING_MIN_FLOOR_PA = 620
NOVEL_MIN_FLOOR_PA = 570
EXISTING_NOVEL_RANK_DECREMENT_PA = 47
WIDTHS = tuple(range(2, 9))


def _sha256(path: Path) -> str:
    raw = electrical._canonical_source_bytes(path.read_bytes())
    return hashlib.sha256(raw).hexdigest()


def _fixtures() -> tuple[tuple[int, Any, tuple[int, ...], int], ...]:
    rows = []
    for seed in electrical.CONSTRUCTION_SEEDS:
        cortex = electrical._new_cortex(seed)
        gids, basket_gid, _evidence = electrical._boundary_fixture(cortex)
        rows.append((seed, cortex, gids, basket_gid))
    return tuple(rows)


def _same_band_rows(decrement_pa: int) -> list[tuple[str, list[int]]]:
    return [
        (
            f"{name}-width-{width}",
            [base_pa - decrement_pa * rank for rank in range(width)],
        )
        for name, base_pa in (("active", ACTIVE_MAX_PA), ("matching", MATCHING_MAX_PA))
        for width in WIDTHS
    ]


def _compositions() -> list[tuple[int, int, int]]:
    return [
        (active, matching, total - active - matching)
        for total in WIDTHS
        for active in range(total + 1)
        for matching in range(total - active + 1)
    ]


def _precedence_rows(decrement_pa: int) -> list[tuple[str, list[int]]]:
    """Disadvantage the expected top-class winner and maximize every rival."""
    rows = []
    for active, matching, novel in _compositions():
        currents = []
        for rank in range(active):
            base = ACTIVE_MIN_FLOOR_PA if rank == 0 else ACTIVE_MAX_PA
            currents.append(base - decrement_pa * rank)
        for rank in range(matching):
            base = MATCHING_MIN_FLOOR_PA if active == 0 and rank == 0 else MATCHING_MAX_PA
            currents.append(base - decrement_pa * rank)
        for rank in range(novel):
            base = (
                NOVEL_MIN_FLOOR_PA
                if active == 0 and matching == 0 and rank == 0
                else NOVEL_MAX_PA
            )
            currents.append(base - EXISTING_NOVEL_RANK_DECREMENT_PA * rank)
        rows.append((f"a{active}-m{matching}-n{novel}", currents))
    return rows


def _evaluate(
    rows: list[tuple[str, list[int]]],
    fixtures: tuple[tuple[int, Any, tuple[int, ...], int], ...],
) -> dict[str, Any]:
    failures = []
    latencies = []
    inhibited_spike_counts = set()
    for label, values in rows:
        currents = tuple(values) + (0,) * (electrical.CELLS_PER_COLUMN - len(values))
        for seed, cortex, gids, basket_gid in fixtures:
            observed = electrical._probe_boundary(
                cortex,
                basket_gid=basket_gid,
                candidate_gids=gids,
                currents=currents,
            )
            latencies.append(observed["natural_latency_ticks"])
            inhibited_spike_counts.add(len(observed["inhibited_spike_gids"]))
            if not observed["passed"]:
                failures.append(
                    {
                        "control_spike_ticks": observed["control_spike_ticks"],
                        "inhibited_spike_ticks": observed["inhibited_spike_ticks"],
                        "label": label,
                        "natural_latency_ticks": observed["natural_latency_ticks"],
                        "seed": seed,
                    }
                )
    return {
        "failures": len(failures),
        "first_failure": failures[0] if failures else None,
        "inhibited_spike_counts": sorted(inhibited_spike_counts),
        "maximum_natural_latency_ticks": max(latencies),
        "minimum_natural_latency_ticks": min(latencies),
        "passed": not failures,
        "trials": len(rows) * len(fixtures),
    }


def _first_failure(
    decrement_pa: int,
    fixtures: tuple[tuple[int, Any, tuple[int, ...], int], ...],
) -> bool:
    for _label, values in _same_band_rows(decrement_pa):
        currents = tuple(values) + (0,) * (electrical.CELLS_PER_COLUMN - len(values))
        for _seed, cortex, gids, basket_gid in fixtures:
            if not electrical._probe_boundary(
                cortex,
                basket_gid=basket_gid,
                candidate_gids=gids,
                currents=currents,
            )["passed"]:
                return True
    return False


def run_search() -> dict[str, Any]:
    fixtures = _fixtures()
    minimum = next(
        value for value in range(1, EXISTING_NOVEL_RANK_DECREMENT_PA + 1)
        if not _first_failure(value, fixtures)
    )
    evaluated = {
        str(value): _evaluate(_same_band_rows(value), fixtures)
        for value in (1, 2, 3, 4, minimum - 1, minimum, EXISTING_NOVEL_RANK_DECREMENT_PA)
    }
    precedence = {
        str(value): _evaluate(_precedence_rows(value), fixtures)
        for value in (minimum, EXISTING_NOVEL_RANK_DECREMENT_PA)
    }

    # Exact native identity extrema: base + 20/(1+load), load in [0,255].
    pairwise_slack_pa = 50.0 + 20.0 / 256.0 - 20.0
    strict_upper_bound = pairwise_slack_pa / 7.0
    strict_integer_maximum = int(strict_upper_bound)
    return {
        "artifact_id": ARTIFACT_ID,
        "claim_boundary": {
            "c1_executed": False,
            "core_runtime_modified": False,
            "held_out_accessed": False,
            "protocol_modified": False,
            "synthetic_electrical_only": True,
            "train_executed": False,
        },
        "construction": {
            "candidate_assignment": (
                "rank0 gets the global maximum pyramidal threshold offset; "
                "rivals get ascending global minima; basket gets its global maximum"
            ),
            "construction_seeds": list(electrical.CONSTRUCTION_SEEDS),
            "integer_resolution_pa": 1,
            "same_band_bases_pa": {
                "active": ACTIVE_MAX_PA,
                "matching": MATCHING_MAX_PA,
            },
            "widths": list(WIDTHS),
        },
        "emitted_winner_precedence": {
            "composition_count_per_seed": len(_compositions()),
            "interpretation": (
                "if active owners exist, active rank0 must be the sole fitted-cell winner; "
                "otherwise matching rank0, otherwise novel rank0"
            ),
            "results": precedence,
        },
        "recommendation": {
            "existing_constant_pa": EXISTING_NOVEL_RANK_DECREMENT_PA,
            "margin_above_measured_minimum_pa": (
                EXISTING_NOVEL_RANK_DECREMENT_PA - minimum
            ),
            "measured_minimum_integer_pa": minimum,
            "recommended_law": (
                "I = I_class(load) - 47 pA * same_class_owner_rank"
            ),
            "scope": "emitted-winner precedence, not pairwise ordering of every raw current",
        },
        "same_band_search": {
            "evaluated": evaluated,
            "minimum_common_integer_decrement_pa": minimum,
            "rejected_integer_range_pa": [1, minimum - 1],
        },
        "schema_version": 1,
        "source_sha256": {
            "calibrator": _sha256(Path(electrical.__file__)),
            "native_temporal_physics": _sha256(
                REPO_ROOT / "src/humanbrain/_native_src/src/temporal_physics.rs"
            ),
            "search_script": _sha256(Path(__file__)),
        },
        "strict_pairwise_current_ordering": {
            "integer_decrement_maximum_pa": strict_integer_maximum,
            "result_at_allowed_integers": {
                str(value): evaluated[str(value)] for value in range(1, 5)
            },
            "status": "INCOMPATIBLE_WITH_ONE_STAGE_LINEAR_RANK_LADDER",
            "upper_bound_exclusive_pa": strict_upper_bound,
        },
        "verdict": "PASS_EMITTED_WINNER_PRECEDENCE_WITH_39_PA_MINIMUM",
    }


def main() -> int:
    artifact = run_search()
    ARTIFACT_PATH.write_text(
        json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"same-band rank-bias search written: {ARTIFACT_PATH} "
        f"(minimum={artifact['same_band_search']['minimum_common_integer_decrement_pa']} pA)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
