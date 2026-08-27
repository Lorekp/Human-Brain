"""Execute the frozen reserve-valid R3 electrical-v3 fixture.

The fixture is synthetic and electrical-only. It never reads corpus, held-out,
recognition, temporal-learning, or C1 data. ``--write`` replaces the single
artifact with a deterministic replay against the current fitted-cell sources;
``--check`` validates that artifact without running the sweep.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(__file__).with_name("minicolumn_electrical_calibration.json")
PROTOCOL_SOURCE = Path(__file__).with_name("calibration_v3_protocol.py")
CORTEX_SOURCE = REPO_ROOT / "src" / "humanbrain" / "cortex.py"
CELLS_SOURCE = REPO_ROOT / "src" / "humanbrain" / "cells.py"
SPARSE_SOURCE = REPO_ROOT / "src" / "humanbrain" / "sparse_temporal_cortex.py"

SOURCE_PATHS = {
    "experiments/r3_pilot/calibrate_minicolumn_electrical.py": Path(__file__),
    "experiments/r3_pilot/calibration_v3_protocol.py": PROTOCOL_SOURCE,
    "src/humanbrain/cells.py": CELLS_SOURCE,
    "src/humanbrain/cortex.py": CORTEX_SOURCE,
    "src/humanbrain/sparse_temporal_cortex.py": SPARSE_SOURCE,
}
SOURCE_IDENTITY_RULE = "sha256(live file after CRLF-to-LF normalization; lone CR forbidden)"

ARTIFACT_ID = "r3-minicolumn-electrical-v3"
SCHEMA_VERSION = 3

COLUMNS = 4_096
CELLS_PER_COLUMN = 8
BASE_PYRAMIDAL_CELLS = COLUMNS * CELLS_PER_COLUMN
MAX_PYRAMIDAL_CELLS = 131_072
RELAY_OFFSET = MAX_PYRAMIDAL_CELLS
BASKET_OFFSET = 2 * MAX_PYRAMIDAL_CELLS
PROXIMAL_OFFSET = BASKET_OFFSET + COLUMNS
PROXIMAL_RELAY_OFFSET = PROXIMAL_OFFSET + COLUMNS
CELL_COUNT = PROXIMAL_RELAY_OFFSET + COLUMNS
CONSTRUCTION_SEEDS = (3, 7, 17, 29, 43, 59, 83, 101)

BASKET_CURRENT_PA = 200_000
GABA_CURRENT_PA = 500
PROXIMAL_CURRENT_PA = 570
RELAY_CURRENT_PA = 20_000
DT_US = 50
LATENCY_MUST_EXCEED_US = 100
LATENCY_MUST_EXCEED_TICKS = LATENCY_MUST_EXCEED_US // DT_US

PASS_VERDICT = "PASS_RESERVE_VALID_ELECTRICAL_V3"
REJECT_VERDICT = "REJECT_RESERVE_VALID_ELECTRICAL_V3"


class CalibrationError(RuntimeError):
    """The implementation or artifact violates the electrical-v3 contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_source_bytes(raw: bytes) -> bytes:
    normalized = raw.replace(b"\r\n", b"\n")
    _require(b"\r" not in normalized, "canonical source contains a lone CR byte")
    return normalized


def _source_hashes() -> dict[str, str]:
    return {
        name: _sha256(_canonical_source_bytes(path.read_bytes()))
        for name, path in sorted(SOURCE_PATHS.items())
    }


def _claim_boundary() -> dict[str, bool]:
    return {
        "capability_evidence": False,
        "held_out_accessed": False,
        "reserve_valid_electrical_fixture": True,
        "synthetic_electrical_calibration": True,
        "temporal_learning_evidence": False,
        "transfer_evidence": False,
    }


def _scenario_currents() -> dict[str, tuple[int, ...]]:
    return {
        "active_vs_matching": (650, 640, 590, 543, 496, 449, 402, 355),
        "matching_vs_novel": (600, 590, 543, 496, 449, 402, 355, 308),
        "novel_rank_boundary": (550, 543, 496, 449, 402, 355, 308, 261),
    }


def _protocol() -> dict[str, Any]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from experiments.r3_pilot.calibration_v3_protocol import (
        build_protocol,
    )

    screen = build_protocol()["screens"]["electrical_v3"]
    _require(
        screen["construction_seeds"] == list(CONSTRUCTION_SEEDS),
        "electrical-v3 seed contract drift",
    )
    _require(
        screen["boundary_scenarios_pa"]
        == {name: list(row) for name, row in _scenario_currents().items()},
        "electrical-v3 boundary rows drift",
    )
    _require(screen["required_verdict"] == PASS_VERDICT, "required verdict drift")
    return screen


def _runtime_config_hashes() -> dict[str, str]:
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from humanbrain.cortex import HumanCellParameters
    from humanbrain.sparse_temporal_cortex import (
        SparseTemporalConfig,
    )

    sparse = SparseTemporalConfig(
        column_count=COLUMNS,
        theta_match=10,
        theta_active=13,
    )
    electrical = {
        "active_current_pa": sparse.active_current_pa,
        "base_cells_per_column": sparse.base_cells_per_column,
        "basket_current_pa": sparse.basket_current_pa,
        "basket_inhibition_pa": sparse.basket_inhibition_pa,
        "column_count": sparse.column_count,
        "homeostatic_current_pa": sparse.homeostatic_current_pa,
        "matching_current_pa": sparse.matching_current_pa,
        "max_pyramidal_cells": sparse.max_pyramidal_cells,
        "novel_current_pa": sparse.novel_current_pa,
        "novel_race_cap": sparse.novel_race_cap,
        "novel_rank_spacing_pa": sparse.novel_rank_spacing_pa,
        "proximal_current_pa": sparse.proximal_current_pa,
        "relay_current_pa": sparse.relay_current_pa,
    }
    return {
        "human_cell_parameters": _sha256(_canonical_json_bytes(asdict(HumanCellParameters()))),
        "sparse_electrical_defaults": _sha256(_canonical_json_bytes(electrical)),
    }


def _runtime_versions() -> dict[str, str]:
    return {
        "numpy": np.__version__,
        "python": platform.python_version(),
    }


def _new_cortex(seed: int) -> Any:
    if str(REPO_ROOT / "src") not in sys.path:
        sys.path.insert(0, str(REPO_ROOT / "src"))
    from humanbrain.cortex import DendriticCortex

    cortex = DendriticCortex(cell_count=CELL_COUNT, seed=seed)
    _require(
        round(cortex.parameters.cell.dt_ms * 1_000) == DT_US,
        "GLIF3 integration step drift",
    )
    return cortex


def _ms_to_ticks(value_ms: float | None) -> int:
    _require(value_ms is not None, "required physical event did not occur")
    ticks = round(float(value_ms) * 1_000.0 / DT_US)
    _require(
        abs(float(value_ms) - ticks * DT_US / 1_000.0) < 1e-12,
        "physical event is not aligned to the GLIF3 integration step",
    )
    return ticks


def _domain_ids(start: int, stop: int) -> np.ndarray:
    return np.arange(start, stop, dtype=np.intp)


def _domain_threshold_evidence(cortex: Any, ids: np.ndarray) -> dict[str, Any]:
    offsets = np.asarray(cortex._threshold_offset[ids], dtype=np.float64)
    minimum_index = int(np.argmin(offsets))
    maximum_index = int(np.argmax(offsets))
    return {
        "cells": int(ids.size),
        "maximum_gid": int(ids[maximum_index]),
        "maximum_offset_mv_hex": float(offsets[maximum_index]).hex(),
        "minimum_gid": int(ids[minimum_index]),
        "minimum_offset_mv_hex": float(offsets[minimum_index]).hex(),
        "threshold_offsets_sha256": _sha256(np.asarray(offsets, dtype="<f8").tobytes(order="C")),
    }


def _boundary_fixture(cortex: Any) -> tuple[tuple[int, ...], int, dict[str, Any]]:
    pyramidal_ids = _domain_ids(0, MAX_PYRAMIDAL_CELLS)
    offsets = np.asarray(cortex._threshold_offset[pyramidal_ids], dtype=np.float64)
    order = np.lexsort((pyramidal_ids, offsets))
    minimum_gid = int(pyramidal_ids[order[0]])
    maximum_gid = int(pyramidal_ids[order[-1]])
    remaining = [
        int(pyramidal_ids[index])
        for index in order
        if int(pyramidal_ids[index]) not in {minimum_gid, maximum_gid}
    ]
    candidates = (maximum_gid, minimum_gid, *remaining[:6])

    basket_ids = _domain_ids(BASKET_OFFSET, BASKET_OFFSET + COLUMNS)
    basket_offsets = np.asarray(cortex._threshold_offset[basket_ids], dtype=np.float64)
    basket_gid = int(basket_ids[int(np.argmax(basket_offsets))])
    evidence = {
        "basket": _domain_threshold_evidence(cortex, basket_ids),
        "candidate_gids_by_rank": list(candidates),
        "pyramidal": _domain_threshold_evidence(cortex, pyramidal_ids),
    }
    return candidates, basket_gid, evidence


def _first_spike_ticks(
    cortex: Any,
    ids: np.ndarray,
    *,
    current_pa: int,
) -> np.ndarray:
    cell = cortex.parameters.cell
    thresholds = cell.threshold_relative_mv + cortex._threshold_offset[ids]
    response_needed = thresholds / float(current_pa)
    rising_response = cortex._unit_nmda_response[: cortex._unit_nmda_peak_step + 1]
    first_steps = np.searchsorted(rising_response, response_needed, side="left")
    no_spike_tick = cortex._race_steps + 1
    return np.where(
        first_steps < rising_response.size,
        first_steps + 1,
        no_spike_tick,
    ).astype(np.int32, copy=False)


def _prove_tick_monotonicity(
    cortex: Any,
    ids: np.ndarray,
    *,
    currents_pa: Sequence[int],
) -> dict[str, Any]:
    offsets = np.asarray(cortex._threshold_offset[ids], dtype=np.float64)
    threshold_order = np.lexsort((ids, offsets))
    hasher = hashlib.sha256()
    rows: dict[str, dict[str, int]] = {}
    total_violations = 0
    for current_pa in sorted(set(int(value) for value in currents_pa)):
        ticks = _first_spike_ticks(cortex, ids, current_pa=current_pa)
        ordered_ticks = ticks[threshold_order]
        violations = int(np.count_nonzero(np.diff(ordered_ticks) < 0))
        total_violations += violations
        hasher.update(struct.pack("<q", current_pa))
        hasher.update(np.asarray(ticks, dtype="<i4").tobytes(order="C"))
        rows[str(current_pa)] = {
            "maximum_tick": int(ticks.max()),
            "minimum_tick": int(ticks.min()),
            "no_spike_cells": int(np.count_nonzero(ticks > cortex._race_steps)),
            "order_violations": violations,
        }
    return {
        "cells": int(ids.size),
        "comparisons": int((ids.size - 1) * len(rows)),
        "currents_pa": [int(value) for value in sorted(int(key) for key in rows)],
        "first_spike_ticks_sha256": hasher.hexdigest(),
        "method": ("exact compiled rising-response search used by DendriticCortex.race_currents"),
        "nondecreasing": total_violations == 0,
        "rows": rows,
        "total_order_violations": total_violations,
    }


def _probe_boundary(
    cortex: Any,
    *,
    basket_gid: int,
    candidate_gids: Sequence[int],
    currents: Sequence[int],
) -> dict[str, Any]:
    candidates = tuple(int(value) for value in candidate_gids)
    current_tuple = tuple(int(value) for value in currents)
    _require(len(candidates) == CELLS_PER_COLUMN, "boundary candidate width drift")
    _require(len(current_tuple) == CELLS_PER_COLUMN, "boundary current width drift")

    control = cortex.race_currents_with_inhibition(
        candidates[:2],
        current_tuple[:2],
        basket_cell_id=basket_gid,
        basket_current_pa=BASKET_CURRENT_PA,
        inhibition_current_pa=0.0,
    )
    inhibited = cortex.race_currents_with_inhibition(
        candidates,
        current_tuple,
        basket_cell_id=basket_gid,
        basket_current_pa=BASKET_CURRENT_PA,
        inhibition_current_pa=GABA_CURRENT_PA,
    )

    control_ticks = [_ms_to_ticks(value) for value in control.pyramidal_spike_times_ms]
    inhibited_ticks = [_ms_to_ticks(value) for value in inhibited.pyramidal_spike_times_ms]
    winner_tick = _ms_to_ticks(inhibited.winner_spike_time_ms)
    basket_tick = _ms_to_ticks(inhibited.basket_spike_time_ms)
    gaba_tick = _ms_to_ticks(inhibited.inhibition_onset_ms)
    natural_latency = control_ticks[1] - control_ticks[0] if len(control_ticks) >= 2 else -1
    observed = {
        "basket_gid": int(basket_gid),
        "basket_to_gaba_ticks": gaba_tick - basket_tick,
        "candidate_gids_by_rank": list(candidates),
        "control_spike_gids": [int(value) for value in control.pyramidal_spikes],
        "control_spike_ticks": control_ticks,
        "control_winner": int(control.winner),
        "expected_winner": candidates[0],
        "inhibited_spike_gids": [int(value) for value in inhibited.pyramidal_spikes],
        "inhibited_spike_ticks": inhibited_ticks,
        "inhibited_winner": int(inhibited.winner),
        "natural_latency_ticks": natural_latency,
        "winner_to_basket_ticks": basket_tick - winner_tick,
    }
    observed["passed"] = _boundary_passed(observed)
    return observed


def _boundary_passed(observed: dict[str, Any]) -> bool:
    return (
        len(observed["control_spike_gids"]) == 2
        and observed["control_winner"] == observed["expected_winner"]
        and len(observed["inhibited_spike_gids"]) == 1
        and observed["inhibited_winner"] == observed["expected_winner"]
        and observed["natural_latency_ticks"] > LATENCY_MUST_EXCEED_TICKS
        and observed["winner_to_basket_ticks"] == 1
        and observed["basket_to_gaba_ticks"] == 1
    )


def _probe_singleton(cortex: Any, *, gid: int, current_pa: int) -> dict[str, int]:
    race = cortex.race_currents([gid], [current_pa])
    return {
        "active_cells": int(race.active_cells),
        "expected_winner": int(gid),
        "spike_tick": _ms_to_ticks(race.spike_time_ms),
        "winner": int(race.winner),
    }


def _probe_proximal_channel(cortex: Any, *, column: int) -> dict[str, int]:
    _require(0 <= column < COLUMNS, "column outside calibrated sheet")
    source_gid = PROXIMAL_OFFSET + column
    relay_gid = PROXIMAL_RELAY_OFFSET + column
    source = cortex.race_currents([source_gid], [PROXIMAL_CURRENT_PA])
    relay = cortex.race_currents([relay_gid], [RELAY_CURRENT_PA])
    source_tick = _ms_to_ticks(source.spike_time_ms)
    relay_intrinsic_tick = _ms_to_ticks(relay.spike_time_ms)
    return {
        "causal_relay_tick": source_tick + relay_intrinsic_tick,
        "expected_relay": relay_gid,
        "expected_source": source_gid,
        "relay_active_cells": int(relay.active_cells),
        "relay_intrinsic_tick": relay_intrinsic_tick,
        "relay_winner": int(relay.winner),
        "source_active_cells": int(source.active_cells),
        "source_tick": source_tick,
        "source_winner": int(source.winner),
    }


def _run_proximal_seed(cortex: Any) -> dict[str, Any]:
    hasher = hashlib.sha256()
    summary: dict[str, Any] = {
        "causal_relay_after_source_trials": 0,
        "dedicated_relay_single_spike_trials": 0,
        "expected_relay_winner_trials": 0,
        "expected_source_winner_trials": 0,
        "maximum_causal_relay_tick": 0,
        "maximum_relay_intrinsic_tick": 0,
        "maximum_source_tick": 0,
        "minimum_causal_relay_tick": None,
        "minimum_relay_intrinsic_tick": None,
        "minimum_source_tick": None,
        "source_single_spike_trials": 0,
        "trials": 0,
    }
    for column in range(COLUMNS):
        observed = _probe_proximal_channel(cortex, column=column)
        summary["trials"] += 1
        summary["source_single_spike_trials"] += int(observed["source_active_cells"] == 1)
        summary["dedicated_relay_single_spike_trials"] += int(observed["relay_active_cells"] == 1)
        summary["expected_source_winner_trials"] += int(
            observed["source_winner"] == observed["expected_source"]
        )
        summary["expected_relay_winner_trials"] += int(
            observed["relay_winner"] == observed["expected_relay"]
        )
        summary["causal_relay_after_source_trials"] += int(
            observed["causal_relay_tick"] > observed["source_tick"]
        )
        for field in ("causal_relay_tick", "relay_intrinsic_tick", "source_tick"):
            minimum = f"minimum_{field}"
            maximum = f"maximum_{field}"
            current_minimum = summary[minimum]
            value = observed[field]
            summary[minimum] = value if current_minimum is None else min(current_minimum, value)
            summary[maximum] = max(summary[maximum], value)
        packed = tuple(observed[key] for key in sorted(observed))
        hasher.update(struct.pack("<" + "q" * len(packed), *packed))
    summary["observations_sha256"] = hasher.hexdigest()
    required = (
        "source_single_spike_trials",
        "dedicated_relay_single_spike_trials",
        "expected_source_winner_trials",
        "expected_relay_winner_trials",
        "causal_relay_after_source_trials",
    )
    summary["passed"] = summary["trials"] == COLUMNS and all(
        summary[field] == COLUMNS for field in required
    )
    return summary


def _run_seed(seed: int) -> dict[str, Any]:
    cortex = _new_cortex(seed)
    candidates, basket_gid, fixture = _boundary_fixture(cortex)
    boundary_currents = sorted({value for row in _scenario_currents().values() for value in row})
    pyramidal_ids = _domain_ids(0, MAX_PYRAMIDAL_CELLS)
    relay_ids = _domain_ids(RELAY_OFFSET, RELAY_OFFSET + MAX_PYRAMIDAL_CELLS)
    pyramidal_monotonicity = _prove_tick_monotonicity(
        cortex,
        pyramidal_ids,
        currents_pa=boundary_currents,
    )
    relay_monotonicity = _prove_tick_monotonicity(
        cortex,
        relay_ids,
        currents_pa=(RELAY_CURRENT_PA,),
    )

    boundaries = {
        name: _probe_boundary(
            cortex,
            basket_gid=basket_gid,
            candidate_gids=candidates,
            currents=currents,
        )
        for name, currents in sorted(_scenario_currents().items())
    }

    relay_domain = _domain_threshold_evidence(cortex, relay_ids)
    relay_singleton = _probe_singleton(
        cortex,
        gid=relay_domain["maximum_gid"],
        current_pa=RELAY_CURRENT_PA,
    )
    relay_singleton["passed"] = (
        relay_singleton["active_cells"] == 1
        and relay_singleton["winner"] == relay_singleton["expected_winner"]
        and relay_singleton["spike_tick"] > 0
        and relay_monotonicity["nondecreasing"]
        and relay_monotonicity["rows"][str(RELAY_CURRENT_PA)]["no_spike_cells"] == 0
    )

    proximal = _run_proximal_seed(cortex)
    passed = (
        pyramidal_monotonicity["nondecreasing"]
        and all(row["passed"] for row in boundaries.values())
        and relay_singleton["passed"]
        and proximal["passed"]
    )
    return {
        "boundaries": boundaries,
        "fixture": fixture,
        "monotonicity": {
            "pyramidal": pyramidal_monotonicity,
            "winner_relay": relay_monotonicity,
        },
        "proximal_channel": proximal,
        "seed": seed,
        "verdict": PASS_VERDICT if passed else REJECT_VERDICT,
        "winner_relay": {
            "domain": relay_domain,
            "singleton_global_worst": relay_singleton,
        },
    }


def _aggregate_seeds(seed_rows: dict[str, dict[str, Any]]) -> tuple[dict[str, Any], list[str]]:
    rejection_reasons: list[str] = []
    boundary_summary: dict[str, Any] = {}
    for name in sorted(_scenario_currents()):
        rows = [seed_rows[str(seed)]["boundaries"][name] for seed in CONSTRUCTION_SEEDS]
        for seed, row in zip(CONSTRUCTION_SEEDS, rows, strict=True):
            if not row["passed"]:
                rejection_reasons.append(
                    f"seed-{seed}:{name}:natural_latency_ticks="
                    f"{row['natural_latency_ticks']} (must be >"
                    f"{LATENCY_MUST_EXCEED_TICKS})"
                )
        boundary_summary[name] = {
            "maximum_natural_latency_ticks": max(row["natural_latency_ticks"] for row in rows),
            "minimum_natural_latency_ticks": min(row["natural_latency_ticks"] for row in rows),
            "passed_trials": sum(int(row["passed"]) for row in rows),
            "trials": len(rows),
        }

    proximal_fields = (
        "trials",
        "source_single_spike_trials",
        "dedicated_relay_single_spike_trials",
        "expected_source_winner_trials",
        "expected_relay_winner_trials",
        "causal_relay_after_source_trials",
    )
    proximal = {
        field: sum(seed_rows[str(seed)]["proximal_channel"][field] for seed in CONSTRUCTION_SEEDS)
        for field in proximal_fields
    }
    proximal["passed_seeds"] = sum(
        int(seed_rows[str(seed)]["proximal_channel"]["passed"]) for seed in CONSTRUCTION_SEEDS
    )
    if proximal["passed_seeds"] != len(CONSTRUCTION_SEEDS):
        rejection_reasons.append("generic_proximal_relay_screen_failed")

    monotonic = {
        domain: {
            "cells_covered": sum(
                seed_rows[str(seed)]["monotonicity"][domain]["cells"] for seed in CONSTRUCTION_SEEDS
            ),
            "comparisons": sum(
                seed_rows[str(seed)]["monotonicity"][domain]["comparisons"]
                for seed in CONSTRUCTION_SEEDS
            ),
            "passed_seeds": sum(
                int(seed_rows[str(seed)]["monotonicity"][domain]["nondecreasing"])
                for seed in CONSTRUCTION_SEEDS
            ),
        }
        for domain in ("pyramidal", "winner_relay")
    }
    if any(row["passed_seeds"] != len(CONSTRUCTION_SEEDS) for row in monotonic.values()):
        rejection_reasons.append("first_spike_tick_monotonicity_failed")

    relay_passed = sum(
        int(seed_rows[str(seed)]["winner_relay"]["singleton_global_worst"]["passed"])
        for seed in CONSTRUCTION_SEEDS
    )
    if relay_passed != len(CONSTRUCTION_SEEDS):
        rejection_reasons.append("winner_relay_global_worst_singleton_failed")

    return (
        {
            "boundaries": boundary_summary,
            "construction_seeds": len(CONSTRUCTION_SEEDS),
            "monotonicity": monotonic,
            "proximal_channel": proximal,
            "winner_relay": {
                "domain_cells_covered": MAX_PYRAMIDAL_CELLS * len(CONSTRUCTION_SEEDS),
                "passed_global_worst_singletons": relay_passed,
                "singleton_trials": len(CONSTRUCTION_SEEDS),
            },
        },
        rejection_reasons,
    )


def run_replay() -> dict[str, Any]:
    protocol = _protocol()
    seed_rows = {str(seed): _run_seed(seed) for seed in CONSTRUCTION_SEEDS}
    summary, rejection_reasons = _aggregate_seeds(seed_rows)
    return {
        "artifact_id": ARTIFACT_ID,
        "claim_boundary": _claim_boundary(),
        "config_sha256": _runtime_config_hashes(),
        "protocol": protocol,
        "rejection_reasons": rejection_reasons,
        "runtime_versions": _runtime_versions(),
        "schema_version": SCHEMA_VERSION,
        "seeds": seed_rows,
        "source_identity_rule": SOURCE_IDENTITY_RULE,
        "source_sha256": _source_hashes(),
        "summary": summary,
        "verdict": PASS_VERDICT if not rejection_reasons else REJECT_VERDICT,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("artifact_id") == ARTIFACT_ID, "artifact id mismatch")
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
    _require(artifact.get("claim_boundary") == _claim_boundary(), "claim drift")
    _require(artifact.get("protocol") == _protocol(), "protocol drift")
    _require(
        artifact.get("source_identity_rule") == SOURCE_IDENTITY_RULE,
        "source identity rule drift",
    )
    _require(artifact.get("source_sha256") == _source_hashes(), "source changed")
    _require(
        artifact.get("config_sha256") == _runtime_config_hashes(),
        "electrical config changed",
    )
    _require(
        artifact.get("runtime_versions") == _runtime_versions(),
        "calibration runtime changed",
    )
    seeds = artifact.get("seeds")
    _require(isinstance(seeds, dict), "per-seed evidence is missing")
    _require(
        set(seeds) == {str(seed) for seed in CONSTRUCTION_SEEDS},
        "construction-seed evidence drift",
    )
    for seed in CONSTRUCTION_SEEDS:
        row = seeds[str(seed)]
        _require(row.get("seed") == seed, f"seed-{seed} identity drift")
        _require(
            set(row.get("boundaries", {})) == set(_scenario_currents()),
            f"seed-{seed} boundary grid drift",
        )
        _require(
            row.get("fixture", {}).get("pyramidal", {}).get("cells") == MAX_PYRAMIDAL_CELLS,
            f"seed-{seed} pyramidal domain drift",
        )
        _require(
            row.get("winner_relay", {}).get("domain", {}).get("cells") == MAX_PYRAMIDAL_CELLS,
            f"seed-{seed} winner-relay domain drift",
        )
        _require(
            row.get("proximal_channel", {}).get("trials") == COLUMNS,
            f"seed-{seed} proximal trial count drift",
        )
        seed_passed = (
            row["monotonicity"]["pyramidal"]["nondecreasing"]
            and all(boundary["passed"] for boundary in row["boundaries"].values())
            and row["winner_relay"]["singleton_global_worst"]["passed"]
            and row["proximal_channel"]["passed"]
        )
        _require(
            row.get("verdict") == (PASS_VERDICT if seed_passed else REJECT_VERDICT),
            f"seed-{seed} verdict drift",
        )

    summary, rejection_reasons = _aggregate_seeds(seeds)
    _require(artifact.get("summary") == summary, "aggregate evidence drift")
    _require(
        artifact.get("rejection_reasons") == rejection_reasons,
        "rejection evidence drift",
    )
    expected_verdict = PASS_VERDICT if not rejection_reasons else REJECT_VERDICT
    _require(artifact.get("verdict") == expected_verdict, "global verdict drift")


def load_and_validate(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(artifact, dict), "artifact root must be an object")
    validate_artifact(artifact)
    return artifact


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(artifact, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true", help="fast artifact check")
    actions.add_argument(
        "--replay-check",
        action="store_true",
        help="repeat electrical-v3 and compare the artifact",
    )
    actions.add_argument(
        "--write",
        action="store_true",
        help="repeat electrical-v3 and replace the artifact",
    )
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.check:
            artifact = load_and_validate(args.artifact)
            print(
                "minicolumn electrical-v3 artifact check: PASS "
                f"(scientific verdict: {artifact['verdict']})"
            )
            return 0

        started = time.perf_counter()
        observed = run_replay()
        validate_artifact(observed)
        elapsed = time.perf_counter() - started
        if args.write:
            _write_artifact(args.artifact, observed)
            print(
                "minicolumn electrical-v3 artifact written: "
                f"{args.artifact} ({elapsed:.1f}s, {observed['verdict']})"
            )
            return 0

        expected = load_and_validate(args.artifact)
        _require(
            _canonical_json_bytes(observed) == _canonical_json_bytes(expected),
            "electrical-v3 replay differs from artifact",
        )
        print(
            f"minicolumn electrical-v3 replay check: PASS ({elapsed:.1f}s, {observed['verdict']})"
        )
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError) as error:
        print(f"minicolumn electrical-v3: FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
