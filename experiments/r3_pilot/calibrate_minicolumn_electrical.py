"""Exhaustive coupled-race calibration for one 4096 x 8 minicolumn sheet.

``--check`` validates only hashes and the frozen artifact. ``--replay-check``
repeats the exhaustive electrical sweep. The sweep contains no corpus, task,
held-out split, recognition outcome, or temporal-code selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(__file__).with_name("minicolumn_electrical_calibration.json")
CORTEX_SOURCE = REPO_ROOT / "src" / "humanbrain" / "cortex.py"
CELLS_SOURCE = REPO_ROOT / "src" / "humanbrain" / "cells.py"
SPARSE_SOURCE = REPO_ROOT / "src" / "humanbrain" / "sparse_temporal_cortex.py"

ARTIFACT_ID = "r3-minicolumn-electrical-v1"
SCHEMA_VERSION = 1
SOURCE_COMMIT = "df26cc2"
CORTEX_SOURCE_SHA256 = "16545b390c516b6f5817366946f0f516cec5b049bb9e1f6021184d9e156e0a8c"
CELLS_SOURCE_SHA256 = "5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6"
SPARSE_SOURCE_SHA256 = "0aca721d0c10b88ad91764f499b0b8e6cdde4b08b35e862f7dcfecf42f46fac3"
HUMAN_CELL_CONFIG_SHA256 = (
    "946fe684aea009b78a4d6892dc66cdfdf79aa523c49382bc6e72a6efcf745726"
)
SPARSE_ELECTRICAL_CONFIG_SHA256 = (
    "f69d0addd9619cae6b0d03eb621ec7ca324b0016058e82aef12a61a0f1cf8515"
)

COLUMNS = 4_096
CELLS_PER_COLUMN = 8
BASE_PYRAMIDAL_CELLS = COLUMNS * CELLS_PER_COLUMN
MAX_PYRAMIDAL_CELLS = 131_072
RELAY_OFFSET = MAX_PYRAMIDAL_CELLS
BASKET_OFFSET = 2 * MAX_PYRAMIDAL_CELLS
PROXIMAL_OFFSET = BASKET_OFFSET + COLUMNS
PROXIMAL_RELAY_OFFSET = PROXIMAL_OFFSET + COLUMNS
CELL_COUNT = PROXIMAL_RELAY_OFFSET + COLUMNS
CONSTRUCTION_SEEDS = (3, 7, 17, 29, 43)

NOVEL_CURRENT_LADDER_PA = tuple(570 - 47 * rank for rank in range(8))
MATCHING_CURRENT_PA = 620
ACTIVE_CURRENT_PA = 670
HOMEOSTASIS_ADVERSARIAL_PA = 20
BASKET_CURRENT_PA = 200_000
GABA_CURRENT_PA = 500
PROXIMAL_CURRENT_PA = 570
RELAY_CURRENT_PA = 20_000
DT_US = 50
LATENCY_MUST_EXCEED_US = 100


class CalibrationError(RuntimeError):
    """The implementation or artifact violates the frozen electrical screen."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CalibrationError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _source_hashes() -> dict[str, str]:
    return {
        "cells.py": _sha256(CELLS_SOURCE.read_bytes()),
        "cortex.py": _sha256(CORTEX_SOURCE.read_bytes()),
        "sparse_temporal_cortex.py": _sha256(SPARSE_SOURCE.read_bytes()),
    }


def _claim_boundary() -> dict[str, bool]:
    return {
        "capability_evidence": False,
        "sparse_temporal_integration_evidence": False,
        "synthetic_electrical_calibration": True,
        "transfer_evidence": False,
    }


def _scenario_currents() -> dict[str, tuple[int, ...]]:
    novel_losers = tuple(
        current + HOMEOSTASIS_ADVERSARIAL_PA
        for current in NOVEL_CURRENT_LADDER_PA[1:]
    )
    return {
        "active_vs_matching": (
            ACTIVE_CURRENT_PA - HOMEOSTASIS_ADVERSARIAL_PA,
            MATCHING_CURRENT_PA + HOMEOSTASIS_ADVERSARIAL_PA,
            NOVEL_CURRENT_LADDER_PA[0] + HOMEOSTASIS_ADVERSARIAL_PA,
            *novel_losers[:5],
        ),
        "matching_vs_novel": (
            MATCHING_CURRENT_PA - HOMEOSTASIS_ADVERSARIAL_PA,
            NOVEL_CURRENT_LADDER_PA[0] + HOMEOSTASIS_ADVERSARIAL_PA,
            *novel_losers[:6],
        ),
        "novel_rank_boundary": (
            NOVEL_CURRENT_LADDER_PA[0] - HOMEOSTASIS_ADVERSARIAL_PA,
            *novel_losers,
        ),
    }


def _protocol() -> dict[str, Any]:
    return {
        "active_current_pa": ACTIVE_CURRENT_PA,
        "basket_current_pa": BASKET_CURRENT_PA,
        "basket_offset": BASKET_OFFSET,
        "base_pyramidal_cells": BASE_PYRAMIDAL_CELLS,
        "cells_per_column": CELLS_PER_COLUMN,
        "columns": COLUMNS,
        "construction_seeds": list(CONSTRUCTION_SEEDS),
        "dt_us": DT_US,
        "gaba_current_pa": GABA_CURRENT_PA,
        "homeostasis_adversary_rule": (
            "boundary winner receives -20 pA; every competitor receives +20 pA"
        ),
        "homeostasis_adversarial_pa": HOMEOSTASIS_ADVERSARIAL_PA,
        "latency_must_exceed_us": LATENCY_MUST_EXCEED_US,
        "matching_current_pa": MATCHING_CURRENT_PA,
        "max_pyramidal_cells": MAX_PYRAMIDAL_CELLS,
        "novel_current_ladder_pa": list(NOVEL_CURRENT_LADDER_PA),
        "physical_cell_count": CELL_COUNT,
        "proximal_current_pa": PROXIMAL_CURRENT_PA,
        "proximal_offset": PROXIMAL_OFFSET,
        "proximal_relay_offset": PROXIMAL_RELAY_OFFSET,
        "rank_assignment": (
            "rank 0 is assigned to each column's maximum threshold offset, rank 1 "
            "to its minimum, and descending remainder currents to ascending "
            "remaining thresholds"
        ),
        "relay_offset": RELAY_OFFSET,
        "relay_current_pa": RELAY_CURRENT_PA,
        "scenario_currents_pa": {
            name: list(currents)
            for name, currents in sorted(_scenario_currents().items())
        },
        "singleton_matching_active_assumption": True,
    }


def _runtime_config_hashes() -> dict[str, str]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from humanbrain.cortex import HumanCellParameters  # noqa: PLC0415
    from humanbrain.sparse_temporal_cortex import (  # noqa: PLC0415
        SparseTemporalConfig,
    )

    sparse = SparseTemporalConfig(column_count=COLUMNS)
    electrical_defaults = {
        "active_current_pa": sparse.active_current_pa,
        "base_cells_per_column": sparse.base_cells_per_column,
        "basket_current_pa": sparse.basket_current_pa,
        "basket_inhibition_pa": sparse.basket_inhibition_pa,
        "basket_offset": BASKET_OFFSET,
        "column_count": sparse.column_count,
        "homeostatic_current_pa": sparse.homeostatic_current_pa,
        "matching_current_pa": sparse.matching_current_pa,
        "max_pyramidal_cells": sparse.max_pyramidal_cells,
        "novel_current_pa": sparse.novel_current_pa,
        "novel_race_cap": sparse.novel_race_cap,
        "novel_rank_spacing_pa": sparse.novel_rank_spacing_pa,
        "physical_cell_count": CELL_COUNT,
        "proximal_current_pa": sparse.proximal_current_pa,
        "proximal_offset": PROXIMAL_OFFSET,
        "proximal_relay_offset": PROXIMAL_RELAY_OFFSET,
        "relay_current_pa": sparse.relay_current_pa,
        "relay_offset": RELAY_OFFSET,
    }
    return {
        "human_cell_parameters": _sha256(
            _canonical_json_bytes(asdict(HumanCellParameters()))
        ),
        "sparse_electrical_defaults": _sha256(
            _canonical_json_bytes(electrical_defaults)
        ),
    }


def _new_cortex(seed: int) -> Any:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from humanbrain.cortex import DendriticCortex  # noqa: PLC0415

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


def _probe_trial(
    cortex: Any,
    *,
    column: int,
    currents: Sequence[int],
) -> dict[str, int]:
    _require(0 <= column < COLUMNS, "column outside calibrated sheet")
    current_tuple = tuple(int(value) for value in currents)
    _require(len(current_tuple) == CELLS_PER_COLUMN, "scenario width drift")
    first_cell = column * CELLS_PER_COLUMN
    column_cells = tuple(
        first_cell + offset for offset in range(CELLS_PER_COLUMN)
    )
    by_threshold = tuple(
        sorted(
            column_cells,
            key=lambda cell: (float(cortex._threshold_offset[cell]), cell),
        )
    )
    cells = (by_threshold[-1], by_threshold[0], *by_threshold[1:-1])
    basket_cell = BASKET_OFFSET + column

    control = cortex.race_currents_with_inhibition(
        cells[:2],
        current_tuple[:2],
        basket_cell_id=basket_cell,
        basket_current_pa=BASKET_CURRENT_PA,
        inhibition_current_pa=0.0,
    )
    inhibited = cortex.race_currents_with_inhibition(
        cells,
        current_tuple,
        basket_cell_id=basket_cell,
        basket_current_pa=BASKET_CURRENT_PA,
        inhibition_current_pa=GABA_CURRENT_PA,
    )

    control_times = tuple(_ms_to_ticks(value) for value in control.pyramidal_spike_times_ms)
    inhibited_winner_tick = _ms_to_ticks(inhibited.winner_spike_time_ms)
    basket_tick = _ms_to_ticks(inhibited.basket_spike_time_ms)
    gaba_tick = _ms_to_ticks(inhibited.inhibition_onset_ms)
    natural_latency = (
        control_times[1] - control_times[0]
        if len(control_times) >= 2
        else -1
    )
    return {
        "basket_to_gaba_ticks": gaba_tick - basket_tick,
        "control_spike_count": len(control.pyramidal_spikes),
        "control_winner": int(control.winner),
        "expected_winner": cells[0],
        "inhibited_spike_count": len(inhibited.pyramidal_spikes),
        "inhibited_winner": int(inhibited.winner),
        "natural_latency_ticks": natural_latency,
        "winner_to_basket_ticks": basket_tick - inhibited_winner_tick,
    }


def _probe_proximal_channel(cortex: Any, *, column: int) -> dict[str, int]:
    _require(0 <= column < COLUMNS, "column outside calibrated sheet")
    source_gid = PROXIMAL_OFFSET + column
    relay_gid = PROXIMAL_RELAY_OFFSET + column
    source = cortex.race_currents([source_gid], [PROXIMAL_CURRENT_PA])
    relay = cortex.race_currents([relay_gid], [RELAY_CURRENT_PA])
    source_ticks = _ms_to_ticks(source.spike_time_ms)
    relay_intrinsic_ticks = _ms_to_ticks(relay.spike_time_ms)
    return {
        "causal_relay_ticks": source_ticks + relay_intrinsic_ticks,
        "expected_relay": relay_gid,
        "expected_source": source_gid,
        "relay_active_cells": relay.active_cells,
        "relay_intrinsic_ticks": relay_intrinsic_ticks,
        "relay_winner": int(relay.winner),
        "source_active_cells": source.active_cells,
        "source_ticks": source_ticks,
        "source_winner": int(source.winner),
    }


def _new_boundary_summary() -> dict[str, Any]:
    return {
        "control_two_spike_trials": 0,
        "expected_control_winner_trials": 0,
        "expected_inhibited_winner_trials": 0,
        "inhibited_single_spike_trials": 0,
        "maximum_basket_to_gaba_ticks": 0,
        "maximum_natural_latency_ticks": 0,
        "maximum_winner_to_basket_ticks": 0,
        "minimum_basket_to_gaba_ticks": None,
        "minimum_natural_latency_ticks": None,
        "minimum_winner_to_basket_ticks": None,
        "seed_hashes_sha256": {},
        "trials": 0,
    }


def _update_extrema(summary: dict[str, Any], name: str, value: int) -> None:
    minimum = f"minimum_{name}"
    maximum = f"maximum_{name}"
    current_minimum = summary[minimum]
    summary[minimum] = value if current_minimum is None else min(current_minimum, value)
    summary[maximum] = max(summary[maximum], value)


def _record_trial(summary: dict[str, Any], observed: dict[str, int]) -> None:
    summary["trials"] += 1
    summary["control_two_spike_trials"] += int(
        observed["control_spike_count"] == 2
    )
    summary["expected_control_winner_trials"] += int(
        observed["control_winner"] == observed["expected_winner"]
    )
    summary["inhibited_single_spike_trials"] += int(
        observed["inhibited_spike_count"] == 1
    )
    summary["expected_inhibited_winner_trials"] += int(
        observed["inhibited_winner"] == observed["expected_winner"]
    )
    for name in (
        "basket_to_gaba_ticks",
        "natural_latency_ticks",
        "winner_to_basket_ticks",
    ):
        _update_extrema(summary, name, observed[name])


def _trial_bytes(observed: dict[str, int]) -> bytes:
    ordered = (
        observed["expected_winner"],
        observed["control_winner"],
        observed["control_spike_count"],
        observed["natural_latency_ticks"],
        observed["inhibited_winner"],
        observed["inhibited_spike_count"],
        observed["winner_to_basket_ticks"],
        observed["basket_to_gaba_ticks"],
    )
    return struct.pack("<" + "q" * len(ordered), *ordered)


def _run_proximal_sweep() -> dict[str, Any]:
    summary: dict[str, Any] = {
        "causal_relay_after_source_trials": 0,
        "dedicated_relay_single_spike_trials": 0,
        "expected_relay_winner_trials": 0,
        "expected_source_winner_trials": 0,
        "maximum_causal_relay_ticks": 0,
        "maximum_relay_intrinsic_ticks": 0,
        "maximum_source_ticks": 0,
        "minimum_causal_relay_ticks": None,
        "minimum_relay_intrinsic_ticks": None,
        "minimum_source_ticks": None,
        "seed_hashes_sha256": {},
        "source_single_spike_trials": 0,
        "trials": 0,
    }
    for seed in CONSTRUCTION_SEEDS:
        cortex = _new_cortex(seed)
        seed_hasher = hashlib.sha256()
        for column in range(COLUMNS):
            observed = _probe_proximal_channel(cortex, column=column)
            summary["trials"] += 1
            summary["source_single_spike_trials"] += int(
                observed["source_active_cells"] == 1
            )
            summary["dedicated_relay_single_spike_trials"] += int(
                observed["relay_active_cells"] == 1
            )
            summary["expected_source_winner_trials"] += int(
                observed["source_winner"] == observed["expected_source"]
            )
            summary["expected_relay_winner_trials"] += int(
                observed["relay_winner"] == observed["expected_relay"]
            )
            summary["causal_relay_after_source_trials"] += int(
                observed["causal_relay_ticks"] > observed["source_ticks"]
            )
            for name in (
                "causal_relay_ticks",
                "relay_intrinsic_ticks",
                "source_ticks",
            ):
                _update_extrema(summary, name, observed[name])
            packed = tuple(observed[key] for key in sorted(observed))
            seed_hasher.update(struct.pack("<" + "q" * len(packed), *packed))
        summary["seed_hashes_sha256"][str(seed)] = seed_hasher.hexdigest()

    trials = COLUMNS * len(CONSTRUCTION_SEEDS)
    passed = (
        summary["trials"] == trials
        and summary["source_single_spike_trials"] == trials
        and summary["dedicated_relay_single_spike_trials"] == trials
        and summary["expected_source_winner_trials"] == trials
        and summary["expected_relay_winner_trials"] == trials
        and summary["causal_relay_after_source_trials"] == trials
    )
    summary["verdict"] = (
        "PASS_EXHAUSTIVE_PROXIMAL_RELAY_SCREEN"
        if passed
        else "REJECT_EXHAUSTIVE_PROXIMAL_RELAY_SCREEN"
    )
    return summary


def _finalize_boundary(summary: dict[str, Any]) -> dict[str, Any]:
    trials = COLUMNS * len(CONSTRUCTION_SEEDS)
    passed = (
        summary["trials"] == trials
        and summary["control_two_spike_trials"] == trials
        and summary["expected_control_winner_trials"] == trials
        and summary["inhibited_single_spike_trials"] == trials
        and summary["expected_inhibited_winner_trials"] == trials
        and summary["minimum_natural_latency_ticks"] * DT_US
        > LATENCY_MUST_EXCEED_US
        and summary["minimum_winner_to_basket_ticks"] == 1
        and summary["maximum_winner_to_basket_ticks"] == 1
        and summary["minimum_basket_to_gaba_ticks"] == 1
        and summary["maximum_basket_to_gaba_ticks"] == 1
    )
    summary["verdict"] = (
        "PASS_EXHAUSTIVE_COUPLED_SCREEN"
        if passed
        else "REJECT_EXHAUSTIVE_COUPLED_SCREEN"
    )
    return summary


def run_replay() -> dict[str, Any]:
    source_hashes = _source_hashes()
    _require(source_hashes["cortex.py"] == CORTEX_SOURCE_SHA256, "cortex source drift")
    _require(source_hashes["cells.py"] == CELLS_SOURCE_SHA256, "cell source drift")
    _require(
        source_hashes["sparse_temporal_cortex.py"] == SPARSE_SOURCE_SHA256,
        "sparse source drift",
    )
    config_hashes = _runtime_config_hashes()
    _require(
        config_hashes
        == {
            "human_cell_parameters": HUMAN_CELL_CONFIG_SHA256,
            "sparse_electrical_defaults": SPARSE_ELECTRICAL_CONFIG_SHA256,
        },
        "electrical config drift",
    )

    boundaries: dict[str, Any] = {}
    for boundary_name, currents in sorted(_scenario_currents().items()):
        summary = _new_boundary_summary()
        for seed in CONSTRUCTION_SEEDS:
            cortex = _new_cortex(seed)
            seed_hasher = hashlib.sha256()
            for column in range(COLUMNS):
                observed = _probe_trial(cortex, column=column, currents=currents)
                _record_trial(summary, observed)
                seed_hasher.update(_trial_bytes(observed))
            summary["seed_hashes_sha256"][str(seed)] = seed_hasher.hexdigest()
        boundaries[boundary_name] = _finalize_boundary(summary)

    verdict = (
        "PASS_EXHAUSTIVE_COUPLED_SCREEN"
        if all(
            row["verdict"] == "PASS_EXHAUSTIVE_COUPLED_SCREEN"
            for row in boundaries.values()
        )
        else "REJECT_EXHAUSTIVE_COUPLED_SCREEN"
    )
    proximal_channel = _run_proximal_sweep()
    if proximal_channel["verdict"] != "PASS_EXHAUSTIVE_PROXIMAL_RELAY_SCREEN":
        verdict = "REJECT_EXHAUSTIVE_COUPLED_SCREEN"
    return {
        "artifact_id": ARTIFACT_ID,
        "boundaries": boundaries,
        "claim_boundary": _claim_boundary(),
        "config_sha256": config_hashes,
        "protocol": _protocol(),
        "proximal_channel": proximal_channel,
        "schema_version": SCHEMA_VERSION,
        "source_commit": SOURCE_COMMIT,
        "source_sha256": source_hashes,
        "verdict": verdict,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    _require(artifact.get("artifact_id") == ARTIFACT_ID, "artifact id mismatch")
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
    _require(artifact.get("source_commit") == SOURCE_COMMIT, "source commit drift")
    _require(artifact.get("claim_boundary") == _claim_boundary(), "claim drift")
    _require(artifact.get("protocol") == _protocol(), "protocol drift")
    _require(
        artifact.get("source_sha256")
        == {
            "cells.py": CELLS_SOURCE_SHA256,
            "cortex.py": CORTEX_SOURCE_SHA256,
            "sparse_temporal_cortex.py": SPARSE_SOURCE_SHA256,
        },
        "source hash evidence drift",
    )
    _require(
        artifact.get("config_sha256")
        == {
            "human_cell_parameters": HUMAN_CELL_CONFIG_SHA256,
            "sparse_electrical_defaults": SPARSE_ELECTRICAL_CONFIG_SHA256,
        },
        "electrical config evidence drift",
    )
    boundaries = artifact.get("boundaries")
    _require(isinstance(boundaries, dict), "boundary evidence is missing")
    _require(set(boundaries) == set(_scenario_currents()), "boundary grid drift")
    expected_trials = COLUMNS * len(CONSTRUCTION_SEEDS)
    for name, row in boundaries.items():
        _require(row.get("trials") == expected_trials, f"{name} trial count drift")
        _require(
            row.get("control_two_spike_trials") == expected_trials,
            f"{name} natural-latency control failed",
        )
        _require(
            row.get("expected_control_winner_trials") == expected_trials,
            f"{name} control winner drift",
        )
        _require(
            row.get("inhibited_single_spike_trials") == expected_trials,
            f"{name} causal inhibition emitted extra spikes",
        )
        _require(
            row.get("expected_inhibited_winner_trials") == expected_trials,
            f"{name} inhibited winner drift",
        )
        _require(
            row.get("minimum_natural_latency_ticks", 0) * DT_US
            > LATENCY_MUST_EXCEED_US,
            f"{name} natural latency is inside the GABA onset window",
        )
        for event in ("winner_to_basket_ticks", "basket_to_gaba_ticks"):
            _require(row.get(f"minimum_{event}") == 1, f"{name} {event} minimum drift")
            _require(row.get(f"maximum_{event}") == 1, f"{name} {event} maximum drift")
        seed_hashes = row.get("seed_hashes_sha256")
        _require(
            isinstance(seed_hashes, dict)
            and set(seed_hashes) == {str(seed) for seed in CONSTRUCTION_SEEDS}
            and all(
                isinstance(value, str) and len(value) == 64
                for value in seed_hashes.values()
            ),
            f"{name} seed hash evidence drift",
        )
        _require(
            row.get("verdict") == "PASS_EXHAUSTIVE_COUPLED_SCREEN",
            f"{name} did not pass",
        )
    proximal = artifact.get("proximal_channel")
    _require(isinstance(proximal, dict), "proximal-channel evidence is missing")
    for field in (
        "trials",
        "source_single_spike_trials",
        "dedicated_relay_single_spike_trials",
        "expected_source_winner_trials",
        "expected_relay_winner_trials",
        "causal_relay_after_source_trials",
    ):
        _require(
            proximal.get(field) == expected_trials,
            f"proximal-channel {field} drift",
        )
    for field in (
        "minimum_source_ticks",
        "maximum_source_ticks",
        "minimum_relay_intrinsic_ticks",
        "maximum_relay_intrinsic_ticks",
        "minimum_causal_relay_ticks",
        "maximum_causal_relay_ticks",
    ):
        _require(
            isinstance(proximal.get(field), int) and proximal[field] > 0,
            f"proximal-channel {field} is missing",
        )
    _require(
        proximal["minimum_causal_relay_ticks"]
        > proximal["minimum_source_ticks"],
        "proximal relay is not causally after its source",
    )
    proximal_hashes = proximal.get("seed_hashes_sha256")
    _require(
        isinstance(proximal_hashes, dict)
        and set(proximal_hashes) == {str(seed) for seed in CONSTRUCTION_SEEDS}
        and all(
            isinstance(value, str) and len(value) == 64
            for value in proximal_hashes.values()
        ),
        "proximal-channel seed hash evidence drift",
    )
    _require(
        proximal.get("verdict") == "PASS_EXHAUSTIVE_PROXIMAL_RELAY_SCREEN",
        "proximal-channel electrical screen did not pass",
    )
    _require(
        artifact.get("verdict") == "PASS_EXHAUSTIVE_COUPLED_SCREEN",
        "global electrical verdict did not pass",
    )


def load_and_validate(path: Path) -> dict[str, Any]:
    source_hashes = _source_hashes()
    _require(source_hashes["cortex.py"] == CORTEX_SOURCE_SHA256, "cortex source changed")
    _require(source_hashes["cells.py"] == CELLS_SOURCE_SHA256, "cell source changed")
    _require(
        source_hashes["sparse_temporal_cortex.py"] == SPARSE_SOURCE_SHA256,
        "sparse source changed",
    )
    _require(
        _runtime_config_hashes()
        == {
            "human_cell_parameters": HUMAN_CELL_CONFIG_SHA256,
            "sparse_electrical_defaults": SPARSE_ELECTRICAL_CONFIG_SHA256,
        },
        "electrical config changed",
    )
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
        help="repeat the exhaustive coupled-race sweep and compare the artifact",
    )
    actions.add_argument(
        "--write",
        action="store_true",
        help="repeat the exhaustive coupled-race sweep and write the artifact",
    )
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.check:
            load_and_validate(args.artifact)
            print("minicolumn electrical calibration artifact check: PASS")
            return 0
        started = time.perf_counter()
        observed = run_replay()
        validate_artifact(observed)
        elapsed = time.perf_counter() - started
        if args.write:
            _write_artifact(args.artifact, observed)
            print(
                "minicolumn electrical calibration artifact written: "
                f"{args.artifact} ({elapsed:.1f}s)"
            )
            return 0
        expected = load_and_validate(args.artifact)
        _require(
            _canonical_json_bytes(observed) == _canonical_json_bytes(expected),
            "electrical replay differs from committed artifact",
        )
        print(f"minicolumn electrical calibration replay check: PASS ({elapsed:.1f}s)")
        return 0
    except (CalibrationError, OSError, json.JSONDecodeError) as error:
        print(f"minicolumn electrical calibration: FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
