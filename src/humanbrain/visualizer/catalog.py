"""Reader for canonical ladder experiment result files."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from .model import (
    CURRENT_CONCURRENT_ACTIVITY_REFERENCE,
    DifferenceEstimate,
    ExperimentRun,
    PopulationSummary,
    ProportionEstimate,
    RunOutcome,
    StateSeparation,
)


REQUIRED_FIELDS = frozenset(
    {
        "experiment_id",
        "arm",
        "N",
        "seed",
        "readout_rule",
        "schedule",
        "data_scale",
        "supervision",
        "curriculum_sha256",
        "train_replay_tasks",
        "train_cases_total",
        "train_cases_passed",
        "calib_tasks",
        "calib_cases_total",
        "calib_cases_passed",
        "margin_count",
        "margin_min_pa",
        "margin_median_pa",
        "wall_s",
        "git_head",
        "config_dump",
    }
)


class ExperimentFormatError(ValueError):
    """A JSON file claims the ladder shape but violates its current contract."""


@dataclass(frozen=True, slots=True)
class CatalogIssue:
    path: Path
    message: str
    unsupported: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentCatalog:
    root: Path
    runs: tuple[ExperimentRun, ...]
    issues: tuple[CatalogIssue, ...]

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        return tuple(sorted({run.experiment_id for run in self.runs}))

    def for_experiment(self, experiment_id: str) -> tuple[ExperimentRun, ...]:
        return tuple(run for run in self.runs if run.experiment_id == experiment_id)


def _integer(record: Mapping[str, Any], key: str, *, minimum: int = 0) -> int:
    value = record.get(key)
    if type(value) is not int or value < minimum:
        raise ExperimentFormatError(f"{key} must be an integer >= {minimum}")
    return value


def _number(record: Mapping[str, Any], key: str, *, minimum: float | None = None) -> float:
    value = record.get(key)
    if type(value) not in (int, float):
        raise ExperimentFormatError(f"{key} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (minimum is not None and number < minimum):
        raise ExperimentFormatError(f"{key} must be finite and >= {minimum}")
    return number


def _string(record: Mapping[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        raise ExperimentFormatError(f"{key} must be a non-empty string")
    return value


def _mapping(record: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ExperimentFormatError(f"{key} must be an object")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        return None
    return float(value)


def _state_separation(record: Mapping[str, Any]) -> StateSeparation:
    diagnostics = record.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return StateSeparation()
    state = diagnostics.get("state_separation")
    if not isinstance(state, dict):
        return StateSeparation()
    eligibility = state.get("eligibility")
    current = state.get("synaptic_current")
    return StateSeparation(
        eligibility_median_1minuscos=(
            _optional_float(eligibility.get("median_1minuscos_pooled"))
            if isinstance(eligibility, dict)
            else None
        ),
        synaptic_current_median_1minuscos=(
            _optional_float(current.get("median_1minuscos_pooled"))
            if isinstance(current, dict)
            else None
        ),
    )


def _population(
    *,
    n_cells: int,
    readout_rule: str,
    config_dump: Mapping[str, Any],
) -> PopulationSummary:
    circuit = config_dump.get("circuit_config")
    if not isinstance(circuit, dict):
        raise ExperimentFormatError("config_dump.circuit_config must be an object")
    configured = _integer(circuit, "reservoir_cells", minimum=1)
    if configured != n_cells:
        raise ExperimentFormatError("N does not match circuit_config.reservoir_cells")
    active_value = circuit.get("active_reservoir_cells")
    if active_value is None:
        enabled = configured
    elif type(active_value) is int and 1 <= active_value <= configured:
        enabled = active_value
    else:
        raise ExperimentFormatError("active_reservoir_cells must be null or fit N")

    inhibitory_fraction = _number(circuit, "inhibitory_fraction", minimum=0.0)
    if inhibitory_fraction >= 1.0:
        raise ExperimentFormatError("inhibitory_fraction must be below 1")
    inhibitory = round(inhibitory_fraction * enabled)
    excitatory = enabled - inhibitory
    recurrent_fanout = _integer(circuit, "recurrent_fanout", minimum=1)
    input_assembly = _integer(config_dump, "input_fanout", minimum=1)
    token_cells = _integer(config_dump, "token_cells", minimum=1)
    readout_fan_in = _integer(circuit, "readout_fan_in", minimum=1)
    readout_contacts = (
        token_cells * readout_fan_in
        if readout_rule == "sparse_three_factor"
        else token_cells * enabled
    )
    return PopulationSummary(
        model_family="GLIF5",
        configured_neurons=configured,
        enabled_neurons=enabled,
        excitatory_neurons=excitatory,
        inhibitory_neurons=inhibitory,
        concurrent_activity_reference_neurons=math.ceil(
            enabled * CURRENT_CONCURRENT_ACTIVITY_REFERENCE
        ),
        input_contacts=token_cells * input_assembly,
        recurrent_contacts=enabled * recurrent_fanout,
        readout_contacts=readout_contacts,
        recurrent_fanout=recurrent_fanout,
        input_assembly_size=input_assembly,
    )


def load_run(path: str | Path) -> ExperimentRun:
    source_path = Path(path)
    try:
        record = json.loads(source_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ExperimentFormatError(f"cannot read experiment JSON: {error}") from error
    if not isinstance(record, dict):
        raise ExperimentFormatError("experiment JSON root must be an object")
    missing = sorted(REQUIRED_FIELDS.difference(record))
    if missing:
        raise ExperimentFormatError(f"missing ladder fields: {', '.join(missing)}")

    experiment_id = _string(record, "experiment_id")
    arm = _string(record, "arm")
    if source_path.stem != arm:
        raise ExperimentFormatError("arm must match the JSON filename stem")
    if source_path.parent.name != experiment_id:
        raise ExperimentFormatError("experiment_id must match the parent directory")

    n_cells = _integer(record, "N", minimum=1)
    seed = _integer(record, "seed")
    data_scale = _integer(record, "data_scale", minimum=1)
    readout_rule = _string(record, "readout_rule")
    schedule = _string(record, "schedule")
    supervision = _string(record, "supervision")
    config_dump = _mapping(record, "config_dump")

    training = ProportionEstimate.from_counts(
        _integer(record, "train_cases_passed"),
        _integer(record, "train_cases_total"),
    )
    calibration = ProportionEstimate.from_counts(
        _integer(record, "calib_cases_passed"),
        _integer(record, "calib_cases_total"),
    )
    replay = ProportionEstimate.from_counts(
        _integer(record, "train_replay_tasks"),
        _integer(config_dump, "train_task_count", minimum=1),
    )
    outcome = RunOutcome(
        training=training,
        calibration=calibration,
        replay=replay,
        generalization_gap=DifferenceEstimate.between(training, calibration),
        margin_count=_integer(record, "margin_count"),
        margin_min_pa=_number(record, "margin_min_pa"),
        margin_median_pa=_number(record, "margin_median_pa"),
        wall_seconds=_number(record, "wall_s", minimum=0.0),
    )
    population = _population(
        n_cells=n_cells,
        readout_rule=readout_rule,
        config_dump=config_dump,
    )

    known = REQUIRED_FIELDS | {"diagnostics"}
    extensions = {key: value for key, value in record.items() if key not in known}
    return ExperimentRun(
        source_path=source_path,
        experiment_id=experiment_id,
        arm=arm,
        seed=seed,
        data_scale=data_scale,
        readout_rule=readout_rule,
        schedule=schedule,
        supervision=supervision,
        codebook=str(config_dump.get("codebook", "explicit")),
        simulation_backend=str(config_dump.get("simulation_backend", "cpu")),
        ridge_backend=str(config_dump.get("ridge_backend", "numpy")),
        population=population,
        outcome=outcome,
        state_separation=_state_separation(record),
        curriculum_sha256=_string(record, "curriculum_sha256"),
        git_head=_string(record, "git_head"),
        configuration=config_dump,
        extensions=extensions,
    )


def load_catalog(root: str | Path) -> ExperimentCatalog:
    root_path = Path(root).resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"experiment directory does not exist: {root_path}")
    runs: list[ExperimentRun] = []
    issues: list[CatalogIssue] = []
    for path in sorted(root_path.glob("*/*.json")):
        if path.name.startswith("_"):
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            issues.append(CatalogIssue(path=path, message=str(error)))
            continue
        if not isinstance(raw, dict) or not REQUIRED_FIELDS.issubset(raw):
            issues.append(
                CatalogIssue(
                    path=path,
                    message="not a ladder-experiment metrics record",
                    unsupported=True,
                )
            )
            continue
        try:
            runs.append(load_run(path))
        except ExperimentFormatError as error:
            issues.append(CatalogIssue(path=path, message=str(error)))
    runs.sort(key=lambda run: (run.experiment_id, run.population.configured_neurons, run.arm, run.seed))
    return ExperimentCatalog(root=root_path, runs=tuple(runs), issues=tuple(issues))
