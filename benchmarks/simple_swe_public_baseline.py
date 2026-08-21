from __future__ import annotations

import argparse
import ast
from collections import Counter
import gc
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXPERIMENT_PROTOCOL = (
    ROOT / "benchmarks" / "simple_swe_public_baseline_protocol_v1.json"
)
EXPECTED_SCHEMA = "human-cell-simple-swe-public-development-baseline-protocol-v1"
EXPECTED_STATUS = "noncanonical-public-development-only"
OUTCOME_GROUPS = {
    "syntax": frozenset({"syntax_error"}),
    "format": frozenset({"format_error"}),
    "policy": frozenset({"policy_violation"}),
    "signature": frozenset({"signature_error"}),
    "runtime": frozenset(
        {"runtime_error", "memory_limit", "timeout", "harness_error"}
    ),
    "assertion": frozenset({"regression_failure", "property_failure"}),
    "passed": frozenset({"passed"}),
}
SEMANTIC_TEST_OUTCOMES = frozenset(
    {"regression_failure", "property_failure", "passed"}
)
EXACT_SIGNATURE_OUTCOMES = frozenset(
    {
        "runtime_error",
        "memory_limit",
        "timeout",
        "regression_failure",
        "property_failure",
        "passed",
    }
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _load_experiment_protocol(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict) or payload.get("schema") != EXPECTED_SCHEMA:
        raise ValueError("unsupported public baseline protocol")
    if payload.get("status") != EXPECTED_STATUS:
        raise ValueError("public baseline protocol status is not noncanonical")
    task_protocol = payload.get("task_protocol")
    if not isinstance(task_protocol, dict):
        raise ValueError("task protocol declaration is missing")
    if task_protocol.get("partitions") != ["train", "dev"]:
        raise ValueError("public baseline must name exactly train then dev")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("scope declaration is missing")
    required_false = (
        "external_models",
        "external_pretrained_language_models",
        "heuristic_program_synthesis",
        "template_answers",
        "tool_use_during_synthesis",
    )
    if any(scope.get(name) is not False for name in required_false):
        raise ValueError("public baseline scope permits a forbidden synthesis aid")
    if scope.get("cell_language_model_required") is not True:
        raise ValueError("public baseline must require the cell language model")
    training = payload.get("training")
    if not isinstance(training, dict) or training.get("partition") != "train":
        raise ValueError("training must use the visible train partition")
    if any(
        training.get(name) is not False
        for name in ("solutions", "test_cases", "expected_outputs")
    ):
        raise ValueError("training may not consume solutions or grading data")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("at least one fixed candidate is required")
    candidate_ids = [candidate.get("candidate_id") for candidate in candidates]
    if any(not isinstance(candidate_id, str) or not candidate_id for candidate_id in candidate_ids):
        raise ValueError("candidate IDs must be non-empty strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    configured_groups = payload.get("scoring", {}).get("outcome_groups")
    expected_groups = {name: sorted(values) for name, values in OUTCOME_GROUPS.items()}
    if not isinstance(configured_groups, dict) or {
        name: sorted(values) for name, values in configured_groups.items()
    } != expected_groups:
        raise ValueError("outcome taxonomy differs from the runner")
    payload["_sha256"] = sha256(raw).hexdigest()
    payload["_path"] = str(path.resolve())
    return payload


def _verify_cell_source(base_commit: str) -> None:
    if not isinstance(base_commit, str) or len(base_commit) != 40:
        raise ValueError("cell source commit must be a full SHA-1")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", base_commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if ancestor.returncode != 0:
        raise RuntimeError("preregistered cell source is not an ancestor of HEAD")
    diff = subprocess.run(
        ["git", "diff", "--quiet", base_commit, "--", "src"],
        cwd=ROOT,
        check=False,
    )
    if diff.returncode != 0:
        raise RuntimeError("production cell source differs from the preregistered commit")


def _parsed_python(source: str) -> bool:
    try:
        ast.parse(source, mode="exec")
    except SyntaxError:
        return False
    return True


def _emit_with_activity(model: Any, prompt: str, max_new_tokens: int) -> dict[str, Any]:
    tokenizer = model._tokenizer
    context = tokenizer.encode(prompt)
    if not context:
        raise ValueError("public completion prompt cannot be empty")
    emitted: list[int] = []
    active_cells: list[int] = []
    spike_times_ms: list[float] = []
    simulated_steps: list[int] = []
    started = perf_counter()
    for _ in range(max_new_tokens):
        race = model._cortex.race(context)
        emitted.append(int(race.winner))
        context.append(int(race.winner))
        active_cells.append(int(race.active_cells))
        spike_times_ms.append(float(race.spike_time_ms))
        simulated_steps.append(int(race.simulated_steps))
    elapsed = perf_counter() - started
    text = tokenizer.decode(emitted)
    cell_steps = sum(
        active * steps for active, steps in zip(active_cells, simulated_steps, strict=True)
    )
    return {
        "emitted_token_ids": tuple(emitted),
        "emitted_text": text,
        "prompt_tokens": len(context) - len(emitted),
        "inference_seconds": elapsed,
        "activity": {
            "cell_races": len(emitted),
            "active_cell_events": sum(active_cells),
            "active_cells_min": min(active_cells),
            "active_cells_max": max(active_cells),
            "active_cells_mean": sum(active_cells) / len(active_cells),
            "unique_winner_cells": len(set(emitted)),
            "winner_cell_ids": sorted(set(emitted)),
            "mean_spike_time_ms": sum(spike_times_ms) / len(spike_times_ms),
            "simulated_cell_steps": cell_steps,
        },
    }


def _run_task(
    *,
    model: Any,
    task: Any,
    max_new_tokens: int,
    compiler: Any,
    limits: Any,
    python_executable: Path,
    python_executable_sha256: str,
    worker_source: bytes,
    worker_sha256: str,
) -> tuple[dict[str, Any], set[int]]:
    from humanbrain.simple_swe.compiler import CompileRequest
    from humanbrain.simple_swe.sandbox import execute_completion

    synthesis = _emit_with_activity(model, task.public.prompt, max_new_tokens)
    token_ids = synthesis["emitted_token_ids"]
    emitted_text = synthesis["emitted_text"]
    source = compiler.compile(
        CompileRequest(
            emitted_text=emitted_text,
            emitted_token_ids=token_ids,
            entry_point=task.public.entry_point,
            parameter_names=task.public.parameter_names,
        )
    )
    result = execute_completion(
        source,
        task.public,
        task.private,
        limits,
        python_executable=python_executable,
        python_executable_sha256=python_executable_sha256,
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )
    activity = dict(synthesis["activity"])
    winners = set(activity.pop("winner_cell_ids"))
    inference_seconds = float(synthesis["inference_seconds"])
    record = {
        "task_id": task.public.task_id,
        "task_sha256": task.public.task_sha256,
        "prompt_sha256": sha256(task.public.prompt.encode("utf-8")).hexdigest(),
        "outcome": result.outcome,
        "outcome_group": next(
            name for name, values in OUTCOME_GROUPS.items() if result.outcome in values
        ),
        "passed": result.passed,
        "detail": result.detail,
        "parsed_python": _parsed_python(source),
        "prompt_tokens": synthesis["prompt_tokens"],
        "emitted_tokens": len(token_ids),
        "emitted_token_sha256": sha256(
            _canonical_json_bytes(list(token_ids))
        ).hexdigest(),
        "source_bytes": len(source.encode("utf-8")),
        "source_sha256": sha256(source.encode("utf-8")).hexdigest(),
        "source_preview": source[:160],
        "inference_seconds": inference_seconds,
        "inference_tokens_per_second": (
            len(token_ids) / inference_seconds if inference_seconds else float("inf")
        ),
        "sandbox_seconds": result.wall_seconds,
        "sandboxed": result.sandboxed,
        "activity": activity,
    }
    return record, winners


def _aggregate_task_records(
    records: list[dict[str, Any]], winner_cells: set[int]
) -> dict[str, Any]:
    outcomes = Counter(record["outcome"] for record in records)
    groups = Counter(record["outcome_group"] for record in records)
    inference_seconds = sum(float(record["inference_seconds"]) for record in records)
    sandbox_seconds = sum(float(record["sandbox_seconds"]) for record in records)
    emitted_tokens = sum(int(record["emitted_tokens"]) for record in records)
    prompt_tokens = sum(int(record["prompt_tokens"]) for record in records)
    cell_races = sum(int(record["activity"]["cell_races"]) for record in records)
    active_cell_events = sum(
        int(record["activity"]["active_cell_events"]) for record in records
    )
    simulated_cell_steps = sum(
        int(record["activity"]["simulated_cell_steps"]) for record in records
    )
    passed = outcomes["passed"]
    return {
        "tasks": len(records),
        "passed": passed,
        "pass_rate": passed / len(records),
        "semantic_test_reached": sum(outcomes[name] for name in SEMANTIC_TEST_OUTCOMES),
        "exact_signature_reached": sum(outcomes[name] for name in EXACT_SIGNATURE_OUTCOMES),
        "parsed_python": sum(bool(record["parsed_python"]) for record in records),
        "outcomes": dict(sorted(outcomes.items())),
        "outcome_groups": dict(sorted(groups.items())),
        "prompt_tokens": prompt_tokens,
        "emitted_tokens": emitted_tokens,
        "inference_seconds": inference_seconds,
        "inference_tokens_per_second": (
            emitted_tokens / inference_seconds if inference_seconds else float("inf")
        ),
        "sandbox_seconds": sandbox_seconds,
        "cell_activity": {
            "cell_races": cell_races,
            "active_cell_events": active_cell_events,
            "mean_active_cells_per_race": active_cell_events / cell_races,
            "active_cells_min": min(
                int(record["activity"]["active_cells_min"]) for record in records
            ),
            "active_cells_max": max(
                int(record["activity"]["active_cells_max"]) for record in records
            ),
            "unique_winner_cells": len(winner_cells),
            "simulated_cell_steps": simulated_cell_steps,
            "cell_steps_per_second": (
                simulated_cell_steps / inference_seconds
                if inference_seconds
                else float("inf")
            ),
            "mean_spike_time_ms": sum(
                float(record["activity"]["mean_spike_time_ms"])
                * int(record["activity"]["cell_races"])
                for record in records
            )
            / cell_races,
        },
    }


def _score_partition(
    *,
    model: Any,
    tasks: Iterable[Any],
    max_new_tokens: int,
    compiler: Any,
    limits: Any,
    python_executable: Path,
    python_executable_sha256: str,
    worker_source: bytes,
    worker_sha256: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    winner_cells: set[int] = set()
    for task in tasks:
        record, task_winners = _run_task(
            model=model,
            task=task,
            max_new_tokens=max_new_tokens,
            compiler=compiler,
            limits=limits,
            python_executable=python_executable,
            python_executable_sha256=python_executable_sha256,
            worker_source=worker_source,
            worker_sha256=worker_sha256,
        )
        records.append(record)
        winner_cells.update(task_winners)
    return {
        "aggregate": _aggregate_task_records(records, winner_cells),
        "task_results": records,
    }


def _grow_candidate(candidate: Mapping[str, Any], prompts: list[str]) -> tuple[Any, float]:
    from humanbrain.language import CellLanguageModel, ModelConfig

    if candidate.get("model_class") != "humanbrain.language.CellLanguageModel":
        raise ValueError("public baseline supports only CellLanguageModel")
    raw_config = candidate.get("model_config")
    if not isinstance(raw_config, dict):
        raise ValueError("candidate model_config must be an object")
    config = ModelConfig(**raw_config)
    started = perf_counter()
    model = CellLanguageModel.grow(prompts, config)
    return model, perf_counter() - started


def _selection_key(candidate_result: Mapping[str, Any]) -> tuple[int, int, int, int]:
    aggregate = candidate_result["train"]["aggregate"]
    return (
        int(aggregate["passed"]),
        int(aggregate["semantic_test_reached"]),
        int(aggregate["exact_signature_reached"]),
        int(aggregate["parsed_python"]),
    )


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _last_commit_for_path(path: Path) -> str:
    return subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def run(
    *,
    experiment_protocol_path: Path,
    python_executable: Path,
    output_path: Path | None,
    runtime_directory: Path,
) -> dict[str, Any]:
    from humanbrain.simple_swe import _worker
    from humanbrain.simple_swe.compiler import RawPythonCompiler
    from humanbrain.simple_swe.protocol import load_protocol
    from humanbrain.simple_swe.sandbox import (
        load_worker_snapshot,
        python_runtime_identity,
    )
    from humanbrain.simple_swe.tasks import generate_visible_partition

    experiment = _load_experiment_protocol(experiment_protocol_path.resolve())
    _verify_cell_source(experiment["cell_source_commit"])
    task_protocol_declaration = experiment["task_protocol"]
    task_protocol_path = (ROOT / task_protocol_declaration["path"]).resolve()
    task_protocol_sha256 = _sha256_file(task_protocol_path)
    if task_protocol_sha256 != task_protocol_declaration["sha256"]:
        raise RuntimeError("visible task protocol differs from preregistration")
    task_protocol = load_protocol(task_protocol_path)
    if task_protocol.sha256 != task_protocol_sha256:
        raise RuntimeError("loaded task protocol hash mismatch")
    if task_protocol.max_new_tokens != experiment["synthesis"]["max_new_tokens"]:
        raise RuntimeError("synthesis token bound differs from task protocol")

    train_tasks = generate_visible_partition(task_protocol, "train")
    train_prompts = [task.public.prompt for task in train_tasks]
    runtime = python_runtime_identity(
        python_executable, task_protocol.raw["execution"]["language"]
    )
    worker_path = Path(_worker.__file__).resolve()
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)
    compiler = RawPythonCompiler()
    compiler_identity = compiler.identity
    max_new_tokens = int(experiment["synthesis"]["max_new_tokens"])

    candidate_results: list[dict[str, Any]] = []
    for candidate in experiment["candidates"]:
        model, growth_seconds = _grow_candidate(candidate, train_prompts)
        scored = _score_partition(
            model=model,
            tasks=train_tasks,
            max_new_tokens=max_new_tokens,
            compiler=compiler,
            limits=task_protocol.execution,
            python_executable=python_executable,
            python_executable_sha256=runtime["executable_sha256"],
            worker_source=worker_source,
            worker_sha256=worker_sha256,
        )
        candidate_results.append(
            {
                "candidate_id": candidate["candidate_id"],
                "model_class": candidate["model_class"],
                "model_config": candidate["model_config"],
                "learning_rule": (
                    "additive structurally grown context-to-token dendritic branches "
                    "with cell-race Witten-Bell synaptic-current interpolation"
                ),
                "training_sequences": len(train_prompts),
                "training_characters": sum(len(prompt) for prompt in train_prompts),
                "growth_seconds": growth_seconds,
                "training_digest": model.training_digest,
                "token_cells": model.vocabulary_size,
                "grown_branches": model.grown_branches,
                "train": scored,
            }
        )
        del model
        gc.collect()

    ordered = sorted(candidate_results, key=lambda result: result["candidate_id"])
    selected_result = max(ordered, key=_selection_key)
    selected_id = selected_result["candidate_id"]
    selected_spec = next(
        candidate
        for candidate in experiment["candidates"]
        if candidate["candidate_id"] == selected_id
    )

    selected_model, checkpoint_growth_seconds = _grow_candidate(
        selected_spec, train_prompts
    )
    if selected_model.training_digest != selected_result["training_digest"]:
        raise RuntimeError("selected candidate retraining digest changed")
    if selected_model.vocabulary_size != selected_result["token_cells"]:
        raise RuntimeError("selected candidate retraining cell count changed")
    if selected_model.grown_branches != selected_result["grown_branches"]:
        raise RuntimeError("selected candidate retraining branches changed")
    runtime_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = runtime_directory / "selected_prompt_only_cells.npz"
    checkpoint_sha256 = selected_model.save(checkpoint_path)
    checkpoint_bytes = checkpoint_path.stat().st_size
    del selected_model
    gc.collect()

    from humanbrain.language import CellLanguageModel

    development_model = CellLanguageModel.load(checkpoint_path)
    if development_model.training_digest != selected_result["training_digest"]:
        raise RuntimeError("reloaded checkpoint training digest changed")
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("selected checkpoint hash changed before development")

    # This is intentionally the first generation of the development partition.
    # Candidate choice and checkpoint bytes are already fixed above.
    dev_tasks = generate_visible_partition(task_protocol, "dev")
    development = _score_partition(
        model=development_model,
        tasks=dev_tasks,
        max_new_tokens=max_new_tokens,
        compiler=compiler,
        limits=task_protocol.execution,
        python_executable=python_executable,
        python_executable_sha256=runtime["executable_sha256"],
        worker_source=worker_source,
        worker_sha256=worker_sha256,
    )
    if _sha256_file(checkpoint_path) != checkpoint_sha256:
        raise RuntimeError("selected checkpoint file changed during development")

    task_source = ROOT / "src" / "humanbrain" / "simple_swe" / "tasks.py"
    runner_path = Path(__file__).resolve()
    command_argv = [str(Path(sys.executable).resolve()), *sys.argv]
    artifact = {
        "schema": "human-cell-simple-swe-public-development-baseline-result-v1",
        "status": EXPECTED_STATUS,
        "capability_claim": (
            "Only passed tasks count as solved; emission, parsing, exact signatures, "
            "cell activity, checkpointing, and sandbox operation are plumbing evidence."
        ),
        "experiment_protocol": {
            "path": str(experiment_protocol_path.resolve()),
            "sha256": experiment["_sha256"],
            "preregistered_commit": _last_commit_for_path(
                experiment_protocol_path.resolve()
            ),
        },
        "task_protocol": {
            "protocol_id": task_protocol.protocol_id,
            "path": str(task_protocol_path),
            "sha256": task_protocol.sha256,
            "train_tasks": len(train_tasks),
            "dev_tasks": len(dev_tasks),
        },
        "source": {
            "cell_source_commit": experiment["cell_source_commit"],
            "run_head": _git_head(),
            "production_source_unchanged_from_cell_commit": True,
            "task_generator_sha256": _sha256_file(task_source),
            "runner_sha256": _sha256_file(runner_path),
        },
        "boundary_audit": {
            "generated_partitions": ["train", "dev"],
            "development_generated_after_train_selection": True,
            "model_training_inputs": "train public prompts only",
            "model_received_solutions": False,
            "model_received_test_cases": False,
            "model_received_expected_outputs": False,
            "external_model_calls": 0,
            "heuristic_solver_calls": 0,
            "template_answer_count": 0,
            "tool_calls_during_synthesis": 0,
        },
        "runtime": {
            **runtime,
            "executable": str(python_executable.resolve()),
            "worker_path": str(worker_path),
            "worker_sha256": worker_sha256,
        },
        "compiler": {
            "compiler_id": compiler_identity.compiler_id,
            "compiler_sha256": compiler_identity.compiler_sha256,
            "mode": compiler_identity.mode,
            "algorithm_source": compiler_identity.algorithm_source,
        },
        "command": {
            "cwd": str(ROOT),
            "argv": command_argv,
            "windows_command_line": subprocess.list2cmdline(command_argv),
        },
        "candidate_train_results": candidate_results,
        "selection": {
            "candidate_id": selected_id,
            "ranking": list(_selection_key(selected_result)),
            "ranking_fields": experiment["selection"]["maximize_lexicographically"],
            "tie_break": experiment["selection"]["final_tiebreak"],
            "development_changed_selection": False,
        },
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_sha256,
            "bytes": checkpoint_bytes,
            "format": "human-cell-brain-checkpoint-v1",
            "candidate_id": selected_id,
            "model_config": selected_spec["model_config"],
            "training_digest": development_model.training_digest,
            "token_cells": development_model.vocabulary_size,
            "grown_branches": development_model.grown_branches,
            "retraining_seconds": checkpoint_growth_seconds,
            "reloaded_before_development": True,
        },
        "development": development,
    }
    destination = output_path or (ROOT / experiment["reporting"]["artifact"])
    destination = destination.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
        + b"\n"
    )
    temporary.replace(destination)
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the preregistered noncanonical public Simple-SWE cell baseline."
    )
    parser.add_argument(
        "--protocol", type=Path, default=DEFAULT_EXPERIMENT_PROTOCOL
    )
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--runtime-directory",
        type=Path,
        default=ROOT / ".runtime" / "simple-swe-public-baseline-v1",
    )
    args = parser.parse_args()
    artifact = run(
        experiment_protocol_path=args.protocol,
        python_executable=args.python_executable.resolve(),
        output_path=args.output,
        runtime_directory=args.runtime_directory.resolve(),
    )
    summary = {
        "status": artifact["status"],
        "protocol_sha256": artifact["experiment_protocol"]["sha256"],
        "selected_candidate": artifact["selection"]["candidate_id"],
        "train": next(
            result["train"]["aggregate"]
            for result in artifact["candidate_train_results"]
            if result["candidate_id"] == artifact["selection"]["candidate_id"]
        ),
        "dev": artifact["development"]["aggregate"],
        "checkpoint_sha256": artifact["checkpoint"]["sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
