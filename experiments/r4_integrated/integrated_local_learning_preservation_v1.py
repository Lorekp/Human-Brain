"""Bounded retained-forward preservation; never an acquisition trial.

Run predecessor and current separately in fresh monitored interpreters. The outer
launcher must enforce the declared wall-time and process-tree RSS ceilings; local
admission checks cannot interrupt a running tensor operation. Comparison reads
only the new authenticated JSON/execution artifacts, with no neuronal imports.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

from experiments.r4_integrated import integrated_local_learning_runtime_v1 as runtime

DRAFT = False
ROOT = runtime.ROOT
CASES = ("A00-R", "A00-E", "R02-R", "R12-R")
MANIFEST_SHA = "528f6599d0ff5af487a418a7a0fa06b340ba0d026328df4bab534af671bd64cf"
SECONDS = 180
FINAL_SECONDS = 20
RSS_BYTES = 4096 * 1024**2
ARTIFACT_BYTES = 256 * 1024**2
PENDING_BYTES = 64 * 1024**2
JSON_BYTES = 64 * 1024**2
FREE_BYTES = 8 * 1024**3
RNG_NAME = "reader.controller.rng_state"
METADATA_NAMES = ("_extra_state", "reader._extra_state", "reader.controller._extra_state")
IMPLEMENTATION_REVISION = "7e164e79d4152c7009afb2411bc2f970c62c1f3f"
CONTROLLER_SHA = "bc53c0f5a641c1b0e0d05034db1b93cacd32250abb1b2295b36c77f7278afd2f"
LEARNER_SHA = "c268563ca357663803d524c4db6f52ee1bfdfbb410f2497ed4bd9874a1ffac0d"
RUNTIME_SHA = "e7dc49253f681976e516770ab2cc1eb8c4ce2b9199fdfbadd40f6bc96a46c34b"
INITIALIZER_SHA = "ea1c2692721b9da49b36c3e6fac8756322ab13641125537b2a443664fdd370e1"
IMPLEMENTATION_SOURCES = {
    "src/humanbrain/recurrent_content_controller.py": CONTROLLER_SHA,
    "src/humanbrain/recurrent_content_learner.py": LEARNER_SHA,
    "experiments/r4_integrated/integrated_local_learning_runtime_v1.py": RUNTIME_SHA,
    "experiments/r4_integrated/integrated_local_learning_initialize_v1.py": INITIALIZER_SHA,
}


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, expected=None, *, limit=JSON_BYTES):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit or (expected is not None and _sha(raw) != expected):
        raise ValueError("preservation evidence exceeds its bound or changed")
    return raw


def _inventory(directory):
    result = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("preservation output cannot contain symbolic links")
        if path.is_file():
            result[path.relative_to(directory).as_posix()] = path.stat().st_size
    return result


def _save(directory, name, value, *, artifact_limit=ARTIFACT_BYTES):
    raw = _encode(value)
    used = sum(_inventory(directory).values())
    if (
        len(raw) > JSON_BYTES
        or used + len(raw) > artifact_limit
        or shutil.disk_usage(directory).free - len(raw) < FREE_BYTES
    ):
        raise RuntimeError("preservation receipt exceeds artifact/free-space bounds")
    with (directory / name).open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"bytes": len(raw), "sha256": _sha(raw)}


def _sources():
    """Bind actual imported project source, including the selected two originals."""
    # This pure reference module is imported lazily by retained history restore.
    paths = {
        Path(__file__).resolve(),
        ROOT / "experiments/r4_integrated/persistent_language_action_baseline_v2.py",
    }
    for module_name, module in tuple(sys.modules.items()):
        origin = getattr(module, "__file__", None)
        if origin is None:
            continue
        path = Path(origin).resolve()
        if (
            path.suffix == ".py"
            and path.is_relative_to(ROOT)
            and (
                path.is_relative_to(ROOT / "experiments")
                or path.is_relative_to(ROOT / "src/humanbrain")
                or module_name.startswith("humanbrain.")
            )
        ):
            paths.add(path)
    return {path.relative_to(ROOT).as_posix(): _sha(_read(path)) for path in sorted(paths)}


def _check_sources(sources):
    for name, identity in sources.items():
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT) or path.is_symlink():
            raise ValueError("source binding escaped the checkout")
        _read(path, identity)


def _manifest():
    manifest = json.loads(_read(runtime.PARENT / "checkpoints/manifest.json", MANIFEST_SHA))
    if tuple(manifest["checkpoint_cases"]) != CASES or set(manifest["tasks"]) != set(CASES):
        raise ValueError("preservation requires exactly the four existing cases")
    return manifest


def _observation_record(manifest, name):
    episode = manifest["tasks"][name]
    if name.startswith("R"):
        record = copy.deepcopy(
            manifest["prior_regression_bindings"]["routes"]["online"][episode["prior_episode_id"]][
                "observation"
            ]
        )
        if record["request_hex"] != episode["request_hex"]:
            raise ValueError("historical preservation request differs from its saved observation")
        return record
    return {
        "schema": "sequence-assembly-raw-observation-v1",
        "request_hex": episode["request_hex"],
        "files": episode["files"] if episode["kind"] == "edit" else None,
        "feedback": None,
    }


def _forward(summary):
    """Separate declared rule/custody differences from every saved forward value."""
    expected = {
        "schema",
        "attempt_id",
        "model_sha256",
        "memory_sha256",
        "trace_sha256",
        "observation_sha256",
        "max_bytes",
        "answer_hex",
        "stop_reason",
        "decisions",
        "scan_steps",
        "controller",
        "steps",
        "work",
    }
    if set(summary) != expected:
        raise ValueError("reader summary schema changed")
    result = {
        key: value
        for key, value in summary.items()
        if key not in {"attempt_id", "model_sha256", "trace_sha256", "work"}
    }
    result["controller"] = dict(result["controller"])
    del result["controller"]["pending_stage_sha256"]
    result["scan_steps"] = [dict(row) for row in result["scan_steps"]]
    for row in result["scan_steps"]:
        del row["credit"]
    return result


def _execution_behavior(execution):
    """Only machine location, PID, elapsed time and location-bearing stderr vary."""
    excluded = {
        "workspace",
        "child_pid",
        "child_command",
        "child_seconds",
        "elapsed_seconds",
        "delegated_execution_seconds",
        "stderr_hex",
        "stderr_sha256",
        "stderr_observed_bytes",
    }
    return {key: value for key, value in execution.items() if key not in excluded}


def _response_counts(row):
    summary, resolution = row["reader"], row["resolution"]
    scan, free = summary["scan_steps"], summary["steps"]
    draws = sum(len(tick["draws"]) for tick in scan) + sum(
        len(step["controller"]["draws"]) for step in free
    )
    if (
        resolution["learning_enabled"] is not False
        or resolution["rng_published"] is not True
        or resolution["draws"] != draws
        or resolution["ticks"] != {"scan": len(scan), "free": len(free), "teacher": 0}
        or summary["controller"]["scan_offers"] != len(scan)
    ):
        raise ValueError("disabled resolution differs from its actual scan/free/RNG work")
    return {
        "scan_offers": len(scan),
        "free_ticks": len(free),
        "controller_draws": draws,
        "rng_publications": 1,
    }


def _no_neuronal_imports():
    if any(name == "humanbrain" or name.startswith("humanbrain.") for name in sys.modules):
        raise ValueError("saved-only comparison cannot import neuronal modules")


def run(args):
    if DRAFT:
        raise RuntimeError("DRAFT preservation worker requires root source/protocol release")
    started = time.perf_counter()
    output = Path(args.output).resolve()
    if output == ROOT or not output.is_relative_to(ROOT):
        raise ValueError("preservation output needs a new checkout-local directory")
    if shutil.disk_usage(ROOT).free < FREE_BYTES + ARTIFACT_BYTES:
        raise RuntimeError("preservation needs its artifact allowance above the8GiB free floor")
    output.mkdir(parents=True, exist_ok=False)
    records, completed, work, phases = {}, [], {}, {}
    live = {
        "free_responses": 0,
        "executions": 0,
        "teachers": 0,
        "ingests": 0,
        "learning_publications": 0,
        "scan_offers": 0,
        "free_ticks": 0,
        "controller_draws": 0,
        "rng_publications": 0,
        "rss_boundary_peak_bytes": 0,
    }
    pending = None
    try:
        _check_sources(IMPLEMENTATION_SOURCES)
        phase_start = time.perf_counter()
        selection = runtime.select(args.rule, args.predecessor_snapshot)
        import psutil
        import torch

        from experiments.r4_integrated.integrated_local_learning_initialize_v1 import (
            initialize_current,
            initialize_predecessor,
        )
        from experiments.r4_integrated.persistent_language_action_execution_v2 import (
            grade_execution,
            workspace_from_records,
        )
        from experiments.r4_integrated.persistent_language_action_v8 import _plain, reader_summary
        from experiments.r4_integrated.persistent_language_action_v8_execution import (
            execute_proposal,
            qualified_launch,
        )
        from humanbrain.neuronal_splice_learner import _digest
        from humanbrain.recurrent_content_learner import RecurrentContentSpliceLearner
        from humanbrain.sequence_assembly_observation import RawObservation, ToolFeedback

        torch.set_num_threads(1)
        # Retained reference restore imports this pure module lazily. Include
        # its transitive source closure before any parent initialization work.
        importlib.import_module("experiments.r4_integrated.persistent_language_action_baseline_v2")
        process = psutil.Process()
        phases["selection_and_import_seconds"] = time.perf_counter() - phase_start

        def admit(*, pending_write=True):
            tick = time.perf_counter()
            used, free = sum(_inventory(output).values()), shutil.disk_usage(output).free
            rss = process.memory_info().rss
            live.update(artifact_bytes=used, free_bytes=free)
            live["rss_boundary_peak_bytes"] = max(live["rss_boundary_peak_bytes"], rss)
            reserve = PENDING_BYTES if pending_write else 0
            if used + reserve > ARTIFACT_BYTES or free < FREE_BYTES + reserve or rss > RSS_BYTES:
                raise RuntimeError("preservation resource admission refused")
            limit = SECONDS - (FINAL_SECONDS if pending_write else 0)
            if time.perf_counter() - started >= limit:
                raise TimeoutError("preservation time reserve reached")
            phases["admission_seconds"] = (
                phases.get("admission_seconds", 0.0) + time.perf_counter() - tick
            )

        def publish(name, value):
            tick = time.perf_counter()
            records[name] = _save(output, name, value)
            phases["serialization_publication_seconds"] = (
                phases.get("serialization_publication_seconds", 0.0) + time.perf_counter() - tick
            )

        def numerical(model, state):
            tick = time.perf_counter()
            values = {
                key: value
                for key, value in model.state_dict().items()
                if isinstance(value, torch.Tensor)
            }
            result = {
                "numeric_sha256": _digest(values, work),
                "non_rng_numeric_sha256": _digest(
                    {k: v for k, v in values.items() if k != RNG_NAME}, work
                ),
                "rng_sha256": _digest({"rng_state": values[RNG_NAME]}, work),
                "memory_sha256": _digest(vars(state), work),
                "memory_used": state.used,
                "experience": model._experience,
                "reader_experience": model.reader._active_experience,
                "views_sha256": model.reader._view_sha256,
                "self_wiring_sha256": model.reader._self_wiring_sha256,
            }
            phases["numerical_hash_seconds"] = (
                phases.get("numerical_hash_seconds", 0.0) + time.perf_counter() - tick
            )
            return result

        admit()
        manifest = _manifest()
        publish("manifest.json", manifest)
        sources = _sources()
        origins_before = runtime.check_origins(selection)
        publish(
            "sources.json",
            {"selection": selection, "source_sha256": sources, "origins_before": origins_before},
        )
        initializer = initialize_predecessor if args.rule == "predecessor" else initialize_current
        phase_start = time.perf_counter()
        initialized = initializer(
            runtime.PARENT / "checkpoints",
            runtime.BANK,
            actor_class=RecurrentContentSpliceLearner,
            source_receipt=selection,
            predecessor_snapshot=args.predecessor_snapshot,
        )
        phases["initialization_seconds"] = time.perf_counter() - phase_start
        model, state = initialized["models"]["online"], initialized["states"]["online"]
        unused = initialized["models"]["learning_disabled"]
        unused_state = initialized["states"]["learning_disabled"]
        initial, unused_initial = numerical(model, state), numerical(unused, unused_state)
        metadata = {key: _plain(model.state_dict()[key]) for key in METADATA_NAMES}
        publish("initialization.json", initialized["receipt"])
        launch = qualified_launch(ROOT)
        publish("launch.json", launch.to_record())
        for index, name in enumerate(CASES):
            admit()
            tick = time.perf_counter()
            episode, public = manifest["tasks"][name], _observation_record(manifest, name)
            workspace = (
                workspace_from_records(public["files"]) if episode["kind"] == "edit" else None
            )
            feedback = public["feedback"]
            if feedback is not None:
                feedback = ToolFeedback(
                    **{k: v for k, v in feedback.items() if k not in {"stdout_hex", "stderr_hex"}},
                    stdout=bytes.fromhex(feedback["stdout_hex"]),
                    stderr=bytes.fromhex(feedback["stderr_hex"]),
                )
            observation = RawObservation(bytes.fromhex(public["request_hex"]), workspace, feedback)
            if observation.to_record() != public:
                raise ValueError("preservation observation reconstruction changed saved bytes")
            before = numerical(model, state)
            pending = {"index": index, "episode_id": name, "phase": "free_response"}
            phase_start = time.perf_counter()
            prediction = (
                model.reader.predict_with_receipt(observation, state, 128)
                if workspace is None
                else model.propose(observation, state)
            )
            prediction_seconds = time.perf_counter() - phase_start
            live["free_responses"] += 1
            payload = prediction if workspace is None else prediction.payload
            phase_start = time.perf_counter()
            summary = reader_summary(payload)
            row = {
                "index": index,
                "episode_id": name,
                "kind": episode["kind"],
                "observation": public,
                "before": before,
                "reader": summary,
                "forward": _forward(summary),
                "heads": None,
                "proposal": None,
                "execution": None,
                "grade": None,
                "phase_seconds": {"prediction": prediction_seconds, "actual_execution": 0.0},
            }
            row["phase_seconds"]["trace_serialization"] = time.perf_counter() - phase_start
            if workspace is not None:
                row["heads"] = _plain(prediction.choices)
                row["proposal"] = prediction.proposal.to_record()
                pending["phase"] = "public_execution"
                admit()
                phase_start = time.perf_counter()
                _, execution = execute_proposal(
                    workspace,
                    prediction.proposal,
                    output / "execution",
                    launch=launch,
                    deadline=started + SECONDS - FINAL_SECONDS,
                )
                row["phase_seconds"]["actual_execution"] = time.perf_counter() - phase_start
                live["executions"] += 1
                row["execution"] = execution
                row["grade"] = grade_execution(execution, episode["grade"], public["files"])
                row["correct"] = row["grade"]["task_success"]
                envelope = {
                    "attempt_id": prediction.attempt_id,
                    "proposal_sha256": prediction.proposal_sha256,
                    "workspace_sha256": workspace.sha256,
                    "execution": execution,
                }
                pending["phase"] = "disabled_resolution"
                phase_start = time.perf_counter()
                row["ack_work"] = model.finish_without_teaching(
                    prediction, observation, state, envelope, learning_enabled=False
                )
            else:
                row["correct"] = (
                    summary["answer_hex"] in episode["grade"]["accepted_answer_hex"]
                    and summary["stop_reason"] == "eos"
                )
                pending["phase"] = "disabled_resolution"
                phase_start = time.perf_counter()
                row["ack_work"] = model.reader.acknowledge(
                    prediction, state, learning_enabled=False
                )
            row["phase_seconds"]["disabled_resolution"] = time.perf_counter() - phase_start
            for name_phase, seconds in row["phase_seconds"].items():
                phases[name_phase + "_seconds"] = phases.get(name_phase + "_seconds", 0.0) + seconds
            model._require_idle()
            row["resolution"] = _plain(model.reader.last_controller_resolution)
            for key, count in _response_counts(row).items():
                live[key] += count
            row["after"] = numerical(model, state)
            for key in (
                "non_rng_numeric_sha256",
                "memory_sha256",
                "memory_used",
                "views_sha256",
                "self_wiring_sha256",
            ):
                if row["after"][key] != initial[key]:
                    raise ValueError(
                        "disabled preservation changed a retained learned/factual value"
                    )
            row["elapsed_seconds"] = time.perf_counter() - tick
            publish(f"case-{index}.json", row)
            completed.append(name)
            pending = None
            del prediction, payload, summary, row
        if numerical(unused, unused_state) != unused_initial:
            raise ValueError("unused independent initializer route changed")
        origins_after = runtime.check_origins(selection)
        _check_sources(sources)
        _check_sources(IMPLEMENTATION_SOURCES)
        if _sources() != sources or origins_after != origins_before:
            raise ValueError("preservation source closure changed during execution")
        admit(pending_write=False)
        final = numerical(model, state)
        report = {
            "schema": "integrated-local-learning-preservation-worker-v1",
            "status": "completed",
            "rule": args.rule,
            "cases": completed,
            "manifest_sha256": MANIFEST_SHA,
            "parent_bank_sha256": runtime.BANK_SHA,
            "parent_boundary_sha256": runtime.BOUNDARY_SHA,
            "implementation_revision": IMPLEMENTATION_REVISION,
            "initial": initial,
            "final": final,
            "metadata": metadata,
            "origins_after": origins_after,
            "work": work,
            "counts": dict(live),
            "unused_initial": unused_initial,
            "unused_final": numerical(unused, unused_state),
            "initialization_seconds": initialized["receipt"]["elapsed_seconds"],
            "phase_seconds": dict(phases),
            "timing_scope": {
                "initialization": "parent authentication, one decode, two model copies and hashes",
                "prediction": "includes its model hashing, forward work and trace construction",
                "actual_execution": "includes executor child and captured files/streams",
                "disabled_resolution": "fork, update proposal, validation, RNG publish and hashes",
                "numerical_hash": "explicit worker hashes only; excludes nested model hashes",
                "trace_serialization": "reader summary/forward projection; excludes JSON encoding",
                "serialization_publication": (
                    "JSON, SHA, disk checks, write/fsync; excludes final report/completion"
                ),
                "admission": "boundary checks only; overlaps before-edit case elapsed",
                "case_elapsed": "all case phases except JSON publication; overlaps listed phases",
                "total": "all phases; nested initializer/executor timers must not be added again",
            },
            "elapsed_seconds": time.perf_counter() - started,
            "budget": {
                "seconds": SECONDS,
                "final_reserve_seconds": FINAL_SECONDS,
                "rss_bytes": RSS_BYTES,
                "artifact_bytes": ARTIFACT_BYTES,
                "pending_bytes_inside_cap": PENDING_BYTES,
                "free_reserve_bytes": FREE_BYTES,
            },
            "resource_scope": "boundary RSS/admission; outer launcher enforces hard tree limits",
            "scope": "forward preservation; zero teachers, ingests or learning publication",
        }
        publish("report.json", report)
        # Include actual execution files as well as the complete response records.
        for name in _inventory(output):
            if name not in records:
                raw = _read(output / name)
                records[name] = {"bytes": len(raw), "sha256": _sha(raw)}
        admit(pending_write=False)
        _save(
            output,
            "completion.json",
            {
                "schema": "integrated-local-learning-preservation-completion-v1",
                "status": "completed",
                "rule": args.rule,
                "files": records,
                "elapsed_seconds": time.perf_counter() - started,
            },
        )
        return report
    except BaseException as exc:
        try:
            _save(
                output,
                "failure.json",
                {
                    "status": "failed",
                    "rule": args.rule,
                    "error": type(exc).__name__ + ": " + str(exc),
                    "completed": completed,
                    "pending": pending,
                    "counts": live,
                    "elapsed_seconds": time.perf_counter() - started,
                },
            )
        except BaseException as receipt_error:
            print(
                json.dumps(
                    {
                        "failure_receipt_error": type(receipt_error).__name__
                        + ": "
                        + str(receipt_error)
                    }
                ),
                file=sys.stderr,
            )
        raise


def _saved(directory, rule):
    directory = Path(directory).resolve()
    completion = json.loads(_read(directory / "completion.json", limit=1024**2))
    if (
        completion["schema"] != "integrated-local-learning-preservation-completion-v1"
        or completion["status"] != "completed"
        or completion["rule"] != rule
    ):
        raise ValueError("preservation worker did not complete its declared rule")
    files = completion["files"]
    if set(_inventory(directory)) != set(files) | {"completion.json"}:
        raise ValueError("preservation completion inventory changed")
    if sum(_inventory(directory).values()) > ARTIFACT_BYTES:
        raise ValueError("preservation worker exceeded its artifact cap")
    values = {}
    for name, entry in files.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or path.is_symlink():
            raise ValueError("preservation file escaped its completed output")
        raw = _read(path, entry["sha256"])
        if len(raw) != entry["bytes"]:
            raise ValueError("preservation completed file length changed")
        if name in {
            "report.json",
            "sources.json",
            "manifest.json",
            "initialization.json",
            "launch.json",
        }:
            values[name] = json.loads(raw)
    sources = values["sources.json"]
    _check_sources(IMPLEMENTATION_SOURCES)
    _check_sources(sources["source_sha256"])
    _check_sources(sources["selection"]["source_sha256"])
    if sources["selection"]["rule"] != rule:
        raise ValueError("saved source selection differs")
    selection = sources["selection"]
    if (
        selection["parent_bank_sha256"] != runtime.BANK_SHA
        or selection["parent_config_sha256"] != runtime.CONFIG_SHA
    ):
        raise ValueError("saved selection changed the retained parent")
    for module, old_hash in runtime.MODULES.items():
        relative = Path(selection["module_origins"][module]).resolve().relative_to(ROOT).as_posix()
        expected = (
            old_hash if rule == "predecessor" else IMPLEMENTATION_SOURCES[runtime._relative(module)]
        )
        if selection["source_sha256"].get(relative) != expected:
            raise ValueError("saved worker did not select its exact implementation")
    report = values["report.json"]
    if (
        report["status"] != "completed"
        or report["rule"] != rule
        or report["cases"] != list(CASES)
        or report["manifest_sha256"] != MANIFEST_SHA
        or report["parent_bank_sha256"] != runtime.BANK_SHA
        or report["parent_boundary_sha256"] != runtime.BOUNDARY_SHA
        or report["implementation_revision"] != IMPLEMENTATION_REVISION
        or _sha(_encode(values["manifest.json"])) != MANIFEST_SHA
        or report["elapsed_seconds"] >= SECONDS
        or completion["elapsed_seconds"] >= SECONDS
        or report["counts"]["rss_boundary_peak_bytes"] > RSS_BYTES
        or any(
            report["counts"][key] != count
            for key, count in {
                "free_responses": 4,
                "executions": 1,
                "teachers": 0,
                "ingests": 0,
                "learning_publications": 0,
            }.items()
        )
    ):
        raise ValueError("preservation worker contract differs")
    initialized = values["initialization.json"]
    if (
        initialized["status"] != "initialized"
        or initialized["rule"] != rule
        or initialized["parent_boundary_sha256"] != runtime.BOUNDARY_SHA
        or initialized["work"]["tensor_decodes"] != 1
        or initialized["work"]["model_constructions"] != 2
        or report["origins_after"] != sources["origins_before"]
        or report["unused_initial"] != report["unused_final"]
        or report["unused_initial"] != report["initial"]
    ):
        raise ValueError("retained initialization or inactive route parity differs")
    values["_directory"], values["_files"] = directory, files
    return values


def compare(predecessor_directory, current_directory, output):
    """Authenticate saved worker artifacts, then compare actual forward values exactly."""
    if DRAFT:
        raise RuntimeError("DRAFT preservation comparator requires root source/protocol release")
    _no_neuronal_imports()
    started = time.perf_counter()
    destination = Path(output).resolve()
    if destination.exists() or not destination.is_relative_to(ROOT):
        raise ValueError("comparison receipt must be a new checkout-local file")
    if any(
        destination.is_relative_to(Path(p).resolve())
        for p in (predecessor_directory, current_directory)
    ):
        raise ValueError("comparison receipt cannot mutate a completed worker directory")
    old, new = _saved(predecessor_directory, "predecessor"), _saved(current_directory, "current")
    from experiments.r4_integrated.integrated_local_learning_initialize_v1 import (
        _predecessor_metadata,
    )

    _no_neuronal_imports()

    before, after = old["report.json"], new["report.json"]
    metadata = copy.deepcopy(after["metadata"])
    _predecessor_metadata(metadata)
    # JSON represents metadata tuples as lists on both sides.
    if metadata != before["metadata"] or before["metadata"] == after["metadata"]:
        raise ValueError("paired metadata differs beyond the exact declared v2/v3 transition")
    if before["initial"] != after["initial"] or before["final"] != after["final"]:
        raise ValueError("paired retained numerical/RNG/factual state differs")
    cases = []
    counted = {"scan_offers": 0, "free_ticks": 0, "controller_draws": 0, "rng_publications": 0}
    previous = before["initial"]
    for index, name in enumerate(CASES):
        case_name = f"case-{index}.json"
        a, b = (
            json.loads(_read(saved["_directory"] / case_name, saved["_files"][case_name]["sha256"]))
            for saved in (old, new)
        )
        public = _observation_record(old["manifest.json"], name)
        for row in (a, b):
            if (
                row["index"] != index
                or row["episode_id"] != name
                or row["observation"] != public
                or row["forward"] != _forward(row["reader"])
                or row["before"] != previous
            ):
                raise ValueError("saved response differs from its exact case/forward record")
            if row["resolution"]["learning_enabled"] is not False:
                raise ValueError("preservation published a learning update")
            _response_counts(row)
            initial = before["initial"]
            for key in (
                "non_rng_numeric_sha256",
                "memory_sha256",
                "memory_used",
                "views_sha256",
                "self_wiring_sha256",
            ):
                if row["before"][key] != initial[key] or row["after"][key] != initial[key]:
                    raise ValueError("saved disabled response changed retained slow/factual state")
            if row["after"]["reader_experience"] != row["before"]["reader_experience"] + 1 or row[
                "after"
            ]["experience"] != row["before"]["experience"] + int(name == "A00-E"):
                raise ValueError(
                    "preservation counters do not describe exactly one actual response"
                )
        for key in (
            "kind",
            "observation",
            "before",
            "after",
            "forward",
            "heads",
            "proposal",
            "grade",
            "correct",
        ):
            if a[key] != b[key]:
                raise ValueError(f"retained forward mismatch at {name}: {key}")
        if name == "A00-E":
            if _execution_behavior(a["execution"]) != _execution_behavior(b["execution"]):
                raise ValueError("paired actual execution behavior differs")
        elif a["execution"] is not None or b["execution"] is not None:
            raise ValueError("reading case performed an undeclared execution")
        diagnostic_equal = (
            None
            if name != "A00-E"
            else {
                key: a["execution"][key] == b["execution"][key]
                for key in (
                    "stdout_hex",
                    "stderr_hex",
                    "stdout_observed_bytes",
                    "stderr_observed_bytes",
                )
            }
        )
        cases.append(
            {
                "episode_id": name,
                "forward_equal": True,
                "numeric_rng_equal": True,
                "complete_outcome_equal": True,
                "correct": a["correct"],
                "raw_execution_diagnostics_equal": diagnostic_equal,
            }
        )
        for key, count in _response_counts(a).items():
            counted[key] += count
        previous = a["after"]
        del a, b, row
    if previous != before["final"]:
        raise ValueError("saved preservation chain does not reach the final retained state")
    if any(
        before["counts"][key] != count or after["counts"][key] != count
        for key, count in counted.items()
    ):
        raise ValueError("saved scan/free/draw/RNG counts differ from actual responses")
    result = {
        "schema": "integrated-local-learning-preservation-comparison-v1",
        "status": "passed",
        "cases": cases,
        "actual_controller_counts_per_worker": counted,
        "metadata": "only exact declared v2/v3 dictionary replacements",
        "completion_sha256": {
            rule: _sha(_read(Path(path) / "completion.json"))
            for rule, path in (
                ("predecessor", predecessor_directory),
                ("current", current_directory),
            )
        },
        "elapsed_seconds": time.perf_counter() - started,
        "model_calls": 0,
        "tensor_decodes": 0,
        "scope": "actual retained forward/numeric preservation; no acquisition evidence",
    }
    # The comparison is a small sibling receipt; both256MiB worker allowances
    # remain inside the combined parent directory budget.
    if len(_encode(result)) > 1024**2:
        raise RuntimeError("preservation comparison receipt exceeds1MiB")
    _no_neuronal_imports()
    _save(destination.parent, destination.name, result, artifact_limit=2 * ARTIFACT_BYTES + 1024**2)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker")
    worker.add_argument("--rule", required=True, choices=("predecessor", "current"))
    worker.add_argument("--predecessor-snapshot", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    comparison = sub.add_parser("compare")
    comparison.add_argument("--predecessor", type=Path, required=True)
    comparison.add_argument("--current", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (
        run(args)
        if args.command == "worker"
        else compare(args.predecessor, args.current, args.output)
    )
    print(
        json.dumps(
            {"status": result["status"], "elapsed_seconds": result["elapsed_seconds"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
