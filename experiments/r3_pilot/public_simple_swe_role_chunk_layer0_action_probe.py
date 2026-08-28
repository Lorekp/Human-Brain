"""One-shot visible-TRAIN route-to-Layer-0 action probe.

The already licensed R3-SR0 ``concept_true`` checkpoint receives exactly one
read-only normalized prompt for scalar-nonnegative /009 and makes exactly one
no-argument native generation call.  The emitted surface route token alone
selects a fixed Layer-0 actuator.  Only after generation may the existing
literal compiler and canonical Simple-SWE TRAIN sandbox see the action and the
private grading cases, respectively.

Importing or compiling this module does not spend the target.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
PROBE_DIR = Path(__file__).resolve().parent
for import_root in (SRC_ROOT, REPO_ROOT, PROBE_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import public_simple_swe_role_chunk_route_probe as route_probe
from humanbrain.native import loaded_library_identity
from humanbrain.native_temporal_runtime import (
    NativeTemporalCircuit,
    NativeTemporalCircuitDefinition,
    TemporalStepMode,
)
from humanbrain.program import (
    Layer0ProgramCompiler,
    Layer0ProgramFormatError,
    Layer0ProgramGrammar,
    Layer0ProgramSignature,
)
from humanbrain.role_chunk_episode import RoleChunkEpisode, RolePayloadKind
from humanbrain.simple_swe.sandbox import (
    SandboxConfigurationError,
    execute_completion,
    load_worker_snapshot,
    python_runtime_identity,
    resolve_python_executable,
)
from humanbrain.simple_swe.tasks import GeneratedTask, PublicTask, generate_visible_partition


OUTPUT_PATH = Path(__file__).with_suffix(".json")
SPEND_MARKER_PATH = Path(__file__).with_name(
    f"{Path(__file__).stem}.t0_scalar_nonnegative-009.spent.json"
)
PARENT_ARTIFACT_PATH = Path(__file__).with_name(
    "public_simple_swe_role_chunk_route_probe.json"
)

TARGET_FAMILY = "t0_scalar_nonnegative"
TARGET_ORDINAL = 9
TARGET_TASK_ID = "simple-swe-v1/train/tier-0/t0_scalar_nonnegative/009"
TARGET_TASK_SHA256 = "987630286c577aa77caa6857d9ecf526f6ad42ca4eae3e5a48ca93f1f76ec44c"
TARGET_HIDDEN_TEST_SHA256 = (
    "23c38860c359c1a6cd4431de5dac799d00713c089da2840052cf0849997ebea2"
)
TARGET_STRUCTURAL_SHA256 = (
    "736854ba550d057d152c9701e65fe4a6d4528d4d0581db6454333e487bd3f899"
)
TARGET_CONCEPT_STREAM_SHA256 = (
    "c4fc0c6d9eb257e2655453fb9c644ce7d82465bfe54a512c028fcb5dba3c42c7"
)

EXPECTED_ROUTE_TOKEN_ID = 1536
EXPECTED_CHECKPOINT_SHA256 = (
    "fad5974f59d0b6ee527ace9cd7a817fb31d92f78ce68b0dce0515b90ca982549"
)
EXPECTED_PARENT_ARTIFACT_SHA256 = (
    "9ee0df214df3525a2e2898129b03295d9ad20706e26a7c52039db40c7a62884b"
)
ACTUATOR_BY_ROUTE_TOKEN = MappingProxyType(
    {EXPECTED_ROUTE_TOKEN_ID: ("MAX", "ZERO", "ARG0", "END")}
)

PINNED_SOURCE_SHA256 = {
    "experiments/r3_pilot/public_simple_swe_role_chunk_route_probe.py": "5b4c80b9fe3a8d780b5a286fb33ad69885316e8b09a2a2723f7b2a46db1c1001",
    "src/humanbrain/native.py": "8963db700914828a0d5212419431ff50d50089c750be259c23fea69087748a6f",
    "src/humanbrain/native_temporal_runtime.py": "77b41b014d35648adbe989fe3c31059856f60410758dddb3d9794e54a0ed227a",
    "src/humanbrain/program.py": "ca5c46aa04688954099d029cc94829f4b8ca1e0fddc651d4aff2bdbe5fb05ddd",
    "src/humanbrain/role_chunk_episode.py": "8dc8a9102569f6bdc3135479bd38f8074993efc2a93234202fcf4e96d65940f5",
    "src/humanbrain/concept_tokens.py": "9070deaf7d9a8a54add24e3a213d8b6aa7bbb24b3d35ad66a7b68b3d1148f43e",
    "src/humanbrain/tokenizer.py": "08ef114a758006868944ad3091ca66619d3c082086ea2d44d93211d82a9791ce",
    "src/humanbrain/cortex.py": "7a7704a14324203d2cce427a11fbd58ffbbc345185ba34633449f59cf294624d",
    "src/humanbrain/sdr_union.py": "1de7f6edd914584c4c72d7e42426254590049cfb69cbbfa730afa3179afa63f9",
    "src/humanbrain/simple_swe/protocol.py": "22993997004a156b7d94b8d79198deae3e2f71f04d1aebc9740b342c0aa94849",
    "src/humanbrain/simple_swe/tasks.py": "d247bb1a81dca47846a787292a8a3f29c01938771070517b6fa35c9d8f7eed23",
    "src/humanbrain/simple_swe/sandbox.py": "2da8083bb48638bb0119b7e6a552a2134b915e07f2eecd1d69fd2dbf7d365b35",
    "src/humanbrain/simple_swe/_worker.py": "258aaf45decb8ed6635a85b92ee970591932b2cf53b691149a0d66777601c870",
    "src/humanbrain/_native_src/src/temporal_runtime.rs": "012d4f79505e3e0e381ebf9397db0c74d695120e7dfe8d511a338f8b896b9dc4",
    "src/humanbrain/_native_src/src/temporal_runtime/checkpoint.rs": "bdb1ab444a826e706b9c8284816a6e54847b0bb6c156723135324fffb377faba",
    "benchmarks/simple_swe_protocol_v1.json": "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-executable", type=Path, default=None)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _display_path(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _assert_parent_license() -> dict[str, object]:
    if _sha256_file(PARENT_ARTIFACT_PATH) != EXPECTED_PARENT_ARTIFACT_SHA256:
        raise RuntimeError("licensed R3-SR0 parent artifact changed")
    payload = json.loads(PARENT_ARTIFACT_PATH.read_text(encoding="utf-8"))
    if payload.get("outcome") != "LICENSE_CONCEPT_REPETITION_CONDITIONED_ROUTE":
        raise RuntimeError("parent artifact did not license concept routing")
    gate = payload.get("gate")
    train = payload.get("train")
    if not isinstance(gate, dict) or gate.get("selected_representation") != "concept":
        raise RuntimeError("parent artifact selected a different representation")
    if (
        not isinstance(train, dict)
        or not isinstance(train.get("concept_true"), dict)
        or train["concept_true"].get("checkpoint_sha256")
        != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("parent artifact checkpoint identity changed")
    return payload


def _assert_frozen_sources() -> None:
    for source, expected in PINNED_SOURCE_SHA256.items():
        if _sha256_file(REPO_ROOT / source) != expected:
            raise RuntimeError(f"frozen Layer-0 action dependency changed: {source}")


def _assert_committed_and_unspent() -> None:
    if OUTPUT_PATH.exists() or SPEND_MARKER_PATH.exists():
        raise RuntimeError("refusing spent role-route Layer-0 action probe")
    _assert_parent_license()
    _assert_frozen_sources()
    paths = (
        Path(__file__).resolve(),
        PARENT_ARTIFACT_PATH,
        *(REPO_ROOT / source for source in PINNED_SOURCE_SHA256),
    )
    for path in paths:
        relative = path.resolve().relative_to(REPO_ROOT).as_posix()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all", "--", relative],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        if status.stdout:
            raise RuntimeError(f"Layer-0 action source must be committed: {relative}")


def _durable_marker_inventory() -> list[dict[str, str]]:
    inventory: list[dict[str, str]] = []
    for path in sorted(PROBE_DIR.glob("*.spent.json")):
        text = path.read_text(encoding="utf-8")
        if TARGET_TASK_ID in text:
            raise RuntimeError(f"target already appears in durable marker: {path}")
        inventory.append({"path": _display_path(path), "sha256": _sha256_file(path)})
    return inventory


def _select_target(protocol: object) -> GeneratedTask:
    matches = tuple(
        task
        for task in generate_visible_partition(protocol, "train")
        if task.public.family == TARGET_FAMILY and task.public.ordinal == TARGET_ORDINAL
    )
    if len(matches) != 1:
        raise RuntimeError("target selection is not unique")
    target = matches[0]
    public = target.public
    if (
        public.task_id != TARGET_TASK_ID
        or public.partition != "train"
        or public.task_sha256 != TARGET_TASK_SHA256
        or public.hidden_test_sha256 != TARGET_HIDDEN_TEST_SHA256
    ):
        raise RuntimeError("public /009 target identity changed")
    return target


def _prepare_probe() -> tuple[
    object,
    GeneratedTask,
    RoleChunkEpisode,
    NativeTemporalCircuitDefinition,
    bytes,
    object,
    dict[str, object],
    dict[str, int],
]:
    protocol, donors, _parent_probe_records = route_probe._select_tasks()
    target = _select_target(protocol)
    encoder, concept_streams, base_streams, _records = (
        route_probe._prepare_representations(donors, (target,))
    )
    definition, capacity = route_probe._definition(
        donors, (target,), concept_streams, base_streams
    )
    with route_probe._public_train_authority():
        checkpoint, identity, train_record = route_probe._train_arm(
            "concept_true", definition, donors, concept_streams, base_streams
        )
    checkpoint_sha256 = _sha256_file_bytes(checkpoint)
    if (
        checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256
        or train_record.get("checkpoint_sha256") != EXPECTED_CHECKPOINT_SHA256
    ):
        raise RuntimeError("rebuilt concept_true checkpoint differs from license")

    episode = encoder.encode(target.public)
    if (
        episode.structural_sha256 != TARGET_STRUCTURAL_SHA256
        or _sha256_json(tuple(episode.token_ids)) != TARGET_CONCEPT_STREAM_SHA256
        or tuple(episode.token_ids)
        != concept_streams[(TARGET_FAMILY, route_probe.DONOR_ORDINALS[0])]
    ):
        raise RuntimeError("/009 role-chunk representation differs from licensed template")
    if set(episode.token_ids) & set(route_probe.ROUTE_TOKEN_IDS):
        raise RuntimeError("/009 prompt entered the private route-token band")
    return (
        protocol,
        target,
        episode,
        definition,
        checkpoint,
        identity,
        train_record,
        capacity,
    )


def _sha256_file_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _layer0_signature(episode: RoleChunkEpisode, public: PublicTask) -> Layer0ProgramSignature:
    if type(episode) is not RoleChunkEpisode or type(public) is not PublicTask:
        raise TypeError("Layer-0 action requires exact role episode and PublicTask values")
    if episode.task_id != public.task_id or episode.restore_prompt() != public.prompt:
        raise ValueError("role episode does not match its public task")
    if public.partition != "train" or len(public.signature.parameters) != 1:
        raise ValueError("Layer-0 action requires one public TRAIN parameter")
    parameter = public.signature.parameters[0]
    if parameter.type.constructor != "Int" or public.signature.result.constructor != "Int":
        raise ValueError("Layer-0 action requires a scalar Int -> Int signature")

    entries = episode.payloads_for(RolePayloadKind.ENTRY_POINT)
    parameters = episode.payloads_for(RolePayloadKind.PARAMETER)
    numeric = (
        *episode.payloads_for(RolePayloadKind.INTEGER_LITERAL),
        *episode.payloads_for(RolePayloadKind.FLOAT_LITERAL),
    )
    if len(entries) != 1 or len(parameters) != 1 or numeric:
        raise ValueError("scalar-nonnegative sidecar shape changed")
    if entries[0].value != public.signature.entry_point or parameters[0].value != parameter.name:
        raise ValueError("role sidecar disagrees with the public signature")
    return Layer0ProgramSignature(
        entry_point=entries[0].value,
        parameter_name=parameters[0].value,
    )


def _dispatch_action(surface_route_token_id: int | None) -> tuple[str, ...] | None:
    if surface_route_token_id is None:
        return None
    if type(surface_route_token_id) is not int:
        raise TypeError("surface route token must be an exact integer")
    return ACTUATOR_BY_ROUTE_TOKEN.get(surface_route_token_id)


def _compile_action(
    action: tuple[str, ...], episode: RoleChunkEpisode, public: PublicTask
) -> dict[str, object]:
    if action not in ACTUATOR_BY_ROUTE_TOKEN.values():
        raise ValueError("action is outside the fixed route actuator table")
    signature = _layer0_signature(episode, public)
    compiler = Layer0ProgramCompiler(Layer0ProgramGrammar(max_nodes=3))
    source = compiler.compile(action, signature)
    expected = (
        f"def {signature.entry_point}({signature.parameter_name}):\n"
        f"    return max(0, {signature.parameter_name})\n"
    )
    if source != expected:
        raise RuntimeError("existing Layer-0 compiler emitted unexpected source")
    return {
        "action_tokens": list(action),
        "entry_point_from_sidecar": signature.entry_point,
        "parameter_from_sidecar": signature.parameter_name,
        "compiler": "humanbrain.program.Layer0ProgramCompiler",
        "max_nodes": 3,
        "source": source,
        "source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        "complete_function_exact": True,
        "compiler_received_prompt": False,
        "compiler_received_private_cases": False,
    }


def _create_spend_marker(target: PublicTask) -> dict[str, object]:
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-layer0-action",
        "state": "TARGET_ATTEMPT_SPENT",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "target_task_id": target.task_id,
        "family": target.family,
        "ordinal": target.ordinal,
        "representation": "concept_true",
        "checkpoint_sha256": EXPECTED_CHECKPOINT_SHA256,
        "parent_artifact_sha256": EXPECTED_PARENT_ARTIFACT_SHA256,
        "atomic_no_argument_generation_calls_planned": 1,
        "model_retries": 0,
        "harness_retries": 0,
        "probe_source_sha256": _sha256_file(Path(__file__)),
        "canonical_artifact_path": _display_path(OUTPUT_PATH),
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with SPEND_MARKER_PATH.open("x", encoding="utf-8", newline="\n") as marker:
            marker.write(serialized)
            marker.flush()
            os.fsync(marker.fileno())
    except FileExistsError as error:
        raise RuntimeError("refusing already spent Layer-0 action target") from error
    return {
        "path": _display_path(SPEND_MARKER_PATH),
        "sha256": _sha256_file(SPEND_MARKER_PATH),
        "exclusive_create": True,
        "flushed_and_fsynced_before_single_generate_call": True,
    }


def _sandbox_record(
    source: str,
    target: GeneratedTask,
    protocol: object,
    *,
    python_executable: Path,
    runtime_identity: dict[str, str],
    worker_source: bytes,
    worker_sha256: str,
) -> dict[str, object]:
    try:
        result = execute_completion(
            source,
            target.public,
            target.private,
            protocol.execution,
            python_executable=python_executable,
            python_executable_sha256=runtime_identity["executable_sha256"],
            worker_source=worker_source,
            worker_sha256=worker_sha256,
        )
    except SandboxConfigurationError as error:
        return {
            "attempted": True,
            "passed": False,
            "sandboxed": False,
            "outcome": "harness_error",
            "detail": str(error),
        }
    return {
        "attempted": True,
        **asdict(result),
        "regression_case_count": len(target.private.regression_cases),
        "property_case_count": len(target.private.property_cases),
        "all_private_cases_held_post_generation": True,
        "worker_sha256": worker_sha256,
        "python_runtime": runtime_identity,
    }


def main() -> int:
    args = _parse_args()
    _assert_committed_and_unspent()
    marker_inventory = _durable_marker_inventory()
    started_ns = time.perf_counter_ns()
    parent_artifact = _assert_parent_license()
    (
        protocol,
        target,
        episode,
        definition,
        checkpoint,
        checkpoint_identity,
        train_record,
        capacity,
    ) = _prepare_probe()

    executable = resolve_python_executable(args.python_executable)
    runtime_identity = python_runtime_identity(executable, protocol.raw.get("language"))
    if runtime_identity.get("python_version") != "3.11.15":
        raise RuntimeError("Layer-0 action probe requires repo Python 3.11.15")
    worker_path = SRC_ROOT / "humanbrain" / "simple_swe" / "_worker.py"
    worker_sha256 = _sha256_file(worker_path)
    worker_source = load_worker_snapshot(worker_sha256)

    with route_probe._public_train_authority():
        with NativeTemporalCircuit.from_checkpoint(definition, checkpoint) as circuit:
            if circuit.checkpoint_identity() != checkpoint_identity:
                raise RuntimeError("restored concept_true checkpoint differs")
            prompt_steps = circuit.advance_block(
                tuple(episode.token_ids),
                mode=TemporalStepMode.READ_ONLY,
                reset_before=True,
            )
            marker = _create_spend_marker(target.public)
            raw_generation = route_probe._generate_once(circuit)

    generation = raw_generation["generation"]
    observed_route = (
        None if generation is None else int(generation["surface_token_id"])
    )
    action = _dispatch_action(observed_route)
    action_dispatch_count = int(action is not None)
    scored_generation = route_probe._score_generation(
        raw_generation, EXPECTED_ROUTE_TOKEN_ID
    )

    compile_record: dict[str, object] = {
        "attempted": False,
        "complete_function_exact": False,
        "error": "no actuator matched the emitted route token",
    }
    if action is not None:
        try:
            compile_record = {
                "attempted": True,
                **_compile_action(action, episode, target.public),
                "error": None,
            }
        except (TypeError, ValueError, RuntimeError, Layer0ProgramFormatError) as error:
            compile_record = {
                "attempted": True,
                "complete_function_exact": False,
                "error": f"{type(error).__name__}: {error}",
            }

    sandbox: dict[str, object] = {
        "attempted": False,
        "passed": False,
        "sandboxed": False,
        "outcome": None,
        "detail": "not run because no complete Layer-0 function was compiled",
    }
    source = compile_record.get("source")
    if compile_record.get("complete_function_exact") is True and isinstance(source, str):
        sandbox = _sandbox_record(
            source,
            target,
            protocol,
            python_executable=executable,
            runtime_identity=runtime_identity,
            worker_source=worker_source,
            worker_sha256=worker_sha256,
        )

    gates = {
        "exact_physical_route": bool(
            scored_generation["physical_route_election_valid"]
            and observed_route == EXPECTED_ROUTE_TOKEN_ID
        ),
        "exactly_one_action": action_dispatch_count == 1
        and action == ACTUATOR_BY_ROUTE_TOKEN[EXPECTED_ROUTE_TOKEN_ID],
        "complete_layer0_compile": compile_record.get("complete_function_exact") is True,
        "canonical_train_sandbox_all_cases": bool(
            sandbox.get("passed") is True
            and sandbox.get("sandboxed") is True
            and sandbox.get("outcome") == "passed"
        ),
    }
    passed = all(gates.values())
    outcome = "PASS_ROLE_ROUTE_LAYER0_ACTION" if passed else "STOP_ROLE_ROUTE_LAYER0_ACTION"
    native_identity = loaded_library_identity()
    payload = {
        "schema_version": 1,
        "experiment": "r3-public-simple-swe-role-chunk-layer0-action",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "outcome": outcome,
        "claim_boundary": (
            "One visible-TRAIN /009 normalized-template replay and one fixed scalar Layer-0 actuator. "
            "The prompt was historically disclosed to a prompt-only baseline, but no durable native/role-route marker had spent it. "
            "This is not paraphrase transfer, structural generalization, DEV/eval, held-out, M0, C1, or broad SWE capability."
        ),
        "authority": {
            "partition": "train",
            "target_task_id": target.public.task_id,
            "read_only_prompt_presentations": 1,
            "atomic_no_argument_generation_calls": 1,
            "model_retries": 0,
            "harness_retries": 0,
            "expected_route_used_as_runtime_input": False,
            "private_cases_entered_cell_runtime": False,
            "private_cases_used_only_after_generation_for_sandbox_grading": True,
            "preexisting_durable_marker_inventory": marker_inventory,
        },
        "parent_license": {
            "artifact_path": _display_path(PARENT_ARTIFACT_PATH),
            "artifact_sha256": EXPECTED_PARENT_ARTIFACT_SHA256,
            "outcome": parent_artifact["outcome"],
            "selected_representation": parent_artifact["gate"]["selected_representation"],
            "rebuilt_concept_true_checkpoint_sha256": _sha256_file_bytes(checkpoint),
        },
        "target": {
            "task_id": target.public.task_id,
            "task_sha256": target.public.task_sha256,
            "hidden_test_sha256": target.public.hidden_test_sha256,
            "structural_sha256": episode.structural_sha256,
            "concept_stream_sha256": _sha256_json(tuple(episode.token_ids)),
            "concept_prompt_events": len(episode.token_ids),
            "sidecar_payload_kinds": [payload.kind.value for payload in episode.payloads],
        },
        "physical_definition": {
            "circuit_identity_sha256": definition.circuit_identity_sha256(),
            "column_count": definition.config.column_count,
            "active_columns": route_probe.ACTIVE_COLUMNS,
            "theta_match": definition.config.theta_match,
            "theta_active": definition.config.theta_active,
            **capacity,
        },
        "train": train_record,
        "prompt": route_probe._step_summary(prompt_steps),
        "spend_marker": marker,
        "generation": scored_generation,
        "actuator": {
            "mapping": {
                str(route): list(tokens)
                for route, tokens in ACTUATOR_BY_ROUTE_TOKEN.items()
            },
            "observed_surface_route_token_id": observed_route,
            "dispatch_count": action_dispatch_count,
            "dispatched_action": None if action is None else list(action),
            "dispatch_input_was_only_observed_surface_route_token": True,
        },
        "compile": compile_record,
        "sandbox": sandbox,
        "gates": gates,
        "runtime": {
            "elapsed_wall_ns": time.perf_counter_ns() - started_ns,
            "python": sys.version,
            "platform": platform.platform(),
            "native_source_digest": native_identity.source_digest,
            "native_binary_sha256": native_identity.binary_sha256,
            "probe_source_sha256": _sha256_file(Path(__file__)),
            "pinned_source_sha256": {
                source: _sha256_file(REPO_ROOT / source)
                for source in PINNED_SOURCE_SHA256
            },
        },
    }
    with OUTPUT_PATH.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({"outcome": outcome, "gates": gates}, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
