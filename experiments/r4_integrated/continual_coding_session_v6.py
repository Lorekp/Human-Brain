"""Prepared V6 recurrent-assembly continual coding session; import is inert.

The fresh public-TRAIN V6 stream drives three independently grown learners from
one identical public bootstrap corpus: a plastic treatment, a no-update control,
and a no-update STACK_OFF causal arm.  Evaluator-private policy and tests open
only after a learner has sealed one generic ``write_source`` action.  Importing
this module, and its schema preflight, never grow or invoke a model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

from experiments.r4_integrated.continual_toy_swe_tasks_v6 import (
    MAX_ATTEMPTS,
    TASK_COUNT,
    TRAIN_STREAM_CUSTODY_SHA256,
    TRAIN_STREAM_SHA256,
    ContinualToySweTrainTaskV6,
    continual_toy_swe_train_stream_v6,
    train_stream_custody_v6,
)
from humanbrain import persistent_coding as coding

SCHEMA = "continual-coding-session-v6"
RESERVATION_SCHEMA = "continual-coding-session-reservation-v6"
WRITE_TOOL = "write_source"
PUBLIC_PRIOR_CASES = 8

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = REPO_ROOT / "benchmarks/r4_integrated_continual_coding_session_v6.json"
PROTOCOL_PATH = REPO_ROOT / "benchmarks/simple_swe_protocol_v1.json"
WORKER_PATH = REPO_ROOT / "src/humanbrain/simple_swe/_worker.py"
STACK_GATE_PATH = REPO_ROOT / "benchmarks/r4_recurrent_assembly_stack_gate_v1.json"

V6_TASK_SOURCE_SHA256 = (
    "34a19e89e8893f2a496cda422e7e2d5693c7d65cb68a4ebc1a0699e8b9049e24"
)
STACK_GATE_SHA256 = (
    "11dbbbd205c2ba1f59d8c2369a1190f5dc6f232205e425f3be07c951c579912a"
)

_TASK_SOURCE = REPO_ROOT / "experiments/r4_integrated/continual_toy_swe_tasks_v6.py"
_ASSEMBLY_SOURCE = REPO_ROOT / "src/humanbrain/assembly_factorized_pointer.py"
_PERSISTENT_SOURCE = REPO_ROOT / "src/humanbrain/persistent_coding.py"
_FRAGMENT_SOURCE = REPO_ROOT / "src/humanbrain/factorized_fragment_memory.py"
_STACK_SOURCE = REPO_ROOT / "src/humanbrain/recurrent_assembly_stack.py"

_CUSTODY_PATHS = {
    "runner_source_sha256": Path(__file__),
    "task_stream_source_sha256": _TASK_SOURCE,
    "public_prior_source_sha256": (
        REPO_ROOT / "experiments/r4_integrated/public_coding_prior_v1.py"
    ),
    "assembly_pointer_source_sha256": _ASSEMBLY_SOURCE,
    "fragment_memory_source_sha256": _FRAGMENT_SOURCE,
    "recurrent_stack_source_sha256": _STACK_SOURCE,
    "persistent_coding_source_sha256": _PERSISTENT_SOURCE,
    "pointer_binding_source_sha256": REPO_ROOT / "src/humanbrain/pointer_binding.py",
    "tokenizer_source_sha256": REPO_ROOT / "src/humanbrain/tokenizer.py",
    "cortex_source_sha256": REPO_ROOT / "src/humanbrain/cortex.py",
    "cells_source_sha256": REPO_ROOT / "src/humanbrain/cells.py",
    "typed_program_types_source_sha256": (
        REPO_ROOT / "src/humanbrain/typed_program_types.py"
    ),
    "simple_swe_tasks_source_sha256": REPO_ROOT / "src/humanbrain/simple_swe/tasks.py",
    "simple_swe_protocol_source_sha256": (
        REPO_ROOT / "src/humanbrain/simple_swe/protocol.py"
    ),
    "simple_swe_sandbox_source_sha256": (
        REPO_ROOT / "src/humanbrain/simple_swe/sandbox.py"
    ),
    "worker_source_sha256": WORKER_PATH,
    "protocol_sha256": PROTOCOL_PATH,
    "stack_gate_artifact_sha256": STACK_GATE_PATH,
    "public_prior_loader_source_sha256": (
        REPO_ROOT / "experiments/r3_pilot/fresh_coding_checkpoint.py"
    ),
    "public_prior_corpus_sha256": (
        REPO_ROOT / "experiments/r3_pilot/fresh_coding_generic_corpus.json"
    ),
    "public_prior_manifest_sha256": (
        REPO_ROOT / "experiments/r3_pilot/fresh_coding_checkpoint.manifest.json"
    ),
}

_EXPECTED_DEPENDENCY_SHA256 = {
    "task_stream_source_sha256": (
        "34a19e89e8893f2a496cda422e7e2d5693c7d65cb68a4ebc1a0699e8b9049e24"
    ),
    "public_prior_source_sha256": (
        "95b0bbae62cc1fc08dd94e0f25e32116c2bbe671c7d5ceb10b50a4617ff0ac5b"
    ),
    "assembly_pointer_source_sha256": (
        "b27c67724af115e520fd9d8663ae0cf2b74b45f6a883d560c0f2f25e25e2537e"
    ),
    "fragment_memory_source_sha256": (
        "5e47aba9747cec7e87212047020d9beb94a7550b2ad4a9a0b6dc5a424ec3e18c"
    ),
    "recurrent_stack_source_sha256": (
        "2d49700ab899a4ce817a0b98c18de3133dc682c5306e516ef87a82c8895c9a6c"
    ),
    "persistent_coding_source_sha256": (
        "43aca8cbba287b073f5658951de6402f760b23c691929e487c3e6cb3b4d1138b"
    ),
    "pointer_binding_source_sha256": (
        "0cf13bd13be981c27d1eb00c1feb3fa08039c543654b17b02fa1b0e07e38171d"
    ),
    "tokenizer_source_sha256": (
        "08ef114a758006868944ad3091ca66619d3c082086ea2d44d93211d82a9791ce"
    ),
    "cortex_source_sha256": (
        "9cf37fc36978c3389a86a2856401f6c528d1a95f491dab0eb1b8662edcd055db"
    ),
    "cells_source_sha256": (
        "5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6"
    ),
    "typed_program_types_source_sha256": (
        "5f06a43bc771a8d61881d7ebf7c7decefcadbe245d77ce10640dbfff5174d099"
    ),
    "simple_swe_tasks_source_sha256": (
        "b0edfc2eedc25335bcb2318a7b13026f6ab7affc29643af76e16f4c6b57240c3"
    ),
    "simple_swe_protocol_source_sha256": (
        "22993997004a156b7d94b8d79198deae3e2f71f04d1aebc9740b342c0aa94849"
    ),
    "simple_swe_sandbox_source_sha256": (
        "2da8083bb48638bb0119b7e6a552a2134b915e07f2eecd1d69fd2dbf7d365b35"
    ),
    "worker_source_sha256": (
        "258aaf45decb8ed6635a85b92ee970591932b2cf53b691149a0d66777601c870"
    ),
    "protocol_sha256": (
        "f7bc723457bdb733bb560feb7e7f509b39cb756b2eec130e7ead6168e0bcb2e2"
    ),
    "stack_gate_artifact_sha256": STACK_GATE_SHA256,
    "public_prior_loader_source_sha256": (
        "40492212e78f32baf976291f32ead91c48d8e83568faee0e39d7675bddbe815f"
    ),
    "public_prior_corpus_sha256": (
        "b35a2edeabfb9e78755c4eb7bf3c77babacdddeaf2c0b4164af0c934e54ea56c"
    ),
    "public_prior_manifest_sha256": (
        "0942545cd03e21ac94c047db90c15e59b9327c5171db74afd3963d3cbeef7362"
    ),
}

_PRIVATE_FIELDS = frozenset(
    {
        "attempt_budget",
        "capability_group",
        "expected_action",
        "feedback_pressure",
        "generated_task",
        "private_environment_sha256",
        "private_test_sha256",
        "property_cases",
        "regression_cases",
        "retention_anchor_ordinal",
        "stage",
        "test_sha256",
    }
)
_STOP = re.compile(r"<stop>([^<>]+)</stop>", re.DOTALL)


def _json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(domain: str, value: object) -> str:
    payload = domain.encode("ascii") + b"\0" + _json(value).encode("utf-8")
    return sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _text_sha256(value: str | None) -> str | None:
    return None if value is None else sha256(value.encode("utf-8")).hexdigest()


def _audit_value(value: object) -> object:
    """Convert dataclass custody to canonical JSON without losing float bits."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, float):
        return {"float_hex": value.hex()}
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return {"bytes_sha256": sha256(value).hexdigest(), "length": len(value)}
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _audit_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        return {str(key): _audit_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_audit_value(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted((_audit_value(item) for item in value), key=_json)
    if value is None or isinstance(value, (str, int, bool)):
        return value
    raise TypeError(f"unsupported custody value: {type(value).__name__}")


def _assert_fixture_custody() -> None:
    if _sha256_file(_TASK_SOURCE) != V6_TASK_SOURCE_SHA256:
        raise RuntimeError("frozen V6 task source changed")
    if TRAIN_STREAM_SHA256 != (
        "4291faf1c0035dbff79b40250ab8cfd0e718d0a610ffd73684481a93fed60303"
    ):
        raise RuntimeError("frozen V6 stream root changed")
    if TRAIN_STREAM_CUSTODY_SHA256 != (
        "e19d14234b9d3bef90927e176590ede49f17c8280208712ee0a455b743cf672b"
    ):
        raise RuntimeError("frozen V6 custody root changed")
    if TASK_COUNT != 16 or MAX_ATTEMPTS != 3:
        raise RuntimeError("V6 count or uniform attempt policy changed")


def _assert_dependency_custody() -> None:
    dependency_paths = {
        name: path
        for name, path in _CUSTODY_PATHS.items()
        if name != "runner_source_sha256"
    }
    if set(dependency_paths) != set(_EXPECTED_DEPENDENCY_SHA256):
        raise RuntimeError("V6 dependency custody manifest is incomplete")
    for name, path in dependency_paths.items():
        if _sha256_file(path) != _EXPECTED_DEPENDENCY_SHA256[name]:
            raise RuntimeError(f"V6 dependency custody changed: {name}")


def _assert_projection_launch_ready() -> dict[str, object]:
    """Refuse the known incomplete projection/pending seam before model growth."""

    if _sha256_file(STACK_GATE_PATH) != STACK_GATE_SHA256:
        raise RuntimeError("immutable recurrent-assembly mechanism artifact changed")
    assembly_source = _ASSEMBLY_SOURCE.read_text(encoding="utf-8")
    fragment_source = _FRAGMENT_SOURCE.read_text(encoding="utf-8")
    persistent_source = _PERSISTENT_SOURCE.read_text(encoding="utf-8")
    stack_source = _STACK_SOURCE.read_text(encoding="utf-8")
    required_assembly = (
        "pointer_prompt = pointerize(prompt, inventory)",
        "FactorizedFragmentMemory",
        "if _EXPLICIT_STOP.fullmatch(raw_output):",
        "def resolve_pending_neutral(",
    )
    if any(marker not in assembly_source for marker in required_assembly):
        raise RuntimeError("full authenticated prompt projection has not landed")
    if (
        "if not explicit_neuronal_stop:" not in persistent_source
        or "feedback_text=_generation_failure_feedback(completion_trace)"
        not in persistent_source
    ):
        raise RuntimeError("generic host-rejected-output consequence has not landed")
    required_fragment = (
        "def _bounded_competition_drives(",
        "def _build_factor_refinements(",
        "self._factor_refinements",
    )
    if any(marker not in fragment_source for marker in required_fragment):
        raise RuntimeError("bounded fragment competition is not launch-ready")
    required_stack = (
        "def _respond_projection(",
        'phase="factor-relay"',
        'phase="factor-recall"',
        "ablation is AssemblyAblation.NONE",
        "candidate.signed_drive > 0.0",
        "factor_state[slot.factor_cell] = chosen.cohort_cell",
    )
    if any(marker not in stack_source for marker in required_stack):
        raise RuntimeError("fragment-to-recurrent projection seam has not landed")

    broken_stop_invariant = re.compile(
        r"stopped\s+and self\.learning_applied\s+and "
        r"self\.completion_trace\.materialization_error is None"
    )
    if broken_stop_invariant.search(persistent_source):
        raise RuntimeError(
            "EpisodeResult still rejects a learned generic host StopDecision"
        )

    from humanbrain.assembly_factorized_pointer import (
        AssemblyFactorizedPointerConfig,
    )

    config = AssemblyFactorizedPointerConfig()
    if (
        not hasattr(config, "fragments")
        or config.stack.max_cue_tokens < 4_096
        or config.stack.max_output_tokens < 512
    ):
        raise RuntimeError("projected prompt/output bounds are not V6-ready")
    return {
        "assembly_pointer_source_sha256": _sha256_file(_ASSEMBLY_SOURCE),
        "fragment_memory_source_sha256": _sha256_file(_FRAGMENT_SOURCE),
        "persistent_coding_source_sha256": _sha256_file(_PERSISTENT_SOURCE),
        "recurrent_stack_source_sha256": _sha256_file(_STACK_SOURCE),
        "full_prompt_projection_present": True,
        "explicit_stop_neutral_closure_present": True,
        "bounded_fragment_competition_present": True,
        "factor_refinement_relays_present": True,
        "factor_recall_race_present": True,
        "generic_host_rejection_callback_present": True,
        "stopped_learning_invariant_generalized": True,
        "default_config": _audit_value(config),
    }


@dataclass(frozen=True, slots=True)
class _SandboxRuntime:
    protocol: object
    executable: Path
    executable_sha256: str
    worker_source: bytes
    worker_sha256: str


def _sandbox_runtime() -> _SandboxRuntime:
    from humanbrain.simple_swe.protocol import load_protocol
    from humanbrain.simple_swe.sandbox import (
        load_worker_snapshot,
        python_runtime_identity,
        resolve_python_executable,
    )

    protocol = load_protocol(PROTOCOL_PATH)
    executable = resolve_python_executable()
    identity = python_runtime_identity(executable, protocol.raw["language"])
    worker_sha256 = _sha256_file(WORKER_PATH)
    return _SandboxRuntime(
        protocol=protocol,
        executable=executable,
        executable_sha256=identity["executable_sha256"],
        worker_source=load_worker_snapshot(worker_sha256),
        worker_sha256=worker_sha256,
    )


_WRITE_AFFORDANCE = coding.ToolAffordance.seal(
    WRITE_TOOL,
    {
        "additionalProperties": False,
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
        "type": "object",
    },
)


def _generic_receipt_record(receipt: coding.ExecutionReceipt) -> dict[str, object]:
    return {
        "accepted": receipt.accepted,
        "observations": [
            {"kind": block.kind, "text": block.text}
            for block in receipt.observations
        ],
        "receipt_sha256": receipt.sha256,
        "succeeded": receipt.succeeded,
    }


def _frame_blocks(
    *,
    starter_source: str,
    current_source: str,
    signature: dict[str, object],
    public_prompt: str,
    task_commitment_sha256: str | None = None,
    relative_path: str | None = None,
    previous: coding.ExecutionReceipt | None = None,
) -> tuple[coding.ObservationBlock, ...]:
    blocks = [
        coding.ObservationBlock.seal("starter-source", starter_source),
        coding.ObservationBlock.seal("current-source", current_source),
    ]
    if task_commitment_sha256 is not None:
        blocks.append(
            coding.ObservationBlock.seal(
                "task-commitment",
                task_commitment_sha256,
            )
        )
    if relative_path is not None:
        blocks.append(coding.ObservationBlock.seal("repository-path", relative_path))
    blocks.append(coding.ObservationBlock.seal("typed-signature", _json(signature)))
    if previous is not None:
        blocks.append(
            coding.ObservationBlock.seal(
                "previous-receipt",
                _json(_generic_receipt_record(previous)),
            )
        )
    blocks.append(coding.ObservationBlock.seal("public-task", public_prompt))
    return tuple(blocks)


def _training_frame(
    starter_source: str,
    entry_point: str,
    public_prompt: str,
) -> coding.ObservationFrame:
    return coding.ObservationFrame.seal(
        _frame_blocks(
            starter_source=starter_source,
            current_source=starter_source,
            signature={"entry_point": entry_point},
            public_prompt=public_prompt,
        ),
        (_WRITE_AFFORDANCE,),
    )


def _bootstrap_examples(example_type: type) -> tuple[object, ...]:
    from experiments.r4_integrated.public_coding_prior_v1 import (
        public_coding_prior_v1,
    )
    from humanbrain.pointer_binding import python_interface_inventory

    prior = public_coding_prior_v1()
    if len(prior) != PUBLIC_PRIOR_CASES:
        raise RuntimeError("public coding prior must contain exactly eight cases")
    examples = []
    for index, case in enumerate(prior):
        frame = _training_frame(
            case.starter_source,
            case.entrypoint,
            case.public_prompt,
        )
        action = {
            "arguments": {"source": case.successful_source},
            "name": WRITE_TOOL,
        }
        examples.append(
            example_type(
                prompt=coding._prompt(frame),
                output=f"<tool>{_json(action)}</tool>",
                inventory=python_interface_inventory(
                    case.starter_source,
                    case.entrypoint,
                ),
                cue_text=case.public_prompt,
                provenance_id=(
                    f"public-coding-prior-v1:{index}:"
                    f"{sha256(case.public_prompt.encode()).hexdigest()}"
                ),
            )
        )
    return tuple(examples)


def _bootstrap_corpus_sha256(
    examples: tuple[object, ...],
    learned_config: object,
) -> str:
    return _digest(
        "continual-coding-bootstrap-corpus-v6",
        {
            "examples": [
                {
                    "cue_text": example.cue_text,
                    "inventory_sha256": example.inventory.sha256,
                    "inventory_symbols": example.inventory.symbols,
                    "output": example.output,
                    "prompt": example.prompt,
                    "provenance_id": example.provenance_id,
                }
                for example in examples
            ],
            "learned_config": _audit_value(learned_config),
        },
    )


def _model_identity(model: object) -> dict[str, object]:
    return {
        "training_digest": model.training_digest,
        "vocabulary_size": model.vocabulary_size,
    }


def _grow_arms() -> tuple[
    dict[str, object],
    dict[str, object],
    tuple[object, ...],
    str,
]:
    """Lazy model import and deterministic independent growth, never inference."""

    from humanbrain.assembly_factorized_pointer import (
        AssemblyFactorizedPointerConfig,
        AssemblyFactorizedPointerModel,
        AssemblyPointerTrainingExample,
    )
    from humanbrain.recurrent_assembly_stack import AssemblyAblation

    base = AssemblyFactorizedPointerConfig()
    policies = {
        "treatment": (True, AssemblyAblation.NONE),
        "no_update": (False, AssemblyAblation.NONE),
        "stack_off": (False, AssemblyAblation.STACK_OFF),
    }
    configs = {
        name: AssemblyFactorizedPointerConfig(
            stack=base.stack,
            tokenizer=base.tokenizer,
            fragments=base.fragments,
            plasticity=plasticity,
            ablation=ablation,
        )
        for name, (plasticity, ablation) in policies.items()
    }
    examples = _bootstrap_examples(AssemblyPointerTrainingExample)
    models = {
        name: AssemblyFactorizedPointerModel.grow(examples, config)
        for name, config in configs.items()
    }
    identities = {name: _model_identity(model) for name, model in models.items()}
    if len({_json(identity) for identity in identities.values()}) != 1:
        raise RuntimeError("V6 independently grown arm checkpoints diverged")

    learned_config = {
        "fragments": base.fragments,
        "stack": base.stack,
        "tokenizer": base.tokenizer,
    }
    config_record = {
        name: _audit_value(config) for name, config in configs.items()
    }
    corpus_sha = _bootstrap_corpus_sha256(examples, learned_config)
    return models, config_record, examples, corpus_sha


def _public_prior_retention_probe(
    model: object,
    examples: tuple[object, ...],
) -> dict[str, object]:
    """Probe every public bootstrap row at one read-only learned checkpoint."""

    from humanbrain.pointer_binding import pointerize

    checkpoint = str(model.training_digest)
    rows: list[dict[str, object]] = []
    for index, example in enumerate(examples):
        before = str(model.training_digest)
        expected_output = pointerize(example.output, example.inventory)
        raw_output = model.raw_complete(
            example.prompt,
            example.inventory,
            cue_text=example.cue_text,
            max_new_tokens=512,
            stop_text=("</tool>", "</stop>"),
        )
        trace = model.last_trace
        if trace is None:
            raise RuntimeError("public-prior probe has no physical trace")
        emitted_stop = _STOP.fullmatch(raw_output.strip()) is not None
        if trace.selected_stop:
            resolution = "physical-stop-neutral"
        elif emitted_stop:
            resolution = "emitted-stop-neutral"
        else:
            update = model.resolve_pending_neutral(
                example.prompt,
                example.inventory,
                cue_text=example.cue_text,
            )
            if (
                update.valence.value != "neutral"
                or update.changed_edges
                or update.prior_training_digest != update.training_digest
            ):
                raise RuntimeError("public-prior neutral closure changed assembly state")
            resolution = "executable-neutral"
        after = str(model.training_digest)
        if before != checkpoint or after != checkpoint:
            raise RuntimeError("public-prior probe changed its learned checkpoint")
        rows.append(
            {
                "exact_match": raw_output == expected_output,
                "expected_output_sha256": _text_sha256(expected_output),
                "index": index,
                "physical_max_depth": trace.stack_trace.max_depth,
                "physical_operations": list(trace.stack_trace.operations),
                "physical_selected_stop": trace.selected_stop,
                "raw_output_sha256": _text_sha256(raw_output),
                "resolution": resolution,
            }
        )
    exact_matches = sum(int(row["exact_match"]) for row in rows)
    return {
        "checkpoint_digest": checkpoint,
        "exact_matches": exact_matches,
        "read_only_digest_preserved": str(model.training_digest) == checkpoint,
        "rows": rows,
        "total": len(rows),
    }


class V6ExecutionAdapter:
    """Execute only one sealed generic write; private policy opens afterward."""

    def __init__(
        self,
        tasks: tuple[ContinualToySweTrainTaskV6, ...],
        runtime: _SandboxRuntime,
    ) -> None:
        self._tasks = {
            task.public.task_commitment_sha256: task for task in tasks
        }
        self._runtime = runtime
        self._task_commitment_by_frame: dict[str, str] = {}
        self._source_by_commitment = {
            task.public.task_commitment_sha256: task.public.starter_source
            for task in tasks
        }
        self._audit_by_action: dict[str, dict[str, object]] = {}
        self._executed_actions: set[str] = set()
        self.sandbox_executions = 0
        self.policy_feedback_events = 0

    def bind(
        self,
        frame: coding.ObservationFrame,
        task: ContinualToySweTrainTaskV6,
    ) -> None:
        public = task.public
        public.authenticate()
        commitment = public.task_commitment_sha256
        if commitment not in self._tasks:
            raise ValueError("frame task is outside the V6 TRAIN stream")
        prior = self._task_commitment_by_frame.setdefault(frame.sha256, commitment)
        if prior != commitment:
            raise ValueError("frame was rebound to another public task")

    def source(self, task: ContinualToySweTrainTaskV6) -> str:
        return self._source_by_commitment[task.public.task_commitment_sha256]

    def audit(self, action_sha256: str | None) -> dict[str, object] | None:
        if action_sha256 is None:
            return None
        value = self._audit_by_action.get(action_sha256)
        return None if value is None else dict(value)

    @staticmethod
    def _generic_receipt(
        action: coding.ToolAction,
        *,
        accepted: bool,
        succeeded: bool,
        outcome: str,
    ) -> coding.ExecutionReceipt:
        return coding.ExecutionReceipt.seal(
            action,
            accepted=accepted,
            succeeded=succeeded,
            observations=(
                coding.ObservationBlock.seal("execution-outcome", outcome),
            ),
        )

    def execute(self, action: coding.ToolAction) -> coding.ExecutionReceipt:
        if action.sha256 in self._executed_actions:
            raise RuntimeError("sealed V6 action was executed more than once")
        self._executed_actions.add(action.sha256)

        commitment = self._task_commitment_by_frame.get(action.frame_sha256)
        if commitment is None:
            audit = {
                "host_status": "rejected",
                "reason": "unbound-frame",
                "sandbox_invoked": False,
            }
            self._audit_by_action[action.sha256] = audit
            return self._generic_receipt(
                action,
                accepted=False,
                succeeded=False,
                outcome="rejected",
            )
        try:
            arguments = json.loads(action.arguments_json)
        except json.JSONDecodeError:
            arguments = None
        if (
            action.tool_name != WRITE_TOOL
            or not isinstance(arguments, dict)
            or set(arguments) != {"source"}
            or not isinstance(arguments["source"], str)
        ):
            audit = {
                "host_status": "rejected",
                "reason": "invalid-write-source-shape",
                "sandbox_invoked": False,
            }
            self._audit_by_action[action.sha256] = audit
            return self._generic_receipt(
                action,
                accepted=False,
                succeeded=False,
                outcome="rejected",
            )

        source = arguments["source"]
        source_sha256 = sha256(source.encode("utf-8")).hexdigest()
        self._source_by_commitment[commitment] = source

        # The evaluator opens only after the exact action above is sealed and bound.
        task = self._tasks[commitment]
        task.authenticate()
        if task.private.expected_action == "stop":
            self.policy_feedback_events += 1
            audit = {
                "host_status": "unsuccessful",
                "sandbox_invoked": False,
                "sealed_action_sha256": action.sha256,
                "source_sha256": source_sha256,
            }
            self._audit_by_action[action.sha256] = audit
            return self._generic_receipt(
                action,
                accepted=True,
                succeeded=False,
                outcome="unsuccessful",
            )

        from humanbrain.simple_swe.sandbox import execute_completion

        generated = task.private.generated_task
        result = execute_completion(
            source,
            generated.public,
            generated.private,
            self._runtime.protocol.execution,
            python_executable=self._runtime.executable,
            python_executable_sha256=self._runtime.executable_sha256,
            worker_source=self._runtime.worker_source,
            worker_sha256=self._runtime.worker_sha256,
        )
        if result.sandboxed:
            self.sandbox_executions += 1
        audit = {
            "host_status": result.outcome,
            "passed": result.passed,
            "sandbox_invoked": result.sandboxed,
            "sandboxed": result.sandboxed,
            "sealed_action_sha256": action.sha256,
            "source_sha256": source_sha256,
            "wall_seconds_hex": float(result.wall_seconds).hex(),
        }
        self._audit_by_action[action.sha256] = audit
        return self._generic_receipt(
            action,
            accepted=True,
            succeeded=result.passed,
            outcome="succeeded" if result.passed else "unsuccessful",
        )


def _frame(
    adapter: V6ExecutionAdapter,
    task: ContinualToySweTrainTaskV6,
    previous: coding.ExecutionReceipt | None = None,
) -> coding.ObservationFrame:
    public = task.public
    public.authenticate()
    frame = coding.ObservationFrame.seal(
        _frame_blocks(
            starter_source=public.starter_source,
            current_source=adapter.source(task),
            signature=public.signature.to_record(),
            public_prompt=public.prompt,
            task_commitment_sha256=public.task_commitment_sha256,
            relative_path=public.relative_path,
            previous=previous,
        ),
        (_WRITE_AFFORDANCE,),
    )
    adapter.bind(frame, task)
    return frame


def _block_record(block: coding.ObservationBlock) -> dict[str, str]:
    return {"kind": block.kind, "sha256": block.sha256, "text": block.text}


def _completion_trace_record(trace: coding.CompletionTrace) -> dict[str, object]:
    return {
        "frame_sha256": trace.frame_sha256,
        "inventory_sha256": trace.inventory_sha256,
        "inventory_symbols": list(trace.inventory_symbols),
        "materialization_error": trace.materialization_error,
        "materialized_output": trace.materialized_output,
        "materialized_output_sha256": _text_sha256(trace.materialized_output),
        "model_digest": trace.model_digest,
        "raw_pointer_output": trace.raw_pointer_output,
        "raw_pointer_output_sha256": _text_sha256(trace.raw_pointer_output),
        "starter_block_sha256": trace.starter_block_sha256,
        "trace_sha256": trace.sha256,
    }


def _decision_record(decision: coding.Decision) -> dict[str, object]:
    if isinstance(decision, coding.StopDecision):
        return {
            "frame_sha256": decision.frame_sha256,
            "kind": "stop",
            "model_digest": decision.model_digest,
            "reason": decision.reason,
            "sha256": decision.sha256,
        }
    return {
        "arguments_json": decision.arguments_json,
        "completion_trace_sha256": decision.completion_trace_sha256,
        "frame_sha256": decision.frame_sha256,
        "kind": "tool",
        "model_digest": decision.model_digest,
        "sha256": decision.sha256,
        "tool_name": decision.tool_name,
    }


def _receipt_record(
    receipt: coding.ExecutionReceipt | None,
) -> dict[str, object] | None:
    if receipt is None:
        return None
    return {
        "accepted": receipt.accepted,
        "action_sha256": receipt.action_sha256,
        "frame_sha256": receipt.frame_sha256,
        "observations": [_block_record(block) for block in receipt.observations],
        "sha256": receipt.sha256,
        "succeeded": receipt.succeeded,
    }


def _pending_resolution(
    model: object,
    result: coding.EpisodeResult,
    *,
    plasticity: bool,
) -> str:
    trace = model.last_trace
    update = model.last_update
    if trace is None:
        raise RuntimeError("V6 episode has no assembly completion trace")
    if trace.raw_output != result.completion_trace.raw_pointer_output:
        raise RuntimeError("assembly trace output differs from completion custody")
    if trace.prior_training_digest != result.prior_model_digest:
        raise RuntimeError("assembly trace is not bound to the episode checkpoint")

    if trace.selected_stop:
        if update is not None or result.receipt is not None:
            raise RuntimeError("physical STOP was not closed neutrally in the adapter")
        return "physical-stop-neutral"

    materialized = result.completion_trace.materialized_output
    explicit_stop = (
        isinstance(materialized, str)
        and _STOP.fullmatch(materialized.strip()) is not None
    )
    if isinstance(result.decision, coding.StopDecision) and explicit_stop:
        if update is not None or result.receipt is not None:
            raise RuntimeError("emitted explicit STOP received a second consequence")
        return "emitted-stop-neutral"

    if update is None:
        raise RuntimeError("non-STOP neuronal decision remained unresolved")
    expected_polarity = (
        "success"
        if result.receipt is not None
        and result.receipt.accepted
        and result.receipt.succeeded
        else "failure"
    )
    if update.polarity != expected_polarity:
        raise RuntimeError("pending consequence polarity does not match host outcome")
    expected_valence = (
        "positive" if expected_polarity == "success" else "negative"
    ) if plasticity else "neutral"
    actual_valence = update.valence.value
    if actual_valence != expected_valence:
        raise RuntimeError("pending consequence valence does not match arm policy")
    return f"{expected_polarity}-{actual_valence}"


def _attempt_record(
    *,
    arm: str,
    attempt: int,
    frame: coding.ObservationFrame,
    model: object,
    adapter: V6ExecutionAdapter,
    result: coding.EpisodeResult,
    plasticity: bool,
) -> dict[str, object]:
    if not result.is_authentic():
        raise RuntimeError("V6 episode result is not authentic")
    if result.completion_trace.frame_sha256 != frame.sha256:
        raise RuntimeError("V6 episode is not bound to its public frame")
    trace = model.last_trace
    if trace is None:
        raise RuntimeError("V6 model did not expose its physical trace")
    receipt = result.receipt
    if isinstance(result.decision, coding.ToolAction):
        if receipt is None or receipt.action_sha256 != result.decision.sha256:
            raise RuntimeError("tool decision has no bound execution receipt")
    elif receipt is not None:
        raise RuntimeError("STOP decision unexpectedly has an execution receipt")

    materialized = result.completion_trace.materialized_output
    emitted_stop = (
        isinstance(result.decision, coding.StopDecision)
        and isinstance(materialized, str)
        and _STOP.fullmatch(materialized.strip()) is not None
    )
    status = (
        "physical_stop"
        if trace.selected_stop
        else "emitted_stop"
        if emitted_stop
        else "host_rejected_output"
        if isinstance(result.decision, coding.StopDecision)
        else "passed"
        if receipt is not None and receipt.succeeded
        else "rejected"
        if receipt is not None and not receipt.accepted
        else "failed"
    )
    action_sha = (
        result.decision.sha256
        if isinstance(result.decision, coding.ToolAction)
        else None
    )
    execution_audit = adapter.audit(action_sha)
    if action_sha is not None and execution_audit is None:
        raise RuntimeError("sealed tool action has no execution custody")
    resolution = _pending_resolution(model, result, plasticity=plasticity)
    stack_trace = trace.stack_trace
    return {
        "arm": arm,
        "assembly_trace": _audit_value(trace),
        "attempt": attempt,
        "completion_trace": _completion_trace_record(result.completion_trace),
        "decision": _decision_record(result.decision),
        "episode_sha256": result.sha256,
        "execution_audit": execution_audit,
        "frame": {
            "affordances": [item.sha256 for item in frame.affordances],
            "blocks": [_block_record(block) for block in frame.blocks],
            "sha256": frame.sha256,
        },
        "learning_applied": result.learning_applied,
        "learning_outcome": result.learning_outcome,
        "model_digest": result.model_digest,
        "pending_resolution": resolution,
        "physical_max_depth": stack_trace.max_depth,
        "physical_operations": list(stack_trace.operations),
        "physical_selected_stop": trace.selected_stop,
        "physical_simulated_steps": stack_trace.simulated_steps,
        "prior_model_digest": result.prior_model_digest,
        "receipt": _receipt_record(receipt),
        "status": status,
        "update": None if model.last_update is None else _audit_value(model.last_update),
    }


def _run_task(
    *,
    arm: str,
    learner: coding.PersistentCodingLearner,
    model: object,
    adapter: V6ExecutionAdapter,
    task: ContinualToySweTrainTaskV6,
    plasticity: bool,
    invariant_digest: str | None,
) -> list[dict[str, object]]:
    attempts: list[dict[str, object]] = []
    previous: coding.ExecutionReceipt | None = None
    for attempt_index in range(MAX_ATTEMPTS):
        frame = _frame(adapter, task, previous)
        result = learner.episode(frame)
        record = _attempt_record(
            arm=arm,
            attempt=attempt_index + 1,
            frame=frame,
            model=model,
            adapter=adapter,
            result=result,
            plasticity=plasticity,
        )
        attempts.append(record)
        if invariant_digest is not None and model.training_digest != invariant_digest:
            raise RuntimeError(f"{arm} changed its no-update checkpoint")
        if record["status"] in {
            "emitted_stop",
            "passed",
            "physical_stop",
        }:
            break
        if result.receipt is None:
            if record["status"] != "host_rejected_output":
                raise RuntimeError("retry lacks an authenticated previous receipt")
            # Parser/materialization rejection already delivered one internal
            # negative consequence. Retry the same public frame without
            # inventing a repository receipt that never existed.
            previous = None
        else:
            previous = result.receipt
    return attempts


def _run_stack_off_first_attempt(
    *,
    learner: coding.PersistentCodingLearner,
    model: object,
    adapter: V6ExecutionAdapter,
    task: ContinualToySweTrainTaskV6,
    invariant_digest: str,
) -> dict[str, object]:
    frame = _frame(adapter, task)
    result = learner.episode(frame)
    record = _attempt_record(
        arm="stack_off",
        attempt=1,
        frame=frame,
        model=model,
        adapter=adapter,
        result=result,
        plasticity=False,
    )
    if model.training_digest != invariant_digest:
        raise RuntimeError("STACK_OFF changed its no-update checkpoint")
    if record["assembly_trace"]["ablation"] != "stack_off":
        raise RuntimeError("STACK_OFF physical trace lost its ablation tag")
    return record


def _arm_score(
    expected_action: str,
    attempts: list[dict[str, object]],
) -> dict[str, object]:
    first = attempts[0]
    if expected_action == "stop":
        first_correct = bool(
            first["physical_selected_stop"]
            and first["receipt"] is None
            and not (
                first["execution_audit"]
                and first["execution_audit"].get("sandbox_invoked")
            )
        )
        return {
            "correct": first_correct,
            "first_attempt_correct": first_correct,
            "late_stop_attempts": [
                row["attempt"]
                for row in attempts[1:]
                if row["physical_selected_stop"]
            ],
            "winning_attempt": 1 if first_correct else None,
        }

    winning = next(
        (row["attempt"] for row in attempts if row["status"] == "passed"),
        None,
    )
    return {
        "correct": winning is not None,
        "first_attempt_correct": winning == 1,
        "late_stop_attempts": [
            row["attempt"]
            for row in attempts[1:]
            if row["physical_selected_stop"]
        ],
        "winning_attempt": winning,
    }


def _retry_pairs(attempts: list[dict[str, object]]) -> list[dict[str, object]]:
    pairs = []
    for left, right in pairwise(attempts):
        left_audit = left["execution_audit"] or {}
        right_audit = right["execution_audit"] or {}
        left_source = left_audit.get("source_sha256")
        right_source = right_audit.get("source_sha256")
        changed = bool(
            isinstance(left_source, str)
            and isinstance(right_source, str)
            and left_source != right_source
        )
        left_sandbox_failed = bool(
            left["status"] == "failed"
            and left_audit.get("sandbox_invoked") is True
            and left_audit.get("sandboxed") is True
            and left_audit.get("passed") is False
            and left_audit.get("host_status") != "harness_error"
        )
        if right["status"] == "passed":
            classification = "fail-to-changed-pass" if changed else "fail-to-same-pass"
        elif right["physical_selected_stop"]:
            classification = "fail-to-stop"
        elif changed:
            classification = "changed-but-unsuccessful"
        else:
            classification = "unchanged-failure"
        pairs.append(
            {
                "changed": changed,
                "classification": classification,
                "from_attempt": left["attempt"],
                "left_sandbox_failed": left_sandbox_failed,
                "left_source_sha256": left_source,
                "left_update": left["update"],
                "right_source_sha256": right_source,
                "source_changed": changed,
                "to_attempt": right["attempt"],
            }
        )
    return pairs


def _wilson(successes: int, total: int) -> dict[str, float | int]:
    if total == 0:
        return {"successes": successes, "total": 0, "rate": 0.0, "low": 0.0, "high": 1.0}
    z = 1.959963984540054
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return {
        "high": min(1.0, center + margin),
        "low": max(0.0, center - margin),
        "rate": rate,
        "successes": successes,
        "total": total,
    }


def _paired_counts(
    records: list[dict[str, object]],
    left: str,
    right: str,
) -> dict[str, int]:
    counts = Counter()
    for record in records:
        left_value = bool(record["scores"][left]["correct"])
        right_value = bool(record["scores"][right]["correct"])
        key = (
            "both"
            if left_value and right_value
            else f"{left}_only"
            if left_value
            else f"{right}_only"
            if right_value
            else "neither"
        )
        counts[key] += 1
    return dict(counts)


def _posthoc_score(
    tasks: tuple[ContinualToySweTrainTaskV6, ...],
    records: list[dict[str, object]],
    public_prior_retention: dict[str, dict[str, object]],
) -> dict[str, object]:
    """Open evaluator labels only after every arm sealed every decision."""

    if not tasks or len(records) != len(tasks):
        raise RuntimeError("V6 task records do not match the frozen stream")
    stage_counts: dict[str, Counter[str]] = {}
    arity_counts: dict[str, Counter[str]] = {}
    capability_counts: dict[str, Counter[str]] = {}
    retry_counts = {"treatment": Counter(), "no_update": Counter()}
    cumulative = []
    running = Counter()
    correction_events = []
    recurrence_events = []

    for ordinal, (task, record) in enumerate(zip(tasks, records, strict=True)):
        task.authenticate()
        private = task.private
        scores = {
            "treatment": _arm_score(private.expected_action, record["attempts"]),
            "no_update": _arm_score(
                private.expected_action,
                record["no_update_attempts"],
            ),
            "stack_off": _arm_score(
                private.expected_action,
                [record["stack_off_first_attempt"]],
            ),
        }
        record["scores"] = scores
        record["posthoc"] = {
            "authored_attempt_budget": private.attempt_budget,
            "capability_group": private.capability_group,
            "expected_action": private.expected_action,
            "feedback_pressure": private.feedback_pressure,
            "interface_arity": len(task.public.signature.parameters),
            "private_environment_sha256": private.environment_sha256,
            "retention_anchor_ordinal": private.retention_anchor_ordinal,
            "stage": private.stage,
        }
        record["retry_pairs"] = {
            "treatment": _retry_pairs(record["attempts"]),
            "no_update": _retry_pairs(record["no_update_attempts"]),
        }
        for arm in ("treatment", "no_update"):
            retry_counts[arm].update(
                pair["classification"] for pair in record["retry_pairs"][arm]
            )
        for arm, score in scores.items():
            running[arm] += int(score["correct"])
            stage_counts.setdefault(private.stage, Counter())[arm] += int(
                score["correct"]
            )
            capability_counts.setdefault(
                private.capability_group,
                Counter(),
            )[arm] += int(score["correct"])
            arity = str(len(task.public.signature.parameters))
            arity_counts.setdefault(arity, Counter())[arm] += int(score["correct"])
        stage_counts[private.stage]["tasks"] += 1
        capability_counts[private.capability_group]["tasks"] += 1
        arity_counts[str(len(task.public.signature.parameters))]["tasks"] += 1
        cumulative.append(
            {
                "no_update_correct": running["no_update"],
                "seen": ordinal + 1,
                "stack_off_correct": running["stack_off"],
                "treatment_correct": running["treatment"],
            }
        )

        for pair in record["retry_pairs"]["treatment"]:
            update = pair["left_update"]
            if (
                pair["classification"] == "fail-to-changed-pass"
                and pair["left_sandbox_failed"]
                and pair["source_changed"]
                and update is not None
                and update["polarity"] == "failure"
                and update["valence"] == "negative"
                and bool(update["changed_edges"])
            ):
                correction_events.append(
                    {
                        "from_attempt": pair["from_attempt"],
                        "task_ordinal": ordinal,
                        "to_attempt": pair["to_attempt"],
                    }
                )

        control_first = record["no_update_attempts"][0]
        operations = set(control_first["physical_operations"])
        if (
            private.expected_action == "execute"
            and scores["no_update"]["first_attempt_correct"]
            and not scores["stack_off"]["first_attempt_correct"]
            and control_first["physical_max_depth"] >= 2
            and {"push", "pop"} <= operations
        ):
            recurrence_events.append(
                {
                    "control_depth": control_first["physical_max_depth"],
                    "control_operations": control_first["physical_operations"],
                    "task_ordinal": ordinal,
                }
            )

    totals = {
        arm: sum(int(record["scores"][arm]["correct"]) for record in records)
        for arm in ("treatment", "no_update", "stack_off")
    }
    first_totals = {
        arm: sum(
            int(record["scores"][arm]["first_attempt_correct"])
            for record in records
        )
        for arm in ("treatment", "no_update", "stack_off")
    }
    treatment_only = sum(
        int(
            record["scores"]["treatment"]["correct"]
            and not record["scores"]["no_update"]["correct"]
        )
        for record in records
    )
    execute_passes = sum(
        int(
            record["posthoc"]["expected_action"] == "execute"
            and record["scores"]["treatment"]["correct"]
        )
        for record in records
    )
    uncertainty_rows = [
        record
        for record in records
        if record["posthoc"]["expected_action"] == "stop"
    ]
    uncertainty_correct = sum(
        int(record["scores"]["treatment"]["first_attempt_correct"])
        for record in uncertainty_rows
    )
    uncertainty_no_sandbox = all(
        not record["attempts"][0]["execution_audit"]
        for record in uncertainty_rows
    )
    executable_stages = sorted(
        {
            str(record["posthoc"]["stage"])
            for record in records
            if record["posthoc"]["expected_action"] == "execute"
        }
    )
    executable_stage_coverage = {
        stage: stage_counts[stage]["treatment"] > 0
        for stage in executable_stages
    }
    retention_events = []
    for ordinal, record in enumerate(records):
        anchor = record["posthoc"]["retention_anchor_ordinal"]
        if anchor is None:
            continue
        anchor_solved = bool(records[anchor]["scores"]["treatment"]["correct"])
        retention_first = bool(
            record["scores"]["treatment"]["first_attempt_correct"]
        )
        control_retention_first = bool(
            record["scores"]["no_update"]["first_attempt_correct"]
        )
        if anchor_solved and retention_first and not control_retention_first:
            retention_events.append(
                {
                    "anchor_ordinal": anchor,
                    "no_update_first_attempt_correct": False,
                    "retention_ordinal": ordinal,
                }
            )

    criteria = {
        "at_least_one_executable_pass": execute_passes >= 1,
        "all_public_prior_cases_retained": bool(
            public_prior_retention["treatment"]["read_only_digest_preserved"]
            and public_prior_retention["treatment"]["exact_matches"]
            == public_prior_retention["treatment"]["total"]
            == PUBLIC_PRIOR_CASES
        ),
        "every_executable_stage_has_treatment_success": all(
            executable_stage_coverage.values()
        ),
        "fail_changed_pass_after_scoped_negative_update": bool(correction_events),
        "causal_delayed_retention_at_least_one": bool(retention_events),
        "same_state_recurrence_vs_stack_off": bool(recurrence_events),
        "treatment_advantage": (
            totals["treatment"] > totals["no_update"] and treatment_only >= 1
        ),
        "uncertainty_two_of_two_physical_stop_without_sandbox": (
            len(uncertainty_rows) == 2
            and uncertainty_correct == 2
            and uncertainty_no_sandbox
        ),
    }
    return {
        "capability_curve": cumulative,
        "correction_events": correction_events,
        "criteria": criteria,
        "execute_passes": execute_passes,
        "first_attempt": {
            arm: _wilson(first_totals[arm], len(records)) for arm in first_totals
        },
        "gate_pass": all(criteria.values()),
        "overall": {arm: _wilson(totals[arm], len(records)) for arm in totals},
        "paired_treatment_no_update": _paired_counts(
            records,
            "treatment",
            "no_update",
        ),
        "paired_no_update_stack_off": _paired_counts(
            records,
            "no_update",
            "stack_off",
        ),
        "public_prior_retention": public_prior_retention,
        "recurrence_events": recurrence_events,
        "retention_events": retention_events,
        "retry_outcomes": {
            arm: dict(sorted(counts.items())) for arm, counts in retry_counts.items()
        },
        "stage_scores": {
            stage: {
                arm: _wilson(counts[arm], counts["tasks"])
                for arm in ("treatment", "no_update", "stack_off")
            }
            for stage, counts in sorted(stage_counts.items())
        },
        "arity_scores": {
            arity: {
                arm: _wilson(counts[arm], counts["tasks"])
                for arm in ("treatment", "no_update", "stack_off")
            }
            for arity, counts in sorted(arity_counts.items())
        },
        "capability_group_scores": {
            group: {
                arm: _wilson(counts[arm], counts["tasks"])
                for arm in ("treatment", "no_update", "stack_off")
            }
            for group, counts in sorted(capability_counts.items())
        },
        "executable_stage_coverage": executable_stage_coverage,
        "treatment_only_wins": treatment_only,
        "uncertainty": {
            "correct": uncertainty_correct,
            "no_sandbox": uncertainty_no_sandbox,
            "tasks": len(uncertainty_rows),
        },
    }


def _schema_preflight() -> dict[str, object]:
    """Authenticate public V6 frames only; never construct or invoke a model."""

    _assert_fixture_custody()
    _assert_dependency_custody()
    tasks = continual_toy_swe_train_stream_v6()
    frame_roots = []
    for task in tasks:
        public = task.public
        public.authenticate()
        public_record = public.record()
        if _PRIVATE_FIELDS & set(public_record):
            raise RuntimeError("private evaluator metadata entered public projection")
        frame = coding.ObservationFrame.seal(
            _frame_blocks(
                starter_source=public.starter_source,
                current_source=public.starter_source,
                signature=public.signature.to_record(),
                public_prompt=public.prompt,
                task_commitment_sha256=public.task_commitment_sha256,
                relative_path=public.relative_path,
            ),
            (_WRITE_AFFORDANCE,),
        )
        signatures = [
            block for block in frame.blocks if block.kind == "typed-signature"
        ]
        if len(signatures) != 1:
            raise RuntimeError("V6 frame lacks exactly one typed signature")
        signature = json.loads(signatures[0].text)
        if signature != public.signature.to_record() or _json(signature) != signatures[0].text:
            raise RuntimeError("V6 typed signature is not canonical or task-bound")
        if any(block.kind in _PRIVATE_FIELDS for block in frame.blocks):
            raise RuntimeError("private evaluator block entered learner cognition")
        frame_roots.append(frame.sha256)
    return {
        "adapter_instantiations": 0,
        "artifact_created_by_preflight": False,
        "artifact_exists_before_preflight": OUTPUT_PATH.exists(),
        "dependency_count": len(_EXPECTED_DEPENDENCY_SHA256),
        "dependency_custody_sha256": _digest(
            "v6-dependency-custody",
            _EXPECTED_DEPENDENCY_SHA256,
        ),
        "frame_roots_sha256": _digest("v6-preflight-frame-roots", frame_roots),
        "learner_invocations": 0,
        "model_bootstrap_calls": 0,
        "assembly_model_imports": 0,
        "output_path": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
        "private_labels_used_for_frame_construction": 0,
        "schema": SCHEMA,
        "task_count": len(tasks),
        "task_source_sha256": V6_TASK_SOURCE_SHA256,
        "train_stream_custody_sha256": TRAIN_STREAM_CUSTODY_SHA256,
        "train_stream_sha256": TRAIN_STREAM_SHA256,
        "typed_signature_blocks_per_frame": 1,
    }


def _execution_custody(runtime: _SandboxRuntime) -> dict[str, object]:
    custody: dict[str, object] = {
        name: _sha256_file(path) for name, path in _CUSTODY_PATHS.items()
    }
    custody.update(
        {
            "python_executable_sha256": runtime.executable_sha256,
            "worker_runtime_sha256": runtime.worker_sha256,
        }
    )
    return custody


def _corpus_custody(
    *,
    bootstrap_corpus_sha256: str,
    model_configs: dict[str, object],
) -> dict[str, object]:
    from experiments.r4_integrated.public_coding_prior_v1 import (
        PUBLIC_CODING_PRIOR_V1_SHA256,
        SOURCE_CORPUS_SHA256,
        SOURCE_FREEZE_COMMIT,
        SOURCE_MANIFEST_SHA256,
    )

    if (
        PUBLIC_CODING_PRIOR_V1_SHA256
        != "81a60454b9af6b8592ea8b422c25bdd98115c101e5580c8cd571555fdf0abfb1"
        or SOURCE_CORPUS_SHA256
        != "b35a2edeabfb9e78755c4eb7bf3c77babacdddeaf2c0b4164af0c934e54ea56c"
        or SOURCE_MANIFEST_SHA256
        != "0942545cd03e21ac94c047db90c15e59b9327c5171db74afd3963d3cbeef7362"
    ):
        raise RuntimeError("public coding prior custody changed")

    return {
        "bootstrap_cases": PUBLIC_PRIOR_CASES,
        "bootstrap_corpus_sha256": bootstrap_corpus_sha256,
        "cohort_novel_to_model_at_spend": True,
        "cohort_novel_to_orchestrator_developer": False,
        "cohort_structurally_inspected_by_development_agents": True,
        "cohort_unseen_by_representation_implementation_agent": True,
        "model_configs": model_configs,
        "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
        "source_corpus_sha256": SOURCE_CORPUS_SHA256,
        "source_freeze_commit": SOURCE_FREEZE_COMMIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "synthetic_stop_supervision_examples": 0,
        "target_task_teacher_sources_used": 0,
        "target_tasks_present_at_prior_freeze": False,
        "task_count": TASK_COUNT,
        "train_stream_custody": train_stream_custody_v6(),
        "train_stream_custody_sha256": TRAIN_STREAM_CUSTODY_SHA256,
        "train_stream_sha256": TRAIN_STREAM_SHA256,
    }


@dataclass(slots=True)
class _Prepared:
    runtime: _SandboxRuntime
    models: dict[str, object]
    model_configs: dict[str, object]
    bootstrap_corpus_sha256: str
    bootstrap_examples: tuple[object, ...]
    initial_models: dict[str, dict[str, object]]
    initial_public_prior_retention: dict[str, dict[str, object]]
    preflight: dict[str, object]
    launch_check: dict[str, object]
    corpus_custody: dict[str, object]
    execution_custody: dict[str, object]


def _reservation_record(prepared: _Prepared) -> dict[str, object]:
    return {
        "artifact_path": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
        "corpus_custody": prepared.corpus_custody,
        "execution_custody": prepared.execution_custody,
        "initial_models": prepared.initial_models,
        "initial_public_prior_retention": prepared.initial_public_prior_retention,
        "launch_check": prepared.launch_check,
        "reservation_policy": "absent-only-exclusive-create-before-first-episode",
        "schema": RESERVATION_SCHEMA,
        "schema_preflight": prepared.preflight,
        "status": "RUNNING",
    }


def _prepare() -> _Prepared:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"V6 session artifact already exists: {OUTPUT_PATH}")
    _assert_fixture_custody()
    preflight = _schema_preflight()
    launch_check = _assert_projection_launch_ready()
    runtime = _sandbox_runtime()
    models, configs, examples, corpus_sha = _grow_arms()
    initial = {name: _model_identity(model) for name, model in models.items()}
    if len({_json(identity) for identity in initial.values()}) != 1:
        raise RuntimeError("V6 arms do not share one initial learned-state root")
    initial_prior_retention = {
        name: _public_prior_retention_probe(models[name], examples)
        for name in ("treatment", "no_update")
    }
    if any(
        result["exact_matches"] != PUBLIC_PRIOR_CASES
        or result["total"] != PUBLIC_PRIOR_CASES
        or not result["read_only_digest_preserved"]
        for result in initial_prior_retention.values()
    ):
        raise RuntimeError("public bootstrap prior is not replayable before V6 spend")
    corpus = _corpus_custody(
        bootstrap_corpus_sha256=corpus_sha,
        model_configs=configs,
    )
    execution = _execution_custody(runtime)
    return _Prepared(
        runtime=runtime,
        models=models,
        model_configs=configs,
        bootstrap_corpus_sha256=corpus_sha,
        bootstrap_examples=examples,
        initial_models=initial,
        initial_public_prior_retention=initial_prior_retention,
        preflight=preflight,
        launch_check=launch_check,
        corpus_custody=corpus,
        execution_custody=execution,
    )


def run_session(prepared: _Prepared) -> dict[str, object]:
    tasks = continual_toy_swe_train_stream_v6()
    if len(tasks) != TASK_COUNT:
        raise RuntimeError("V6 task count changed after reservation")

    adapters = {
        name: V6ExecutionAdapter(tasks, prepared.runtime)
        for name in ("treatment", "no_update", "stack_off")
    }
    learners = {
        name: coding.PersistentCodingLearner(prepared.models[name], adapters[name])
        for name in adapters
    }
    control_root = prepared.initial_models["no_update"]["training_digest"]
    stack_off_root = prepared.initial_models["stack_off"]["training_digest"]
    task_records: list[dict[str, object]] = []

    for ordinal, task in enumerate(tasks):
        public = task.public
        public.authenticate()
        attempts = _run_task(
            arm="treatment",
            learner=learners["treatment"],
            model=prepared.models["treatment"],
            adapter=adapters["treatment"],
            task=task,
            plasticity=True,
            invariant_digest=None,
        )
        no_update_attempts = _run_task(
            arm="no_update",
            learner=learners["no_update"],
            model=prepared.models["no_update"],
            adapter=adapters["no_update"],
            task=task,
            plasticity=False,
            invariant_digest=str(control_root),
        )
        stack_off = _run_stack_off_first_attempt(
            learner=learners["stack_off"],
            model=prepared.models["stack_off"],
            adapter=adapters["stack_off"],
            task=task,
            invariant_digest=str(stack_off_root),
        )
        task_records.append(
            {
                "attempts": attempts,
                "no_update_attempts": no_update_attempts,
                "public_observation_sha256": public.observation_sha256,
                "stack_off_first_attempt": stack_off,
                "task_commitment_sha256": public.task_commitment_sha256,
                "task_ordinal": ordinal,
                "task_sha256": task.task_sha256,
            }
        )

    task_final_models = {
        name: _model_identity(model) for name, model in prepared.models.items()
    }
    public_prior_retention = {
        name: _public_prior_retention_probe(
            prepared.models[name],
            prepared.bootstrap_examples,
        )
        for name in ("treatment", "no_update")
    }
    if {
        name: _model_identity(model) for name, model in prepared.models.items()
    } != task_final_models:
        raise RuntimeError("read-only public-prior probes changed a V6 checkpoint")

    # This is the first runner point that opens scoring labels and retention anchors.
    evaluation = _posthoc_score(
        tasks,
        task_records,
        public_prior_retention,
    )
    final_models = {
        name: _model_identity(model) for name, model in prepared.models.items()
    }
    if final_models["no_update"] != prepared.initial_models["no_update"]:
        raise RuntimeError("no-update model changed across V6")
    if final_models["stack_off"] != prepared.initial_models["stack_off"]:
        raise RuntimeError("STACK_OFF model changed across V6")

    final_execution = _execution_custody(prepared.runtime)
    final_corpus = _corpus_custody(
        bootstrap_corpus_sha256=prepared.bootstrap_corpus_sha256,
        model_configs=prepared.model_configs,
    )
    pending_resolution_count = sum(
        1
        for record in task_records
        for key in ("attempts", "no_update_attempts")
        for attempt in record[key]
        if attempt["pending_resolution"]
    ) + len(task_records)
    attempt_count = sum(
        len(record["attempts"]) + len(record["no_update_attempts"]) + 1
        for record in task_records
    )
    if pending_resolution_count != attempt_count:
        raise RuntimeError("V6 pending-resolution custody is incomplete")

    return {
        "claim_boundary": (
            "fresh public-TRAIN integrated V6 continual coding evidence only; "
            "not protected evaluation, family-disjoint transfer, Simple-SWE M0, "
            "SWE-bench, external-model comparison, or broad intelligence evidence"
        ),
        "corpus_custody": final_corpus,
        "evaluation": evaluation,
        "execution_custody": final_execution,
        "final_models": final_models,
        "full_replay_between_tasks": False,
        "hard_validity": {
            "all_attempts_authentic_and_resolved": True,
            "attempt_count": attempt_count,
            "initial_arm_roots_equal": True,
            "no_duplicate_action_execution": True,
            "no_update_digest_unchanged": True,
            "pending_resolution_count": pending_resolution_count,
            "private_authored_budgets_used_for_cognition": False,
            "private_fields_in_learner_frames": 0,
            "stack_off_digest_unchanged": True,
            "target_task_teacher_sources_used": 0,
        },
        "initial_models": prepared.initial_models,
        "initial_public_prior_retention": prepared.initial_public_prior_retention,
        "max_attempts_for_every_main_arm_task": MAX_ATTEMPTS,
        "one_persistent_no_update_control": True,
        "one_persistent_treatment_learner": True,
        "external_retry_feedback_if_present_is_generic_receipt_only": True,
        "public_prior_retention": public_prior_retention,
        "schema": SCHEMA,
        "stack_off_protocol": (
            "independently-grown same-baseline no-update STACK_OFF; one original-"
            "frame first attempt; no retry or retained feedback"
        ),
        "status": "COMPLETED",
        "task_count": len(tasks),
        "task_final_models": task_final_models,
        "tasks": task_records,
    }


def _write_exclusive(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_terminal(path: Path, record: dict[str, object]) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            delete=False,
            dir=path.parent,
            encoding="utf-8",
            prefix=f"{path.name}.",
            suffix=".tmp",
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(record, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--preflight-schema",
        action="store_true",
        help="authenticate only frozen public schema and inert frame seals",
    )
    mode.add_argument(
        "--run",
        action="store_true",
        help="irreversibly reserve and spend the fresh V6 TRAIN cohort",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.preflight_schema:
        print(json.dumps(_schema_preflight(), indent=2, sort_keys=True))
        return 0
    if not args.run:
        raise RuntimeError("V6 execution requires the explicit --run flag")

    prepared = _prepare()
    reservation = _reservation_record(prepared)
    _write_exclusive(OUTPUT_PATH, reservation)
    try:
        result = run_session(prepared)
        if (
            result["corpus_custody"] != reservation["corpus_custody"]
            or result["execution_custody"] != reservation["execution_custody"]
        ):
            raise RuntimeError("final V6 custody differs from exact reservation")
        result["reservation_custody_matches_final"] = True
    except BaseException as error:
        failed = dict(reservation)
        failed["error"] = str(error)
        failed["error_type"] = type(error).__name__
        failed["status"] = "FAILED"
        _write_terminal(OUTPUT_PATH, failed)
        raise

    completed_reservation = dict(reservation)
    completed_reservation["status"] = "COMPLETED"
    result["reservation"] = completed_reservation
    _write_terminal(OUTPUT_PATH, result)
    print(f"wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
