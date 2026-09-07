"""One persistent sequence reader and raw action circuit with actual feedback custody."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from humanbrain.neuronal_splice_learner import (
    LEARNING_RATE,
    HeadDecision,
    NeuronalSpliceLearner,
    _add,
    _digest,
    _finite,
)
from humanbrain.sequence_assembly_action_cells import (
    WIDTHS,
    SequenceAssemblyActionCircuit,
    encode,
    end_offsets,
)
from humanbrain.sequence_assembly_memory_learner import (
    SequenceAssemblyFeedback,
    SequenceAssemblyMemoryLearner,
    SequenceAssemblyPrediction,
)
from humanbrain.sequence_assembly_observation import (
    MAX_REPLACEMENT_BYTES,
    MAX_STREAM_BYTES,
    RawObservation,
    SpliceProposal,
    apply_splice,
)


@dataclass(frozen=True)
class SequenceAssemblySplicePrediction:
    proposal: SpliceProposal
    attempt_id: str
    proposal_sha256: str
    observation_sha256: str
    model_sha256: str
    memory_sha256: str
    choices: dict[str, HeadDecision]
    work: dict
    payload: SequenceAssemblyPrediction


@dataclass(frozen=True)
class SequenceAssemblySpliceFeedback:
    attempt_id: str
    action_updates: dict[str, bool]
    action_losses: dict[str, float]
    teacher_choices: dict[str, HeadDecision]
    reader_feedback: SequenceAssemblyFeedback | None
    work: dict


def _action_trace(prediction):
    return {
        "attempt_id": prediction.attempt_id,
        "proposal_sha256": prediction.proposal_sha256,
        "observation_sha256": prediction.observation_sha256,
        "model_sha256": prediction.model_sha256,
        "memory_sha256": prediction.memory_sha256,
        "choices": {head: vars(choice) for head, choice in prediction.choices.items()},
        "work": prediction.work,
    }


class SequenceAssemblySpliceLearner(NeuronalSpliceLearner):
    """Same model/tape through propose, real action, and one teach or acknowledgement.

    Reading uses this same ``reader`` directly with RawObservation. An unresolved
    reading or action prevents another action or checkpoint replacement. All
    action update proposals precede the reader's atomic publication.
    """

    def __init__(self, *, mode="learned", seed=733, reader=None):
        # Do not construct a legacy action circuit merely to replace it.
        nn.Module.__init__(self)
        if mode not in self.MODES or type(seed) is not int:
            raise ValueError("invalid sequence action mode or seed")
        self.mode, self.seed = mode, seed
        self.reader = (
            reader
            if reader is not None
            else SequenceAssemblyMemoryLearner(mode=self.MODES[mode], seed=seed)
        )
        if (
            not isinstance(self.reader, SequenceAssemblyMemoryLearner)
            or self.reader.mode != self.MODES[mode]
            or self.reader.seed != seed
        ):
            raise ValueError("actions require the selected same-instance sequence reader")
        self.actions = SequenceAssemblyActionCircuit(seed=seed)
        self._experience = 0
        self._pending = None
        self._check()

    def _check(self, work=None):
        if (
            self.mode not in self.MODES
            or self.reader.mode != self.MODES[self.mode]
            or self.reader.seed != self.seed
            or self.actions.seed != self.seed
        ):
            raise ValueError("sequence action control configuration changed")
        self.reader._check_conditioned_parameters(work)
        self.actions.check_parameters()

    def _require_idle(self):
        if self._pending is not None or self.reader.pending_prediction is not None:
            raise ValueError("resolve both actual action and reader response first")

    def get_extra_state(self):
        return {
            "schema": "sequence-assembly-splice-learner-v1",
            "mode": self.mode,
            "seed": self.seed,
            "widths": dict(WIDTHS),
            "feature_energy": 32,
            "operation_augmented_energy": 33,
            "learning_rate": LEARNING_RATE,
            "payload_cap": MAX_REPLACEMENT_BYTES,
            "observation": "whole raw request, workspace and actual public diagnostics",
            "publication": "all action proposals validated then atomic reader then action copies",
            "stop_teacher": "operation only; acknowledge actual reader without payload labels",
            "execution": "four-key owned-action envelope with nested actual execution-v2",
            "work": "reader prevalidation counted in action; reader call has its own work",
        }

    def snapshot(self, state):
        self._require_idle()
        return super().snapshot(state)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._require_idle()
        self._check()
        if not strict or assign or not isinstance(state_dict, Mapping):
            raise ValueError("sequence action loading requires strict in-place state")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("sequence action checkpoint fields mismatch")
        # Validate the entire recursive state before nn.Module starts copying.
        for name, prototype in expected.items():
            value = state_dict[name]
            if isinstance(prototype, Tensor):
                if (
                    not isinstance(value, Tensor)
                    or value.shape != prototype.shape
                    or value.dtype != prototype.dtype
                    or value.device != prototype.device
                    or value.requires_grad
                    or value.grad_fn is not None
                    or value.grad is not None
                ):
                    raise ValueError("sequence action checkpoint tensor contract mismatch")
                _finite(value, "sequence action checkpoint tensor")
            elif value != prototype:
                raise ValueError("sequence action checkpoint metadata mismatch")
        for name in self.reader.FIXED_STATE_NAMES:
            if not torch.equal(state_dict["reader." + name], expected["reader." + name]):
                raise ValueError("checkpoint changes inherited fixed reader wiring")
        self.reader.context.validate_fixed_state(state_dict, "reader.context.")
        self.reader.assemblies.validate_state(state_dict, "reader.assemblies.")
        # The inherited strict loader repeats its generic checks, then publishes
        # already validated values. No child custom load method is relied upon.
        return super().load_state_dict(state_dict, strict=True, assign=False)

    @torch.no_grad()
    def propose(self, observation, state):
        self._require_idle()
        work = {}
        self._check(work)
        self.reader._check_state(state)
        if not isinstance(observation, RawObservation) or observation.workspace is None:
            raise ValueError("action requires the complete raw workspace observation")
        workspace = observation.workspace
        encoded = encode(observation, self.reader.input_codes, work)
        model_hash = _digest(self.state_dict(), work)
        memory_hash = _digest(vars(state), work)
        start = self.actions.score("start", encoded, work=work)
        path, offset = workspace.boundaries[start.selected_index]
        ends = end_offsets(encoded, path, offset)
        end = self.actions.score("end", encoded, path=path, start=offset, work=work)
        stop = ends[end.selected_index]
        payload = self.reader.predict_with_receipt(observation, state, MAX_REPLACEMENT_BYTES)
        generated = payload.generation
        operation = self.actions.score(
            "operation",
            encoded,
            path=path,
            start=offset,
            end=stop,
            payload=generated.data,
            stop_reason=generated.stop_reason,
            work=work,
        )
        del encoded
        proposal = SpliceProposal(
            "STOP" if operation.selected_index == 0 else "SPLICE",
            path,
            offset,
            stop,
            generated.data,
            generated.stop_reason,
            workspace.sha256,
        )
        if (
            _digest(self.state_dict(), work) != model_hash
            or _digest(vars(state), work) != memory_hash
        ):
            raise RuntimeError("free action changed the neuronal model or memory")
        self._experience += 1
        prediction = SequenceAssemblySplicePrediction(
            proposal,
            f"{id(self)}:{self._experience}",
            proposal.sha256,
            observation.sha256,
            model_hash,
            memory_hash,
            {"start": start, "end": end, "operation": operation},
            {"action": work, "reader": dict(payload.work)},
            payload,
        )
        self._pending = prediction, observation.sha256, _digest(_action_trace(prediction))
        return prediction

    def _validate_execution(self, prediction, observation, state, receipt, work):
        if (
            self._pending is None
            or self._pending[0] is not prediction
            or not isinstance(prediction, SequenceAssemblySplicePrediction)
        ):
            raise ValueError("execution must resolve this instance's actual pending action")
        if (
            not isinstance(observation, RawObservation)
            or observation.workspace is None
            or observation.sha256 != self._pending[1]
            or prediction.observation_sha256 != observation.sha256
            or prediction.proposal.workspace_sha256 != observation.workspace.sha256
            or prediction.proposal.sha256 != prediction.proposal_sha256
            or _digest(_action_trace(prediction)) != self._pending[2]
            or _digest(self.state_dict(), work) != prediction.model_sha256
            or _digest(vars(state), work) != prediction.memory_sha256
        ):
            raise ValueError("actual action observation, choices, model or memory changed")
        if (
            not isinstance(receipt, Mapping)
            or set(receipt) != {"attempt_id", "proposal_sha256", "workspace_sha256", "execution"}
            or receipt["attempt_id"] != prediction.attempt_id
            or receipt["proposal_sha256"] != prediction.proposal_sha256
            or receipt["workspace_sha256"] != observation.workspace.sha256
        ):
            raise ValueError("execution envelope does not bind the actual action")
        execution = receipt["execution"]
        if (
            not isinstance(execution, Mapping)
            or execution.get("schema") != "persistent-language-action-execution-v2"
            or execution.get("proposal_sha256") != prediction.proposal_sha256
            or execution.get("workspace_sha256") != observation.workspace.sha256
            or type(execution.get("child_pid")) is not int
            or execution["child_pid"] <= 0
            or "returncode" not in execution
            or (
                execution.get("returncode") is not None and type(execution["returncode"]) is not int
            )
        ):
            raise ValueError("actual execution identity or outcome is missing")
        for field in (
            "accepted",
            "timeout",
            "capture_complete",
            "files_complete",
            "output_limit_exceeded",
        ):
            if type(execution.get(field)) is not bool:
                raise ValueError("actual execution completeness must be explicit")
        for name in ("stdout", "stderr"):
            text = execution.get(name + "_hex")
            if not isinstance(text, str):
                raise ValueError("actual diagnostic bytes must retain canonical hex")
            raw = bytes.fromhex(text)
            observed = execution.get(name + "_observed_bytes")
            if (
                raw.hex() != text
                or len(raw) > MAX_STREAM_BYTES
                or hashlib.sha256(raw).hexdigest() != execution.get(name + "_sha256")
                or type(observed) is not int
                or observed < len(raw)
                or (observed > len(raw) and not execution["output_limit_exceeded"])
            ):
                raise ValueError("actual diagnostic bytes changed or were silently omitted")
        if execution["output_limit_exceeded"] and execution["capture_complete"]:
            raise ValueError("capped diagnostics cannot be complete")
        before = [
            {"path": item.path, "content_hex": item.content.hex()}
            for item in observation.workspace.files
        ]
        proposed = (
            apply_splice(observation.workspace, prediction.proposal)
            if execution["accepted"]
            else observation.workspace.files
        )
        if execution.get("before_files") != before or execution.get("proposed_files") != [
            {"path": item.path, "content_hex": item.content.hex()} for item in proposed
        ]:
            raise ValueError("execution applied different bytes than the pending action")

    @torch.no_grad()
    def teach(
        self,
        prediction,
        observation,
        state,
        teacher,
        execution_receipt,
        learning_rate=0.01,
        *,
        learning_enabled=True,
    ):
        if (
            type(learning_rate) not in (float, int)
            or learning_rate != LEARNING_RATE
            or type(learning_enabled) is not bool
        ):
            raise ValueError("local learning requires eta.01 and an explicit Boolean gate")
        work = {}
        self._check(work)
        self.reader._check_state(state)
        self._validate_execution(prediction, observation, state, execution_receipt, work)
        self.reader.validate_pending(prediction.payload, state, observation=observation, work=work)
        workspace = observation.workspace
        if not isinstance(teacher, SpliceProposal) or teacher.workspace_sha256 != workspace.sha256:
            raise ValueError("teacher must bind the actual pre-action workspace")
        apply_splice(workspace, teacher)
        encoded = encode(observation, self.reader.input_codes, work)
        updates = {}
        if teacher.operation == "SPLICE":
            target = workspace.boundaries.index((teacher.path, teacher.start))
            updates["start"] = self.actions.prepare_update("start", encoded, target, work=work)
            ends = end_offsets(encoded, teacher.path, teacher.start)
            updates["end"] = self.actions.prepare_update(
                "end",
                encoded,
                ends.index(teacher.end),
                path=teacher.path,
                start=teacher.start,
                work=work,
            )
            source = teacher
        else:
            source = prediction.proposal
        updates["operation"] = self.actions.prepare_update(
            "operation",
            encoded,
            int(teacher.operation == "SPLICE"),
            path=source.path,
            start=source.start,
            end=source.end,
            payload=source.replacement,
            stop_reason=source.stop_reason,
            work=work,
        )
        del encoded
        enabled = learning_enabled and self.mode not in {"action_frozen", "slow_frozen"}
        changed = {
            head: enabled and any(not torch.equal(old, new) for old, new in update.parameters)
            for head, update in updates.items()
        }
        for head in updates:
            _add(work, f"{head}_enabled_update_groups", int(enabled))
        feedback = SequenceAssemblySpliceFeedback(
            prediction.attempt_id,
            changed,
            {head: update.nll for head, update in updates.items()},
            {head: update.decision for head, update in updates.items()},
            None,
            {"action": work, "reader": {}},
        )
        # Every action tensor and receipt is already allocated and validated.
        # Reader teaching stages its own state; only plain field replacement and
        # infallible in-place action copies follow its atomic publication.
        if teacher.operation == "SPLICE":
            reader_feedback = self.reader.teach(
                observation,
                teacher.replacement,
                state,
                LEARNING_RATE,
                prediction=prediction.payload,
                learning_enabled=learning_enabled,
            )
            object.__setattr__(feedback, "reader_feedback", reader_feedback)
            feedback.work["reader"] = reader_feedback.work
        else:
            feedback.work["reader"] = self.reader.acknowledge(prediction.payload, state)
        if enabled:
            for update in updates.values():
                for parameter, proposal in update.parameters:
                    parameter.copy_(proposal)
        self._pending = None
        return feedback

    @torch.no_grad()
    def finish_without_teaching(self, prediction, observation, state, execution_receipt):
        work = {}
        self._check(work)
        self.reader._check_state(state)
        self._validate_execution(prediction, observation, state, execution_receipt, work)
        self.reader.validate_pending(prediction.payload, state, observation=observation, work=work)
        result = {"action": work, "reader": {}}
        result["reader"] = self.reader.acknowledge(prediction.payload, state)
        self._pending = None
        return result
