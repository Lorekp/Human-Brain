"""Bind local payload correction to the same reader that actually chose an action."""

import torch

from humanbrain.active_path_memory_learner import ActivePathMemoryLearner
from humanbrain.neuronal_splice_action import SpliceProposal, apply_splice
from humanbrain.neuronal_splice_learner import (
    LEARNING_RATE,
    SpliceFeedback,
    _add,
    encode,
    end_features,
    operation_features,
    start_features,
)
from humanbrain.request_conditioned_splice_learner import RequestConditionedSpliceLearner


class ActivePathSpliceLearner(RequestConditionedSpliceLearner):
    """One pending action and its exact nested neuronal output response."""

    def __init__(self, *, mode="learned", seed=733, reader=None):
        if mode not in self.MODES:
            raise ValueError("invalid active-path splice mode")
        if reader is None:
            reader = ActivePathMemoryLearner(mode=self.MODES[mode], seed=seed)
        if not isinstance(reader, ActivePathMemoryLearner):
            raise TypeError("active-path actions require the same active-path reader")
        super().__init__(mode=mode, seed=seed, reader=reader)

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "active-path-splice-learner-v1",
            "payload_custody": "exact pending reader response; actual action before correction",
            "stop_teacher": "operation correction only; acknowledge reader with zero payload offers",
            "teacher_free": "resolve both action and reader responses without local updates",
        })
        return metadata

    def _require_idle(self):
        if self._pending is not None or self.reader.pending_prediction is not None:
            raise ValueError("resolve both action and reader responses before state replacement")

    def load_state_dict(self, state_dict, strict=True, assign=False):
        self._require_idle()
        return super().load_state_dict(state_dict, strict=strict, assign=assign)

    def snapshot(self, state):
        self._require_idle()
        return super().snapshot(state)

    @torch.no_grad()
    def teach(self, prediction, workspace, state, teacher, execution_receipt,
              learning_rate=0.01, *, learning_enabled=True):
        if (type(learning_rate) not in {float, int} or learning_rate != LEARNING_RATE
                or type(learning_enabled) is not bool):
            raise ValueError("local learning requires eta .01 and an explicit boolean gate")
        self._check()
        self.reader._check_state(state)
        work = {}
        self._validate_execution(prediction, workspace, state, execution_receipt, work)
        request = self._pending[1]
        reader_prevalidation = {}
        self.reader.validate_pending(
            prediction.payload, state, query=request, work=reader_prevalidation)
        if not isinstance(teacher, SpliceProposal) or teacher.workspace_sha256 != workspace.sha256:
            raise ValueError("teacher must describe the authenticated pre-action workspace")
        apply_splice(workspace, teacher)
        encoded = encode(request, workspace, self.reader.input_codes, work)
        updates = {}
        if teacher.operation == "SPLICE":
            features = start_features(encoded, work)
            target = workspace.boundaries.index((teacher.path, teacher.start))
            updates["start"] = self.actions.prepare_update("start", features, target, work)
            del features
            ends, features = end_features(encoded, teacher.path, teacher.start, work)
            updates["end"] = self.actions.prepare_update(
                "end", features, ends.index(teacher.end), work)
            del features
            operation_source = teacher
        else:
            operation_source = prediction.proposal
        features = operation_features(
            encoded, operation_source.path, operation_source.start, operation_source.end,
            operation_source.replacement, operation_source.stop_reason, work)
        updates["operation"] = self.actions.prepare_update(
            "operation", features, int(teacher.operation == "SPLICE"), work)
        del features, encoded
        enabled = learning_enabled and self.mode not in {"action_frozen", "slow_frozen"}
        changed = {
            head: enabled and any(not torch.equal(old, new) for old, new in update.parameters)
            for head, update in updates.items()
        }
        for head in updates:
            _add(work, f"{head}_enabled_update_groups", int(enabled))
        # Validate and allocate every action proposal before reader publication.
        feedback = SpliceFeedback(
            prediction.attempt_id, changed,
            {head: update.nll for head, update in updates.items()},
            {head: update.decision for head, update in updates.items()}, None,
            {"action": work, "reader": {}})
        if teacher.operation == "SPLICE":
            reader_feedback = self.reader.teach(
                request, teacher.replacement, state, LEARNING_RATE,
                prediction=prediction.payload, learning_enabled=learning_enabled)
            object.__setattr__(feedback, "reader_feedback", reader_feedback)
            feedback.work["reader"] = reader_feedback.work
        else:
            feedback.work["reader"] = self.reader.acknowledge(prediction.payload, state)
        for key, count in reader_prevalidation.items():
            _add(feedback.work["reader"], key, count)
        if enabled:
            for update in updates.values():
                for parameter, proposal in update.parameters:
                    parameter.copy_(proposal)
        self._pending = None
        return feedback

    def finish_without_teaching(self, prediction, workspace, state, execution_receipt):
        self._check()
        self.reader._check_state(state)
        work = {}
        self._validate_execution(prediction, workspace, state, execution_receipt, work)
        reader_prevalidation = {}
        self.reader.validate_pending(
            prediction.payload, state, query=self._pending[1], work=reader_prevalidation)
        reader_work = self.reader.acknowledge(prediction.payload, state)
        for key, count in reader_prevalidation.items():
            _add(reader_work, key, count)
        self._pending = None
        return {"action": work, "reader": reader_work}
