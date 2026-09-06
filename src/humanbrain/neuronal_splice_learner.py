"""Local output-neuron choices for an authenticated raw byte-splice tool."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn

from humanbrain.byte_sequence_decoder import BYTE_EOS
from humanbrain.local_memory_learner import (
    LocalMemoryLearner,
    SequenceLessonFeedback,
    SequencePrediction,
    _finite,
)
from humanbrain.neuronal_sequence_memory import TapeState
from humanbrain.neuronal_splice_action import (
    MAX_REPLACEMENT_BYTES,
    MAX_REQUEST_BYTES,
    SpliceProposal,
    SpliceWorkspace,
    apply_splice,
)

WIDTHS = {"start": 6177, "end": 2049, "operation": 4244}
LEARNING_RATE = 0.01


def _add(work: dict | None, name: str, value: int) -> None:
    if work is not None:
        work[name] = work.get(name, 0) + value


def _check_tensor(value: Tensor, shape: tuple, name: str) -> None:
    if (not isinstance(value, Tensor) or tuple(value.shape) != shape
            or value.dtype != torch.float32 or value.device.type != "cpu"
            or value.requires_grad or value.grad_fn is not None or value.grad is not None):
        raise ValueError(f"{name} must retain fixed CPU float32 non-autograd geometry")
    _finite(value, name)


def _rows(raw: bytes, size: int, codes: Tensor, work: dict | None,
          *, reverse: bool = False, eos: bool = True) -> Tensor:
    tokens = ([BYTE_EOS, *reversed(raw)] if reverse else [*raw, BYTE_EOS]) if eos else list(raw)
    if len(tokens) > size:
        raise ValueError("raw input exceeds its code population")
    result = torch.zeros(size, 32, dtype=torch.float32)
    if tokens:
        result[:len(tokens)].copy_(codes[torch.tensor(tokens, dtype=torch.long)])
    _add(work, "raw_bytes_encoded", len(raw))
    _add(work, "eos_codes_encoded", int(eos))
    _add(work, "code_rows_loaded", len(tokens))
    return result


@dataclass(frozen=True)
class SpliceEncoding:
    workspace: SpliceWorkspace
    codes: Tensor
    query: Tensor
    paths: tuple[Tensor, ...]
    files: tuple[Tensor, ...]


@torch.no_grad()
def encode(request: bytes, workspace: SpliceWorkspace, input_codes: Tensor,
           work: dict | None = None) -> SpliceEncoding:
    if not isinstance(request, bytes) or len(request) > MAX_REQUEST_BYTES:
        raise ValueError("request must contain at most 31 raw bytes")
    if not isinstance(workspace, SpliceWorkspace):
        raise TypeError("encoding requires a SpliceWorkspace")
    _check_tensor(input_codes, (258, 32), "input codes")
    query = _rows(request, 32, input_codes, work, reverse=True)
    paths = tuple(_rows(item.path.encode("utf-8"), 32, input_codes, work, reverse=True)
                  for item in workspace.files)
    files = tuple(_rows(item.content, 129, input_codes, work) for item in workspace.files)
    return SpliceEncoding(workspace, input_codes, query, paths, files)


def _file(encoding: SpliceEncoding, path: str):
    for index, item in enumerate(encoding.workspace.files):
        if item.path == path:
            return index, item
    raise ValueError("selected file is absent")


def _contexts(encoding: SpliceEncoding, content: bytes, offsets: tuple[int, ...],
              work: dict | None) -> Tensor:
    if not offsets or any(type(i) is not int or not 0 <= i <= len(content) for i in offsets):
        raise ValueError("boundary offsets must be actual file boundaries")
    positions = torch.tensor(offsets, dtype=torch.long)[:, None]
    indices = torch.cat((positions - torch.arange(1, 17), positions + torch.arange(16)), dim=1)
    valid = (indices >= 0) & (indices <= len(content))
    source = _rows(content, len(content) + 1, encoding.codes, work)
    result = source[indices.clamp(0, len(content))] * valid[:, :, None]
    _add(work, "boundary_contexts", len(offsets))
    _add(work, "context_code_gathers", len(offsets) * 32)
    return result


def _normalize(raw: Tensor, work: dict | None, head: str) -> Tensor:
    norms = torch.linalg.vector_norm(raw, dim=1, keepdim=True)
    _finite(norms, "whole feature norm")
    result = raw / norms.clamp_min(1e-8) * math.sqrt(32)
    _finite(result, "normalized action features")
    _add(work, f"{head}_feature_normalizations", raw.shape[0])
    _add(work, f"{head}_feature_components", raw.numel())
    return result


@torch.no_grad()
def start_features(encoding: SpliceEncoding, work: dict | None = None) -> Tensor:
    result = torch.empty(len(encoding.workspace.boundaries), WIDTHS["start"], dtype=torch.float32)
    first = 0
    for index, item in enumerate(encoding.workspace.files):
        count = len(item.content) + 1
        # These actual fixed forward activities are invariant within one file.
        path = (encoding.query @ encoding.paths[index].T).flatten()
        whole = (encoding.query @ encoding.files[index].T).flatten()
        contexts = _contexts(encoding, item.content, tuple(range(count)), work)
        local = torch.matmul(encoding.query[None], contexts.transpose(1, 2)).flatten(1)
        offset = torch.arange(count, dtype=torch.float32)[:, None] / 128
        raw = torch.cat((path.expand(count, -1), whole.expand(count, -1), local, offset), dim=1)
        result[first:first + count].copy_(_normalize(raw, work, "start"))
        _add(work, "start_feature_products", (1024 + 4128 + count * 1024) * 32)
        _add(work, "start_feature_channel_reductions", (1024 + 4128 + count * 1024) * 31)
        first += count
        del raw, local, contexts, path, whole
    return result


@torch.no_grad()
def end_features(encoding: SpliceEncoding, path: str, start: int,
                 work: dict | None = None) -> tuple[tuple[int, ...], Tensor]:
    _, item = _file(encoding, path)
    if type(start) is not int or not 0 <= start <= len(item.content):
        raise ValueError("start is not a file boundary")
    offsets = tuple(range(start, len(item.content) + 1))
    contexts = _contexts(encoding, item.content, offsets, work)
    selected = contexts[0]
    query = torch.matmul(encoding.query[None], contexts.transpose(1, 2)).flatten(1)
    relative = torch.matmul(selected[None], contexts.transpose(1, 2)).flatten(1)
    distance = (torch.tensor(offsets, dtype=torch.float32)[:, None] - start) / 128
    raw = torch.cat((query, relative, distance), dim=1)
    _add(work, "end_feature_products", len(offsets) * 2048 * 32)
    _add(work, "end_feature_channel_reductions", len(offsets) * 2048 * 31)
    return offsets, _normalize(raw, work, "end")


@torch.no_grad()
def operation_features(encoding: SpliceEncoding, path: str, start: int, end: int,
                       payload: bytes, stop_reason: str, work: dict | None = None) -> Tensor:
    index, item = _file(encoding, path)
    if (type(start) is not int or type(end) is not int
            or not 0 <= start <= end <= len(item.content)
            or not isinstance(payload, bytes) or len(payload) > MAX_REPLACEMENT_BYTES
            or stop_reason not in {"eos", "limit"}):
        raise ValueError("operation features require an actual bounded provisional span/payload")
    context = _contexts(encoding, item.content, (start,), work)[0]
    span = _rows(item.content[start:end], 129, encoding.codes, work)
    replacement = _rows(payload, 17, encoding.codes, work, eos=stop_reason == "eos")
    raw = torch.cat((
        (encoding.query @ context.T).flatten(),
        (encoding.query @ encoding.paths[index].T).flatten(),
        (span @ replacement.T).flatten(),
        torch.tensor([(end - start) / 128, len(payload) / 16, float(stop_reason == "eos")]),
    ))[None]
    _add(work, "operation_feature_products", (1024 + 1024 + 2193) * 32)
    _add(work, "operation_feature_channel_reductions", (1024 + 1024 + 2193) * 31)
    return _normalize(raw, work, "operation")


@dataclass(frozen=True)
class HeadDecision:
    logits: Tensor
    probabilities: Tensor
    selected_index: int


@dataclass(frozen=True)
class HeadUpdate:
    decision: HeadDecision
    target: int
    nll: float
    parameters: tuple[tuple[Tensor, Tensor], ...]


class SpliceActionCircuit(nn.Module):
    """Three direct output populations; fixed features receive no backward error."""

    def __init__(self, *, seed: int = 733) -> None:
        super().__init__()
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.seed = seed
        for head, offset in (("start", 5), ("end", 6), ("operation", 7)):
            outputs = 2 if head == "operation" else 1
            matrix = torch.empty(outputs, WIDTHS[head], dtype=torch.float32)
            generator = torch.Generator(device="cpu").manual_seed(seed + offset)
            nn.init.xavier_uniform_(matrix, generator=generator)
            self.register_parameter(f"{head}_weight", nn.Parameter(
                matrix if outputs == 2 else matrix[0], requires_grad=False))
        self.operation_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=False)

    def check_parameters(self) -> None:
        for head in WIDTHS:
            shape = (2, WIDTHS[head]) if head == "operation" else (WIDTHS[head],)
            _check_tensor(getattr(self, f"{head}_weight"), shape, f"{head} weight")
        _check_tensor(self.operation_bias, (2,), "operation bias")

    @torch.no_grad()
    def score(self, head: str, features: Tensor, work: dict | None = None) -> HeadDecision:
        self.check_parameters()
        if head not in WIDTHS or features.ndim != 2 or features.shape[0] < 1:
            raise ValueError("unknown head or empty feature population")
        _check_tensor(features, (features.shape[0], WIDTHS[head]), "action features")
        weight = getattr(self, f"{head}_weight")
        if head == "operation":
            if features.shape[0] != 1:
                raise ValueError("operation has one input and two output neurons")
            logits = (weight * features).sum(1) + self.operation_bias
            products, reductions = 2 * WIDTHS[head], 2 * (WIDTHS[head] - 1)
            _add(work, "operation_bias_additions", 2)
        else:
            logits = (features * weight).sum(1)
            products, reductions = features.numel(), features.shape[0] * (WIDTHS[head] - 1)
        _finite(logits, "action output logits")
        probabilities = torch.softmax(logits.to(torch.float64), dim=0)
        _finite(probabilities, "action output probabilities")
        _add(work, f"{head}_score_products", products)
        _add(work, f"{head}_score_reductions", reductions)
        _add(work, f"{head}_score_evaluations", logits.numel())
        _add(work, f"{head}_softmaxes", 1)
        return HeadDecision(logits, probabilities, int(probabilities.argmax()))

    @torch.no_grad()
    def prepare_update(self, head: str, features: Tensor, target: int,
                       work: dict | None = None) -> HeadUpdate:
        decision = self.score(head, features, work)
        if type(target) is not int or not 0 <= target < decision.probabilities.numel():
            raise ValueError("teacher must select an existing output neuron")
        error = decision.probabilities.clone()
        error[target] -= 1
        activity64 = features.to(torch.float64)
        if head == "operation":
            products = error[:, None] * activity64
            contribution = products
            pairs = [(self.operation_weight, contribution), (self.operation_bias, error)]
            product_count, reduction_count = products.numel(), 0
        else:
            products = error[:, None] * activity64
            contribution = products.sum(0)
            pairs = [(getattr(self, f"{head}_weight"), contribution)]
            product_count = products.numel()
            reduction_count = (features.shape[0] - 1) * features.shape[1]
        proposals = []
        for parameter, signal in pairs:
            _finite(signal, "action local contribution")
            cast = signal.to(torch.float32)
            _finite(cast, "cast action local contribution")
            proposal = parameter - LEARNING_RATE * cast
            _finite(proposal, "action parameter proposal")
            proposals.append((parameter, proposal))
        nll = float(-torch.log_softmax(decision.logits.to(torch.float64), dim=0)[target])
        _add(work, f"{head}_log_softmaxes", 1)
        if not math.isfinite(nll):
            raise ValueError("action teacher loss must be finite")
        _add(work, f"{head}_local_products", product_count)
        _add(work, f"{head}_local_reductions", reduction_count)
        _add(work, f"{head}_teacher_offers", 1)
        return HeadUpdate(decision, target, nll, tuple(proposals))


def _digest(value, work: dict | None = None) -> str:
    result = hashlib.sha256()

    def visit(item):
        if isinstance(item, Tensor):
            raw = item.detach().contiguous().numpy().tobytes()
            result.update(str((str(item.dtype), tuple(item.shape), len(raw))).encode())
            result.update(raw)
            _add(work, "custody_tensor_bytes_hashed", len(raw))
        elif isinstance(item, Mapping):
            for key in sorted(item):
                result.update(json.dumps(key).encode())
                visit(item[key])
        else:
            result.update(json.dumps(item, sort_keys=True, allow_nan=False).encode())
    visit(value)
    return result.hexdigest()


@dataclass(frozen=True)
class SplicePrediction:
    proposal: SpliceProposal
    attempt_id: str
    proposal_sha256: str
    model_sha256: str
    memory_sha256: str
    choices: dict[str, HeadDecision]
    work: dict
    payload: SequencePrediction


@dataclass(frozen=True)
class SpliceFeedback:
    attempt_id: str
    action_updates: dict[str, bool]
    action_losses: dict[str, float]
    teacher_choices: dict[str, HeadDecision]
    reader_feedback: SequenceLessonFeedback | None
    work: dict


class NeuronalSpliceLearner(nn.Module):
    """One reader/action model with one outstanding actual action at a time."""

    MODES: ClassVar[dict[str, str]] = {
        "learned": "learned", "action_frozen": "learned",
        "slow_frozen": "slow_frozen", "memory_disabled": "writes_disabled",
    }

    def __init__(self, *, mode: str = "learned", seed: int = 733,
                 reader: LocalMemoryLearner | None = None) -> None:
        super().__init__()
        if mode not in self.MODES or type(seed) is not int:
            raise ValueError("invalid splice learner mode or seed")
        self.mode, self.seed = mode, seed
        self.reader = reader if reader is not None else LocalMemoryLearner(
            mode=self.MODES[mode], seed=seed)
        if (not isinstance(self.reader, LocalMemoryLearner)
                or self.reader.mode != self.MODES[mode] or self.reader.seed != seed):
            raise ValueError("reader must have the selected seed and control configuration")
        self.actions = SpliceActionCircuit(seed=seed)
        self._experience = 0
        self._pending: tuple[SplicePrediction, bytes] | None = None
        self._check()

    def _check(self) -> None:
        if self.mode not in self.MODES or self.reader.mode != self.MODES[self.mode]:
            raise ValueError("splice control configuration changed")
        self.reader._check_parameters()
        self.actions.check_parameters()

    def get_extra_state(self) -> dict:
        return {"schema": "neuronal-byte-splice-learner-v1", "mode": self.mode,
                "seed": self.seed, "widths": dict(WIDTHS), "feature_energy": 32,
                "operation_augmented_energy": 33, "learning_rate": LEARNING_RATE,
                "payload_cap": 16, "publication": "validated action then atomic reader then copies"}

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("splice learner checkpoint configuration mismatch")

    @torch.no_grad()
    def ingest(self, documents: Sequence[bytes], state: TapeState | None = None) -> TapeState:
        if self._pending is not None:
            raise ValueError("resolve the pending action before ingesting another episode")
        self._check()
        return self.reader.ingest(documents, state)

    @torch.no_grad()
    def propose(
        self, request: bytes, workspace: SpliceWorkspace, state: TapeState,
    ) -> SplicePrediction:
        if self._pending is not None:
            raise ValueError("an action already awaits execution feedback")
        self._check()
        self.reader._check_state(state)
        work = {}
        encoded = encode(request, workspace, self.reader.input_codes, work)
        model_hash = _digest(self.state_dict(), work)
        memory_hash = _digest(state.snapshot(), work)
        features = start_features(encoded, work)
        start = self.actions.score("start", features, work)
        path, offset = workspace.boundaries[start.selected_index]
        del features
        ends, features = end_features(encoded, path, offset, work)
        end = self.actions.score("end", features, work)
        stop = ends[end.selected_index]
        del features
        payload = self.reader.predict_with_receipt(request, state, MAX_REPLACEMENT_BYTES)
        generated = payload.generation
        features = operation_features(encoded, path, offset, stop, generated.data,
                                      generated.stop_reason, work)
        operation = self.actions.score("operation", features, work)
        del features, encoded
        proposal = SpliceProposal(
            "STOP" if operation.selected_index == 0 else "SPLICE", path, offset, stop,
            generated.data, generated.stop_reason, workspace.sha256)
        if (model_hash != _digest(self.state_dict(), work)
                or memory_hash != _digest(state.snapshot(), work)):
            raise RuntimeError("free action changed model or retained memory")
        self._experience += 1
        prediction = SplicePrediction(
            proposal, f"{id(self)}:{self._experience}", proposal.sha256, model_hash, memory_hash,
            {"start": start, "end": end, "operation": operation},
            {"action": work, "reader": dict(payload.work)}, payload)
        self._pending = prediction, request
        return prediction

    def _validate_execution(self, prediction, workspace, state, execution_receipt, work):
        if self._pending is None or self._pending[0] is not prediction:
            raise ValueError("feedback must resolve this instance's pending prediction")
        if (not isinstance(workspace, SpliceWorkspace)
                or workspace.sha256 != prediction.proposal.workspace_sha256
                or prediction.proposal.sha256 != prediction.proposal_sha256
                or _digest(self.state_dict(), work) != prediction.model_sha256
                or _digest(state.snapshot(), work) != prediction.memory_sha256):
            raise ValueError("pending action model, memory or workspace changed")
        if (not isinstance(execution_receipt, Mapping)
                or set(execution_receipt) != {
                    "attempt_id", "proposal_sha256", "workspace_sha256", "execution"}
                or execution_receipt["attempt_id"] != prediction.attempt_id
                or execution_receipt["proposal_sha256"] != prediction.proposal_sha256
                or execution_receipt["workspace_sha256"] != workspace.sha256):
            raise ValueError("execution receipt does not bind the actual pending action")
        execution = execution_receipt["execution"]
        if (not isinstance(execution, Mapping)
                or not {"returncode", "stdout_hex", "stderr_hex"}.issubset(execution)
                or (execution["returncode"] is not None
                    and type(execution["returncode"]) is not int)):
            raise ValueError("actual execution receipt is missing its outcome")
        for field in ("stdout_hex", "stderr_hex"):
            raw = execution[field]
            if not isinstance(raw, str) or bytes.fromhex(raw).hex() != raw:
                raise ValueError("execution bytes must retain canonical hex")

    @torch.no_grad()
    def teach(self, prediction: SplicePrediction, workspace: SpliceWorkspace, state: TapeState,
              teacher: SpliceProposal, execution_receipt: Mapping, learning_rate: float = 0.01,
              *, learning_enabled: bool = True) -> SpliceFeedback:
        if (type(learning_rate) not in {float, int} or learning_rate != LEARNING_RATE
                or type(learning_enabled) is not bool):
            raise ValueError("local learning requires eta .01 and an explicit boolean gate")
        self._check()
        self.reader._check_state(state)
        work = {}
        self._validate_execution(prediction, workspace, state, execution_receipt, work)
        if not isinstance(teacher, SpliceProposal) or teacher.workspace_sha256 != workspace.sha256:
            raise ValueError("teacher must describe the authenticated pre-action workspace")
        # Tool validity is checked before any publication, never used to choose a free span.
        apply_splice(workspace, teacher)
        request = self._pending[1]
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
        changed = {head: enabled and any(not torch.equal(old, new)
                                        for old, new in update.parameters)
                   for head, update in updates.items()}
        for head in updates:
            _add(work, f"{head}_enabled_update_groups", int(enabled))
        # All possible ordinary action validation and allocation precede reader publication.
        feedback = SpliceFeedback(
            prediction.attempt_id, changed, {head: update.nll for head, update in updates.items()},
            {head: update.decision for head, update in updates.items()}, None,
            {"action": work, "reader": {}})
        if teacher.operation == "SPLICE":
            reader_feedback = self.reader.teach(
                request, teacher.replacement, state, LEARNING_RATE,
                learning_enabled=learning_enabled)
            object.__setattr__(feedback, "reader_feedback", reader_feedback)
            feedback.work["reader"] = reader_feedback.work
        if enabled:
            for update in updates.values():
                for parameter, proposal in update.parameters:
                    parameter.copy_(proposal)
        self._pending = None
        return feedback

    def finish_without_teaching(
        self, prediction: SplicePrediction, workspace: SpliceWorkspace, state: TapeState,
        execution_receipt: Mapping,
    ) -> dict:
        """Consume a completed evaluation attempt with no forward pass or teacher."""
        self._check()
        self.reader._check_state(state)
        work = {}
        self._validate_execution(prediction, workspace, state, execution_receipt, work)
        self._pending = None
        return {"action": work, "reader": {}}

    def snapshot(self, state: TapeState) -> dict:
        self._check()
        self.reader._check_state(state, populated=False)
        return {"model": copy.deepcopy(self.state_dict()), "memory": state.snapshot(),
                "experience": self._experience}

    def load_state_dict(self, state_dict: Mapping, strict: bool = True, assign: bool = False):
        if self._pending is not None or not strict or assign or not isinstance(state_dict, Mapping):
            raise ValueError("splice checkpoints require idle strict in-place loading")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("splice checkpoint fields mismatch")
        for name, prototype in expected.items():
            value = state_dict[name]
            if isinstance(prototype, Tensor):
                if (not isinstance(value, Tensor) or value.shape != prototype.shape
                        or value.dtype != prototype.dtype or value.device != prototype.device
                        or value.requires_grad or value.grad_fn is not None):
                    raise ValueError("splice checkpoint tensor contract mismatch")
                _finite(value, "checkpoint tensor")
                if name == "reader.input_codes" and not torch.equal(value, prototype):
                    raise ValueError("checkpoint changes fixed input codes")
            elif value != prototype:
                raise ValueError("splice checkpoint metadata mismatch")
        return super().load_state_dict(state_dict, strict=True, assign=False)

    def restore_snapshot(self, snapshot: dict) -> TapeState:
        if (not isinstance(snapshot, dict) or set(snapshot) != {"model", "memory", "experience"}
                or type(snapshot["experience"]) is not int or snapshot["experience"] < 0):
            raise ValueError("snapshot must bind model, memory and experience count")
        state = self.reader.memory.restore_state(snapshot["memory"])
        self.reader._check_state(state, populated=False)
        self.load_state_dict(snapshot["model"])
        # Restoring weights never reissues an earlier instance-owned attempt token.
        self._experience = max(self._experience, snapshot["experience"])
        return state
