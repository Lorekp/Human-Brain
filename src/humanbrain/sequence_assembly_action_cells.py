"""Long raw-observation action heads with bounded direct-synapse simulation.

All candidate features exist before their learned output products. Teaching
scores all outputs, then evaluates the same fixed features a second time for
their own local error products. No full candidate-by-feature bank is retained.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from humanbrain.neuronal_splice_learner import (
    LEARNING_RATE,
    HeadDecision,
    HeadUpdate,
    _add,
    _check_tensor,
    _finite,
    _normalize,
    _rows,
)
from humanbrain.neuronal_splice_learner import WIDTHS as LEGACY_WIDTHS
from humanbrain.sequence_assembly_observation import (
    FILE_SLOTS,
    MAX_REPLACEMENT_BYTES,
    PATH_SLOTS,
    QUERY_FLAG_NAMES,
    QUERY_SYMBOLS,
    RawObservation,
)

QUERY_ROWS = QUERY_SYMBOLS + len(QUERY_FLAG_NAMES)
PAYLOAD_ROWS = MAX_REPLACEMENT_BYTES + 1
BLOCK_SIZE = 16
WIDTHS = {
    "start": QUERY_ROWS * (PATH_SLOTS + FILE_SLOTS + 32) + 1,
    "end": QUERY_ROWS * 32 + 1024 + 1,
    "operation": QUERY_ROWS * (32 + PATH_SLOTS) + FILE_SLOTS * PAYLOAD_ROWS + 3,
}


@dataclass(frozen=True)
class ActionEncoding:
    observation: RawObservation
    codes: Tensor
    query: Tensor
    paths: tuple[Tensor, ...]
    files: tuple[Tensor, ...]

    @property
    def workspace(self):
        return self.observation.workspace


@torch.no_grad()
def encode(observation: RawObservation, input_codes: Tensor, work=None) -> ActionEncoding:
    """Frame every raw request/tool field; flags have their own byte0/1 rows."""
    if not isinstance(observation, RawObservation) or observation.workspace is None:
        raise ValueError("action encoding requires a whole raw workspace observation")
    _check_tensor(input_codes, (258, 32), "input codes")
    frame = observation.query_frame()
    symbols = list(frame.symbols)
    # Preserve the old request-coordinate direction inside the expanded field.
    # Other typed segments keep the exact public sensor-frame order.
    length = len(observation.request) + 1
    symbols[:length] = reversed(symbols[:length])
    symbols.extend(int(flag) for flag in frame.flags)
    if len(symbols) != QUERY_ROWS or len(frame.flags) != len(QUERY_FLAG_NAMES):
        raise ValueError("action observation framing width changed")
    tokens = torch.tensor(symbols, dtype=torch.int64)
    present = tokens >= 0
    query = input_codes.index_select(0, tokens.clamp_min(0))
    query.mul_(present[:, None])
    paths = tuple(_rows(item.path.encode("utf-8"), PATH_SLOTS, input_codes, work, reverse=True)
                  for item in observation.workspace.files)
    files = tuple(_rows(item.content, FILE_SLOTS, input_codes, work)
                  for item in observation.workspace.files)
    for name, value in {
        "action_query_code_gathers": QUERY_ROWS,
        "action_query_mask_products": QUERY_ROWS * 32,
        "action_query_status_code_rows": len(QUERY_FLAG_NAMES),
        "action_query_present_rows": int(present.sum()),
        "action_query_named_allocated_bytes": QUERY_ROWS * (8 + 1 + 8 + 128),
    }.items():
        _add(work, name, value)
    return ActionEncoding(observation, input_codes, query, paths, files)


def _file(encoding, path):
    for index, item in enumerate(encoding.workspace.files):
        if item.path == path:
            return index, item
    raise ValueError("selected file is absent")


def end_offsets(encoding, path, start):
    _, item = _file(encoding, path)
    if type(start) is not int or not 0 <= start <= len(item.content):
        raise ValueError("start must be a real file boundary")
    return tuple(range(start, len(item.content) + 1))


def _contexts(encoding, index, offsets, work):
    content = encoding.workspace.files[index].content
    positions = torch.tensor(offsets, dtype=torch.int64)[:, None]
    indices = torch.cat((positions - torch.arange(1, 17), positions + torch.arange(16)), dim=1)
    valid = (indices >= 0) & (indices <= len(content))
    result = encoding.files[index][indices.clamp(0, len(content))] * valid[:, :, None]
    _add(work, "boundary_contexts", len(offsets))
    _add(work, "context_code_gathers", len(offsets) * 32)
    return result


def _feature_work(work, head, components):
    _add(work, f"{head}_feature_products", components * 32)
    _add(work, f"{head}_feature_channel_reductions", components * 31)


def feature_blocks(head, encoding, *, path=None, start=None, end=None,
                   payload=b"", stop_reason="eos", work=None):
    """Yield full normalized feature vectors in canonical candidate order.

    Must be consumed inside a no-grad caller. Each pass evaluates all candidates;
    file-invariant path/whole activities are shared only within that pass.
    """
    if not isinstance(encoding, ActionEncoding):
        raise TypeError("action features require a complete ActionEncoding")
    query = encoding.query
    if head == "start":
        for index, item in enumerate(encoding.workspace.files):
            path_activity = (query @ encoding.paths[index].T).flatten()
            whole_activity = (query @ encoding.files[index].T).flatten()
            _feature_work(work, head, QUERY_ROWS * (PATH_SLOTS + FILE_SLOTS))
            for first in range(0, len(item.content) + 1, BLOCK_SIZE):
                offsets = tuple(range(first, min(first + BLOCK_SIZE, len(item.content) + 1)))
                contexts = _contexts(encoding, index, offsets, work)
                local = torch.matmul(query[None], contexts.transpose(1, 2)).flatten(1)
                scalar = torch.tensor(offsets, dtype=torch.float32)[:, None] / 128
                raw = torch.cat((path_activity.expand(len(offsets), -1),
                                 whole_activity.expand(len(offsets), -1), local, scalar), dim=1)
                features = _normalize(raw, work, head)
                _feature_work(work, head, len(offsets) * QUERY_ROWS * 32)
                del contexts, local, scalar, raw
                _add(work, f"{head}_feature_blocks", 1)
                _add(work, f"{head}_feature_block_bytes", features.numel() * 4)
                yield features
                del features
    elif head == "end":
        index, _ = _file(encoding, path)
        offsets = end_offsets(encoding, path, start)
        selected = _contexts(encoding, index, (start,), work)[0]
        for first in range(0, len(offsets), BLOCK_SIZE):
            block_offsets = offsets[first:first + BLOCK_SIZE]
            contexts = _contexts(encoding, index, block_offsets, work)
            q_activity = torch.matmul(query[None], contexts.transpose(1, 2)).flatten(1)
            relative = torch.matmul(selected[None], contexts.transpose(1, 2)).flatten(1)
            distance = (torch.tensor(block_offsets, dtype=torch.float32)[:, None] - start) / 128
            raw = torch.cat((q_activity, relative, distance), dim=1)
            features = _normalize(raw, work, head)
            _feature_work(work, head, len(block_offsets) * (QUERY_ROWS * 32 + 1024))
            del contexts, q_activity, relative, distance, raw
            _add(work, f"{head}_feature_blocks", 1)
            _add(work, f"{head}_feature_block_bytes", features.numel() * 4)
            yield features
            del features
    elif head == "operation":
        index, item = _file(encoding, path)
        if (type(start) is not int or type(end) is not int
                or not 0 <= start <= end <= len(item.content)
                or not isinstance(payload, bytes) or len(payload) > MAX_REPLACEMENT_BYTES
                or stop_reason not in ("eos", "limit")):
            raise ValueError("operation requires an actual bounded provisional span/payload")
        context = _contexts(encoding, index, (start,), work)[0]
        span = _rows(item.content[start:end], FILE_SLOTS, encoding.codes, work)
        replacement = _rows(payload, PAYLOAD_ROWS, encoding.codes, work, eos=stop_reason == "eos")
        raw = torch.cat((
            (query @ context.T).flatten(),
            (query @ encoding.paths[index].T).flatten(),
            (span @ replacement.T).flatten(),
            torch.tensor([(end - start) / 128, len(payload) / 16, float(stop_reason == "eos")]),
        ))[None]
        features = _normalize(raw, work, head)
        _feature_work(work, head, QUERY_ROWS * (32 + PATH_SLOTS) + FILE_SLOTS * PAYLOAD_ROWS)
        del context, span, replacement, raw
        _add(work, f"{head}_feature_blocks", 1)
        _add(work, f"{head}_feature_block_bytes", features.numel() * 4)
        yield features
    else:
        raise ValueError("unknown action head")


def _embedded(head, value):
    """Map old two-dimensional coordinates; never assume flattened strides match."""
    if head == "start":
        result = torch.zeros(WIDTHS[head], dtype=torch.float32)
        result[:QUERY_ROWS * 32].reshape(QUERY_ROWS, 32)[:32].copy_(value[:1024].reshape(32, 32))
        first = QUERY_ROWS * 32
        whole = result[first:first + QUERY_ROWS * FILE_SLOTS].reshape(QUERY_ROWS, FILE_SLOTS)
        whole[:32, :129].copy_(value[1024:5152].reshape(32, 129))
        first += QUERY_ROWS * FILE_SLOTS
        result[first:first + QUERY_ROWS * 32].reshape(QUERY_ROWS, 32)[:32].copy_(
            value[5152:6176].reshape(32, 32))
        result[-1].copy_(value[-1])
    elif head == "end":
        result = torch.zeros(WIDTHS[head], dtype=torch.float32)
        result[:QUERY_ROWS * 32].reshape(QUERY_ROWS, 32)[:32].copy_(value[:1024].reshape(32, 32))
        result[QUERY_ROWS * 32:-1].copy_(value[1024:-1])
        result[-1].copy_(value[-1])
    else:
        result = torch.zeros(2, WIDTHS[head], dtype=torch.float32)
        first = 0
        for old_first in (0, 1024):
            result[:, first:first + QUERY_ROWS * 32].reshape(2, QUERY_ROWS, 32)[:, :32].copy_(
                value[:, old_first:old_first + 1024].reshape(2, 32, 32))
            first += QUERY_ROWS * 32
        result[:, first:-3].reshape(2, FILE_SLOTS, PAYLOAD_ROWS)[:, :129, :17].copy_(
            value[:, 2048:-3].reshape(2, 129, 17))
        result[:, -3:].copy_(value[:, -3:])
    return result


class SequenceAssemblyActionCircuit(nn.Module):
    """The same direct output errors, with extended coordinates and bounded passes."""

    def __init__(self, *, seed=733):
        super().__init__()
        if type(seed) is not int:
            raise ValueError("action seed must be an integer")
        self.seed = seed
        for head, offset in (("start", 5), ("end", 6), ("operation", 7)):
            outputs = 2 if head == "operation" else 1
            legacy = torch.empty(outputs, LEGACY_WIDTHS[head], dtype=torch.float32)
            generator = torch.Generator(device="cpu").manual_seed(seed + offset)
            nn.init.xavier_uniform_(legacy, generator=generator)
            value = _embedded(head, legacy if outputs == 2 else legacy[0])
            self.register_parameter(head + "_weight", nn.Parameter(value, requires_grad=False))
        self.operation_bias = nn.Parameter(torch.zeros(2, dtype=torch.float32), requires_grad=False)

    def get_extra_state(self):
        return {
            "schema": "sequence-assembly-action-circuit-v1", "seed": self.seed,
            "widths": dict(WIDTHS), "query_rows": QUERY_ROWS, "block_size": BLOCK_SIZE,
            "status_codes": list(QUERY_FLAG_NAMES), "status_encoding": "separate byte0/1 code rows",
            "request_direction": "reverse actual request plus EOS within first257 rows",
            "local_teacher": "two feature passes; float64 block accumulation; cast once; eta.01",
            "coordinates": "old coordinates embedded; new coordinates zero; no float-bit parity",
            "offset_divisor": 128, "payload_length_divisor": 16, "feature_energy": 32,
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("extended action configuration mismatch")

    def check_parameters(self):
        for head, width in WIDTHS.items():
            shape = (2, width) if head == "operation" else (width,)
            _check_tensor(getattr(self, head + "_weight"), shape, head + " weight")
        _check_tensor(self.operation_bias, (2,), "operation bias")

    @torch.no_grad()
    def load_legacy_state(self, state):
        names = {head + "_weight" for head in WIDTHS} | {"operation_bias"}
        if not isinstance(state, Mapping) or set(state) != names:
            raise ValueError("legacy action state requires the exact four numeric tensors")
        proposals = {}
        for head in WIDTHS:
            name = head + "_weight"
            shape = (2, LEGACY_WIDTHS[head]) if head == "operation" else (LEGACY_WIDTHS[head],)
            _check_tensor(state[name], shape, name)
            proposals[name] = _embedded(head, state[name])
        _check_tensor(state["operation_bias"], (2,), "legacy operation bias")
        proposals["operation_bias"] = state["operation_bias"].clone()
        self.check_parameters()
        for name, value in proposals.items():
            getattr(self, name).copy_(value)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not strict or assign or not isinstance(state_dict, Mapping):
            raise ValueError("extended action loading requires strict in-place state")
        expected = self.state_dict()
        if set(state_dict) != set(expected):
            raise ValueError("extended action state fields mismatch")
        for name, prototype in expected.items():
            if isinstance(prototype, Tensor):
                _check_tensor(state_dict[name], tuple(prototype.shape), name)
            elif state_dict[name] != prototype:
                raise ValueError("extended action metadata mismatch")
        return super().load_state_dict(state_dict, strict=True, assign=False)

    @torch.no_grad()
    def score(self, head, encoding, *, work=None, **coordinates):
        self.check_parameters()
        if head not in WIDTHS:
            raise ValueError("unknown action head")
        weight, blocks = getattr(self, head + "_weight"), []
        for features in feature_blocks(head, encoding, work=work, **coordinates):
            if head == "operation":
                logits = (weight * features).sum(1) + self.operation_bias
                products, reductions = 2 * WIDTHS[head], 2 * (WIDTHS[head] - 1)
                _add(work, "operation_bias_additions", 2)
            else:
                logits = (features * weight).sum(1)
                products = features.numel()
                reductions = features.shape[0] * (features.shape[1] - 1)
            _finite(logits, "extended action logits")
            blocks.append(logits)
            _add(work, f"{head}_score_products", products)
            _add(work, f"{head}_score_reductions", reductions)
            del features
        logits = torch.cat(blocks)
        probabilities = torch.softmax(logits.to(torch.float64), dim=0)
        _finite(probabilities, "extended action probabilities")
        _add(work, f"{head}_score_evaluations", logits.numel())
        _add(work, f"{head}_softmaxes", 1)
        return HeadDecision(logits, probabilities, int(probabilities.argmax()))

    @torch.no_grad()
    def prepare_update(self, head, encoding, target, *, work=None, **coordinates):
        decision = self.score(head, encoding, work=work, **coordinates)
        if type(target) is not int or not 0 <= target < decision.probabilities.numel():
            raise ValueError("teacher must select an existing output")
        error = decision.probabilities.clone()
        error[target] -= 1
        shape = (2, WIDTHS[head]) if head == "operation" else (WIDTHS[head],)
        contribution = torch.zeros(shape, dtype=torch.float64)
        first = 0
        for features in feature_blocks(head, encoding, work=work, **coordinates):
            activity = features.to(torch.float64)
            if head == "operation":
                products = error[:, None] * activity
                contribution.copy_(products)
                reductions = 0
            else:
                count = features.shape[0]
                products = error[first:first + count, None] * activity
                reduced = products.sum(0)
                contribution.add_(reduced)
                reductions = (count - 1) * WIDTHS[head]
                _add(work, f"{head}_local_accumulator_additions", WIDTHS[head])
                del reduced
                first += count
            _add(work, f"{head}_local_products", products.numel())
            _add(work, f"{head}_local_reductions", reductions)
            _add(work, f"{head}_local_activity_cast_values", activity.numel())
            block_bytes = activity.numel() * 8 + products.numel() * 8
            _add(work, f"{head}_local_named_block_bytes", block_bytes)
            del activity, products, features
        _finite(contribution, "extended action local contribution")
        pairs = [(getattr(self, head + "_weight"), contribution)]
        if head == "operation":
            pairs.append((self.operation_bias, error))
        proposals = []
        for parameter, signal in pairs:
            cast = signal.to(torch.float32)
            _finite(cast, "extended action cast local contribution")
            proposal = parameter - LEARNING_RATE * cast
            _finite(proposal, "extended action parameter proposal")
            proposals.append((parameter, proposal))
        nll = float(-torch.log_softmax(decision.logits.to(torch.float64), dim=0)[target])
        if not math.isfinite(nll):
            raise ValueError("extended action loss must be finite")
        _add(work, f"{head}_log_softmaxes", 1)
        _add(work, f"{head}_teacher_offers", 1)
        return HeadUpdate(decision, target, nll, tuple(proposals))

    def accounting(self):
        return {
            "schema": "sequence-assembly-action-accounting-v1", "widths": dict(WIDTHS),
            "query_rows": QUERY_ROWS, "block_size": BLOCK_SIZE,
            "model_tensor_bytes": sum(p.numel() * p.element_size() for p in self.parameters()),
            "maximum_start_block_bytes": BLOCK_SIZE * WIDTHS["start"] * 4,
            "maximum_start_local_activity_bytes": BLOCK_SIZE * WIDTHS["start"] * 8,
            "maximum_start_local_product_bytes": BLOCK_SIZE * WIDTHS["start"] * 8,
            "maximum_start_accumulator_bytes": WIDTHS["start"] * 8,
            "teacher_passes": 2,
            "scope": "named arrays only; normalization scratch, copies and process RSS separate",
        }
