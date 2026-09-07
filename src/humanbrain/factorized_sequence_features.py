"""Fixed byte/field populations for all-owner local sequence routing.

Only canonical receptive bits enter this module. Numeric histogram pooling is
fixed encoding, not a semantic parser or a candidate search. Sparse columns
represent fixed zero population entries; no occupied owner is omitted.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import Tensor

from humanbrain.neuronal_byte_coincidence import validate_symbol_rails
from humanbrain.plastic_memory import _all_finite

SYMBOLS = 258
CHANNEL_WIDTHS = (SYMBOLS, SYMBOLS * SYMBOLS, 8 * SYMBOLS)
FIELD_WIDTH = sum(CHANNEL_WIDTHS)
FIELD_NAMES = (
    "request", "path_0", "path_1", "path_2", "path_3",
    "file_0", "file_1", "file_2", "file_3", "stdout", "stderr",
    "returncode", "prefix", "memory_preceding", "memory_following",
)
LEXICAL_WIDTH = len(FIELD_NAMES) * FIELD_WIDTH
FEATURE_WIDTH = LEXICAL_WIDTH + 14
INPUT_WIDTH = 66093
SENSOR_SYMBOLS = 3471
PREFIX_SLOTS = 129
MAX_SPARSE_ENTRIES = 10992
HALF_GAIN = 1 / math.sqrt(2)
WORK_KEYS = (
    "factorized_decode_calls", "factorized_decode_symbol_rows",
    "factorized_decode_bit_products", "factorized_decode_bit_reductions",
    "factorized_raw_bytes_hashed", "factorized_field_tokens",
    "factorized_pair_tokens", "factorized_position_bins",
    "factorized_histogram_increments", "factorized_active_histogram_bins",
    "factorized_channel_norm_products", "factorized_channel_norm_reductions",
    "factorized_channel_square_roots", "factorized_channel_divisions",
    "factorized_candidate_builds", "factorized_context_builds",
    "factorized_sparse_entries", "factorized_dense_population_values",
    "factorized_dense_population_writes", "factorized_validation_values",
    "factorized_validation_norm_products", "factorized_validation_norm_reductions",
    "factorized_validation_cast_values",
    "factorized_canonical_rebuilds", "factorized_activity_calls",
    "factorized_population_gathers", "factorized_activity_cast_values",
    "factorized_coincidence_products", "factorized_baseline_reductions",
    "factorized_baseline_divisions", "factorized_flag_subtractions",
    "factorized_position_subtractions", "factorized_position_divisions",
    "factorized_position_absolute_values", "factorized_block_norm_products",
    "factorized_block_norm_reductions", "factorized_block_square_roots",
    "factorized_block_divisions", "factorized_half_energy_scalings",
    "factorized_energy_products", "factorized_energy_reductions",
    "factorized_named_tensor_bytes",
)


def _add(work, name, value):
    if work is not None:
        work[name] = work.get(name, 0) + value


def _named(value, work):
    """Count this explicitly created tensor once, not RSS or temporary liveness."""
    _add(work, "factorized_named_tensor_bytes", value.numel() * value.element_size())
    return value


def _tensor(value, dtype, shape, name):
    if (not isinstance(value, Tensor) or value.dtype != dtype or tuple(value.shape) != shape
            or value.device.type != "cpu" or value.requires_grad
            or value.grad_fn is not None or value.grad is not None):
        raise ValueError(f"{name} requires detached CPU {dtype} {shape}")


def _finite(value, name):
    if not _all_finite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class CandidateView:
    indices: Tensor
    values: Tensor
    flags: Tensor
    position: int
    raw_context_sha256: str


@dataclass(frozen=True)
class FactorizedContext:
    populations: Tensor
    flags: Tensor
    position: int
    raw_context_sha256: str


@dataclass(frozen=True)
class FactorizedActivity:
    indices: Tensor
    values: Tensor
    baseline: float
    energy: float


def _padded(raw, *, special=None):
    """Check a packed field; never connect pairs across absent padding."""
    end = raw.index(-1) if -1 in raw else len(raw)
    if any(token != -1 for token in raw[end:]):
        raise ValueError("field padding must be a suffix")
    active = raw[:end]
    if special == "eos":
        if not active or active[-1] != 256 or any(not 0 <= x < 256 for x in active[:-1]):
            raise ValueError("present raw fields require exactly one terminal EOS")
    elif any(not 0 <= x < 256 for x in active):
        raise ValueError("raw field contains a special symbol")
    return active


def _raw_field(raw, present):
    if not present:
        if any(token != -1 for token in raw):
            raise ValueError("absent fields must contain only padding")
        return []
    return _padded(raw, special="eos")


@torch.no_grad()
def decode_context(pattern: Tensor, work=None):
    """Return 15 ordered symbol fields, owned flags, position and raw identity.

    Memory preceding rows retain newest-first order. Following rows retain
    successor order and repeated absorbing EOS. No BOS/EOS is synthesized.
    """
    _tensor(pattern, torch.bool, (INPUT_WIDTH,), "receptive pattern")
    sensor_end = SENSOR_SYMBOLS * 18
    prefix_start = sensor_end + 12
    context_start = prefix_start + PREFIX_SLOTS * 18
    position_start = context_start + 64 * 18
    rails = _named(torch.cat((
        pattern[:sensor_end], pattern[prefix_start:position_start],
    )).reshape(-1, 18), work)
    validate_symbol_rails(rails, work=work)
    occupied = _named(rails.any(dim=1), work)
    ones = _named(rails[:, 1::2].to(torch.int64), work)
    weights = _named(torch.tensor([1 << bit for bit in range(9)], dtype=torch.int64), work)
    products = _named(ones * weights, work)
    tokens = _named(products.sum(dim=1), work)
    tokens.masked_fill_(~occupied, -1)
    raw = tokens.tolist()
    del rails, occupied, ones, weights, products, tokens
    flags = _named(pattern[sensor_end:prefix_start].clone(), work)
    status = flags.tolist()
    position_bits = pattern[position_start:]
    if int(position_bits.sum().item()) != 1:
        raise ValueError("context requires exactly one output position")
    position = int(position_bits.nonzero()[0, 0].item())
    if status[0] != any(status[1:5]) or status[1:5] != sorted(status[1:5], reverse=True):
        raise ValueError("workspace flags require a contiguous nonempty file prefix")
    if not status[5] and any(status[6:]):
        raise ValueError("absent feedback cannot carry outcome flags")
    request = _raw_field(raw[:257], True)
    stdout = _raw_field(raw[257:770], status[5])
    stderr = _raw_field(raw[770:1283], status[5])
    returncode = raw[1283:1291]
    if status[6]:
        if any(not 0 <= x < 256 for x in returncode):
            raise ValueError("present return code requires eight raw bytes")
    elif any(x != -1 for x in returncode):
        raise ValueError("absent return code must be padding")
    else:
        returncode = []
    paths, files = [], []
    for index in range(4):
        start = 1291 + index * 545
        paths.append(_raw_field(raw[start:start + 32], status[index + 1]))
        files.append(_raw_field(raw[start + 32:start + 545], status[index + 1]))
    prefix = raw[SENSOR_SYMBOLS:SENSOR_SYMBOLS + PREFIX_SLOTS]
    if (prefix[0] != 257 or any(not 0 <= x < 256 for x in prefix[1:position + 1])
            or any(x != -1 for x in prefix[position + 1:])):
        raise ValueError("prefix must be BOS, exactly earlier raw bytes, then padding")
    prefix = prefix[:position + 1]
    memory = raw[SENSOR_SYMBOLS + PREFIX_SLOTS:]
    preceding = _padded(memory[:32])
    following = memory[32:]
    if all(x == -1 for x in following):
        following = []
    elif any(not 0 <= x <= 256 for x in following):
        raise ValueError("following context requires raw/EOS or entirely empty activity")
    elif 256 in following and any(x != 256 for x in following[following.index(256):]):
        raise ValueError("following context must preserve absorbing EOS")
    fields = (request, *paths, *files, stdout, stderr, returncode, prefix, preceding, following)
    identity = hashlib.sha256(pattern.contiguous().numpy().tobytes()).hexdigest()
    _add(work, "factorized_decode_calls", 1)
    _add(work, "factorized_decode_symbol_rows", 3664)
    _add(work, "factorized_decode_bit_products", 3664 * 9)
    _add(work, "factorized_decode_bit_reductions", 3664 * 8)
    _add(work, "factorized_raw_bytes_hashed", INPUT_WIDTH)
    return fields, flags, position, identity


def _populations(fields, work):
    """Exact integer pooling of fixed numeric columns; normalize after pooling."""
    columns, values = [], []
    for field, tokens in enumerate(fields):
        length = len(tokens)
        if length > 513:
            raise ValueError("field exceeds the selected symbol population")
        pairs = [SYMBOLS * a + b for a, b in pairwise(tokens)]
        bins = [min(7, 8 * i // length) * SYMBOLS + token for i, token in enumerate(tokens)]
        offset = field * FIELD_WIDTH
        for activity_ids, width in zip((tokens, pairs, bins), CHANNEL_WIDTHS, strict=True):
            histogram = Counter(activity_ids)
            if histogram:
                # Integer counts remain exact; only normalized output rates are f32.
                norm = math.sqrt(sum(count * count for count in histogram.values()))
                for index in sorted(histogram):
                    columns.append(offset + index)
                    values.append(histogram[index] / norm)
                count = len(histogram)
                _add(work, "factorized_active_histogram_bins", count)
                _add(work, "factorized_channel_norm_products", count)
                _add(work, "factorized_channel_norm_reductions", max(count - 1, 0))
                _add(work, "factorized_channel_square_roots", 1)
                _add(work, "factorized_channel_divisions", count)
            _add(work, "factorized_histogram_increments", len(activity_ids))
            offset += width
        _add(work, "factorized_field_tokens", length)
        _add(work, "factorized_pair_tokens", len(pairs))
        _add(work, "factorized_position_bins", length)
    if len(columns) > MAX_SPARSE_ENTRIES:
        raise ValueError("sparse population exceeds the fixed geometry")
    indices = _named(torch.tensor(columns, dtype=torch.int32), work)
    rates = _named(torch.tensor(values, dtype=torch.float32), work)
    _add(work, "factorized_sparse_entries", len(columns))
    return indices, rates


@torch.no_grad()
def build_candidate_view(pattern: Tensor, work=None) -> CandidateView:
    fields, flags, position, identity = decode_context(pattern, work)
    indices, values = _populations(fields, work)
    view = CandidateView(indices, values, flags, position, identity)
    check_candidate_view(view, work)
    _add(work, "factorized_candidate_builds", 1)
    return view


@torch.no_grad()
def build_context(pattern: Tensor, work=None) -> FactorizedContext:
    fields, flags, position, identity = decode_context(pattern, work)
    indices, values = _populations(fields, work)
    populations = _named(torch.zeros(FEATURE_WIDTH, dtype=torch.float32), work)
    population_indices = _named(indices.to(torch.int64), work)
    populations.index_copy_(0, population_indices, values)
    context = FactorizedContext(populations, flags, position, identity)
    validate_context(context, work)
    _add(work, "factorized_context_builds", 1)
    _add(work, "factorized_dense_population_values", FEATURE_WIDTH)
    _add(work, "factorized_dense_population_writes", indices.numel())
    return context


def _metadata(value):
    _tensor(value.flags, torch.bool, (12,), "context flags")
    if type(value.position) is not int or not 0 <= value.position <= 128:
        raise ValueError("context position must be0..128")
    identity = value.raw_context_sha256
    if (not isinstance(identity, str) or len(identity) != 64
            or any(c not in "0123456789abcdef" for c in identity)):
        raise ValueError("context identity must be a lowercase SHA256")


def _check_context(context):
    if not isinstance(context, FactorizedContext):
        raise ValueError("current activity must be a FactorizedContext")
    _tensor(context.populations, torch.float32, (FEATURE_WIDTH,), "current population")
    _metadata(context)


@torch.no_grad()
def validate_context(context: FactorizedContext, work=None) -> None:
    """Once per public router call, not once per occupied owner."""
    _check_context(context)
    _finite(context.populations, "current population")
    if (bool((context.populations < 0).any())
            or bool(context.populations[LEXICAL_WIDTH:].ne(0).any())):
        raise ValueError("current population requires nonnegative lexical rates and zero tail")
    for field in range(15):
        offset = field * FIELD_WIDTH
        for width in CHANNEL_WIDTHS:
            channel = _named(context.populations[offset:offset + width].to(torch.float64), work)
            squared = _named(channel.square(), work)
            energy = float(squared.sum().item())
            if energy != 0 and not math.isclose(energy, 1, rel_tol=2e-7, abs_tol=2e-7):
                raise ValueError("current channel must have unit or zero energy")
            offset += width
            _add(work, "factorized_validation_cast_values", width)
            _add(work, "factorized_validation_norm_products", width)
            _add(work, "factorized_validation_norm_reductions", width - 1)
            del channel, squared
    _add(work, "factorized_validation_values", FEATURE_WIDTH)


@torch.no_grad()
def check_candidate_view(view: CandidateView, work=None) -> None:
    """Structural read guard; installation additionally binds the raw pattern."""
    if not isinstance(view, CandidateView) or not isinstance(view.indices, Tensor):
        raise ValueError("candidate must be a sparse CandidateView")
    count = view.indices.numel()
    if not 0 <= count <= MAX_SPARSE_ENTRIES:
        raise ValueError("candidate exceeds the sparse bound")
    _tensor(view.indices, torch.int32, (count,), "candidate columns")
    _tensor(view.values, torch.float32, (count,), "candidate rates")
    _metadata(view)
    _finite(view.values, "candidate rates")
    if (bool((view.indices < 0).any()) or bool((view.indices >= LEXICAL_WIDTH).any())
            or bool((view.indices[1:] <= view.indices[:-1]).any())
            or bool((view.values <= 0).any()) or bool((view.values > 1).any())):
        raise ValueError("candidate columns/rates must be sorted, unique and positive")
    _add(work, "factorized_validation_values", 2 * count + 12)


@torch.no_grad()
def validate_candidate_view(view: CandidateView, pattern: Tensor, work=None) -> None:
    """Authenticate every derived column/rate against the actual receptive bits."""
    check_candidate_view(view, work)
    expected = build_candidate_view(pattern, work)
    if (view.position != expected.position
            or view.raw_context_sha256 != expected.raw_context_sha256
            or not torch.equal(view.flags, expected.flags)
            or not torch.equal(view.indices, expected.indices)
            or not torch.equal(view.values, expected.values)):
        raise ValueError("candidate view is not the canonical encoding of its owner")
    _add(work, "factorized_canonical_rebuilds", 1)


def _half_normalize(values, work):
    squared = _named(values.square(), work)
    norm = float(squared.sum().sqrt().item())
    count = values.numel()
    _add(work, "factorized_block_norm_products", count)
    _add(work, "factorized_block_norm_reductions", max(count - 1, 0))
    _add(work, "factorized_block_square_roots", 1)
    if norm:
        values.div_(norm).mul_(HALF_GAIN)
        _add(work, "factorized_block_divisions", count)
        _add(work, "factorized_half_energy_scalings", count)


@torch.no_grad()
def activity(current: FactorizedContext, view: CandidateView, work=None) -> FactorizedActivity:
    """One owner's full sparse presynaptic activity; caller visits every owner.

    The router validates the full current population once on entry. This read
    checks its structure and every gathered value, without a candidate-by-D scan.
    """
    _check_context(current)
    check_candidate_view(view, work)
    indices64 = _named(view.indices.to(torch.int64), work)
    gathered = _named(current.populations.index_select(0, indices64), work)
    current64 = _named(gathered.to(torch.float64), work)
    candidate64 = _named(view.values.to(torch.float64), work)
    lexical = _named(current64 * candidate64, work)
    _finite(lexical, "lexical coincidence")
    if bool((lexical < 0).any()):
        raise ValueError("lexical activity must be nonnegative")
    baseline = float(lexical.sum().item()) / 45
    metadata = _named(torch.empty(14, dtype=torch.float64), work)
    current_flags = _named(current.flags.to(torch.float64), work)
    candidate_flags = _named(view.flags.to(torch.float64), work)
    flag_difference = _named(current_flags - candidate_flags, work)
    metadata[:12].copy_(flag_difference)
    difference = (current.position - view.position) / 128
    metadata[12], metadata[13] = difference, abs(difference)
    _half_normalize(lexical, work)
    _half_normalize(metadata, work)
    tail = _named(torch.arange(LEXICAL_WIDTH, FEATURE_WIDTH, dtype=torch.int32), work)
    indices = _named(torch.cat((view.indices, tail)), work)
    values = _named(torch.cat((lexical, metadata)), work)
    squared = _named(values.square(), work)
    energy = float(squared.sum().item())
    _finite(values, "factorized activity")
    if not math.isfinite(energy) or energy > 1 + 1e-12:
        raise ValueError("factorized activity exceeds the two-block energy bound")
    count = view.indices.numel()
    for name, value in {
        "factorized_activity_calls": 1,
        "factorized_population_gathers": count,
        "factorized_activity_cast_values": 3 * count + 24,
        "factorized_coincidence_products": count,
        "factorized_baseline_reductions": max(count - 1, 0),
        "factorized_baseline_divisions": 1,
        "factorized_flag_subtractions": 12,
        "factorized_position_subtractions": 1,
        "factorized_position_divisions": 1,
        "factorized_position_absolute_values": 1,
        "factorized_energy_products": count + 14,
        "factorized_energy_reductions": count + 13,
    }.items():
        _add(work, name, value)
    return FactorizedActivity(indices, values, baseline, energy)
