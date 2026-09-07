"""Paired raw-prefix/request dendrites with observed-byte local learning.

No module-level simulation, task parsing, output readout fit or upstream error
transport. The owner supplies the causal prefix and a frozen prior per decision.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import pairwise

import torch
from torch import Tensor, nn

OUTPUTS = 257
CELLS_PER_OUTPUT = 16
BRANCHES_PER_CELL = 4
BRANCHES_PER_OUTPUT = CELLS_PER_OUTPUT * BRANCHES_PER_CELL
BRANCHES = OUTPUTS * BRANCHES_PER_OUTPUT
BOS = 257
BASAL_WIDTH = 32 * 258
REQUEST_WIDTH = 256 + 258 * 258 + 8 * 256
CAPACITIES = (16, 8, 16, 8)
CONTACT_OFFSETS = (0, 16, 24, 40, 48)
INPUT_WIDTHS = (BASAL_WIDTH, 256, 258 * 258, 8 * 256)
REQUEST_OFFSETS = (0, 256, 256 + 258 * 258, REQUEST_WIDTH)
TENSOR_BYTES = 4_095_552
BANK_NAMES = ("indices", "permanences", "counts", "thresholds", "occupied")
MASK64 = (1 << 64) - 1
WORK_KEYS = tuple(
    "dendritic_" + key
    for key in (
        "request_encodings",
        "request_raw_bytes",
        "request_zero_values",
        "request_presence_writes",
        "request_pair_writes",
        "request_position_writes",
        "basal_encodings",
        "basal_prefix_bytes",
        "basal_encoded_symbols",
        "basal_zero_values",
        "basal_presence_writes",
        "validation_values",
        "custody_bytes_hashed",
        "stage_copy_bytes",
        "record_decisions",
        "record_tensor_bytes",
        "evaluations",
        "branch_slots",
        "contact_index_cast_values",
        "contact_connectivity_tests",
        "contact_gathers",
        "contact_input_gates",
        "contact_reduction_additions",
        "compartment_thresholds",
        "branch_boolean_gates",
        "column_reduction_values",
        "prior_values_checked",
        "competition_pairs",
        "competition_levels",
        "competition_class_promotions",
        "competition_class_subtractions",
        "competition_class_negations",
        "competition_prior_subtractions",
        "competition_thresholds",
        "competition_boolean_gates",
        "competition_relay_branches",
        "competition_selected_values",
        "competition_carried_values",
        "competition_named_bytes",
        "named_tensor_bytes",
        "offers",
        "source_offers",
        "apical_offers",
        "recruited_branches",
        "attached_apical",
        "exhausted_offers",
        "selection_candidates",
        "selection_scan_values",
        "active_index_scan_values",
        "proposal_copy_bytes",
        "private_copy_values",
        "private_copy_bytes",
        "selection_apical_additions",
        "selection_subtractions",
        "selection_thresholds",
        "selection_boolean_gates",
        "allocation_count_additions",
        "allocation_comparisons",
        "sampling_entries",
        "sampling_identity_additions",
        "sampling_identity_multiplications",
        "sampling_masks",
        "sampling_mix_multiplications",
        "sampling_mix_shifts",
        "sampling_mix_xors",
        "sampling_key_comparisons",
        "sampling_sort_copies",
        "contact_initializations",
        "local_signed_cast_values",
        "local_additions",
        "local_subtractions",
        "local_clamp_values",
        "local_changed_contacts",
        "false_prediction_branches",
        "false_prediction_contact_values",
        "staged_offers",
        "publication_copy_bytes",
        "publications",
    )
)


def _add(work, key, value):
    if work is not None:
        name = "dendritic_" + key
        work[name] = work.get(name, 0) + int(value)


def _tensor(value, shape, dtype, label):
    if (
        not isinstance(value, Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device.type != "cpu"
        or value.requires_grad
        or value.grad_fn is not None
        or value.grad is not None
    ):
        raise ValueError(f"{label} must be detached CPU {dtype} {shape}")


def _hash(*values, work=None):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, Tensor):
            raw = value.contiguous().numpy().tobytes()
            descriptor = str((tuple(value.shape), value.dtype)).encode("ascii")
        elif isinstance(value, bytes):
            raw, descriptor = value, b"bytes"
        else:
            raw, descriptor = repr(value).encode("utf-8"), b"scalar"
        digest.update(len(descriptor).to_bytes(8, "little"))
        digest.update(descriptor)
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        _add(work, "custody_bytes_hashed", len(raw))
    return digest.hexdigest()


@dataclass(frozen=True)
class RequestActivity:
    raw_request: bytes
    population: Tensor
    seal: str


def encode_request(raw: bytes, work=None) -> RequestActivity:
    if type(raw) is not bytes or len(raw) > 256:
        raise ValueError("request must be at most256 raw bytes")
    population = torch.zeros(REQUEST_WIDTH, dtype=torch.bool)
    previous = BOS
    for position, byte in enumerate(raw):
        population[byte] = True
        population[256 + previous * 258 + byte] = True
        population[REQUEST_OFFSETS[2] + (position // 32) * 256 + byte] = True
        previous = byte
    population[256 + previous * 258 + 256] = True
    _add(work, "request_encodings", 1)
    _add(work, "request_raw_bytes", len(raw))
    _add(work, "request_zero_values", REQUEST_WIDTH)
    _add(work, "request_presence_writes", len(raw))
    _add(work, "request_pair_writes", len(raw) + 1)
    _add(work, "request_position_writes", len(raw))
    _add(work, "named_tensor_bytes", REQUEST_WIDTH)
    return RequestActivity(raw, population, _hash(raw, population, work=work))


def _check_request(request, work=None):
    if request is None:
        return
    if not isinstance(request, RequestActivity) or type(request.raw_request) is not bytes:
        raise ValueError("invalid encoded request")
    if len(request.raw_request) > 256:
        raise ValueError("encoded request exceeds256 raw bytes")
    _tensor(request.population, (REQUEST_WIDTH,), torch.bool, "request population")
    if request.seal != _hash(request.raw_request, request.population, work=work):
        raise ValueError("encoded request was changed")
    _add(work, "validation_values", REQUEST_WIDTH)


def encode_prefix(prefix: bytes, work=None) -> Tensor:
    """Encode only the causal last32 symbols; full prefix length remains counted."""
    if type(prefix) is not bytes:
        raise ValueError("prefix must be raw bytes")
    symbols = list(prefix[-32:])
    if len(prefix) < 32:
        symbols.insert(0, BOS)
    activity = torch.zeros(BASAL_WIDTH, dtype=torch.bool)
    for lag, symbol in enumerate(reversed(symbols)):
        activity[lag * 258 + symbol] = True
    _add(work, "basal_encodings", 1)
    _add(work, "basal_prefix_bytes", len(prefix))
    _add(work, "basal_encoded_symbols", len(symbols))
    _add(work, "basal_zero_values", BASAL_WIDTH)
    _add(work, "basal_presence_writes", len(symbols))
    _add(work, "named_tensor_bytes", BASAL_WIDTH)
    return activity


def _mix64(value, work=None):
    value = (value ^ (value >> 30)) * 0xBF58476D1CE4E5B9 & MASK64
    value = (value ^ (value >> 27)) * 0x94D049BB133111EB & MASK64
    value ^= value >> 31
    _add(work, "sampling_mix_multiplications", 2)
    _add(work, "sampling_mix_shifts", 3)
    _add(work, "sampling_mix_xors", 3)
    _add(work, "sampling_masks", 2)
    return value


def _merge_sort(pairs, work=None):
    """Stable bottom-up tuple ordering with explicit comparison/copy counts."""
    width = 1
    while width < len(pairs):
        result = []
        for start in range(0, len(pairs), 2 * width):
            middle, end = min(start + width, len(pairs)), min(start + 2 * width, len(pairs))
            left, right = start, middle
            while left < middle and right < end:
                _add(work, "sampling_key_comparisons", 1)
                if pairs[left] <= pairs[right]:
                    result.append(pairs[left])
                    left += 1
                else:
                    result.append(pairs[right])
                    right += 1
            result.extend(pairs[left:middle])
            result.extend(pairs[right:end])
        _add(work, "sampling_sort_copies", len(result))
        pairs = result
        width *= 2
    return pairs


def _sample(active, count, seed, branch, compartment, work=None):
    identity = 4 * branch + compartment + 1
    base = (seed + 0x9E3779B97F4A7C15 * identity) & MASK64
    _add(work, "sampling_identity_additions", 3 + len(active))
    _add(work, "sampling_identity_multiplications", 2)
    _add(work, "sampling_masks", 1 + len(active))
    ranked = [(_mix64((base + int(index)) & MASK64, work), int(index)) for index in active]
    _add(work, "sampling_entries", len(ranked))
    ranked = _merge_sort(ranked, work)[:count]
    return [pair[0] for pair in _merge_sort([(index, index) for _, index in ranked], work)]


@dataclass(frozen=True)
class TupleCompetition:
    winner: int
    levels: tuple[tuple[int, ...], ...]
    pairs: int


def compete_bytes(classes: Tensor, prior: Tensor, work=None) -> TupleCompetition:
    _tensor(classes, (OUTPUTS,), torch.int16, "byte classes")
    _tensor(prior, (OUTPUTS,), torch.float64, "byte prior")
    if bool(((classes < 0) | (classes > 2)).any()) or not bool(torch.isfinite(prior).all()):
        raise ValueError("invalid byte competition values")
    _add(work, "prior_values_checked", OUTPUTS)
    current_class, current_prior = classes.clone(), prior.clone()
    identities = torch.arange(OUTPUTS, dtype=torch.int64)
    levels, pairs_total = [], 0
    _add(work, "competition_named_bytes", OUTPUTS * 18)
    while identities.numel() > 1:
        count = identities.numel()
        pairs = count // 2
        left_class, right_class = current_class[: 2 * pairs : 2], current_class[1 : 2 * pairs : 2]
        dc = left_class.to(torch.int32) - right_class.to(torch.int32)
        dv = current_prior[: 2 * pairs : 2] - current_prior[1 : 2 * pairs : 2]
        if not bool(torch.isfinite(dv).all()):
            raise ValueError("byte prior subtraction overflow")
        greater = dc >= 1
        equal = (dc >= 0) & (-dc >= 0)
        left_wins = greater | (equal & (dv >= 0))
        chosen_class = torch.where(left_wins, left_class, right_class)
        chosen_prior = torch.where(
            left_wins, current_prior[: 2 * pairs : 2], current_prior[1 : 2 * pairs : 2]
        )
        chosen_ids = torch.where(
            left_wins, identities[: 2 * pairs : 2], identities[1 : 2 * pairs : 2]
        )
        if count % 2:
            chosen_class = torch.cat((chosen_class, current_class[-1:]))
            chosen_prior = torch.cat((chosen_prior, current_prior[-1:]))
            chosen_ids = torch.cat((chosen_ids, identities[-1:]))
            _add(work, "competition_carried_values", 3)
        current_class, current_prior, identities = chosen_class, chosen_prior, chosen_ids
        levels.append(tuple(int(value) for value in identities.tolist()))
        pairs_total += pairs
        _add(work, "competition_pairs", pairs)
        _add(work, "competition_levels", 1)
        _add(work, "competition_class_promotions", 2 * pairs)
        _add(work, "competition_class_subtractions", pairs)
        _add(work, "competition_class_negations", pairs)
        _add(work, "competition_prior_subtractions", pairs)
        _add(work, "competition_thresholds", 4 * pairs)
        _add(work, "competition_boolean_gates", 3 * pairs)
        _add(work, "competition_relay_branches", 6 * pairs)
        _add(work, "competition_selected_values", 3 * pairs)
        # Explicit casts/current/gates/three selected arrays, plus odd concatenations.
        _add(work, "competition_named_bytes", 4 * pairs * 4 + 8 * pairs + 7 * pairs + 18 * pairs)
        if count % 2:
            _add(work, "competition_named_bytes", 18 * (pairs + 1))
    return TupleCompetition(int(identities[0]), tuple(levels), pairs_total)


@dataclass
class BindingStage:
    bank: dict[str, Tensor]
    base_sha256: str
    revision: int = 0
    _owner: object = field(default=None, repr=False)
    _base_revision: int = field(default=0, repr=False)
    _seal: str = field(default="", repr=False)
    _tensor_identity: tuple = field(default=(), repr=False)
    _committed: bool = field(default=False, repr=False)


@dataclass(frozen=True)
class BindingDecision:
    prefix: bytes
    request: RequestActivity | None
    prior: Tensor
    classes: Tensor
    winner: int
    selected_branch: int | None
    matches: Tensor
    basal_active: Tensor
    joint_active: Tensor
    competition: TupleCompetition
    bank_sha256: str
    stage_revision: int
    _basal: Tensor = field(repr=False)
    _stage: BindingStage | None = field(repr=False)
    _owner: object = field(repr=False)
    _seal: str = field(default="", repr=False)
    _used: bool = field(default=False, repr=False)

    def record(self, work=None):
        selected = (
            None if self.selected_branch is None else self.matches[self.selected_branch].clone()
        )
        _add(work, "record_decisions", 1)
        _add(work, "record_tensor_bytes", OUTPUTS * 10 + (0 if selected is None else 8))
        return {
            "schema": "dendritic-binding-decision-v1",
            "prefix_hex": self.prefix.hex(),
            "request_sha256": None if self.request is None else self.request.seal,
            "classes": self.classes.clone(),
            "prior": self.prior.clone(),
            "winner": self.winner,
            "selected_branch": self.selected_branch,
            "selected_matches": selected,
            "competition": {
                "winner": self.competition.winner,
                "pairs": self.competition.pairs,
                "levels": {str(i): row for i, row in enumerate(self.competition.levels)},
            },
            "bank_sha256": self.bank_sha256,
            "stage_revision": self.stage_revision,
        }


@dataclass(frozen=True)
class BindingUpdate:
    symbol: int
    winner_before: int
    selected_branch: int | None
    recruited: bool
    attached_apical: bool
    exhausted: bool
    staged: bool
    false_prediction_branches: tuple[int, ...]
    changed_contacts: int
    selected_contacts: tuple[dict, ...]
    before_bank_sha256: str
    proposed_bank_sha256: str | None
    proposal_sha256: str
    stage_revision_before: int
    stage_revision_after: int

    def record(self):
        return {
            "schema": "dendritic-binding-update-v1",
            "symbol": self.symbol,
            "winner_before": self.winner_before,
            "selected_branch": self.selected_branch,
            "recruited": self.recruited,
            "attached_apical": self.attached_apical,
            "exhausted": self.exhausted,
            "staged": self.staged,
            "false_prediction_branches": self.false_prediction_branches,
            "changed_contacts": self.changed_contacts,
            "selected_contacts": {str(i): item for i, item in enumerate(self.selected_contacts)},
            "before_bank_sha256": self.before_bank_sha256,
            "proposed_bank_sha256": self.proposed_bank_sha256,
            "proposal_sha256": self.proposal_sha256,
            "stage_revision_before": self.stage_revision_before,
            "stage_revision_after": self.stage_revision_after,
        }


class DendriticRequestBinding(nn.Module):
    """Bounded byte-cell dendrites; caller owns whole-response/action custody."""

    def __init__(self, seed=733):
        super().__init__()
        if type(seed) is not int or not 0 <= seed < (1 << 63):
            raise ValueError("dendritic seed must be an integer in[0,2**63)")
        self.seed = seed
        self.register_buffer("indices", torch.zeros((BRANCHES, 48), dtype=torch.int32))
        self.register_buffer("permanences", torch.zeros((BRANCHES, 48), dtype=torch.uint8))
        self.register_buffer("counts", torch.zeros((BRANCHES, 4), dtype=torch.uint8))
        self.register_buffer("thresholds", torch.zeros((BRANCHES, 4), dtype=torch.uint8))
        self.register_buffer("occupied", torch.zeros(BRANCHES, dtype=torch.bool))
        self._owner = object()
        self._revision = 0

    @property
    def tensor_bytes(self):
        return TENSOR_BYTES

    def get_extra_state(self):
        return {
            "schema": "dendritic-request-binding-v1",
            "seed": self.seed,
            "outputs": OUTPUTS,
            "cells_per_output": CELLS_PER_OUTPUT,
            "branches_per_cell": BRANCHES_PER_CELL,
            "contact_capacities": list(CAPACITIES),
            "request_encoding": "byte-presence+boundary-pairs+absolute32-byte-bins-v1",
            "basal_encoding": "BOS+newest-first32-raw-symbols-v1",
            "sampling": "splitmix64-identity-rank-stable-mergesort-v1",
            "permanence": "uint8-0..3-connected>=1-signed-local-update",
            "forward": "same-branch-basal-apical-AND;class/prior/lower-byte-tournament",
            "learning": "arriving-symbol-column-local;source-basal;correction-basal+apical",
            "exhaustion": "skip-new-association-no-eviction",
        }

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("dendritic metadata mismatch")

    def _bank(self):
        return {name: getattr(self, name) for name in BANK_NAMES}

    def _bank_hash(self, bank, work=None):
        return _hash(self.get_extra_state(), *(bank[name] for name in BANK_NAMES), work=work)

    def _check_bank(self, bank, work=None, *, rows=BRANCHES):
        if not isinstance(bank, Mapping) or set(bank) != set(BANK_NAMES):
            raise ValueError("dendritic bank keys mismatch")
        _tensor(bank["indices"], (rows, 48), torch.int32, "contact indices")
        _tensor(bank["permanences"], (rows, 48), torch.uint8, "contact permanences")
        _tensor(bank["counts"], (rows, 4), torch.uint8, "contact counts")
        _tensor(bank["thresholds"], (rows, 4), torch.uint8, "contact thresholds")
        _tensor(bank["occupied"], (rows,), torch.bool, "branch occupancy")
        counts = bank["counts"].to(torch.int16)
        if not torch.equal(bank["thresholds"].to(torch.int16), (3 * counts + 3) // 4):
            raise ValueError("branch threshold mismatch")
        if not torch.equal(bank["occupied"], counts[:, 0] > 0):
            raise ValueError("empty basal branch cannot be occupied")
        if bool(((counts[:, 1:].sum(1) > 0) & (counts[:, 2] == 0)).any()):
            raise ValueError("apical branch requires an ordered-pair compartment")
        for compartment, (capacity, width) in enumerate(zip(CAPACITIES, INPUT_WIDTHS, strict=True)):
            start, end = CONTACT_OFFSETS[compartment : compartment + 2]
            index = bank["indices"][:, start:end]
            permanence = bank["permanences"][:, start:end]
            n = counts[:, compartment]
            if bool(((n < 0) | (n > capacity)).any()):
                raise ValueError("contact count outside compartment capacity")
            allocated = torch.arange(capacity)[None, :] < n[:, None]
            if bool(((index < 0) | (index >= width) | (permanence > 3)).any()):
                raise ValueError("invalid contact index or permanence")
            if bool(((~allocated) & ((index != 0) | (permanence != 0))).any()):
                raise ValueError("unused contact payload must be zero")
            if capacity > 1:
                pair = torch.arange(1, capacity)[None, :] < n[:, None]
                if bool((pair & (index[:, 1:] <= index[:, :-1])).any()):
                    raise ValueError("allocated contacts must have unique increasing indices")
            if bool(((~bank["occupied"]) & (n != 0)).any()):
                raise ValueError("unused branch has contacts")
        _add(work, "validation_values", sum(value.numel() for value in bank.values()))

    def check_parameters(self, work=None):
        self._check_bank(self._bank(), work)

    def validate_state(self, state, prefix="", work=None):
        keys = {prefix + name for name in (*BANK_NAMES, "_extra_state")}
        if (
            not isinstance(state, Mapping)
            or {key for key in state if key.startswith(prefix)} != keys
        ):
            raise ValueError("dendritic state keys mismatch")
        if state[prefix + "_extra_state"] != self.get_extra_state():
            raise ValueError("dendritic state metadata mismatch")
        self._check_bank({name: state[prefix + name] for name in BANK_NAMES}, work)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if strict is not True or assign is not False:
            raise ValueError("dendritic loading requires strict copy semantics")
        self.check_parameters()
        self.validate_state(state_dict)
        result = super().load_state_dict(state_dict, strict=True, assign=False)
        self.invalidate_stages_after_load()
        return result

    def invalidate_stages_after_load(self):
        """Owner calls after a validated recursive nn.Module load, which bypasses ours."""
        self._revision += 1

    def begin_stage(self, work=None):
        self.check_parameters(work)
        bank = {name: value.clone() for name, value in self._bank().items()}
        digest = self._bank_hash(bank, work)
        _add(work, "stage_copy_bytes", TENSOR_BYTES)
        return BindingStage(
            bank,
            digest,
            _owner=self._owner,
            _base_revision=self._revision,
            _seal=digest,
            _tensor_identity=tuple((id(bank[name]), bank[name].data_ptr()) for name in BANK_NAMES),
        )

    def validate_stage(self, stage, work=None):
        if (
            not isinstance(stage, BindingStage)
            or stage._owner is not self._owner
            or stage._committed
            or stage._base_revision != self._revision
        ):
            raise ValueError("stale or foreign dendritic stage")
        self.check_parameters(work)
        self._check_bank(stage.bank, work)
        if tuple((id(stage.bank[name]), stage.bank[name].data_ptr()) for name in BANK_NAMES) != (
            stage._tensor_identity
        ) or any(
            stage.bank[name].data_ptr() == getattr(self, name).data_ptr() for name in BANK_NAMES
        ):
            raise ValueError("dendritic stage storage was replaced or aliases the live bank")
        if self._bank_hash(self._bank(), work) != stage.base_sha256:
            raise ValueError("live dendritic bank changed during stage")
        if self._bank_hash(stage.bank, work) != stage._seal:
            raise ValueError("dendritic stage was modified")

    def commit_validated(self, stage, work=None):
        """Copy-only seam; owner must validate all proposals immediately beforehand.

        No callbacks or borrowed-tensor mutations may intervene. Journal IO is
        outside this atomic validation boundary and can fail after live commit.
        """
        for name in BANK_NAMES:
            getattr(self, name).copy_(stage.bank[name])
        self._revision += 1
        stage._committed = True
        _add(work, "publication_copy_bytes", TENSOR_BYTES)
        _add(work, "publications", 1)

    def _decision_hash(self, decision, work=None):
        return _hash(
            decision.prefix,
            None if decision.request is None else decision.request.seal,
            decision.prior,
            decision.classes,
            decision.winner,
            decision.selected_branch,
            decision.matches,
            decision.basal_active,
            decision.joint_active,
            decision.competition,
            decision.bank_sha256,
            decision.stage_revision,
            decision._basal,
            work=work,
        )

    @staticmethod
    def _inputs(basal, request):
        if request is None:
            return (basal, None, None, None)
        return (basal, *(request.population[start:end] for start, end in pairwise(REQUEST_OFFSETS)))

    @staticmethod
    def _rank(candidates, matches, work=None):
        ids = candidates.nonzero(as_tuple=False).flatten().tolist()
        _add(work, "selection_candidates", len(ids))
        _add(work, "selection_scan_values", candidates.numel())
        _add(work, "named_tensor_bytes", 8 * len(ids))
        if not ids:
            return None
        best = None
        best_basal = best_apical = 0
        for index in ids:
            basal = int(matches[index, 0])
            apical = int(matches[index, 1:].sum(dtype=torch.int32))
            _add(work, "selection_apical_additions", 2)
            _add(work, "named_tensor_bytes", 4)
            if best is None:
                best, best_basal, best_apical = index, basal, apical
                continue
            db, da, di = basal - best_basal, apical - best_apical, best - index
            # Fixed scalar local-WTA simulation; distinct from byte tournament work.
            bg, be, ag, ae, lower = db > 0, db == 0, da > 0, da == 0, di > 0
            replace = bg | (be & (ag | (ae & lower)))
            _add(work, "selection_subtractions", 3)
            _add(work, "selection_thresholds", 5)
            _add(work, "selection_boolean_gates", 4)
            if replace:
                best, best_basal, best_apical = index, basal, apical
        return best

    def evaluate(
        self,
        prefix: bytes,
        request: RequestActivity | None,
        prior: Tensor,
        *,
        stage: BindingStage | None = None,
        work=None,
    ) -> BindingDecision:
        _check_request(request, work)
        _tensor(prior, (OUTPUTS,), torch.float64, "frozen prior")
        if not bool(torch.isfinite(prior).all()):
            raise ValueError("frozen prior must be finite")
        if stage is None:
            self.check_parameters(work)
            bank, revision = self._bank(), self._revision
            bank_hash = self._bank_hash(bank, work)
        else:
            self.validate_stage(stage, work)
            bank, revision, bank_hash = stage.bank, stage.revision, stage._seal
        basal = encode_prefix(prefix, work)
        inputs = self._inputs(basal, request)
        matches = torch.zeros((BRANCHES, 4), dtype=torch.int16)
        active = torch.zeros((BRANCHES, 4), dtype=torch.bool)
        rows = bank["occupied"].nonzero(as_tuple=False).flatten()
        occupied_count = rows.numel()
        _add(work, "evaluations", 1)
        _add(work, "branch_slots", BRANCHES)
        _add(work, "named_tensor_bytes", BRANCHES * 12 + rows.numel() * 8)
        for compartment, population in enumerate(inputs):
            if population is None:
                continue
            start, end = CONTACT_OFFSETS[compartment : compartment + 2]
            width = end - start
            index32 = bank["indices"][rows, start:end]
            index = index32.to(torch.int64)
            permanence = bank["permanences"][rows, start:end]
            connected = permanence >= 1
            gathered = population[index]
            gated = connected & gathered
            count = gated.sum(dim=1, dtype=torch.int16)
            thresholds = bank["thresholds"][rows, compartment].to(torch.int16)
            nonempty = bank["counts"][rows, compartment] > 0
            # Empty/unoccupied basal compartments never receive vacuous support.
            activated = nonempty & (count >= thresholds)
            matches[rows, compartment] = count
            active[rows, compartment] = activated
            contacts = occupied_count * width
            _add(work, "contact_index_cast_values", contacts)
            _add(work, "contact_connectivity_tests", contacts)
            _add(work, "contact_gathers", contacts)
            _add(work, "contact_input_gates", contacts)
            _add(work, "contact_reduction_additions", occupied_count * (width - 1))
            _add(work, "compartment_thresholds", occupied_count)
            _add(work, "branch_boolean_gates", occupied_count)
            _add(work, "named_tensor_bytes", contacts * 16 + occupied_count * 9)
        basal_active = active[:, 0].clone()
        if request is None:
            joint = torch.zeros(BRANCHES, dtype=torch.bool)
            _add(work, "named_tensor_bytes", BRANCHES)
        else:
            apical_nonempty = bank["counts"][:, 1:] > 0
            apical_exists = apical_nonempty.any(dim=1)
            apical = ((~apical_nonempty) | active[:, 1:]).all(dim=1)
            joint = basal_active & apical_exists & apical
            _add(work, "branch_boolean_gates", BRANCHES * 5)
            _add(work, "named_tensor_bytes", BRANCHES * 12)
        basal_columns = basal_active.reshape(OUTPUTS, BRANCHES_PER_OUTPUT).any(dim=1)
        joint_columns = joint.reshape(OUTPUTS, BRANCHES_PER_OUTPUT).any(dim=1)
        classes = basal_columns.to(torch.int16) + joint_columns.to(torch.int16)
        held_prior = prior.clone()
        competition = compete_bytes(classes, held_prior, work)
        mask = joint.clone() if int(classes[competition.winner]) == 2 else basal_active.clone()
        start = competition.winner * BRANCHES_PER_OUTPUT
        mask[:start] = False
        mask[start + BRANCHES_PER_OUTPUT :] = False
        selected = self._rank(mask, matches, work)
        _add(work, "column_reduction_values", BRANCHES * 2)
        _add(work, "named_tensor_bytes", BRANCHES * 2 + OUTPUTS * 16)
        decision = BindingDecision(
            prefix,
            request,
            held_prior,
            classes,
            competition.winner,
            selected,
            matches,
            basal_active,
            joint,
            competition,
            bank_hash,
            revision,
            basal,
            stage,
            self._owner,
        )
        object.__setattr__(decision, "_seal", self._decision_hash(decision, work))
        return decision

    def validate_decision(self, decision, stage=None, work=None):
        if (
            not isinstance(decision, BindingDecision)
            or decision._owner is not self._owner
            or decision._stage is not stage
            or decision._used
        ):
            raise ValueError("used, foreign or incorrectly staged dendritic decision")
        _check_request(decision.request, work)
        if stage is None:
            self.check_parameters(work)
            bank, revision = self._bank(), self._revision
        else:
            self.validate_stage(stage, work)
            bank, revision = stage.bank, stage.revision
        if (
            revision != decision.stage_revision
            or self._bank_hash(bank, work) != decision.bank_sha256
        ):
            raise ValueError("dendritic decision is stale")
        if self._decision_hash(decision, work) != decision._seal:
            raise ValueError("dendritic decision was modified")

    def _allocate_compartment(self, row, branch, compartment, population, work=None):
        if population is None:
            return False
        active = population.nonzero(as_tuple=False).flatten().tolist()
        _add(work, "active_index_scan_values", population.numel())
        _add(work, "named_tensor_bytes", 8 * len(active))
        n = min(CAPACITIES[compartment], len(active))
        selected = _sample(active, n, self.seed, branch, compartment, work)
        start = CONTACT_OFFSETS[compartment]
        row["indices"][0, start : start + n] = torch.tensor(selected, dtype=torch.int32)
        row["permanences"][0, start : start + n] = 1
        row["counts"][0, compartment] = n
        row["thresholds"][0, compartment] = (3 * n + 3) // 4
        _add(work, "contact_initializations", n)
        _add(work, "named_tensor_bytes", 4 * n)
        return n > 0

    def prepare_observation(
        self,
        stage: BindingStage,
        decision: BindingDecision,
        symbol: int,
        *,
        apical_learning: bool,
        enabled: bool,
        work=None,
    ) -> BindingUpdate:
        if type(symbol) is not int or not 0 <= symbol < OUTPUTS:
            raise ValueError("observed symbol must be byte or EOS")
        if type(apical_learning) is not bool or type(enabled) is not bool:
            raise ValueError("learning flags must be Boolean")
        self.validate_decision(decision, stage, work)
        if apical_learning and decision.request is None:
            raise ValueError("apical learning requires an actual encoded request")
        bank = stage.bank
        column_start = symbol * BRANCHES_PER_OUTPUT
        column_end = column_start + BRANCHES_PER_OUTPUT
        column = torch.zeros(BRANCHES, dtype=torch.bool)
        column[column_start:column_end] = True
        _add(work, "named_tensor_bytes", BRANCHES)
        _add(work, "offers", 1)
        _add(work, "apical_offers" if apical_learning else "source_offers", 1)
        inputs = self._inputs(decision._basal, decision.request)
        if apical_learning:
            selected = self._rank(column & decision.joint_active, decision.matches, work)
            if selected is None:
                empty_apical = bank["counts"][:, 1:].sum(dim=1, dtype=torch.int16) == 0
                selected = self._rank(
                    column & decision.basal_active & empty_apical, decision.matches, work
                )
        else:
            selected = self._rank(column & decision.basal_active, decision.matches, work)
        recruited = selected is None
        if recruited:
            usage = bank["occupied"][column_start:column_end].reshape(
                CELLS_PER_OUTPUT, BRANCHES_PER_CELL
            )
            used_counts = usage.sum(dim=1, dtype=torch.int16)
            cell = None
            best_count = BRANCHES_PER_CELL
            _add(work, "allocation_count_additions", CELLS_PER_OUTPUT * (BRANCHES_PER_CELL - 1))
            for candidate in range(CELLS_PER_OUTPUT):
                count = int(used_counts[candidate])
                _add(work, "allocation_comparisons", 1)
                if count < best_count:
                    cell, best_count = candidate, count
            if cell is not None:
                slot = None
                for index in range(BRANCHES_PER_CELL):
                    _add(work, "allocation_comparisons", 1)
                    if not bool(usage[cell, index]):
                        slot = index
                        break
                selected = column_start + cell * BRANCHES_PER_CELL + slot
            _add(work, "selection_candidates", CELLS_PER_OUTPUT)
        exhausted = selected is None
        recruited = recruited and not exhausted
        row = None
        selected_receipts = []
        attached = False
        positive_changes = 0
        if selected is not None:
            row = {name: value[selected : selected + 1].clone() for name, value in bank.items()}
            _add(
                work,
                "proposal_copy_bytes",
                sum(value.numel() * value.element_size() for value in row.values()),
            )
            created = set()
            if recruited:
                self._allocate_compartment(row, selected, 0, inputs[0], work)
                created.add(0)
                row["occupied"][0] = True
            if apical_learning and not bool(row["counts"][0, 1:].any()):
                for compartment in (1, 2, 3):
                    self._allocate_compartment(
                        row, selected, compartment, inputs[compartment], work
                    )
                    created.add(compartment)
                attached = True
            for compartment in range(4 if apical_learning else 1):
                start = CONTACT_OFFSETS[compartment]
                n = int(row["counts"][0, compartment])
                indices = row["indices"][0, start : start + n].to(torch.int64)
                previous = bank["permanences"][selected, start : start + n].clone()
                activity = inputs[compartment][indices].to(torch.int16)
                if compartment not in created:
                    signed = row["permanences"][0, start : start + n].to(torch.int16)
                    proposed = (signed + activity - (1 - activity)).clamp(0, 3).to(torch.uint8)
                    row["permanences"][0, start : start + n] = proposed
                    _add(work, "local_signed_cast_values", 2 * n)
                    _add(work, "local_additions", n)
                    _add(work, "local_subtractions", 2 * n)
                    _add(work, "local_clamp_values", n)
                after = row["permanences"][0, start : start + n]
                positive_changes += int((previous != after).sum())
                selected_receipts.append(
                    {
                        "compartment": compartment,
                        "created": compartment in created,
                        "indices": tuple(int(x) for x in indices.tolist()),
                        "activity": tuple(int(x) for x in activity.tolist()),
                        "before": tuple(int(x) for x in previous.tolist()),
                        "after": tuple(int(x) for x in after.tolist()),
                        "count": n,
                        "threshold": int(row["thresholds"][0, compartment]),
                    }
                )
                _add(work, "named_tensor_bytes", n * 18)
            self._check_bank(row, work, rows=1)
        false_ids = (
            (decision.joint_active & (~column)).nonzero(as_tuple=False).flatten()
            if apical_learning
            else torch.empty(0, dtype=torch.int64)
        )
        false_before = bank["permanences"][false_ids, 16:48]
        false_after = false_before.clone()
        _add(work, "proposal_copy_bytes", false_before.numel() + false_after.numel())
        negative_changes = 0
        if false_ids.numel():
            for compartment in (1, 2, 3):
                start, end = CONTACT_OFFSETS[compartment : compartment + 2]
                indices = bank["indices"][false_ids, start:end].to(torch.int64)
                activity = inputs[compartment][indices].to(torch.int16)
                signed = false_before[:, start - 16 : end - 16].to(torch.int16)
                # Disconnected allocated contacts and zero tails stay at zero.
                proposed = (signed - activity).clamp(0, 3).to(torch.uint8)
                false_after[:, start - 16 : end - 16] = proposed
                count = proposed.numel()
                _add(work, "local_signed_cast_values", 2 * count)
                _add(work, "local_subtractions", count)
                _add(work, "local_clamp_values", count)
                _add(work, "named_tensor_bytes", 4 * count + indices.numel() * 12 + count)
            negative_changes = int((false_before != false_after).sum())
            if bool((false_after > 3).any()):
                raise ValueError("invalid negative contact proposal")
        changed = positive_changes + negative_changes
        proposal_hash = _hash(
            symbol,
            selected,
            tuple(selected_receipts),
            false_ids,
            false_before,
            false_after,
            work=work,
        )
        before_revision = stage.revision
        staged = enabled and bool(recruited or attached or changed)
        after_hash = None
        # Every fallible contact/receipt calculation above precedes these private
        # writes. A later private hashing failure still cannot publish live state.
        if staged:
            if row is not None:
                for name in BANK_NAMES:
                    bank[name][selected : selected + 1].copy_(row[name])
                _add(
                    work,
                    "private_copy_bytes",
                    sum(value.numel() * value.element_size() for value in row.values()),
                )
                _add(work, "private_copy_values", sum(value.numel() for value in row.values()))
            if false_ids.numel():
                bank["permanences"][false_ids, 16:48] = false_after
                _add(work, "private_copy_bytes", false_after.numel())
                _add(work, "private_copy_values", false_after.numel())
            stage.revision += 1
            stage._seal = self._bank_hash(bank, work)
            after_hash = stage._seal
            _add(work, "staged_offers", 1)
        elif enabled:
            after_hash = stage._seal
        object.__setattr__(decision, "_used", True)
        _add(work, "recruited_branches", int(recruited))
        _add(work, "attached_apical", int(attached))
        _add(work, "exhausted_offers", int(exhausted))
        _add(work, "local_changed_contacts", changed)
        _add(work, "false_prediction_branches", false_ids.numel())
        _add(work, "false_prediction_contact_values", false_before.numel())
        return BindingUpdate(
            symbol,
            decision.winner,
            selected,
            recruited,
            attached,
            exhausted,
            staged,
            tuple(int(x) for x in false_ids.tolist()),
            changed,
            tuple(selected_receipts),
            decision.bank_sha256,
            after_hash,
            proposal_hash,
            before_revision,
            stage.revision,
        )
