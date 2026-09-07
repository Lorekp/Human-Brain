"""Every-owner float64 routing and direct own-emission outcome teaching.

Feature construction and assembly recruitment are target-free, separate callers.
No posterior, output derivative or chosen target owner enters this module.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from humanbrain.factorized_sequence_features import (
    FEATURE_WIDTH,
    CandidateView,
    FactorizedContext,
    activity,
    validate_context,
)
from humanbrain.neuronal_address_competition import AddressCompetition
from humanbrain.plastic_memory import _all_finite

CAPACITY = 512
WORK_KEYS = (
    "router_evaluations", "router_candidates", "router_weight_products",
    "router_score_reductions", "router_baseline_additions", "router_index_cast_values",
    "router_stage_copy_bytes", "router_validation_values", "router_custody_bytes_hashed",
    "router_outcome_offers", "router_outcome_comparisons", "router_skipped_outcomes",
    "router_local_error_values", "router_local_products", "router_local_divisions",
    "router_local_additions", "router_staged_updates", "router_publications",
    "router_publication_copy_bytes", "router_named_allocated_bytes",
    "router_competition_candidates", "router_competition_pairs",
    "router_competition_threshold_layers", "router_competition_comparator_neurons",
    "router_competition_winner_neurons", "router_competition_comparator_synapse_applications",
    "router_competition_winner_synapse_applications", "router_competition_comparator_subtractions",
    "router_competition_winner_accumulations", "router_competition_winner_bias_values",
    "router_competition_winner_threshold_subtractions", "router_competition_threshold_evaluations",
    "router_competition_identity_synapse_applications", "router_competition_identity_reductions",
    "router_competition_winner_count_reductions", "router_competition_index_values_constructed",
    "router_competition_receipt_tensor_bytes", "router_competition_named_allocated_tensor_bytes",
    "router_competition_pair_index_bytes", "router_competition_score_gather_bytes",
)


def _add(work, name, value):
    if work is not None:
        work[name] = work.get(name, 0) + value


def _tensor(value, shape, dtype, name):
    if (not isinstance(value, Tensor) or tuple(value.shape) != shape or value.dtype != dtype
            or value.device.type != "cpu" or value.requires_grad or value.grad_fn is not None
            or value.grad is not None):
        raise ValueError(f"{name} must be detached CPU {dtype} {shape}")


def _finite(value, name):
    if not _all_finite(value):
        raise ValueError(f"{name} must be finite")


def _hash(*values, work=None):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, Tensor):
            raw = value.contiguous().numpy().tobytes()
            digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
        else:
            raw = repr(value).encode("utf-8")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
        _add(work, "router_custody_bytes_hashed", len(raw))
    return digest.hexdigest()


@torch.no_grad()
def compete_float64(scores: Tensor) -> AddressCompetition:
    """Two actual threshold layers over all 1..512 scores; lower-index ties."""
    if not isinstance(scores, Tensor) or scores.ndim != 1 or not 1 <= scores.numel() <= CAPACITY:
        raise ValueError("routing scores must have 1..512 entries")
    n = scores.numel()
    _tensor(scores, (n,), torch.float64, "routing scores")
    _finite(scores, "routing scores")
    held = scores.clone()
    pairs = n * (n - 1) // 2
    indices = torch.triu_indices(n, n, offset=1, dtype=torch.int64, device="cpu")
    left, right = held.index_select(0, indices[0]), held.index_select(0, indices[1])
    currents = left - right
    _finite(currents, "routing comparator currents")
    comparators = currents >= 0
    positions = torch.arange(n, dtype=torch.int64, device="cpu")
    winner_currents = positions.to(torch.float64)
    comparator64 = comparators.to(torch.float64)
    winner_currents.index_add_(0, indices[0], comparator64)
    winner_currents.index_add_(0, indices[1], comparator64, alpha=-1)
    winner_currents.sub_(n - 1)
    _finite(winner_currents, "routing winner currents")
    winners = winner_currents >= 0
    if int(winners.sum().item()) != 1:
        raise ValueError("routing competition must have exactly one winner")
    identity = winners.to(torch.int64)
    products = positions * identity
    selected = int(products.sum().item())
    work = {
        "competition_candidates": n, "competition_pairs": pairs,
        "competition_threshold_layers": 2, "competition_comparator_neurons": pairs,
        "competition_winner_neurons": n,
        "competition_comparator_synapse_applications": 2 * pairs,
        "competition_winner_synapse_applications": 2 * pairs,
        "competition_comparator_subtractions": pairs,
        "competition_winner_accumulations": 2 * pairs,
        "competition_winner_bias_values": n, "competition_winner_threshold_subtractions": n,
        "competition_threshold_evaluations": pairs + n,
        "competition_identity_synapse_applications": n,
        "competition_identity_reductions": n - 1,
        "competition_winner_count_reductions": n - 1,
        "competition_index_values_constructed": 2 * pairs + n,
        "competition_receipt_tensor_bytes": 17 * n + 9 * pairs,
        "competition_named_allocated_tensor_bytes": 41 * n + 49 * pairs + 16,
        "competition_pair_index_bytes": 16 * pairs,
        "competition_score_gather_bytes": 16 * pairs,
    }
    return AddressCompetition(held, currents, comparators, winner_currents, winners, selected, work)


@dataclass
class RouterStage:
    theta: Tensor
    source_sha256: str
    revision: int = 0
    _owner: object = field(default=None, repr=False)


@dataclass(frozen=True)
class _OwnedFreeRouting:
    """Call-local borrowed weights; never installed on the router or checkpoint."""

    theta: Tensor
    version: int
    weights_sha256: str
    owner: object


@dataclass(frozen=True)
class RoutingDecision:
    scores: Tensor
    selected_index: int | None
    selected_score: float
    selected_indices: Tensor
    selected_activity: Tensor
    selected_baseline: float
    energy: float
    competition: AddressCompetition | None
    context_sha256: str
    weights_sha256: str
    stage_revision: int | None
    _stage: RouterStage | None = field(repr=False)
    _owner: object = field(repr=False)
    _seal: str = field(repr=False)


@dataclass(frozen=True)
class RoutingUpdate:
    selected_index: int | None
    emitted_byte: int
    offered_byte: int
    desired: float
    score_before: float
    score_after: float
    energy: float
    error: float
    staged: bool
    skipped: bool
    indices: Tensor
    delta: Tensor


class FactorizedSequenceRouter(nn.Module):
    """A shared output potential vector, with private sequential local staging."""

    WORK_KEYS = WORK_KEYS

    def __init__(self, input_width: int = FEATURE_WIDTH):
        super().__init__()
        if type(input_width) is not int or input_width != FEATURE_WIDTH:
            raise ValueError("router input width must match the fixed factorized schema")
        self.input_width = input_width
        self.register_buffer("theta", torch.zeros(input_width, dtype=torch.float64, device="cpu"))
        self._owner = object()

    @property
    def tensor_bytes(self):
        return self.input_width * 8

    def get_extra_state(self):
        return {
            "schema": "factorized-sequence-router-v1", "input_width": self.input_width,
            "capacity": CAPACITY, "dtype": "float64", "initialization": "zero theta",
            "selection": "all owners; two float64 threshold layers; lower-index ties",
            "learning": "own pre-clamp byte equals target: +1 else -1; local potential clamp",
            "publication": "private sequential stage; joint prevalidation; copy-only commit",
        }

    def set_extra_state(self, value):
        if value != self.get_extra_state():
            raise ValueError("router metadata mismatch")

    def _check_theta(self, value, work=None):
        _tensor(value, (self.input_width,), torch.float64, "router theta")
        _finite(value, "router theta")
        _add(work, "router_validation_values", self.input_width)

    def check_parameters(self, work=None):
        self._check_theta(self.theta, work)

    def validate_state(self, state: Mapping, prefix="", work=None):
        if not isinstance(state, Mapping) or not isinstance(prefix, str):
            raise ValueError("router state must be a mapping with string prefix")
        if state.get(prefix + "_extra_state") != self.get_extra_state():
            raise ValueError("router metadata mismatch")
        self._check_theta(state.get(prefix + "theta"), work)

    def load_state_dict(self, state_dict: Mapping, strict=True, assign=False):
        self.check_parameters()
        if not strict or assign or not isinstance(state_dict, Mapping) or set(state_dict) != {
                "theta", "_extra_state"}:
            raise ValueError("router loading requires exact strict in-place state")
        self.validate_state(state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=False)

    @torch.no_grad()
    def snapshot(self):
        self.check_parameters()
        return {"theta": self.theta.clone(), "_extra_state": self.get_extra_state()}

    @torch.no_grad()
    def begin_stage(self, work=None):
        self.check_parameters(work)
        source = _hash(self.theta, work=work)
        _add(work, "router_stage_copy_bytes", self.tensor_bytes)
        return RouterStage(self.theta.clone(), source, _owner=self._owner)

    def validate_stage(self, stage, work=None):
        self.check_parameters(work)
        if (not isinstance(stage, RouterStage) or stage._owner is not self._owner
                or type(stage.revision) is not int or stage.revision < 0):
            raise ValueError("router stage owner or revision mismatch")
        self._check_theta(stage.theta, work)
        if stage.theta.untyped_storage().data_ptr() == self.theta.untyped_storage().data_ptr():
            raise ValueError("router stage must not alias live weights")
        if stage.source_sha256 != _hash(self.theta, work=work):
            raise ValueError("live router changed after staging")

    @torch.no_grad()
    def commit_validated(self, stage):
        """Copy-only seam: owner validates every component immediately beforehand.

        No validation, allocation or counter mutation belongs inside a joint
        multi-component commit. The synchronous owner counts publication after
        all copies and invalidates/abandons the consumed stages.
        """
        self.theta.copy_(stage.theta)
        stage._owner = None

    def _seal(self, decision, work=None):
        competition = decision.competition
        fields = [decision.scores, decision.selected_index, decision.selected_score,
                  decision.selected_indices, decision.selected_activity, decision.selected_baseline,
                  decision.energy, decision.context_sha256, decision.weights_sha256,
                  decision.stage_revision]
        if competition is not None:
            fields.extend((competition.scores, competition.pair_currents,
                           competition.comparator_activity, competition.winner_currents,
                           competition.winner_activity, competition.winner_index,
                           tuple(sorted(competition.work.items()))))
        return _hash(*fields, work=work)

    def _begin_validated_free_response(self, work=None):
        """Borrow weights after the synchronous reader's full entry validation.

        The reader must retain its end model/view/tape byte checks before
        publishing anything. Version checks do not detect NumPy writes.
        """
        return _OwnedFreeRouting(self.theta, self.theta._version,
                                 _hash(self.theta, work=work), self._owner)

    def _check_owned_free_response(self, owned):
        if (not isinstance(owned, _OwnedFreeRouting) or owned.owner is not self._owner
                or owned.theta is not self.theta or owned.version != self.theta._version):
            raise RuntimeError("owned free routing weights changed or were rebound")

    @torch.no_grad()
    def _evaluate_owned_free(self, current_context, candidate_views, owned, *, work=None):
        """Use a freshly build_context-validated population inside its owner call.

        Candidate activity checks and every numerical operation remain in the
        shared kernel. Public evaluation and all teacher stages stay checked.
        """
        self._check_owned_free_response(owned)
        return self._evaluate_validated(
            current_context, candidate_views, theta=owned.theta, stage=None,
            weights_sha256=owned.weights_sha256, work=work)

    @torch.no_grad()
    def evaluate(self, current_context: FactorizedContext, candidate_views: Sequence[CandidateView],
                 *, stage: RouterStage | None = None, work=None) -> RoutingDecision:
        """Visit every owner; reconstruct one selected row after neuronal selection."""
        if not isinstance(candidate_views, (tuple, list)) or len(candidate_views) > CAPACITY:
            raise ValueError("routing requires all 0..512 ordered candidate views")
        if not isinstance(current_context, FactorizedContext):
            raise ValueError("routing needs a factorized current context")
        validate_context(current_context, work=work)
        if stage is None:
            self.check_parameters(work)
            theta = self.theta
        else:
            self.validate_stage(stage, work)
            theta = stage.theta
        return self._evaluate_validated(
            current_context, candidate_views, theta=theta, stage=stage, work=work)

    def _evaluate_validated(self, current_context, candidate_views, *, theta, stage,
                            work=None, weights_sha256=None):
        """Identical ordered float64 arithmetic for public and owned callers."""
        count = len(candidate_views)
        scores = torch.empty(count, dtype=torch.float64, device="cpu")
        _add(work, "router_evaluations", 1)
        _add(work, "router_candidates", count)
        _add(work, "router_named_allocated_bytes", 8 * count)
        for index, view in enumerate(candidate_views):
            row = activity(current_context, view, work=work)
            indices = row.indices.to(torch.int64)
            products = theta.index_select(0, indices) * row.values
            score = float(products.sum().item()) + row.baseline
            if not math.isfinite(score):
                raise ValueError("routing potential must be finite")
            scores[index] = score
            _add(work, "router_index_cast_values", indices.numel())
            _add(work, "router_weight_products", indices.numel())
            _add(work, "router_score_reductions", max(0, indices.numel() - 1))
            _add(work, "router_baseline_additions", 1)
            _add(work, "router_named_allocated_bytes", 24 * indices.numel())
        competition = compete_float64(scores) if count else None
        selected = competition.winner_index if competition is not None else None
        # No host ranking chooses this row. Its extra fixed-feature computation
        # is counted by activity; there is no second score/output prediction.
        best_row = (activity(current_context, candidate_views[selected], work=work)
                    if selected is not None else None)
        if competition is not None:
            for key, value in competition.work.items():
                _add(work, "router_" + key, value)
        indices = (best_row.indices.clone() if best_row is not None else
                   torch.empty(0, dtype=torch.int32, device="cpu"))
        values = (best_row.values.clone() if best_row is not None else
                  torch.empty(0, dtype=torch.float64, device="cpu"))
        decision = RoutingDecision(
            scores, selected, float(scores[selected].item()) if count else 0.0, indices, values,
            best_row.baseline if count else 0.0, best_row.energy if count else 0.0,
            competition, current_context.raw_context_sha256,
            _hash(theta, work=work) if weights_sha256 is None else weights_sha256,
            stage.revision if stage is not None else None, stage, self._owner, "")
        object.__setattr__(decision, "_seal", self._seal(decision, work))
        _add(work, "router_named_allocated_bytes", indices.numel() * 12)
        return decision

    def validate_decision(self, decision, *, stage=None, work=None):
        if (not isinstance(decision, RoutingDecision) or decision._owner is not self._owner
                or decision._stage is not stage):
            raise ValueError("routing decision owner or stage mismatch")
        if stage is None:
            self.check_parameters(work)
            theta, revision = self.theta, None
        else:
            self.validate_stage(stage, work)
            theta, revision = stage.theta, stage.revision
        if (decision.stage_revision != revision
                or decision.weights_sha256 != _hash(theta, work=work)
                or decision._seal != self._seal(decision, work)):
            raise ValueError("routing decision, trace or weights changed")

    @torch.no_grad()
    def prepare_outcome(self, decision: RoutingDecision, emitted_byte: int, offered_byte: int,
                        *, stage: RouterStage, learning_enabled: bool = True,
                        work=None) -> RoutingUpdate:
        """Caller binds emitted_byte to the selected owner's pre-clamp output.

        Current labels enter here only. No output likelihood or derivative is
        computed; the routing output receives its direct +1/-1 outcome current.
        """
        if not isinstance(stage, RouterStage):
            raise ValueError("routing teaching requires a private stage")
        self.validate_decision(decision, stage=stage, work=work)
        if (type(emitted_byte) is not int or not 0 <= emitted_byte <= 256
                or type(offered_byte) is not int or not 0 <= offered_byte <= 256
                or type(learning_enabled) is not bool):
            raise ValueError("routing outcome needs raw byte/EOS labels and Boolean enable")
        desired = 1.0 if emitted_byte == offered_byte else -1.0
        error = desired - decision.selected_score
        _add(work, "router_outcome_offers", 1)
        _add(work, "router_outcome_comparisons", 1)
        skipped = decision.selected_index is None or decision.energy == 0
        if skipped:
            _add(work, "router_skipped_outcomes", 1)
            _add(work, "router_named_allocated_bytes", 12 * decision.selected_indices.numel())
            return RoutingUpdate(decision.selected_index, emitted_byte, offered_byte, desired,
                                 decision.selected_score, decision.selected_score, decision.energy,
                                 error, False, True, decision.selected_indices.clone(),
                                 torch.zeros_like(decision.selected_activity))
        indices = decision.selected_indices.to(torch.int64)
        before = stage.theta.index_select(0, indices)
        delta = error * decision.selected_activity / decision.energy
        after = before + delta
        _finite(delta, "routing local delta")
        _finite(after, "routing local proposal")
        score_after = float((after * decision.selected_activity).sum().item())
        score_after += decision.selected_baseline
        if not math.isfinite(score_after):
            raise ValueError("routing corrected potential must be finite")
        n = indices.numel()
        _add(work, "router_index_cast_values", n)
        _add(work, "router_local_error_values", 1)
        _add(work, "router_local_products", n)
        _add(work, "router_local_divisions", n)
        _add(work, "router_local_additions", n)
        _add(work, "router_weight_products", n)
        _add(work, "router_score_reductions", max(0, n - 1))
        _add(work, "router_baseline_additions", 1)
        _add(work, "router_named_allocated_bytes", 52 * n)
        # Build fallible receipt objects before private advancement too.
        result = RoutingUpdate(decision.selected_index, emitted_byte, offered_byte, desired,
                               decision.selected_score, score_after, decision.energy, error,
                               learning_enabled, False, decision.selected_indices.clone(), delta)
        if learning_enabled:
            stage.theta.index_copy_(0, indices, after)
            stage.revision += 1
            _add(work, "router_staged_updates", 1)
        return result
