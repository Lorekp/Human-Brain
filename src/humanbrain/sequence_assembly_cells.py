"""Bounded input-recruited assemblies with direct local byte/EOS currents.

Agreement and winner cells select using public input only. Recruitment has a
separate target-free entry point; teaching advances a private bank, never live
state. The caller owns causal input construction and whole-lesson publication.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from humanbrain.neuronal_address_competition import AddressCompetition
from humanbrain.persistent_coincidence_reader import compete512, compete1024
from humanbrain.plastic_memory import _all_finite

CAPACITY = 512
OUTPUT_WIDTH = 257
TARGET_CURRENT = math.log(256)
BLOCK_SIZE = 64
WORK_KEYS = (
    "assembly_evaluations", "assembly_candidates", "assembly_agreement_cells",
    "assembly_input_cast_values", "assembly_prototype_cast_values",
    "assembly_agreement_input_applications", "assembly_agreement_additions",
    "assembly_agreement_thresholds", "assembly_agreement_reductions",
    "assembly_score_cast_values", "assembly_exact_thresholds",
    "assembly_sparse_output_reads", "assembly_fallback_output_reads",
    "assembly_named_agreement_allocated_bytes", "assembly_stage_copy_bytes",
    "assembly_validation_values", "assembly_custody_bytes_hashed",
    "assembly_target_free_preparations", "assembly_recruitment_proposals",
    "assembly_receptive_copy_values", "assembly_initial_current_copy_values",
    "assembly_output_offers", "assembly_output_error_values",
    "assembly_output_local_products", "assembly_output_local_divisions",
    "assembly_output_additions", "assembly_staged_output_updates",
    "assembly_staged_recruitments", "assembly_exhausted_offers",
    "assembly_publications", "assembly_publication_copy_bytes",
    "assembly_competition_candidates", "assembly_competition_pairs",
    "assembly_competition_threshold_layers", "assembly_competition_comparator_neurons",
    "assembly_competition_winner_neurons", "assembly_competition_comparator_synapse_applications",
    "assembly_competition_winner_synapse_applications",
    "assembly_competition_comparator_subtractions",
    "assembly_competition_winner_accumulations", "assembly_competition_winner_bias_values",
    "assembly_competition_winner_threshold_subtractions",
    "assembly_competition_threshold_evaluations",
    "assembly_competition_identity_synapse_applications",
    "assembly_competition_identity_reductions",
    "assembly_competition_winner_count_reductions", "assembly_competition_index_values_constructed",
    "assembly_competition_receipt_tensor_bytes",
    "assembly_competition_named_allocated_tensor_bytes",
    "assembly_competition_pair_index_bytes", "assembly_competition_score_gather_bytes",
)


def _add(work: dict | None, name: str, value: int) -> None:
    if work is not None:
        work[name] = work.get(name, 0) + value


def _tensor(value, shape, dtype, name):
    if (not isinstance(value, Tensor) or tuple(value.shape) != shape
            or value.dtype != dtype or value.device.type != "cpu"
            or value.requires_grad or value.grad_fn is not None or value.grad is not None):
        raise ValueError(f"{name} must be detached CPU {dtype} {shape}")


def _finite(value: Tensor, name: str) -> None:
    if not _all_finite(value):
        raise ValueError(f"{name} must be finite")


def _hash_tensors(*values: Tensor, work=None) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(str((tuple(value.shape), value.dtype)).encode("ascii"))
        raw = value.contiguous().numpy().tobytes()
        digest.update(raw)
        _add(work, "assembly_custody_bytes_hashed", len(raw))
    return digest.hexdigest()


@dataclass
class AssemblyStage:
    """Private sequential state; only publish after the owning lesson validates."""

    prototypes: Tensor
    outgoing: Tensor
    used: Tensor
    source_sha256: str
    revision: int = 0
    _owner: object = field(default=None, repr=False)


@dataclass(frozen=True)
class AssemblyEvaluation:
    currents: Tensor
    winner_index: int | None
    exact: bool
    agreement_counts: Tensor
    competition: AddressCompetition | None


@dataclass(frozen=True)
class AssemblyContext:
    """Target-independent selection/reservation bound to one private revision."""

    inputs: Tensor
    initial_currents: Tensor
    evaluation: AssemblyEvaluation
    selected_index: int | None
    recruited: bool
    exhausted: bool
    stage_revision: int
    stage_sha256: str
    _stage: AssemblyStage = field(repr=False)
    _seal: str = field(repr=False)


@dataclass(frozen=True)
class AssemblyUpdate:
    selected_index: int | None
    target: int
    recruited: bool
    exhausted: bool
    staged: bool
    currents_before: Tensor
    currents_after: Tensor
    error: Tensor


class SequenceAssemblyCells(nn.Module):
    """512 Boolean prototypes and separate float64 outgoing current columns.

    No optimizer/autograd participates. Occupied slots form a contiguous prefix.
    Scores are exact integer agreement counts represented in float32; the input
    width bound keeps that conversion exact. Empty slots are canonically zero.
    """

    WORK_KEYS = WORK_KEYS
    capacity = CAPACITY
    _compete = staticmethod(compete512)
    schema = "sequence-assembly-cells-v1"

    def __init__(self, input_width: int) -> None:
        super().__init__()
        if type(input_width) is not int or not 1 <= input_width <= 2**24:
            raise ValueError("assembly input_width must be an integer in1..2**24")
        self.input_width = input_width
        self.register_buffer("prototypes", torch.zeros(
            self.capacity, input_width, dtype=torch.bool, device="cpu"))
        self.register_buffer("outgoing", torch.zeros(
            self.capacity, OUTPUT_WIDTH, dtype=torch.float64, device="cpu"))
        self.register_buffer("used", torch.zeros((), dtype=torch.int64, device="cpu"))
        self._stage_owner = object()

    @property
    def tensor_bytes(self) -> int:
        return self.capacity * self.input_width + self.capacity * OUTPUT_WIDTH * 8 + 8

    def get_extra_state(self) -> dict:
        return {
            "schema": self.schema, "capacity": self.capacity,
            "input_width": self.input_width, "output_width": OUTPUT_WIDTH,
            "receptive_dtype": "bool", "outgoing_dtype": "float64",
            "novelty": "exact input agreement before current target; first free slot",
            "retrieval": "nearest Hamming; fixed two-layer competition; lower-index ties",
            "agreement": "uint8 sum; <=0 and >=2 cells; OR threshold; int64 sum",
            "block_size": BLOCK_SIZE, "target_current": TARGET_CURRENT,
            "learning": "one active assembly; local current error times own activity",
            "exhaustion": "skip novel output learning; exact context reuse remains",
            "publication": "private sequential staging; disabled has no stage advancement",
        }

    def set_extra_state(self, value: dict) -> None:
        if value != self.get_extra_state():
            raise ValueError("assembly metadata mismatch")

    def _validate_bank(self, prototypes, outgoing, used, work=None) -> int:
        _tensor(prototypes, (self.capacity, self.input_width), torch.bool, "assembly prototypes")
        _tensor(outgoing, (self.capacity, OUTPUT_WIDTH), torch.float64, "assembly outgoing")
        _tensor(used, (), torch.int64, "assembly occupancy")
        count = int(used.item())
        if not 0 <= count <= self.capacity:
            raise ValueError("assembly occupancy exceeds capacity")
        _finite(outgoing, "assembly outgoing")
        if bool(prototypes[count:].any()) or bool(outgoing[count:].ne(0).any()):
            raise ValueError("unused assemblies must remain zero")
        _add(work, "assembly_validation_values",
             self.capacity * OUTPUT_WIDTH
             + (self.capacity - count) * (self.input_width + OUTPUT_WIDTH) + 1)
        return count

    def check_parameters(self, work=None) -> None:
        self._validate_bank(self.prototypes, self.outgoing, self.used, work)

    def validate_state(self, state: Mapping, prefix: str = "", work=None) -> None:
        """Also usable by an owning module before PyTorch recursive publication."""
        if not isinstance(state, Mapping) or not isinstance(prefix, str):
            raise ValueError("assembly state must be a mapping with a string prefix")
        if state.get(prefix + "_extra_state") != self.get_extra_state():
            raise ValueError("assembly metadata mismatch")
        self._validate_bank(state.get(prefix + "prototypes"), state.get(prefix + "outgoing"),
                            state.get(prefix + "used"), work)

    def load_state_dict(self, state_dict: Mapping, strict: bool = True, assign: bool = False):
        self.check_parameters()
        if (not strict or assign or not isinstance(state_dict, Mapping)
                or set(state_dict) != {"prototypes", "outgoing", "used", "_extra_state"}):
            raise ValueError("assembly loading requires exact strict in-place state")
        self.validate_state(state_dict)
        return super().load_state_dict(state_dict, strict=True, assign=False)

    @torch.no_grad()
    def snapshot(self) -> dict:
        self.check_parameters()
        return {"prototypes": self.prototypes.clone(), "outgoing": self.outgoing.clone(),
                "used": self.used.clone(), "_extra_state": self.get_extra_state()}

    def _bank_hash(self, prototypes, outgoing, used, work=None) -> str:
        return _hash_tensors(prototypes, outgoing, used, work=work)

    @torch.no_grad()
    def begin_stage(self, work=None) -> AssemblyStage:
        self.check_parameters(work)
        source = self._bank_hash(self.prototypes, self.outgoing, self.used, work)
        _add(work, "assembly_stage_copy_bytes", self.tensor_bytes)
        return AssemblyStage(self.prototypes.clone(), self.outgoing.clone(), self.used.clone(),
                             source, _owner=self._stage_owner)

    def _stage_bank(self, stage: AssemblyStage, work=None) -> int:
        if (not isinstance(stage, AssemblyStage) or stage._owner is not self._stage_owner
                or type(stage.revision) is not int or stage.revision < 0):
            raise ValueError("assembly stage belongs to a different owner or revision")
        for staged, live in ((stage.prototypes, self.prototypes), (stage.outgoing, self.outgoing),
                             (stage.used, self.used)):
            if (isinstance(staged, Tensor)
                    and staged.untyped_storage().data_ptr() == live.untyped_storage().data_ptr()):
                raise ValueError("assembly stage must not alias live storage")
        return self._validate_bank(stage.prototypes, stage.outgoing, stage.used, work)

    def validate_stage(self, stage: AssemblyStage, work=None) -> None:
        self.check_parameters(work)
        self._stage_bank(stage, work)
        if stage.source_sha256 != self._bank_hash(self.prototypes, self.outgoing, self.used, work):
            raise ValueError("live assembly state changed after staging")

    @torch.no_grad()
    def publish(self, stage: AssemblyStage, work=None) -> None:
        self.validate_stage(stage, work)
        # All semantic/shape/finite checks precede every live copy.
        self.commit_validated(stage)
        _add(work, "assembly_publications", 1)
        _add(work, "assembly_publication_copy_bytes", self.tensor_bytes)

    @torch.no_grad()
    def commit_validated(self, stage: AssemblyStage) -> None:
        """Copy-only seam for an owner that prevalidated all component stages.

        Call validate_stage immediately before the synchronous joint commit.
        Publication counters belong after all components have copied.
        """
        self.prototypes.copy_(stage.prototypes)
        self.outgoing.copy_(stage.outgoing)
        self.used.copy_(stage.used)
        stage._owner = None

    def _check_evaluation_inputs(self, inputs: Tensor, fallback_currents: Tensor) -> None:
        _tensor(inputs, (self.input_width,), torch.bool, "assembly input")
        _tensor(fallback_currents, (OUTPUT_WIDTH,), torch.float64, "fallback currents")
        _finite(fallback_currents, "fallback currents")

    @torch.no_grad()
    def evaluate(self, inputs: Tensor, fallback_currents: Tensor, *,
                 stage: AssemblyStage | None = None, work=None) -> AssemblyEvaluation:
        """One public context, bounded64-prototype gate blocks, sparse output read."""
        self._check_evaluation_inputs(inputs, fallback_currents)
        if stage is None:
            self.check_parameters(work)
            prototypes, outgoing, count = self.prototypes, self.outgoing, int(self.used.item())
        else:
            count = self._stage_bank(stage, work)
            prototypes, outgoing = stage.prototypes, stage.outgoing
        return self._evaluate_validated(
            inputs, fallback_currents, prototypes, outgoing, count, work=work)

    @torch.no_grad()
    def _evaluate_validated(self, inputs: Tensor, fallback_currents: Tensor,
                            prototypes: Tensor, outgoing: Tensor, count: int, *,
                            work=None) -> AssemblyEvaluation:
        """Read checked inputs/bank within one synchronous owning call only.

        The caller must keep the bank unchanged and retain its public custody
        checks. This method carries no validity state across calls.
        """
        _add(work, "assembly_evaluations", 1)
        _add(work, "assembly_candidates", count)
        if count == 0:
            _add(work, "assembly_fallback_output_reads", OUTPUT_WIDTH)
            return AssemblyEvaluation(fallback_currents.clone(), None, False,
                                      torch.empty(0, dtype=torch.int64, device="cpu"), None)
        input8 = inputs.to(torch.uint8)
        counts = torch.empty(count, dtype=torch.int64, device="cpu")
        _add(work, "assembly_input_cast_values", self.input_width)
        _add(work, "assembly_named_agreement_allocated_bytes", self.input_width + 8 * count)
        for start in range(0, count, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, count)
            prototype8 = prototypes[start:end].to(torch.uint8)
            currents = prototype8 + input8.unsqueeze(0)
            # Two coincidence cells distinguish both-off and both-on; a fixed
            # OR threshold emits exactly one agreement activity for either.
            both_off = currents <= 0
            both_on = currents >= 2
            off8, on8 = both_off.to(torch.uint8), both_on.to(torch.uint8)
            agreement_current = off8 + on8
            agreement = agreement_current >= 1
            count_values = agreement.sum(dim=1, dtype=torch.int64)
            counts[start:end].copy_(count_values)
            cells = (end - start) * self.input_width
            _add(work, "assembly_agreement_cells", cells)
            _add(work, "assembly_prototype_cast_values", cells)
            _add(work, "assembly_agreement_input_applications", 4 * cells)
            _add(work, "assembly_agreement_additions", 2 * cells)
            _add(work, "assembly_agreement_thresholds", 3 * cells)
            _add(work, "assembly_agreement_reductions", (end - start) * (self.input_width - 1))
            _add(work, "assembly_named_agreement_allocated_bytes", 8 * cells + 8 * (end - start))
            del prototype8, currents, both_off, both_on, off8, on8
            del agreement_current, agreement, count_values
        scores = counts.to(torch.float32)
        competition = self._compete(scores)
        for key, value in competition.work.items():
            _add(work, "assembly_" + key, value)
        winner = competition.winner_index
        # Exact-match detection is a threshold on the winning agreement neuron,
        # not a target-driven search. Counts cannot exceed input_width.
        exact_activity = counts[winner] >= self.input_width
        exact = bool(exact_activity.item())
        _add(work, "assembly_score_cast_values", count)
        _add(work, "assembly_exact_thresholds", 1)
        _add(work, "assembly_sparse_output_reads", OUTPUT_WIDTH)
        _add(work, "assembly_named_agreement_allocated_bytes", 4 * count + 1)
        return AssemblyEvaluation(outgoing[winner].clone(), winner, exact, counts, competition)

    def _context_seal(self, inputs, initial, selected, recruited, exhausted,
                      revision, stage_hash, work=None) -> str:
        tensors = _hash_tensors(inputs, initial, work=work)
        return hashlib.sha256(repr((tensors, selected, recruited, exhausted,
                                   revision, stage_hash)).encode("ascii")).hexdigest()

    @torch.no_grad()
    def prepare_context(self, inputs: Tensor, fallback_currents: Tensor, *,
                        stage: AssemblyStage, work=None) -> AssemblyContext:
        """Reserve one input-defined context without seeing a current target."""
        self.validate_stage(stage, work)
        evaluation = self.evaluate(inputs, fallback_currents, stage=stage, work=work)
        count = int(stage.used.item())
        recruited = not evaluation.exact and count < self.capacity
        exhausted = not evaluation.exact and count == self.capacity
        selected = count if recruited else evaluation.winner_index
        if exhausted:
            selected = None
        # Novel units inherit old currents at this public input, never the
        # nearest assembly's already learned answer or the upcoming target.
        initial = fallback_currents.clone() if recruited else evaluation.currents.clone()
        held_input = inputs.clone()
        stage_hash = self._bank_hash(stage.prototypes, stage.outgoing, stage.used, work)
        seal = self._context_seal(held_input, initial, selected, recruited, exhausted,
                                  stage.revision, stage_hash, work)
        _add(work, "assembly_target_free_preparations", 1)
        _add(work, "assembly_recruitment_proposals", int(recruited))
        return AssemblyContext(held_input, initial, evaluation, selected, recruited, exhausted,
                               stage.revision, stage_hash, stage, seal)

    @torch.no_grad()
    def prepare_teach(self, context: AssemblyContext, target: int, *,
                      stage: AssemblyStage, learning_enabled: bool = True,
                      work=None) -> AssemblyUpdate:
        """One local target-current offer; disabled leaves even private state intact."""
        self.validate_stage(stage, work)
        if type(target) is not int or not 0 <= target < OUTPUT_WIDTH:
            raise ValueError("assembly target must be a raw byte or EOS")
        if type(learning_enabled) is not bool:
            raise ValueError("assembly learning_enabled must be Boolean")
        if (not isinstance(context, AssemblyContext) or context._stage is not stage
                or context.stage_revision != stage.revision):
            raise ValueError("assembly context belongs to another stage or revision")
        _tensor(context.inputs, (self.input_width,), torch.bool, "prepared assembly input")
        _tensor(context.initial_currents, (OUTPUT_WIDTH,), torch.float64, "prepared currents")
        _finite(context.initial_currents, "prepared currents")
        stage_hash = self._bank_hash(stage.prototypes, stage.outgoing, stage.used, work)
        seal = self._context_seal(context.inputs, context.initial_currents,
                                  context.selected_index, context.recruited, context.exhausted,
                                  context.stage_revision, stage_hash, work)
        if stage_hash != context.stage_sha256 or seal != context._seal:
            raise ValueError("prepared assembly context or private bank changed")
        _add(work, "assembly_output_offers", 1)
        before = context.initial_currents.clone()
        if context.exhausted:
            _add(work, "assembly_exhausted_offers", 1)
            return AssemblyUpdate(None, target, False, True, False, before,
                                  before.clone(), torch.zeros_like(before))
        desired = torch.zeros_like(before)
        desired[target] = TARGET_CURRENT
        error = desired - before
        # h=1 and sum(h*h)=1: retain the actual direct synaptic arithmetic.
        proposal = before + error * 1.0 / 1.0
        _add(work, "assembly_output_error_values", OUTPUT_WIDTH)
        _add(work, "assembly_output_local_products", OUTPUT_WIDTH)
        _add(work, "assembly_output_local_divisions", OUTPUT_WIDTH)
        _add(work, "assembly_output_additions", OUTPUT_WIDTH)
        _finite(error, "assembly output error")
        _finite(proposal, "assembly output proposal")
        if learning_enabled:
            index = context.selected_index
            if type(index) is not int or not 0 <= index < self.capacity:
                raise ValueError("prepared assembly selection is invalid")
            # Every fallible proposal check precedes private advancement too.
            if context.recruited:
                stage.prototypes[index].copy_(context.inputs)
                stage.outgoing[index].copy_(before)
                stage.used.add_(1)
                _add(work, "assembly_receptive_copy_values", self.input_width)
                _add(work, "assembly_initial_current_copy_values", OUTPUT_WIDTH)
                _add(work, "assembly_staged_recruitments", 1)
            stage.outgoing[index].copy_(proposal)
            stage.revision += 1
            _add(work, "assembly_staged_output_updates", 1)
        return AssemblyUpdate(context.selected_index, target, context.recruited,
                              False, learning_enabled, before, proposal, error)


class SequenceAssemblyCells1024(SequenceAssemblyCells):
    """Lossless larger event bank; the original512-cell contract stays fixed."""

    capacity = 1024
    _compete = staticmethod(compete1024)
    schema = "sequence-assembly-cells-1024-v1"

    @torch.no_grad()
    def promote_512_state(self, state: Mapping, *, work=None) -> dict:
        """Return a strict successor state, preserving all old rows and zeros.

        This is a shape adapter, not a model load or recognition replay. The
        owning learner validates the complete adapted state before publication.
        """
        if not isinstance(state, Mapping) or set(state) != {
                "prototypes", "outgoing", "used", "_extra_state"}:
            raise ValueError("promotion requires the complete512-cell bank state")
        metadata = self.get_extra_state()
        metadata.update(schema="sequence-assembly-cells-v1", capacity=512)
        if state["_extra_state"] != metadata:
            raise ValueError("promotion source must have the original512-cell metadata")
        _tensor(state["prototypes"], (512, self.input_width), torch.bool, "prior prototypes")
        _tensor(state["outgoing"], (512, OUTPUT_WIDTH), torch.float64, "prior outgoing")
        _tensor(state["used"], (), torch.int64, "prior occupancy")
        count = int(state["used"].item())
        if (not 0 <= count <= 512 or not _all_finite(state["outgoing"])
                or bool(state["prototypes"][count:].any())
                or bool(state["outgoing"][count:].ne(0).any())):
            raise ValueError("promotion requires a finite canonical512-cell bank")
        prototypes = torch.zeros((1024, self.input_width), dtype=torch.bool, device="cpu")
        outgoing = torch.zeros((1024, OUTPUT_WIDTH), dtype=torch.float64, device="cpu")
        prototypes[:512].copy_(state["prototypes"])
        outgoing[:512].copy_(state["outgoing"])
        result = {"prototypes": prototypes, "outgoing": outgoing,
                  "used": state["used"].clone(), "_extra_state": self.get_extra_state()}
        self.validate_state(result, work=work)
        _add(work, "assembly_promotion_allocated_bytes", self.tensor_bytes)
        _add(work, "assembly_promotion_copy_bytes", 512 * (self.input_width + OUTPUT_WIDTH * 8) + 8)
        _add(work, "assembly_promotion_prior_validation_values",
             512 * OUTPUT_WIDTH + (512 - count) * (self.input_width + OUTPUT_WIDTH) + 1)
        return result
