"""Committed neuronal free recall with the inherited local corrective surrogate.

The same address/codec synapses act and learn. Free recall uses one actual
cursor after fixed neuronal competition. Inherited teaching explicitly reopens
all addresses; its local updates are not derivatives through the hard gate.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from humanbrain.byte_sequence_decoder import BYTE_EOS, ByteGeneration
from humanbrain.local_memory_learner import LocalMemoryLearner, SequencePrediction
from humanbrain.neuronal_address_competition import AddressCompetition, compete
from humanbrain.neuronal_sequence_memory import TapeState


@dataclass(frozen=True)
class CompetitiveSequencePrediction(SequencePrediction):
    """Actual one-branch beliefs, with separate uncommitted address diagnostics."""

    competition: AddressCompetition
    diagnostic_log_prior: Tensor


class CompetitiveLocalMemoryLearner(LocalMemoryLearner):
    """Only free recall changes; query features and teacher arithmetic are inherited."""

    def get_extra_state(self) -> dict:
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "local-neuronal-competitive-address-learner-v1",
            "recall": (
                "two fixed threshold layers select first maximum float32 address score; "
                "one actual cursor follows forward EOS-absorbing adjacency"
            ),
            "competition": (
                "float64 pair currents for i<j; nonnegative comparator thresholds; "
                "fixed signed winner currents and nonnegative thresholds; "
                "lower-index ties; sum(i*w_i) identity readout"
            ),
            "free_belief": (
                "one normalized branch; full soft address prior is diagnostic only"
            ),
            "codec_activity": (
                "evaluate every occupied sensory slot once with held weights; "
                "free recall gathers one actual cursor at every decision including first; "
                "teacher retains inherited all-address bank and prefix gathers"
            ),
            "teacher_phase": (
                "after actual execution, reopen all address candidates on the same synapses; "
                "inherited raw-byte/EOS prefix evidence and local p-minus-final-posterior "
                "surrogate; no competition pass, hard-gate derivative or backward transport"
            ),
        })
        return metadata

    @torch.no_grad()
    def predict_with_receipt(
        self, query: bytes, state: TapeState, max_bytes: int
    ) -> CompetitiveSequencePrediction:
        """One query, one competition, and one actually traversed content branch."""
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        event = self._query(query, state)
        competition = compete(event.address_scores)
        diagnostic_log_prior = event.log_prior.clone()
        work = dict(event.work)
        work.update(competition.work)
        # Free recall does not need the forward X bank after address competition.
        del event
        cursors = torch.tensor([competition.winner_index], dtype=torch.int64, device="cpu")
        self.memory._check_cursors(state, cursors)
        work["branch_cursor_initializations"] = 1
        # This bank is deliberately still evaluated over ALL occupied slots.
        bank_cursors = torch.arange(state.used, dtype=torch.int64, device="cpu")
        work["codec_bank_slot_indices"] = state.used
        bank = self._codec_bank(state, bank_cursors, work)
        del bank_cursors
        initial_log_belief = torch.zeros(1, dtype=torch.float64, device="cpu")
        log_belief = initial_log_belief
        output, steps = bytearray(), []
        for position in range(max_bytes + 1):
            self.memory._check_cursors(state, cursors)
            conditional = bank._gather(cursors)
            work["codec_response_gathers"] += 1
            work["codec_response_values_gathered"] += 257 * 4
            work["codec_response_bytes_gathered"] += 257 * (4 + 3 * 8)
            mixture = self._mixture(conditional, log_belief)
            token = int(mixture.argmax())
            receipt = self._observe(position, token, cursors, log_belief, conditional, mixture)
            steps.append(receipt)
            log_belief = receipt.posterior_log_belief
            self._count_step(work, 1)
            if token == BYTE_EOS or position == max_bytes:
                return CompetitiveSequencePrediction(
                    ByteGeneration(bytes(output), "eos" if token == BYTE_EOS else "limit"),
                    tuple(steps), work, initial_log_belief.clone(), log_belief.clone(),
                    competition, diagnostic_log_prior,
                )
            output.append(token)
            cursors = self.memory._advance_checked(state, cursors)
            work["branch_cursor_advances"] += 1
        raise AssertionError("bounded generation must have a stopping decision")

    def accounting(self) -> dict:
        accounting = super().accounting()
        accounting.update({
            "allocated_address_competition_neurons": 128 * 127 // 2 + 128,
            "maximum_address_comparator_neurons": 128 * 127 // 2,
            "maximum_address_winner_neurons": 128,
            "maximum_competition_signed_synapse_applications": 2 * 128 * 127,
            "competition_slow_synapses": 0,
            "free_active_cursor_count": 1,
            "free_log_belief_bytes": 8,
            "maximum_free_diagnostic_log_prior_bytes": 128 * 8,
            "free_codec_response_gather_bytes": 257 * (4 + 3 * 8),
            "maximum_free_overlapping_codec_response_bytes": (128 + 2) * 257 * (4 + 3 * 8),
            "codec_clock": (
                "one full occupied bank per free or teacher call; free gathers one cursor "
                "at every decision including first; teacher retains inherited all-branch work"
            ),
            "competition_clock": (
                "free-only two fixed threshold layers; actual trace and tensor payloads "
                "counted in competition receipt; no competition evaluation during teaching"
            ),
        })
        return accounting
