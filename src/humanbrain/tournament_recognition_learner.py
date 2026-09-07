"""The existing recognition learner with exact factual comparator/copy relays.

Only factual competition changes. All occupied addresses retain their original
float32 scores; float64 event-owner routing and local teaching are inherited.
"""

import torch

from humanbrain.byte_sequence_decoder import BYTE_BOS
from humanbrain.factual_address_tournament import compete_factual
from humanbrain.local_memory_learner import _finite
from humanbrain.recognition_sequence_memory_learner import RecognitionSequenceMemoryLearner
from humanbrain.recognition_sequence_splice_learner import RecognitionSequenceSpliceLearner
from humanbrain.sequence_assembly_memory_learner import PREFIX_SLOTS, _add


class TournamentRecognitionMemoryLearner(RecognitionSequenceMemoryLearner):
    """Same persistent reader, with all-address factual tournament competition."""

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "tournament-recognition-memory-learner-v1",
            "recall": (
                "fresh factual competition at each output step over every occupied address; "
                "original float32 scores, lowest-index ties, unchanged context and fallback"),
            "competition": (
                "factual adjacent comparator and gated potential/identity copy-relay tree; "
                "float32 potentials widened to float64, nonnegative left comparator ties; "
                "odd original-order carry; ceil(log2(N)) comparator/relay stages; "
                "not the two-layer all-pairs threshold construction"),
            "factual_competition_work": "factual_tournament_*; event-owner competition unchanged",
        })
        return metadata

    def _context_forward(self, prepared, prefix, cache, work):
        """Preserve the frozen factual score, fallback and input arithmetic."""
        q, xq, contexts, legacy, sensor = prepared
        xp = cache.features()
        scores = (xq * self.address_weight).sum((1, 2)) + (xp * self.prefix_weight).sum((1, 2))
        _finite(scores, "recognition memory address scores")
        competition = compete_factual(scores)
        for key, value in competition.work.items():
            _add(work, key, value)
        selected = competition.winner_index
        selected_context = contexts[selected:selected + 1]
        position = len(prefix)
        p = self._rails([BYTE_BOS, *prefix], PREFIX_SLOTS, work)
        if q is not None and position <= 16:
            _, logits, _ = self._conditional(
                q, p[:17], selected_context, legacy[selected:selected + 1],
                position, self.output_weight, work)
            fallback = logits[0]
            _add(work, "sequence_conditioned_fallbacks", 1)
        else:
            fallback = legacy[selected].clone()
            _add(work, "sequence_codec_fallbacks", 1)
        position_cells = torch.zeros(PREFIX_SLOTS, dtype=torch.bool)
        position_cells[position] = True
        inputs = torch.cat((sensor, p.flatten(), selected_context.flatten(), position_cells))
        _add(work, "sequence_input_values", inputs.numel())
        _add(work, "output_decisions_contexts", 1)
        _add(work, "output_address_steps", 1)
        _add(work, "output_address_synapse_products", xq.shape[0] * 2048)
        return inputs, fallback, selected, scores

    def accounting(self):
        result = super().accounting()
        # These inherited entries describe the replaced factual all-pairs circuit,
        # not the separate unchanged float64 event-owner competition.
        del result["maximum_address_comparator_neurons"]
        del result["maximum_address_winner_neurons"]
        result.update({
            "schema": "tournament-recognition-accounting-v1",
            "factual_competition": {
                "circuit": "fixed adjacent comparator and gated potential/identity copy-relay tree",
                "eligible_addresses": "all occupied facts, N=1..2048; no shortlist or padding",
                "comparators_per_decision": "N-1",
                "comparator_levels_per_decision": "ceil(log2(N)); zero for N=1",
                "relay_levels_per_decision": "ceil(log2(N)); zero for N=1",
                "maximum_comparators_per_decision": 2047,
                "maximum_comparator_levels_per_decision": 11,
                "maximum_relay_levels_per_decision": 11,
                "maximum_logical_gated_payload_branches_per_decision": 4 * 2047,
                # Exact scalar recurrence maxima over every N=1..2048. Odd
                # concatenations make N=2048 alone an insufficient byte bound.
                "maximum_named_allocated_bytes_per_decision": 124_086,
                "named_allocated_maximum_at_candidates": 2017,
                "maximum_retained_tensor_bytes_per_decision": 59_367,
                "retained_maximum_at_candidates": 2048,
                "maximum_peak_named_live_bytes_per_decision": 82_899,
                "peak_named_maximum_at_candidates": 2047,
                "allocation_scope": (
                    "actual per-call factual_tournament_* bytes include odd carries; "
                    "named tensor buffers only, not RSS or Python/framework allocations"),
                "checkpoint_tensor_bytes": 0,
            },
        })
        return result


class TournamentRecognitionSpliceLearner(RecognitionSequenceSpliceLearner):
    """Use the same tournament reader for language, payload and local correction."""

    def __init__(self, *, mode="learned", seed=733, reader=None):
        if mode not in self.MODES:
            raise ValueError("invalid tournament recognition action mode")
        if reader is None:
            reader = TournamentRecognitionMemoryLearner(mode=self.MODES[mode], seed=seed)
        if not isinstance(reader, TournamentRecognitionMemoryLearner):
            raise ValueError("tournament actions require the same tournament recognition reader")
        super().__init__(mode=mode, seed=seed, reader=reader)

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "tournament-recognition-splice-learner-v1",
            "reader": "tournament-recognition-memory-learner-v1",
        })
        return metadata
