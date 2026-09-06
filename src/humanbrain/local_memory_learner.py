"""Local document writing, question addressing and raw-byte teacher feedback.

This composition owns no corpus, answer inventory, optimizer or action solver.
Fixed byte codes feed learned writer/query cells. First generation is separate
from explicitly supplied teacher bytes; every byte/EOS teacher event is counted.
Fixed broadcasts approximate upstream credit, not reverse derivatives.
"""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from humanbrain.byte_sequence_decoder import BYTE_BOS, BYTE_EOS, ByteGeneration, ByteSequenceDecoder
from humanbrain.plastic_memory import (
    LocalMemoryState,
    PlasticMemory,
    PlasticMemoryState,
    StreamingRateCell,
    _all_finite,
)


@dataclass(frozen=True)
class LocalLessonFeedback:
    """One lesson's explicit exposure, updates and retired persistent state."""

    state: LocalMemoryState
    teacher_count: int
    mean_loss: float
    decoder_updated: bool
    query_updated: bool
    memory_updated: bool
    mean_context_signal: Tensor


class LocalMemoryLearner(nn.Module):
    """One persistent memory stream with five matched learning controls.

    ``tagged`` retains write-key-tagged eligibility; ``aggregate`` uses the
    memory's scalar overwrite credit. ``writer_maps_frozen`` fixes controller,
    key/value/strength maps but trains the query map, query cell and decoder.
    ``frozen`` fixes every slow parameter, while ``writes_disabled`` permits
    local slow teaching but cannot change fast synapses or write credit.

    The decoder's fixed embedding is shared for documents and queries. Each raw
    input is followed by one terminator (256); no textual normalization is applied.
    Memory, query and decoder seeds are seed, seed+1, seed+2. Upstream feedback
    uses a private seed+3 stream, controller/key/value/strength/query order,
    normal(0,1)/sqrt(context_size). Decoder feedback uses its own seed+2 offset.
    All fixed matrices serialize; controls have identical initial tensor bytes.
    """

    MODES = frozenset({"tagged", "aggregate", "writer_maps_frozen", "frozen", "writes_disabled"})

    def __init__(
        self,
        *,
        mode: str = "tagged",
        embedding_size: int = 32,
        hidden_size: int = 64,
        key_size: int = 64,
        value_size: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError("unknown local memory control mode")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        self.mode = mode
        self.memory = PlasticMemory(embedding_size, hidden_size, key_size, value_size, seed=seed)
        self.query_cell = StreamingRateCell(embedding_size, embedding_size, seed=seed + 1)
        self.decoder = ByteSequenceDecoder(
            embedding_size + value_size,
            hidden_size,
            embedding_size,
            seed=seed + 2,
            local_learning=True,
        )
        generator = torch.Generator(device="cpu").manual_seed(seed + 3)
        for name, size in (
            ("controller", hidden_size),
            ("key", key_size),
            ("value", value_size),
            ("strength", 1),
            ("query", key_size),
        ):
            self.register_buffer(
                f"{name}_feedback",
                torch.randn(size, embedding_size + value_size, generator=generator)
                / math.sqrt(embedding_size + value_size),
            )
        if mode == "writer_maps_frozen":
            for name, parameter in self.memory.named_parameters():
                if not name.startswith("query_"):
                    parameter.requires_grad_(False)
        if mode == "frozen":
            self.requires_grad_(False)

    def get_extra_state(self) -> dict:
        return {
            "schema": "local-memory-learner-v1",
            "mode": self.mode,
            "embedding_size": self.decoder.embedding_size,
            "hidden_size": self.decoder.hidden_size,
            "key_size": self.memory.key_size,
            "value_size": self.memory.value_size,
            "input_code": "shared-fixed-decoder-embedding; raw-bytes+256",
            "feedback": "normal/sqrt(context); private seed+3; controller,key,value,strength,query",
            "clock": "mean teacher-byte signals once; upstream once; retire write/query credit",
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("local learner checkpoint configuration mismatch")

    def _encoded(self, raw: bytes) -> Tensor:
        if not isinstance(raw, bytes):
            raise ValueError("inputs must be raw bytes")
        tokens = torch.tensor(
            [*raw, BYTE_EOS], dtype=torch.long, device=self.decoder.embedding.weight.device
        )
        return self.decoder.embedding(tokens)[None]

    def _check_state_mode(self, state: LocalMemoryState) -> None:
        if self.mode == "writer_maps_frozen" and any(
            parameter.requires_grad for parameter in self.memory._writer_parameters().values()
        ):
            raise ValueError("writer_maps_frozen requires writer/controller/maps to remain frozen")
        expected = "aggregate" if self.mode == "aggregate" else "tagged"
        if (
            not isinstance(state, LocalMemoryState)
            or state.credit_mode != expected
            or state.synapses.shape[0] != 1
        ):
            raise ValueError("memory state must match the control mode and single stream")

    @torch.no_grad()
    def ingest(
        self, documents: Sequence[bytes], state: LocalMemoryState | None = None
    ) -> LocalMemoryState:
        """Ingest complete documents once, then purge writer activity/traces.

        No question/target enters this method. Starting without state creates an
        empty episode; supplying state is explicit cumulative ingestion. Memory
        and credit survive document boundaries; no byte/token arrays are retained.
        """
        if isinstance(documents, (bytes, str)) or not documents:
            raise ValueError("documents must be a nonempty sequence of raw byte strings")
        if any(not isinstance(document, bytes) for document in documents):
            raise ValueError("documents must contain only raw bytes")
        if state is None:
            state = self.memory.initial_local_state(
                1, credit_mode="aggregate" if self.mode == "aggregate" else "tagged"
            )
        self._check_state_mode(state)
        if self.mode in {"writer_maps_frozen", "frozen"}:
            # These fixed control modes cannot consume writer eligibility. Keep
            # the allocated tag schema but omit dead trace/tag arithmetic. Mode
            # validation rejects later writer unfreezing, which would require
            # credit from writes that deliberately did not record it.
            self.memory._check_local_state(state)
            forward = PlasticMemoryState(state.synapses, state.cell.activity)
            for document in documents:
                values = self._encoded(document)
                mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
                forward = self.memory.write(values, mask, forward)
            result = LocalMemoryState(
                forward.synapses.clone(),
                self.memory.controller.initial_learning_state(1),
                {name: torch.zeros_like(value) for name, value in state.credit.items()},
                state.credit_mode,
            )
            self.memory._check_local_state(result)
            return result
        for document in documents:
            values = self._encoded(document)
            mask = torch.ones(values.shape[:2], dtype=torch.bool, device=values.device)
            state = self.memory.write_local(
                values, mask, state, writes_enabled=self.mode != "writes_disabled"
            )
        return self.memory.purge_local_writer(state)

    def _query(self, query: bytes, state: LocalMemoryState):
        self._check_state_mode(state)
        values = self._encoded(query)
        activity = self.query_cell.initial_learning_state(1)
        for inputs in values.unbind(1):
            activity = self.query_cell.local_step(inputs, activity)
        event = self.memory.read_local(activity.activity, state)
        return torch.cat((activity.activity, event.value), -1), activity, event

    @torch.no_grad()
    def _prediction_context(self, query: bytes, state: LocalMemoryState) -> Tensor:
        """Read-only forward context without unused teaching eligibility.

        Validate supplied state and every actual forward activity/read. Overflow
        solely in omitted query eligibility is no longer a prediction error;
        teaching still constructs and validates its original local traces.
        """
        self._check_state_mode(state)
        values = self._encoded(query)
        self.memory._check_local_state(state)
        if values.ndim != 3 or values.shape[0] != 1 or values.shape[2] != self.query_cell.input_size:
            raise ValueError("encoded query must be [1, tokens, query input size]")
        if (
            values.dtype != self.query_cell.weight.dtype
            or values.device != self.query_cell.weight.device
        ):
            raise ValueError("inputs dtype/device must match parameters")
        if not _all_finite(values):
            raise ValueError("active inputs must be finite")
        activity = self.query_cell.initial_state(1)
        for inputs in values.unbind(1):
            activity, _ = self.query_cell._step(inputs, activity)
            if not _all_finite(activity):
                raise ValueError("prediction activity must be finite")
        value = self.memory.read(activity, PlasticMemoryState(state.synapses, state.cell.activity))
        context = torch.cat((activity, value), -1)
        if not _all_finite(context):
            raise ValueError("context must be finite")
        return context

    @torch.no_grad()
    def predict(self, query: bytes, state: LocalMemoryState, max_bytes: int) -> ByteGeneration:
        """A genuine read-only attempt: no teacher input, traces or state retained."""
        context = self._prediction_context(query, state)
        return self.decoder.generate(context, max_bytes)[0]

    @torch.no_grad()
    def teach(
        self,
        query: bytes,
        teacher: bytes,
        state: LocalMemoryState,
        learning_rate: float,
        *,
        learning_enabled: bool = True,
    ) -> LocalLessonFeedback:
        """One explicitly authorized correction, distinct from first inference.

        The caller binds the teacher to an action and owns replay/episode counts.
        Query/writer/maps stay fixed during all teacher bytes. The decoder updates
        per byte/EOS; its BEFORE-update broadcasts are averaged once per lesson.
        That mean supplies upstream neuron signals, applied once. Fast memory is
        never reconstructed after a slow update. Returned state retires write and
        query credit; it must replace the caller's state for subsequent lessons.
        """
        if not isinstance(teacher, bytes):
            raise ValueError("teacher must be raw bytes")
        if not math.isfinite(learning_rate) or learning_rate < 0:
            raise ValueError("learning_rate must be finite and nonnegative")
        if type(learning_enabled) is not bool:
            raise ValueError("learning_enabled must be boolean")
        enabled = learning_enabled and self.mode != "frozen"
        context, query_state, read_event = self._query(query, state)
        total_signal = torch.zeros_like(context)
        losses = 0.0
        count = 0
        decoder_updated = False
        decoder_state = None
        previous = BYTE_BOS
        for target in (*teacher, BYTE_EOS):
            step = self.decoder.local_step(
                context, torch.tensor([previous], device=context.device), decoder_state
            )
            feedback = self.decoder.teach_local(
                step,
                torch.tensor([target], device=context.device),
                learning_rate,
                learning_enabled=enabled,
            )
            total_signal.add_(feedback.context_signal)
            losses += feedback.loss
            count += feedback.teacher_count
            decoder_updated = decoder_updated or feedback.updated
            decoder_state = step.state
            previous = target
        mean_signal = total_signal / count
        signals = {
            name: F.linear(mean_signal, getattr(self, f"{name}_feedback"))
            for name in ("controller", "key", "value", "strength", "query")
        }
        memory_updated = self.memory.apply_local_feedback(
            state, read_event, signals, learning_rate, learning_enabled=enabled
        )
        query_updated = self.query_cell.apply_learning_signal(
            query_state,
            mean_signal[:, : self.decoder.embedding_size],
            learning_rate,
            learning_enabled=enabled,
        )
        retired = self.memory.retire_local_traces(state)
        return LocalLessonFeedback(
            retired,
            count,
            losses / count,
            decoder_updated,
            query_updated,
            memory_updated,
            mean_signal,
        )

    def snapshot(self, state: LocalMemoryState) -> dict:
        """Detached slow/fixed parameters and source-derived fast/credit tensors.

        No documents, queries, targets, decoder activity or input graphs enter
        this transport. Credit is retained plasticity state and must be counted.
        """
        self._check_state_mode(state)
        return {"model": copy.deepcopy(self.state_dict()), "memory": state.snapshot()}

    def restore_snapshot(self, snapshot: dict) -> LocalMemoryState:
        if set(snapshot) != {"model", "memory"}:
            raise ValueError("snapshot must contain model and memory")
        state = self.memory.restore_local_state(snapshot["memory"])
        self._check_state_mode(state)
        self.load_state_dict(snapshot["model"], strict=True)
        return state
