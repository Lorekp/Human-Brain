"""Learned autoregressive bytes from a context vector, with no output grammar.

Bytes 0..255 and EOS 256 are predicted; BOS 257 is input-only. The caller owns
the context producer, optimizer, and training examples. Generation never repairs
UTF-8, punctuation, field structure, or a missing EOS.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from humanbrain.plastic_memory import StreamingRateCell

BYTE_EOS = 256
BYTE_BOS = 257


@dataclass(frozen=True)
class ByteGeneration:
    data: bytes
    stop_reason: Literal["eos", "limit"]


def byte_teacher_forcing(
    outputs: Sequence[bytes], *, device: torch.device | str | None = None
) -> tuple[Tensor, Tensor, Tensor]:
    """Return BOS-shifted inputs, byte-plus-EOS targets, and an active-token mask.

    Empty bytes have one active EOS target. Padding uses byte zero but contributes
    no loss; actual zero bytes remain active. No text encoding or normalization.
    """
    if not outputs or any(not isinstance(value, bytes) for value in outputs):
        raise ValueError("outputs must be a nonempty sequence of bytes")
    length = max(map(len, outputs)) + 1
    inputs = torch.zeros((len(outputs), length), dtype=torch.long, device=device)
    targets = torch.zeros_like(inputs)
    mask = torch.zeros_like(inputs, dtype=torch.bool)
    inputs[:, 0] = BYTE_BOS
    for row, value in enumerate(outputs):
        tokens = torch.tensor(list(value), dtype=torch.long, device=device)
        inputs[row, 1 : len(value) + 1] = tokens
        targets[row, : len(value)] = tokens
        targets[row, len(value)] = BYTE_EOS
        mask[row, : len(value) + 1] = True
    return inputs, targets, mask


def byte_sequence_loss(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
    """Mean cross-entropy over active byte/EOS targets, including empty outputs."""
    if logits.ndim != 3 or logits.shape[-1] != BYTE_EOS + 1:
        raise ValueError("logits must have shape [batch, time, 257]")
    if targets.dtype != torch.long or targets.shape != logits.shape[:2]:
        raise ValueError("targets must be int64 with matching batch/time dimensions")
    if mask.dtype != torch.bool or mask.shape != targets.shape:
        raise ValueError("mask must be boolean with matching target dimensions")
    if targets.device != logits.device or mask.device != logits.device:
        raise ValueError("logits, targets and mask must share a device")
    if not mask.numel() or not bool(mask[:, 0].all()):
        raise ValueError("every sequence must include at least EOS")
    if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("mask must be right-padded")
    active = targets[mask]
    if bool(((active < 0) | (active > BYTE_EOS)).any()):
        raise ValueError("active targets must be bytes or EOS")
    last = mask.sum(1) - 1
    eos_positions = torch.zeros_like(mask).scatter_(1, last[:, None], True)
    if not torch.equal((targets == BYTE_EOS) & mask, eos_positions):
        raise ValueError("each sequence must end with exactly one active EOS")
    selected = logits[mask]
    if not bool(torch.isfinite(selected).all()):
        raise ValueError("active logits must be finite")
    return F.cross_entropy(selected, active)


class ByteSequenceDecoder(nn.Module):
    """A shared streaming rate cell conditioned on context at every byte step.

    Parameters initialize reproducibly without advancing the caller's CPU RNG.
    Standard ``state_dict`` includes configuration and all learned tensors.
    Hidden state starts at zero for each call and is never retained on the module.
    Greedy generation resolves exact logit ties by the lowest token index.
    """

    def __init__(
        self,
        context_size: int,
        hidden_size: int = 64,
        embedding_size: int = 32,
        *,
        kind: str = "cell",
        seed: int = 0,
    ) -> None:
        super().__init__()
        if any(
            type(size) is not int or size < 1
            for size in (context_size, hidden_size, embedding_size)
        ):
            raise ValueError("context, hidden and embedding sizes must be positive integers")
        if kind not in {"cell", "bypass"} or type(seed) is not int:
            raise ValueError("kind must be cell/bypass and seed must be an integer")
        self.context_size = context_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.kind = kind
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.embedding = nn.Embedding(BYTE_BOS + 1, embedding_size)
            self.cell = StreamingRateCell(
                context_size + embedding_size, hidden_size, kind=kind, seed=seed + 1
            )
            self.output = nn.Linear(hidden_size, BYTE_EOS + 1)

    def get_extra_state(self) -> dict:
        return {
            "schema": "byte-sequence-decoder-v1",
            "context_size": self.context_size,
            "hidden_size": self.hidden_size,
            "embedding_size": self.embedding_size,
            "kind": self.kind,
        }

    def set_extra_state(self, state: dict) -> None:
        if state != self.get_extra_state():
            raise ValueError("decoder checkpoint configuration mismatch")

    def _validate_context(self, context: Tensor) -> None:
        if context.ndim != 2 or context.shape[1] != self.context_size or not context.shape[0]:
            raise ValueError("context must have nonempty shape [batch, context_size]")
        if (
            context.dtype != self.embedding.weight.dtype
            or context.device != self.embedding.weight.device
        ):
            raise ValueError("context must match the decoder's floating dtype and device")
        if not bool(torch.isfinite(context).all()):
            raise ValueError("context must be finite")

    def _step(self, context: Tensor, tokens: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        state = self.cell(torch.cat((context, self.embedding(tokens)), dim=-1), state)
        return self.output(state), state

    def forward(self, context: Tensor, input_tokens: Tensor) -> Tensor:
        """Predict next-token logits after each supplied input, without shifting it."""
        self._validate_context(context)
        if input_tokens.ndim != 2 or input_tokens.shape[0] != context.shape[0]:
            raise ValueError("input_tokens must have shape [batch, time]")
        if input_tokens.dtype != torch.long or input_tokens.device != context.device:
            raise ValueError("input_tokens must be int64 on the context device")
        if bool(((input_tokens < 0) | (input_tokens > BYTE_BOS)).any()):
            raise ValueError("inputs must be bytes, EOS or BOS")
        state = self.cell.initial_state(
            context.shape[0], device=context.device, dtype=context.dtype
        )
        logits = []
        for tokens in input_tokens.unbind(1):
            values, state = self._step(context, tokens, state)
            logits.append(values)
        return torch.stack(logits, 1) if logits else context.new_empty(context.shape[0], 0, 257)

    def loss(self, context: Tensor, outputs: Sequence[bytes]) -> Tensor:
        inputs, targets, mask = byte_teacher_forcing(outputs, device=context.device)
        return byte_sequence_loss(self(context, inputs), targets, mask)

    @torch.no_grad()
    def generate(self, context: Tensor, max_bytes: int) -> tuple[ByteGeneration, ...]:
        """Emit at most max_bytes per row; one final decision can confirm EOS.

        A non-EOS prediction at the cap yields ``limit`` without appending that
        extra byte. EOS is never synthesized. Returned bytes may be invalid UTF-8.
        No training/eval mode, parameter, persistent state or RNG is changed.
        """
        self._validate_context(context)
        if type(max_bytes) is not int or max_bytes < 0:
            raise ValueError("max_bytes must be a nonnegative integer")
        batch = context.shape[0]
        state = self.cell.initial_state(batch, device=context.device, dtype=context.dtype)
        tokens = torch.full((batch,), BYTE_BOS, dtype=torch.long, device=context.device)
        outputs = [bytearray() for _ in range(batch)]
        reasons: list[Literal["eos", "limit"] | None] = [None] * batch
        for position in range(max_bytes + 1):
            logits, state = self._step(context, tokens, state)
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("generation logits must be finite")
            tokens = logits.argmax(-1)
            for row, token in enumerate(tokens.tolist()):
                if reasons[row] is not None:
                    continue
                if token == BYTE_EOS:
                    reasons[row] = "eos"
                elif position == max_bytes:
                    reasons[row] = "limit"
                else:
                    outputs[row].append(token)
            if all(reason is not None for reason in reasons):
                break
        return tuple(
            ByteGeneration(bytes(value), reason)
            for value, reason in zip(outputs, reasons, strict=True)
            if reason is not None
        )
