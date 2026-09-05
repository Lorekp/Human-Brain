"""Batched token-pointer acquisition readers; no tokenizer, labels or actions inside forward.

All positions are original word positions; masks are right-padded. Endpoint
normalization includes one shared NULL, while decoding permits only ordered
within-source pairs. Dropout uses PyTorch's caller-owned RNG: resetting its state
before matched forwards supplies identical masks across the three producer kinds.
"""

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def _mask(mask: Tensor, shape: tuple[int, int]) -> None:
    if mask.dtype != torch.bool or tuple(mask.shape) != shape:
        raise ValueError("mask must be boolean with matching batch/token dimensions")
    if bool((mask[:, 1:] & ~mask[:, :-1]).any()):
        raise ValueError("masks must be right-padded")


class _SequenceStage(nn.Module):
    def __init__(self, kind: str, inputs: int, dropout: float):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        self.norm = nn.LayerNorm(inputs, elementwise_affine=False)
        if kind == "lstm":
            self.lstm = nn.LSTM(inputs, 64, batch_first=True, bidirectional=True)
        else:
            self.weight = nn.Parameter(torch.empty(2, 64, inputs))
            self.bias = nn.Parameter(torch.zeros(2, 64))
            if kind == "cell":
                self.recurrent = nn.Parameter(torch.empty(2, 64, 64))

    def reset_parameters(self) -> None:
        if self.kind == "lstm":
            for name, parameter in self.lstm.named_parameters():
                if name.startswith("weight_ih"):
                    nn.init.xavier_uniform_(parameter)
                elif name.startswith("weight_hh"):
                    # Each gate's recurrent transform is independently orthogonal.
                    for gate in parameter.chunk(4, dim=0):
                        nn.init.orthogonal_(gate)
                else:
                    nn.init.zeros_(parameter)
                    # PyTorch adds the two biases; set the forget offset only once.
                    if name.startswith("bias_ih"):
                        with torch.no_grad():
                            parameter[64:128].fill_(1)
        else:
            for weight in self.weight:
                nn.init.xavier_uniform_(weight)
            nn.init.zeros_(self.bias)
            if self.kind == "cell":
                with torch.no_grad():
                    self.recurrent.copy_(0.9 * torch.eye(64).expand(2, -1, -1))

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        batch, length, _ = values.shape
        if length == 0:
            return values.new_zeros(batch, 0, 128)
        values = self.norm(values.masked_fill(~mask[..., None], 0))
        if self.training and self.dropout:
            # One input mask per sequence, held across tokens and both directions.
            keep = F.dropout(torch.ones_like(values[:, :1]), self.dropout, True)
            values = values * keep
        if self.kind == "lstm":
            lengths = mask.sum(1)
            selected = torch.where(lengths > 0)[0]
            result = values.new_zeros(batch, length, 128)
            if selected.numel():
                packed = pack_padded_sequence(
                    values[selected],
                    lengths[selected].cpu(),
                    batch_first=True,
                    enforce_sorted=False,
                )
                packed_result, _ = self.lstm(packed)
                output, _ = pad_packed_sequence(
                    packed_result,
                    batch_first=True,
                    total_length=length,
                )
                result = result.index_copy(0, selected, output)
            return result
        drives = torch.einsum("bti,dhi->dbth", values, self.weight)
        drives = drives + self.bias[:, None, None, :]
        if self.kind == "bypass":
            return torch.cat(tuple(torch.tanh(drives)), -1).masked_fill(~mask[..., None], 0)
        directions = []
        for direction in range(2):
            state = values.new_zeros(batch, 64)
            positions = range(length) if direction == 0 else reversed(range(length))
            outputs = [None] * length
            for position in positions:
                activation = torch.tanh(
                    drives[direction, :, position] + F.linear(state, self.recurrent[direction])
                )
                candidate = 0.5 * state + 0.5 * activation
                active = mask[:, position, None]
                state = torch.where(active, candidate, state)
                outputs[position] = torch.where(active, state, 0)
            directions.append(torch.stack(outputs, 1))
        return torch.cat(directions, -1)


def _masked_softmax(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    has_values = mask.any(dim=dim, keepdim=True)
    safe = torch.where(has_values, values.masked_fill(~mask, -torch.inf), 0)
    return torch.softmax(safe, dim=dim).masked_fill(~mask, 0)


class TokenPointerReader(nn.Module):
    """Shared 64-cell-per-direction encoder and modeling stage with BiDAF fusion.

    Initialization: Xavier uniform input/endpoint/trilinear weights, zero biases,
    rate recurrence .9 I; LSTM recurrent gates orthogonal and total forget bias 1.
    Fixed LayerNorm (epsilon 1e-5) precedes each sequence input projection. Module
    construction preserves the caller's RNG. Use state_dict plus the declared
    kind/dropout and optimizer state for restoration; train/eval and frozen masks
    are caller-owned policy state, not encoded in a parameter checkpoint.
    """

    def __init__(self, kind: str = "cell", seed: int = 20260905, dropout: float = 0.2):
        super().__init__()
        if kind not in {"cell", "bypass", "lstm"}:
            raise ValueError("kind must be cell, bypass or lstm")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.kind, self.seed, self.dropout = kind, seed, dropout
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoder = _SequenceStage(kind, 384, dropout)
            self.modeling = _SequenceStage(kind, 512, dropout)
            self.similarity = nn.Parameter(torch.empty(3, 128))
            self.start_head = nn.Linear(128, 1)
            self.end_head = nn.Linear(128, 1)
            self.null_bias = nn.Parameter(torch.zeros(()))
            # Reset after construction so framework constructor draws do not
            # determine the documented initialization stream.
            torch.manual_seed(seed)
            self.encoder.reset_parameters()
            self.modeling.reset_parameters()
            nn.init.xavier_uniform_(self.similarity.view(1, -1))
            for head in (self.start_head, self.end_head):
                nn.init.xavier_uniform_(head.weight)
                nn.init.zeros_(head.bias)

    def freeze_producer(self) -> None:
        """Freeze both sequence stages and attention; retain training-time dropout."""
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(
                name == "null_bias" or name.startswith(("start_head.", "end_head."))
            )

    def forward(
        self,
        query: Tensor,
        query_mask: Tensor,
        context: Tensor,
        context_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        for values, mask in ((query, query_mask), (context, context_mask)):
            if values.ndim != 3 or values.shape[-1] != 384 or not values.is_floating_point():
                raise ValueError("word inputs must be floating [batch,tokens,384]")
            _mask(mask, tuple(values.shape[:2]))
            if values.device != mask.device or not bool(torch.isfinite(values[mask]).all()):
                raise ValueError("valid word inputs must be finite and colocated with mask")
        if (
            query.shape[0] != context.shape[0]
            or query.device != context.device
            or query.dtype != context.dtype
        ):
            raise ValueError("query/context batch, dtype and device must match")
        q = self.encoder(query, query_mask)
        h = self.encoder(context, context_mask)
        if query.shape[1] == 0 or context.shape[1] == 0:
            attended = torch.zeros_like(h)
            summary = torch.zeros_like(h)
        else:
            similarity = (
                (h @ self.similarity[0])[:, :, None]
                + (q @ self.similarity[1])[:, None, :]
                + (h * self.similarity[2]) @ q.transpose(1, 2)
            )
            attended = _masked_softmax(similarity, query_mask[:, None, :], -1) @ q
            strongest = similarity.masked_fill(~query_mask[:, None, :], -torch.inf).amax(-1)
            has_query = query_mask.any(-1)
            strongest = torch.where(has_query[:, None], strongest, 0)
            weights = _masked_softmax(strongest, context_mask, -1)
            summary = (weights[:, None, :] @ h).expand_as(h)
            summary = summary * has_query[:, None, None]
        fused = torch.cat((h, attended, h * attended, h * summary), -1)
        modeled = self.modeling(fused, context_mask)
        return tuple(
            head(modeled).squeeze(-1).masked_fill(~context_mask, -torch.inf)
            for head in (self.start_head, self.end_head)
        )


def _endpoints(start: Tensor, end: Tensor, mask: Tensor, null_bias: Tensor) -> Tensor:
    if start.ndim != 2 or start.shape != end.shape or not start.is_floating_point():
        raise ValueError("endpoint logits must have matching [batch,tokens] shape")
    _mask(mask, tuple(start.shape))
    if start.device != end.device or start.device != mask.device or start.dtype != end.dtype:
        raise ValueError("endpoint tensors must share device and dtype")
    null = torch.as_tensor(null_bias, dtype=start.dtype, device=start.device)
    if null.numel() != 1 or not bool(torch.isfinite(null).all()):
        raise ValueError("NULL must be one finite scalar")
    if not bool(torch.isfinite(start[mask]).all() & torch.isfinite(end[mask]).all()):
        raise ValueError("valid endpoint logits must be finite")
    return null.reshape(())


def pointer_loss(
    start: Tensor,
    end: Tensor,
    mask: Tensor,
    gold_pairs: Sequence[Sequence[tuple[int, int]] | None],
    null_bias: Tensor,
) -> Tensor:
    """Mean negative log mass of unique accepted inclusive pairs, or NULL/NULL.

    One row is one complete normalization inventory. A pooled caller may flatten
    sources for this training objective, but must supply source-custodied gold
    pairs and retain source boundaries when calling decode_pointer.
    """
    null = _endpoints(start, end, mask, null_bias)
    if len(gold_pairs) != start.shape[0] or not len(gold_pairs):
        raise ValueError("one target per nonempty batch row is required")
    probabilities = [
        F.log_softmax(
            torch.cat((logits.masked_fill(~mask, -torch.inf), null.expand(start.shape[0], 1)), -1),
            -1,
        )
        for logits in (start, end)
    ]
    losses = []
    for row, alternatives in enumerate(gold_pairs):
        if alternatives is None:
            losses.append(-probabilities[0][row, -1] - probabilities[1][row, -1])
            continue
        pairs = sorted(set(tuple(pair) for pair in alternatives))
        if not pairs:
            raise ValueError("supported targets require at least one accepted pair")
        for pair in pairs:
            if len(pair) != 2 or any(type(index) is not int for index in pair):
                raise ValueError("gold endpoints must be integer pairs")
            left, right = pair
            if not 0 <= left <= right < start.shape[1] or not bool(
                mask[row, left] & mask[row, right]
            ):
                raise ValueError("gold pair is outside valid ordered word positions")
        left, right = zip(*pairs, strict=True)
        joint = probabilities[0][row, list(left)] + probabilities[1][row, list(right)]
        losses.append(-torch.logsumexp(joint, 0))
    return torch.stack(losses).mean()


def decode_pointer(
    start: Tensor,
    end: Tensor,
    mask: Tensor,
    null_bias: Tensor,
) -> tuple[int, int, int] | None:
    """O(tokens) unique maximum legal occurrence versus NULL; read-only.

    Each row is one original source. Promote individual stored logits to Python
    float (float64 arithmetic) before sums and exact comparisons. Distinct tied
    occurrences STOP even if their eventual emitted byte strings are identical.
    """
    null = _endpoints(start, end, mask, null_bias)
    best = 2 * float(null.detach())
    winner, tied = None, True
    for document, (starts, ends, valid) in enumerate(
        zip(
            start.detach().cpu().tolist(),
            end.detach().cpu().tolist(),
            mask.cpu().tolist(),
            strict=True,
        )
    ):
        prefix, prefix_index, prefix_count = -math.inf, 0, 0
        for position, active in enumerate(valid):
            if not active:
                break
            if starts[position] > prefix:
                prefix, prefix_index, prefix_count = starts[position], position, 1
            elif starts[position] == prefix:
                prefix_count += 1
            score = prefix + ends[position]
            if not math.isfinite(score):
                raise ValueError("endpoint sum overflow")
            if score > best:
                best, winner, tied = score, (document, prefix_index, position), prefix_count != 1
            elif score == best:
                tied = True
    return None if tied else winner
