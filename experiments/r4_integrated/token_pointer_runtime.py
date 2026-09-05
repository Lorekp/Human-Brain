"""Small shared training/state seam for the declared token-pointer comparison.

Labels enter update/scoring only. Predictions receive original inputs and source
boundaries. Each policy owns its dropout RNG; probes and action execution leave
parameters, optimizer, counters and RNG unchanged.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from hashlib import sha256

import torch

from humanbrain.token_pointer_reader import (
    TokenPointerReader,
    decode_answerable_pointer,
    decode_pointer,
    pointer_loss,
    separate_answerability_loss,
)

SEED = 20260905


@dataclass(frozen=True, slots=True)
class PointerDocument:
    document_id: bytes
    provenance_id: bytes
    tokens: object


def state_hash(value) -> str:
    """Canonical recursive hash, including optimizer tensors and policy flags."""
    digest = sha256()

    def add(item):
        if isinstance(item, torch.Tensor):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(json.dumps(["tensor", str(array.dtype), list(array.shape)]).encode())
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"{")
            for key in sorted(item, key=lambda k: (type(k).__name__, str(k))):
                add(key)
                add(item[key])
            digest.update(b"}")
        elif isinstance(item, (list, tuple)):
            digest.update(b"[")
            for child in item:
                add(child)
            digest.update(b"]")
        else:
            digest.update(json.dumps(item, allow_nan=False).encode() + b"\n")

    add(value)
    return digest.hexdigest()


def batch_inputs(words):
    if not words:
        raise ValueError("nonempty batch required")
    lengths = [len(word.words) for word in words]
    values = torch.zeros(len(words), max(lengths), 384, dtype=torch.float32)
    mask = torch.arange(max(lengths))[None, :] < torch.tensor(lengths)[:, None]
    for row, word in enumerate(words):
        # Copy only this bounded batch from immutable arrays/memmaps.
        values[row, : lengths[row]].copy_(torch.tensor(word.vectors))
    return values, mask


class PointerPolicy:
    def __init__(
        self, kind="cell", *, frozen_producer=False, seed=SEED, objective="joint_null"
    ):
        self.reader = TokenPointerReader(kind, seed=seed, objective=objective).eval()
        self.frozen_producer = frozen_producer
        if frozen_producer:
            self.reader.freeze_producer()
        self.optimizer = torch.optim.Adam(
            [p for p in self.reader.parameters() if p.requires_grad],
            lr=0.001,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0,
        )
        self.rng = torch.Generator().manual_seed(seed + 1).get_state()
        self.steps = self.presentations = 0
        self.updating = True

    @property
    def objective(self):
        return self.reader.objective

    def checkpoint(self, *, copy_tensors=True):
        """Return an independent restorable snapshot; hash-only callers may borrow views."""
        state = {
            "kind": self.reader.kind,
            "seed": self.reader.seed,
            "dropout": self.reader.dropout,
            "training": self.reader.training,
            "frozen_producer": self.frozen_producer,
            "requires_grad": {n: p.requires_grad for n, p in self.reader.named_parameters()},
            "model": self.reader.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng": self.rng,
            "steps": self.steps,
            "presentations": self.presentations,
            "updating": self.updating,
        }
        # Historical joint-NULL checkpoints retain their exact original preimage.
        if self.objective != "joint_null":
            state["objective"] = self.objective
        return copy.deepcopy(state) if copy_tensors else state

    @property
    def checkpoint_sha256(self):
        return state_hash(self.checkpoint(copy_tensors=False))

    @classmethod
    def restore(cls, state):
        policy = cls(
            state["kind"],
            frozen_producer=state["frozen_producer"],
            seed=state["seed"],
            objective=state.get("objective", "joint_null"),
        )
        if state["dropout"] != policy.reader.dropout or state["requires_grad"] != {
            n: p.requires_grad for n, p in policy.reader.named_parameters()
        }:
            raise ValueError("checkpoint architecture or frozen parameter policy differs")
        policy.reader.load_state_dict(state["model"], strict=True)
        policy.optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        policy.reader.train(state["training"])
        policy.rng = state["rng"].clone()
        policy.steps, policy.presentations = state["steps"], state["presentations"]
        policy.updating = state["updating"]
        if policy.checkpoint_sha256 != state_hash(state):
            raise ValueError("restored complete policy state differs")
        return policy

    def clone(self, *, frozen=False):
        result = self.restore(self.checkpoint())
        if frozen:
            result.updating = False
        return result

    def _update(self, query, qm, context, cm, gold, *, pooled=False):
        if not self.updating or self.reader.training:
            raise ValueError("update requires an enabled policy at its evaluation boundary")
        self.optimizer.zero_grad(set_to_none=True)
        self.reader.train()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.random.set_rng_state(self.rng)
                if self.objective == "separate_answerability":
                    start, end, answerability = self.reader.forward_with_answerability(
                        query, qm, context, cm, pooled=pooled
                    )
                else:
                    start, end = self.reader(query, qm, context, cm)
                if pooled:
                    start, end = start[cm][None, :], end[cm][None, :]
                    cm = torch.ones_like(start, dtype=torch.bool)
                if self.objective == "separate_answerability":
                    loss = separate_answerability_loss(start, end, cm, gold, answerability)
                else:
                    loss = pointer_loss(start, end, cm, gold, self.reader.null_bias)
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.reader.parameters() if p.requires_grad],
                    1.0,
                    error_if_nonfinite=True,
                )
                self.optimizer.step()
                self.rng = torch.random.get_rng_state().clone()
            self.steps += 1
            self.presentations += len(gold)
            return {"loss": float(loss.detach()), "gradient_norm_before_clip": float(norm)}
        finally:
            self.reader.eval()
            self.optimizer.zero_grad(set_to_none=True)

    def update_batch(self, queries, contexts, gold):
        query, qm = batch_inputs(queries)
        context, cm = batch_inputs(contexts)
        return self._update(query, qm, context, cm, gold)

    def update_pool(self, query, documents, gold):
        """Gold is None or (source index, inclusive start, inclusive end) alternatives."""
        documents = tuple(documents)
        q, qm = batch_inputs([query] * len(documents))
        c, cm = batch_inputs([d.tokens for d in documents])
        target = None
        if gold is not None:
            counts = cm.sum(1).tolist()
            offsets = [sum(counts[:i]) for i in range(len(counts))]
            target = []
            for d, a, b in gold:
                if not 0 <= d < len(documents) or not 0 <= a <= b < counts[d]:
                    raise ValueError("pooled gold is outside its original source")
                target.append((offsets[d] + a, offsets[d] + b))
        return self._update(q, qm, c, cm, [target], pooled=True)

    @torch.no_grad()
    def predict(self, query, documents):
        if self.reader.training:
            raise ValueError("prediction requires dropout disabled")
        documents = tuple(documents)
        q, qm = batch_inputs([query] * len(documents))
        c, cm = batch_inputs([d.tokens for d in documents])
        if self.objective == "separate_answerability":
            start, end, z = self.reader.forward_with_answerability(q, qm, c, cm, pooled=True)
            return decode_answerable_pointer(start, end, cm, z)
        return decode_pointer(*self.reader(q, qm, c, cm), cm, self.reader.null_bias)

    @torch.no_grad()
    def predict_local_batch(self, queries, contexts):
        if self.reader.training:
            raise ValueError("prediction requires dropout disabled")
        q, qm = batch_inputs(queries)
        c, cm = batch_inputs(contexts)
        if self.objective == "separate_answerability":
            start, end, z = self.reader.forward_with_answerability(q, qm, c, cm)
            return [
                decode_answerable_pointer(
                    start[i : i + 1], end[i : i + 1], cm[i : i + 1], z[i]
                )
                for i in range(len(queries))
            ]
        start, end = self.reader(q, qm, c, cm)
        return [
            decode_pointer(start[i : i + 1], end[i : i + 1], cm[i : i + 1], self.reader.null_bias)
            for i in range(len(queries))
        ]


def local_examples(cases, documents, get_word):
    queries = [get_word(sha256(row["question"].encode()).hexdigest()) for row in cases]
    contexts = [
        get_word(sha256(documents[row["paragraph_id"]]["context"].encode()).hexdigest())
        for row in cases
    ]
    gold = [
        None
        if row["is_impossible"]
        else sorted({(a["word_start"], a["word_end"] - 1) for a in row["answers"]})
        for row in cases
    ]
    return queries, contexts, gold


def score_local(policy, cases, documents, get_word):
    before = policy.checkpoint_sha256
    rows = []
    for offset in range(0, len(cases), 32):
        batch = cases[offset : offset + 32]
        queries, contexts, gold = local_examples(batch, documents, get_word)
        predictions = policy.predict_local_batch(queries, contexts)
        for case, context, target, selected in zip(batch, contexts, gold, predictions, strict=True):
            occurrence = None
            if selected is not None:
                _, a, b = selected
                start, end = context.word_starts[a], context.word_ends[b]
                occurrence = {
                    "paragraph_id": case["paragraph_id"],
                    "word_start": a,
                    "word_end_inclusive": b,
                    "byte_start": start,
                    "byte_end": end,
                    "emitted_hex": context.source[start:end].hex(),
                }
            rows.append(
                {
                    "id": case["id"],
                    "supported": target is not None,
                    "correct": selected is None
                    if target is None
                    else (selected is not None and (selected[1], selected[2]) in target),
                    "occurrence": occurrence,
                }
            )
    if policy.checkpoint_sha256 != before:
        raise RuntimeError("read-only local probes changed policy state")
    return rows


def score_counts(rows):
    return {
        "supported": sum(row["correct"] and row["supported"] for row in rows),
        "STOP": sum(row["correct"] and not row["supported"] for row in rows),
        "supported_total": sum(row["supported"] for row in rows),
        "STOP_total": sum(not row["supported"] for row in rows),
    }
