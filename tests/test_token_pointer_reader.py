"""Synthetic math/custody QA only; no acquisition or benchmark qualification."""

import copy
import math
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch", reason="optional pointer runtime is not installed")

from experiments.r4_integrated.token_pointer_runtime import (  # noqa: E402
    PointerDocument,
    PointerPolicy,
    batch_inputs,
    state_hash,
)
from humanbrain.token_pointer_reader import (  # noqa: E402
    TokenPointerReader,
    _SequenceStage,
    decode_pointer,
    pointer_loss,
)


def inputs():
    generator = torch.Generator().manual_seed(719)
    query = torch.randn(2, 2, 384, generator=generator, dtype=torch.float64)
    context = torch.randn(2, 4, 384, generator=generator, dtype=torch.float64)
    return (
        query,
        torch.tensor([[1, 1], [1, 0]], dtype=torch.bool),
        context,
        torch.tensor(
            [[1, 1, 1, 1], [1, 1, 0, 0]],
            dtype=torch.bool,
        ),
    )


@pytest.mark.parametrize("kind", ["cell", "bypass", "lstm"])
def test_seed_batch_padding_and_restore(kind):
    rng = torch.random.get_rng_state().clone()
    reader = TokenPointerReader(kind).double().eval()
    assert torch.equal(rng, torch.random.get_rng_state())
    clone = TokenPointerReader(kind, seed=31).double().eval()
    clone.load_state_dict(reader.state_dict())
    query, qm, context, cm = inputs()
    first = reader(query, qm, context, cm)
    for a, b in zip(first, clone(query, qm, context, cm), strict=True):
        assert torch.equal(a, b)
    # Padding is neither a recurrent input nor a bidirectional boundary change.
    query[~qm] = torch.nan
    context[~cm] = torch.nan
    for a, b in zip(first, reader(query, qm, context, cm), strict=True):
        assert torch.equal(a, b)
    alone = reader(query[1:, :1], qm[1:, :1], context[1:, :2], cm[1:, :2])
    for a, b in zip(first, alone, strict=True):
        torch.testing.assert_close(a[1, :2], b[0], atol=1e-12, rtol=1e-12)
    context[1, :2] *= 10
    changed = reader(query, qm, context, cm)
    for a, b in zip(first, changed, strict=True):
        torch.testing.assert_close(a[0], b[0], atol=1e-12, rtol=1e-12)


def test_cell_recurrence_matches_direct_bidirectional_reference():
    torch.manual_seed(18)
    stage = _SequenceStage("cell", 3, 0).double()
    stage.reset_parameters()
    x = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.bool)
    actual = stage(x, mask)
    expected = torch.zeros_like(actual)
    for batch, count in enumerate(mask.sum(1).tolist()):
        for direction in range(2):
            state = torch.zeros(64, dtype=x.dtype)
            positions = range(count) if direction == 0 else reversed(range(count))
            for position in positions:
                word = x[batch, position]
                normalized = (word - word.mean()) / torch.sqrt(word.var(unbiased=False) + 1e-5)
                activation = torch.tanh(
                    stage.weight[direction] @ normalized
                    + stage.recurrent[direction] @ state
                    + stage.bias[direction]
                )
                state = (state + activation) / 2
                expected[batch, position, direction * 64 : (direction + 1) * 64] = state
    torch.testing.assert_close(actual, expected, atol=1e-14, rtol=1e-13)
    actual.sum().backward()
    assert torch.equal(x.grad[~mask], torch.zeros_like(x.grad[~mask]))
    assert stage.recurrent.grad.abs().sum() > 0


def test_initialization_and_matched_locked_dropout(monkeypatch):
    readers = {kind: TokenPointerReader(kind).double() for kind in ("cell", "bypass", "lstm")}
    repeated = TokenPointerReader().double()
    assert all(
        torch.equal(v, repeated.state_dict()[k]) for k, v in readers["cell"].state_dict().items()
    )
    for stage in (readers["cell"].encoder, readers["cell"].modeling):
        torch.testing.assert_close(stage.recurrent, 0.9 * torch.eye(64).expand(2, -1, -1).double())
        assert not stage.bias.count_nonzero()
    for stage in (readers["lstm"].encoder, readers["lstm"].modeling):
        for suffix in ("", "_reverse"):
            bias = getattr(stage.lstm, "bias_ih_l0" + suffix) + getattr(
                stage.lstm, "bias_hh_l0" + suffix
            )
            assert torch.equal(bias[64:128], torch.ones(64, dtype=bias.dtype))
            assert not torch.cat((bias[:64], bias[128:])).count_nonzero()
    original = torch.nn.functional.dropout
    captured = []

    def record(values, *args, **kwargs):
        result = original(values, *args, **kwargs)
        captured.append(result.detach().clone())
        return result

    monkeypatch.setattr(torch.nn.functional, "dropout", record)
    q, qm, c, cm = inputs()
    for reader in readers.values():
        torch.manual_seed(91)
        reader(q, qm, c, cm)
    assert len(captured) == 9
    for stage in range(3):
        assert captured[stage].shape[1] == 1
        assert all(torch.equal(captured[stage], captured[stage + offset]) for offset in (3, 6))


def test_loss_unique_alternatives_null_and_analytic_gradient():
    start = torch.tensor(
        [[0.2, 0.4, -0.1], [0.3, -0.8, 4]], dtype=torch.float64, requires_grad=True
    )
    end = torch.tensor([[0.1, -0.3, 0.7], [0.1, 0.2, 8]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[1, 1, 1], [1, 1, 0]], dtype=torch.bool)
    null = torch.tensor(0.25, dtype=torch.float64, requires_grad=True)
    gold = [[(0, 1), (1, 2), (0, 1)], None]
    loss = pointer_loss(start, end, mask, gold, null)
    ps = torch.softmax(torch.cat((start[0], null[None])), 0)
    pe = torch.softmax(torch.cat((end[0], null[None])), 0)
    masses = torch.stack((ps[0] * pe[1], ps[1] * pe[2]))
    ns = torch.softmax(torch.cat((start[1, :2], null[None])), 0)
    ne = torch.softmax(torch.cat((end[1, :2], null[None])), 0)
    expected = -(masses.sum().log() + ns[-1].log() + ne[-1].log()) / 2
    torch.testing.assert_close(loss, expected)
    loss.backward()
    posterior = masses.detach() / masses.detach().sum()
    gs = ps.detach().clone()
    ge = pe.detach().clone()
    gs[:2] -= posterior
    ge[1:3] -= posterior
    torch.testing.assert_close(start.grad[0], gs[:3] / 2)
    torch.testing.assert_close(end.grad[0], ge[:3] / 2)
    expected_null = (gs[-1] + ge[-1] + ns[-1].detach() + ne[-1].detach() - 2) / 2
    torch.testing.assert_close(null.grad, expected_null)
    assert start.grad[1, 2] == end.grad[1, 2] == 0
    with pytest.raises(ValueError, match="outside"):
        pointer_loss(start, end, mask, [[(2, 1)], None], null)


@pytest.mark.parametrize("kind", ["cell", "bypass", "lstm"])
def test_end_to_end_float64_parameter_directional_derivatives(kind):
    reader = TokenPointerReader(kind, dropout=0).double().eval()
    q, qm, c, cm = inputs()
    gold = [[(1, 2), (2, 3)], None]
    loss = pointer_loss(*reader(q, qm, c, cm), cm, gold, reader.null_bias)
    loss.backward()
    # Whole-parameter random directions reach all input/recurrent/attention/head
    # coordinates, rather than checking only a convenient nonzero scalar.
    generator = torch.Generator().manual_seed(37)
    families = ["encoder", "modeling", "similarity", "start_head", "end_head", "null_bias"]
    step = 1e-5
    for family in families:
        selected = [(name, p) for name, p in reader.named_parameters() if name.startswith(family)]
        directions = [torch.randn(p.shape, generator=generator, dtype=p.dtype) for _, p in selected]
        norm = torch.sqrt(sum(d.square().sum() for d in directions))
        directions = [d / norm for d in directions]
        analytic = sum((p.grad * d).sum() for (_, p), d in zip(selected, directions, strict=True))
        originals = [p.detach().clone() for _, p in selected]
        values = []
        with torch.no_grad():
            for sign in (1, -1):
                for (_, p), original, direction in zip(
                    selected, originals, directions, strict=True
                ):
                    p.copy_(original + sign * step * direction)
                values.append(pointer_loss(*reader(q, qm, c, cm), cm, gold, reader.null_bias))
            for (_, p), original in zip(selected, originals, strict=True):
                p.copy_(original)
        numerical = (values[0] - values[1]) / (2 * step)
        torch.testing.assert_close(analytic, numerical, atol=1e-7, rtol=1e-4)


@pytest.mark.parametrize("kind", ["cell", "bypass", "lstm"])
def test_frozen_producer_optimizer_restore_and_training_dropout(kind):
    reader = TokenPointerReader(kind).double()
    reader.freeze_producer()
    q, qm, c, cm = inputs()
    optimizer = torch.optim.Adam([p for p in reader.parameters() if p.requires_grad], lr=0.001)
    before = {name: value.detach().clone() for name, value in reader.named_parameters()}
    torch.manual_seed(999)
    first = reader(q, qm, c, cm)
    torch.manual_seed(999)
    second = reader(q, qm, c, cm)
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))
    third = reader(q, qm, c, cm)
    assert any(not torch.equal(a, b) for a, b in zip(first, third, strict=True))
    pointer_loss(*first, cm, [[(0, 1)], None], reader.null_bias).backward()
    optimizer.step()
    for name, parameter in reader.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None
            assert torch.equal(parameter, before[name])
    assert any(
        not torch.equal(p, before[n]) for n, p in reader.named_parameters() if p.requires_grad
    )
    saved = copy.deepcopy((reader.state_dict(), optimizer.state_dict()))
    restored = TokenPointerReader(kind).double()
    restored.freeze_producer()
    restored.load_state_dict(saved[0])
    other_optimizer = torch.optim.Adam(
        [p for p in restored.parameters() if p.requires_grad], lr=0.001
    )
    other_optimizer.load_state_dict(saved[1])
    for model, opt in ((reader, optimizer), (restored, other_optimizer)):
        torch.manual_seed(101)
        opt.zero_grad(set_to_none=True)
        pointer_loss(*model(q, qm, c, cm), cm, [None, [(0, 1)]], model.null_bias).backward()
        opt.step()
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in reader.state_dict().items())


def test_decode_exhaustive_reference_ties_empty_and_long_span():
    generator = torch.Generator().manual_seed(12)
    start = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    end = torch.randn(3, 8, generator=generator, dtype=torch.float64)
    mask = torch.arange(8)[None, :] < torch.tensor([8, 5, 0])[:, None]
    null = torch.tensor(-0.3)
    all_pairs = [(2 * float(null), None)]
    for d, count in enumerate(mask.sum(1).tolist()):
        all_pairs.extend(
            (float(start[d, s]) + float(end[d, e]), (d, s, e))
            for s in range(count)
            for e in range(s, count)
        )
    leaders = [pair for score, pair in all_pairs if score == max(score for score, _ in all_pairs)]
    assert decode_pointer(start, end, mask, null) == (leaders[0] if len(leaders) == 1 else None)
    start = torch.tensor([[4.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0]])
    end = torch.tensor([[-3.0, -3.0, -3.0, -3.0, -3.0, -3.0, -3.0, 4.0]])
    mask = torch.ones_like(start, dtype=torch.bool)
    assert decode_pointer(start, end, mask, torch.tensor(0.0)) == (0, 0, 7)
    assert decode_pointer(start, end, mask, torch.tensor(4.0)) is None
    start[0, 1] = 4
    assert decode_pointer(start, end, mask, torch.tensor(0.0)) is None
    assert decode_pointer(start, end, torch.zeros_like(mask), torch.tensor(0.0)) is None
    assert decode_pointer(start[:0], end[:0], mask[:0], torch.tensor(0.0)) is None
    start[0, 0] = math.nan
    with pytest.raises(ValueError, match="finite"):
        decode_pointer(start, end, mask, torch.tensor(0.0))


@pytest.mark.parametrize("kind", ["cell", "bypass", "lstm"])
def test_empty_queries_context_and_mask_rejection(kind):
    reader = TokenPointerReader(kind).double().eval()
    q, qm, c, cm = inputs()
    start, end = reader(q[:, :0], qm[:, :0], c, cm)
    assert torch.isfinite(start[cm]).all() and torch.isfinite(end[cm]).all()
    start, end = reader(q, qm, c[:, :0], cm[:, :0])
    loss = pointer_loss(start, end, cm[:, :0], [None, None], reader.null_bias)
    assert loss == 0
    qm[1] = torch.tensor([False, True])
    with pytest.raises(ValueError, match="right-padded"):
        reader(q, qm, c, cm)


def word_inputs(length, seed):
    vectors = torch.randn(length, 384, generator=torch.Generator().manual_seed(seed)).numpy()
    return SimpleNamespace(words=tuple(range(length)), vectors=vectors)


@pytest.mark.parametrize(
    "kind,frozen_producer", [("cell", False), ("cell", True), ("bypass", False), ("lstm", False)]
)
def test_policy_full_snapshot_clone_rng_and_read_only_predict(kind, frozen_producer):
    policy = PointerPolicy(kind, frozen_producer=frozen_producer)
    query, context = word_inputs(2, 4), word_inputs(4, 7)
    outer_rng = torch.random.get_rng_state().clone()
    policy.update_batch([query], [context], [[(0, 2)]])
    assert torch.equal(outer_rng, torch.random.get_rng_state())
    snapshot = policy.checkpoint()
    saved_hash = state_hash(snapshot)
    restored = PointerPolicy.restore(snapshot)
    clone = policy.clone()
    frozen = policy.clone(frozen=True)
    assert (
        saved_hash
        == policy.checkpoint_sha256
        == restored.checkpoint_sha256
        == clone.checkpoint_sha256
    )
    for name, parameter in policy.reader.named_parameters():
        assert parameter.data_ptr() != dict(restored.reader.named_parameters())[name].data_ptr()
    assert policy.rng.data_ptr() != restored.rng.data_ptr()
    for state in policy.optimizer.state.values():
        assert state["step"] == 1
    frozen_before = frozen.checkpoint_sha256
    with pytest.raises(ValueError, match="enabled"):
        frozen.update_batch([query], [context], [None])
    assert frozen.checkpoint_sha256 == frozen_before
    document = PointerDocument(b"source", b"provenance", context)
    first_prediction = policy.predict(query, [document])
    assert policy.predict(query, [document]) == first_prediction
    assert policy.predict_local_batch([query], [context])[0] == first_prediction
    assert policy.checkpoint_sha256 == saved_hash  # action adapter's checkpoint property
    assert len(saved_hash) == 64
    receipts = [p.update_batch([query], [context], [None]) for p in (policy, restored, clone)]
    assert receipts[0] == receipts[1] == receipts[2]
    assert policy.checkpoint_sha256 == restored.checkpoint_sha256 == clone.checkpoint_sha256
    assert state_hash(snapshot) == saved_hash  # both model and warmed Adam snapshot stay unchanged
    assert policy.steps == policy.presentations == 2
    assert all(state["step"] == 2 for state in policy.optimizer.state.values())
    assert frozen.checkpoint_sha256 == frozen_before
    if frozen_producer:
        assert all(
            torch.equal(parameter, snapshot["model"][name])
            for name, parameter in policy.reader.named_parameters()
            if not parameter.requires_grad
        )
        snapshot["requires_grad"]["similarity"] = True
    else:
        snapshot["dropout"] = 0
    with pytest.raises(ValueError, match="architecture"):
        PointerPolicy.restore(snapshot)


def test_policies_advance_matched_private_dropout_streams():
    policies = [
        PointerPolicy(),
        PointerPolicy(frozen_producer=True),
        PointerPolicy("bypass"),
        PointerPolicy("lstm"),
    ]
    queries = [word_inputs(2, 11), word_inputs(1, 12)]
    contexts = [word_inputs(4, 13), word_inputs(2, 14)]
    outer = torch.random.get_rng_state().clone()
    for policy in policies:
        policy.update_batch(queries, contexts, [[(1, 3)], None])
    assert all(torch.equal(p.rng, policies[0].rng) for p in policies)
    assert torch.equal(outer, torch.random.get_rng_state())
    assert all(p.steps == 1 and p.presentations == 2 for p in policies)


def manual_policy_gradients(policy, query, qm, context, cm, targets, pooled=False):
    """Same-state autograd reference: per-example log normalizers, then global mean."""
    policy.reader.train()
    policy.optimizer.zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[]):
        torch.random.set_rng_state(policy.rng)
        start, end = policy.reader(query, qm, context, cm)
        if pooled:
            start = torch.cat(
                [row[: int(length)] for row, length in zip(start, cm.sum(1), strict=True)]
            )[None]
            end = torch.cat(
                [row[: int(length)] for row, length in zip(end, cm.sum(1), strict=True)]
            )[None]
            cm = torch.ones_like(start, dtype=torch.bool)
        losses = []
        for row, target in enumerate(targets):
            valid_start, valid_end = start[row, cm[row]], end[row, cm[row]]
            norm_start = torch.logsumexp(torch.cat((valid_start, policy.reader.null_bias[None])), 0)
            norm_end = torch.logsumexp(torch.cat((valid_end, policy.reader.null_bias[None])), 0)
            if target is None:
                gold_score = 2 * policy.reader.null_bias
            else:
                gold_score = torch.logsumexp(
                    torch.stack([valid_start[a] + valid_end[b] for a, b in sorted(set(target))]), 0
                )
            losses.append(norm_start + norm_end - gold_score)
        loss = torch.stack(losses).mean()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            policy.reader.parameters(), 1, error_if_nonfinite=True
        )
        gradients = {name: p.grad.detach().clone() for name, p in policy.reader.named_parameters()}
        policy.rng = torch.random.get_rng_state().clone()
    policy.reader.eval()
    policy.optimizer.zero_grad(set_to_none=True)
    return loss.detach(), norm, gradients


def capture_step_gradients(monkeypatch, policy):
    gradients = {}
    original_step = policy.optimizer.step

    def step():
        gradients.update(
            {name: p.grad.detach().clone() for name, p in policy.reader.named_parameters()}
        )
        return original_step()

    monkeypatch.setattr(policy.optimizer, "step", step)
    return gradients


def check_gradients_and_exact_adam(policy, reference, expected, captured):
    # Different equivalent float32 objective expressions incur rounding. Check
    # local derivatives directly; near-zero gradients must not be mistaken for
    # an exact independently ordered Adam trajectory. Adam receives the captured
    # actual gradients in the separate exact optimizer-application check below.
    for name, parameter in reference.reader.named_parameters():
        torch.testing.assert_close(captured[name], expected[name], atol=1e-7, rtol=1e-5)
        parameter.grad = captured[name].clone()
    reference.optimizer.step()
    for name, parameter in policy.reader.named_parameters():
        assert torch.equal(parameter, dict(reference.reader.named_parameters())[name])


def test_update_batch_is_one_clipped_global_mean_step(monkeypatch):
    policy = PointerPolicy()
    reference = policy.clone()
    queries = [word_inputs(2, 111), word_inputs(1, 112), word_inputs(2, 113)]
    contexts = [word_inputs(4, 114), word_inputs(2, 115), word_inputs(3, 116)]
    gold = [[(1, 3), (2, 3)], None, [(0, 1)]]
    query, qm = batch_inputs(queries)
    context, cm = batch_inputs(contexts)
    expected_loss, expected_norm, expected_gradients = manual_policy_gradients(
        reference, query, qm, context, cm, gold
    )
    captured = capture_step_gradients(monkeypatch, policy)
    receipt = policy.update_batch(queries, contexts, gold)
    assert receipt["loss"] == pytest.approx(float(expected_loss), abs=1e-6)
    assert receipt["gradient_norm_before_clip"] == pytest.approx(float(expected_norm), rel=1e-6)
    assert expected_norm > 1
    check_gradients_and_exact_adam(policy, reference, expected_gradients, captured)
    assert policy.steps == 1 and policy.presentations == 3
    assert all(state["step"] == 1 for state in policy.optimizer.state.values())
    assert torch.equal(policy.rng, reference.rng)


def test_pooled_unpadding_gold_offsets_and_source_boundary_rejection(monkeypatch):
    policy = PointerPolicy()
    query = word_inputs(2, 121)
    documents = [
        PointerDocument(b"one", b"p1", word_inputs(2, 122)),
        PointerDocument(b"two", b"p2", word_inputs(4, 123)),
    ]
    before = policy.checkpoint_sha256
    with pytest.raises(ValueError, match="original source"):
        policy.update_pool(query, documents, [(0, 1, 2)])
    assert policy.checkpoint_sha256 == before
    reference = policy.clone()
    q, qm = batch_inputs([query, query])
    c, cm = batch_inputs([document.tokens for document in documents])
    expected_loss, expected_norm, expected_gradients = manual_policy_gradients(
        reference, q, qm, c, cm, [[(3, 5)]], pooled=True
    )
    captured = capture_step_gradients(monkeypatch, policy)
    receipt = policy.update_pool(query, documents, [(1, 1, 3)])
    assert receipt["loss"] == pytest.approx(float(expected_loss), abs=1e-6)
    assert receipt["gradient_norm_before_clip"] == pytest.approx(float(expected_norm), rel=1e-6)
    check_gradients_and_exact_adam(policy, reference, expected_gradients, captured)
    assert policy.steps == policy.presentations == 1  # one inventory, not two lessons
