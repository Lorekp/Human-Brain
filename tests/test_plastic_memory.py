"""Synthetic state and derivative checks; no prose acquisition qualification."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional pointer runtime is not installed")

from humanbrain.plastic_memory import (  # noqa: E402
    PlasticMemory,
    PlasticMemoryState,
    StreamingRateCell,
)


@pytest.mark.parametrize("kind", ["cell", "bypass"])
def test_local_rule_independent_scalar_oracle_and_delayed_feedback(kind):
    cell = StreamingRateCell(2, 2, kind, seed=41).double()
    if kind == "cell":
        with torch.no_grad():
            cell.recurrent.copy_(torch.tensor([[0.7, -0.3], [0.2, 0.6]]))
    inputs = np.array(
        [[[0.3, -0.2], [0.5, 0.1]], [[-0.4, 0.7], [0.2, -0.5]], [[0.8, 0.3], [-0.2, 0.4]]]
    )
    w, bias = cell.weight.detach().numpy().copy(), cell.bias.detach().numpy().copy()
    u = cell.recurrent.detach().numpy().copy() if kind == "cell" else np.zeros((2, 2))
    h = np.zeros((2, 2))
    ew, eu, eb = np.zeros((2, 2, 2)), np.zeros((2, 2, 2)), np.zeros((2, 2))
    state = cell.initial_learning_state(2)
    for x in inputs:
        previous = h.copy()
        for b in range(2):
            for j in range(2):
                a = np.tanh(
                    sum(w[j, i] * x[b, i] for i in range(2))
                    + bias[j]
                    + sum(u[j, i] * previous[b, i] for i in range(2))
                )
                d = (0.5 if kind == "cell" else 1.0) * (1 - a * a)
                carry = 0.5 + d * u[j, j] if kind == "cell" else 0.0
                h[b, j] = previous[b, j] + 0.5 * (a - previous[b, j]) if kind == "cell" else a
                for i in range(2):
                    ew[b, j, i] = carry * ew[b, j, i] + d * x[b, i]
                    eu[b, j, i] = carry * eu[b, j, i] + d * previous[b, i]
                eb[b, j] = carry * eb[b, j] + d
        state = cell.local_step(torch.from_numpy(x).requires_grad_(), state)
        np.testing.assert_allclose(state.activity, h, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(state.input_eligibility, ew, atol=1e-14, rtol=1e-13)
        np.testing.assert_allclose(state.bias_eligibility, eb, atol=1e-14, rtol=1e-13)
        if kind == "cell":
            np.testing.assert_allclose(state.recurrent_eligibility, eu, atol=1e-14, rtol=1e-13)
        else:
            assert state.recurrent_eligibility is None
    saved = state.snapshot()
    signal = torch.tensor([[0.7, -0.6], [-0.2, 0.8]], dtype=torch.float64, requires_grad=True)
    cell.bias.requires_grad_(False)
    assert cell.apply_learning_signal(state, signal, 0.03)
    np.testing.assert_allclose(
        cell.weight.detach(),
        w - 0.03 * (signal.detach().numpy()[:, :, None] * ew).mean(0),
        atol=1e-14,
    )
    if kind == "cell":
        np.testing.assert_allclose(
            cell.recurrent.detach(),
            u - 0.03 * (signal.detach().numpy()[:, :, None] * eu).mean(0),
            atol=1e-14,
        )
    assert torch.equal(cell.bias, torch.from_numpy(bias))
    assert signal.grad is None
    assert all(p.grad is None for p in cell.parameters())
    for name, value in state.snapshot().items():
        if value is not None:
            assert torch.equal(value, saved[name])
            assert not value.requires_grad and value.grad_fn is None


def test_local_eligibility_matches_diagonal_recurrence_finite_difference():
    cell = StreamingRateCell(2, 2, seed=5).double()
    xs = np.array([[0.4, -0.7], [0.1, 0.6], [-0.3, 0.9]])
    state = cell.initial_learning_state(1)
    for x in xs:
        state = cell.local_step(torch.from_numpy(x[None]), state)
    arrays = {name: p.detach().numpy().copy() for name, p in cell.named_parameters()}

    def reference():
        h = np.zeros(2)
        for x in xs:
            a = np.tanh(arrays["weight"] @ x + arrays["recurrent"] @ h + arrays["bias"])
            h = h + 0.5 * (a - h)
        return h

    trace_names = {
        "weight": "input_eligibility",
        "recurrent": "recurrent_eligibility",
        "bias": "bias_eligibility",
    }
    for name, array in arrays.items():
        trace = getattr(state, trace_names[name])[0].numpy()
        for index in np.ndindex(array.shape):
            old = array[index]
            array[index] = old + 1e-6
            plus = reference()[index[0]]
            array[index] = old - 1e-6
            minus = reference()[index[0]]
            array[index] = old
            assert trace[index] == pytest.approx((plus - minus) / 2e-6, abs=1e-9, rel=1e-7)


@pytest.mark.parametrize("kind", ["cell", "bypass"])
def test_local_masking_freeze_snapshot_reset_and_no_history(kind):
    cell = StreamingRateCell(2, 3, kind, seed=2).double()
    x = torch.tensor([[0.2, -0.1], [0.5, 0.8]], dtype=torch.float64, requires_grad=True)
    state = cell.local_step(x)
    original = state.snapshot()
    next_x = torch.tensor([[float("nan"), float("nan")], [0.9, -0.2]], dtype=torch.float64)
    next_state = cell.local_step(next_x, state, mask=torch.tensor([False, True]))
    for name, value in vars(next_state).items():
        if value is not None:
            assert torch.equal(value[0], original[name][0])
            assert value.data_ptr() != getattr(state, name).data_ptr()
            assert not value.requires_grad and value.grad_fn is None
    before = copy.deepcopy(cell.state_dict())
    signal = torch.tensor([[float("nan")] * 3, [0.2, -0.4, 0.6]], dtype=torch.float64)
    mask = torch.tensor([False, True])
    assert not cell.apply_learning_signal(
        next_state, signal, 0.1, mask=mask, learning_enabled=False
    )
    assert not cell.apply_learning_signal(next_state, signal, 0.0, mask=mask)
    assert not cell.apply_learning_signal(
        next_state, signal, 0.1, mask=torch.zeros(2, dtype=torch.bool)
    )
    cell.requires_grad_(False)
    assert not cell.apply_learning_signal(next_state, signal, 0.1, mask=mask)
    for name, value in cell.state_dict().items():
        assert torch.equal(value, before[name])
    cell.requires_grad_(True)
    assert cell.apply_learning_signal(next_state, signal, 0.1, mask=mask)
    expected = before["weight"] - 0.1 * signal[1, :, None] * next_state.input_eligibility[1]
    torch.testing.assert_close(cell.weight, expected, rtol=1e-14, atol=1e-15)
    restored = cell.restore_learning_state(next_state.snapshot())
    for name, value in vars(restored).items():
        if value is not None:
            assert torch.equal(value, getattr(next_state, name))
            assert value.data_ptr() != getattr(next_state, name).data_ptr()
    replica = StreamingRateCell(2, 3, kind).double()
    replica.load_state_dict(copy.deepcopy(cell.state_dict()))
    continued = replica.local_step(x, restored)
    expected_state = cell.local_step(x, next_state)
    for name, value in vars(continued).items():
        if value is not None:
            assert torch.equal(value, getattr(expected_state, name))
    reset = cell.reset_eligibility(restored)
    assert torch.equal(reset.activity, restored.activity)
    assert not torch.count_nonzero(reset.input_eligibility)
    assert not torch.count_nonzero(reset.bias_eligibility)
    assert not torch.count_nonzero(
        cell.reset_eligibility(restored, preserve_activity=False).activity
    )
    tampered = restored.snapshot()
    tampered["activity"].requires_grad_()
    with pytest.raises(ValueError, match="autograd"):
        cell.restore_learning_state(tampered)
    assert x.grad is None


@pytest.mark.parametrize("kind", ["cell", "bypass"])
def test_streaming_arithmetic_private_rng_and_restore(kind):
    rng = torch.random.get_rng_state().clone()
    cell = StreamingRateCell(2, 3, kind, seed=91).double()
    assert torch.equal(rng, torch.random.get_rng_state())
    x = torch.tensor([[0.2, -0.7], [0.8, 0.3]], dtype=torch.float64)
    previous = torch.tensor([[0.1, 0.5, -0.3], [0.2, -0.1, 0.6]], dtype=torch.float64)
    drive = x.numpy() @ cell.weight.detach().numpy().T + cell.bias.detach().numpy()
    if kind == "cell":
        np.testing.assert_array_equal(cell.recurrent.detach().numpy(), np.eye(3) * np.float32(0.9))
        activation = np.tanh(drive + previous.numpy() @ cell.recurrent.detach().numpy().T)
        expected = previous.numpy() + 0.5 * (activation - previous.numpy())
    else:
        assert "recurrent" not in dict(cell.named_parameters())
        expected = np.tanh(drive)
        torch.testing.assert_close(cell(x, previous), cell(x, -previous), rtol=0, atol=0)
    np.testing.assert_allclose(cell(x, previous).detach(), expected, rtol=1e-14, atol=1e-14)
    restored = StreamingRateCell(2, 3, kind, seed=3).double()
    restored.load_state_dict(copy.deepcopy(cell.state_dict()))
    assert torch.equal(cell(x, previous), restored(x, previous))
    assert torch.count_nonzero(cell.initial_state(2)) == 0


def test_seed_matched_bypass_and_explicit_state_only():
    rng = torch.random.get_rng_state().clone()
    cell = PlasticMemory(2, 3, 2, 2, seed=19)
    bypass = PlasticMemory(2, 3, 2, 2, kind="bypass", seed=19)
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, value in bypass.state_dict().items():
        assert torch.equal(value, cell.state_dict()[name])
    before = copy.deepcopy(cell.state_dict())
    state = cell.write(torch.ones(2, 3, 2), torch.ones(2, 3, dtype=torch.bool))
    assert set(vars(state)) == {"synapses", "writer"}
    assert state.synapses.shape == (2, 2, 2)
    assert not list(cell.buffers())
    for name, value in cell.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("kind", ["cell", "bypass"])
def test_masking_chunk_continuation_and_batch_isolation(kind):
    model = PlasticMemory(2, 3, 2, 2, kind=kind, seed=7).double()
    generator = torch.Generator().manual_seed(11)
    values = torch.randn(2, 5, 2, generator=generator, dtype=torch.float64)
    mask = torch.tensor([[1, 0, 1, 1, 0], [0, 1, 0, 0, 1]], dtype=torch.bool)
    values[~mask] = torch.nan
    values.requires_grad_()
    before = copy.deepcopy(model.state_dict())
    full = model.write(values, mask)
    split = model.write(values[:, :2], mask[:, :2])
    split = model.write(values[:, 2:], mask[:, 2:], split)
    assert torch.equal(full.synapses, split.synapses)
    assert torch.equal(full.writer, split.writer)
    for row in range(2):
        compact = model.write(
            values[row : row + 1, mask[row]], torch.ones(1, int(mask[row].sum()), dtype=torch.bool)
        )
        torch.testing.assert_close(full.synapses[row], compact.synapses[0], rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(full.writer[row], compact.writer[0], rtol=1e-13, atol=1e-13)
    read = model.read(torch.ones(2, 2, dtype=torch.float64), full)
    read.sum().backward()
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad[~mask]) == 0
    unchanged = model.write(values, torch.zeros_like(mask), full)
    assert torch.equal(unchanged.synapses, full.synapses)
    assert torch.equal(unchanged.writer, full.writer)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_snapshot_purge_restore_and_writes_off_causality(tmp_path):
    model = PlasticMemory(2, 3, 2, 2, seed=12).double()
    values = torch.tensor([[[0.5, -0.2], [-0.1, 0.8]]], dtype=torch.float64, requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)
    query = torch.tensor([[0.4, -0.7]], dtype=torch.float64)
    live = model.write(values, mask)
    assert live.synapses.grad_fn is not None
    snapshot = live.snapshot()
    assert all(not x.requires_grad and x.grad_fn is None for x in snapshot.values())
    path = tmp_path / "synapses.pt"
    torch.save(snapshot, path)
    restored = model.restore_state(torch.load(path, weights_only=True))
    assert restored.synapses.data_ptr() != snapshot["synapses"].data_ptr()
    purged = model.purge_writer(restored)
    assert torch.count_nonzero(purged.writer) == 0
    assert torch.equal(model.read(query, live), model.read(query, purged))
    disabled = model.write(-values, mask, purged, writes_enabled=False)
    assert torch.equal(disabled.synapses, purged.synapses)
    assert not torch.equal(disabled.writer, purged.writer)
    assert torch.equal(model.read(query, disabled), model.read(query, purged))
    empty_disabled = model.write(values, mask, writes_enabled=False)
    assert torch.count_nonzero(model.read(query, empty_disabled)) == 0
    assert torch.count_nonzero(empty_disabled.writer) > 0
    snapshot["synapses"].zero_()
    assert torch.count_nonzero(restored.synapses) > 0
    with pytest.raises(ValueError, match="only synapses"):
        model.restore_state({**restored.snapshot(), "inputs": values})
    with pytest.raises(ValueError, match="shape"):
        model.restore_state(
            {"synapses": torch.zeros(1, 3, 2, dtype=torch.float64), "writer": purged.writer}
        )
    del live, values, snapshot
    assert torch.isfinite(
        model.read(query, model.restore_state(torch.load(path, weights_only=True)))
    ).all()


def test_delta_correction_and_key_interference_are_visible():
    model = PlasticMemory(2, 2, 2, 1, kind="bypass", seed=2).double()
    with torch.no_grad():
        model.controller.weight.copy_(torch.eye(2))
        model.controller.bias.zero_()
        model.key_weight.copy_(torch.eye(2))
        model.key_bias.zero_()
        model.query_weight.copy_(torch.eye(2))
        model.query_bias.zero_()
        model.value_weight.copy_(torch.tensor([[0.8, -0.6]]))
        model.value_bias.zero_()
        model.strength_weight.zero_()
        model.strength_bias.zero_()
    x = torch.tensor([[[0.7, 0.0]]], dtype=torch.float64)
    active = torch.ones(1, 1, dtype=torch.bool)
    one = model.write(x, active)
    target = torch.tanh(0.8 * torch.tanh(x[0, 0, 0]))
    q = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    assert model.read(q, one).item() == pytest.approx(0.5 * target.item())
    two = model.write(x, active, one)
    assert model.read(q, two).item() == pytest.approx(0.75 * target.item())
    orthogonal = model.write(torch.tensor([[[0.0, 0.6]]], dtype=torch.float64), active, one)
    assert torch.equal(model.read(q, orthogonal), model.read(q, one))
    competing = model.write(torch.tensor([[[0.7, 0.7]]], dtype=torch.float64), active, one)
    assert not torch.equal(model.read(q, competing), model.read(q, one))


def _reference(parameters, values, mask, query, synapses, writer):
    """Scalar-stream NumPy forward, independent of Torch's batched implementation."""
    synapses, writer = synapses.copy(), writer.copy()
    for batch in range(len(values)):
        for position in range(values.shape[1]):
            if not mask[batch, position]:
                continue
            drive = parameters["controller.weight"] @ values[batch, position]
            drive += (
                parameters["controller.bias"] + parameters["controller.recurrent"] @ writer[batch]
            )
            writer[batch] += 0.5 * (np.tanh(drive) - writer[batch])
            key = parameters["key_weight"] @ writer[batch] + parameters["key_bias"]
            key /= max(np.linalg.norm(key), 1e-8)
            value = np.tanh(parameters["value_weight"] @ writer[batch] + parameters["value_bias"])
            gate = parameters["strength_weight"] @ writer[batch] + parameters["strength_bias"]
            strength = 1 / (1 + np.exp(-gate))
            synapses[batch] += strength * np.outer(value - synapses[batch] @ key, key)
    reads = []
    for batch in range(len(query)):
        key = parameters["query_weight"] @ query[batch] + parameters["query_bias"]
        key /= max(np.linalg.norm(key), 1e-8)
        reads.append(synapses[batch] @ key)
    reads = np.stack(reads)
    objective = (reads * np.array([[0.8, -0.3], [-0.2, 0.6]])).sum()
    objective += 0.17 * (synapses**2).sum() + 0.11 * (writer**2).sum()
    return objective, reads, synapses, writer


def test_independent_multiwrite_read_gradients_all_coordinates():
    model = PlasticMemory(2, 3, 2, 2, seed=15).double()
    rng = np.random.default_rng(231)
    values, query = rng.normal(size=(2, 3, 2)) * 0.3, rng.normal(size=(2, 2)) * 0.3
    synapses, writer = rng.normal(size=(2, 2, 2)) * 0.1, rng.normal(size=(2, 3)) * 0.1
    mask = np.array([[True, True, True], [True, False, True]])
    tensors = [
        torch.tensor(x, dtype=torch.float64, requires_grad=True)
        for x in (values, query, synapses, writer)
    ]
    state = model.write(tensors[0], torch.tensor(mask), PlasticMemoryState(tensors[2], tensors[3]))
    result = model.read(tensors[1], state)
    objective = (result * torch.tensor([[0.8, -0.3], [-0.2, 0.6]], dtype=torch.float64)).sum()
    objective = (
        objective + 0.17 * state.synapses.square().sum() + 0.11 * state.writer.square().sum()
    )
    objective.backward()
    parameters = {name: value.detach().numpy().copy() for name, value in model.named_parameters()}
    expected = _reference(parameters, values, mask, query, synapses, writer)
    for actual, reference in zip(
        (objective, result, state.synapses, state.writer), expected, strict=True
    ):
        np.testing.assert_allclose(actual.detach().numpy(), reference, atol=1e-14, rtol=1e-13)
    arrays = [*parameters.values(), values, query, synapses, writer]
    gradients = [*(p.grad for p in model.parameters()), *(t.grad for t in tensors)]
    for array, gradient in zip(arrays, gradients, strict=True):
        finite_difference = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + 1e-6
            plus = _reference(parameters, values, mask, query, synapses, writer)[0]
            array[index] = original - 1e-6
            minus = _reference(parameters, values, mask, query, synapses, writer)[0]
            array[index] = original
            finite_difference[index] = (plus - minus) / 2e-6
        np.testing.assert_allclose(gradient.numpy(), finite_difference, atol=1e-7, rtol=2e-5)


def test_empty_invalid_inputs_and_frozen_parameters():
    model = PlasticMemory(2, 3, 2, 2, seed=4).double()
    empty = model.write(
        torch.empty(1, 0, 2, dtype=torch.float64), torch.empty(1, 0, dtype=torch.bool)
    )
    assert torch.count_nonzero(empty.synapses) == 0
    with pytest.raises(ValueError, match="boolean"):
        model.write(torch.ones(1, 2, 2, dtype=torch.float64), torch.ones(1, 2))
    with pytest.raises(ValueError, match="finite"):
        model.write(
            torch.full((1, 1, 2), torch.inf, dtype=torch.float64),
            torch.ones(1, 1, dtype=torch.bool),
        )
    model.requires_grad_(False)
    before = copy.deepcopy(model.state_dict())
    x = torch.ones(1, 2, 2, dtype=torch.float64, requires_grad=True)
    state = model.write(x, torch.ones(1, 2, dtype=torch.bool))
    model.read(torch.ones(1, 2, dtype=torch.float64), state).sum().backward()
    assert x.grad is not None
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())
