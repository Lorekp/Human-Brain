"""Synthetic state and derivative checks; no prose acquisition qualification."""

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional pointer runtime is not installed")

from humanbrain.plastic_memory import (  # noqa: E402
    LocalMemoryState,
    PlasticMemory,
    PlasticMemoryState,
    StreamingRateCell,
)


def _unbuffered_tag_reference(model, values, mask, state):
    """Old allocating tag expression; cell/map arithmetic has separate oracle tests."""
    with torch.no_grad():
        for index in range(values.shape[1]):
            stepped = model.write_local(
                values[:, index : index + 1], mask[:, index : index + 1], state
            )
            h = stepped.cell.activity
            key, gain = model._normalized_local_map(
                torch.nn.functional.linear(h, model.key_weight, model.key_bias)
            )
            value = torch.tanh(torch.nn.functional.linear(h, model.value_weight, model.value_bias))
            strength = torch.sigmoid(
                torch.nn.functional.linear(h, model.strength_weight, model.strength_bias)
            )
            beta = strength * mask[:, index, None]
            traces = {
                "controller.weight": stepped.cell.input_eligibility,
                "controller.recurrent": stepped.cell.recurrent_eligibility,
                "controller.bias": stepped.cell.bias_eligibility,
            }
            for name, derivative in (
                ("key", gain),
                ("value", 1 - value.square()),
                ("strength", strength * (1 - strength)),
            ):
                traces[name + "_weight"] = derivative[:, :, None] * h[:, None, :]
                traces[name + "_bias"] = derivative
            credit = {}
            for name, old in state.credit.items():
                flat = old.reshape(values.shape[0], -1, model.key_size)
                prediction = (flat * key[:, None, :]).sum(-1)
                e = traces[name].reshape(values.shape[0], -1)
                updated = flat + (beta * (e - prediction))[:, :, None] * key[:, None, :]
                credit[name] = updated.reshape_as(old)
                assert torch.equal(stepped.credit[name], credit[name]), name
            state = LocalMemoryState(stepped.synapses, stepped.cell, credit, "tagged")
    return state


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_buffered_tags_exact_repeated_write_feedback_and_noncontiguous_restore(dtype):
    model = PlasticMemory(32, 64, 64, 64, seed=733).to(dtype=dtype)
    reference_model = copy.deepcopy(model)
    generator = torch.Generator().manual_seed(57)
    state = model.initial_local_state(2)
    reference_state = reference_model.initial_local_state(2)
    for _ in range(2):
        # Same logical values with noncontiguous tag storage exercise reshape copies.
        snapshot = state.snapshot()
        snapshot["credit"] = {
            name: value.transpose(-1, -2).contiguous().transpose(-1, -2)
            for name, value in snapshot["credit"].items()
        }
        state = model.restore_local_state(snapshot)
        reference_state = reference_model.restore_local_state(snapshot)
        assert any(not value.is_contiguous() for value in state.credit.values())
        before = state.snapshot()
        values = torch.randn(2, 8, 32, generator=generator, dtype=dtype) * 0.2
        mask = torch.tensor([[1, 1, 0, 1, 1, 1, 1, 1], [1, 0, 1, 1, 1, 0, 1, 1]], dtype=torch.bool)
        values[~mask] = torch.nan
        result = model.write_local(values, mask, state)
        reference_state = _unbuffered_tag_reference(reference_model, values, mask, reference_state)
        assert torch.equal(result.synapses, reference_state.synapses)
        for name, value in result.credit.items():
            assert torch.equal(value, reference_state.credit[name]), name
            assert torch.equal(state.credit[name], before["credit"][name])
            assert value.data_ptr() != state.credit[name].data_ptr()
        query = torch.randn(2, 32, generator=generator, dtype=dtype)
        signals = {
            name: torch.randn(2, size, generator=generator, dtype=dtype) * 0.1
            for name, size in {
                "controller": 64,
                "key": 64,
                "value": 64,
                "strength": 1,
                "query": 64,
            }.items()
        }
        model.apply_local_feedback(result, model.read_local(query, result), signals, 0.001)
        reference_model.apply_local_feedback(
            reference_state, reference_model.read_local(query, reference_state), signals, 0.001
        )
        assert all(
            torch.equal(value, reference_model.state_dict()[name])
            for name, value in model.state_dict().items()
        )
        state = model.purge_local_writer(result)
        reference_state = reference_model.purge_local_writer(reference_state)


def test_buffered_tag_nonfinite_rejection_preserves_caller_state_and_slow_weights():
    model = PlasticMemory(2, 2, 2, 2, seed=13)
    with torch.no_grad():
        model.key_weight.zero_()
        model.key_bias.fill_(1)
    state = model.initial_local_state(1)
    state.credit["controller.weight"].fill_(torch.finfo(torch.float32).max)
    before = state.snapshot()
    parameters = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="finite"):
        model.write_local(torch.ones(1, 1, 2), torch.ones(1, 1, dtype=torch.bool), state)
    assert torch.equal(state.synapses, before["synapses"])
    assert all(torch.equal(value, before["credit"][name]) for name, value in state.credit.items())
    assert all(torch.equal(value, parameters[name]) for name, value in model.state_dict().items())


def test_local_write_rejects_intermediate_cell_overflow_before_masked_suffix():
    model = PlasticMemory(2, 2, 2, 2, seed=13)
    with torch.no_grad():
        model.controller.recurrent.copy_(torch.eye(2) * torch.finfo(torch.float32).max)
    state = model.initial_local_state(1)
    state.cell.input_eligibility.fill_(4)
    before = state.snapshot()
    # Output validation still runs at every private step, even with writes off.
    with pytest.raises(ValueError, match="finite"):
        model.write_local(
            torch.zeros(1, 2, 2), torch.tensor([[True, False]]), state, writes_enabled=False
        )
    for name, value in vars(state.cell).items():
        assert torch.equal(value, before["cell"][name])
    assert torch.equal(state.synapses, before["synapses"])
    assert all(torch.equal(value, before["credit"][name]) for name, value in state.credit.items())


@pytest.mark.parametrize("credit_mode", ["tagged", "aggregate"])
def test_local_memory_overwrite_and_query_retrieval_scalar_oracle(credit_mode):
    model = PlasticMemory(2, 2, 2, 2, kind="bypass", seed=31).double()
    with torch.no_grad():
        model.controller.weight.copy_(torch.eye(2))
        model.key_weight.copy_(torch.eye(2))
        model.query_weight.copy_(torch.eye(2))
        model.value_weight.copy_(torch.eye(2))
        model.strength_weight.zero_()
    state = model.initial_local_state(1, credit_mode=credit_mode)
    xs = np.array([[0.8, 0.0], [0.0, -0.6], [0.4, 0.0]])
    memory = np.zeros((2, 2))
    tags = np.zeros((2, 2)) if credit_mode == "tagged" else np.zeros(2)
    for x in xs:
        h = np.tanh(x)
        key = h / np.linalg.norm(h)
        value = np.tanh(h)
        e = 1 - h * h
        old_memory, old_tags = memory.copy(), tags.copy()
        for i in range(2):
            predicted = sum(old_memory[i, k] * key[k] for k in range(2))
            for k in range(2):
                memory[i, k] = old_memory[i, k] + 0.5 * (value[i] - predicted) * key[k]
            if credit_mode == "tagged":
                tagged_prediction = sum(old_tags[i, k] * key[k] for k in range(2))
                for k in range(2):
                    tags[i, k] = old_tags[i, k] + 0.5 * (e[i] - tagged_prediction) * key[k]
            else:
                tags[i] = old_tags[i] + 0.5 * (e[i] - old_tags[i])
        state = model.write_local(
            torch.from_numpy(x[None, None]), torch.ones(1, 1, dtype=torch.bool), state
        )
        np.testing.assert_allclose(state.synapses[0], memory, atol=1e-14)
        np.testing.assert_allclose(state.credit["controller.bias"][0], tags, atol=1e-14)
    # Each query reads the same M; credit is never a decoder/read input.
    event = model.read_local(torch.tensor([[1.0, 0.0]], dtype=torch.float64), state)
    np.testing.assert_allclose(event.value[0], memory[:, 0], atol=1e-14)
    before = model.controller.bias.detach().clone()
    signals = {
        "controller": torch.ones(1, 2, dtype=torch.float64),
        "key": torch.zeros(1, 2, dtype=torch.float64),
        "value": torch.zeros(1, 2, dtype=torch.float64),
        "strength": torch.zeros(1, 1, dtype=torch.float64),
        "query": torch.zeros(1, 2, dtype=torch.float64),
    }
    assert model.apply_local_feedback(state, event, signals, 0.03)
    expected_e = tags[:, 0] if credit_mode == "tagged" else tags
    np.testing.assert_allclose(
        model.controller.bias.detach(), before.numpy() - 0.03 * expected_e, atol=1e-14
    )
    np.testing.assert_allclose(state.synapses[0], memory, atol=1e-14)


@pytest.mark.parametrize("credit_mode", ["tagged", "aggregate"])
def test_local_memory_all_groups_feedback_controls_and_retirement(credit_mode):
    model = PlasticMemory(2, 3, 2, 2, seed=9).double()
    values = torch.tensor(
        [[[0.4, -0.2], [0.8, 0.3]], [[-0.3, 0.7], [0.1, -0.9]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    initial = model.initial_local_state(2, credit_mode=credit_mode)
    before = copy.deepcopy(model.state_dict())
    state = model.write_local(values, torch.ones(2, 2, dtype=torch.bool), initial)
    query = torch.tensor(
        [[0.6, -0.4], [float("nan"), float("nan")]], dtype=torch.float64, requires_grad=True
    )
    event = model.read_local(query, state, mask=torch.tensor([True, False]))
    signals = {
        name: torch.tensor([[0.3] * size, [float("nan")] * size], dtype=torch.float64)
        for name, size in {"controller": 3, "key": 2, "value": 2, "strength": 1, "query": 2}.items()
    }
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert not model.apply_local_feedback(state, event, signals, 0.01, learning_enabled=False)
    model.requires_grad_(False)
    assert not model.apply_local_feedback(state, event, signals, 0.01)
    model.requires_grad_(True)
    assert model.apply_local_feedback(state, event, signals, 0.01)
    for name, value in model.state_dict().items():
        assert torch.isfinite(value).all()
        assert not torch.equal(value, before[name]), name
    assert all(p.grad is None for p in model.parameters())
    assert values.grad is None and query.grad is None
    restored = model.restore_local_state(state.snapshot())
    restored_event = model.restore_local_read(event.snapshot())
    assert torch.equal(restored.synapses, state.synapses)
    assert restored.synapses.data_ptr() != state.synapses.data_ptr()
    for name, tensor in restored.credit.items():
        assert torch.equal(tensor, state.credit[name])
        assert tensor.data_ptr() != state.credit[name].data_ptr()
        assert tensor.grad_fn is None and not tensor.requires_grad
    assert torch.equal(restored_event.value, event.value)
    purged = model.purge_local_writer(restored)
    assert not torch.count_nonzero(purged.cell.activity)
    assert not torch.count_nonzero(purged.cell.input_eligibility)
    assert torch.equal(purged.synapses, state.synapses)
    for name, value in purged.credit.items():
        assert torch.equal(value, state.credit[name])
    retired = model.retire_local_traces(purged)
    assert all(not torch.count_nonzero(tensor) for tensor in retired.credit.values())
    assert torch.equal(retired.synapses, state.synapses)
    assert torch.equal(
        model.read_local(torch.nan_to_num(query), retired).value,
        model.read_local(torch.nan_to_num(query), purged).value,
    )


def test_local_memory_padding_disabled_writes_and_reference_parity():
    model = PlasticMemory(2, 3, 2, 2, seed=6).double()
    values = torch.tensor(
        [[[0.2, -0.4], [float("nan"), float("nan")], [0.9, 0.1]]], dtype=torch.float64
    )
    mask = torch.tensor([[True, False, True]])
    state = model.write_local(values, mask)
    reference = model.write(values, mask)
    torch.testing.assert_close(state.synapses, reference.synapses, atol=0, rtol=0)
    torch.testing.assert_close(state.cell.activity, reference.writer, atol=0, rtol=0)
    compact = model.write_local(values[:, mask[0]], torch.ones(1, 2, dtype=torch.bool))
    for name in state.credit:
        assert torch.equal(state.credit[name], compact.credit[name])
    blocked = model.write_local(values, mask, writes_enabled=False)
    assert not torch.count_nonzero(blocked.synapses)
    assert all(not torch.count_nonzero(t) for t in blocked.credit.values())
    assert torch.equal(blocked.cell.activity, state.cell.activity)
    assert torch.equal(
        model.read_local(torch.ones(1, 2, dtype=torch.float64), blocked).value,
        torch.zeros(1, 2, dtype=torch.float64),
    )
    unchanged = model.write_local(values, torch.zeros_like(mask), state)
    assert torch.equal(unchanged.synapses, state.synapses)
    for name, tensor in unchanged.credit.items():
        assert torch.equal(tensor, state.credit[name])
    with pytest.raises(ValueError, match="five neuron"):
        model.apply_local_feedback(
            state, model.read_local(torch.ones(1, 2, dtype=torch.float64), state), {}, 0.1
        )


def test_local_normalization_diagonal_approximation():
    drive = torch.tensor([[0.3, -0.4], [0.0, 0.0]], dtype=torch.float64)
    key, gain = PlasticMemory._normalized_local_map(drive)
    torch.testing.assert_close(key[0], torch.tensor([0.6, -0.8], dtype=torch.float64))
    torch.testing.assert_close(gain[0], torch.tensor([1.28, 0.72], dtype=torch.float64))
    assert torch.equal(gain[1], torch.full((2,), 1e8, dtype=torch.float64))
    for j in range(2):
        plus, minus = drive[0].clone(), drive[0].clone()
        plus[j] += 1e-6
        minus[j] -= 1e-6
        fd = ((plus / plus.norm())[j] - (minus / minus.norm())[j]) / 2e-6
        assert float(gain[0, j]) == pytest.approx(float(fd), abs=1e-9)


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
