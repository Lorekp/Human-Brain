"""Synthetic state and derivative checks; no prose acquisition qualification."""

import ast
import copy
import subprocess
from unittest.mock import patch

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="optional pointer runtime is not installed")

from humanbrain.plastic_memory import (  # noqa: E402
    LocalMemoryState,
    PlasticMemory,
    PlasticMemoryState,
    StreamingRateCell,
    _all_finite,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_all_finite_cpu_views_are_exact_read_only_detached_and_do_not_mutate(dtype):
    base = torch.arange(16, dtype=dtype).reshape(4, 4).requires_grad_()
    views = [base, base.T, base[:, ::2], base[:1, :1].expand(3, 4),
             base.as_strided((3, 3), (1, 1)), base[:0]]
    numpy_finite = np.isfinite
    for special in (torch.finfo(dtype).max, float("nan"), float("inf"), -float("inf")):
        with torch.no_grad():
            base[0, 0] = special
        for view in views:
            before = view.detach().clone()
            expected = bool(torch.isfinite(view).all())

            def checked_array(array, view=view):
                assert not array.flags.writeable
                if view.numel():
                    assert array.ctypes.data == view.data_ptr()
                return numpy_finite(array)

            with patch("humanbrain.plastic_memory.np.isfinite", side_effect=checked_array):
                assert _all_finite(view) is expected
            torch.testing.assert_close(view, before, rtol=0, atol=0, equal_nan=True)
            assert view.requires_grad and base.grad is None
    # Read-only ndarray flags never freeze or retain the caller's tensor storage.
    with torch.no_grad():
        base.fill_(3)
    assert _all_finite(base)


def test_all_finite_unsupported_dtype_layout_device_and_view_use_torch_fallback():
    tensors = [
        torch.tensor([1, 2], dtype=dtype)
        for dtype in (torch.float16, torch.bfloat16, torch.int64, torch.bool, torch.complex64)
    ]
    tensors.extend([
        torch._neg_view(torch.tensor([1.0, float("inf")])),
        torch.tensor([1.0 + 2.0j, complex(float("inf"), 0)]).conj(),
        torch.tensor([1.0, 0.0]).to_sparse(),
        torch.empty(2, device="meta"),
    ])
    original = torch.isfinite
    for value in tensors:
        try:
            expected = bool(original(value).all())
        except (RuntimeError, NotImplementedError) as error:
            with patch.object(torch, "isfinite", wraps=original) as fallback:
                with pytest.raises(type(error)) as captured:
                    _all_finite(value)
                assert str(captured.value) == str(error)
        else:
            with patch.object(torch, "isfinite", wraps=original) as fallback:
                assert _all_finite(value) is expected
        fallback.assert_called_once_with(value)


@pytest.mark.parametrize("error", [TypeError("unsupported conversion"), RuntimeError("view bits")])
def test_all_finite_conversion_failure_preserves_original_torch_predicate(error):
    value = torch.tensor([1.0, float("nan")], requires_grad=True)
    with patch.object(torch.Tensor, "numpy", side_effect=error):
        assert not _all_finite(value)
    assert value.requires_grad and value.grad is None


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
        [[[0.2, -0.4], [float("nan"), float("inf")], [0.9, 0.1]]], dtype=torch.float64
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_read_path_signals_scalar_chain_and_raw_map_update(dtype):
    model = PlasticMemory(2, 3, 2, 2, seed=9).to(dtype=dtype)
    with torch.no_grad():
        model.query_weight.copy_(torch.tensor([[0.7, -0.2], [0.4, 0.8]], dtype=dtype))
        model.query_bias.copy_(torch.tensor([0.1, -0.3], dtype=dtype))
    state = model.initial_local_state(2)
    state.synapses[0].copy_(torch.tensor([[0.3, -0.6], [0.2, 0.8]], dtype=dtype))
    query = torch.tensor([[0.6, -0.4], [torch.nan, torch.nan]], dtype=dtype)
    read_error = torch.tensor([[0.8, -0.5], [torch.inf, torch.nan]], dtype=dtype)
    event = model.read_local(query, state, mask=torch.tensor([True, False]))
    before = copy.deepcopy(model.state_dict())
    state_before = state.snapshot()
    error, indirect = model.read_learning_signals(state, event, read_error)
    # Independent scalar chain for one active stream; no autodiff or local helper.
    qweight = before["query_weight"].double().numpy()
    bias, u = before["query_bias"].double().numpy(), query[0].double().numpy()
    drive = np.array([sum(qweight[j, i] * u[i] for i in range(2)) + bias[j] for j in range(2)])
    magnitude = np.sqrt(sum(x*x for x in drive))
    key = drive / magnitude
    matrix, signal = state.synapses[0].double().numpy(), read_error[0].double().numpy()
    key_error = np.array([sum(matrix[i, j] * signal[i] for i in range(2)) for j in range(2)])
    expected = (key_error - key * sum(key[j] * key_error[j] for j in range(2))) / magnitude
    expected_indirect = np.array([
        sum(qweight[j, i] * expected[j] for j in range(2)) for i in range(2)
    ])
    tol = 2e-6 if dtype == torch.float32 else 1e-12
    np.testing.assert_allclose(error[0], expected, atol=tol, rtol=tol)
    np.testing.assert_allclose(indirect[0], expected_indirect, atol=tol, rtol=tol)
    assert not error[1].any() and not indirect[1].any()
    assert torch.equal(event.query_weight_eligibility[0], query[0].expand(2, 2))
    assert torch.equal(event.query_bias_eligibility[0], torch.ones(2, dtype=dtype))
    assert not event.query_weight_eligibility[1].any() and not event.query_bias_eligibility[1].any()
    broadcasts = {name: torch.zeros(2, size, dtype=dtype) for name, size in (
        ("controller", 3), ("key", 2), ("value", 2), ("strength", 1))}
    broadcasts["query"] = error
    assert model.apply_local_feedback(state, event, broadcasts, 0.01)
    np.testing.assert_allclose(model.query_weight.detach(), qweight - 0.01 * expected[:, None] * u,
                               atol=tol, rtol=tol)
    np.testing.assert_allclose(
        model.query_bias.detach(), bias - 0.01 * expected, atol=tol, rtol=tol
    )
    for name, value in before.items():
        if not name.startswith("query_"):
            assert torch.equal(value, model.state_dict()[name])
    assert torch.equal(state.synapses, state_before["synapses"])
    restored = model.restore_local_read(event.snapshot())
    assert torch.equal(restored.query_norm, event.query_norm)
    assert restored.query_norm.data_ptr() != event.query_norm.data_ptr()
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("magnitude", [0.0, 0.5e-8, 1e-8, 2e-8])
def test_read_path_normalization_clamp_and_equality_branch(dtype, magnitude):
    model = PlasticMemory(2, 2, 2, 2).to(dtype=dtype)
    with torch.no_grad():
        model.query_weight.copy_(torch.eye(2, dtype=dtype))
        model.query_bias.zero_()
    state = model.initial_local_state(1)
    state.synapses[0].copy_(torch.eye(2, dtype=dtype))
    event = model.read_local(torch.tensor([[magnitude, 0.0]], dtype=dtype), state)
    error, indirect = model.read_learning_signals(
        state, event, torch.tensor([[1.5, 0.75]], dtype=dtype)
    )
    expected = torch.tensor(
        [[0.0, 0.75 / magnitude]] if magnitude > 1e-8 else [[1.5e8, 0.75e8]], dtype=dtype
    )
    torch.testing.assert_close(error, expected)
    assert torch.equal(indirect, error)
    assert float(event.query_norm[0, 0]) == pytest.approx(magnitude)


@pytest.mark.parametrize("failure", ["read_transport", "normalization", "query_return"])
def test_read_path_nonfinite_intermediate_rejects_without_mutation(failure):
    model = PlasticMemory(2, 2, 2, 2)
    with torch.no_grad():
        model.query_weight.zero_()
        model.query_bias.zero_()
        if failure == "query_return":
            model.query_weight.fill_(1e32)
    state = model.initial_local_state(1)
    state.synapses.fill_(1e20 if failure == "read_transport" else
                         1e32 if failure == "normalization" else 1.0)
    event = model.read_local(torch.zeros(1, 2), state)
    before = copy.deepcopy(model.state_dict())
    signal = torch.full((1, 2), 1e20 if failure == "read_transport" else 1.0)
    with pytest.raises(ValueError, match="finite"):
        model.read_learning_signals(state, event, signal)
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_joined_read_credit_uses_old_query_weights_and_keeps_single_upstream_cast(dtype):
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(input_code_size=2, value_size=94, hidden_size=3, key_size=2,
                                 writer_update_fraction=0.5)
    if dtype == torch.float64:
        learner.double()
    with torch.no_grad():
        learner.decoder.output_weights.copy_(torch.linspace(
            -0.1, 0.1, learner.decoder.output_weights.numel(), dtype=torch.float64
        ).reshape_as(learner.decoder.output_weights))
    state = learner.ingest([b"x=ab", b"y=bc"])
    _, query_state, event = learner._query(b"x?", state)
    before = copy.deepcopy(learner.state_dict())
    feedback = learner.teach(b"x?", b"ab", state, 0.01)
    gradient = feedback.mean_context_signal
    assert gradient.dtype == dtype
    key_error = torch.bmm(state.synapses.transpose(1, 2), gradient[:, 2:, None]).squeeze(-1)
    e_a = (key_error - event.key * (event.key * key_error).sum(-1, keepdim=True)) / event.query_norm
    old_q = before["memory.query_weight"]
    cell_error = gradient[:, :2] + torch.nn.functional.linear(e_a, old_q.T)
    expected_map = old_q - 0.01 * (e_a[:, :, None] * event.query_weight_eligibility).sum(0)
    expected_cell = before["query_cell.weight"] - 0.01 * (
        cell_error[:, :, None] * query_state.input_eligibility
    ).sum(0)
    assert torch.equal(learner.memory.query_weight, expected_map)
    assert torch.equal(learner.query_cell.weight, expected_cell)
    for name in ("value_weight", "value_bias"):
        credit = state.credit[name]
        retrieved = (credit * event.key.reshape(1, *([1] * (credit.ndim - 2)), 2)).sum(-1)
        signal = gradient[:, 2:, None] if name.endswith("weight") else gradient[:, 2:]
        expected_value = before["memory." + name] - 0.01 * (signal * retrieved).sum(0)
        assert torch.equal(getattr(learner.memory, name), expected_value)
        assert not torch.equal(expected_value, before["memory." + name])
    assert not hasattr(learner, "query_feedback")
    assert not hasattr(learner, "value_feedback")
    assert learner.get_extra_state()["schema"] == "local-memory-phase-lms-v4"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_value_tags_match_independent_multiwrite_partial_and_read_update(dtype):
    model = PlasticMemory(2, 3, 2, 2, seed=15).to(dtype=dtype)
    rng = np.random.default_rng(327)
    values = rng.normal(size=(2, 5, 2)) * 0.4
    values[:, 4] = values[:, 2]  # Revisit a write input after intervening overwrite.
    mask = np.array([[True, False, True, True, True], [True, True, False, True, True]])
    values[~mask] = np.nan
    query = rng.normal(size=(2, 2)) * 0.3
    initial = model.initial_local_state(2)
    initial.synapses.copy_(torch.tensor(rng.normal(size=(2, 2, 2)) * 0.2, dtype=dtype))
    initial.cell.activity.copy_(torch.tensor(rng.normal(size=(2, 3)) * 0.1, dtype=dtype))
    initial_snapshot = initial.snapshot()
    inputs = torch.tensor(values, dtype=dtype)
    queries = torch.tensor(query, dtype=dtype)
    state = model.write_local(inputs, torch.tensor(mask), initial)
    event = model.read_local(queries, state)
    parameters = {name: value.detach().double().numpy().copy()
                  for name, value in model.named_parameters()}
    values, query = inputs.double().numpy(), queries.double().numpy()
    matrix = initial.synapses.double().numpy()
    writer = initial.cell.activity.double().numpy()
    base = _reference(parameters, values, mask, query, matrix, writer)
    tolerance = 3e-6 if dtype == torch.float32 else 1e-10
    np.testing.assert_allclose(state.synapses, base[2], atol=tolerance, rtol=tolerance)
    np.testing.assert_allclose(event.value, base[1], atol=tolerance, rtol=tolerance)
    read_signal = torch.tensor([[0.8, -0.3], [-0.2, 0.6]], dtype=dtype)
    expected_gradients = {}
    for name in ("value_weight", "value_bias"):
        array = parameters[name]
        gradient = np.zeros_like(array)
        for index in np.ndindex(array.shape):
            original = array[index]
            array[index] = original + 1e-6
            plus = _reference(parameters, values, mask, query, matrix, writer)
            array[index] = original - 1e-6
            minus = _reference(parameters, values, mask, query, matrix, writer)
            array[index] = original
            matrix_partial = (plus[2] - minus[2]) / 2e-6
            expected_partial = np.zeros_like(matrix_partial)
            expected_partial[:, index[0], :] = state.credit[name][(slice(None), *index)].double()
            np.testing.assert_allclose(
                matrix_partial, expected_partial, atol=tolerance, rtol=tolerance
            )
            gradient[index] = (
                ((plus[1] - minus[1]) / 2e-6) * read_signal.double().numpy()
            ).sum() / 2
        expected_gradients[name] = gradient
    before = copy.deepcopy(model.state_dict())
    signals = {name: torch.zeros(2, size, dtype=dtype) for name, size in (
        ("controller", 3), ("key", 2), ("strength", 1), ("query", 2))}
    signals["value"] = read_signal
    assert model.apply_local_feedback(state, event, signals, 0.01)
    for name, expected in expected_gradients.items():
        np.testing.assert_allclose(
            getattr(model, name).detach(), parameters[name] - 0.01 * expected,
            atol=tolerance, rtol=tolerance,
        )
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items()
               if name not in expected_gradients)
    assert torch.equal(initial.synapses, initial_snapshot["synapses"])
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize("seed", [0, 733])
def test_direct_value_credit_preserves_feedback_initialization_slots(seed):
    from humanbrain.local_memory_learner import LocalMemoryLearner

    global_rng = torch.random.get_rng_state().clone()
    learner = LocalMemoryLearner(input_code_size=2, value_size=94, hidden_size=3,
                                 key_size=2, seed=seed, writer_update_fraction=0.5)
    generator = torch.Generator(device="cpu").manual_seed(seed + 3)
    for name, size in (("controller", 3), ("key", 2), ("value", 94), ("strength", 1)):
        original = torch.randn(size, 96, generator=generator) / np.sqrt(96)
        if name == "value":
            assert not hasattr(learner, "value_feedback")
        else:
            assert torch.equal(getattr(learner, name + "_feedback"), original)
    assert torch.equal(global_rng, torch.random.get_rng_state())
    restored = copy.deepcopy(learner)
    restored.load_state_dict(copy.deepcopy(learner.state_dict()), strict=True)
    obsolete = learner.get_extra_state()
    obsolete["schema"] = "local-memory-phase-lms-v2"
    with pytest.raises(ValueError, match="configuration"):
        restored.set_extra_state(obsolete)


def test_value_read_signal_nonfinite_rejects_before_memory_group_publication():
    model = PlasticMemory(2, 3, 2, 2)
    state = model.write_local(torch.ones(1, 2, 2), torch.ones(1, 2, dtype=torch.bool))
    event = model.read_local(torch.ones(1, 2), state)
    signals = {name: torch.ones(1, size) for name, size in (
        ("controller", 3), ("key", 2), ("value", 2), ("strength", 1), ("query", 2))}
    signals["value"][0, 0] = torch.inf
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="finite"):
        model.apply_local_feedback(state, event, signals, 0.01)
    assert all(torch.equal(value, model.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_fraction_exact_frozen_source_forward_trace_and_teacher_parity(dtype):
    import humanbrain.plastic_memory as memory_module

    # Authenticated predecessor source is a QA oracle only, never evaluation loading.
    source = subprocess.run(
        ["git", "show",
         "86cc7232320cf4cef9337cf4f3b79a017bb55171:src/humanbrain/plastic_memory.py"],
        check=True, capture_output=True, text=True, timeout=30,
    ).stdout
    old_class = next(node for node in ast.parse(source).body
                     if isinstance(node, ast.ClassDef) and node.name == "StreamingRateCell")
    namespace = dict(vars(memory_module))
    code = compile(ast.Module(body=[old_class], type_ignores=[]), "frozen-rate-cell", "exec")
    exec(code, namespace)
    old = namespace["StreamingRateCell"](2, 3, seed=7).to(dtype=dtype)
    models = [StreamingRateCell(2, 3, seed=7).to(dtype=dtype),
              StreamingRateCell(2, 3, seed=7,
                                update_fraction=torch.full((3,), 0.5)).to(dtype=dtype)]
    assert set(models[0].state_dict()) == set(old.state_dict())
    states = [None, None, None]
    rng = torch.Generator().manual_seed(19)
    for _ in range(2):
        for step in range(5):
            x = torch.randn(2, 2, generator=rng, dtype=dtype)
            mask = torch.tensor([step != 2, step != 3])
            x[~mask] = torch.nan
            for index, model in enumerate([old, *models]):
                states[index] = model.local_step(x, states[index], mask=mask)
            for state in states[1:]:
                for name, value in states[0].snapshot().items():
                    assert torch.equal(getattr(state, name), value)
            clean = torch.where(mask[:, None], x, 0)
            expected = old(clean, states[0].activity)
            for model in models:
                assert torch.equal(model(clean, states[0].activity), expected)
        signal = torch.tensor([[0.1, -0.3, 0.2], [-0.2, 0.1, 0.3]], dtype=dtype)
        for model, state in zip([old, *models], states, strict=True):
            assert model.apply_learning_signal(state, signal, 0.01)
        for model in models:
            assert all(torch.equal(value, model.state_dict()[name])
                       for name, value in old.state_dict().items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multiscale_diagonal_cell_independent_finite_difference_and_local_update(dtype):
    alpha = np.array([0.5, 0.125, 0.03125])
    cell = StreamingRateCell(2, 3, update_fraction=torch.tensor(alpha, dtype=torch.float32))
    cell = cell.to(dtype=dtype)
    with torch.no_grad():
        cell.weight.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.2], [-0.2, 0.4]], dtype=dtype))
        cell.recurrent.copy_(torch.diag(torch.tensor([0.8, -0.3, 0.6], dtype=dtype)))
        cell.bias.copy_(torch.tensor([0.1, -0.2, 0.05], dtype=dtype))
    inputs = np.array([[0.2, -0.4], [0.6, 0.1], [np.nan, np.nan], [-0.3, 0.2], [0.5, 0.7]])
    mask = [True, True, False, True, True]
    initial = np.array([0.1, -0.2, 0.3])
    arrays = {name: p.detach().double().numpy().copy() for name, p in cell.named_parameters()}

    def forward():
        h = initial.copy()
        for active, x in zip(mask, inputs, strict=True):
            if active:
                # Independent scalar drive and neuron update; no model/autograd oracle.
                activation = np.array([np.tanh(sum(arrays['weight'][j, i] * x[i] for i in range(2))
                    + sum(arrays['recurrent'][j, k] * h[k] for k in range(3))
                    + arrays['bias'][j]) for j in range(3)])
                h = np.array([h[j] + alpha[j] * (activation[j] - h[j]) for j in range(3)])
        return h

    state = cell.initial_learning_state(1)
    state.activity.copy_(torch.tensor(initial, dtype=dtype)[None])
    before = state.snapshot()
    for active, x in zip(mask, inputs, strict=True):
        state = cell.local_step(torch.tensor(x, dtype=dtype)[None], state,
                                mask=torch.tensor([active]))
    tolerance = 2e-6 if dtype == torch.float32 else 2e-9
    np.testing.assert_allclose(state.activity[0], forward(), atol=tolerance, rtol=tolerance)
    gradients = {}
    for name, trace_name in [('weight', 'input_eligibility'),
                             ('recurrent', 'recurrent_eligibility'), ('bias', 'bias_eligibility')]:
        parameter = arrays[name]
        numerical = np.zeros_like(parameter)
        for index in np.ndindex(parameter.shape):
            value = parameter[index]
            parameter[index] = value + 1e-6
            plus = forward()
            parameter[index] = value - 1e-6
            minus = forward()
            parameter[index] = value
            numerical[index] = (plus[index[0]] - minus[index[0]]) / 2e-6
        np.testing.assert_allclose(getattr(state, trace_name)[0], numerical,
                                   atol=tolerance, rtol=tolerance)
        gradients[name] = numerical
    signal = torch.tensor([[0.3, -0.2, 0.4]], dtype=dtype)
    assert cell.apply_learning_signal(state, signal, 0.01)
    for name, parameter in cell.named_parameters():
        s = signal.numpy()[0] if name == 'bias' else signal.numpy()[0, :, None]
        np.testing.assert_allclose(parameter.detach(), arrays[name] - 0.01 * s * gradients[name],
                                   atol=tolerance, rtol=tolerance)
        assert parameter.grad is None
    assert torch.equal(before['activity'], torch.tensor(initial, dtype=dtype)[None])
    assert not cell.update_fraction.requires_grad
    assert 'update_fraction' not in dict(cell.named_parameters())


def test_fixed_fraction_validation_cast_loading_and_caller_ownership():
    source = torch.tensor([0.5, 0.125, 0.03125])
    cell = StreamingRateCell(2, 3, update_fraction=source)
    source.zero_()
    assert cell.update_fraction.tolist() == [0.5, 0.125, 0.03125]
    for bad in (0.0, -0.1, 1.1, float('nan'), True,
                torch.ones(2), torch.ones(3, dtype=torch.int64),
                torch.ones(3, requires_grad=True), torch.tensor([0.5, float('inf'), 0.1])):
        with pytest.raises(ValueError, match='fraction'):
            StreamingRateCell(2, 3, update_fraction=bad)
    cell.double()
    restored = StreamingRateCell(2, 3, update_fraction=torch.tensor([0.5, 0.125, 0.03125])).double()
    restored.load_state_dict(copy.deepcopy(cell.state_dict()))
    for name, value in cell.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name])
        assert value.data_ptr() != restored.state_dict()[name].data_ptr()
    for bad in (torch.tensor([0.5, 0.25, 0.03125], dtype=torch.float64),
                torch.ones(2, dtype=torch.float64), torch.ones(3, dtype=torch.float32)):
        state = copy.deepcopy(cell.state_dict())
        state['update_fraction'] = bad
        with pytest.raises(RuntimeError, match='fraction'):
            restored.load_state_dict(state)
        assert torch.equal(restored.update_fraction, cell.update_fraction)
    restored.update_fraction[1] = 0.25
    with pytest.raises(ValueError, match='fraction'):
        restored.local_step(torch.zeros(1, 2, dtype=torch.float64))
    restored.update_fraction = restored.update_fraction.float()
    with pytest.raises(ValueError, match='fraction'):
        restored(torch.zeros(1, 2, dtype=torch.float64))
    # Ordinary dtype casts may round a generic fixed fraction without retraining it.
    generic = StreamingRateCell(2, 3, update_fraction=torch.tensor([0.1, 0.2, 0.3]))
    generic.double().float()
    assert generic.local_step(torch.zeros(1, 2)).activity.shape == (1, 3)


def test_multiscale_masked_noop_and_nonfinite_learning_proposal_are_atomic():
    cell = StreamingRateCell(2, 3, update_fraction=torch.tensor([0.5, 0.125, 0.03125]))
    state = cell.local_step(torch.ones(1, 2))
    masked = cell.local_step(torch.full((1, 2), torch.nan), state, mask=torch.tensor([False]))
    assert all(torch.equal(value, getattr(masked, name))
               for name, value in state.snapshot().items())
    state.input_eligibility.fill_(torch.finfo(torch.float32).max)
    before = copy.deepcopy(cell.state_dict())
    with pytest.raises(ValueError, match='finite'):
        cell.apply_learning_signal(state, torch.full((1, 3), 2.0), 1.0)
    assert all(torch.equal(value, cell.state_dict()[name]) for name, value in before.items())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_multiscale_controller_traces_reach_unchanged_overwrite_tags(dtype):
    model = PlasticMemory(
        2, 3, 2, 2, writer_update_fraction=torch.tensor([0.5, 0.125, 0.03125])
    ).to(dtype=dtype)
    inputs = torch.tensor([[[0.1, 0.3], [0.4, -0.2], [float('nan'), 0], [0.1, 0.3]]],
                          dtype=dtype)
    mask = torch.tensor([[True, True, False, True]])
    initial = model.initial_local_state(1)
    expected = _unbuffered_tag_reference(model, inputs, mask, initial)
    actual = model.write_local(inputs, mask, initial)
    assert torch.equal(actual.synapses, expected.synapses)
    assert all(torch.equal(value, expected.credit[name]) for name, value in actual.credit.items())
    assert all(torch.equal(value, getattr(expected.cell, name))
               for name, value in actual.cell.snapshot().items())
