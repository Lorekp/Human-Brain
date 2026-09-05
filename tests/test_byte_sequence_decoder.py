"""Byte decoder numerical/interface QA, without a task fit or language claim."""

import copy
import io
import math
from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch", reason="optional pointer runtime is not installed")

from humanbrain.byte_sequence_decoder import (  # noqa: E402
    BYTE_BOS,
    BYTE_EOS,
    ByteGeneration,
    ByteSequenceDecoder,
    byte_sequence_loss,
    byte_teacher_forcing,
)


def test_teacher_forcing_preserves_bytes_empty_and_shift():
    values = [b"", bytes([0, 255]), b'{"a":1,"b":2}\n']
    inputs, targets, mask = byte_teacher_forcing(values)
    for row, value in enumerate(values):
        count = len(value) + 1
        assert inputs[row, :count].tolist() == [BYTE_BOS, *value]
        assert targets[row, :count].tolist() == [*value, BYTE_EOS]
        assert mask[row].sum().item() == count
        assert not mask[row, count:].any()


def test_loss_masks_padding_but_includes_empty_eos_and_zero_bytes():
    _, targets, mask = byte_teacher_forcing([b"", b"\x00\xff"])
    logits = torch.zeros(2, 3, 257, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits[~mask] = torch.nan
    loss = byte_sequence_loss(logits, targets, mask)
    assert loss.item() == pytest.approx(math.log(257))
    loss.backward()
    assert torch.equal(logits.grad[~mask], torch.zeros_like(logits.grad[~mask]))
    expected = torch.full((257,), 1 / (4 * 257), dtype=torch.float64)
    expected[BYTE_EOS] -= 1 / 4
    torch.testing.assert_close(logits.grad[0, 0], expected)
    assert logits.grad[1, 0, 0] < 0


@pytest.mark.parametrize("kind", ["cell", "bypass"])
def test_forward_is_causal_matches_stream_and_has_context_gradient(kind):
    decoder = ByteSequenceDecoder(3, 4, 2, kind=kind, seed=7).double()
    context = torch.tensor([[0.3, -0.2, 0.7]], dtype=torch.float64, requires_grad=True)
    inputs, _, _ = byte_teacher_forcing([b"ab"])
    actual = decoder(context, inputs)
    state = decoder.cell.initial_state(1, dtype=context.dtype)
    manual = []
    for token in inputs.unbind(1):
        state = decoder.cell(torch.cat((context, decoder.embedding(token)), -1), state)
        manual.append(decoder.output(state))
    assert torch.equal(actual, torch.stack(manual, 1))
    changed = inputs.clone()
    changed[:, 2] = 99
    assert torch.equal(actual[:, :2], decoder(context, changed)[:, :2])
    assert torch.autograd.gradcheck(
        lambda c: decoder.loss(c, [b"ab"]), (context,), eps=1e-6, atol=1e-6, rtol=1e-4
    )
    decoder.loss(context, [b"ab"]).backward()
    assert context.grad.abs().sum() > 0
    for name, parameter in decoder.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name


@pytest.mark.parametrize("kind", ["cell", "bypass"])
def test_restore_seed_and_inference_are_read_only(kind):
    rng = torch.random.get_rng_state().clone()
    decoder = ByteSequenceDecoder(3, 4, 2, kind=kind, seed=11).double().train()
    assert torch.equal(rng, torch.random.get_rng_state())
    context = torch.tensor([[1.0, 2.0, 3.0], [-1.0, 0.0, 1.0]], dtype=torch.float64)
    optimizer = torch.optim.SGD(decoder.parameters(), lr=0.01)
    decoder.loss(context, [b"a", b""]).backward()
    optimizer.step()  # One synthetic numerical state mutation, not a scored fit.
    before = copy.deepcopy(decoder.state_dict())
    buffer = io.BytesIO()
    torch.save(before, buffer)
    buffer.seek(0)
    restored = ByteSequenceDecoder(3, 4, 2, kind=kind, seed=80).double()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    inputs, _, _ = byte_teacher_forcing([b"a", b""])
    assert torch.equal(decoder(context, inputs), restored(context, inputs))
    assert decoder.generate(context, 7) == restored.generate(context, 7)
    assert decoder.training  # generate must not change module mode.
    assert torch.equal(rng, torch.random.get_rng_state())
    for key, value in before.items():
        current = decoder.state_dict()[key]
        assert torch.equal(current, value) if torch.is_tensor(value) else current == value
    with pytest.raises(ValueError, match="configuration mismatch"):
        wrong = ByteSequenceDecoder(3, 4, 2, kind=kind).state_dict()
        wrong["_extra_state"]["context_size"] = 5
        restored.load_state_dict(wrong)


def test_generation_raw_bytes_per_row_eos_and_no_repair():
    decoder = ByteSequenceDecoder(2, 3, 2)
    scripts = [[BYTE_EOS], [255, 0, 10, BYTE_EOS], [123, 34, 97, 34, 58, 49, BYTE_EOS]]
    position = 0

    def scripted_step(context, tokens, state):
        nonlocal position
        logits = context.new_full((3, 257), -100)
        for row, sequence in enumerate(scripts):
            logits[row, sequence[min(position, len(sequence) - 1)]] = 100
        position += 1
        return logits, state

    with patch.object(decoder, "_step", side_effect=scripted_step):
        result = decoder.generate(torch.zeros(3, 2), 6)
    assert result == (
        ByteGeneration(b"", "eos"),
        ByteGeneration(b"\xff\x00\n", "eos"),
        ByteGeneration(b'{"a":1', "eos"),
    )
    # Invalid UTF-8 and unfinished JSON are intentionally returned unchanged.
    assert position == 7


def test_generation_byte_limit_zero_limit_and_tie_rule_use_neural_logits():
    decoder = ByteSequenceDecoder(2, 3, 2)
    with torch.no_grad():
        decoder.output.weight.zero_()
        decoder.output.bias.zero_()  # Complete tie selects byte zero, never injected EOS.
    context = torch.zeros(1, 2)
    assert decoder.generate(context, 3) == (ByteGeneration(b"\x00" * 3, "limit"),)
    assert decoder.generate(context, 0) == (ByteGeneration(b"", "limit"),)
    with torch.no_grad():
        decoder.output.bias[BYTE_EOS] = 1
    assert decoder.generate(context, 0) == (ByteGeneration(b"", "eos"),)


def test_matched_bypass_removes_previous_byte_influence_only():
    cell = ByteSequenceDecoder(2, 4, 3, seed=19).double()
    bypass = ByteSequenceDecoder(2, 4, 3, kind="bypass", seed=19).double()
    for name, parameter in bypass.named_parameters():
        assert torch.equal(parameter, dict(cell.named_parameters())[name])
    context = torch.tensor([[0.2, 0.4]], dtype=torch.float64)
    first = torch.tensor([[BYTE_BOS, 65, 90]])
    second = torch.tensor([[BYTE_BOS, 66, 90]])
    assert not torch.equal(cell(context, first)[:, -1], cell(context, second)[:, -1])
    assert torch.equal(bypass(context, first)[:, -1], bypass(context, second)[:, -1])


def test_invalid_teacher_masks_tokens_and_context_fail_explicitly():
    with pytest.raises(ValueError):
        byte_teacher_forcing(["text"])
    with pytest.raises(ValueError):
        byte_teacher_forcing([])
    inputs, targets, mask = byte_teacher_forcing([b"ab"])
    decoder = ByteSequenceDecoder(2, 3, 2)
    with pytest.raises(ValueError):
        decoder(torch.zeros(1, 2), inputs + 258)
    with pytest.raises(ValueError):
        decoder(torch.full((1, 2), torch.nan), inputs)
    logits = torch.zeros(1, 3, 257)
    targets[0, 0] = BYTE_EOS
    with pytest.raises(ValueError, match="exactly one"):
        byte_sequence_loss(logits, targets, mask)
    mask[0, 1] = False
    with pytest.raises(ValueError, match="right-padded"):
        byte_sequence_loss(logits, targets, mask)


def _assert_decoder_state_equal(first, second):
    assert first.keys() == second.keys()
    for name, value in first.items():
        assert torch.equal(value, second[name]) if torch.is_tensor(value) else value == second[name]


@pytest.mark.parametrize("kind", ["cell", "bypass"])
def test_local_two_step_rule_matches_independent_scalar_oracle(kind):
    decoder = ByteSequenceDecoder(2, 3, 2, kind=kind, seed=91, local_learning=True).double()
    context = torch.tensor([[0.2, -0.4], [0.3, 0.1]], dtype=torch.float64, requires_grad=True)
    h = torch.zeros(2, 3, dtype=torch.float64)
    ew = torch.zeros(2, 3, 4, dtype=torch.float64)
    eu = torch.zeros(2, 3, 3, dtype=torch.float64)
    eb = torch.zeros(2, 3, dtype=torch.float64)
    state = None
    with torch.no_grad():
        for tokens in (torch.tensor([BYTE_BOS, BYTE_BOS]), torch.tensor([97, 98])):
            x = torch.cat((context.detach(), decoder.embedding(tokens)), -1)
            old = h.clone()
            for row in range(2):
                for j in range(3):
                    drive = float(decoder.cell.bias[j]) + sum(
                        float(decoder.cell.weight[j, i] * x[row, i]) for i in range(4)
                    )
                    if kind == "cell":
                        drive += sum(
                            float(decoder.cell.recurrent[j, i] * old[row, i]) for i in range(3)
                        )
                    activation = math.tanh(drive)
                    derivative = (1 - activation**2) * (0.5 if kind == "cell" else 1)
                    carry = (
                        0.5 + derivative * float(decoder.cell.recurrent[j, j])
                        if kind == "cell"
                        else 0
                    )
                    h[row, j] = (
                        old[row, j] + 0.5 * (activation - old[row, j])
                        if kind == "cell"
                        else activation
                    )
                    ew[row, j] = carry * ew[row, j] + derivative * x[row]
                    eu[row, j] = carry * eu[row, j] + derivative * old[row]
                    eb[row, j] = carry * eb[row, j] + derivative
            step = decoder.local_step(context, tokens, state)
            state = step.state
        torch.testing.assert_close(state.activity, h, atol=1e-14, rtol=1e-14)
        torch.testing.assert_close(state.input_eligibility, ew, atol=1e-14, rtol=1e-14)
        targets = torch.tensor([BYTE_EOS, 255])
        delta = step.logits.softmax(-1)
        delta[0, BYTE_EOS] -= 1
        delta[1, 255] -= 1
        signal = delta @ decoder.hidden_feedback.T
        expected = copy.deepcopy(decoder.state_dict())
        eta = 0.07
        expected["output.weight"] -= eta * delta.T @ h / 2
        expected["output.bias"] -= eta * delta.mean(0)
        expected["cell.weight"] -= eta * (signal[:, :, None] * ew).mean(0)
        expected["cell.bias"] -= eta * (signal * eb).mean(0)
        if kind == "cell":
            expected["cell.recurrent"] -= eta * (signal[:, :, None] * eu).mean(0)
    with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")):
        old_cell_weight = decoder.cell.weight.detach().clone()
        old_output_weight = decoder.output.weight.detach().clone()
        feedback = decoder.teach_local(step, targets, eta)
    assert feedback.teacher_count == 2 and feedback.updated
    assert not torch.equal(old_cell_weight, decoder.cell.weight)
    assert not torch.equal(old_output_weight, decoder.output.weight)
    torch.testing.assert_close(feedback.hidden_signal, signal)
    torch.testing.assert_close(feedback.context_signal, delta @ decoder.context_feedback.T)
    for name, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(decoder.state_dict()[name], value, atol=1e-14, rtol=1e-14)
    assert context.grad is None
    assert all(parameter.grad is None for parameter in decoder.parameters())
    assert all(
        not value.requires_grad
        for value in (step.logits, feedback.hidden_signal, feedback.context_signal)
    )
    assert not decoder.embedding.weight.requires_grad


def test_local_masks_freezing_and_fixed_state_are_honest_noops():
    decoder = ByteSequenceDecoder(2, 3, 2, local_learning=True).double()
    initial = copy.deepcopy(decoder.state_dict())
    context = torch.tensor([[0.1, 0.2], [torch.nan, torch.nan]], dtype=torch.float64)
    step = decoder.local_step(
        context, torch.tensor([BYTE_BOS, -999]), mask=torch.tensor([True, False])
    )
    receipt = decoder.teach_local(step, torch.tensor([BYTE_EOS, -999]), 0.1, learning_enabled=False)
    assert receipt.teacher_count == 1 and not receipt.updated
    assert torch.count_nonzero(receipt.hidden_signal[1]) == 0
    _assert_decoder_state_equal(initial, decoder.state_dict())
    single = copy.deepcopy(decoder)
    one_step = single.local_step(context[:1], torch.tensor([BYTE_BOS]))
    single.teach_local(one_step, torch.tensor([65]), 0.1)
    mixed = decoder.teach_local(step, torch.tensor([65, -999]), 0.1)
    assert mixed.teacher_count == 1 and mixed.updated
    _assert_decoder_state_equal(single.state_dict(), decoder.state_dict())
    initial = copy.deepcopy(decoder.state_dict())
    decoder.requires_grad_(False)
    assert not decoder.teach_local(step, torch.tensor([65, -999]), 0.1).updated
    _assert_decoder_state_equal(initial, decoder.state_dict())
    empty = decoder.local_step(
        torch.full((2, 2), torch.nan, dtype=torch.float64),
        torch.tensor([-1, -1]),
        step.state,
        mask=torch.zeros(2, dtype=torch.bool),
    )
    for name, value in step.state.snapshot().items():
        other = empty.state.snapshot()[name]
        assert value is None if other is None else torch.equal(value, other)
    receipt = decoder.teach_local(empty, torch.tensor([-1, -1]), 0.1)
    assert receipt.teacher_count == 0 and receipt.loss == 0 and not receipt.updated
    _assert_decoder_state_equal(initial, decoder.state_dict())


def test_local_delayed_feedback_reload_and_read_only_generation():
    decoder = ByteSequenceDecoder(2, 3, 2, seed=4, local_learning=True)
    context = torch.tensor([[0.1, 0.3]])
    first = decoder.local_step(context, torch.tensor([BYTE_BOS]))
    snapshot = first.snapshot()
    second = decoder.local_step(context, torch.tensor([65]), first.state)
    decoder.teach_local(second, torch.tensor([66]), 0.02)
    buffer = io.BytesIO()
    torch.save({"model": decoder.state_dict(), "pending": snapshot}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = ByteSequenceDecoder(2, 3, 2, seed=99, local_learning=True)
    restored.load_state_dict(saved["model"])
    pending = restored.restore_local_step(saved["pending"])
    assert pending.logits.data_ptr() != saved["pending"]["logits"].data_ptr()
    assert torch.equal(pending.logits, first.logits)
    receipt = decoder.teach_local(first, torch.tensor([70]), 0.03)
    again = restored.teach_local(pending, torch.tensor([70]), 0.03)
    assert receipt.loss == again.loss and receipt.teacher_count == again.teacher_count == 1
    _assert_decoder_state_equal(decoder.state_dict(), restored.state_dict())
    before = copy.deepcopy(restored.state_dict())
    assert restored.generate(context, 3) == decoder.generate(context, 3)
    _assert_decoder_state_equal(before, restored.state_dict())


def test_local_initial_identity_private_rng_and_invalid_feedback_atomicity():
    rng = torch.random.get_rng_state().clone()
    reference = ByteSequenceDecoder(2, 3, 2, seed=5)
    local = ByteSequenceDecoder(2, 3, 2, seed=5, local_learning=True)
    assert torch.equal(rng, torch.random.get_rng_state())
    for name, parameter in reference.named_parameters():
        assert torch.equal(parameter, dict(local.named_parameters())[name])
    assert "local_learning" not in reference.get_extra_state()
    assert local.get_extra_state()["local_learning"] == "fixed-embedding-random-feedback-v1"
    context = torch.tensor([[0.1, -0.2]])
    tokens = torch.tensor([[BYTE_BOS, 65]])
    assert torch.equal(reference(context, tokens), local(context, tokens))
    with pytest.raises(ValueError, match="local_learning=True"):
        reference.local_step(context, tokens[:, 0])
    step = local.local_step(context, tokens[:, 0])
    before = copy.deepcopy(local.state_dict())
    for target, rate in ((torch.tensor([BYTE_BOS]), 0.1), (torch.tensor([65]), float("nan"))):
        with pytest.raises(ValueError):
            local.teach_local(step, target, rate)
        _assert_decoder_state_equal(before, local.state_dict())
    bad = step.snapshot()
    bad["state"]["input_eligibility"][0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        local.restore_local_step(bad)
    _assert_decoder_state_equal(before, local.state_dict())


def _assert_nested_equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif torch.is_tensor(first):
        assert torch.equal(first, second)
        assert not second.requires_grad and second.grad_fn is None
    else:
        assert first == second


@pytest.mark.parametrize(
    "mode", ["tagged", "aggregate", "writer_maps_frozen", "frozen", "writes_disabled"]
)
def test_local_composition_one_lesson_control_groups_and_retired_credit(mode):
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(
        mode=mode, embedding_size=2, hidden_size=3, key_size=2, value_size=3, seed=19
    ).double()
    before = copy.deepcopy(learner.state_dict())
    state = learner.ingest([b"x=a", b"y=b"])
    _assert_nested_equal(before, learner.state_dict())  # Ingestion updates fast state only.
    assert torch.count_nonzero(state.cell.activity) == 0
    assert torch.count_nonzero(state.cell.input_eligibility) == 0
    if mode == "writes_disabled":
        assert torch.count_nonzero(state.synapses) == 0
        assert all(torch.count_nonzero(credit) == 0 for credit in state.credit.values())
    else:
        assert torch.count_nonzero(state.synapses) > 0
        assert any(torch.count_nonzero(credit) > 0 for credit in state.credit.values())
    persistent = state.snapshot()
    first = learner.predict(b"x?", state, max_bytes=3)
    assert isinstance(first, ByteGeneration)
    _assert_nested_equal(before, learner.state_dict())
    _assert_nested_equal(persistent, state.snapshot())
    with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")):
        receipt = learner.teach(b"x?", b"a", state, 0.001)
    assert receipt.teacher_count == 2 and math.isfinite(receipt.mean_loss)
    assert receipt.decoder_updated == (mode != "frozen")
    assert receipt.query_updated == (mode != "frozen")
    assert receipt.memory_updated == (mode != "frozen")
    _assert_nested_equal(persistent, state.snapshot())  # Caller explicitly adopts returned state.
    assert torch.equal(receipt.state.synapses, state.synapses)
    assert all(torch.count_nonzero(credit) == 0 for credit in receipt.state.credit.values())
    assert torch.count_nonzero(receipt.state.cell.activity) == 0
    after = learner.state_dict()
    for name, value in before.items():
        if not torch.is_tensor(value):
            continue
        if name.endswith("feedback") or name == "decoder.embedding.weight":
            assert torch.equal(value, after[name]), name
    for name in (
        "memory.controller.weight",
        "memory.key_weight",
        "memory.value_weight",
        "memory.strength_weight",
    ):
        changed = not torch.equal(before[name], after[name])
        assert changed == (mode in {"tagged", "aggregate"}), name
    assert (
        not torch.equal(before["memory.query_weight"], after["memory.query_weight"])
        if mode != "frozen"
        else torch.equal(before["memory.query_weight"], after["memory.query_weight"])
    )
    assert all(parameter.grad is None for parameter in learner.parameters())
    if mode == "frozen":
        _assert_nested_equal(before, after)


def test_local_composition_before_update_signals_mean_once_and_upstream_last():
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(embedding_size=2, hidden_size=3, key_size=2, value_size=3)
    state = learner.ingest([b"one", b"two"])
    writer_before = copy.deepcopy(learner.memory.state_dict())
    query_before = copy.deepcopy(learner.query_cell.state_dict())
    signals, events = [], []
    original_teach = learner.decoder.teach_local
    original_memory = learner.memory.apply_local_feedback
    original_query = learner.query_cell.apply_learning_signal

    def teach(*args, **kwargs):
        _assert_nested_equal(writer_before, learner.memory.state_dict())
        _assert_nested_equal(query_before, learner.query_cell.state_dict())
        feedback = original_teach(*args, **kwargs)
        signals.append(feedback.context_signal.clone())
        events.append("decoder")
        return feedback

    def apply_memory(current, read, incoming, rate, **kwargs):
        assert len(signals) == 3
        mean = sum(signals) / 3
        for name in incoming:
            torch.testing.assert_close(
                incoming[name], mean @ getattr(learner, name + "_feedback").T
            )
        events.append("memory")
        return original_memory(current, read, incoming, rate, **kwargs)

    def apply_query(current, incoming, rate, **kwargs):
        torch.testing.assert_close(incoming, (sum(signals) / 3)[:, :2])
        events.append("query")
        return original_query(current, incoming, rate, **kwargs)

    with (
        patch.object(learner.decoder, "teach_local", side_effect=teach),
        patch.object(learner.memory, "apply_local_feedback", side_effect=apply_memory),
        patch.object(learner.query_cell, "apply_learning_signal", side_effect=apply_query),
    ):
        receipt = learner.teach(b"q", b"a!", state, 0.001)
    assert events == ["decoder", "decoder", "decoder", "memory", "query"]
    assert receipt.teacher_count == 3
    torch.testing.assert_close(receipt.mean_context_signal, sum(signals) / 3)


def test_local_composition_source_free_transport_and_disabled_empty_teacher():
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(embedding_size=2, hidden_size=3, key_size=2, value_size=3)
    rng = torch.random.get_rng_state().clone()
    state = learner.ingest([b"private-source-ONE", b"private-source-TWO"])
    snapshot = learner.snapshot(state)

    def check_transport(value):
        if isinstance(value, dict):
            for item in value.values():
                check_transport(item)
        elif torch.is_tensor(value):
            assert not value.requires_grad and value.grad_fn is None
        else:
            assert isinstance(value, (str, int, type(None)))
            assert "private-source" not in str(value)

    check_transport(snapshot)
    buffer = io.BytesIO()
    torch.save(snapshot, buffer)
    buffer.seek(0)
    restored = LocalMemoryLearner(
        embedding_size=2, hidden_size=3, key_size=2, value_size=3, seed=91
    )
    transported = restored.restore_snapshot(torch.load(buffer, weights_only=True))
    assert learner.predict(b"q", state, 5) == restored.predict(b"q", transported, 5)
    before = restored.snapshot(transported)
    feedback = restored.teach(b"q", b"", transported, 0.01, learning_enabled=False)
    assert feedback.teacher_count == 1
    assert not any((feedback.decoder_updated, feedback.query_updated, feedback.memory_updated))
    _assert_nested_equal(before, restored.snapshot(transported))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not hasattr(restored, "documents") and not hasattr(restored, "query")
    wrong_mode = LocalMemoryLearner(
        mode="aggregate", embedding_size=2, hidden_size=3, key_size=2, value_size=3
    )
    with pytest.raises(ValueError, match="control mode"):
        wrong_mode.restore_snapshot(snapshot)
