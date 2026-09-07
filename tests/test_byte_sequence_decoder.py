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
@pytest.mark.parametrize("feedback_mode", ["random", "symmetric"])
def test_local_two_step_rule_matches_independent_scalar_oracle(kind, feedback_mode):
    decoder = ByteSequenceDecoder(
        2, 3, 2, kind=kind, seed=91, local_learning=True, hidden_feedback_mode=feedback_mode
    ).double()
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
        signal = delta @ (
            decoder.hidden_feedback.T if feedback_mode == "random" else decoder.output.weight
        )
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
    assert local.get_extra_state()["local_learning"] == "fixed-embedding-local-feedback-v2"
    assert local.get_extra_state()["hidden_feedback_mode"] == "random"
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


def _set_upstream_dtype(learner, dtype):
    """The default is mixed precision; .double() also tests float64 upstream."""
    if dtype == torch.float64:
        learner.double()


@pytest.mark.parametrize(
    "mode", ["tagged", "aggregate", "writer_maps_frozen", "frozen", "writes_disabled"]
)
def test_local_composition_one_lesson_control_groups_and_retired_credit(mode):
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    learner = LocalMemoryLearner(writer_update_fraction=0.5,
        mode=mode, input_code_size=2, hidden_size=3, key_size=2, value_size=94, seed=19
    ).double()
    # First exposure writes output synapses but correctly supplies zero upstream signal.
    warm_state = learner.ingest([b"x=a", b"y=b"])
    warm = learner.teach(b"x?", b"aa", warm_state, 0.01)
    assert not warm.mean_context_signal.any()
    assert not warm.query_updated and not warm.memory_updated
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
        if mode in {"writer_maps_frozen", "frozen"}:
            assert all(torch.count_nonzero(credit) == 0 for credit in state.credit.values())
        else:
            assert any(torch.count_nonzero(credit) > 0 for credit in state.credit.values())
    persistent = state.snapshot()
    first = learner.predict(b"x?", state, max_bytes=2)
    assert isinstance(first, ByteGeneration)
    _assert_nested_equal(before, learner.state_dict())

    _assert_nested_equal(persistent, state.snapshot())
    with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")):
        receipt = learner.teach(b"x?", b"aa", state, 0.01)
    assert receipt.teacher_count == 3 and math.isfinite(receipt.mean_loss)
    assert receipt.decoder_updated == (mode != "frozen")
    assert receipt.query_updated == (mode != "frozen")
    assert receipt.memory_updated == (mode not in {"frozen", "writes_disabled"})
    _assert_nested_equal(persistent, state.snapshot())  # Caller explicitly adopts returned state.
    assert torch.equal(receipt.state.synapses, state.synapses)
    assert all(torch.count_nonzero(credit) == 0 for credit in receipt.state.credit.values())
    assert torch.count_nonzero(receipt.state.cell.activity) == 0
    after = learner.state_dict()
    for name, value in before.items():
        if not torch.is_tensor(value):
            continue
        if name.endswith("feedback") or name == "input_codes":
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
        if mode not in {"frozen", "writes_disabled"}
        else torch.equal(before["memory.query_weight"], after["memory.query_weight"])
    )
    assert all(parameter.grad is None for parameter in learner.parameters())
    if mode == "frozen":
        _assert_nested_equal(before, after)


def test_local_composition_before_update_signals_mean_once_and_upstream_last():
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    learner = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=2, hidden_size=3, key_size=2, value_size=94
    )
    state = learner.ingest([b"one", b"two"])
    learner.teach(b"q", b"a!", state, 0.01)
    state = learner.ingest([b"one", b"two"])
    writer_before = copy.deepcopy(learner.memory.state_dict())
    query_before = copy.deepcopy(learner.query_cell.state_dict())
    signals, events, expected_query = [], [], []
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
        mean = mean.to(learner.input_codes.dtype)
        key_error = current.synapses.transpose(1, 2) @ mean[:, 2:, None]
        key_error = key_error.squeeze(-1)
        drive_error = (key_error - read.key * (read.key * key_error).sum(-1, keepdim=True))
        drive_error = drive_error / read.query_norm
        for name in incoming:
            expected = (drive_error if name == "query" else mean[:, 2:]) if (
                name in {"query", "value"}
            ) else mean @ getattr(learner, name + "_feedback").T
            torch.testing.assert_close(incoming[name], expected)
        expected_query.append(mean[:, :2] + drive_error @ writer_before["query_weight"])
        events.append("memory")
        return original_memory(current, read, incoming, rate, **kwargs)

    def apply_query(current, incoming, rate, **kwargs):
        torch.testing.assert_close(
            incoming, expected_query[0]
        )
        events.append("query")
        return original_query(current, incoming, rate, **kwargs)

    with (
        patch.object(learner.decoder, "teach_local", side_effect=teach),
        patch.object(learner.memory, "apply_local_feedback", side_effect=apply_memory),
        patch.object(learner.query_cell, "apply_learning_signal", side_effect=apply_query),
    ):
        receipt = learner.teach(b"q", b"a!", state, 0.01)
    assert events == ["decoder", "decoder", "decoder", "memory", "query"]
    assert receipt.teacher_count == 3
    torch.testing.assert_close(
        receipt.mean_context_signal, (sum(signals) / 3).to(learner.input_codes.dtype)
    )


def test_local_composition_source_free_transport_and_disabled_teacher():
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    learner = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=2, hidden_size=3, key_size=2, value_size=94
    )
    rng = torch.random.get_rng_state().clone()
    state = learner.ingest([b"private-source-ONE", b"private-source-TWO"])
    snapshot = learner.snapshot(state)

    def check_transport(value):
        if isinstance(value, dict):
            for item in value.values():
                check_transport(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                check_transport(item)
        elif torch.is_tensor(value):
            assert not value.requires_grad and value.grad_fn is None
        else:
            assert isinstance(value, (str, int, float, type(None)))
            assert "private-source" not in str(value)

    check_transport(snapshot)
    buffer = io.BytesIO()
    torch.save(snapshot, buffer)
    buffer.seek(0)
    restored = LocalMemoryLearner(writer_update_fraction=0.5,
        input_code_size=2, hidden_size=3, key_size=2, value_size=94, seed=0
    )
    transported = restored.restore_snapshot(torch.load(buffer, weights_only=True))
    assert learner.predict(b"q", state, 2) == restored.predict(b"q", transported, 2)
    before = restored.snapshot(transported)
    feedback = restored.teach(b"q", b"aa", transported, 0.01, learning_enabled=False)
    assert feedback.teacher_count == 3
    assert not any((feedback.decoder_updated, feedback.query_updated, feedback.memory_updated))
    _assert_nested_equal(before, restored.snapshot(transported))
    assert torch.equal(rng, torch.random.get_rng_state())
    assert not hasattr(restored, "documents") and not hasattr(restored, "query")
    wrong_mode = LocalMemoryLearner(writer_update_fraction=0.5,
        mode="aggregate", input_code_size=2, hidden_size=3, key_size=2, value_size=94
    )
    with pytest.raises(ValueError, match="control mode"):
        wrong_mode.restore_snapshot(snapshot)


@pytest.fixture(params=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11))
def local_cost_runner(request):
    if request.param == 2:
        from experiments.r4_integrated import local_memory_byte_v2 as runner

        return runner, "experiments.r4_integrated.local_memory_byte_v2"
    from functools import partial
    from types import SimpleNamespace

    from experiments.r4_integrated import local_memory_byte_harness as harness
    if request.param == 11:
        from experiments.r4_integrated import local_sequence_memory_v1 as harness
        SPEC = harness.SPEC
    elif request.param == 10:
        from experiments.r4_integrated import local_memory_event_v1 as harness
        SPEC = harness.SPEC
    elif request.param == 9:
        from experiments.r4_integrated import local_memory_multiscale_v1 as harness
        SPEC = harness.SPEC
    elif request.param == 8:
        from experiments.r4_integrated import local_memory_value_credit_v1 as harness
        SPEC = harness.SPEC
    elif request.param == 7:
        from experiments.r4_integrated import local_memory_query_credit_v1 as harness
        SPEC = harness.SPEC
    elif request.param == 6:
        from experiments.r4_integrated import local_memory_two_byte_v1 as harness
        SPEC = harness.SPEC
    elif request.param == 3:
        from experiments.r4_integrated.local_memory_byte_v3 import SPEC
    elif request.param == 4:
        from experiments.r4_integrated.local_memory_byte_v4 import SPEC
    else:
        from experiments.r4_integrated.local_memory_byte_v5 import SPEC

    return SimpleNamespace(
        ROOT=harness.ROOT, ARMS=harness.ARMS, CONFIG=SPEC.config,
        file_hash=harness.file_hash, write_json=harness.write_json,
        project_cost=partial(harness.project_cost, SPEC),
        validate_cost_launch=partial(harness.validate_cost_launch, SPEC),
        duration_clock=harness.duration_clock,
    ), SPEC.module


def test_local_successor_cost_keeps_original_margin_and_rejects_bad_samples(local_cost_runner):
    runner, _ = local_cost_runner

    report = {
        "status": "completed",
        "mode": "cost",
        "config": runner.CONFIG,
        "selected_lesson_indices": [3, 4, 5, 6],
        "lessons": [
            {"arm": arm, "index": index, "seconds": 0.01}
            for index in [3, 4, 5, 6]
            for arm in runner.ARMS
        ],
        "cap_generation": {
            arm: {
                "seconds": 0.02,
                "stop_reason": "limit",
                "decisions": runner.CONFIG["max_bytes"] + 1,
                "generated_hex": bytes(runner.CONFIG["max_bytes"]).hex(),
            }
            for arm in runner.ARMS
        },
        "actions": [{}] * (20 * len(runner.ARMS)),
        "motor_demonstrations": [{}, {}],
        "counts": {
            "first_attempts": 4 * len(runner.ARMS),
            "teacher_byte_offers": (
                4 * len(runner.ARMS) * (3 if runner.CONFIG["max_bytes"] == 2 else 10)
            ),
            "enabled_decoder_step_opportunities": (
                4 * (len(runner.ARMS) - 1) * (3 if runner.CONFIG["max_bytes"] == 2 else 10)
            ),
        },
        "memory": {"peak_working_set_mib": 300},
        "worker": {"memory": {"peak_working_set_mib": 400}},
    }
    launch = {"elapsed_seconds": 10.0, "observed_tree_peak_sum_mib": 750}
    if hasattr(runner, "duration_clock"):
        report["duration_clock"] = runner.duration_clock()
    result = runner.project_cost(report, launch)
    assert result["projected_seconds"] == pytest.approx(
        2 * 1024 * len(runner.ARMS) * 0.03 + 2 * (10 - 0.04 * len(runner.ARMS)) + 30
    )
    assert result["peak_upper_mib"] == 750 and result["admissible"]
    frozen = "slow_frozen" if "slow_frozen" in runner.ARMS else "frozen"
    report["cap_generation"][frozen]["seconds"] = 0.0
    with pytest.raises(AssertionError):
        runner.project_cost(report, launch)
    report["cap_generation"][frozen]["seconds"] = 0.02
    if hasattr(runner, "duration_clock"):
        report["duration_clock"]["name"] = "monotonic"
        with pytest.raises(AssertionError):
            runner.project_cost(report, launch)
        report["duration_clock"] = runner.duration_clock()
    launch["observed_tree_peak_sum_mib"] = 2049
    assert not runner.project_cost(report, launch)["admissible"]
    launch["observed_tree_peak_sum_mib"] = 750
    report["lessons"][0]["seconds"] = 1.0
    assert not runner.project_cost(report, launch)["admissible"]
    report["lessons"][0]["seconds"] = float("nan")
    with pytest.raises(AssertionError):
        runner.project_cost(report, launch)


def test_local_successor_cost_binds_redirector_to_observed_owned_tree(tmp_path, local_cost_runner):
    runner, module = local_cost_runner

    admission = tmp_path / "admission.json"
    admission.write_text("{}")
    report = {
        "pid": 22, "worker": {"worker_pid": 33},
        "launch_started_unix": 100, "deadline_unix": 1000,
        "source_revision": "revision", "admission_sha256": runner.file_hash(admission),
        "source_sha256": {
            "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1": "launcher"
        },
    }
    runner.write_json(tmp_path / "report.json", report)
    launch = {
        "status": "passed", "exit_code": 0, "child_pid": 11,
        "started_unix": 100, "deadline_unix": 1000, "seconds_limit": 900,
        "tree_peak_sum_mib_limit": 2048, "launcher_sha256": "launcher",
        "command": [str(runner.ROOT / ".venv/Scripts/python.exe"), "-X", "utf8", "-m",
                    module, "cost", "--output",
                    str(tmp_path), "--admission", str(admission), "--admission-sha",
                    report["admission_sha256"], "--revision", "revision"],
        "observed_process_peaks_mib": {"11/101": 5, "22/102": 300, "33/103": 400},
    }
    runner.validate_cost_launch(launch, report)
    del launch["observed_process_peaks_mib"]["22/102"]
    with pytest.raises(AssertionError, match="process absent"):
        runner.validate_cost_launch(launch, report)
    launch["observed_process_peaks_mib"].update({"22/102": 300, "22/104": 400})
    with pytest.raises(AssertionError, match="PID reused"):
        runner.validate_cost_launch(launch, report)


@pytest.mark.parametrize("version", [4, 5])
def test_local_harness_spec_binds_source_and_preserves_fixed_workload(version):
    from dataclasses import FrozenInstanceError, replace
    from importlib import import_module

    from experiments.r4_integrated import local_memory_byte_harness as harness
    from experiments.r4_integrated import local_memory_byte_v2 as previous
    SPEC = import_module(f"experiments.r4_integrated.local_memory_byte_v{version}").SPEC

    assert {**SPEC.config, "schema": previous.CONFIG["schema"]} == previous.CONFIG
    modified = SPEC.config
    modified["arms"].clear()
    modified["seed"] = 0
    assert SPEC.config["arms"] == list(harness.ARMS) and SPEC.config["seed"] == 733
    with pytest.raises(FrozenInstanceError):
        SPEC.schema = "mutated"
    with pytest.raises(ValueError, match="module and entrypoint"):
        replace(SPEC, module="experiments.r4_integrated.local_memory_byte_v2")
    manifest = harness.sources(SPEC)
    for relative in (
        "experiments/r4_integrated/local_memory_byte_harness.py",
        "experiments/r4_integrated/local_memory_byte_v3.py",
        "experiments/r4_integrated/local_memory_byte_v3.md",
        f"experiments/r4_integrated/local_memory_byte_v{version}.py",
        f"experiments/r4_integrated/local_memory_byte_v{version}.md",
        "experiments/r4_integrated/local_memory_byte_v2.py",
        "src/humanbrain/plastic_memory.py",
        "src/humanbrain/byte_sequence_decoder.py",
    ):
        assert manifest[relative] == harness.file_hash(harness.ROOT / relative)
    # Closed-model initialization parity is preserved at its historical revision.


def test_local_harness_capped_prediction_measures_subtick_duration(monkeypatch):
    from experiments.r4_integrated import local_memory_two_byte_v1 as harness
    from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    from experiments.r4_integrated.local_memory_two_byte_v1 import SPEC

    model = LocalMemoryLearner(mode="frozen", seed=733)
    monkeypatch.setattr(harness.time, "monotonic", lambda: 1000.0)
    ticks = iter((2000.0, 2000.003))
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(ticks))
    with torch.no_grad():
        sample = harness.capped_prediction(SPEC, "frozen", model)
    assert sample["decisions"] == 3 and sample["stop_reason"] == "limit"
    assert sample["seconds"] == pytest.approx(0.003)


def test_local_harness_export_uses_same_high_resolution_clock(tmp_path, monkeypatch):
    from experiments.r4_integrated import local_memory_two_byte_v1 as harness
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    model = LocalMemoryLearner(writer_update_fraction=0.5,
        mode="frozen", input_code_size=2, hidden_size=3, key_size=2, value_size=94
    )
    checks = [
        {"id": index, "memory_id": str(index), "base_memory_id": None,
         "document_hex": [b"x".hex()], "query_hex": b"q".hex(), "delay_reads": 0}
        for index in range(12)
    ]
    report = {"source_sha256": {}}
    monkeypatch.setattr(harness.time, "monotonic", lambda: 1000.0)
    ticks = iter((2000.0, 2000.004, 2000.004, 2000.005))
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(ticks))
    harness.export_memories({"frozen": model}, checks, tmp_path, report, lambda: None)
    assert report["final_ingestion_seconds"] == pytest.approx(0.004)
    assert report["source_free_serialization_seconds"] == pytest.approx(0.001)
    assert (tmp_path / "source_free.pt").is_file()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mode", ["tagged", "aggregate", "writer_maps_frozen", "frozen", "writes_disabled"]
)
def test_local_forward_prediction_matches_full_query_and_subsequent_teaching(dtype, mode):
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    candidate = LocalMemoryLearner(writer_update_fraction=0.5,
        mode=mode, input_code_size=2, hidden_size=3, key_size=2, value_size=94, seed=19
    )
    _set_upstream_dtype(candidate, dtype)
    reference = copy.deepcopy(candidate)
    for documents, query, target in (([b"x=a", b"y=b"], b"x?", b"aa"),
                                     ([b"x=b", b"y=a"], b"y?", b"a!")):
        state = candidate.ingest(documents)
        baseline = reference.ingest(documents)
        before = candidate.snapshot(state)
        with patch.object(
            candidate.query_cell, "local_step", side_effect=AssertionError("no traces")
        ):
            actual_context = candidate._prediction_context(query, state)
            actual = candidate.predict(query, state, 2)
        with torch.no_grad():
            context, _, _ = reference._query(query, baseline)
            expected = reference.decoder.generate(context, 2)[0]
        assert torch.equal(actual_context, context)
        assert actual == expected
        _assert_nested_equal(before, candidate.snapshot(state))
        feedback = candidate.teach(query, target, state, 0.01)
        expected_feedback = reference.teach(query, target, baseline, 0.01)
        assert feedback.teacher_count == expected_feedback.teacher_count
        assert feedback.mean_loss == expected_feedback.mean_loss
        assert (feedback.decoder_updated, feedback.query_updated, feedback.memory_updated) == (
            expected_feedback.decoder_updated, expected_feedback.query_updated,
            expected_feedback.memory_updated
        )
        assert torch.equal(feedback.mean_context_signal, expected_feedback.mean_context_signal)
        _assert_nested_equal(candidate.snapshot(feedback.state),
                             reference.snapshot(expected_feedback.state))


def test_local_forward_prediction_uses_supplied_memory_and_preserves_it():
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    # Authored transport witness: this tests causal routing, not learned capability.
    learner = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=1, hidden_size=1, key_size=1, value_size=95
    )
    with torch.no_grad():
        learner.query_cell.weight.zero_()
        learner.query_cell.recurrent.zero_()
        learner.query_cell.bias.fill_(1)
        learner.memory.query_weight.fill_(1)
        learner.memory.query_bias.zero_()
        learner.input_codes.zero_()
        learner.decoder.output_weights.zero_()
        # Phase-zero read component, then a phase-only second byte to hit the cap.
        learner.decoder.output_weights[65, 3 + learner.query_size] = 1
        learner.decoder.output_weights[66, 3 + learner.query_size] = -1
    positive = learner.memory.initial_local_state(1)
    negative = learner.memory.initial_local_state(1)
    positive.synapses.fill_(1)
    negative.synapses.fill_(-1)
    for state, expected in ((positive, b"A"), (negative, b"B")):
        before = learner.snapshot(state)
        assert learner.predict(b"same question", state, 1) == ByteGeneration(expected, "limit")
        _assert_nested_equal(before, learner.snapshot(state))


@pytest.mark.parametrize(
    "corrupt", ["synapses", "credit", "cell", "input", "activity", "read", "logits"]
)
def test_local_forward_prediction_rejects_actual_nonfinite_state_and_computation(corrupt):
    from experiments.r4_integrated.local_memory_byte_harness import digest
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    learner = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=1, hidden_size=1, key_size=1, value_size=95
    )
    state = learner.memory.initial_local_state(1)
    with torch.no_grad():
        if corrupt == "synapses":
            state.synapses.fill_(torch.nan)
        elif corrupt == "credit":
            next(iter(state.credit.values())).fill_(torch.nan)
        elif corrupt == "cell":
            state.cell.activity.fill_(torch.nan)
        elif corrupt == "input":
            learner.input_codes[ord("q")].fill_(torch.nan)
        elif corrupt == "activity":
            learner.query_cell.bias.fill_(torch.nan)
        elif corrupt == "read":
            learner.memory.query_weight.fill_(torch.nan)
        else:
            learner.decoder.output_weights.fill_(torch.nan)
    before = digest(learner.snapshot(state))
    with pytest.raises(ValueError, match="finite"):
        learner.predict(b"q", state, 1)
    assert digest(learner.snapshot(state)) == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_forward_prediction_omits_unused_eligibility_overflow_but_teaching_rejects_it(dtype):
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    learner = LocalMemoryLearner(writer_update_fraction=0.5,
        input_code_size=1, hidden_size=1, key_size=1, value_size=95
    )
    _set_upstream_dtype(learner, dtype)
    with torch.no_grad():
        learner.input_codes.zero_()
        learner.query_cell.weight.zero_()
        learner.query_cell.bias.zero_()
        learner.query_cell.recurrent.fill_(3)
    state = learner.memory.initial_local_state(1)
    before = learner.snapshot(state)
    # h stays zero while unused bias eligibility doubles per step and overflows.
    # Singleton punctuation keeps the declared number of query-cell events.
    query = b"!" * (130 if dtype == torch.float32 else 1030)
    assert isinstance(learner.predict(query, state, 1), ByteGeneration)
    _assert_nested_equal(before, learner.snapshot(state))
    with pytest.raises(ValueError, match="finite"):
        learner.teach(query, b"aa", state, 0.01)
    _assert_nested_equal(before, learner.snapshot(state))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decoder_finite_validation_rejects_feedback_and_proposal_before_mutation(dtype):
    decoder = ByteSequenceDecoder(2, hidden_size=3, embedding_size=2, local_learning=True).to(dtype)
    context = torch.zeros(1, 2, dtype=dtype)
    step = decoder.local_step(context, torch.tensor([BYTE_BOS]))
    for name in ("hidden_feedback", "context_feedback"):
        buffer = getattr(decoder, name)
        original = buffer.clone()
        buffer[0, 0] = torch.inf
        before = copy.deepcopy(decoder.state_dict())
        with pytest.raises(ValueError, match="feedback signals must be finite"):
            decoder.teach_local(step, torch.tensor([65]), 0.01)
        _assert_nested_equal(before, decoder.state_dict())
        buffer.copy_(original)
    step.state.activity.fill_(torch.finfo(dtype).max)
    before = copy.deepcopy(decoder.state_dict())
    with pytest.raises(ValueError, match="nonfinite proposed byte readout update"):
        decoder.teach_local(step, torch.tensor([65]), 4.0)
    _assert_nested_equal(before, decoder.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_decoder_finite_validation_preserves_differentiable_strided_context(dtype):
    from humanbrain import byte_sequence_decoder as module

    decoder = ByteSequenceDecoder(2, hidden_size=3, embedding_size=2).to(dtype)
    base = torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=dtype, requires_grad=True)
    context = base.T
    assert not context.is_contiguous()
    outputs = [b"a", b"b"]
    candidate = decoder.loss(context, outputs)
    candidate.backward()
    gradients = [base.grad.clone(), *(p.grad.clone() for p in decoder.parameters())]
    decoder.zero_grad(set_to_none=True)
    base.grad = None
    with patch.object(module, "_all_finite", side_effect=lambda x: bool(torch.isfinite(x).all())):
        reference = decoder.loss(context, outputs)
        reference.backward()
    assert torch.equal(candidate, reference)
    for actual, expected in zip(
        [base.grad, *(p.grad for p in decoder.parameters())], gradients, strict=True
    ):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("mode", ["writer_maps_frozen", "frozen"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_frozen_writer_omits_dead_credit_with_exact_forward_and_learning_parity(mode, dtype):
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    learner = LocalMemoryLearner(writer_update_fraction=0.5,
        mode=mode, input_code_size=4, hidden_size=6, key_size=4, value_size=92, seed=39
    )
    _set_upstream_dtype(learner, dtype)
    reference = copy.deepcopy(learner)
    state = learner.memory.initial_local_state(1)
    reference_state = reference.memory.initial_local_state(1)
    for documents in ([b"first red", b"second blue"], [b"third green"], [b"first orange"]):
        previous = state
        before = learner.snapshot(previous)
        state = learner.ingest(documents, state)
        _assert_nested_equal(before["memory"], previous.snapshot())
        with torch.no_grad():
            for document in documents:
                values = reference._encoded(document)
                reference_state = reference.memory.write_local(
                    values, torch.ones(values.shape[:2], dtype=torch.bool), reference_state
                )
            reference_state = reference.memory.purge_local_writer(reference_state)
        assert torch.equal(state.synapses, reference_state.synapses)
        _assert_nested_equal(state.cell.snapshot(), reference_state.cell.snapshot())
        assert all(not tensor.count_nonzero() for tensor in state.credit.values())
        assert any(tensor.count_nonzero() for tensor in reference_state.credit.values())
        assert {
            name: (value.shape, value.numel() * value.element_size())
            for name, value in state.credit.items()
        } == {
            name: (value.shape, value.numel() * value.element_size())
            for name, value in reference_state.credit.items()
        }
        assert learner.predict(b"which?", state, 2) == reference.predict(
            b"which?", reference_state, 2
        )
        feedback = learner.teach(b"which?", b"bl", state, 0.01)
        reference_feedback = reference.teach(b"which?", b"bl", reference_state, 0.01)
        assert feedback.mean_loss == reference_feedback.mean_loss
        assert torch.equal(feedback.mean_context_signal, reference_feedback.mean_context_signal)
        assert feedback.teacher_count == reference_feedback.teacher_count == 3
        _assert_nested_equal(learner.state_dict(), reference.state_dict())
        # Retirement makes both full states equal again; fresh-process transport
        # preserves the fixed mode and permits the next cumulative ingestion.
        _assert_nested_equal(feedback.state.snapshot(), reference_feedback.state.snapshot())
        snapshot = learner.snapshot(feedback.state)
        restored = copy.deepcopy(learner)
        state = restored.restore_snapshot(snapshot)
        learner = restored
        reference_state = reference_feedback.state
        assert state.synapses.data_ptr() != snapshot["memory"]["synapses"].data_ptr()
        assert all(
            value.data_ptr() != snapshot["memory"]["credit"][name].data_ptr()
            for name, value in state.credit.items()
        )
        # The supplied snapshot remains an independent object through writes.
        assert before["memory"]["synapses"].data_ptr() != state.synapses.data_ptr()


def test_frozen_writer_control_rejects_later_unfreezing_before_ingest_or_teaching():
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    learner = LocalMemoryLearner(writer_update_fraction=0.5,
        mode="writer_maps_frozen", input_code_size=2, hidden_size=3, key_size=2, value_size=94
    )
    state = learner.ingest([b"a"])
    learner.memory.key_weight.requires_grad_(True)
    before = copy.deepcopy(learner.state_dict())
    with pytest.raises(ValueError, match="remain frozen"):
        learner.ingest([b"b"], state)
    with pytest.raises(ValueError, match="remain frozen"):
        learner.teach(b"q", b"aa", state, 0.01)
    _assert_nested_equal(before, learner.state_dict())


def test_direct_context_roster_balances_factors_and_excludes_heldout_training():
    import random
    from collections import Counter

    from experiments.r4_integrated import local_context_use_v1 as runner

    rows = runner.schedule()
    assert len(rows) == 384
    assert rows == runner.schedule()
    assert set(runner.TRAIN).isdisjoint(runner.HELDOUT)
    assert len(runner.TRAIN) == 12 and len(runner.HELDOUT) == 4
    rng = random.Random(734)
    for cycle in range(32):
        selected = rows[cycle * 12:(cycle + 1) * 12]
        shuffled = list(runner.TRAIN)
        rng.shuffle(shuffled)
        assert [row["contexts"]["shuffled"] for row in selected] == shuffled
        assert Counter(shuffled) == Counter(runner.TRAIN)
        for position in (0, 1):
            assert Counter(row["target"][position] for row in selected) == dict.fromkeys("abcd", 3)
        for row in selected:
            assert row["contexts"]["correct"] == row["contexts"]["frozen"] == row["target"]
            assert row["contexts"]["zero"] is None
            assert row["contexts"]["shuffled"] in runner.TRAIN
    for pair in runner.TRAIN + runner.HELDOUT:
        value = runner.context(pair)
        assert value.shape == (1, 96) and value.sum() == 2
        assert value[0, "abcd".index(pair[0])] == 1
        assert value[0, 4 + "abcd".index(pair[1])] == 1
        assert not value[0, 8:].any()
        assert torch.equal(runner.final_context("shuffled", pair), value)
        assert not runner.final_context("zero", pair).any()


def test_direct_context_matched_initialization_and_one_lesson_accounting():
    from experiments.r4_integrated import local_context_use_v1 as runner

    models = {arm: runner.new_decoder() for arm in runner.ARMS}
    initial = copy.deepcopy(models["correct"].state_dict())
    assert all(runner.same_state(initial, model.state_dict()) for model in models.values())
    assert runner.CONFIG["decoder_seed"] == runner.CONFIG["seed"] + 2 == 735
    row = runner.schedule()[0]
    for arm, model in models.items():
        with patch.object(model, "generate", wraps=model.generate) as generate, patch.object(
            model, "teach_local", wraps=model.teach_local
        ) as teach:
            receipt = runner.lesson(model, runner.context(row["contexts"][arm]), row["target"],
                                    enabled=arm != "frozen")
        assert generate.call_count == 1 and teach.call_count == 3
        assert receipt["byte_offers"] == 3
        assert receipt["enabled_steps"] == (0 if arm == "frozen" else 3)
        assert len(receipt["teacher_losses"]) == 3
        assert receipt["mean_loss"] == pytest.approx(sum(receipt["teacher_losses"]) / 3)
        assert receipt["before_sha256"] == receipt["after_first_sha256"]
        assert receipt["after_teacher_sha256"] == runner.state_hash(model)
        assert (receipt["before_sha256"] == receipt["after_teacher_sha256"]) == (arm == "frozen")
        assert all(parameter.grad is None for parameter in model.parameters())
    assert runner.same_state(initial, models["frozen"].state_dict())


def test_direct_context_first_generation_precedes_teacher_and_final_is_read_only():
    from experiments.r4_integrated import local_context_use_v1 as runner

    model = runner.new_decoder()
    order = []
    original_generate, original_teach = model.generate, model.teach_local

    def generate(*args, **kwargs):
        order.append("generate")
        return original_generate(*args, **kwargs)

    def teach(*args, **kwargs):
        order.append("teach")
        return original_teach(*args, **kwargs)

    with patch.object(model, "generate", side_effect=generate), patch.object(
        model, "teach_local", side_effect=teach
    ):
        runner.lesson(model, runner.context("aa"), "aa", enabled=True)
    assert order == ["generate", "teach", "teach", "teach"]
    before = runner.state_hash(model)
    generated = runner.generation(model, runner.context("aa"))
    assert runner.state_hash(model) == before
    assert generated["decisions"] == len(bytes.fromhex(generated["hex"])) + 1 <= 3


def test_direct_context_manifest_binds_seed_rule_and_rejects_wrong_source_before_allocation(
    tmp_path,
):
    from experiments.r4_integrated import local_context_use_v1 as runner

    sources = runner.sources()
    assert "src/humanbrain/local_memory_learner.py" in sources
    assert "experiments/r4_integrated/local_context_use_v1.md" in sources
    assert "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1" in sources
    output = tmp_path / "no-run"
    with (
        patch.object(runner.subprocess, "check_output", return_value="revision\n"),
        patch.object(runner, "new_decoder", side_effect=AssertionError("must not allocate")),
        pytest.raises(ValueError, match="source manifest"),
    ):
        runner.run(output, "revision", "wrong")
    assert not output.exists()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_symmetric_feedback_uses_preupdate_weights_and_same_read_only_forward(dtype):
    random = ByteSequenceDecoder(2, 3, 2, seed=11, local_learning=True).to(dtype=dtype)
    symmetric = ByteSequenceDecoder(
        2, 3, 2, seed=11, local_learning=True, hidden_feedback_mode="symmetric"
    ).to(dtype=dtype)
    context = torch.tensor([[0.2, -0.4]], dtype=dtype)
    inputs = torch.tensor([[BYTE_BOS, 97]])
    before = copy.deepcopy(symmetric.state_dict())
    with torch.no_grad():
        assert torch.equal(random(context, inputs), symmetric(context, inputs))
    assert random.generate(context, 2) == symmetric.generate(context, 2)
    _assert_decoder_state_equal(before, symmetric.state_dict())
    step = symmetric.local_step(context, inputs[:, 0])
    delta = step.logits.softmax(-1)
    delta[0, 98] -= 1
    expected_signal = torch.nn.functional.linear(delta, before["output.weight"].T)
    expected_context = torch.nn.functional.linear(delta, before["context_feedback"])
    with (
        patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")),
        patch.object(torch.autograd, "grad", side_effect=AssertionError("no reverse gradients")),
    ):
        receipt = symmetric.teach_local(step, torch.tensor([98]), 0.05)
    assert torch.equal(receipt.hidden_signal, expected_signal)
    assert torch.equal(receipt.context_signal, expected_context)
    assert not torch.equal(
        receipt.hidden_signal, torch.nn.functional.linear(delta, symmetric.output.weight.T)
    )
    for name in ("embedding.weight", "hidden_feedback", "context_feedback"):
        assert torch.equal(before[name], symmetric.state_dict()[name])
    assert all(parameter.grad is None for parameter in symmetric.parameters())


def test_symmetric_mode_round_trip_and_mismatched_configuration_rejection():
    decoder = ByteSequenceDecoder(2, 3, 2, local_learning=True, hidden_feedback_mode="symmetric")
    context = torch.tensor([[0.4, 0.3]])
    step = decoder.local_step(context, torch.tensor([BYTE_BOS]))
    decoder.teach_local(step, torch.tensor([65]), 0.01)
    buffer = io.BytesIO()
    torch.save(decoder.state_dict(), buffer)
    buffer.seek(0)
    saved = torch.load(buffer, weights_only=True)
    restored = ByteSequenceDecoder(
        2, 3, 2, seed=99, local_learning=True, hidden_feedback_mode="symmetric"
    )
    restored.load_state_dict(saved)
    _assert_decoder_state_equal(decoder.state_dict(), restored.state_dict())
    assert decoder.generate(context, 2) == restored.generate(context, 2)
    random = ByteSequenceDecoder(2, 3, 2, local_learning=True)
    with pytest.raises(ValueError, match="configuration mismatch"):
        random.set_extra_state(saved["_extra_state"])
    old = {**saved["_extra_state"], "local_learning": "fixed-embedding-random-feedback-v1"}
    old.pop("hidden_feedback_mode")
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.set_extra_state(old)
    with pytest.raises(ValueError, match="requires local_learning"):
        ByteSequenceDecoder(2, hidden_feedback_mode="symmetric")
    with pytest.raises(ValueError, match="must be random or symmetric"):
        ByteSequenceDecoder(2, local_learning=True, hidden_feedback_mode="invalid")


def test_context_feedback_comparison_has_exact_inputs_initial_tensors_and_strict_decision():
    from experiments.r4_integrated import local_context_use_v1 as previous
    from experiments.r4_integrated import local_context_use_v2 as runner

    assert tuple(
        f"{mode}_{condition}"
        for mode in ("random", "symmetric")
        for condition in ("correct", "zero", "shuffled", "frozen")
    ) == runner.ARMS
    assert runner.schedule() == previous.schedule()
    assert runner.lesson is previous.lesson and runner.generation is previous.generation
    for key, value in previous.CONFIG.items():
        if key != "schema":
            assert runner.CONFIG[key] == value
    states = {
        arm: runner.new_decoder(mode).state_dict() for arm, (mode, _) in runner.ARM_SPECS.items()
    }
    assert runner.matched_initial_tensors(states)
    baseline = previous.new_decoder().state_dict()
    assert all(
        torch.equal(value, states["symmetric_correct"][key])
        for key, value in baseline.items()
        if isinstance(value, torch.Tensor)
    )
    assert states["symmetric_correct"]["_extra_state"]["hidden_feedback_mode"] == "symmetric"
    states["symmetric_correct"]["output.bias"][0] += 1
    assert not runner.matched_initial_tensors(states)
    scores = {arm: {"train": 0, "heldout": 0} for arm in runner.ARMS}
    scores["symmetric_correct"] = {"train": 12, "heldout": 4}
    assert runner.qualifies(scores)
    for comparator in (
        "random_correct",
        "symmetric_zero",
        "symmetric_shuffled",
        "symmetric_frozen",
    ):
        changed = copy.deepcopy(scores)
        changed[comparator]["heldout"] = 4
        assert not runner.qualifies(changed)
    scores["symmetric_correct"]["train"] = 11
    assert not runner.qualifies(scores)
    manifest = runner.sources()
    assert manifest.items() >= previous.sources().items()
    assert "experiments/r4_integrated/local_context_use_v2.md" in manifest



def test_learnability_bracket_fixed_balanced_orders_certificate_and_decisions():
    import random
    from collections import Counter

    from experiments.r4_integrated import local_context_learnability_v1 as runner

    rows = runner.schedule()
    assert len(rows) == 768 and runner.CONFIG["checkpoints"] == [384, 768]
    expected_arms = (
        "decoder_lexical",
        "decoder_permuted",
        "reference_lexical",
        "reference_permuted",
        "reference_zero",
        "reference_frozen",
    )
    assert expected_arms == runner.ARMS
    rng = random.Random(734)
    for cycle in range(64):
        permuted = list(runner.TRAIN)
        rng.shuffle(permuted)
        group = rows[cycle * 12 : (cycle + 1) * 12]
        assert [row["permuted_target"] for row in group] == permuted
        assert [row["lexical_target"] for row in group] == list(runner.TRAIN)
        assert Counter(permuted) == Counter(runner.TRAIN)
        assert set(permuted).isdisjoint(runner.HELDOUT)
        for row in group:
            assert runner.assignment("decoder_permuted", row) == runner.assignment(
                "reference_permuted", row
            )
            target, supplied = runner.assignment("reference_zero", row)
            assert target == row["lexical_target"] and supplied is None
    fake = [
        {"teacher_losses": [int(index % 12 == 0 and index // 12 < 54), 0, 0]}
        for index in range(768)
    ]
    certificate = runner.certificate(fake, 12)
    assert certificate["passed"] and certificate["teacher_mistakes"] == 54
    assert certificate["first_zero_mistake_cycle"] == 55
    assert not runner.certificate(fake, 11)["passed"]
    fake[54 * 12]["teacher_losses"][0] = 1
    assert not runner.certificate(fake, 12)["passed"]
    scores = {arm: {"train": 0, "heldout": 0} for arm in runner.ARMS}
    scores["decoder_permuted"] = {"train": 12, "heldout": 4}
    assert runner.useful(scores)["decoder_permuted"]
    for control in ("reference_zero", "reference_frozen"):
        changed = copy.deepcopy(scores)
        changed[control]["heldout"] = 4
        assert not runner.useful(changed)["decoder_permuted"]


def test_learnability_decoder_initialization_and_loss_types_are_explicit():
    from experiments.r4_integrated import local_context_learnability_v1 as runner
    from experiments.r4_integrated import local_context_use_v1 as previous

    assert runner.lesson is previous.lesson
    decoder = runner.new_model("decoder_lexical")
    assert runner.same_state(decoder.state_dict(), previous.new_decoder().state_dict())
    assert runner.same_state(
        decoder.state_dict(), runner.new_model("decoder_permuted").state_dict()
    )
    initial = runner.new_model("reference_lexical").state_dict()
    for arm in ("reference_permuted", "reference_zero", "reference_frozen"):
        assert runner.same_state(initial, runner.new_model(arm).state_dict())
    fixture = [
        {
            "loss_kind": "reference_teacher_mistake",
            "first": {"exact": False, "stop_reason": "limit"},
            "mean_loss": 1 / 3,
            "teacher_losses": [1, 0, 0],
        }
    ]
    assert runner.summarize(fixture)["loss_kind"] == "reference_teacher_mistake"
    assert runner.summarize(fixture)["mean_loss_by_byte1_byte2_eos"] == [1, 0, 0]
    assert "src/humanbrain/phase_context_perceptron.py" in runner.sources()



@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_phase_reference_generic_clock_features_and_integer_local_update(dtype):
    from humanbrain.phase_context_perceptron import PhaseContextPerceptron

    model = PhaseContextPerceptron()
    context = torch.zeros(1, 96, dtype=dtype)
    context[0, [13, 72]] = 1
    initial = copy.deepcopy(model.state_dict())
    state = None
    for phase, token in enumerate((BYTE_BOS, 65, 255)):
        before_phase = None if state is None else state.phase.clone()
        step = model.local_step(context, torch.tensor([token]), state)
        expected = torch.zeros(1, 291, dtype=torch.int32)
        expected[0, [phase, 3 + phase * 96 + 13, 3 + phase * 96 + 72]] = 1
        assert torch.equal(step.features, expected)
        assert int(step.features.square().sum()) == 3
        assert torch.equal(step.scores, torch.zeros(1, 257, dtype=torch.int64))
        if state is not None:
            alternative = model.local_step(context, torch.tensor([23]), state)
            assert torch.equal(alternative.features, step.features)
            assert torch.equal(state.phase, before_phase)
        state = step.state
    assert not state.phase.any()
    with pytest.raises(ValueError, match="exhausted"):
        model.local_step(context, torch.tensor([65]), state)
    _assert_decoder_state_equal(initial, model.state_dict())
    step = model.local_step(context, torch.tensor([BYTE_BOS]))
    with (
        patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")),
        patch.object(torch.autograd, "grad", side_effect=AssertionError("no gradients")),
    ):
        receipt = model.teach_local(step, torch.tensor([65]), 0.01)
    assert (receipt.winner, receipt.target, receipt.phase, receipt.loss) == (0, 65, 0, 1)
    assert receipt.updated and receipt.teacher_count == 1
    expected_ticks = torch.zeros(257, 291, dtype=torch.int32)
    expected_ticks[65] = step.features[0]
    expected_ticks[0] = -step.features[0]
    assert torch.equal(model.output_ticks, expected_ticks)
    with pytest.raises(ValueError, match="stale step"):
        model.teach_local(step, torch.tensor([65]), 0.01)
    current = model.local_step(context, torch.tensor([BYTE_BOS]))
    assert not model.teach_local(current, torch.tensor([65]), 0.01).updated
    frozen = model.teach_local(current, torch.tensor([66]), 0.01, learning_enabled=False)
    assert frozen.loss == 1 and not frozen.updated and frozen.teacher_count == 1
    assert torch.equal(model.output_ticks, expected_ticks)
    snapshot = current.snapshot()
    snapshot["features"].zero_()
    assert current.features.any()
    assert current.tensor_bytes == 3232 and current.state.tensor_bytes == 12
    assert (
        model.accounting()["serialized_tensor_bytes"]
        == sum(value.numel() * value.element_size() for value in model.buffers())
        == 302652
    )
    assert not list(model.parameters())


def test_phase_reference_binary_validation_overflow_and_fixed_state_rejection():
    from humanbrain.phase_context_perceptron import PhaseContextPerceptron

    model = PhaseContextPerceptron()
    context = torch.zeros(1, 96)
    context[0, [13, 72]] = 1
    for value in (0.5, float("nan"), float("inf")):
        invalid = context.clone()
        invalid[0, 0] = value
        with pytest.raises(ValueError, match="binary"):
            model.local_step(invalid, torch.tensor([BYTE_BOS]))
    model.output_ticks[65, 0] = torch.iinfo(torch.int32).max
    model.output_ticks[65, 3 + 13] = -torch.iinfo(torch.int32).max
    step = model.local_step(context, torch.tensor([BYTE_BOS]))
    assert int(step.scores.argmax()) == 0
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="overflow"):
        model.teach_local(step, torch.tensor([65]), 0.01)
    _assert_decoder_state_equal(before, model.state_dict())
    with pytest.raises(ValueError, match=r"0\.01 tick"):
        model.teach_local(step, torch.tensor([65]), 0.02)
    model.clock_transition[0, 0] = 1
    with pytest.raises(ValueError, match="remain fixed"):
        model.generate(context, 2)


def test_phase_reference_lesson_receipts_restore_and_read_only_generation():
    from experiments.r4_integrated import local_context_learnability_v1 as runner
    from humanbrain.phase_context_perceptron import PhaseContextPerceptron

    model = PhaseContextPerceptron()
    baseline = copy.deepcopy(model)
    value = runner.context("aa")
    first = runner.reference_lesson(model, value, "aa", enabled=True)
    ordinary = runner.lesson(baseline, value, "aa", enabled=True)
    assert {key: value for key, value in first.items() if key != "teachers"} == ordinary
    assert [row["phase"] for row in first["teachers"]] == [0, 1, 2]
    assert [row["target"] for row in first["teachers"]] == [97, 97, BYTE_EOS]
    assert all(row["winner_score"] >= row["target_score"] for row in first["teachers"])
    assert all(len(row["active_features"]) == 3 for row in first["teachers"])
    assert first["byte_offers"] == 3 and first["enabled_steps"] == 3
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = PhaseContextPerceptron()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    before = runner.state_hash(restored)
    assert model.generate(value, 2) == restored.generate(value, 2)
    assert runner.state_hash(restored) == before
    frozen = runner.reference_lesson(restored, value, "aa", enabled=False)
    assert frozen["byte_offers"] == 3 and frozen["enabled_steps"] == 0
    assert frozen["before_sha256"] == frozen["after_teacher_sha256"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.set_extra_state({**restored.get_extra_state(), "tick_size": 0.02})



@pytest.mark.parametrize("input_dtype", [torch.float32, torch.float64])
def test_phase_lms_signed_features_and_preupdate_all_output_residuals(input_dtype):
    from humanbrain.phase_context_lms import PhaseContextLMS

    model = PhaseContextLMS()
    value = torch.zeros(1, 96, dtype=input_dtype)
    value[0, 13], value[0, 72] = -0.25, 0.5
    state = None
    for phase, token in enumerate((BYTE_BOS, 0, 255)):
        step = model.local_step(value, torch.tensor([token]), state)
        expected = torch.zeros(1, 291, dtype=torch.float64)
        expected[0, phase] = 1
        expected[0, 3 + phase * 96 + 13] = -0.25
        expected[0, 3 + phase * 96 + 72] = 0.5
        assert torch.equal(step.features, expected)
        if state is not None:
            assert torch.equal(
                step.features, model.local_step(value, torch.tensor([42]), state).features
            )
        state = step.state
    assert not state.phase.any()
    with pytest.raises(ValueError, match="exhausted"):
        model.local_step(value, torch.tensor([1]), state)
    # A correct winner still learns its nonzero residual; another neuron's
    # nonzero potential receives its own correction even though it is not target.
    model.output_weights[65, 0] = 0.8
    model.output_weights[123, 0] = 0.25
    step = model.local_step(value, torch.tensor([BYTE_BOS]))
    before = model.output_weights.clone()
    saved_scores = step.scores.clone()
    residual = -saved_scores
    residual[0, 65] += 1
    # Independent scalar oracle for the only two nonzero output residuals and
    # three nonzero input rates. Full equality also checks all untouched entries.
    expected = before.clone()
    for output_id, error in ((65, 1.0 - 0.8), (123, -0.25)):
        for feature_id, rate in ((0, 1.0), (3 + 13, -0.25), (3 + 72, 0.5)):
            expected[output_id, feature_id] = (
                float(before[output_id, feature_id]) + (0.01 * error) * rate
            )
    with (
        patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")),
        patch.object(torch.autograd, "grad", side_effect=AssertionError("no gradients")),
    ):
        receipt = model.teach_local(step, torch.tensor([65]), 0.01)
    assert receipt.winner == receipt.target == 65 and receipt.updated
    assert receipt.teacher_count == 1 and receipt.phase == 0
    assert torch.equal(receipt.residual, residual)
    assert torch.equal(step.scores, saved_scores)
    assert torch.equal(model.output_weights, expected)
    assert receipt.loss == pytest.approx(
        0.5 * math.fsum(float(x) ** 2 for x in residual[0]), rel=1e-15
    )
    assert model.output_weights[123, 0] < before[123, 0]
    with pytest.raises(ValueError, match="stale step"):
        model.teach_local(step, torch.tensor([65]), 0.01)
    current = model.local_step(value, torch.tensor([BYTE_BOS]))
    frozen_before = copy.deepcopy(model.state_dict())
    frozen = model.teach_local(current, torch.tensor([66]), 0.01, learning_enabled=False)
    assert frozen.teacher_count == 1 and not frozen.updated and frozen.residual.shape == (1, 257)
    _assert_decoder_state_equal(frozen_before, model.state_dict())
    snapshot = current.snapshot()
    snapshot["features"].zero_()
    assert current.features.any()
    assert current.tensor_bytes == 4396 and current.state.tensor_bytes == 12
    assert model.accounting()["serialized_tensor_bytes"] == 602952
    assert not list(model.parameters())


def test_phase_lms_nonfinite_input_and_atomic_proposed_overflow_rejection():
    from humanbrain.phase_context_lms import PhaseContextLMS

    model = PhaseContextLMS()
    for invalid in (float("nan"), float("inf")):
        value = torch.zeros(1, 96, dtype=torch.float64)
        value[0, 13] = invalid
        with pytest.raises(ValueError, match="finite"):
            model.generate(value, 2)
    value = torch.zeros(1, 96, dtype=torch.float64)
    value[0, 13] = 1e200
    model.output_weights[0, 0] = 1e154
    step = model.local_step(value, torch.tensor([BYTE_BOS]))
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="proposed synapses"):
        model.teach_local(step, torch.tensor([65]), 0.01)
    _assert_decoder_state_equal(before, model.state_dict())
    model.gating_synapses[0, 0, 0] = 0
    with pytest.raises(ValueError, match="remain fixed"):
        model.generate(torch.zeros(1, 96), 2)


def test_lms_runner_saved_free_margins_use_single_call_and_missing_phase_fails():
    from experiments.r4_integrated import local_context_lms_v1 as runner
    from humanbrain.phase_context_lms import PhaseContextLMS

    model = PhaseContextLMS()
    model.output_weights[97, 0] = 0.8
    model.output_weights[97, 1] = 0.7
    model.output_weights[BYTE_EOS, 2] = 0.6
    value = runner.context("aa")
    before = runner.state_hash(model)
    with patch.object(model, "local_step", wraps=model.local_step) as steps:
        emitted = runner.generation(model, value)
    assert steps.call_count == 3
    assert emitted["hex"] == b"aa".hex() and emitted["stop_reason"] == "eos"
    assert model.generate(value, 2)[0] == ByteGeneration(b"aa", "eos")
    assert runner.state_hash(model) == before
    assert runner.target_margins(emitted, "aa") == [0.8, 0.7, 0.6]
    model.output_weights.zero_()
    model.output_weights[BYTE_EOS, 0] = 1
    with patch.object(model, "local_step", wraps=model.local_step) as steps:
        early = runner.generation(model, value)
        margins = runner.target_margins(early, "aa")
    assert steps.call_count == 1 and margins == [-1, None, None]
    assert runner.sparse(torch.tensor([0.0, -0.25, 0.0, 0.5])) == [[1, -0.25], [3, 0.5]]
    scores = {arm: {"train": 0, "heldout": 0} for arm in runner.ARMS}
    scores["lexical"] = {"train": 12, "heldout": 4}
    checks = [{"arm": "lexical", "margin_passed": True} for _ in range(16)]
    assert runner.qualifies(scores, checks)["lexical"]
    checks[0]["margin_passed"] = False
    assert not runner.qualifies(scores, checks)["lexical"]
    checks[0]["margin_passed"] = True
    scores["zero"]["heldout"] = 4
    assert not runner.qualifies(scores, checks)["lexical"]


def test_lms_runner_single_disposable_lesson_receipts_restore_and_counts():
    from experiments.r4_integrated import local_context_lms_v1 as runner
    from humanbrain.phase_context_lms import PhaseContextLMS

    model = PhaseContextLMS()
    value = runner.context("aa")
    with patch.object(model, "local_step", wraps=model.local_step) as steps:
        record = runner.lesson(model, value, "aa", enabled=True)
    assert steps.call_count == record["first"]["decisions"] + 3
    assert record["byte_offers"] == 3 and record["residual_offers"] == 771
    assert record["enabled_steps"] == 3 and record["enabled_residual_offers"] == 771
    assert record["nonzero_residual_offers"] == record["enabled_nonzero_residual_offers"] == 3
    assert record["loss_kind"] == "half_sum_squared_potential_error"
    assert [row["target"] for row in record["teachers"]] == [97, 97, BYTE_EOS]
    assert [row["phase"] for row in record["teachers"]] == [0, 1, 2]
    for receipt in record["teachers"]:
        scores = dict(receipt["nonzero_scores"])
        expected = [
            (index, float(index == receipt["target"]) - scores.get(index, 0.0))
            for index in range(257)
        ]
        assert receipt["nonzero_residuals"] == [
            [index, value] for index, value in expected if value != 0
        ]
        assert receipt["loss"] == 0.5 * math.fsum(value**2 for _, value in expected)
    assert record["before_sha256"] == record["after_first_sha256"]
    assert record["after_teacher_sha256"] == runner.state_hash(model)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = PhaseContextLMS()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert runner.state_hash(restored) == runner.state_hash(model)
    frozen = runner.lesson(restored, value, "aa", enabled=False)
    assert frozen["byte_offers"] == 3 and frozen["residual_offers"] == 771
    assert frozen["enabled_steps"] == frozen["enabled_residual_offers"] == 0
    assert frozen["before_sha256"] == frozen["after_teacher_sha256"]
    with pytest.raises(ValueError, match="configuration mismatch"):
        restored.set_extra_state({**restored.get_extra_state(), "dtype": "float32"})


def test_lms_runner_fixed_160_cycle_balanced_schedule_without_training():
    import random

    from experiments.r4_integrated import local_context_lms_v1 as runner

    assert runner.ARMS == ("lexical", "permuted", "zero", "frozen")
    assert runner.CONFIG["lessons_per_arm"] == 1920
    assert runner.CONFIG["minimum_final_margin_exclusive"] == 0.10
    rows = runner.schedule()
    assert len(rows) == 1920
    rng = random.Random(734)
    for cycle in range(160):
        permuted = list(runner.TRAIN)
        rng.shuffle(permuted)
        group = rows[cycle * 12:(cycle + 1) * 12]
        assert [row["permuted_target"] for row in group] == permuted
        assert [row["lexical_target"] for row in group] == list(runner.TRAIN)
        assert set(permuted).isdisjoint(runner.HELDOUT)
        for row in group:
            target, supplied = runner.assignment("zero", row)
            assert target == row["lexical_target"] and supplied is None
    assert runner.context("aa").dtype == torch.float64
    assert "docs/research/local-composition-return-review.md" in runner.sources()
    assert "src/humanbrain/phase_context_lms.py" in runner.sources()


def test_phase_lms_context_signal_is_loss_sign_and_preupdate_active_phase():
    from humanbrain.phase_context_lms import PhaseContextLMS

    model = PhaseContextLMS()
    context = torch.zeros(1, 96, dtype=torch.float64)
    context[0, 17] = 0.25
    model.output_weights[65, 3 + 17] = 2
    model.output_weights[66, 3 + 17] = -1
    model.output_weights[65, 3 + 96 + 17] = 100  # Inactive phase must not contribute.
    before = model.output_weights.clone()
    step = model.local_step(context, torch.tensor([BYTE_BOS]))
    with patch.object(torch.autograd, "grad", side_effect=AssertionError("no backward")):
        receipt = model.teach_local(step, torch.tensor([65]), 0.01)
    # Independent scalar oracle: (0.5-1)*2 + (-0.25-0)*(-1).
    assert float(receipt.context_signal[0, 17]) == -0.75
    assert receipt.context_signal.count_nonzero() == 1
    assert model.output_weights[65, 20] == float(before[65, 20]) + 0.01 * 0.5 * 0.25
    assert torch.equal(model.output_weights[:, 99:], before[:, 99:])
    assert not receipt.context_signal.requires_grad


def test_local_lms_owns_old_seed_input_codes_and_rejects_invalid_teacher_before_query():
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    rng = torch.random.get_rng_state().clone()
    model = LocalMemoryLearner(writer_update_fraction=0.5, seed=733, hidden_size=3, key_size=2)
    old_codes = ByteSequenceDecoder(96, 3, 32, seed=735).embedding.weight
    assert torch.equal(model.input_codes, old_codes)
    assert not model.input_codes.requires_grad and not hasattr(model.decoder, "embedding")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert model.decoder.output_weights.dtype == torch.float64
    assert not model.decoder.output_weights.any()
    state = model.memory.initial_local_state(1)
    before = model.snapshot(state)
    with patch.object(model, "_query", side_effect=AssertionError("query must not run")):
        for teacher, rate in ((b"", 0.01), (b"a", 0.01), (b"aaa", 0.01), (b"aa", 0.02)):
            with pytest.raises(ValueError):
                model.teach(b"q", teacher, state, rate)
    _assert_nested_equal(before, model.snapshot(state))


@pytest.mark.parametrize("failure", ["cast", "loss"])
def test_local_lms_nonfinite_aggregate_rejects_before_upstream_mutation(failure):
    from dataclasses import replace

    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    model = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=2, value_size=94, hidden_size=3, key_size=2
    )
    state = model.ingest([b"x=aa"])
    before = model.snapshot(state)
    teach = model.decoder.teach_local

    def injected(*args, **kwargs):
        receipt = teach(*args, **kwargs)
        if failure == "cast":
            return replace(receipt, context_signal=torch.full((1, 96), 1e40, dtype=torch.float64))
        return replace(receipt, loss=1e308)

    with patch.object(model.decoder, "teach_local", side_effect=injected), pytest.raises(
        ValueError, match="after upstream cast" if failure == "cast" else "teacher loss"
    ):
        model.teach(b"q", b"aa", state, 0.01, learning_enabled=False)
    _assert_nested_equal(before, model.snapshot(state))


def test_two_byte_roster_counterfactuals_and_frozen_schedule_without_fit():
    from collections import Counter
    from dataclasses import FrozenInstanceError

    from experiments.r4_integrated import local_memory_two_byte_v1 as runner

    train, checks = runner.roster()
    values = [x["value"] for x in train]
    assert values == ["aa", "ab", "bb", "bc", "cc", "cd", "dd", "da"]
    assert all(
        Counter(x[position] for x in values) == dict.fromkeys("abcd", 2) for position in (0, 1)
    )
    assert Counter(x["group"] for x in checks) == {
        "fit": 8,
        "unseen_value": 4,
        "delayed": 4,
        "interference": 4,
    }
    assert len({x["memory_id"] for x in checks}) == 12
    novel, delayed, interference = checks[8:12], checks[12:16], checks[16:20]
    assert [bytes.fromhex(x["expected_hex"]) for x in novel] == [b"ac", b"ca", b"bd", b"db"]
    for pair in (novel[:2], novel[2:]):
        assert pair[0]["query_hex"] == pair[1]["query_hex"]
        assert pair[0]["document_hex"] != pair[1]["document_hex"]
        assert pair[0]["memory_id"] != pair[1]["memory_id"]
    for fresh, delay, extra in zip(novel, delayed, interference, strict=True):
        assert delay["memory_id"] == extra["base_memory_id"] == fresh["memory_id"]
        assert delay["delay_reads"] == 32 and not delay["additional_document_hex"]
        assert len(extra["additional_document_hex"]) == 3
        assert extra["expected_hex"] == fresh["expected_hex"]
    schedule = runner.teaching_schedule(train)
    assert len(schedule) == 256 and all(len(group) == 4 for group in schedule)
    assert schedule == runner.teaching_schedule(train)
    query_targets = {}
    for episode in [x for group in schedule for x in group]:
        assert bytes.fromhex(episode["target_hex"]).decode() in values
        assert len(episode["document_hex"]) == 2
        query_targets.setdefault(episode["query_hex"], set()).add(episode["target_hex"])
    assert all(len(targets) == 8 for targets in query_targets.values())
    mutated = runner.SPEC.config
    mutated["arms"].clear()
    assert len(runner.SPEC.config["arms"]) == 5
    with pytest.raises(FrozenInstanceError):
        runner.SPEC.schema = "other"
    manifest = runner.sources()
    for name in ("local_memory_two_byte_v1.py", "local_memory_two_byte_v1.md"):
        path = runner.ROOT / "experiments/r4_integrated" / name
        assert manifest[path.relative_to(runner.ROOT).as_posix()] == runner.file_hash(path)


def test_two_byte_lesson_actual_lms_receipts_before_update_and_retirement():
    from experiments.r4_integrated import local_memory_two_byte_v1 as runner
    from experiments.r4_integrated.local_memory_event_candidate_reference import (
        LocalMemoryLearner,
    )

    model = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=2, value_size=94, hidden_size=3, key_size=2
    )
    episode = {"document_hex": [b"x=aa".hex()], "query_hex": b"x?".hex(), "target_hex": b"aa".hex()}
    with patch.object(model, "predict", wraps=model.predict) as predict, patch.object(
        model, "teach", wraps=model.teach
    ) as teach:
        receipt = runner.lesson(runner.SPEC, model, episode, lambda: None)
    assert predict.call_count == teach.call_count == 1
    assert receipt["teacher_count"] == receipt["decoder_changed_steps"] == 3
    assert receipt["nonzero_residual_offers"] == receipt["enabled_nonzero_residual_offers"] == 3
    assert not receipt["query_updated"] and not receipt["memory_updated"]
    assert receipt["loss_kind"] == "half_sum_squared_potential_error"
    assert receipt["memory_sha256"] != receipt["retired_memory_sha256"]


@pytest.mark.parametrize(
    "runner_module", ["local_shared_temporal_query_v1"]
)
def test_two_byte_source_free_worker_fresh_process_read_only_checkpoint(tmp_path, runner_module):
    import importlib
    import json
    import os
    import subprocess
    import sys
    import time

    runner = importlib.import_module("experiments.r4_integrated." + runner_module)

    saved = {}
    for arm in runner.ARMS:
        model = runner.new_model(runner.SPEC, arm)
        state = model.ingest([b""])
        saved[arm] = {
            "weights": runner.detached_state(model.state_dict()),
            "memories": {str(i): runner.detached_state(state.snapshot()) for i in range(12)},
        }
    checkpoint, requests, output = (
        tmp_path / name for name in ("state.pt", "queries.json", "generated.json")
    )
    torch.save(saved, checkpoint)
    cases = [
        {"id": i, "memory_id": str(i), "query_hex": b"same question".hex(), "delay_reads": 0}
        for i in (0, 11)
    ]
    runner.write_json(requests, {"source_sha256": runner.sources(), "cases": cases})
    hashes = [runner.file_hash(p) for p in (checkpoint, requests)]
    environment = dict(os.environ)
    environment.update(
        HUMANBRAIN_DEVICE="cpu",
        HUMANBRAIN_NATIVE_CACHE=str(runner.ROOT / ".native-cache"),
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
    )
    child = subprocess.run(
        [
            sys.executable,
            "-I",
            "-X",
            "utf8",
            str(runner.SPEC.entrypoint),
            "worker",
            "--output",
            str(output),
            "--checkpoint",
            str(checkpoint),
            "--requests",
            str(requests),
            "--checkpoint-sha",
            hashes[0],
            "--requests-sha",
            hashes[1],
            "--deadline-unix",
            str(time.time() + 45),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert child.returncode == 0, child.stderr
    result = json.loads(output.read_text())
    assert result["worker_pid"] != os.getpid()
    assert (
        result["status"] == "completed"
        and not result["roster_invoked_or_documents_or_targets_supplied_to_model"]
    )
    assert result["query_byte_steps"] == 2 * len(runner.ARMS) * (len(b"same question") + 1)
    assert len(result["outputs"]) == 2 * len(runner.ARMS)
    for arm, work in result["work"].items():
        assert work["query_raw_bytes"] == 26 and work["query_eos_symbols"] == 2
        assert work["query_lag_shift_steps"] == 28
        if arm != "byte_reference":
            assert work["codec_slot_evaluations"] == (6 if arm == "writes_disabled" else 2)
            assert work["teacher_offers"] == 0
    for row in result["outputs"]:
        capped = row["arm"] == "writes_disabled"
        assert row["generated_hex"] == ("0000" if capped else "")
        assert row["stop_reason"] == ("limit" if capped else "eos")
        assert row["decisions"] == (3 if capped else 1)
        assert row["memory_sha256"] == runner.digest(
            saved[row["arm"]]["memories"][row["memory_id"]]
        )
    assert hashes == [runner.file_hash(p) for p in (checkpoint, requests)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mode", ["tagged", "aggregate", "writer_maps_frozen", "frozen", "writes_disabled"]
)
def test_localization_token_trace_matches_whole_ingest_and_single_purge(dtype, mode):
    from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    from experiments.r4_integrated.local_memory_localization_core import traced_ingest

    model = LocalMemoryLearner(writer_update_fraction=0.5,
        mode=mode, input_code_size=2, value_size=94, hidden_size=3, key_size=2
    )
    _set_upstream_dtype(model, dtype)
    weights = copy.deepcopy(model.state_dict())
    documents = [b"A", b"BC"]
    ordinary = model.ingest(documents)
    state, rows = traced_ingest(model, documents)
    _assert_nested_equal(ordinary.snapshot(), state.snapshot())
    _assert_nested_equal(weights, model.state_dict())
    assert [r["token"] for r in rows] == [65, BYTE_EOS, 66, 67, BYTE_EOS]
    assert [r["document_index"] for r in rows] == [0, 0, 1, 1, 1]
    assert torch.equal(rows[2]["writer_before"], rows[1]["writer"])
    assert rows[2]["writer_before"].count_nonzero()
    assert not state.cell.activity.count_nonzero()
    assert torch.equal(rows[-1]["synapses"], state.synapses)
    previous = torch.zeros_like(state.synapses, dtype=torch.float64)
    for row in rows:
        key, value, beta = (row[k].double() for k in ("key", "value", "beta"))
        predicted = torch.bmm(previous, key[:, :, None]).squeeze(-1)
        derived = previous + (beta * (value - predicted))[:, :, None] * key[:, None, :]
        torch.testing.assert_close(derived, row["synapses"].double(), atol=1e-6, rtol=1e-6)
        previous = row["synapses"].double()
    snapshot = state.snapshot()
    rows[-1]["synapses"].zero_()
    rows[-1]["writer"].zero_()
    _assert_nested_equal(snapshot, state.snapshot())


@pytest.mark.parametrize("ending", ["first_eos", "second_eos", "full_eos", "cap"])
def test_localization_observations_match_prediction_and_count_supplemental_phases(ending):
    from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    from experiments.r4_integrated.local_memory_localization_core import observed_query

    model = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=2, value_size=94, hidden_size=3, key_size=2
    )
    with torch.no_grad():
        model.decoder.output_weights[97, :3] = 1
        if ending != "cap":
            phase = {"first_eos": 0, "second_eos": 1, "full_eos": 2}[ending]
            model.decoder.output_weights[BYTE_EOS, phase] = 2
    state = model.ingest([b"A", b"BC"])
    before = model.snapshot(state)
    observation = observed_query(model, b"q", state)
    ordinary = model.predict(b"q", state, 2)
    context, query_state, event = model._query(b"q", state)
    assert observation["generated_hex"] == ordinary.data.hex()
    assert observation["stop_reason"] == ordinary.stop_reason
    assert observation["decisions"] == len(ordinary.data) + 1
    assert observation["phase_evaluations"] == 3
    assert observation["extra_phase_evaluations"] == 3 - observation["decisions"]
    assert torch.equal(observation["context"], context)
    assert torch.equal(observation["query_activity"], query_state.activity)
    assert torch.equal(observation["query_key"], event.key)
    assert torch.equal(observation["read"], event.value)
    _assert_nested_equal(before, model.snapshot(state))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_localization_proposal_matches_actual_teacher_and_isolates_two_blocks(dtype):
    from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    from experiments.r4_integrated.local_memory_localization_core import (
        observed_query,
        perturbed_model,
        propose_blocks,
    )

    model = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=2, value_size=94, hidden_size=3, key_size=2
    )
    _set_upstream_dtype(model, dtype)
    with torch.no_grad():
        model.decoder.output_weights.copy_(torch.linspace(
            -0.002, 0.002, model.decoder.output_weights.numel(), dtype=torch.float64
        ).reshape_as(model.decoder.output_weights))
    state = model.ingest([b"x=ab", b"y=bc"])
    before = model.snapshot(state)
    observation = observed_query(model, b"x?", state)
    reference = copy.deepcopy(model)
    receipt = reference.teach(b"x?", b"ab", state, 0.01)
    proposal = propose_blocks(model, b"x?", b"ab", state, observation)
    assert proposal["teacher_count"] == 3
    assert proposal["mean_loss"] == receipt.mean_loss
    assert torch.equal(proposal["actual_mean_context_signal"], receipt.mean_context_signal)
    assert torch.equal(proposal["discarded_output_delta"],
                       reference.decoder.output_weights - model.decoder.output_weights)
    assert proposal["teacher_context_signals"].count_nonzero()
    expected = dict(reference.named_parameters())
    for block, changes in proposal["blocks"].items():
        assert any(change["delta"].count_nonzero() for change in changes.values())
        for name, change in changes.items():
            assert torch.equal(change["after"], expected[name])
        for sign in (-1, 1):
            candidate = perturbed_model(model, proposal, block, sign)
            original = dict(model.named_parameters())
            for name, parameter in candidate.named_parameters():
                wanted = (
                    (changes[name]["after"] if sign == 1 else
                     original[name] - changes[name]["delta"])
                    if name in changes else original[name]
                )
                assert torch.equal(parameter, wanted)
            assert torch.equal(candidate.decoder.output_weights, model.decoder.output_weights)
    _assert_nested_equal(before, model.snapshot(state))


def test_localization_positive_block_uses_actual_after_despite_float_cancellation():
    from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    from experiments.r4_integrated.local_memory_localization_core import perturbed_model

    model = LocalMemoryLearner(
        writer_update_fraction=0.5, input_code_size=2, value_size=94, hidden_size=3, key_size=2
    )
    with torch.no_grad():
        model.memory.strength_bias.fill_(1.0)
    before = model.memory.strength_bias.detach().clone()
    after = torch.full_like(before, 1e-8)
    delta = after - before
    assert not torch.equal(before + delta, after)
    proposal = {"blocks": {"writer": {"memory.strength_bias": {
        "before": before, "after": after, "delta": delta,
    }}}}
    positive = perturbed_model(model, proposal, "writer", 1)
    negative = perturbed_model(model, proposal, "writer", -1)
    assert torch.equal(positive.memory.strength_bias, after)
    assert torch.equal(negative.memory.strength_bias, before - delta)
    assert torch.equal(model.memory.strength_bias, before)


def test_localization_fixed_roster_and_saved_potential_scorer_without_model_calls():
    from collections import Counter

    from experiments.r4_integrated import local_memory_localization_v1 as runner

    roster = runner.roster()
    memories, cases = roster["memories"], roster["cases"]
    assert len(memories) == 16 and len(cases) == 64
    assert sum(runner.CONFIG[key] for key in (
        "baseline_ingestions", "proposal_ingestions", "perturbed_ingestions"
    )) == 56
    assert runner.CONFIG["free_queries"] == 64 and runner.CONFIG["teacher_offers"] == 24
    assert [row["id"] for row in cases] == list(range(64))
    assert all(row["mode"] == "tagged" for row in cases[:8])
    for index in range(4):
        assert memories[index + 4]["document_hex"] == memories[index]["document_hex"][::-1]
        assert memories[index + 12]["document_hex"] == memories[index + 8]["document_hex"][::-1]
    assert Counter(row["baseline_id"] for row in cases[32:]) == dict.fromkeys(range(8), 4)
    assert Counter((row["block"], row["sign"]) for row in cases[32:]) == {
        ("writer", 1): 8, ("writer", -1): 8, ("address", 1): 8, ("address", -1): 8,
    }
    for row in cases[32:]:
        baseline = cases[row["baseline_id"]]
        assert row["query_hex"] == baseline["query_hex"]
        assert row["target_hex"] == baseline["target_hex"]
    scores = torch.zeros(3, 257, dtype=torch.float64)
    scores[0, 97] = scores[1, 98] = scores[2, BYTE_EOS] = 1
    observation = {"phase_scores": scores}
    result = runner.score(observation, b"ab".hex())
    assert result["phase_losses"] == [0, 0, 0] and result["phase_margins"] == [1, 1, 1]
    assert result["mean_loss"] == 0
    scores[0, 99] = 2
    before = scores.clone()
    result = runner.score(observation, b"ab".hex())
    assert result["phase_losses"] == [2, 0, 0]
    assert result["phase_margins"] == [-1, 1, 1]
    assert result["mean_loss"] == 2 / 3
    assert torch.equal(scores, before)



@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "reference_module",
    ["local_memory_query_credit_reference", "local_memory_value_credit_reference",
     "local_memory_multiscale_reference", "local_memory_event_reference",
     "local_memory_event_candidate_reference"],
)
def test_local_credit_reference_matches_frozen_git_source_and_trajectories(dtype, reference_module):
    """The experiment adapter must reproduce the old dependency, not only its caller."""
    import ast
    import hashlib
    import importlib
    import subprocess
    import sys
    import types
    from dataclasses import fields
    from pathlib import Path

    reference = importlib.import_module("experiments.r4_integrated." + reference_module)

    root = Path(__file__).resolve().parents[1]
    texts = {
        name: subprocess.check_output(
            ["git", "show", f"{reference.REFERENCE_REVISION}:src/humanbrain/{name}.py"],
            cwd=root,
        ).decode("utf-8")
        for name in ("plastic_memory", "local_memory_learner")
    }
    assert hashlib.sha256(texts["plastic_memory"].encode()).hexdigest() == (
        reference.REFERENCE_MEMORY_BLOB_SHA256
    )
    assert hashlib.sha256(texts["local_memory_learner"].encode()).hexdigest() == (
        reference.REFERENCE_LEARNER_BLOB_SHA256
    )

    def definitions(text):
        return {n.name: n for n in ast.parse(text).body if isinstance(n, ast.ClassDef)}

    archived = definitions(Path(reference.__file__).read_text())
    old_memory = definitions(texts["plastic_memory"])
    current_memory = definitions((root / "src/humanbrain/plastic_memory.py").read_text())
    assert ast.dump(archived["LocalMemoryLearner"]) == ast.dump(
        definitions(texts["local_memory_learner"])["LocalMemoryLearner"]
    )
    if "LocalMemoryRead" in archived:
        assert ast.dump(archived["LocalMemoryRead"]) == ast.dump(old_memory["LocalMemoryRead"])
    else:
        assert ast.dump(old_memory["LocalMemoryRead"]) == ast.dump(
            current_memory["LocalMemoryRead"]
        )
    old_methods = {
        n.name: n for n in old_memory["PlasticMemory"].body if isinstance(n, ast.FunctionDef)
    }
    new_methods = {
        n.name: n for n in current_memory["PlasticMemory"].body if isinstance(n, ast.FunctionDef)
    }
    overrides = {} if "PlasticMemory" not in archived else {
        n.name: n for n in archived["PlasticMemory"].body if isinstance(n, ast.FunctionDef)
    }
    expected_overrides = (
        {"read_local", "_check_local_read", "restore_local_read"}
        if reference_module == "local_memory_query_credit_reference" else set()
    )
    assert set(overrides) == expected_overrides
    for name, node in old_methods.items():
        if name == "__init__":
            continue  # New generic rate argument; scalar dependency trajectory checked below.
        assert ast.dump(node) == ast.dump(overrides.get(name, new_methods[name])), name
    for name in old_memory.keys() - {"PlasticMemory", "LocalMemoryRead", "StreamingRateCell"}:
        assert ast.dump(old_memory[name]) == ast.dump(current_memory[name]), name
    for node in ast.parse(texts["plastic_memory"]).body:
        if isinstance(node, ast.FunctionDef):
            current = next(
                n for n in ast.parse((root / "src/humanbrain/plastic_memory.py").read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == node.name
            )
            assert ast.dump(node) == ast.dump(current), node.name

    # Loading old source is QA-only. The experiment itself uses its declared archive.
    old_module = types.ModuleType("_query_credit_old_memory_fixture")
    old_learner = types.ModuleType("_query_credit_old_learner_fixture")
    modules = {old_module.__name__: old_module, old_learner.__name__: old_learner}
    with patch.dict(sys.modules, modules):
        exec(compile(texts["plastic_memory"], "<frozen-memory-QA>", "exec"), old_module.__dict__)
        source = texts["local_memory_learner"].replace(
            "from humanbrain.plastic_memory import", f"from {old_module.__name__} import"
        )
        exec(compile(source, "<frozen-learner-QA>", "exec"), old_learner.__dict__)
        config = dict(input_code_size=2, value_size=94, hidden_size=3, key_size=2, seed=733)
        if reference_module in {"local_memory_event_reference",
                                "local_memory_event_candidate_reference"}:
            config = {"seed": 733}  # Actual 64-neuron multiscale byte comparator.
        expected = old_learner.LocalMemoryLearner(**config)
        actual = reference.LocalMemoryLearner(**config)
        for model in (actual, expected):
            _set_upstream_dtype(model, dtype)
        _assert_nested_equal(actual.state_dict(), expected.state_dict())
        with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")):
            for target in (b"aa", b"bc", b"ab"):
                states = [model.ingest([b"x=a", b"y=b"]) for model in (actual, expected)]
                _assert_nested_equal(states[0].snapshot(), states[1].snapshot())
                reads = [
                    model._query(b"x?", state)
                    for model, state in zip((actual, expected), states, strict=True)
                ]
                assert torch.equal(reads[0][0], reads[1][0])
                assert actual.predict(b"x?", states[0], 2) == expected.predict(b"x?", states[1], 2)
                receipts = [
                    model.teach(b"x?", target, state, 0.01)
                    for model, state in zip((actual, expected), states, strict=True)
                ]
                for field in fields(receipts[0]):
                    x, y = (getattr(receipt, field.name) for receipt in receipts)
                    _assert_nested_equal(x.snapshot() if field.name == "state" else x,
                                         y.snapshot() if field.name == "state" else y)
                _assert_nested_equal(actual.state_dict(), expected.state_dict())


def test_shared_temporal_roster_factory_and_criterion():
    from experiments.r4_integrated import local_memory_two_byte_v1 as previous
    from experiments.r4_integrated import local_shared_temporal_query_v1 as runner

    assert runner.ARMS == ("learned", "address_frozen", "slow_frozen",
                           "writes_disabled", "dense_reference")
    train, checks = runner.roster()
    assert (train, checks) == previous.roster()
    assert runner.teaching_schedule(train) == previous.teaching_schedule(train)
    assert len({c["memory_id"] for c in checks}) == 12
    config = runner.SPEC.config
    config["arms"].clear()
    config["loss_kind"].clear()
    assert len(runner.SPEC.config["arms"]) == len(runner.SPEC.config["loss_kind"]) == 5
    manifest = runner.sources()
    for name in ("local_shared_temporal_query_v1.py", "local_shared_temporal_query_v1.md",
                 "local_lag_query_reference.py",
                 "local_memory_event_reference.py", "local_memory_two_byte_v1.py"):
        path = runner.ROOT / "experiments/r4_integrated" / name
        assert manifest[path.relative_to(runner.ROOT).as_posix()] == runner.file_hash(path)
    models = {arm: runner.new_model(runner.SPEC, arm) for arm in runner.ARMS}
    initial = {arm: model.state_dict() for arm, model in models.items()}
    keys = {k for k, v in initial["learned"].items() if torch.is_tensor(v)}
    assert keys == {"input_codes", "address_weight", "codec.weight", "codec.bias"}
    for arm in runner.ARMS[:4]:
        assert {k for k, v in initial[arm].items() if torch.is_tensor(v)} == keys
        for key in keys:
            assert torch.equal(initial[arm][key], initial["learned"][key])
        assert initial[arm]["_extra_state"]["mode"] == arm
    for key in ("input_codes", "codec.weight", "codec.bias"):
        assert torch.equal(initial["learned"][key], initial["dense_reference"][key])
    assert initial["learned"]["address_weight"].shape == (32, 32)
    with pytest.raises(ValueError, match="configuration mismatch"):
        models["learned"].set_extra_state(models["address_frozen"].get_extra_state())
    scores = {arm: dict(fit=0, unseen_value=0, delayed=0, interference=0) for arm in runner.ARMS}
    scores["learned"] = dict(fit=8, unseen_value=4, delayed=4, interference=4)
    assert runner.qualifies(scores)
    scores["dense_reference"] = dict(fit=7, unseen_value=4, delayed=4, interference=2)
    assert runner.qualifies(scores) and runner.matched_control_advantage(scores)
    scores["address_frozen"]["unseen_value"] = 4
    assert not runner.qualifies(scores) and not runner.matched_control_advantage(scores)
    scores["address_frozen"]["unseen_value"] = 0
    scores["dense_reference"] = dict(fit=8, unseen_value=4, delayed=4, interference=4)
    assert not runner.qualifies(scores)
    scores["dense_reference"] = dict(fit=7, unseen_value=4, delayed=4, interference=2)
    scores["learned"]["interference"] = 3
    assert not runner.qualifies(scores)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_explicit_scalar_forward_matches_current_reference_before_teaching(dtype):
    from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    from experiments.r4_integrated.local_memory_multiscale_reference import (
        LocalMemoryLearner as CurrentLearner,
    )

    config = dict(input_code_size=2, value_size=94, hidden_size=3, key_size=2, seed=733)
    repaired, current = (
        LocalMemoryLearner(writer_update_fraction=0.5, **config),
        CurrentLearner(**config),
    )
    for model in (repaired, current):
        _set_upstream_dtype(model, dtype)
    states = [model.ingest([b"ab", b"cd"]) for model in (repaired, current)]
    _assert_nested_equal(states[0].snapshot(), states[1].snapshot())
    weights = [copy.deepcopy(model.state_dict()) for model in (repaired, current)]
    for query in (b"x?", b"y?"):
        reads = [
            model._query(query, state)
            for model, state in zip((repaired, current), states, strict=True)
        ]
        assert torch.equal(reads[0][0], reads[1][0])
        assert repaired.predict(query, states[0], 2) == current.predict(query, states[1], 2)
    for before, model in zip(weights, (repaired, current), strict=True):
        _assert_nested_equal(before, model.state_dict())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["writer_maps_frozen", "frozen", "writes_disabled"])
def test_explicit_scalar_controls_preserve_immediate_predecessor_trajectory(dtype, mode):
    from dataclasses import fields

    from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    from experiments.r4_integrated.local_memory_multiscale_reference import (
        LocalMemoryLearner as PreviousLearner,
    )

    config = dict(mode=mode, input_code_size=2, value_size=94, hidden_size=3, key_size=2, seed=733)
    candidate, previous = (
        LocalMemoryLearner(writer_update_fraction=0.5, **config),
        PreviousLearner(**config),
    )
    for model in (candidate, previous):
        _set_upstream_dtype(model, dtype)

    def common_weights():
        new = {k: v for k, v in candidate.state_dict().items() if torch.is_tensor(v)}
        old = {k: v for k, v in previous.state_dict().items() if torch.is_tensor(v)}
        assert set(old) == set(new)
        for key, value in new.items():
            assert value.dtype == old[key].dtype and torch.equal(value, old[key]), key

    common_weights()
    assert not hasattr(candidate, "value_feedback") and not hasattr(candidate, "query_feedback")
    for target in (b"aa", b"bc", b"ab"):
        states = [model.ingest([b"x=a", b"y=b"]) for model in (candidate, previous)]
        _assert_nested_equal(states[0].snapshot(), states[1].snapshot())
        assert candidate.predict(b"x?", states[0], 2) == previous.predict(b"x?", states[1], 2)
        with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")):
            receipts = [
                model.teach(b"x?", target, state, 0.01)
                for model, state in zip((candidate, previous), states, strict=True)
            ]
        for field in fields(receipts[0]):
            new, old = (getattr(receipt, field.name) for receipt in receipts)
            _assert_nested_equal(
                new.snapshot() if field.name == "state" else new,
                old.snapshot() if field.name == "state" else old,
            )
        common_weights()



def test_binding_fixed_factorial_roster_and_observation_budget():
    from experiments.r4_integrated import local_memory_binding_v1 as runner

    plan = runner.roster()
    assert len(plan["memories"]) == 16 and len(plan["cases"]) == 32
    assert [case["id"] for case in plan["cases"]] == list(range(32))
    assert sum(len(bytes.fromhex(d)) + 1 for m in plan["memories"]
               for d in m["document_hex"]) == 800
    assert sum(len(bytes.fromhex(c["query_hex"])) + 1 for c in plan["cases"]) == 832
    for mode in runner.CONFIG["checkpoint_modes"]:
        entries = [m for m in plan["memories"] if m["model_id"] == mode]
        for assignment in "ABCD":
            normal, reverse = [m for m in entries if m["assignment"] == assignment]
            assert normal["document_hex"] == reverse["document_hex"][::-1]
        first_documents = [bytes.fromhex(m["document_hex"][0]) for m in entries]
        assert len(set(first_documents)) == 8
        for family, keys, values in (("AB", (b"k00", b"k01"), (b"ab", b"bc")),
                                     ("CD", (b"k02", b"k03"), (b"cd", b"da"))):
            documents = {bytes.fromhex(m["document_hex"][0]) for m in entries
                         if m["assignment"] in family}
            assert documents == {b"Record " + k + b" has value " + v + b"."
                                 for k in keys for v in values}
    assert runner.CONFIG["prefix_milestones"] == [9, 20, 21, 22, 24, 49]
    assert runner.CONFIG["expected_derived_prefix_projections"] == 192
    assert runner.CONFIG["teacher_offers"] == runner.CONFIG["actions"] == 0
    assert runner.CONFIG["traced_ingestions"] + runner.CONFIG["ordinary_parity_ingestions"] == 32
    assert runner.CONFIG["output_phase_evaluations"] == 3 * len(plan["cases"])



@pytest.mark.parametrize("mode", ["tagged", "aggregate", "writer_maps_frozen", "frozen",
                                  "writes_disabled"])
def test_event_candidate_controls_keep_encoder_and_writer_rates_fixed(mode):
    from experiments.r4_integrated.local_memory_event_candidate_reference import LocalMemoryLearner

    model = LocalMemoryLearner(mode=mode, seed=733)
    rates = torch.tensor([0.5] * 32 + [0.125] * 16 + [0.03125] * 16)
    assert torch.equal(model.memory.controller.update_fraction, rates)
    assert "update_fraction" not in dict(model.memory.controller.named_parameters())
    before = copy.deepcopy(model.state_dict())
    for target in (b"aa", b"bc"):
        state = model.ingest([b"x=a", b"y=b"])
        repeated = model.ingest([b"x=a", b"y=b"])
        _assert_nested_equal(state.snapshot(), repeated.snapshot())
        snapshot = model.snapshot(state)
        model.predict(b"x?", state, 2)
        _assert_nested_equal(snapshot, model.snapshot(state))
        with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")):
            feedback = model.teach(b"x?", target, state, 0.01)
        assert feedback.teacher_count == 3
        assert torch.equal(model.memory.controller.update_fraction, rates)
        assert model.query_cell.update_fraction == 0.5
        for key, value in before.items():
            if key.startswith("event_encoder."):
                assert torch.equal(value, model.state_dict()[key])
        if mode == "writes_disabled":
            assert not state.synapses.count_nonzero() and not feedback.memory_updated
    if mode == "frozen":
        _assert_nested_equal(before, model.state_dict())
    elif mode == "writer_maps_frozen":
        for key, value in before.items():
            if key.startswith("memory.") and not key.startswith("memory.query_"):
                _assert_nested_equal(value, model.state_dict()[key])
        assert not torch.equal(before["decoder.output_weights"], model.decoder.output_weights)


    else:
        assert not torch.equal(before["decoder.output_weights"], model.decoder.output_weights)


def test_event_input_work_separates_raw_encoder_controller_and_extra_host_counts():
    from experiments.r4_integrated import local_memory_event_v1 as runner

    actual = runner.harness.input_work(runner.SPEC, "tagged", b"ab !", repeats=2)
    assert actual == {
        "raw_bytes": 8, "string_eos": 2, "encoder_steps": 10, "controller_events": 8,
        "encoding_host_segmentations": 2, "encoding_host_bytes": 8,
        "accounting_host_segmentations": 1, "accounting_host_bytes": 4,
    }
    old = runner.harness.input_work(runner.SPEC, "current_tagged", b"ab !", repeats=2)
    assert old == {
        "raw_bytes": 8, "string_eos": 2, "encoder_steps": 0, "controller_events": 10,
        "encoding_host_segmentations": 0, "encoding_host_bytes": 0,
        "accounting_host_segmentations": 0, "accounting_host_bytes": 0,
    }
    assert runner.harness.sum_input_work([actual, old])["controller_events"] == 18


def test_event_complete_lesson_counts_actual_traversals_without_encoder_replay():
    from experiments.r4_integrated import local_memory_event_v1 as runner
    from experiments.r4_integrated.local_memory_event_candidate_reference import LocalMemoryLearner

    model = LocalMemoryLearner(mode="tagged", seed=733)
    documents, query = [b"alpha=ab", b"beta=ba"], b"alpha?"
    calls = []
    encoded = model._encoded

    def record(raw):
        values = encoded(raw)
        calls.append((raw, values.shape[1]))
        return values

    episode = {"document_hex": [raw.hex() for raw in documents],
               "query_hex": query.hex(), "target_hex": b"ab".hex()}
    with patch.object(model, "_encoded", side_effect=record), patch.object(
        torch.autograd, "backward", side_effect=AssertionError("local only")
    ):
        receipt = runner.harness.event_lesson(runner.SPEC, "tagged", model, episode, lambda: None)
    assert calls == [(documents[0], 4), (documents[1], 4), (query, 3), (query, 3)]
    assert receipt["teacher_count"] == 3
    docs, queries = receipt["input_work"]["document"], receipt["input_work"]["query"]
    assert docs["raw_bytes"] == 15 and docs["string_eos"] == 2
    assert docs["encoder_steps"] == 17 and docs["controller_events"] == 8
    assert queries["raw_bytes"] == 12 and queries["string_eos"] == 2
    assert queries["encoder_steps"] == 14 and queries["controller_events"] == 6
    assert queries["accounting_host_segmentations"] == 1
    assert receipt["seconds"] > 0


def test_sequence_small_end_to_end_and_shared_codec_identity_prior():
    from experiments.r4_integrated.local_sequence_memory_reference import (
        ConditionalByteCodec,
        LocalMemoryLearner,
    )

    codec = ConditionalByteCodec()
    sensory = torch.cat((torch.eye(257), torch.zeros(1, 257)))
    evaluated = codec.evaluate(sensory)
    for token in range(257):
        assert float(evaluated.probabilities[token, token]) == pytest.approx(0.5, abs=2e-16)
        other = (token + 1) % 257
        assert float(evaluated.probabilities[token, other]) == pytest.approx(1 / 512, abs=1e-17)
    torch.testing.assert_close(evaluated.probabilities[-1], torch.full((257,), 1 / 257,
                              dtype=torch.float64), rtol=0, atol=1e-17)
    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    snapshot = model.snapshot(state)
    with patch.object(model.codec, "evaluate", wraps=model.codec.evaluate) as passes:
        prediction = model.predict_with_receipt(b"q", state, 2)
    assert passes.call_count == len(prediction.steps) == len(prediction.generation.data) + 1
    assert prediction.work["codec_slot_evaluations"] == state.used * len(prediction.steps)
    assert prediction.work["branch_cursor_initializations"] == state.used
    assert prediction.work["branch_cursor_advances"] == state.used * (len(prediction.steps) - 1)
    _assert_nested_equal(snapshot, model.snapshot(state))
    with patch.object(torch.autograd, "backward", side_effect=AssertionError("local only")):
        feedback = model.teach(b"q", b"ab", state, 0.01)
    assert feedback.teacher_count == 3 and feedback.codec_updated and feedback.address_updated
    assert feedback.loss_kind == "sequence_mixture_negative_log_likelihood"
    assert feedback.total_loss == sum(s.nll for s in feedback.steps)
    assert feedback.mean_loss == feedback.total_loss / 3
    _assert_nested_equal(snapshot["memory"], feedback.state.snapshot())
    assert feedback.state.keys.data_ptr() != state.keys.data_ptr()
    assert all(p.grad is None for p in model.parameters())
    assert not any(hasattr(model, name) for name in ("decoder", "event_encoder", "value_feedback"))


def _sequence_scalar_teacher(weights, bias, sensory_ids, successor, initial, targets):
    """Independent scalar current-prefix filter and local codec contribution oracle."""
    belief, cursors = list(initial), list(range(len(initial)))
    errors, bias_errors, losses, posteriors = {}, [0.0] * 257, [], []
    for target in targets:
        columns = [sensory_ids[cursor] for cursor in cursors]
        probabilities = []
        for column in columns:
            logits = [bias[k] + (weights[k][column] if column is not None else 0)
                      for k in range(257)]
            peak = max(logits)
            values = [math.exp(x - peak) for x in logits]
            total = math.fsum(values)
            probabilities.append([x / total for x in values])
        mixture = math.fsum(b * p[target] for b, p in zip(belief, probabilities, strict=True))
        losses.append(-math.log(mixture))
        belief = [b * p[target] / mixture
                  for b, p in zip(belief, probabilities, strict=True)]
        posteriors.append(list(belief))
        for branch, column in enumerate(columns):
            for output in range(257):
                error = belief[branch] * (probabilities[branch][output] - (output == target))
                bias_errors[output] += error
                if column is not None:
                    key = (output, column)
                    errors[key] = errors.get(key, 0.0) + error
        cursors = [successor[cursor] for cursor in cursors]
    return errors, bias_errors, losses, posteriors


def test_sequence_teacher_current_prefix_scalar_codec_address_and_preupdate_query_credit():
    from experiments.r4_integrated.local_sequence_memory_reference import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    before = copy.deepcopy(model.state_dict())
    event = model._query(b"q", state, learning=True)
    initial = event.log_prior.exp().tolist()
    expected = _sequence_scalar_teacher(
        before["codec.weight"].tolist(), before["codec.bias"].tolist(),
        [97, 98, 46, 256], state.successor[:4].tolist(), initial, [97, 98, 256],
    )
    errors, bias_errors, losses, posteriors = expected
    address_error = [p - b for p, b in zip(initial, posteriors[-1], strict=True)]
    keys = state.keys[:4].tolist()
    h = [math.fsum(error * key[j] for error, key in zip(address_error, keys, strict=True))
         for j in range(1024)]
    query_signal = torch.tensor([[math.fsum(float(w) * error for w, error in zip(row, h,
                                  strict=True)) for row in before["address_weight"].tolist()]],
                                dtype=torch.float32)
    expected_query = copy.deepcopy(model.query_cell)
    expected_query.apply_learning_signal(event.learning_state, query_signal, 0.01)
    seen = []
    evaluate = model.codec.evaluate

    def observe(sensory):
        seen.append(copy.deepcopy(model.state_dict()))
        return evaluate(sensory)

    with patch.object(model.codec, "evaluate", side_effect=observe), patch.object(
        torch.autograd, "backward", side_effect=AssertionError("no reverse history")
    ):
        feedback = model.teach(b"q", b"ab", state, 0.01)
    assert len(seen) == 3
    for saved in seen:
        _assert_nested_equal(before, saved)
    assert [s.nll for s in feedback.steps] == pytest.approx(losses, abs=2e-14)
    for receipt, posterior in zip(feedback.steps, posteriors, strict=True):
        assert receipt.posterior_log_belief.exp().tolist() == pytest.approx(posterior, abs=2e-15)
    assert feedback.address_error.tolist() == pytest.approx(address_error, abs=2e-15)
    torch.testing.assert_close(feedback.query_signal, query_signal, rtol=2e-6, atol=2e-8)
    for (output, column), error in errors.items():
        expected_weight = float(before["codec.weight"][output, column]) - 0.01 * error
        assert float(model.codec.weight[output, column].detach()) == pytest.approx(
            expected_weight, abs=2e-15
        )
    untouched = [i for i in range(257) if i not in {97, 98, 46, 256}]
    assert torch.equal(model.codec.weight[:, untouched], before["codec.weight"][:, untouched])
    for output, error in enumerate(bias_errors):
        assert float(model.codec.bias[output].detach()) == pytest.approx(-0.01 * error, abs=2e-15)
    for i, j in ((0, 0), (9, 31), (31, 64), (4, 99)):
        # Independent scalar stages preserve the declared float32 publication order.
        contribution = torch.tensor(float(event.activity[0, i]) * h[j], dtype=torch.float32)
        change = float(torch.tensor(0.01, dtype=torch.float32) * contribution)
        expected_weight = float(torch.tensor(float(before["address_weight"][i, j]) - change))
        assert float(model.address_weight[i, j].detach()) == expected_weight
    for name, expected_parameter in expected_query.state_dict().items():
        torch.testing.assert_close(model.query_cell.state_dict()[name], expected_parameter,
                                   rtol=2e-6, atol=2e-8)
    # The address objective is the held-codec complete likelihood; perturb only logits.
    logits = event.log_prior.tolist()

    def objective(displacement):
        values = [math.exp(x + (displacement if i == 1 else 0)) for i, x in enumerate(logits)]
        prior = [x / math.fsum(values) for x in values]
        return math.fsum(_sequence_scalar_teacher(
            before["codec.weight"].tolist(), before["codec.bias"].tolist(),
            [97, 98, 46, 256], [1, 2, 3, 3], prior, [97, 98, 256],
        )[2])

    numeric = (objective(1e-5) - objective(-1e-5)) / 2e-5
    assert numeric == pytest.approx(address_error[1], rel=1e-7, abs=2e-9)
    # Identity ab. and an absorbing EOS start differ by a factor256 over ab+EOS,
    # not by a host punctuation skip or a final-posterior codec oracle.
    assert posteriors[-1][0] / posteriors[-1][3] == pytest.approx(initial[0] / initial[3] * 256)
    assert posteriors[1] != pytest.approx(posteriors[-1])


@pytest.mark.parametrize("mode", ["learned", "address_frozen", "slow_frozen", "writes_disabled"])
def test_sequence_controls_read_only_transport_and_teacher_group_counts(mode):
    from experiments.r4_integrated.local_sequence_memory_reference import LocalMemoryLearner

    model = LocalMemoryLearner(mode=mode, seed=733)
    state = model.ingest([b"ab."])
    snapshot = model.snapshot(state)
    with patch.object(model.codec, "evaluate", side_effect=AssertionError("query only")):
        query = model.query_receipt(b"q", state)
    assert query.work["query_controller_steps"] == 2
    assert query.work["codec_slot_evaluations"] == query.work["branch_cursor_initializations"] == 0
    _assert_nested_equal(snapshot, model.snapshot(state))
    clone = LocalMemoryLearner(mode=mode, seed=733)
    restored = clone.restore_snapshot(snapshot)
    assert restored.keys.data_ptr() != snapshot["memory"]["keys"].data_ptr()
    assert clone.predict(b"q", restored, 2) == model.predict(b"q", state, 2)
    feedback = model.teach(b"q", b"ab", state, 0.01)
    assert feedback.teacher_count == 3
    for group in ("codec", "address", "query"):
        assert feedback.work[group + "_update_group_offers"] == 1
    assert feedback.work["enabled_codec_update_groups"] == (mode != "slow_frozen")
    assert feedback.work["enabled_address_update_groups"] == (
        mode in {"learned", "writes_disabled"}
    )
    assert feedback.work["enabled_query_update_groups"] == (mode in {"learned", "writes_disabled"})
    _assert_nested_equal(snapshot["memory"], state.snapshot())
    assert torch.equal(snapshot["model"]["input_codes"], model.input_codes)
    if mode in {"address_frozen", "slow_frozen", "writes_disabled"}:
        for name in ("address_weight", "query_cell.weight", "query_cell.bias",
                     "query_cell.recurrent"):
            assert torch.equal(snapshot["model"][name], model.state_dict()[name])
    if mode == "slow_frozen":
        _assert_nested_equal(snapshot["model"], model.state_dict())
    if mode == "writes_disabled":
        assert not state.keys.count_nonzero() and not state.sensory.count_nonzero()
        assert not feedback.address_updated and not feedback.query_updated
        assert torch.equal(snapshot["model"]["codec.weight"], model.codec.weight)
        assert feedback.codec_updated and model.codec.bias.count_nonzero()


def test_sequence_generic_longer_generation_and_teacher_no_two_slot_limit():
    from experiments.r4_integrated.local_sequence_memory_reference import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"wxyz"])
    event = model.query_receipt(b"q", state)
    prior = torch.tensor([0.0, -100.0, -100.0, -100.0, -100.0], dtype=torch.float64)
    from dataclasses import replace

    # An explicit initial-address QA fixture; all continuation/output cells remain real.
    with patch.object(model, "_query", return_value=replace(event, log_prior=prior)):
        generation = model.predict(b"q", state, 8)
    assert generation == ByteGeneration(b"wxyz", "eos")
    feedback = model.teach(b"q", b"wxyz", state, 0.01, learning_enabled=False)
    assert feedback.teacher_count == 5 and len(feedback.steps) == 5
    assert not any((feedback.codec_updated, feedback.address_updated, feedback.query_updated))
    assert feedback.steps[-1].observed_token == 256


def test_sequence_rejects_invalid_parameters_checkpoint_and_teacher_without_mutation():
    from experiments.r4_integrated.local_sequence_memory_reference import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"a"])
    before = model.snapshot(state)
    for rate in (0.1, float("nan")):
        with pytest.raises(ValueError, match="learning_rate"):
            model.teach(b"q", b"a", state, rate)
    for key, mutation in (("codec.weight", float("nan")), ("input_codes", 0.0)):
        snapshot = copy.deepcopy(before)
        snapshot["model"][key][0, 0] = mutation
        with pytest.raises(ValueError):
            model.restore_snapshot(snapshot)
        _assert_nested_equal(before, model.snapshot(state))
    bad_state = copy.deepcopy(state)
    bad_state.keys[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.predict(b"q", bad_state, 2)
    _assert_nested_equal(before, model.snapshot(state))


@pytest.mark.parametrize("arm", ["learned", "address_frozen", "slow_frozen",
                                "writes_disabled", "byte_reference"])
def test_sequence_complete_lesson_and_full_capacity_cap_accounting(arm):
    from experiments.r4_integrated import local_sequence_memory_v1 as runner

    if arm == "byte_reference":
        from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
        model = LocalMemoryLearner(mode="tagged", seed=733)
    else:
        from experiments.r4_integrated.local_sequence_memory_reference import LocalMemoryLearner
        model = LocalMemoryLearner(mode=arm, seed=733)
    episode = {"document_hex": [b"ab.".hex()], "query_hex": b"q".hex(), "target_hex": b"ab".hex()}
    row = runner.lesson(runner.SPEC, arm, model, episode, lambda: None)
    assert row["teacher_count"] == 3
    work = row["work"]
    assert work["document_raw_bytes"] == 3 and work["document_eos_symbols"] == 1
    assert work["query_controller_steps"] == 4 and work["query_eligibility_steps"] == 2
    assert work["tape_key_normalizations"] == (4 if arm in runner.ARMS[:3] else 0)
    if arm != "byte_reference":
        assert work["codec_slot_evaluations"] == 4 * (row["decisions"] + 3)
        assert len(row["teacher_steps"]) == 3
        assert work["branch_cursor_initializations"] == 8
    before = copy.deepcopy(model.state_dict())
    cap = runner.capped_prediction(runner.SPEC, arm, model)
    assert cap["generated_hex"] == "0000" and cap["decisions"] == 3
    assert cap["ingestion_work"]["document_raw_bytes"] == 127
    assert cap["ingestion_work"]["document_eos_symbols"] == 1
    assert cap["original_weights_restored"]
    if arm != "byte_reference":
        assert cap["occupied_slots"] == 128 and cap["work"]["codec_slot_evaluations"] == 384
    _assert_nested_equal(before, model.state_dict())


@pytest.mark.parametrize("query", [b"", b"ab", bytes(range(40))])
def test_lag_query_normalization_order_truncation_and_read_only_counts(query):
    from experiments.r4_integrated.local_lag_query_reference import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    before = model.snapshot(state)
    event = model.query_receipt(query, state)
    ids = list(reversed([*query, BYTE_EOS][-32:]))
    values = [float(value) for token in ids for value in model.input_codes[token]]
    values += [0.0] * (1024 - len(values))
    norm = math.sqrt(math.fsum(value * value for value in values))
    expected = torch.tensor([value * math.sqrt(32) / max(norm, 1e-8) for value in values])
    torch.testing.assert_close(event.activity[0], expected, rtol=2e-6, atol=2e-7)
    assert float(event.activity.square().sum()) == pytest.approx(32, abs=1e-5)
    assert event.work["query_lag_shift_steps"] == len(query) + 1
    assert event.work["query_population_normalizations"] == 1
    assert event.work["address_projection_synapse_evaluations"] == 1024 ** 2
    assert event.work["codec_slot_evaluations"] == 0
    assert not any("eligibility" in key or "query_update" in key for key in event.work)
    _assert_nested_equal(before, model.snapshot(state))
    assert not hasattr(model, "query_cell") and not hasattr(event, "learning_state")


def test_lag_local_address_scalar_update_held_codec_and_retired_query_credit():
    from experiments.r4_integrated.local_lag_query_reference import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    before = copy.deepcopy(model.state_dict())
    event = model.query_receipt(b"q", state)
    initial = event.log_prior.exp().tolist()
    errors, bias_errors, losses, posteriors = _sequence_scalar_teacher(
        before["codec.weight"].tolist(), before["codec.bias"].tolist(),
        [97, 98, 46, 256], [1, 2, 3, 3], initial, [97, 98, 256],
    )
    expected_error = [p - b for p, b in zip(initial, posteriors[-1], strict=True)]
    keys = state.keys[:4].tolist()
    h = [math.fsum(error * key[j] for error, key in zip(expected_error, keys, strict=True))
         for j in range(1024)]
    evaluate = model.codec.evaluate
    calls = []

    def held(sensory):
        _assert_nested_equal(before, model.state_dict())
        calls.append(len(sensory))
        return evaluate(sensory)

    with patch.object(model.codec, "evaluate", side_effect=held), patch.object(
        torch.autograd, "backward", side_effect=AssertionError("local only")
    ):
        receipt = model.teach(b"q", b"ab", state, 0.01)
    assert calls == [4, 4, 4]
    assert [step.nll for step in receipt.steps] == pytest.approx(losses, abs=2e-14)
    assert receipt.address_error.tolist() == pytest.approx(expected_error, abs=2e-15)
    for i, j in ((0, 0), (9, 31), (31, 64), (40, 65)):
        contribution = torch.tensor(float(event.activity[0, i]) * h[j], dtype=torch.float32)
        delta = float(torch.tensor(0.01, dtype=torch.float32) * contribution)
        expected = float(torch.tensor(float(before["address_weight"][i, j]) - delta))
        assert float(model.address_weight[i, j].detach()) == expected
    assert torch.equal(model.address_weight[64:], before["address_weight"][64:])
    assert torch.equal(model.address_weight[:, 96:], before["address_weight"][:, 96:])
    for (output, column), error in errors.items():
        assert float(model.codec.weight[output, column].detach()) == pytest.approx(
            float(before["codec.weight"][output, column]) - 0.01 * error, abs=2e-15
        )
    for output, error in enumerate(bias_errors):
        assert float(model.codec.bias[output].detach()) == pytest.approx(-0.01 * error, abs=2e-15)
    assert receipt.address_updated and receipt.codec_updated
    assert not hasattr(receipt, "query_signal") and not hasattr(receipt, "query_updated")
    assert receipt.work["teacher_offers"] == 3
    assert receipt.work["address_local_synapse_contributions"] == 1024 ** 2
    assert not any("query_update" in key for key in receipt.work)
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("mode", ["learned", "address_frozen", "slow_frozen", "writes_disabled"])
def test_lag_controls_and_source_free_snapshot_preserve_real_groups(mode):
    from experiments.r4_integrated.local_lag_query_reference import LocalMemoryLearner

    model = LocalMemoryLearner(mode=mode, seed=733)
    state = model.ingest([b"ab."])
    before = model.snapshot(state)
    first = model.predict(b"q", state, 2)
    _assert_nested_equal(before, model.snapshot(state))
    clone = LocalMemoryLearner(mode=mode, seed=733)
    restored = clone.restore_snapshot(before)
    assert restored.sensory.data_ptr() != state.sensory.data_ptr()
    assert clone.predict(b"q", restored, 2) == first
    feedback = model.teach(b"q", b"ab", state, 0.01)
    assert feedback.teacher_count == 3
    assert feedback.work["enabled_codec_update_groups"] == (mode != "slow_frozen")
    assert feedback.work["enabled_address_update_groups"] == (
        mode in {"learned", "writes_disabled"}
    )
    _assert_nested_equal(before["memory"], feedback.state.snapshot())
    assert torch.equal(before["model"]["input_codes"], model.input_codes)
    if mode != "learned":
        assert torch.equal(before["model"]["address_weight"], model.address_weight)
    if mode == "slow_frozen":
        _assert_nested_equal(before["model"], model.state_dict())
    if mode == "writes_disabled":
        assert not state.sensory.count_nonzero() and not feedback.address_updated
        assert torch.equal(before["model"]["codec.weight"], model.codec.weight)


def test_lag_finite_normalization_and_fixed_metadata_reject_without_publication():
    from experiments.r4_integrated.local_lag_query_reference import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"a"])
    model.input_codes.fill_(1e30)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="query lag norm"):
        model.teach(b"q", b"a", state, 0.01)
    _assert_nested_equal(before, model.state_dict())
    model.input_codes.zero_()
    event = model.query_receipt(b"q", state)
    assert not event.activity.count_nonzero()
    assert torch.isfinite(event.log_prior).all()
    bad = copy.deepcopy(model.get_extra_state())
    bad["query_size"] = 32
    with pytest.raises(ValueError, match="configuration mismatch"):
        model.set_extra_state(bad)


@pytest.mark.parametrize(
    "reference_module", ["local_sequence_memory_reference", "local_lag_query_reference"]
)
def test_lag_sequence_archive_exact_source_and_three_lesson_trajectory(reference_module):
    import ast
    import hashlib
    import importlib
    import subprocess
    import sys
    import types
    from pathlib import Path

    reference = importlib.import_module("experiments.r4_integrated." + reference_module)
    from experiments.r4_integrated.local_lag_query_v1 import plain

    root = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output([
        "git", "show", reference.REFERENCE_REVISION + ":src/humanbrain/local_memory_learner.py"
    ], cwd=root)
    assert hashlib.sha256(raw).hexdigest() == reference.REFERENCE_LEARNER_BLOB_SHA256
    original = ast.parse(raw)
    archived = ast.parse(Path(reference.__file__).read_bytes())
    assert all(ast.dump(a) == ast.dump(b)
               for a, b in zip(original.body, archived.body[:len(original.body)], strict=True))
    tape = subprocess.check_output([
        "git", "show", reference.REFERENCE_REVISION + ":src/humanbrain/neuronal_sequence_memory.py"
    ], cwd=root)
    assert hashlib.sha256(tape).hexdigest() == reference.REFERENCE_TAPE_BLOB_SHA256
    assert ast.dump(ast.parse(tape)) == ast.dump(ast.parse(
        (root / "src/humanbrain/neuronal_sequence_memory.py").read_bytes()
    ))
    old = types.ModuleType("_lag_query_sequence_source_fixture")
    with patch.dict(sys.modules, {old.__name__: old}):
        exec(compile(raw, "<frozen-sequence-QA>", "exec"), old.__dict__)
        models = [reference.LocalMemoryLearner(seed=733), old.LocalMemoryLearner(seed=733)]
        _assert_nested_equal(models[0].state_dict(), models[1].state_dict())
        for target in (b"ab", b"bc", b"aa"):
            states = [model.ingest([b"x=ab.", b"y=bc."]) for model in models]
            _assert_nested_equal(states[0].snapshot(), states[1].snapshot())
            predictions = [model.predict_with_receipt(b"x?", state, 2)
                           for model, state in zip(models, states, strict=True)]
            assert plain(predictions[0]) == plain(predictions[1])
            teachers = [model.teach(b"x?", target, state, 0.01)
                        for model, state in zip(models, states, strict=True)]
            assert plain(teachers[0]) == plain(teachers[1])
            _assert_nested_equal(models[0].state_dict(), models[1].state_dict())


@pytest.mark.parametrize("arm", ["learned", "address_frozen", "slow_frozen",
                                "writes_disabled", "sequence_reference", "byte_reference"])
def test_lag_complete_lesson_and_capacity_cap_accounting(arm):
    from experiments.r4_integrated import local_lag_query_v1 as runner

    if arm in runner.ARMS[:4]:
        from experiments.r4_integrated.local_lag_query_reference import LocalMemoryLearner
        model = LocalMemoryLearner(mode=arm, seed=733)
    else:
        model = runner.new_model(runner.SPEC, arm)
    episode = {"document_hex": [b"ab.".hex()], "query_hex": b"q".hex(), "target_hex": b"ab".hex()}
    row = runner.lesson(runner.SPEC, arm, model, episode, lambda: None)
    assert row["teacher_count"] == 3 and row["work"]["document_raw_bytes"] == 3
    work = row["work"]
    if arm in runner.ARMS[:4]:
        assert work["query_lag_shift_steps"] == 4
        assert work["query_population_normalizations"] == 2
        assert work["address_projection_synapse_evaluations"] == 2 * 1024 ** 2
        assert work["address_local_synapse_contributions"] == 1024 ** 2
        assert "teacher_query_signal" not in row and "query_updated" not in row
    else:
        assert work["query_controller_steps"] == 4
        assert work["query_eligibility_steps"] == 2
    assert work["tape_key_normalizations"] == (4 if arm not in {
        "writes_disabled", "byte_reference"} else 0)
    before = copy.deepcopy(model.state_dict())
    cap = runner.capped_prediction(runner.SPEC, arm, model)
    assert cap["generated_hex"] == "0000" and cap["decisions"] == 3
    assert cap["ingestion_work"]["document_raw_bytes"] == 127
    assert cap["ingestion_work"]["document_eos_symbols"] == 1
    assert cap["original_weights_restored"]
    if arm != "byte_reference":
        assert cap["occupied_slots"] == 128
        assert cap["work"]["codec_slot_evaluations"] == 384
    _assert_nested_equal(before, model.state_dict())


def test_lag_journal_complete_rows_bounds_and_pending_audit_projection(tmp_path):
    import hashlib
    import json

    from experiments.r4_integrated import local_lag_query_v1 as runner

    journal = runner.LessonJournal(tmp_path / "lessons.jsonl", runner.SPEC)
    expected = []
    for index in range(4):
        for arm in runner.ARMS:
            row = dict(arm=arm, index=index, seconds=0.01 * (index + 1),
                       first_exact=False, decisions=3, teacher_count=3, work={"x": 2},
                       codec_group_enabled=arm != "slow_frozen", codec_updated=False,
                       address_group_enabled=arm in {"learned", "writes_disabled",
                                                     "sequence_reference"},
                       address_updated=False, query_group_enabled=True, query_updated=False,
                       decoder_changed_steps=0, memory_updated=False,
                       complete_nested_receipt={"values": [1.0, None, "\u00e9"]})
            expected.append(copy.deepcopy(row))
            journal.append(row)
            row["complete_nested_receipt"]["values"].clear()
    receipt = journal.close()
    raw = (tmp_path / "lessons.jsonl").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert [json.loads(line) for line in raw.splitlines()] == expected
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["rows"] == 24 and receipt["bytes"] == len(raw)
    assert journal.counts["teacher_byte_offers"] == 72
    assert journal.counts["enabled_teacher_offers"] == 60
    assert not hasattr(journal, "lessons")
    assert all(summary["rows"] == 4 and summary["work"] == {"x": 8}
               for summary in journal.summary.values())
    rejected = runner.LessonJournal(tmp_path / "rejected.jsonl", runner.SPEC)
    try:
        with pytest.raises(MemoryError, match="host-tree"):
            rejected.append({"too_large": "x" * 262144})
        with pytest.raises(MemoryError, match="line limit"):
            rejected.append({"escaped": "\u00ff" * 50000})
        assert rejected.rows == 0 and rejected.bytes == 0
    finally:
        rejected.close()
    report = dict(status="completed", mode="cost", config=runner.SPEC.config,
                  duration_clock=runner.duration_clock(), counts=journal.counts,
                  actions=[{}] * 120, motor_demonstrations=[{}, {}],
                  lesson_summary=journal.summary, lesson_log=receipt,
                  cap_generation={arm: dict(stop_reason="limit", decisions=3,
                      generated_hex="0000", seconds=0.001) for arm in runner.ARMS},
                  memory={"peak_working_set_mib": 100},
                  worker={"memory": {"peak_working_set_mib": 80}})
    launch = dict(elapsed_seconds=10, observed_tree_peak_sum_mib=150)
    pending = runner.project_cost(runner.SPEC, report, launch)
    assert pending["audit_peak_pending"] and not pending["admissible"]
    assert pending["receipt_memory_projection"]["run_projected_peak_mib"] == 213.5
    audit = dict(status="passed", exit_code=0, elapsed_seconds=1,
                 observed_tree_peak_sum_mib=200)
    final = runner.project_cost(runner.SPEC, report, launch, audit_launch=audit)
    assert not final["audit_peak_pending"] and final["admissible"]
    assert final["peak_upper_mib"] == 233.5 + 2 * 312831236 / 2**20
    assert final["projected_learning_seconds"] == pytest.approx(2 * 1024 * sum(
        0.04 + item["max_journal_seconds"] + 0.001 for item in journal.summary.values()
    ))


def test_shared_temporal_small_end_to_end_and_snapshot():
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    before = model.snapshot(state)
    first = model.predict_with_receipt(b"q", state, 2)
    _assert_nested_equal(before, model.snapshot(state))
    receipt = model.teach(b"q", b"ab", state, 0.01)
    assert receipt.teacher_count == len(receipt.steps) == 3
    assert model.address_weight.shape == (32, 32)
    assert receipt.address_updated and receipt.codec_updated
    assert first.work["address_projection_channel_applications"] == 32
    assert receipt.work["address_local_synapse_contributions"] == 32768
    assert receipt.work["address_local_channel_reductions"] == 31744
    _assert_nested_equal(before["memory"], state.snapshot())
    clone = LocalMemoryLearner(seed=733)
    restored = clone.restore_snapshot(before)
    assert clone.predict(b"q", restored, 2) == first.generation


def test_shared_temporal_multichannel_scalar_forward_and_local_update():
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    before = copy.deepcopy(model.state_dict())
    event = model.query_receipt(b"q", state)
    z = event.activity.reshape(32, 32).tolist()
    b = before["address_weight"].tolist()
    expected_q = torch.tensor([
        math.fsum(b[i][j] * z[i][d] for i in range(32))
        for j in range(32) for d in range(32)
    ], dtype=torch.float32)
    torch.testing.assert_close(event.projected_query, expected_q, rtol=2e-6, atol=2e-7)
    initial = event.log_prior.exp().tolist()
    errors, bias_errors, losses, posteriors = _sequence_scalar_teacher(
        before["codec.weight"].tolist(), before["codec.bias"].tolist(),
        [97, 98, 46, 256], [1, 2, 3, 3], initial, [97, 98, 256],
    )
    e = [a - b for a, b in zip(initial, posteriors[-1], strict=True)]
    keys = state.keys[:4].tolist()
    h = [[math.fsum(e[s] * keys[s][j * 32 + d] for s in range(4))
          for d in range(32)] for j in range(32)]
    original_evaluate = model.codec.evaluate
    calls = []

    def held(sensory):
        _assert_nested_equal(before, model.state_dict())
        calls.append(len(sensory))
        return original_evaluate(sensory)

    with patch.object(model.codec, "evaluate", side_effect=held), patch.object(
        torch.autograd, "backward", side_effect=AssertionError("local only")
    ):
        receipt = model.teach(b"q", b"ab", state, 0.01)
    assert calls == [4, 4, 4]
    assert [step.nll for step in receipt.steps] == pytest.approx(losses, abs=2e-14)
    assert receipt.address_error.tolist() == pytest.approx(e, abs=2e-15)
    for i, j in ((0, 0), (0, 1), (1, 1), (1, 2)):
        summed = math.fsum(z[i][d] * h[j][d] for d in range(32))
        contribution = torch.tensor(summed, dtype=torch.float32)
        delta = float(torch.tensor(0.01, dtype=torch.float32) * contribution)
        expected = float(torch.tensor(b[i][j] - delta, dtype=torch.float32))
        assert float(model.address_weight[i, j].detach()) == expected
    assert torch.equal(model.address_weight[2:], before["address_weight"][2:])
    assert torch.equal(model.address_weight[:, 3:], before["address_weight"][:, 3:])
    for (output, column), error in errors.items():
        assert float(model.codec.weight[output, column].detach()) == pytest.approx(
            float(before["codec.weight"][output, column]) - 0.01 * error, abs=2e-15
        )
    for output, error in enumerate(bias_errors):
        assert float(model.codec.bias[output].detach()) == pytest.approx(-0.01 * error, abs=2e-15)
    assert receipt.work["address_projection_channel_applications"] == 32
    assert receipt.work["address_projection_synapse_evaluations"] == 32768
    assert receipt.work["address_local_synapse_contributions"] == 32768
    assert receipt.work["address_local_channel_reductions"] == 31744
    assert all(parameter.grad is None for parameter in model.parameters())


def test_shared_temporal_channel_permutation_equivariance():
    from humanbrain.local_memory_learner import LocalMemoryLearner

    original, permuted = LocalMemoryLearner(seed=733), LocalMemoryLearner(seed=733)
    order = torch.arange(31, -1, -1)
    permuted.input_codes.copy_(original.input_codes[:, order])
    states = [model.ingest([b"ab.", b"cd."]) for model in (original, permuted)]
    events = [model.query_receipt(b"qa?", state)
              for model, state in zip((original, permuted), states, strict=True)]
    torch.testing.assert_close(events[1].activity.reshape(32, 32),
                               events[0].activity.reshape(32, 32)[:, order])
    torch.testing.assert_close(events[1].projected_query.reshape(32, 32),
                               events[0].projected_query.reshape(32, 32)[:, order],
                               rtol=2e-6, atol=2e-7)
    torch.testing.assert_close(events[1].log_prior, events[0].log_prior, rtol=2e-6, atol=2e-7)
    assert torch.equal(original.address_weight, permuted.address_weight)


@pytest.mark.parametrize("mode", ["learned", "address_frozen", "slow_frozen", "writes_disabled"])
def test_shared_temporal_control_publications_and_counts(mode):
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(mode=mode, seed=733)
    state = model.ingest([b"ab."])
    before = model.snapshot(state)
    receipt = model.teach(b"q", b"ab", state, 0.01)
    _assert_nested_equal(before["memory"], state.snapshot())
    assert receipt.work["codec_update_group_offers"] == 1
    assert receipt.work["address_update_group_offers"] == 1
    assert receipt.work["address_local_synapse_contributions"] == 32768
    assert receipt.work["address_local_channel_reductions"] == 31744
    assert receipt.work["enabled_codec_update_groups"] == (mode != "slow_frozen")
    assert receipt.work["enabled_address_update_groups"] == (mode in {"learned", "writes_disabled"})
    assert receipt.address_updated == (mode == "learned")
    assert receipt.codec_updated == (mode != "slow_frozen")
    assert torch.equal(before["model"]["input_codes"], model.input_codes)
    if mode != "learned":
        assert torch.equal(before["model"]["address_weight"], model.address_weight)
    if mode == "slow_frozen":
        _assert_nested_equal(before["model"], model.state_dict())
    if mode == "writes_disabled":
        assert not state.keys.count_nonzero() and not state.sensory.count_nonzero()
        assert torch.equal(before["model"]["codec.weight"], model.codec.weight)


def test_shared_temporal_invalid_normalization_and_dense_checkpoint_rejected():
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"a"])
    model.input_codes.fill_(1e30)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="query lag norm"):
        model.teach(b"q", b"a", state, 0.01)
    _assert_nested_equal(before, model.state_dict())
    model.input_codes.zero_()
    event = model.query_receipt(b"q", state)
    assert not event.activity.count_nonzero() and torch.isfinite(event.log_prior).all()
    before = copy.deepcopy(model.state_dict())
    wrong = copy.deepcopy(before)
    wrong["address_weight"] = torch.zeros(1024, 1024)
    with pytest.raises(ValueError, match="tensor contract"):
        model.load_state_dict(wrong)
    _assert_nested_equal(before, model.state_dict())
    invalid = copy.deepcopy(before)
    invalid["address_weight"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.load_state_dict(invalid)
    _assert_nested_equal(before, model.state_dict())


def test_shared_temporal_journal_and_completed_audit_projection(tmp_path):
    import hashlib
    import json

    from experiments.r4_integrated import local_shared_temporal_query_v1 as runner

    journal = runner.LessonJournal(tmp_path / "lessons.jsonl", runner.SPEC)
    expected = []
    for index in range(4):
        for arm in runner.ARMS:
            row = dict(arm=arm, index=index, seconds=0.01 * (index + 1),
                       first_exact=False, decisions=3, teacher_count=3, work={"x": 2},
                       codec_group_enabled=arm != "slow_frozen", codec_updated=False,
                       address_group_enabled=arm in {"learned", "writes_disabled",
                                                     "dense_reference"},
                       address_updated=False, query_group_enabled=True, query_updated=False,
                       decoder_changed_steps=0, memory_updated=False,
                       complete_nested_receipt={"values": [1.0, None, "\u00e9"]})
            expected.append(copy.deepcopy(row))
            journal.append(row)
            row["complete_nested_receipt"]["values"].clear()
    receipt = journal.close()
    raw = (tmp_path / "lessons.jsonl").read_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert [json.loads(line) for line in raw.splitlines()] == expected
    assert receipt["sha256"] == hashlib.sha256(raw).hexdigest()
    assert receipt["rows"] == 20 and receipt["bytes"] == len(raw)
    assert journal.counts["teacher_byte_offers"] == 60
    assert journal.counts["enabled_teacher_offers"] == 48
    assert not hasattr(journal, "lessons")
    assert all(summary["rows"] == 4 and summary["work"] == {"x": 8}
               for summary in journal.summary.values())
    rejected = runner.LessonJournal(tmp_path / "rejected.jsonl", runner.SPEC)
    try:
        with pytest.raises(MemoryError, match="host-tree"):
            rejected.append({"too_large": "x" * 262144})
        with pytest.raises(MemoryError, match="line limit"):
            rejected.append({"escaped": "\u00ff" * 50000})
        assert rejected.rows == 0 and rejected.bytes == 0
    finally:
        rejected.close()
    report = dict(status="completed", mode="cost", config=runner.SPEC.config,
                  duration_clock=runner.duration_clock(), counts=journal.counts,
                  actions=[{}] * 100, motor_demonstrations=[{}, {}],
                  lesson_summary=journal.summary, lesson_log=receipt,
                  cap_generation={arm: dict(stop_reason="limit", decisions=3,
                      generated_hex="0000", seconds=0.001) for arm in runner.ARMS},
                  memory={"peak_working_set_mib": 100},
                  worker={"memory": {"peak_working_set_mib": 80}})
    launch = dict(elapsed_seconds=10, observed_tree_peak_sum_mib=150)
    pending = runner.project_cost(runner.SPEC, report, launch)
    assert pending["audit_peak_pending"] and not pending["admissible"]
    assert pending["receipt_memory_projection"]["run_projected_peak_mib"] == 311
    audit = dict(status="passed", exit_code=0, elapsed_seconds=1,
                 observed_tree_peak_sum_mib=200)
    final = runner.project_cost(runner.SPEC, report, launch, audit_launch=audit)
    assert not final["audit_peak_pending"] and final["admissible"]
    assert final["peak_upper_mib"] == 235 + 2 * 105236476 / 2**20
    assert final["projected_learning_seconds"] == pytest.approx(2 * 1024 * sum(
        0.04 + item["max_journal_seconds"] + 0.001 for item in journal.summary.values()
    ))


@pytest.mark.parametrize("arm", ["learned", "address_frozen", "slow_frozen",
                                "writes_disabled", "dense_reference"])
def test_shared_temporal_actual_lesson_and_full_cap_counts(arm):
    from experiments.r4_integrated import local_shared_temporal_query_v1 as runner

    model = runner.new_model(runner.SPEC, arm)
    episode = {"document_hex": [b"ab.".hex()], "query_hex": b"q".hex(), "target_hex": b"ab".hex()}
    row = runner.lesson(runner.SPEC, arm, model, episode, lambda: None)
    work = row["work"]
    assert row["teacher_count"] == 3 and work["query_lag_shift_steps"] == 4
    assert work["tape_key_normalizations"] == (0 if arm == "writes_disabled" else 4)
    assert work["branch_cursor_initializations"] == 8
    assert work["codec_slot_evaluations"] == 4 * (row["decisions"] + 3)
    if arm != "dense_reference":
        assert work["address_projection_channel_applications"] == 64
        assert work["address_projection_synapse_evaluations"] == 65536
        assert work["address_local_synapse_contributions"] == 32768
        assert work["address_local_channel_reductions"] == 31744
    else:
        assert work["address_projection_synapse_evaluations"] == 2 * 1024 ** 2
        assert work["address_local_synapse_contributions"] == 1024 ** 2
        assert "address_projection_channel_applications" not in work
    before = copy.deepcopy(model.state_dict())
    cap = runner.capped_prediction(runner.SPEC, arm, model)
    assert cap["generated_hex"] == "0000" and cap["decisions"] == 3
    assert cap["occupied_slots"] == 128
    assert cap["work"]["codec_slot_evaluations"] == 384
    assert cap["ingestion_work"]["document_raw_bytes"] == 127
    assert cap["ingestion_work"]["document_eos_symbols"] == 1
    assert cap["original_weights_restored"]
    _assert_nested_equal(before, model.state_dict())


def test_joint_online_same_instance_action_before_teacher(tmp_path):
    """One authored actual action, then three corrections on that same learner."""
    import hashlib
    import json
    import os

    from experiments.r4_integrated import local_joint_address_online_v1 as runner

    model = runner.new_model(runner.SPEC, "learned")
    before = copy.deepcopy(model.state_dict())
    motor, demos = runner.prepare_motor(tmp_path, lambda: None)
    assert len(demos) == 2 and all(item["passed"] for item in demos)
    motor_hash = motor.checkpoint_sha256
    actions = runner.JsonlJournal(tmp_path / "acquisition_actions.jsonl", runner.SPEC)
    lessons = runner.LessonJournal(tmp_path / "lessons.jsonl", runner.SPEC)
    episode = {"document_hex": [b"ab.".hex()], "query_hex": b"q".hex(),
               "target_hex": b"ab".hex()}
    actual_teach = model.teach
    observed = []

    def checked_teach(query, target, state, rate):
        action = json.loads(actions.path.read_bytes())
        assert action["model_instance_token"] == f"{os.getpid()}:{id(model)}"
        assert action["memory_sha256"] == runner.digest(state.snapshot())
        assert action["before_sha256"] == runner.digest(model.state_dict())
        assert action["execution"]["returncode"] == 0
        observed.append((query, target, id(state)))
        return actual_teach(query, target, state, rate)

    try:
        with patch.object(model, "teach", side_effect=checked_teach):
            row = runner.lesson(runner.SPEC, "learned", model, episode, lambda: None,
                index=0, motor=motor, action_root=tmp_path / "actions", action_journal=actions)
        assert len(observed) == 1 and observed[0][:2] == (b"q", b"ab")
        assert row["teacher_count"] == 3
        assert (row["action_completed_perf"] <= row["teacher_started_perf"]
                <= row["teacher_completed_perf"])
        assert row["before_sha256"] == row["teacher_before_sha256"] == runner.digest(before)
        assert row["memory_sha256"] == row["teacher_memory_sha256"]
        assert row["action_row_sha256"] == hashlib.sha256(actions.path.read_bytes()).hexdigest()
        assert row["work"]["address_feature_slots"] == 8  # four slots, free and teacher queries
        assert row["work"]["address_feature_products"] == 8 * 32768
        assert row["work"]["address_local_synapse_contributions"] == 4 * 1024
        assert row["work"]["address_local_slot_reductions"] == 3 * 1024
        assert row["codec_updated"] and row["address_updated"]
        assert row["action_seconds"] > 0 and row["model_seconds"] > 0
        lessons.append(row)
        del row
        saved = json.loads(lessons.path.read_bytes())
        action = json.loads(actions.path.read_bytes())
        assert saved["grade"] == action["grade"]
        assert saved["first_exact"] == action["grade"]["model_correct"]
        if not action["emitted"]:
            assert action["grade"]["pending_fallback"] and not action["grade"]["task_success"]
        assert motor.checkpoint_sha256 == motor_hash
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in model.parameters())
        assert lessons.counts["first_attempts"] == lessons.counts["acquisition_actions"] == 1
        assert lessons.counts["teacher_byte_offers"] == 3
    finally:
        lessons.close()
        actions.close()


def test_joint_address_scalar_forward_and_local_output_synapses():
    """One current learner: scalar forward features and own-synapse teacher products."""
    import math

    import numpy as np

    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    before_state = state.snapshot()
    before_b = model.address_weight.numpy().copy()
    before_w = model.codec.weight.numpy().copy()
    before_bias = model.codec.bias.numpy().copy()
    query = model.query_receipt(b"q", state)
    z = query.activity.numpy().reshape(32, 32)
    keys = state.keys[:state.used].numpy().reshape(state.used, 32, 32)
    expected_x = np.zeros((state.used, 32, 32), dtype=np.float64)
    for slot in range(state.used):
        for i in range(32):
            for j in range(32):
                expected_x[slot, i, j] = sum(
                    float(z[i, channel]) * float(keys[slot, j, channel])
                    for channel in range(32)
                )
    actual_x = query.address_features.numpy()
    np.testing.assert_allclose(actual_x, expected_x, rtol=2e-6, atol=2e-7)
    expected_scores = np.array([
        sum(float(actual_x[slot, i, j]) * float(before_b[i, j])
            for i in range(32) for j in range(32))
        for slot in range(state.used)
    ])
    np.testing.assert_allclose(query.address_scores.numpy(), expected_scores,
                               rtol=2e-6, atol=2e-7)

    actual_query, actual_codec = model._query, model.codec.evaluate
    teacher_queries, held_codec_calls = [], []

    def capture_query(*args):
        result = actual_query(*args)
        teacher_queries.append(result)
        return result

    def checked_codec(sensory):
        np.testing.assert_array_equal(model.address_weight.numpy(), before_b)
        np.testing.assert_array_equal(model.codec.weight.numpy(), before_w)
        np.testing.assert_array_equal(model.codec.bias.numpy(), before_bias)
        held_codec_calls.append(sensory.shape[0])
        return actual_codec(sensory)

    with patch.object(model, "_query", side_effect=capture_query), \
            patch.object(model.codec, "evaluate", side_effect=checked_codec):
        feedback = model.teach(b"q", b"ab", state, 0.01)
    assert len(teacher_queries) == 1 and held_codec_calls == [4]
    np.testing.assert_array_equal(teacher_queries[0].address_features.numpy(), actual_x)

    # Output-neuron activity differences and actual cached preactivities only.
    # No transpose of keys, hidden-neuron error or older learner is an oracle.
    initial = feedback.initial_log_belief.numpy()
    final = feedback.final_log_belief.numpy()
    output_error = np.array([
        math.exp(p) - math.exp(b) for p, b in zip(initial, final, strict=True)
    ])
    np.testing.assert_allclose(feedback.address_error.numpy(), output_error,
                               rtol=2e-14, atol=2e-15)
    local_b = np.zeros((32, 32), dtype=np.float64)
    for i in range(32):
        for j in range(32):
            local_b[i, j] = sum(float(actual_x[slot, i, j]) * output_error[slot]
                                for slot in range(state.used))
    expected_b = before_b - np.float32(0.01) * local_b.astype(np.float32)
    np.testing.assert_allclose(model.address_weight.numpy(), expected_b, rtol=0, atol=3e-8)

    # Each codec synapse receives its own sensory activity and current-prefix gate.
    local_w = np.zeros_like(before_w)
    local_bias = np.zeros_like(before_bias)
    for step in feedback.steps:
        for slot, cursor in enumerate(step.branch_cursors.tolist()):
            sensory = state.sensory[cursor].numpy()
            active, column = float(sensory.sum()), int(sensory.argmax())
            logits = before_w[:, column] * active + before_bias
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            error = probabilities.copy()
            error[step.observed_token] -= 1
            error *= math.exp(float(step.posterior_log_belief[slot]))
            local_w[:, column] += active * error
            local_bias += error
    np.testing.assert_allclose(model.codec.weight.numpy(), before_w - 0.01 * local_w,
                               rtol=2e-14, atol=2e-15)
    np.testing.assert_allclose(model.codec.bias.numpy(), before_bias - 0.01 * local_bias,
                               rtol=2e-14, atol=2e-15)
    assert feedback.codec_updated and feedback.address_updated and feedback.teacher_count == 3
    _assert_nested_equal(before_state, state.snapshot())
    _assert_nested_equal(before_state, feedback.state.snapshot())
    assert not any(parameter.requires_grad or parameter.grad is not None
                   for parameter in model.parameters())
    assert not hasattr(model, "address_features") and not hasattr(model, "projection_signal")


@pytest.mark.parametrize("mode,learning_enabled,codec_enabled,address_enabled,address_changed", [
    ("learned", True, True, True, True),
    ("address_frozen", True, True, False, False),
    ("slow_frozen", True, False, False, False),
    ("writes_disabled", True, True, True, False),
    ("learned", False, False, False, False),
])
def test_joint_address_explicit_local_publication_controls(
    mode, learning_enabled, codec_enabled, address_enabled, address_changed,
):
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(mode=mode, seed=733)
    state = model.ingest([b"ab."])
    before = model.snapshot(state)
    assert all(not parameter.requires_grad for parameter in model.parameters())
    feedback = model.teach(b"q", b"ab", state, 0.01, learning_enabled=learning_enabled)
    assert feedback.codec_updated == codec_enabled
    assert feedback.address_updated == address_changed
    assert feedback.work["enabled_codec_update_groups"] == int(codec_enabled)
    assert feedback.work["enabled_address_update_groups"] == int(address_enabled)
    assert feedback.work["codec_update_group_offers"] == 1
    assert feedback.work["address_update_group_offers"] == 1
    assert feedback.work["address_feature_slots"] == 4
    assert feedback.work["address_feature_products"] == 4 * 32768
    assert feedback.work["address_feature_channel_reductions"] == 4 * 31744
    assert feedback.work["address_score_synapse_evaluations"] == 4 * 1024
    assert feedback.work["address_score_reductions"] == 4 * 1023
    assert feedback.work["address_local_synapse_contributions"] == 4 * 1024
    assert feedback.work["address_local_slot_reductions"] == 3 * 1024
    assert feedback.work["codec_slot_evaluations"] == 4
    assert feedback.work["codec_response_gathers"] == 8
    assert feedback.work["teacher_offers"] == 3
    assert not any("projection" in name for name in feedback.work)
    if not address_changed:
        assert torch.equal(before["model"]["address_weight"], model.address_weight)
    if not codec_enabled:
        assert torch.equal(before["model"]["codec.weight"], model.codec.weight)
        assert torch.equal(before["model"]["codec.bias"], model.codec.bias)
    if mode == "writes_disabled":
        assert not state.keys.count_nonzero() and not state.sensory.count_nonzero()
        assert torch.equal(before["model"]["codec.weight"], model.codec.weight)
    _assert_nested_equal(before["memory"], state.snapshot())
    _assert_nested_equal(before["memory"], feedback.state.snapshot())
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.parameters())


# Authored executor qualification only: no neuronal or motor construction.
_DIRECT_QA_ATTEMPT = "v2"
_DIRECT_IDENTITY_PROGRAM = r'''
import ctypes, encodings, json, os, pathlib, runpy, site, sys
image = ctypes.create_unicode_buffer(32768)
length = ctypes.windll.kernel32.GetModuleFileNameW(None, image, len(image))
assert 0 < length < len(image)
flags = {name: getattr(sys.flags, name) for name in dir(sys.flags)
         if not name.startswith('_') and isinstance(getattr(sys.flags, name), int)}
print(json.dumps(dict(
    executable=sys.executable, base_executable=sys._base_executable,
    prefix=sys.prefix, exec_prefix=sys.exec_prefix,
    base_prefix=sys.base_prefix, base_exec_prefix=sys.base_exec_prefix,
    version=sys.version, implementation=sys.implementation.name,
    path=sys.path, flags=flags, user_site=site.ENABLE_USER_SITE,
    site_packages=site.getsitepackages(), filesystem_encoding=sys.getfilesystemencoding(),
    filesystem_errors=sys.getfilesystemencodeerrors(),
    stdio={name: [getattr(sys, name).encoding, getattr(sys, name).errors]
           for name in ('stdin', 'stdout', 'stderr')},
    stdlib_origins={m.__name__: getattr(m, '__file__', None)
                    for m in (os, pathlib, runpy, site, encodings)},
    launcher_environment=os.environ.get('__PYVENV_LAUNCHER__'),
    os_image=image.value, pid=os.getpid(), parent_pid=os.getppid(), cwd=os.getcwd()
), sort_keys=True))
'''


def _direct_executor_context():
    from pathlib import Path

    from experiments.r4_integrated import direct_python_action_bridge_v1 as direct
    from experiments.r4_integrated import utf8_action_bridge as old

    root = Path(__file__).resolve().parents[1]
    return root, old, direct, direct.discover_launch(root / ".venv/Scripts/python.exe")


def _direct_executor_sources(root, old, direct):
    from hashlib import sha256
    from pathlib import Path

    return {str(path): sha256(path.read_bytes()).hexdigest() for path in (
        Path(__file__).resolve(), Path(old.__file__).resolve(), Path(direct.__file__).resolve(),
        root / "docs/research/local-action-execution-speed-review.md",
    )}


def _direct_executor_fixture(old, name, value, *, program=None, pending=False):
    from dataclasses import replace

    case = old.Case(name, "setting", value, "authored_config", "pending")
    before = case.files()
    if program is not None:
        before = tuple(replace(item, content=program.encode("utf-8"))
                       if item.path == "main.py" else item for item in before)
    files = before if pending else tuple(
        replace(item, content=f"setting = {value!r}\n".encode())
        if item.path == "authored_config.py" else item for item in before)
    return case, files, before


def _direct_executor_pair(old, direct, launch, fixture, directory, order):
    case, files, before = fixture
    receipts = {}
    for route in order:
        workspace = directory / route
        receipts[route] = (old.execute_utf8(case, files, before, workspace)
                           if route == "old" else direct.execute_utf8(
                               case, files, before, workspace, launch=launch))
    return receipts


def _direct_executor_check_files(fixture, directory):
    files = fixture[1]
    for route in ("old", "direct"):
        workspace = directory / route
        assert {item.path: (workspace / item.path).read_bytes() for item in files} == {
            item.path: item.content for item in files}
        assert not list(workspace.rglob("*.pyc"))


def _direct_executor_save(root, name, value):
    import json

    path = root / ".pytest-tmp" / name
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def test_direct_python_executor_identity_pair(tmp_path):
    """Exactly two real stdlib-only children; run before the output-pair node."""
    import json
    import os
    import sys
    from pathlib import Path

    root, old, direct, launch = _direct_executor_context()
    passed_name = f"direct-python-identity-{_DIRECT_QA_ATTEMPT}.json"
    raw_name = f"direct-python-identity-{_DIRECT_QA_ATTEMPT}-raw.json"
    assert not (root / ".pytest-tmp" / passed_name).exists()
    assert not (root / ".pytest-tmp" / raw_name).exists()
    ambient = dict(os.environ)
    sources = _direct_executor_sources(root, old, direct)
    fixture = _direct_executor_fixture(old, "identity", "unused", program=_DIRECT_IDENTITY_PROGRAM)
    receipts = _direct_executor_pair(old, direct, launch, fixture, tmp_path, ("old", "direct"))
    raw_path = _direct_executor_save(root, raw_name, dict(
        schema="direct-python-identity-raw-v1", status="pending_semantic_checks",
        launch_record=launch.to_record(), source_sha256=sources,
        execution_order=["old", "direct"], receipts=receipts, child_executions=2))
    _direct_executor_check_files(fixture, tmp_path)
    identities = {}
    for route, receipt in receipts.items():
        assert receipt["returncode"] == 0 and receipt["stderr_hex"] == ""
        assert receipt["decode_errors"] == {}
        assert receipt["decoys_unchanged"] and receipt["materialized_matches_action"]
        identity = json.loads(receipt["stdout"])
        assert identity["pid"] != os.getpid() and identity["parent_pid"] > 0
        assert identity["pid"] != identity["parent_pid"]
        assert Path(identity["cwd"]).resolve() == (tmp_path / route).resolve()
        assert Path(identity["os_image"]).resolve() == Path(sys._base_executable).resolve()
        assert Path(identity["executable"]).resolve() == Path(sys.executable).resolve()
        assert Path(identity["prefix"]).resolve() == (root / ".venv").resolve()
        assert identity["flags"]["isolated"] == 1 and identity["flags"]["utf8_mode"] == 1
        assert identity["flags"]["dont_write_bytecode"] == 1
        assert identity["stdio"]["stdout"] == ["utf-8", "strict"]
        assert identity["stdio"]["stderr"] == ["utf-8", "strict"]
        assert identity["launcher_environment"] is None
        identities[route] = identity
    varying = {"pid", "parent_pid", "cwd"}
    assert {k: v for k, v in identities["old"].items() if k not in varying} == {
        k: v for k, v in identities["direct"].items() if k not in varying}
    assert dict(os.environ) == ambient
    assert sources == _direct_executor_sources(root, old, direct)
    from hashlib import sha256

    _direct_executor_save(root, passed_name, dict(
        schema="direct-python-identity-qa-v1", status="passed", launch_record=launch.to_record(),
        raw_evidence={str(raw_path): sha256(raw_path.read_bytes()).hexdigest()},
        source_sha256=sources, identity_pair=dict(passed=True, receipts=receipts,
        identities=identities, declared_varying_fields=sorted(varying)), child_executions=2))


def test_direct_python_executor_authored_output_pairs(tmp_path):
    """Exactly eight children, gated by the saved passing identity pair."""
    import json
    import os
    from hashlib import sha256

    root, old, direct, launch = _direct_executor_context()
    output_name = f"direct-python-outputs-{_DIRECT_QA_ATTEMPT}.json"
    assert not (root / ".pytest-tmp" / output_name).exists()
    raw_names = [f"direct-python-output-{_DIRECT_QA_ATTEMPT}-{i}-raw.json" for i in range(4)]
    assert all(not (root / ".pytest-tmp" / name).exists() for name in raw_names)
    sources = _direct_executor_sources(root, old, direct)
    identity_path = root / ".pytest-tmp" / f"direct-python-identity-{_DIRECT_QA_ATTEMPT}.json"
    identity_raw = identity_path.read_bytes()
    identity = json.loads(identity_raw)
    assert identity["status"] == "passed" and identity["child_executions"] == 2
    assert identity["launch_record"] == launch.to_record()
    assert identity["source_sha256"] == sources
    cases = [
        ("ascii", "ab", None, False, b"ab\n", b"", 0, True),
        ("unicode_escaping", "café '\\\n雪", None, False,
         "café '\\\n雪\n".encode(), b"", 0, True),
        ("pending", "desired", None, True, b"pending\n", b"", 0, False),
        ("invalid_utf8", "unused", "import os\nos.write(1, b'\\xff\\n')\n"
         "os.write(2, b'authored failure\\n')\nraise SystemExit(7)\n", False,
         b"\xff\n", b"authored failure\n", 7, False),
    ]
    pairs = []
    raw_evidence = {}
    ambient = dict(os.environ)
    for index, (name, value, program, pending, stdout, stderr, code, passed) in enumerate(cases):
        order = ("old", "direct") if index % 2 == 0 else ("direct", "old")
        fixture = _direct_executor_fixture(old, name, value, program=program, pending=pending)
        receipts = _direct_executor_pair(old, direct, launch, fixture, tmp_path / name, order)
        raw_path = _direct_executor_save(root, raw_names[index], dict(
            schema="direct-python-output-raw-v1", status="pending_semantic_checks",
            name=name, launch_record=launch.to_record(), source_sha256=sources,
            execution_order=list(order), receipts=receipts, child_executions=2))
        raw_evidence[str(raw_path)] = sha256(raw_path.read_bytes()).hexdigest()
        _direct_executor_check_files(fixture, tmp_path / name)
        for receipt in receipts.values():
            assert receipt["returncode"] == code
            assert receipt["stdout_hex"] == stdout.hex() and receipt["stderr_hex"] == stderr.hex()
            assert receipt["passed"] is passed
            assert receipt["decoys_unchanged"] and receipt["materialized_matches_action"]
            assert receipt["stdout"] == (None if name == "invalid_utf8" else stdout.decode("utf-8"))
            assert bool(receipt["decode_errors"]) == (name == "invalid_utf8")
        common = set(receipts["old"]) - {"workspace", "elapsed_seconds"}
        assert {k: receipts["old"][k] for k in common} == {
            k: receipts["direct"][k] for k in common}
        pairs.append(dict(name=name, passed=True, execution_order=list(order), receipts=receipts))
    assert dict(os.environ) == ambient
    assert sources == _direct_executor_sources(root, old, direct)
    _direct_executor_save(root, output_name, dict(
        schema="direct-python-output-qa-v1", status="passed", launch_record=launch.to_record(),
        source_sha256=sources,
        identity_evidence={str(identity_path): sha256(identity_raw).hexdigest()},
        raw_evidence=raw_evidence, output_pairs=pairs, child_executions=8))


def test_direct_python_executor_mocked_timeout_preserves_launch_contract(tmp_path, monkeypatch):
    """No children: injected timeout verifies kwargs, parent environment, and raw partial bytes."""
    import os
    import subprocess
    import sys

    _root, old, direct, launch = _direct_executor_context()
    captured = []
    ambient = dict(os.environ)

    def timeout(command, **kwargs):
        captured.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, 10, output=b"partial\xff", stderr=b"failure\n")

    monkeypatch.setattr(subprocess, "run", timeout)
    fixture = _direct_executor_fixture(old, "timeout", "ab")
    receipts = _direct_executor_pair(old, direct, launch, fixture, tmp_path, ("old", "direct"))
    _direct_executor_check_files(fixture, tmp_path)
    assert len(captured) == 2
    old_command, old_kwargs = captured[0]
    new_command, new_kwargs = captured[1]
    assert old_command[0] == sys.executable and new_command[0] == sys._base_executable
    assert old_command[1:] == new_command[1:]
    assert old_command[1:6] == ["-I", "-X", "utf8", "-B", "-c"]
    assert {k: v for k, v in old_kwargs.items() if k != "cwd"} == {
        k: v for k, v in new_kwargs.items() if k not in {"cwd", "env"}}
    assert new_kwargs["env"] == {**ambient, "__PYVENV_LAUNCHER__": sys.executable}
    assert new_kwargs["capture_output"] is True and new_kwargs["timeout"] == 10
    assert new_kwargs["check"] is False and dict(os.environ) == ambient
    for receipt in receipts.values():
        assert receipt["returncode"] is None and receipt["stdout"] is None
        assert receipt["stdout_hex"] == b"partial\xff".hex()
        assert receipt["stderr"] == "failure\n" and not receipt["passed"]
        assert "stdout" in receipt["decode_errors"]


def test_joint_online_v3_export_publishes_complete_request_once(tmp_path):
    """Real export/serialization, plain stubs: zero neuronal calls or child executions."""
    import json
    from types import SimpleNamespace

    from experiments.r4_integrated import local_joint_address_online_v3 as runner

    ingests, snapshots, guards = [], [], []

    class StubState:
        def snapshot(self):
            snapshots.append(None)
            return {}

    class StubModel:
        def state_dict(self):
            return {}

        def ingest(self, documents, state=None):
            assert state is None
            ingests.append(tuple(documents))
            return StubState()

    checks = [dict(id=i, memory_id=f"authored-{i}", base_memory_id=None,
                   document_hex=[bytes([65 + i]).hex()], additional_document_hex=[],
                   query_hex=b"authored query".hex(), delay_reads=0, target_hex=b"unused".hex())
              for i in range(12)]
    checks.append({**checks[0], "id": 12, "query_hex": b"second request".hex(), "delay_reads": 1})
    report = {"source_sha256": {"authored_source": "a" * 64}}
    launch = SimpleNamespace(sha256="b" * 64)
    runner.export_memories(runner.SPEC, {"learned": StubModel()}, checks, tmp_path,
                           report, lambda: guards.append(None), launch)

    assert ingests == [(bytes([65 + i]),) for i in range(12)]
    assert len(snapshots) == len(guards) == 12
    assert torch.load(tmp_path / "source_free.pt", weights_only=True) == {
        "learned": {"weights": {}, "memories": {f"authored-{i}": {} for i in range(12)}}}
    request_path = tmp_path / "query_only.json"
    original = request_path.read_bytes()
    request = json.loads(original)
    assert request == {
        "source_sha256": report["source_sha256"],
        "action_launch_record_sha256": launch.sha256,
        "cases": [{key: case[key] for key in ("id", "memory_id", "query_hex", "delay_reads")}
                  for case in checks],
    }
    assert all(set(case) == {"id", "memory_id", "query_hex", "delay_reads"}
               for case in request["cases"])
    assert report["source_free_tensor_bytes"] == 0
    assert report["final_ingestion_work"] == {"learned": {
        "document_raw_bytes": 12, "document_eos_symbols": 12,
        "tape_allocation_steps": 24, "tape_key_normalizations": 24,
        "tape_lag_shift_steps": 12, "tape_adjacency_assignments": 24,
        "sensory_association_event_opportunities": 24,
        "key_association_event_opportunities": 24, "byte_reference_writer_steps": 0,
    }}
    assert all(math.isfinite(report[key]) and report[key] >= 0 for key in (
        "final_ingestion_seconds", "source_free_serialization_seconds",
        "action_identity_binding_seconds"))
    with pytest.raises(FileExistsError):
        runner.write_json(request_path, {"must_not_replace": True})
    assert request_path.read_bytes() == original


_SOURCE_BOOTSTRAP_CASES = (
    "namespace_cleanup", "utf8_bom_crlf", "latin1_cookie", "syntax_error",
    "runtime_error", "exit_invalid_utf8", "pending",
)

_SOURCE_NAMESPACE_PROGRAM = r'''"""authored module doc"""
_initial = sorted(globals())
import __main__, atexit, ctypes, encodings, json, os, runpy, site, sys
import authored_config
_script_module = __main__
_image = ctypes.create_unicode_buffer(32768)
_length = ctypes.windll.kernel32.GetModuleFileNameW(None, _image, len(_image))
assert 0 < _length < len(_image)
print(json.dumps(dict(
    phase='body', initial_globals=_initial, name=__name__, file=__file__,
    spec=__spec__, package=__package__, loader=__loader__, cached=__cached__, doc=__doc__,
    main_is_globals=(__main__.__dict__ is globals()), argv=sys.argv,
    sibling=authored_config.setting, executable=sys.executable,
    base_executable=sys._base_executable, prefix=sys.prefix, exec_prefix=sys.exec_prefix,
    base_prefix=sys.base_prefix, base_exec_prefix=sys.base_exec_prefix,
    version=sys.version, path=sys.path, os_image=_image.value, user_site=site.ENABLE_USER_SITE,
    flags={n: getattr(sys.flags, n) for n in dir(sys.flags)
           if not n.startswith('_') and isinstance(getattr(sys.flags, n), int)},
    stdio={n: [getattr(sys, n).encoding, getattr(sys, n).errors]
           for n in ('stdin', 'stdout', 'stderr')},
    stdlib_origins={m.__name__: getattr(m, '__file__', None)
                   for m in (runpy, site, encodings)},
), sort_keys=True))
def _at_exit():
    print(json.dumps(dict(phase='atexit', argv0=sys.argv[0],
        main_restored=(sys.modules['__main__'] is not _script_module),
        retained_name=__name__, retained_file=__file__, retained_doc=__doc__), sort_keys=True))
atexit.register(_at_exit)
'''


def _source_bootstrap_fixture(old, name):
    from dataclasses import replace

    value, program, pending = "ab", None, False
    if name == "namespace_cleanup":
        program = _SOURCE_NAMESPACE_PROGRAM.encode("utf-8")
    elif name == "utf8_bom_crlf":
        value = "café '\\\n雪"
        program = b"\xef\xbb\xbf" + f"# coding: utf-8\r\nprint({value!r})\r\n".encode()
    elif name == "latin1_cookie":
        value, program = "café", b"# coding: latin-1\nprint('caf\xe9')\n"
    elif name == "syntax_error":
        program = b"if True print('bad')\n"
    elif name == "runtime_error":
        program = (b"import atexit, sys\n"
                   b"atexit.register(lambda: print('runtime-exit:' + sys.argv[0]))\n"
                   b"raise ValueError('authored runtime failure')\n")
    elif name == "exit_invalid_utf8":
        program = (b"import os, sys, atexit\n"
                   b"atexit.register(lambda: sys.stderr.write('exit-hook\\n'))\n"
                   b"os.write(1, b'\\xff\\n')\nsys.exit(7)\n")
    elif name == "pending":
        value, pending = "desired", True
    else:
        raise AssertionError("undeclared bootstrap fixture")
    fixture = _direct_executor_fixture(old, name, value, pending=pending)
    if program is None:
        return fixture
    case, files, before = fixture

    def replace_main(items):
        return tuple(replace(item, content=program) if item.path == "main.py" else item
                     for item in items)

    return case, replace_main(files), replace_main(before)


@pytest.mark.parametrize("case_name", _SOURCE_BOOTSTRAP_CASES)
def test_source_python_bootstrap_semantic_pair(tmp_path, case_name):
    """Seven pairs/14 children; wrapper traceback and import-event differences are declared."""
    import json
    import os
    import sys
    from hashlib import sha256
    from pathlib import Path

    from experiments.r4_integrated import direct_python_action_bridge_v1 as old
    from experiments.r4_integrated import source_python_action_bridge_v1 as new

    root = Path(__file__).resolve().parents[1]
    launch = old.discover_launch(root / ".venv/Scripts/python.exe")
    raw_name = f"source-python-bootstrap-v1-{case_name}-raw.json"
    passed_name = f"source-python-bootstrap-v1-{case_name}.json"
    assert not (root / ".pytest-tmp" / raw_name).exists()
    assert not (root / ".pytest-tmp" / passed_name).exists()
    source_paths = (Path(__file__).resolve(), Path(old.__file__).resolve(),
                    Path(new.__file__).resolve())
    source_hashes = {str(path): sha256(path.read_bytes()).hexdigest() for path in source_paths}
    fixture = _source_bootstrap_fixture(old, case_name)
    case, files, before = fixture
    index = _SOURCE_BOOTSTRAP_CASES.index(case_name)
    order = ("old", "source") if index % 2 == 0 else ("source", "old")
    receipts = {}
    ambient = dict(os.environ)
    for route in order:
        bridge = old if route == "old" else new
        receipts[route] = bridge.execute_utf8(case, files, before, tmp_path / route, launch=launch)
    raw_path = _direct_executor_save(root, raw_name, dict(
        schema="source-python-bootstrap-raw-v1", status="pending_semantic_checks", case=case_name,
        execution_order=list(order), launch_record=launch.to_record(), source_sha256=source_hashes,
        fixture_files={item.path: item.content.hex() for item in files},
        receipts=receipts, child_executions=2))

    assert dict(os.environ) == ambient
    assert source_hashes == {
        str(path): sha256(path.read_bytes()).hexdigest() for path in source_paths}
    assert receipts["source"]["bootstrap_policy"] == new.BOOTSTRAP_POLICY
    assert receipts["source"]["bootstrap_sha256"] == new.BOOTSTRAP_SHA256
    assert receipts["source"]["child_command"] == [
        launch.base_executable, "-I", "-X", "utf8", "-B", "-c", new.BOOTSTRAP]
    for route, receipt in receipts.items():
        assert receipt["launch_record_sha256"] == launch.sha256
        assert receipt["child_environment_override"] == {
            "__PYVENV_LAUNCHER__": launch.venv_executable}
        assert receipt["child_python_flags"] == ["-I", "-X", "utf8", "-B"]
        assert receipt["decoys_unchanged"] and receipt["materialized_matches_action"]
        assert {item.path: (tmp_path / route / item.path).read_bytes() for item in files} == {
            item.path: item.content for item in files}
        assert not list((tmp_path / route).rglob("*.pyc"))
        for channel in ("stdout", "stderr"):
            raw_channel = bytes.fromhex(receipt[channel + "_hex"])
            assert receipt[channel + "_sha256"] == sha256(raw_channel).hexdigest()

    if case_name == "namespace_cleanup":
        identities = {}
        for route, receipt in receipts.items():
            assert receipt["returncode"] == 0 and receipt["stderr_hex"] == ""
            body, exit_record = map(json.loads, receipt["stdout"].splitlines())
            assert body["initial_globals"] == sorted([
                "__name__", "__doc__", "__package__", "__loader__", "__spec__",
                "__file__", "__cached__", "__builtins__"])
            assert {key: body[key] for key in ("name", "file", "package", "doc")} == {
                "name": "__main__", "file": "main.py", "package": "", "doc": "authored module doc"}
            assert all(body[key] is None for key in ("spec", "loader", "cached"))
            assert body["main_is_globals"] is True and body["argv"] == ["main.py"]
            assert body["sibling"] == "ab"
            assert body["executable"] == sys.executable
            assert body["base_executable"] == sys._base_executable
            assert Path(body["os_image"]).resolve() == Path(sys._base_executable).resolve()
            assert body["stdio"]["stdout"] == body["stdio"]["stderr"] == ["utf-8", "strict"]
            assert body["flags"]["isolated"] == body["flags"]["utf8_mode"] == 1
            assert exit_record == dict(phase="atexit", argv0="-c", main_restored=True,
                retained_name="__main__", retained_file="main.py",
                retained_doc="authored module doc")
            identities[route] = [body, exit_record]
        assert identities["old"] == identities["source"]
    elif case_name in {"syntax_error", "runtime_error"}:
        for receipt in receipts.values():
            assert receipt["returncode"] == 1 and receipt["decode_errors"] == {}
            assert not receipt["passed"]
            if case_name == "syntax_error":
                assert receipt["stdout_hex"] == ""
                assert 'File "main.py", line 1' in receipt["stderr"]
                assert "if True print('bad')" in receipt["stderr"]
                assert receipt["stderr"].splitlines()[-1] == "SyntaxError: invalid syntax"
            else:
                assert receipt["stdout"] == "runtime-exit:-c\n"
                assert 'File "main.py", line 3, in <module>' in receipt["stderr"]
                assert receipt["stderr"].splitlines()[-1] == "ValueError: authored runtime failure"
    else:
        expected_stdout = b"\xff\n" if case_name == "exit_invalid_utf8" else (
            b"pending\n" if case_name == "pending" else (case.value + "\n").encode("utf-8"))
        expected_stderr = b"exit-hook\n" if case_name == "exit_invalid_utf8" else b""
        for receipt in receipts.values():
            assert receipt["stdout_hex"] == expected_stdout.hex()
            assert receipt["stderr_hex"] == expected_stderr.hex()
            assert receipt["returncode"] == (7 if case_name == "exit_invalid_utf8" else 0)
            assert receipt["passed"] is (case_name in {"utf8_bom_crlf", "latin1_cookie"})
            if case_name == "exit_invalid_utf8":
                assert receipt["stdout"] is None and set(receipt["decode_errors"]) == {"stdout"}
            else:
                assert receipt["stdout"] == expected_stdout.decode("utf-8")
                assert receipt["decode_errors"] == {}
    _direct_executor_save(root, passed_name, dict(
        schema="source-python-bootstrap-case-qa-v1", status="passed", case=case_name,
        case_index=index, execution_order=list(order), launch_record=launch.to_record(),
        source_sha256=source_hashes, bootstrap_policy=new.BOOTSTRAP_POLICY,
        bootstrap_sha256=new.BOOTSTRAP_SHA256,
        raw_evidence={str(raw_path): sha256(raw_path.read_bytes()).hexdigest()},
        declared_differences=["bootstrap_frames", "bootstrap_import_set",
                              "import_path_hook_and_audit_events"],
        child_executions=2))


def test_joint_address_autograd_and_invalid_learning_flag_rejected_before_activity():
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"a."])
    before = model.snapshot(state)
    for parameter in (model.address_weight, model.codec.weight, model.codec.bias):
        parameter.requires_grad_(True)
        try:
            with pytest.raises(ValueError, match="outside autograd"):
                model.query_receipt(b"q", state)
        finally:
            parameter.requires_grad_(False)
    with pytest.raises(ValueError, match="learning_enabled must be boolean"):
        model.teach(b"q", b"ab", state, 0.01, learning_enabled=1)
    _assert_nested_equal(before, model.snapshot(state))


@pytest.mark.parametrize("raw,target,emitted,returncode,stdout,success", [
    (b"ab", b"ab", True, 0, b"ab\n", True),
    (b"ab", b"cd", True, 0, b"ab\n", False),
    (b"", b"ab", False, 0, b"pending\n", False),
    (b"ab", b"ab", True, 1, b"", False),
    (b"ab", b"ab", True, 0, b"different\n", False),
])
def test_joint_online_grade_separates_task_from_execution(
    raw, target, emitted, returncode, stdout, success,
):
    from experiments.r4_integrated import local_joint_address_online_v1 as runner

    generated = dict(generated_hex=raw.hex(), stop_reason="eos")
    action = dict(emitted=emitted, lossless_generated_execution=True,
                  execution=dict(returncode=returncode, stdout_hex=stdout.hex()))
    grade = runner.external_grade(generated, action, target)
    assert grade["task_success"] is success
    assert grade["model_correct"] == (raw == target)
    assert grade["process_succeeded"] == (returncode == 0)
    assert grade["pending_fallback"] == (not emitted)


def test_joint_online_paired_journals_and_two_stage_projection(tmp_path):
    import hashlib
    import json

    from experiments.r4_integrated import local_joint_address_online_v1 as runner

    actions = runner.JsonlJournal(tmp_path / "actions.jsonl", runner.SPEC)
    lessons = runner.LessonJournal(tmp_path / "lessons.jsonl", runner.SPEC)
    for index in range(4):
        for arm in runner.ARMS:
            action_sha = actions.append(dict(arm=arm, index=index, result="wrong"))
            lessons.append(dict(arm=arm, index=index, action_row_sha256=action_sha,
                model_seconds=.001 * (index + 1), action_seconds=.002 * (index + 1),
                first_exact=False, task_success=False, decisions=3, teacher_count=3,
                codec_group_enabled=arm != "slow_frozen", codec_updated=False,
                address_group_enabled=arm in {"learned", "writes_disabled"},
                address_updated=False, work={"teacher_offers": 3},
                observations={"nested": [1.25, None, "\u00e9"]}))
    action_log, lesson_log = actions.close(), lessons.close()
    assert action_log["rows"] == lesson_log["rows"] == 16
    assert lessons.counts["teacher_byte_offers"] == 48
    assert lessons.counts["enabled_teacher_offers"] == 36
    for action_line, lesson_line in zip(actions.path.read_bytes().splitlines(keepends=True),
                                      lessons.path.read_bytes().splitlines(keepends=True),
                                      strict=True):
        row = json.loads(lesson_line)
        assert row["action_row_sha256"] == hashlib.sha256(action_line).hexdigest()
    assert not hasattr(lessons, "lessons")
    report = dict(status="completed", mode="cost", config=runner.SPEC.config,
        duration_clock=runner.duration_clock(), counts=lessons.counts,
        final_action_log={"rows": 80}, motor_demonstrations=[{}, {}],
        lesson_summary=lessons.summary, lesson_log=lesson_log,
        cap_generation={arm: dict(stop_reason="limit", decisions=3,
            generated_hex="0000", seconds=.001) for arm in runner.ARMS},
        memory={"peak_working_set_mib": 100}, worker={"memory": {"peak_working_set_mib": 80}})
    launch = dict(elapsed_seconds=10, observed_tree_peak_sum_mib=150)
    pending = runner.project_cost(runner.SPEC, report, launch)
    assert pending["audit_peak_pending"] and not pending["admissible"]
    assert pending["projected_audit_seconds"] is None
    audit = dict(status="passed", learning_pair_timing={
        arm: dict(rows=4, seconds=.0005, max_seconds=.0002) for arm in runner.ARMS})
    monitor = dict(status="passed", exit_code=0, elapsed_seconds=1,
                   observed_tree_peak_sum_mib=200)
    final = runner.project_cost(runner.SPEC, report, launch, audit_launch=monitor, audit=audit)
    assert final["admissible"] and not final["audit_peak_pending"]
    assert final["peak_upper_mib"] == 311
    assert final["projected_audit_seconds"] == pytest.approx(2 * (1024 * .0008 + .998))
    expected_learning = 2 * 1024 * sum(
        .004 + .008 + row["max_journal_seconds"] + .001 for row in lessons.summary.values())
    assert final["projected_learning_seconds"] == pytest.approx(expected_learning)
    assert final["projected_seconds"] == pytest.approx(
        pending["projected_run_seconds"] + final["projected_audit_seconds"])
    rejected = runner.JsonlJournal(tmp_path / "rejected.jsonl", runner.SPEC)
    try:
        with pytest.raises(MemoryError, match="host-tree"):
            rejected.append({"value": "x" * 262144})
        with pytest.raises(MemoryError, match="line limit"):
            rejected.append({"value": "\u00ff" * 50000})
        assert rejected.rows == 0
    finally:
        rejected.close()
    with pytest.raises(ValueError, match="truncated"):
        runner.read_row(io.BytesIO(b'{"x":1}'), runner.SPEC)
    empty = {arm: dict(fit=0, unseen_value=0, delayed=0, interference=0) for arm in runner.ARMS}
    empty["learned"] = dict(fit=8, unseen_value=4, delayed=4, interference=2)
    assert runner.preserves_workload(empty) and not runner.qualifies(empty)
    empty["learned"]["interference"] = 4
    assert runner.qualifies(empty)


@pytest.mark.parametrize("arm", ["learned", "address_frozen", "slow_frozen", "writes_disabled"])
def test_joint_online_disposable_cap_and_feature_counts(arm):
    from experiments.r4_integrated import local_joint_address_online_v1 as runner

    model = runner.new_model(runner.SPEC, arm)
    before = copy.deepcopy(model.state_dict())
    cap = runner.capped_prediction(runner.SPEC, arm, model)
    assert cap["disposable_clone"] and cap["original_weights_restored"]
    assert cap["occupied_slots"] == 128
    assert cap["generated_hex"] == "0000" and cap["decisions"] == 3
    assert cap["work"]["address_feature_products"] == 128 * 32768
    assert cap["work"]["address_score_synapse_evaluations"] == 128 * 1024
    assert cap["work"]["codec_slot_evaluations"] == 128
    assert cap["work"]["codec_response_gathers"] == 2 * 128
    accounting = model.accounting()
    assert accounting["maximum_held_codec_response_bank_bytes"] == 921088
    assert accounting["maximum_overlapping_codec_response_bytes"] == 2763264
    assert cap["work"]["teacher_offers"] == 0
    assert cap["ingestion_work"]["document_raw_bytes"] == 127
    assert cap["ingestion_work"]["document_eos_symbols"] == 1
    _assert_nested_equal(before, model.state_dict())


def test_joint_online_fresh_worker_query_only_four_modes(tmp_path):
    import json
    import os
    import subprocess
    import time

    from experiments.r4_integrated import local_joint_address_online_v1 as runner

    saved = {}
    for arm in runner.ARMS:
        model = runner.new_model(runner.SPEC, arm)
        state = model.ingest([b""])
        saved[arm] = dict(weights=runner.detached_state(model.state_dict()),
            memories={f"m{i}": runner.detached_state(state.snapshot()) for i in range(12)})
    checkpoint = tmp_path / "source_free.pt"
    torch.save(saved, checkpoint)
    request_path = tmp_path / "query_only.json"
    request = dict(source_sha256=runner.sources(),
                   cases=[dict(id=0, memory_id="m0", query_hex=b"q".hex(), delay_reads=0)])
    runner.write_json(request_path, request)
    output = tmp_path / "generated.json"
    command = [str(runner.ROOT / ".venv/Scripts/python.exe"), "-I", "-X", "utf8",
        str(runner.SPEC.entrypoint), "worker", "--output", str(output),
        "--checkpoint", str(checkpoint), "--requests", str(request_path),
        "--checkpoint-sha", runner.file_hash(checkpoint),
        "--requests-sha", runner.file_hash(request_path),
        "--deadline-unix", str(time.time() + 40)]
    child = subprocess.run(command, cwd=tmp_path, capture_output=True, timeout=45)
    assert child.returncode == 0, child.stderr.decode(errors="replace")
    result = json.loads(output.read_bytes())
    assert result["worker_pid"] != os.getpid()
    assert result["output_log"]["rows"] == 4 and result["query_byte_steps"] == 8
    assert not result["roster_invoked_or_documents_or_targets_supplied_to_model"]
    assert result["checkpoint_sha256"] == runner.file_hash(checkpoint)
    raw = output.with_suffix(".jsonl").read_bytes()
    assert result["output_log"]["sha256"] == __import__("hashlib").sha256(raw).hexdigest()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert [row["arm"] for row in rows] == list(runner.ARMS)
    assert [(row["generated_hex"], row["stop_reason"]) for row in rows] == [
        ("", "eos"), ("", "eos"), ("", "eos"), ("0000", "limit")]
    assert sum(len(row["steps"]) for row in rows) == 6
    assert all(row["work"]["address_feature_slots"] == 1 for row in rows)
    assert runner.sources() == request["source_sha256"]


def test_joint_address_injected_late_validation_failure_preserves_both_groups():
    """Injected proposal rejection checks ordering; it is not an overflow experiment."""
    from humanbrain import local_memory_learner as learner

    model = learner.LocalMemoryLearner(seed=733)
    state = model.ingest([b"ab."])
    before = model.snapshot(state)
    actual_finite = learner._finite
    proposal_checks = []

    def reject_last_proposal(value, name):
        actual_finite(value, name)
        if name == "codec parameter proposal":
            proposal_checks.append(name)
        elif name == "address parameter proposal":
            assert proposal_checks == ["codec parameter proposal"] * 2
            proposal_checks.append(name)
            raise ValueError("injected late address proposal rejection")

    with patch.object(learner, "_finite", side_effect=reject_last_proposal), \
            pytest.raises(ValueError, match="injected late address proposal rejection"):
        model.teach(b"q", b"ab", state, 0.01)
    assert proposal_checks == ["codec parameter proposal"] * 2 + ["address parameter proposal"]
    _assert_nested_equal(before, model.snapshot(state))
    assert all(not parameter.requires_grad and parameter.grad is None
               for parameter in model.parameters())


# New environment qualification: 1 identity + 14 semantic + 2 contamination children.
_STDLIB_IDENTITY_PROGRAM = r'''
import sys, os, json, ctypes, importlib.util
image = ctypes.create_unicode_buffer(32768)
length = ctypes.windll.kernel32.GetModuleFileNameW(None, image, len(image))
assert 0 < length < len(image)
print(json.dumps(dict(executable=sys.executable, base_executable=sys._base_executable,
    prefix=sys.prefix, exec_prefix=sys.exec_prefix, base_prefix=sys.base_prefix,
    base_exec_prefix=sys.base_exec_prefix, version=sys.version, path=sys.path,
    os_image=image.value, pid=os.getpid(), cwd=os.getcwd(),
    no_site=sys.flags.no_site, isolated=sys.flags.isolated, utf8=sys.flags.utf8_mode,
    no_bytecode=sys.flags.dont_write_bytecode,
    site_loaded=('site' in sys.modules),
    hooks=[n for n in sys.modules if n in ('sitecustomize','usercustomize')
           or n.startswith('__editable__')],
    unavailable={n: importlib.util.find_spec(n) is None for n in ('humanbrain','numpy','torch')},
    launcher=os.environ.get('__PYVENV_LAUNCHER__'),
    stdio={n: [getattr(sys,n).encoding,getattr(sys,n).errors] for n in ('stdout','stderr')}
), sort_keys=True))
'''


def _stdlib_context():
    from hashlib import sha256
    from pathlib import Path

    from experiments.r4_integrated import source_python_action_bridge_v1 as old
    from experiments.r4_integrated import stdlib_python_action_bridge_v1 as new

    root = Path(__file__).resolve().parents[1]
    launch = new.discover_launch(root / ".venv/Scripts/python.exe")
    paths = (Path(__file__).resolve(), Path(old.__file__).resolve(), Path(new.__file__).resolve())
    sources = {str(path): sha256(path.read_bytes()).hexdigest() for path in paths}
    return root, old, new, launch, sources


def _stdlib_check_base_identity(identity, launch, workspace):
    from pathlib import Path

    base = Path(launch.base_executable).resolve()
    assert Path(identity["executable"]).resolve() == base
    assert Path(identity["base_executable"]).resolve() == base
    assert Path(identity["os_image"]).resolve() == base
    for key in ("prefix", "exec_prefix", "base_prefix", "base_exec_prefix"):
        assert Path(identity[key]).resolve() == base.parent
    for entry in identity["path"]:
        resolved = ((workspace / entry).resolve() if not Path(entry).is_absolute()
                    else Path(entry).resolve())
        assert resolved == workspace.resolve() or resolved.is_relative_to(base.parent)
        assert ".venv" not in resolved.parts and "site-packages" not in resolved.parts


def _stdlib_execute_saved(root, name, fixture, workspace, routes, launch, sources, old, new):
    """Save all actual receipts before any output or materialization assertions."""
    import os

    raw_name = f"stdlib-python-v1-{name}-raw.json"
    passed_name = f"stdlib-python-v1-{name}.json"
    assert not (root / ".pytest-tmp" / raw_name).exists()
    assert not (root / ".pytest-tmp" / passed_name).exists()
    ambient = dict(os.environ)
    case, files, before = fixture
    receipts = {}
    for route in routes:
        bridge, record = (old, launch.parent_runtime) if route == "old" else (new, launch)
        receipts[route] = bridge.execute_utf8(
            case, files, before, workspace / route, launch=record)
    raw_path = _direct_executor_save(root, raw_name, dict(
        schema="stdlib-python-raw-qa-v1", status="pending_semantic_checks", name=name,
        execution_order=list(routes), source_sha256=sources, launch_record=launch.to_record(),
        fixture_files={item.path: item.content.hex() for item in files}, receipts=receipts,
        child_executions=len(routes)))
    assert dict(os.environ) == ambient
    for route, receipt in receipts.items():
        assert receipt["decoys_unchanged"] and receipt["materialized_matches_action"]
        assert {item.path: (workspace / route / item.path).read_bytes() for item in files} == {
            item.path: item.content for item in files}
        assert not list((workspace / route).rglob("*.pyc"))
        if route != "old":
            assert receipt["child_command"] == [
                launch.base_executable, "-I", "-S", "-X", "utf8", "-B", "-c", new.BOOTSTRAP]
            assert receipt["child_environment_override"] == {}
            assert receipt["child_environment_removed"] == ["__PYVENV_LAUNCHER__"]
            assert receipt["launch_record_sha256"] == launch.sha256
    return raw_path, passed_name, receipts


def _stdlib_pass(root, name, raw_path, launch, sources, count, **details):
    from hashlib import sha256
    from pathlib import Path

    assert sources == {path: sha256(Path(path).read_bytes()).hexdigest() for path in sources}
    _direct_executor_save(root, name, dict(schema="stdlib-python-case-qa-v1", status="passed",
        launch_record=launch.to_record(), source_sha256=sources, child_executions=count,
        raw_evidence={str(raw_path): sha256(raw_path.read_bytes()).hexdigest()}, **details))


def _stdlib_require_identity(root, launch, sources):
    import json

    identity = json.loads((root / ".pytest-tmp/stdlib-python-v1-identity.json").read_bytes())
    assert identity["status"] == "passed" and identity["child_executions"] == 1
    assert identity["launch_record"] == launch.to_record() and identity["source_sha256"] == sources


def test_stdlib_python_action_identity(tmp_path):
    """One new child; no repeat of the historical runtime qualification."""
    import json
    import os
    import sys

    root, old, new, launch, sources = _stdlib_context()
    fixture = _direct_executor_fixture(old, "stdlib-identity", "unused",
                                       program=_STDLIB_IDENTITY_PROGRAM)
    raw_path, passed_name, receipts = _stdlib_execute_saved(
        root, "identity", fixture, tmp_path, ("new",), launch, sources, old, new)
    receipt = receipts["new"]
    assert receipt["returncode"] == 0 and receipt["stderr_hex"] == ""
    identity = json.loads(receipt["stdout"])
    _stdlib_check_base_identity(identity, launch, tmp_path / "new")
    assert identity["version"] == sys.version and identity["pid"] != os.getpid()
    assert all(identity[key] == 1 for key in ("no_site", "isolated", "utf8", "no_bytecode"))
    assert identity["site_loaded"] is False and identity["hooks"] == []
    assert identity["unavailable"] == dict(humanbrain=True, numpy=True, torch=True)
    assert identity["launcher"] is None
    assert identity["stdio"] == {name: ["utf-8", "strict"] for name in ("stdout", "stderr")}
    _stdlib_pass(root, passed_name, raw_path, launch, sources, 1, identity=identity)


@pytest.mark.parametrize("case_name", _SOURCE_BOOTSTRAP_CASES)
def test_stdlib_python_action_semantic_pair(tmp_path, case_name):
    """Fourteen children reuse the frozen source fixtures; new runtime fields are explicit."""
    import json

    root, old, new, launch, sources = _stdlib_context()
    _stdlib_require_identity(root, launch, sources)
    fixture = _source_bootstrap_fixture(old, case_name)
    case = fixture[0]
    index = _SOURCE_BOOTSTRAP_CASES.index(case_name)
    order = ("old", "new") if index % 2 == 0 else ("new", "old")
    raw_path, passed_name, receipts = _stdlib_execute_saved(
        root, case_name, fixture, tmp_path, order, launch, sources, old, new)
    if case_name == "namespace_cleanup":
        old_body, old_exit = map(json.loads, receipts["old"]["stdout"].splitlines())
        body, exit_record = map(json.loads, receipts["new"]["stdout"].splitlines())
        environment_fields = {"executable", "base_executable", "prefix", "exec_prefix",
            "base_prefix", "base_exec_prefix", "path", "user_site", "flags", "stdlib_origins"}
        assert {k: v for k, v in old_body.items() if k not in environment_fields} == {
            k: v for k, v in body.items() if k not in environment_fields}
        _stdlib_check_base_identity(body, launch, tmp_path / "new")
        assert body["flags"] == {**old_body["flags"], "no_site": 1}
        assert old_body["user_site"] is False and body["user_site"] is None
        from pathlib import Path

        assert {k: Path(v).resolve() for k, v in body["stdlib_origins"].items()} == {
            k: Path(v).resolve() for k, v in old_body["stdlib_origins"].items()}
        assert body["main_is_globals"] is True and body["argv"] == ["main.py"]
        assert body["name"] == "__main__" and body["package"] == "" and body["sibling"] == "ab"
        assert exit_record == old_exit == dict(phase="atexit", argv0="-c", main_restored=True,
            retained_name="__main__", retained_file="main.py", retained_doc="authored module doc")
        assert all(r["returncode"] == 0 and r["stderr_hex"] == "" for r in receipts.values())
    elif case_name in {"syntax_error", "runtime_error"}:
        for receipt in receipts.values():
            assert receipt["returncode"] == 1 and receipt["decode_errors"] == {}
            assert not receipt["passed"]
            if case_name == "syntax_error":
                assert receipt["stdout_hex"] == ""
                assert 'File "main.py", line 1' in receipt["stderr"]
                assert "if True print('bad')" in receipt["stderr"]
                assert receipt["stderr"].splitlines()[-1] == "SyntaxError: invalid syntax"
            else:
                assert receipt["stdout"] == "runtime-exit:-c\n"
                assert 'File "main.py", line 3, in <module>' in receipt["stderr"]
                assert receipt["stderr"].splitlines()[-1] == "ValueError: authored runtime failure"
    else:
        stdout = b"\xff\n" if case_name == "exit_invalid_utf8" else (
            b"pending\n" if case_name == "pending" else (case.value + "\n").encode())
        stderr = b"exit-hook\n" if case_name == "exit_invalid_utf8" else b""
        for receipt in receipts.values():
            assert receipt["stdout_hex"] == stdout.hex() and receipt["stderr_hex"] == stderr.hex()
            assert receipt["returncode"] == (7 if case_name == "exit_invalid_utf8" else 0)
            assert receipt["passed"] is (case_name in {"utf8_bom_crlf", "latin1_cookie"})
            expected_text = None if case_name == "exit_invalid_utf8" else stdout.decode()
            expected_errors = {"stdout"} if case_name == "exit_invalid_utf8" else set()
            assert receipt["stdout"] == expected_text
            assert set(receipt["decode_errors"]) == expected_errors
    _stdlib_pass(root, passed_name, raw_path, launch, sources, 2, case=case_name, case_index=index,
        declared_differences=["base_no_site_runtime", "bootstrap_frames",
                              "import_and_audit_events"])


def test_stdlib_python_action_fresh_process_contamination(tmp_path):
    """Two new children; first mutates process memory/cwd, second gets a clean process/workspace."""
    import json
    import os

    root, old, new, launch, sources = _stdlib_context()
    _stdlib_require_identity(root, launch, sources)
    marker = "HUMANBRAIN_AUTHORED_CONTAMINATION_V1"
    assert marker not in os.environ
    producer = (
        "import os,sys,builtins,types,json,authored_config\n"
        f"os.environ[{marker!r}]='dirty'\n"
        "builtins._authored_dirty=True\n"
        "sys.modules['_authored_dirty']=types.ModuleType('_authored_dirty')\n"
        "authored_config.setting='dirty'\nos.mkdir('inner')\nos.chdir('inner')\n"
        "print(json.dumps(dict(pid=os.getpid(),cwd=os.getcwd(),mutated=True)))\n")
    consumer = (
        "import os,sys,builtins,json\n"
        f"clean_env=({marker!r} not in os.environ)\n"
        "clean_modules=('_authored_dirty' not in sys.modules "
        "and 'authored_config' not in sys.modules)\n"
        "import authored_config\n"
        "print(json.dumps(dict(pid=os.getpid(),cwd=os.getcwd(),clean_env=clean_env,"
        "clean_modules=clean_modules,clean_builtins=not hasattr(builtins,'_authored_dirty'),"
        "sibling=authored_config.setting,clean_workspace=not os.path.exists('inner'))))\n")
    raw_paths, receipts = [], []
    # Precheck both pairs before the first launch; each raw result is preserved immediately.
    for name in ("contamination-producer", "contamination-consumer"):
        assert not (root / ".pytest-tmp" / f"stdlib-python-v1-{name}-raw.json").exists()
        assert not (root / ".pytest-tmp" / f"stdlib-python-v1-{name}.json").exists()
    passed_name = "stdlib-python-v1-contamination.json"
    assert not (root / ".pytest-tmp" / passed_name).exists()
    for name, program in (("producer", producer), ("consumer", consumer)):
        fixture = _direct_executor_fixture(old, name, "ab", program=program)
        raw_path, _, record = _stdlib_execute_saved(root, f"contamination-{name}", fixture,
            tmp_path / name, ("new",), launch, sources, old, new)
        raw_paths.append(raw_path)
        receipts.append(record["new"])
    assert all(r["returncode"] == 0 and r["stderr_hex"] == "" for r in receipts)
    first, second = (json.loads(r["stdout"]) for r in receipts)
    assert first["mutated"] is True
    assert first["pid"] != second["pid"] and os.getpid() not in {first["pid"], second["pid"]}
    from hashlib import sha256
    from pathlib import Path

    assert Path(first["cwd"]).resolve() == (tmp_path / "producer/new/inner").resolve()
    assert Path(second["cwd"]).resolve() == (tmp_path / "consumer/new").resolve()
    assert all(second[key] is True for key in (
        "clean_env", "clean_modules", "clean_builtins", "clean_workspace"))
    assert second["sibling"] == "ab" and marker not in os.environ
    _stdlib_pass(root, passed_name, raw_paths[0], launch, sources, 2,
        contamination=dict(passed=True),
        all_raw_evidence={str(path): sha256(path.read_bytes()).hexdigest() for path in raw_paths})


@pytest.mark.parametrize("mode,learning_enabled", [
    ("learned", True), ("address_frozen", True), ("slow_frozen", True),
    ("writes_disabled", True), ("learned", False),
])
def test_held_codec_exact_current_local_reference(mode, learning_enabled):
    """Authored call scheduling parity against the frozen no-backprop source."""
    import dataclasses
    import hashlib
    import subprocess
    import sys
    import types
    from pathlib import Path

    from humanbrain.local_memory_learner import LocalMemoryLearner

    root = Path(__file__).resolve().parents[1]
    raw = subprocess.check_output([
        "git", "show", "64f9e0a60fe042a39a72a80c82d0a78c12960c2a:"
        "src/humanbrain/local_memory_learner.py",
    ], cwd=root)
    assert hashlib.sha256(raw).hexdigest() == (
        "4a4aa24f14f30705cb8a83a0954291ed86e7e4473744920775d3a7961d770ccc")
    old = types.ModuleType("_held_codec_local_reference")

    def same_steps(first, second):
        assert len(first) == len(second)
        for a, b in zip(first, second, strict=True):
            for field in dataclasses.fields(a):
                _assert_nested_equal(getattr(a, field.name), getattr(b, field.name))

    def same_weights(first, second):
        a, b = first.state_dict(), second.state_dict()
        # Only explicitly declared simulator scheduling metadata may differ.
        extra_a, extra_b = a.pop("_extra_state"), b.pop("_extra_state")
        assert extra_a.pop("schema") == "local-neuronal-joint-address-learner-v1"
        assert extra_b.pop("schema") == "local-neuronal-joint-address-held-codec-learner-v1"
        assert extra_b.pop("codec_activity") == (
            "evaluate all occupied sensory slots once per call with held weights; "
            "first decision uses bank, later forward cursors gather responses; "
            "discard before any later call or publication-dependent reuse"
        )
        assert extra_a == extra_b
        _assert_nested_equal(a, b)

    def same_prediction(first, second, slots):
        assert first.generation == second.generation
        same_steps(first.steps, second.steps)
        _assert_nested_equal(first.initial_log_belief, second.initial_log_belief)
        _assert_nested_equal(first.final_log_belief, second.final_log_belief)
        same_work(first.work, second.work, slots, len(first.steps))

    def same_work(first, second, slots, decisions):
        changed = {"codec_slot_evaluations", "codec_output_potentials", "codec_normalizations"}
        added = {key for key in second if key not in first}
        assert added == {
            "codec_response_gathers", "codec_response_values_gathered",
            "codec_response_bytes_gathered",
        }
        assert first.keys() <= second.keys()
        for key in first.keys() - changed:
            assert first[key] == second[key]
        assert first["codec_slot_evaluations"] == slots * decisions
        assert first["codec_normalizations"] == slots * decisions
        assert first["codec_output_potentials"] == slots * decisions * 257
        assert second["codec_slot_evaluations"] == second["codec_normalizations"] == slots
        assert second["codec_output_potentials"] == slots * 257
        gathered = slots * (decisions - 1)
        assert second["codec_response_gathers"] == gathered
        assert second["codec_response_values_gathered"] == gathered * 257 * 4
        assert second["codec_response_bytes_gathered"] == gathered * 257 * 28

    with patch.dict(sys.modules, {old.__name__: old}):
        exec(compile(raw, "<no-backprop-held-codec-reference>", "exec"), old.__dict__)
        models = [old.LocalMemoryLearner(mode=mode, seed=733),
                  LocalMemoryLearner(mode=mode, seed=733)]
        same_weights(*models)
        states = [model.ingest([b"aa", b""]) for model in models]
        _assert_nested_equal(states[0].snapshot(), states[1].snapshot())
        before_state = states[1].snapshot()
        new_calls = []
        decisions = []
        evaluate = models[1].codec.evaluate

        def counted(sensory):
            new_calls.append(sensory.shape[0])
            return evaluate(sensory)

        with patch.object(models[1].codec, "evaluate", side_effect=counted):
            predictions = [model.predict_with_receipt(b"r", state, 4)
                           for model, state in zip(models, states, strict=True)]
            same_prediction(*predictions, states[0].used)
            decisions.extend(len(item.steps) for item in predictions)
            teachers = [model.teach(b"r", b"aaba", state, .01,
                                    learning_enabled=learning_enabled)
                        for model, state in zip(models, states, strict=True)]
            same_steps(teachers[0].steps, teachers[1].steps)
            for field in dataclasses.fields(teachers[0]):
                if field.name not in {"steps", "state", "work"}:
                    _assert_nested_equal(getattr(teachers[0], field.name),
                                         getattr(teachers[1], field.name))
            same_work(teachers[0].work, teachers[1].work, states[0].used, 5)
            same_weights(*models)
            _assert_nested_equal(teachers[0].state.snapshot(), teachers[1].state.snapshot())
            predictions = [model.predict_with_receipt(b"r", state, 4)
                           for model, state in zip(models, states, strict=True)]
            same_prediction(*predictions, states[0].used)
            decisions.extend(len(item.steps) for item in predictions)
        assert new_calls == [states[1].used] * 3
        _assert_nested_equal(before_state, states[1].snapshot())
        assert not any(parameter.requires_grad or parameter.grad is not None
                       for model in models for parameter in model.parameters())
        _direct_executor_save(root, f"held-codec-v1-{mode}-{learning_enabled}.json", dict(
            status="passed", mode=mode, learning_enabled=learning_enabled,
            model_constructions=2, ingestions=2, free_predictions=4,
            free_decisions=decisions, teacher_traversals=2, teacher_offers=10,
            current_codec_evaluation_rows=new_calls, child_actions=0,
            source_sha256={str(Path(__file__).resolve()): hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
                str(root / "src/humanbrain/local_memory_learner.py"): hashlib.sha256(
                    (root / "src/humanbrain/local_memory_learner.py").read_bytes()).hexdigest()},
            reference_sha256=hashlib.sha256(raw).hexdigest()))


def test_held_codec_rechecks_new_calls_after_external_mutation():
    """No held response survives a public call or bypasses changed-input validation."""
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"aa"])
    model.predict_with_receipt(b"q", state, 0)
    before = model.snapshot(state)
    model.codec.bias[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        model.predict_with_receipt(b"q", state, 1)
    model.codec.bias.copy_(before["model"]["codec.bias"])
    state.sensory[0, 0] = 0.5
    with pytest.raises(ValueError):
        model.predict_with_receipt(b"q", state, 1)
    state.sensory.copy_(before["memory"]["sensory"])
    with pytest.raises(ValueError, match="raw bytes"):
        model.teach(b"q", "aa", state, .01)
    _assert_nested_equal(before, model.snapshot(state))


def test_held_codec_nonzero_sensory_free_cursor_response_parity():
    """Force a long free traversal to compare nonzero/EOS gathers with actual evaluations."""
    import dataclasses
    from hashlib import sha256
    from pathlib import Path

    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(seed=733)
    state = model.ingest([b"aa", b""])
    # Authored activity fixture, shared by both response paths; no teaching or task fit.
    model.codec.bias[0] = 8.0
    before = model.snapshot(state)
    actual = model._codec_responses
    positions = []

    def checked(bank, tape, cursors, position, work):
        response = actual(bank, tape, cursors, position, work)
        recomputed = model.codec.evaluate(model.memory._content_checked(tape, cursors))
        for field in dataclasses.fields(response):
            _assert_nested_equal(getattr(recomputed, field.name), getattr(response, field.name))
        positions.append(position)
        return response

    with patch.object(model, "_codec_responses", side_effect=checked):
        prediction = model.predict_with_receipt(b"q", state, 4)
    assert positions == [0, 1, 2, 3, 4]
    assert prediction.generation == ByteGeneration(b"\x00" * 4, "limit")
    assert prediction.work["codec_slot_evaluations"] == 4
    assert prediction.work["codec_response_gathers"] == 16
    _assert_nested_equal(before, model.snapshot(state))
    root = Path(__file__).resolve().parents[1]
    _direct_executor_save(root, "held-codec-v1-nonzero-free-qa.json", dict(
        status="passed", model_constructions=1, ingestions=1, free_predictions=1,
        free_decisions=len(prediction.steps), extra_standalone_codec_evaluations=5,
        extra_standalone_codec_rows=20, teacher_offers=0, child_actions=0,
        source_sha256={str(Path(__file__).resolve()): sha256(
            Path(__file__).read_bytes()).hexdigest(),
            str(root / "src/humanbrain/local_memory_learner.py"): sha256(
                (root / "src/humanbrain/local_memory_learner.py").read_bytes()).hexdigest()}))


_TEMPORARY_MODULE_CASES = (*_SOURCE_BOOTSTRAP_CASES, "callback_lifetime")
_TEMPORARY_MODULE_CALLBACK = '''"""callback lifetime doc"""
import atexit, json, sys, weakref
payload = 'callback-雪'
_module_ref = weakref.ref(sys.modules[__name__])
def _at_exit():
    print(json.dumps(dict(phase='atexit', name=__name__, file=__file__, doc=__doc__,
        payload=payload, module_alive=(_module_ref() is not None), argv0=sys.argv[0],
        main_restored=(sys.modules['__main__'].__dict__ is not globals())), sort_keys=True))
atexit.register(_at_exit)
print('body')
'''


def _temporary_module_normalize_workspace(value, workspace):
    """Only independent workspace roots vary; runtime paths/metadata stay exact."""
    if isinstance(value, str):
        return value.replace(str(workspace), "<workspace>")
    if isinstance(value, list):
        return [_temporary_module_normalize_workspace(item, workspace) for item in value]
    if isinstance(value, dict):
        return {key: _temporary_module_normalize_workspace(item, workspace)
                for key, item in value.items()}
    return value


@pytest.mark.parametrize("case_name", _TEMPORARY_MODULE_CASES)
def test_temporary_module_bootstrap_semantic_pair(tmp_path, case_name):
    """Eight pairs/16 children; no neuronal model, teacher, motor or native call."""
    import json
    import os
    from hashlib import sha256
    from pathlib import Path

    from experiments.r4_integrated import stdlib_python_action_bridge_v1 as old
    from experiments.r4_integrated import temporary_module_action_bridge_v1 as new

    root = Path(__file__).resolve().parents[1]
    qualification = root / ".pytest-tmp/stdlib-python-action-v1-qualification.json"
    qualification_sha = "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c"
    launch = old.load_qualified_launch(
        qualification, qualification_sha, root / ".venv/Scripts/python.exe")
    prefix = f"temporary-module-bootstrap-v1-{case_name}"
    passed_name = f"{prefix}.json"
    raw_names = {route: f"{prefix}-{route}-raw.json" for route in ("old", "new")}
    for name in (passed_name, *raw_names.values()):
        assert not (root / ".pytest-tmp" / name).exists()
    source_paths = (Path(__file__).resolve(), Path(old.__file__).resolve(),
                    Path(new.__file__).resolve())
    sources = {str(path): sha256(path.read_bytes()).hexdigest() for path in source_paths}
    fixture = (_direct_executor_fixture(old, case_name, "unused",
                                       program=_TEMPORARY_MODULE_CALLBACK)
               if case_name == "callback_lifetime" else _source_bootstrap_fixture(old, case_name))
    case, files, before = fixture
    index = _TEMPORARY_MODULE_CASES.index(case_name)
    order = ("old", "new") if index % 2 == 0 else ("new", "old")
    ambient = dict(os.environ)
    receipts, raw_paths = {}, {}
    for route in order:
        bridge = old if route == "old" else new
        receipt = bridge.execute_utf8(case, files, before, tmp_path / route, launch=launch)
        receipts[route] = receipt
        raw_paths[route] = _direct_executor_save(root, raw_names[route], dict(
            schema="temporary-module-bootstrap-raw-v1", status="pending_semantic_checks",
            case=case_name, case_index=index, route=route, execution_order=list(order),
            launch_record=launch.to_record(), runtime_qualification_sha256=qualification_sha,
            source_sha256=sources, fixture_files={item.path: item.content.hex() for item in files},
            receipt=receipt, child_executions=1))

    assert dict(os.environ) == ambient
    assert old.load_qualified_launch(
        qualification, qualification_sha, root / ".venv/Scripts/python.exe") == launch
    assert sources == {str(path): sha256(path.read_bytes()).hexdigest() for path in source_paths}
    for route, receipt in receipts.items():
        bridge = old if route == "old" else new
        assert receipt["launch_record_sha256"] == launch.sha256
        assert receipt["bootstrap_policy"] == bridge.BOOTSTRAP_POLICY
        assert receipt["bootstrap_sha256"] == bridge.BOOTSTRAP_SHA256
        assert receipt["child_command"] == [
            launch.base_executable, "-I", "-S", "-X", "utf8", "-B", "-c", bridge.BOOTSTRAP]
        assert receipt["child_python_flags"] == ["-I", "-S", "-X", "utf8", "-B"]
        assert receipt["child_environment_override"] == {}
        assert receipt["child_environment_removed"] == ["__PYVENV_LAUNCHER__"]
        assert receipt["decoys_unchanged"] and receipt["materialized_matches_action"]
        assert {item.path: (tmp_path / route / item.path).read_bytes() for item in files} == {
            item.path: item.content for item in files}
        assert {path.name for path in (tmp_path / route).iterdir()} == {
            item.path for item in files}

    if case_name == "namespace_cleanup":
        bodies, exits = {}, {}
        for route, receipt in receipts.items():
            assert receipt["returncode"] == 0 and receipt["stderr_hex"] == ""
            assert receipt["decode_errors"] == {}
            body, exit_record = map(json.loads, receipt["stdout"].splitlines())
            _stdlib_check_base_identity(body, launch, tmp_path / route)
            bodies[route] = _temporary_module_normalize_workspace(body, tmp_path / route)
            exits[route] = exit_record
        assert bodies["old"] == bodies["new"]
        body = bodies["new"]
        assert body["initial_globals"] == sorted([
            "__name__", "__file__", "__cached__", "__doc__", "__loader__", "__package__",
            "__spec__", "__builtins__"])
        assert body["main_is_globals"] is True and body["argv"] == ["main.py"]
        assert body["name"] == "__main__" and body["file"] == "main.py"
        assert body["package"] == "" and body["sibling"] == "ab"
        assert body["doc"] == "authored module doc"
        assert body["spec"] is body["loader"] is body["cached"] is None
        assert body["flags"]["no_site"] == 1 and body["user_site"] is None
        assert exits["old"] == exits["new"] == dict(
            phase="atexit", argv0="-c", main_restored=True, retained_name="__main__",
            retained_file="main.py", retained_doc="authored module doc")
    elif case_name == "callback_lifetime":
        for receipt in receipts.values():
            assert receipt["returncode"] == 0 and receipt["stderr_hex"] == ""
            assert receipt["decode_errors"] == {}
            body, callback = receipt["stdout"].splitlines()
            assert body == "body"
            assert json.loads(callback) == dict(
                phase="atexit", name="__main__", file="main.py", doc="callback lifetime doc",
                payload="callback-雪", module_alive=False, argv0="-c", main_restored=True)
    elif case_name in {"syntax_error", "runtime_error"}:
        for receipt in receipts.values():
            assert receipt["returncode"] == 1 and receipt["decode_errors"] == {}
            assert not receipt["passed"]
            if case_name == "syntax_error":
                assert receipt["stdout_hex"] == ""
                assert 'File "main.py", line 1' in receipt["stderr"]
                assert "if True print('bad')" in receipt["stderr"]
                assert receipt["stderr"].splitlines()[-1] == "SyntaxError: invalid syntax"
            else:
                assert receipt["stdout"] == "runtime-exit:-c\n"
                assert 'File "main.py", line 3, in <module>' in receipt["stderr"]
                assert receipt["stderr"].splitlines()[-1] == "ValueError: authored runtime failure"
    else:
        stdout = b"\xff\n" if case_name == "exit_invalid_utf8" else (
            b"pending\n" if case_name == "pending" else (case.value + "\n").encode())
        stderr = b"exit-hook\n" if case_name == "exit_invalid_utf8" else b""
        for receipt in receipts.values():
            assert receipt["stdout_hex"] == stdout.hex() and receipt["stderr_hex"] == stderr.hex()
            assert receipt["returncode"] == (7 if case_name == "exit_invalid_utf8" else 0)
            assert receipt["passed"] is (case_name in {"utf8_bom_crlf", "latin1_cookie"})
            assert receipt["stdout"] == (
                None if case_name == "exit_invalid_utf8" else stdout.decode())
            assert set(receipt["decode_errors"]) == (
                {"stdout"} if case_name == "exit_invalid_utf8" else set())
    _direct_executor_save(root, passed_name, dict(
        schema="temporary-module-bootstrap-case-qa-v1", status="passed", case=case_name,
        case_index=index, child_executions=2, execution_order=list(order),
        launch_record=launch.to_record(), runtime_qualification_sha256=qualification_sha,
        source_sha256=sources, bootstrap_policy=new.BOOTSTRAP_POLICY,
        bootstrap_sha256=new.BOOTSTRAP_SHA256,
        raw_evidence={str(path): sha256(path.read_bytes()).hexdigest()
                      for path in raw_paths.values()},
        declared_differences=["bootstrap_wrapper_frames", "import_and_audit_events"]))


def _splice_qa_scalar_product(left, right):
    """Authored scalar forward coincidence oracle, with no learned weights."""
    import numpy as np

    return np.array([[sum(float(a) * float(b) for a, b in zip(row, other, strict=True))
                      for other in right] for row in left], dtype=np.float64)


def _splice_qa_scalar_population(raw, codes, rows, *, reverse=False, eos=True):
    import numpy as np

    result = np.zeros((rows, 32), dtype=np.float64)
    symbols = ([256, *reversed(raw)] if reverse else [*raw, *([256] if eos else [])])
    assert len(symbols) <= rows
    for index, symbol in enumerate(symbols):
        result[index] = codes[symbol]
    return result


def _splice_qa_scalar_context(raw, index, codes):
    import numpy as np

    result = np.zeros((32, 32), dtype=np.float64)
    for lag in range(16):
        if index - lag - 1 >= 0:
            result[lag] = codes[raw[index - lag - 1]]
        right = index + lag
        if right < len(raw):
            result[16 + lag] = codes[raw[right]]
        elif right == len(raw):
            result[16 + lag] = codes[256]
    return result


def _splice_qa_normalized(*blocks):
    import math

    import numpy as np

    raw = np.concatenate([np.asarray(block).reshape(-1) for block in blocks])
    return raw * (math.sqrt(32) / max(math.sqrt(sum(float(x) ** 2 for x in raw)), 1e-8))


def test_splice_three_leaf_updates_independent_local_oracle():
    """One action circuit, three direct labels; no reader, proposal or execution."""
    import math

    import numpy as np

    from humanbrain.neuronal_splice_learner import SpliceActionCircuit

    circuit = SpliceActionCircuit(seed=733)
    assert all(not value.requires_grad and value.grad is None for value in circuit.parameters())
    for head, width, target in (("start", 6177, 2), ("end", 2049, 1),
                                ("operation", 4244, 1)):
        rows = 1 if head == "operation" else 3
        features = torch.zeros((rows, width), dtype=torch.float32)
        for index in range(rows):
            features[index, index] = 4
            features[index, width - index - 1] = -4
        if rows > 1:
            features[1].copy_(features[0])
        weight = getattr(circuit, f"{head}_weight")
        original = weight.detach().numpy().copy()
        bias = circuit.operation_bias.detach().numpy().copy() if head == "operation" else None
        phi = features.numpy().astype(np.float64)
        if head == "operation":
            logits = np.array([sum(float(original[k, j]) * float(phi[0, j])
                                   for j in range(width)) + float(bias[k]) for k in range(2)])
        else:
            logits = np.array([sum(float(original[j]) * float(row[j])
                                   for j in range(width)) for row in phi])
        exponentials = [math.exp(float(item - max(logits))) for item in logits]
        probabilities = np.array([item / sum(exponentials) for item in exponentials])
        error = probabilities.copy()
        error[target] -= 1
        if head == "operation":
            gradient = np.array([[float(error[k]) * float(phi[0, j])
                                  for j in range(width)] for k in range(2)])
        else:
            gradient = np.array([sum(float(error[k]) * float(phi[k, j])
                                     for k in range(rows)) for j in range(width)])
        expected = original - np.float32(.01) * gradient.astype(np.float32)
        update = circuit.prepare_update(head, features, target)
        if rows > 1:
            assert update.decision.logits[0] == update.decision.logits[1]
        np.testing.assert_allclose(update.decision.logits.numpy().reshape(-1), logits,
                                   rtol=2e-6, atol=2e-7)
        np.testing.assert_allclose(update.decision.probabilities.numpy().reshape(-1),
                                   probabilities, rtol=2e-6, atol=2e-7)
        assert update.target == target
        assert float(update.nll) == pytest.approx(-math.log(probabilities[target]), rel=2e-6)
        assert update.parameters[0][0] is weight
        np.testing.assert_allclose(update.parameters[0][1].numpy(), expected,
                                   rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(weight.numpy(), original)
        if head == "operation":
            assert update.parameters[1][0] is circuit.operation_bias
            np.testing.assert_allclose(update.parameters[1][1].numpy(),
                                       bias - np.float32(.01) * error.astype(np.float32),
                                       rtol=2e-6, atol=2e-7)
            np.testing.assert_array_equal(circuit.operation_bias.numpy(), bias)


def _splice_qa_workspace(files=None):
    from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
    from humanbrain.neuronal_splice_action import SpliceWorkspace

    if files is None:
        files = (("main.py", b"import a\nprint(a.x)\n"),
                 ("a.py", b"x = 'pending'\n"), ("b.py", b"x = 'pending'\n"))
    return SpliceWorkspace(tuple(OpaqueWorkspaceFile(path, raw) for path, raw in files))


def _splice_qa_codes():
    # Authored bounded nonlexical activity, not a learner or a random draw.
    values = torch.arange(258 * 32, dtype=torch.float32).reshape(258, 32)
    return ((values * 7 + torch.arange(32).square()) % 43 - 21) / 23


def test_splice_raw_abi_bounds_and_stale_workspace():
    """Literal transport only: three byte edits/five rejections, zero neural calls."""
    from humanbrain.neuronal_splice_action import SpliceProposal, apply_splice

    workspace = _splice_qa_workspace((("a.py", b"A'B\xffC"), ("b.py", b"untouched")))
    original = workspace.files
    for start, end, raw, expected in (
        (1, 1, b"\x00", b"A\x00'B\xffC"),
        (1, 4, b"", b"AC"),
        (1, 4, b"'\\\xfe", b"A'\\\xfeC"),
    ):
        proposal = SpliceProposal("SPLICE", "a.py", start, end, raw, "eos", workspace.sha256)
        result = apply_splice(workspace, proposal)
        assert result[0].content == expected and result[1] == original[1]
        assert workspace.files == original
    invalid = (
        ("a.py", 0, 1, "eos", "0" * 64),
        ("missing.py", 0, 1, "eos", workspace.sha256),
        ("a.py", 3, 2, "eos", workspace.sha256),
        ("a.py", 0, 6, "eos", workspace.sha256),
        ("a.py", 0, 1, "limit", workspace.sha256),
    )
    for path, start, end, status, digest in invalid:
        with pytest.raises(ValueError):
            apply_splice(workspace, SpliceProposal("SPLICE", path, start, end,
                                                   b"ab", status, digest))
        assert workspace.files == original
    stopped = SpliceProposal("STOP", "missing.py", -1, -2, b"ab", "limit", workspace.sha256)
    assert apply_splice(workspace, stopped) is workspace.files
    with pytest.raises(ValueError):
        _splice_qa_workspace((("a.py", b"x" * 129),))
    with pytest.raises(ValueError):
        _splice_qa_workspace((("a" * 32, b""),))
    with pytest.raises(ValueError):
        _splice_qa_workspace(tuple((f"{i}.py", b"") for i in range(5)))
    with pytest.raises(ValueError):
        _splice_qa_workspace((("a.py", b""), ("A.py", b"")))


def test_splice_features_independent_scalar_oracle():
    """Three head banks from authored byte activity; no model or target selection."""
    import numpy as np

    from humanbrain import neuronal_splice_learner as splice

    codes = _splice_qa_codes()
    raw_codes = codes.numpy()
    workspace = _splice_qa_workspace((("z.py", b"\xff\x00A"), ("a.py", b"B'")))
    request = b"set z.py"
    encoding = splice.encode(request, workspace, codes)
    actual_start = splice.start_features(encoding)
    q = _splice_qa_scalar_population(request, raw_codes, 32, reverse=True)
    for path, start in (("a.py", 0), ("z.py", 3)):
        raw = next(item.content for item in workspace.files if item.path == path)
        p = _splice_qa_scalar_population(path.encode(), raw_codes, 32, reverse=True)
        f = _splice_qa_scalar_population(raw, raw_codes, 129)
        c = _splice_qa_scalar_context(raw, start, raw_codes)
        expected = _splice_qa_normalized(_splice_qa_scalar_product(q, p),
                                         _splice_qa_scalar_product(q, f),
                                         _splice_qa_scalar_product(q, c), [start / 128])
        index = workspace.boundaries.index((path, start))
        np.testing.assert_allclose(actual_start[index].numpy(), expected,
                                   rtol=4e-5, atol=3e-6)
    raw = b"\xff\x00A"
    start, end = 1, 3
    c_start = _splice_qa_scalar_context(raw, start, raw_codes)
    offsets, actual_end = splice.end_features(encoding, "z.py", start)
    assert offsets == (1, 2, 3)
    for index, offset in enumerate(offsets):
        c_end = _splice_qa_scalar_context(raw, offset, raw_codes)
        expected = _splice_qa_normalized(_splice_qa_scalar_product(q, c_end),
                                         _splice_qa_scalar_product(c_start, c_end),
                                         [(offset - start) / 128])
        np.testing.assert_allclose(actual_end[index].numpy(), expected,
                                   rtol=4e-5, atol=3e-6)
    payload = b"'\xff"
    actual_op = splice.operation_features(encoding, "z.py", start, end, payload, "eos")
    p = _splice_qa_scalar_population(b"z.py", raw_codes, 32, reverse=True)
    span = _splice_qa_scalar_population(raw[start:end], raw_codes, 129)
    emitted = _splice_qa_scalar_population(payload, raw_codes, 17)
    expected = _splice_qa_normalized(_splice_qa_scalar_product(q, c_start),
                                     _splice_qa_scalar_product(q, p),
                                     _splice_qa_scalar_product(span, emitted),
                                     [(end - start) / 128, len(payload) / 16, 1])
    np.testing.assert_allclose(actual_op[0].numpy(), expected, rtol=4e-5, atol=3e-6)
    for features, width in ((actual_start, 6177), (actual_end, 2049), (actual_op, 4244)):
        assert features.shape[1] == width and features.dtype == torch.float32
        assert torch.isfinite(features).all()
        assert (features.double().square().sum(1) <= 32 + 2e-5).all()
    with pytest.raises(ValueError):
        splice.encode(b"q" * 32, workspace, codes)


def test_splice_all_boundaries_and_file_order_equivariance():
    """Four pure workspace banks, including all 516 capacity boundaries."""
    from humanbrain import neuronal_splice_learner as splice

    codes = _splice_qa_codes()
    request = b"set a.py"
    base = (("b.py", b"ab"), ("a.py", b"ab"), ("empty.py", b""))
    workspace = _splice_qa_workspace(base)
    assert workspace.boundaries == (("a.py", 0), ("a.py", 1), ("a.py", 2),
                                    ("b.py", 0), ("b.py", 1), ("b.py", 2),
                                    ("empty.py", 0))
    features = splice.start_features(splice.encode(request, workspace, codes))
    reordered = _splice_qa_workspace(tuple(reversed(base)))
    assert reordered.sha256 == workspace.sha256
    other = splice.start_features(splice.encode(request, reordered, codes))
    assert torch.equal(features, other)
    # Identical local content in different files still receives raw path evidence.
    assert not torch.equal(features[0], features[3])
    moved = _splice_qa_workspace((("b.py", b"ab"), ("a.py", b"Xab"), ("empty.py", b"")))
    moved_features = splice.start_features(splice.encode(request, moved, codes))
    assert len(moved_features) == len(features) + 1
    assert not torch.equal(features[1], moved_features[2])
    assert moved.boundaries[moved_features.shape[0] - 1] == ("empty.py", 0)
    del other, moved_features, features
    capacity = _splice_qa_workspace(tuple((f"{index}.py", bytes(
        (index * 128 + offset) % 256 for offset in range(128))) for index in range(4)))
    assert len(capacity.boundaries) == 516
    assert len(set(capacity.boundaries)) == 516
    maximum = splice.start_features(splice.encode(request, capacity, codes))
    assert maximum.shape == (516, 6177) and torch.isfinite(maximum).all()
    assert (maximum.double().square().sum(1) <= 32 + 2e-5).all()


def _splice_qa_model(mode="learned"):
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner

    model = NeuronalSpliceLearner(mode=mode, seed=733)
    return model, model.ingest([b"ab."])


def _splice_qa_outcome(prediction, workspace, execution=None):
    """Default is an explicitly simulated outcome for nonexecution mechanics QA."""
    return {"attempt_id": prediction.attempt_id,
            "proposal_sha256": prediction.proposal_sha256,
            "workspace_sha256": workspace.sha256,
            "execution": execution if execution is not None else {
                "returncode": 0, "stdout_hex": "", "stderr_hex": ""}}


def _splice_qa_teacher(workspace):
    from humanbrain.neuronal_splice_action import SpliceProposal

    # Authored corrective label only. Never called to select a free proposal.
    return SpliceProposal("SPLICE", "a.py", 5, 12, b"ab", "eos", workspace.sha256)


def test_splice_same_instance_execution_before_correction(tmp_path):
    """One actual child, one same-instance correction; no motor or solved edit."""
    import json
    from hashlib import sha256
    from pathlib import Path
    from time import perf_counter

    from experiments.r4_integrated import temporary_module_action_bridge_v1 as bridge
    from humanbrain.neuronal_splice_action import apply_splice

    root = Path(__file__).resolve().parents[1]
    raw_path = root / ".pytest-tmp/neuronal-splice-e2e-v1-raw.json"
    passed_path = root / ".pytest-tmp/neuronal-splice-e2e-v1.json"
    assert not raw_path.exists() and not passed_path.exists()
    runtime = root / ".pytest-tmp/stdlib-python-action-v1-qualification.json"
    bootstrap = root / ".pytest-tmp/temporary-module-bootstrap-v1-qualification.json"
    launch = bridge.load_qualified_bootstrap(
        bootstrap, "0bb809fd33aee145f169776a03a884fd62daf381ff4e67ed333a2a378c46d92b",
        runtime_qualification_path=runtime,
        runtime_qualification_sha="4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c",
        expected_venv=root / ".venv/Scripts/python.exe")
    source_paths = (Path(__file__).resolve(), root / "src/humanbrain/neuronal_splice_learner.py",
                    root / "src/humanbrain/neuronal_splice_action.py",
                    root / "src/humanbrain/local_memory_learner.py", Path(bridge.__file__))
    source_hashes = {str(path): sha256(path.read_bytes()).hexdigest() for path in source_paths}
    model, state = _splice_qa_model()
    workspace = _splice_qa_workspace()
    original = copy.deepcopy(model.state_dict())
    state_before = state.snapshot()
    prediction = model.propose(b"set a.py k00", workspace, state)
    rejected = None
    try:
        after = apply_splice(workspace, prediction.proposal)
    except ValueError as error:
        rejected, after = str(error), workspace.files
    # The old bridge is only process transport: neither Case.files nor actuate is called.
    # Its expected-output field is deliberately empty; task grading happens below.
    case = bridge.Case("splice-qa", "unused", "", "__unselected__", "")
    execution = bridge.execute_utf8(
        case, after, workspace.files, tmp_path / "action", launch=launch)
    completed_at = perf_counter()
    outcome = _splice_qa_outcome(prediction, workspace, execution)
    record = {"schema": "neuronal-splice-e2e-raw-v1", "status": "pending_checks",
              "model_instance": id(model), "state_instance": id(state),
              "proposal": prediction.proposal.to_record(), "outcome": outcome,
              "before": {item.path: item.content.hex() for item in workspace.files},
              "after": {item.path: item.content.hex() for item in after},
              "rejection": rejected, "action_completed_perf": completed_at,
              "child_executions": 1, "source_sha256": source_hashes}
    with raw_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
    _assert_nested_equal(model.state_dict(), original)
    _assert_nested_equal(state.snapshot(), state_before)
    assert execution["returncode"] is None or type(execution["returncode"]) is int
    assert execution["bootstrap_sha256"] == bridge.BOOTSTRAP_SHA256
    assert execution["child_command"] == [launch.base_executable, *bridge.FLAGS,
                                           "-c", bridge.BOOTSTRAP]
    assert {item.path: (tmp_path / "action" / item.path).read_bytes() for item in after} == {
        item.path: item.content for item in after}
    teacher = _splice_qa_teacher(workspace)
    task_success = (rejected is None and execution["returncode"] == 0
                    and execution["stdout_hex"] == b"ab\n".hex()
                    and execution["stderr_hex"] == "" and after == apply_splice(workspace, teacher))
    actual_teach, calls = model.reader.teach, []

    def observed_teach(request, payload, tape, *args, **kwargs):
        saved = json.loads(raw_path.read_bytes())
        assert saved["outcome"] == outcome and saved["model_instance"] == id(model)
        assert tape is state and request == b"set a.py k00" and payload == b"ab"
        assert perf_counter() >= completed_at
        _assert_nested_equal(model.state_dict(), original)
        calls.append(1)
        return actual_teach(request, payload, tape, *args, **kwargs)

    teacher_started = perf_counter()
    with patch.object(model.reader, "teach", side_effect=observed_teach):
        feedback = model.teach(prediction, workspace, state, teacher, outcome)
    assert calls == [1] and feedback.reader_feedback.teacher_count == 3
    assert set(feedback.teacher_choices) == {"start", "end", "operation"}
    assert model._pending is None
    _assert_nested_equal(state.snapshot(), state_before)
    assert source_hashes == {
        str(path): sha256(path.read_bytes()).hexdigest() for path in source_paths}
    with passed_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": "neuronal-splice-e2e-qa-v1", "status": "passed",
            "raw_sha256": sha256(raw_path.read_bytes()).hexdigest(), "source_sha256": source_hashes,
            "task_success": task_success, "child_executions": 1, "free_proposals": 1,
            "free_payload_decisions": len(prediction.payload.steps), "payload_teacher_offers": 3,
            "action_teacher_offers": 3, "action_completed_perf": completed_at,
            "teacher_started_perf": teacher_started, "teacher_completed_perf": perf_counter()},
            sort_keys=True, allow_nan=False) + "\n")


def test_splice_online_receipt_maximum_contract(tmp_path):
    """Authored activity numbers test serialization; no neural model is run."""
    import json
    from types import SimpleNamespace as Value

    from experiments.r4_integrated import neuronal_splice_receipts_v1 as receipts
    from experiments.r4_integrated.neuronal_splice_io_v1 import JsonlJournal, read_row
    from experiments.r4_integrated.neuronal_splice_online_v1 import SPEC

    successor = list(range(1, 128)) + [127]
    state = {"write_index": 128, "successor": successor}
    belief = [-math.log(128)] * 128
    steps, cursors = [], list(range(128))
    for index in range(17):
        token = 65 if index < 16 else 256
        probabilities = [0.0] * 257
        probabilities[token] = 1.0
        steps.append(Value(output_step=index, observed_token=token, nll=0.0,
                           mixture_probabilities=probabilities,
                           prior_log_belief=list(belief), posterior_log_belief=list(belief),
                           branch_cursors=list(cursors)))
        cursors = [successor[cursor] for cursor in cursors]
    common = dict(steps=steps, work={}, initial_log_belief=belief, final_log_belief=belief)
    free = Value(**common, generation=Value(data=b"A" * 16, stop_reason="eos"))
    teacher = Value(**common, teacher_count=17, total_loss=0.0, mean_loss=0.0,
                    address_error=[0.0] * 128, codec_updated=True, address_updated=False)
    compact_free = receipts.compact_reader(free, state)
    compact_teacher = receipts.compact_reader(teacher, state)
    heads = {name: {"logits": [0.0] * count, "selected_index": 0}
             for name, count in (("start", 516), ("end", 129), ("operation", 2))}
    for name, head in heads.items():
        receipts.validate_head(head, name)
    numbers = sum(len(item["initial_log_belief"]) + sum(
        len(step["mixture"]) + len(step["posterior"]) for step in item["steps"])
        for item in (compact_free, compact_teacher)) + 2 * sum(len(head["logits"]) for head in heads.values())
    assert numbers == 14640
    # Reserve the complete named vectors plus the declared maximum free text,
    # full counter dictionaries and host/executable/bootstrap/output fields.
    row = {"free": compact_free, "teacher": compact_teacher,
           "free_heads": heads, "teacher_heads": copy.deepcopy(heads),
           "metadata": "x" * 32768, "host_path": "p" * 1024,
           "executable_path": "e" * 1024, "bootstrap": "b" * 4096,
           "stdout_hex": "ff" * 16384, "stderr_hex": "ff" * 16384,
           "diagnostics": ["d" * 1024, "e" * 1024],
           "ids": ["i" * 128 for _ in range(12)], "hashes": ["a" * 64 for _ in range(16)],
           "work": [{f"counter_{i:03d}".ljust(96, "x"): 10**12 for i in range(160)}
                    for _ in range(4)]}
    encoded = receipts.bounded_record(row)
    assert len(encoded) <= 2**20 and receipts.tree_bytes(row) <= 2**20
    path = tmp_path / "maximum.jsonl"
    journal = JsonlJournal(path, SPEC)
    line_hash = journal.append(row)
    summary = journal.close()
    with path.open("rb") as stream:
        assert read_row(stream, SPEC) == row
        assert read_row(stream, SPEC) is None
    assert summary["sha256"] == line_hash
    bad = copy.deepcopy(free)
    bad.steps[1].prior_log_belief[0] += .1
    with pytest.raises(ValueError, match="omitted prior"):
        receipts.compact_reader(bad, state)
    bad = copy.deepcopy(free)
    bad.steps[1].branch_cursors[0] = 0
    with pytest.raises(ValueError, match="omitted cursor"):
        receipts.compact_reader(bad, state)
    with pytest.raises(ValueError, match="tree limit"):
        receipts.bounded_record({"over": "x" * 2**20})
    (tmp_path / "maximum-summary.json").write_text(json.dumps({
        "authored_vector_numbers": numbers, "line_bytes": len(encoded),
        "tree_bytes": receipts.tree_bytes(row), "models": 0, "children": 0,
        "journal": summary}, sort_keys=True) + "\n", encoding="utf-8")


def test_splice_online_binary_protocol_contract():
    """Four in-memory messages exercise transport and reply validation only."""
    from experiments.r4_integrated.neuronal_splice_endpoint_v1 import check_acknowledgement
    from experiments.r4_integrated.neuronal_splice_io_v1 import read_row, write_message
    from experiments.r4_integrated.neuronal_splice_online_v1 import SPEC
    from experiments.r4_integrated.neuronal_splice_worker_v1 import IPC_SCHEMA, check_executed

    identity = dict(schema=IPC_SCHEMA, type="proposal", arm="learned", index=0,
                    id="P0-0", attempt_id="authored:1")
    outcome = {"attempt_id": identity["attempt_id"], "proposal_sha256": "b" * 64,
               "workspace_sha256": "c" * 64,
               "execution": {"returncode": 1, "stdout_hex": "", "stderr_hex": "657272"}}
    executed = {**identity, "type": "executed", "action_sha256": "a" * 64, "outcome": outcome}
    acknowledged = {**identity, "type": "acknowledged", "action_sha256": "a" * 64,
                    "model_sha256": "d" * 64, "memory_sha256": "e" * 64,
                    "work": {"action": {}, "reader": {}}, "model_seconds": .01}
    messages = [identity, executed, acknowledged, {**identity, "index": 1, "attempt_id": "authored:2"}]
    pipe = io.BytesIO()
    for message in messages:
        write_message(pipe, message, SPEC)
    pipe.seek(0)
    assert [read_row(pipe, SPEC) for _ in range(4)] == messages
    assert read_row(pipe, SPEC) is None
    assert check_executed(executed, identity) == outcome
    check_acknowledgement(acknowledged, identity, "a" * 64, "d" * 64, "e" * 64)
    for field in ("teacher", "grade", "passed"):
        with pytest.raises(ValueError):
            check_executed({**executed, field: True}, identity)
        bad = copy.deepcopy(executed)
        bad["outcome"]["execution"][field] = True
        with pytest.raises(ValueError):
            check_executed(bad, identity)
    for field, value in (("arm", "slow_frozen"), ("index", 1), ("attempt_id", "another")):
        with pytest.raises(ValueError):
            check_executed({**executed, field: value}, identity)
    with pytest.raises(ValueError):
        check_acknowledgement({**acknowledged, "action_sha256": "f" * 64},
                              identity, "a" * 64, "d" * 64, "e" * 64)
    for raw in (b"{}", b"x" * (2**20 + 1), b"not-json\n"):
        with pytest.raises(ValueError):
            read_row(io.BytesIO(raw), SPEC)


def test_splice_online_saved_audit_arithmetic():
    """Authored grades and timing numbers, never an audit of real neural output."""
    from hashlib import sha256

    from experiments.r4_integrated import check_neuronal_splice_online_v1 as audit

    phases = {
        "regular": {"rows": 4, "model_seconds": .04, "model_max_seconds": .01,
                    "action_seconds": .08, "action_max_seconds": .02,
                    "journal_seconds": .004, "journal_max_seconds": .001},
        "cap": {"rows": 1, "model_seconds": .03, "model_max_seconds": .03,
                "action_seconds": .015, "action_max_seconds": .015,
                "journal_seconds": .002, "journal_max_seconds": .002}}
    report = {"mode": "cost", "status": "completed",
              "timing_summary": {arm: copy.deepcopy(phases) for arm in audit.ARMS},
              "category_probes": [{"action_seconds": .025}] * 4,
              "memory": {"peak_working_set_mib": 300},
              "worker": {"worker": {"memory": {"peak_working_set_mib": 200}}}}
    outer = {"elapsed_seconds": 20., "observed_tree_peak_sum_mib": 550.}
    preliminary = audit.project_cost(report, outer)
    assert preliminary["projected_run_seconds"] == pytest.approx(185.744)
    assert preliminary["run_projected_peak_mib"] == 690
    assert preliminary["audit_peak_pending"] and not preliminary["admissible"]
    finished = {"status": "passed", "pid": 7, "learning_pair_timing": {
        arm: {"regular": {"rows": 4, "seconds": .004, "max_seconds": .001},
              "cap": {"rows": 1, "seconds": .002, "max_seconds": .002}}
        for arm in audit.ARMS}}
    audit_outer = {"status": "passed", "exit_code": 0, "seconds_limit": 900,
                   "tree_peak_sum_mib_limit": 2048, "elapsed_seconds": 5.,
                   "observed_tree_peak_sum_mib": 250., "started_unix": 100.,
                   "deadline_unix": 1000., "child_pid": 7,
                   "observed_process_peaks_mib": {f"7/{621355968000000000 + 1010000000}": 250.}}
    projected = audit.project_cost(report, outer, audit=finished, audit_outer=audit_outer)
    assert projected["projected_audit_seconds"] == pytest.approx(14.064)
    assert projected["projected_seconds"] == pytest.approx(199.808)
    assert projected["admissible"] and projected["projected_tree_mib"] == 690
    costly = copy.deepcopy(report)
    costly["category_probes"][0]["action_seconds"] = 1.
    assert not audit.project_cost(costly, outer, audit=finished, audit_outer=audit_outer)["admissible"]
    with pytest.raises(ValueError, match="unfinished"):
        audit.project_cost(report, outer, audit=finished)
    invalid = copy.deepcopy(report)
    invalid["timing_summary"]["learned"]["regular"]["rows"] = 3
    with pytest.raises(ValueError, match="incomplete"):
        audit.project_cost(invalid, outer)

    files = [{"path": "a.py", "content_hex": b"abc".hex()}]
    teacher = {"operation": "SPLICE", "path": "a.py", "start": 1, "end": 2,
               "replacement_hex": b"X".hex(), "stop_reason": "eos",
               "workspace_sha256": audit.workspace_digest(files)}
    expected, rejected = audit.apply_saved(files, teacher)
    assert rejected is None
    task = {"files": files, "teacher": teacher, "required_operation": "SPLICE",
            "expected_after_files": expected, "expected_stdout_hex": b"done\n".hex()}
    execution = {"returncode": 0, "stdout_hex": b"done\n".hex(), "stderr_hex": "",
                 "stdout_sha256": sha256(b"done\n").hexdigest(), "stderr_sha256": sha256(b"").hexdigest(),
                 "materialized_matches_action": True, "decode_errors": {}}
    alternative = {**teacher, "start": 0, "end": 3, "replacement_hex": b"aXc".hex()}
    assert audit.task_grade(task, alternative, expected, None, execution) == {
        "task_success": True, "canonical_tuple_correct": False}
    stopped = {**teacher, "operation": "STOP"}
    unchanged_set = {**task, "expected_after_files": files}
    assert not audit.task_grade(unchanged_set, stopped, files, None, execution)["task_success"]
    limited = {**teacher, "stop_reason": "limit"}
    after, rejection = audit.apply_saved(files, limited)
    assert after == files and rejection == "missing_eos"
    assert not audit.task_grade(task, limited, after, rejection, execution)["task_success"]


def test_splice_online_prior_worker_exchange():
    """One actual teacher-free IPC action; its success is not a task-skill claim."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path

    from experiments.r4_integrated.check_neuronal_splice_online_v1 import initial_action_tensors
    from experiments.r4_integrated.neuronal_splice_endpoint_v1 import (
        check_acknowledgement,
        check_proposal,
    )
    from experiments.r4_integrated.neuronal_splice_io_v1 import (
        JsonlJournal,
        process_identity,
        read_row,
        write_message,
    )
    from experiments.r4_integrated.neuronal_splice_online_v1 import (
        SPEC,
        execute_action,
        qualified_launch,
        workspace_from_records,
    )
    from experiments.r4_integrated.neuronal_splice_prior_v1 import initialize_models, tensor_bytes
    from experiments.r4_integrated.neuronal_splice_receipts_v1 import validate_prediction
    from experiments.r4_integrated.neuronal_splice_worker_v1 import (
        LIMITS,
        SOURCE_MODULES,
        WORKER_SOURCE,
        check_requests,
        file_hash,
        write_json,
    )
    from humanbrain.neuronal_splice_learner import _digest

    root = Path(__file__).resolve().parents[1]
    output = root / ".pytest-tmp/neuronal-splice-prior-worker-qa-v1"
    output.mkdir(exist_ok=False)
    deadline = float(os.environ["HUMANBRAIN_RUN_DEADLINE_UNIX"])
    assert 0 < deadline - time.time() <= 60
    models, custody = initialize_models(root)
    write_json(output / "prior.json", custody)
    initial_actions = initial_action_tensors(torch)
    for model in models.values():
        assert all(torch.equal(model.state_dict()[key], value) for key, value in initial_actions.items())
        assert all(torch.equal(value, models["learned"].reader.state_dict()[key])
                   for key, value in model.reader.state_dict().items() if isinstance(value, torch.Tensor))
    model = models["learned"]
    document = b"Record k00 has value ab."
    state = model.ingest([document])
    model_hash, memory_hash = _digest(model.state_dict()), _digest(state.snapshot())
    workspace = workspace_from_records([
        {"path": "a.py", "content_hex": b"x = 'old'\n".hex()},
        {"path": "main.py", "content_hex": b"import a\nprint(a.x)\n".hex()}])
    task = {"task_id": "endpoint-0", "memory_id": "qa-memory", "request_hex": b"SET a.py x k00".hex(),
            "workspace_sha256": workspace.sha256,
            "files": [{"path": item.path, "content_hex": item.content.hex()} for item in workspace.files]}
    package = output / "package"
    package.mkdir()
    sources = {}
    for relative in (*SOURCE_MODULES.values(), WORKER_SOURCE):
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
        assert file_hash(destination) == file_hash(root / relative)
        sources[relative] = file_hash(destination)
    checkpoint = output / "source_free.pt"
    payload = {"schema": "neuronal-splice-single-exchange-qa-state-v1", "arms": {"learned": {
        "weights": model.state_dict(), "memories": {"qa-memory": state.snapshot()}, "experience": 0}}}
    assert tensor_bytes(payload) == 1291336
    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)
    del payload
    checkpoint_hash = file_hash(checkpoint)
    requests = {"schema": "neuronal-splice-single-exchange-qa-request-v1",
                "checkpoint_sha256": checkpoint_hash, "source_sha256": sources,
                "receipt_limits": LIMITS, "arms": ["learned"], "reading_cases": [],
                "cases": [{"id": task["task_id"], **{key: task[key] for key in ("memory_id", "request_hex", "files")}}]}
    check_requests(requests, single_exchange_qa=True)
    with pytest.raises(ValueError):
        check_requests(requests)
    request_path = output / "query_only.json"
    write_json(request_path, requests)
    requests_hash = file_hash(request_path)
    launch = qualified_launch(root)
    command = [sys.executable, "-I", "-X", "utf8", str(package / WORKER_SOURCE),
               "--package-root", str(package), "--checkpoint", str(checkpoint),
               "--checkpoint-sha", checkpoint_hash, "--requests", str(request_path),
               "--requests-sha", requests_hash, "--output", str(output / "worker"),
               "--deadline-unix", repr(deadline), "--single-exchange-qa"]
    action_journal = JsonlJournal(output / "actions.jsonl", SPEC)
    try:
        with (output / "worker-stderr.bin").open("xb") as stderr:
            process = subprocess.Popen(command, cwd=output, stdin=subprocess.PIPE,
                                       stdout=subprocess.PIPE, stderr=stderr)
            try:
                launcher_identity = process_identity(process.pid)
                write_json(output / "worker-launch.json", {"command": command, "cwd": str(output),
                           "pid": process.pid, "process_identity": launcher_identity, "parent_pid": os.getpid()})
                proposal_row = read_row(process.stdout, SPEC)
                write_json(output / "proposal.json", proposal_row)
                proposal = check_proposal(proposal_row, "learned", 0, task, model_hash, memory_hash)
                validate_prediction(proposal_row["prediction"], state.successor[:state.used].tolist(),
                                    boundaries=workspace.boundaries)
                identity = {key: proposal_row[key] for key in ("schema", "type", "arm", "index", "id", "attempt_id")}
                envelope, action_hash, _, _ = execute_action(
                    task=None, workspace=workspace, proposal=proposal, attempt_id=identity["attempt_id"],
                    phase="endpoint", arm="learned", index=0, output=output, launch=launch, journal=action_journal)
                write_message(process.stdin, {**identity, "type": "executed", "action_sha256": action_hash, "outcome": envelope}, SPEC)
                acknowledgement = read_row(process.stdout, SPEC)
                write_json(output / "acknowledgement.json", acknowledgement)
                check_acknowledgement(acknowledgement, identity, action_hash, model_hash, memory_hash)
                process.stdin.close()
                process.wait(timeout=max(.001, deadline - time.time()))
                assert process.returncode == 0 and read_row(process.stdout, SPEC) is None
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
                if not process.stdin.closed:
                    process.stdin.close()
    finally:
        action_log = action_journal.close()
    completion = json.loads((output / "worker/completion.json").read_bytes())
    assert completion["scope"] == "single_exchange_qa" and completion["status"] == "passed"
    assert completion["pid"] == process.pid or completion["parent_pid"] == process.pid
    assert completion["process_identity"]["pid"] == completion["pid"]
    assert completion["checkpoint_sha256"] == file_hash(checkpoint) == checkpoint_hash
    assert completion["requests_sha256"] == file_hash(request_path) == requests_hash
    assert completion["source_sha256"] == sources
    assert completion["source_origins"] == {key: str(package / path) for key, path in SOURCE_MODULES.items()}
    assert completion["final"] == {"learned": {"model_sha256": model_hash,
               "memories_sha256": {"qa-memory": memory_hash}, "experience": 1}}
    assert completion["endpoint_log"]["rows"] == 2 and completion["reading_log"]["rows"] == 0
    for name, filename in (("endpoint_log", "endpoints.jsonl"), ("reading_log", "reading.jsonl")):
        assert completion[name]["sha256"] == file_hash(output / "worker" / filename)
    assert (output / "worker-stderr.bin").stat().st_size <= 1024
    assert _digest(model.state_dict()) == model_hash and _digest(state.snapshot()) == memory_hash
    assert action_log["rows"] == 1
    assert acknowledgement["work"] == {"action": {"custody_tensor_bytes_hashed": 1291336}, "reader": {}}
    steps = proposal_row["prediction"]["payload"]["steps"]
    assert 1 <= len(steps) <= 17
    write_json(output / "passed.json", {
        "schema": "neuronal-splice-prior-worker-qa-v1", "status": "passed",
        "parent_pid": os.getpid(), "launcher_identity": launcher_identity,
        "worker_identity": completion["process_identity"], "action_log": action_log,
        "prior": custody, "worker_completion_sha256": file_hash(output / "worker/completion.json"),
        "source_sha256": sources, "test_sha256": file_hash(Path(__file__)),
        "combined_model_constructions": 5, "prior_loads": 1, "worker_checkpoint_loads": 1,
        "ingests": 1, "document_raw_bytes": len(document), "document_eos": 1,
        "free_proposals": 1, "actual_payload_decisions": len(steps), "action_children": 1,
        "worker_processes": 1, "teachers": 0, "reading_panel_calls": 0,
        "proposal_sha256": proposal.sha256, "action_line_sha256": action_hash,
        "qualification_scope": "Same pending neuronal worker through actual execution; no acquired task skill."})


def test_splice_free_end_and_corrected_end_use_distinct_prefixes():
    """One model, two free proposals, one SPLICE plus one STOP; simulated outcomes."""
    from humanbrain import neuronal_splice_learner as splice
    from humanbrain.neuronal_splice_action import SpliceProposal

    model, state = _splice_qa_model()
    workspace = _splice_qa_workspace()
    # Explicit mechanics setup: all start scores tie, so raw boundary zero wins.
    model.actions.start_weight.zero_()
    end_calls, op_calls = [], []
    actual_end, actual_op = splice.end_features, splice.operation_features

    def observe_end(encoding, path, start, work=None):
        result = actual_end(encoding, path, start, work)
        end_calls.append((path, start, result[0]))
        return result

    def observe_op(encoding, path, start, end, payload, status, work=None):
        op_calls.append((path, start, end, payload, status))
        return actual_op(encoding, path, start, end, payload, status, work)

    with patch.object(splice, "end_features", side_effect=observe_end), \
            patch.object(splice, "operation_features", side_effect=observe_op):
        prediction = model.propose(b"set a.py k00", workspace, state)
        original_proposal = prediction.proposal.to_record()
        teacher = _splice_qa_teacher(workspace)
        feedback = model.teach(prediction, workspace, state, teacher,
                               _splice_qa_outcome(prediction, workspace))
    assert prediction.proposal.path == "a.py" and prediction.proposal.start == 0
    assert end_calls[0] == ("a.py", 0, tuple(range(15)))
    assert end_calls[1] == ("a.py", 5, tuple(range(5, 15)))
    assert op_calls[0] == (prediction.proposal.path, prediction.proposal.start,
                           prediction.proposal.end, prediction.proposal.replacement,
                           prediction.proposal.stop_reason)
    assert op_calls[1] == ("a.py", 5, 12, b"ab", "eos")
    assert prediction.proposal.to_record() == original_proposal
    assert feedback.reader_feedback.teacher_count == 3
    stopped = model.propose(b"keep a.py", workspace, state)
    actual = stopped.proposal
    teacher_stop = SpliceProposal("STOP", "unused.py", -1, -1, b"", "limit", workspace.sha256)
    op_calls.clear()
    with patch.object(model.reader, "teach", side_effect=AssertionError("STOP taught reader")), \
            patch.object(model.reader, "predict_with_receipt",
                         side_effect=AssertionError("STOP reran reader")), \
            patch.object(splice, "start_features", side_effect=AssertionError("STOP start")), \
            patch.object(splice, "end_features", side_effect=AssertionError("STOP end")), \
            patch.object(splice, "operation_features", side_effect=observe_op):
        feedback_stop = model.teach(stopped, workspace, state, teacher_stop,
                                    _splice_qa_outcome(stopped, workspace))
    assert op_calls == [(actual.path, actual.start, actual.end, actual.replacement,
                         actual.stop_reason)]
    assert feedback_stop.reader_feedback is None and feedback_stop.work["reader"] == {}
    assert set(feedback_stop.teacher_choices) == {"operation"}


def test_splice_stop_delete_and_capped_payload_remain_distinct():
    """One action circuit, three operation forwards, zero readers or teachers."""
    from humanbrain import neuronal_splice_learner as splice
    from humanbrain.neuronal_splice_action import SpliceProposal, apply_splice

    workspace = _splice_qa_workspace((("a.py", b"abc"),))
    encoding = splice.encode(b"edit a.py", workspace, _splice_qa_codes())
    circuit = splice.SpliceActionCircuit(seed=733)
    circuit.operation_weight.zero_()
    for payload, status, operation in ((b"q", "eos", "STOP"),
                                        (b"", "eos", "SPLICE"),
                                        (b"x" * 16, "limit", "STOP")):
        circuit.operation_bias.copy_(torch.tensor([1., 0.] if operation == "STOP" else [0., 1.]))
        features = splice.operation_features(encoding, "a.py", 1, 2, payload, status)
        decision = circuit.score("operation", features)
        selected = "STOP" if decision.selected_index == 0 else "SPLICE"
        assert selected == operation
        proposal = SpliceProposal(selected, "a.py", 1, 2, payload, status, workspace.sha256)
        result = apply_splice(workspace, proposal)
        assert result[0].content == (b"ac" if operation == "SPLICE" else b"abc")
        if status == "limit":
            assert features[0, -1] == 0
            with pytest.raises(ValueError, match="EOS"):
                apply_splice(workspace, SpliceProposal("SPLICE", "a.py", 1, 2,
                                                       payload, status, workspace.sha256))


def test_splice_four_publication_modes():
    """Four fresh modes: four simulated attempts, twelve payload/action labels each."""
    for mode in ("learned", "action_frozen", "slow_frozen", "memory_disabled"):
        model, state = _splice_qa_model(mode)
        workspace = _splice_qa_workspace()
        before = copy.deepcopy(model.state_dict())
        tape = state.snapshot()
        prediction = model.propose(b"set a.py k00", workspace, state)
        feedback = model.teach(prediction, workspace, state, _splice_qa_teacher(workspace),
                               _splice_qa_outcome(prediction, workspace))
        after = model.state_dict()
        action_changed = any(
            not torch.equal(before[key], after[key]) for key in before
            if key.startswith("actions.") and isinstance(before[key], torch.Tensor))
        reader_changed = any(not torch.equal(before[key], after[key]) for key in before
                             if key.startswith("reader.") and isinstance(before[key], torch.Tensor))
        assert action_changed is (mode in {"learned", "memory_disabled"})
        assert reader_changed is (mode != "slow_frozen")
        assert any(feedback.action_updates.values()) is action_changed
        assert feedback.reader_feedback.teacher_count == 3
        for head in ("start", "end", "operation"):
            assert feedback.work["action"][f"{head}_teacher_offers"] == 1
            assert feedback.work["action"][f"{head}_log_softmaxes"] == 1
            assert feedback.work["action"][f"{head}_enabled_update_groups"] == int(action_changed)
        assert all(not parameter.requires_grad and parameter.grad is None
                   for parameter in model.parameters())
        _assert_nested_equal(state.snapshot(), tape)
        if mode == "memory_disabled":
            assert torch.count_nonzero(state.sensory) == torch.count_nonzero(state.keys) == 0


def test_splice_learning_disabled_blocks_all_publication():
    """One full correction offered with both reader/action publication disabled."""
    model, state = _splice_qa_model()
    workspace = _splice_qa_workspace()
    prediction = model.propose(b"set a.py k00", workspace, state)
    before, tape = copy.deepcopy(model.state_dict()), state.snapshot()
    feedback = model.teach(prediction, workspace, state, _splice_qa_teacher(workspace),
                           _splice_qa_outcome(prediction, workspace), learning_enabled=False)
    _assert_nested_equal(model.state_dict(), before)
    _assert_nested_equal(state.snapshot(), tape)
    assert feedback.reader_feedback.teacher_count == 3
    assert not feedback.reader_feedback.codec_updated
    assert not feedback.reader_feedback.address_updated
    assert not any(feedback.action_updates.values()) and model._pending is None
    model.actions.start_weight.requires_grad_(True)
    try:
        with pytest.raises(ValueError, match="non-autograd"):
            model._check()
    finally:
        model.actions.start_weight.requires_grad_(False)


def test_splice_late_validation_failure_is_atomic():
    """Injected final action validation failure: zero reader teacher calls."""
    from humanbrain import neuronal_splice_learner as splice

    model, state = _splice_qa_model()
    workspace = _splice_qa_workspace()
    prediction = model.propose(b"set a.py k00", workspace, state)
    before, tape = copy.deepcopy(model.state_dict()), state.snapshot()
    checked, actual_finite = [], splice._finite

    def inject(value, name):
        actual_finite(value, name)
        if name == "action parameter proposal":
            checked.append(tuple(value.shape))
            if tuple(value.shape) == (2,):
                assert checked == [(6177,), (2049,), (2, 4244), (2,)]
                raise ValueError("injected final action validation failure")

    with (
        patch.object(splice, "_finite", side_effect=inject),
        patch.object(model.reader, "teach", side_effect=AssertionError("reader published first")),
        pytest.raises(ValueError, match="injected final action"),
    ):
        model.teach(prediction, workspace, state, _splice_qa_teacher(workspace),
                    _splice_qa_outcome(prediction, workspace))
    assert checked == [(6177,), (2049,), (2, 4244), (2,)]
    _assert_nested_equal(model.state_dict(), before)
    _assert_nested_equal(state.snapshot(), tape)
    assert model._pending[0] is prediction


def test_splice_combined_snapshot_roundtrip_has_no_external_selector():
    """Two new models, two proposals/acknowledgements, no teachers or stored checkpoint."""
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner

    model, state = _splice_qa_model()
    workspace = _splice_qa_workspace()
    snapshot = model.snapshot(state)
    original = copy.deepcopy(snapshot)
    prediction = model.propose(b"set a.py k00", workspace, state)
    with patch.object(model.reader, "teach", side_effect=AssertionError("evaluation teacher")), \
            patch.object(model.reader, "predict_with_receipt",
                         side_effect=AssertionError("acknowledgement forward")), \
            patch.object(model.reader, "_query", side_effect=AssertionError("ack query")), \
            patch.object(model.actions, "score", side_effect=AssertionError("ack scoring")):
        work = model.finish_without_teaching(prediction, workspace, state,
                                              _splice_qa_outcome(prediction, workspace))
    assert work["reader"] == {} and model._pending is None
    _assert_nested_equal(model.state_dict(), original["model"])
    assert model._experience == 1 and snapshot["experience"] == 0
    model.restore_snapshot(original)
    assert model._experience == 1  # Numerical restore cannot rewind live attempt identities.
    restored = NeuronalSpliceLearner(seed=733)
    restored_state = restored.restore_snapshot(snapshot)
    again = restored.propose(b"set a.py k00", workspace, restored_state)
    assert again.proposal == prediction.proposal
    for head in ("start", "end", "operation"):
        assert torch.equal(again.choices[head].logits, prediction.choices[head].logits)
    assert torch.equal(again.payload.initial_log_belief, prediction.payload.initial_log_belief)
    restored.finish_without_teaching(again, workspace, restored_state,
                                     _splice_qa_outcome(again, workspace))
    snapshot["model"]["actions.start_weight"].zero_()
    snapshot["memory"]["sensory"].zero_()
    _assert_nested_equal(model.state_dict(), original["model"])
    _assert_nested_equal(restored.state_dict(), original["model"])
    _assert_nested_equal(state.snapshot(), original["memory"])
    _assert_nested_equal(restored_state.snapshot(), original["memory"])


def test_splice_online_roster_data_contract(tmp_path):
    """One complete data roster and one bad-hash rejection; zero neural/child calls."""
    import ast
    import json
    from collections import Counter
    from hashlib import sha256
    from pathlib import Path

    from experiments.r4_integrated import neuronal_splice_roster_v1 as data

    root = Path(__file__).resolve().parents[1]
    generated_path = root / ".pytest-tmp/neuronal-splice-roster-qa-generated-v1.json"
    passed_path = root / ".pytest-tmp/neuronal-splice-roster-data-qa-v1.json"
    assert not generated_path.exists() and not passed_path.exists()
    old_path = root / ".pytest-tmp/local-joint-address-online-v6-run/offline_teacher_roster.json"
    roster = data.build_roster(old_path)
    with generated_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(roster, sort_keys=True, allow_nan=False) + "\n")
    assert roster["schema"] == "neuronal-splice-roster-v1"
    expected_counts = {
        "tasks": 64, "train_tasks": 32, "fresh_tasks": 32, "retention_tasks": 16,
        "lessons_per_arm": 256, "memory_packages_per_arm": 64,
        "before_learning_packages": 48, "after_learning_packages": 16,
        "documents_per_arm": 152, "document_raw_bytes_per_arm": 3638,
        "document_eos_per_arm": 152, "ingestion_symbols_per_arm": 3790,
        "maximum_memory_slots": 128, "reader_teacher_offers_per_arm": 480,
        "action_teacher_labels_per_arm": 640, "full_action_verifications": 1216,
        "old_panel_reader_calls": 160, "full_reader_teacher_offers": 1920,
        "full_action_teacher_labels": 2560,
    }
    assert roster["counts"] == expected_counts
    tasks, memories = roster["tasks"], roster["memory_packages"]
    documents = [bytes.fromhex(raw) for memory in memories.values()
                 for raw in memory["document_hex"]]
    assert len(documents) == 152 and sum(map(len, documents)) == 3638
    stage_ids = roster["before_learning_memory_ids"] + roster["after_learning_memory_ids"]
    assert len(set(stage_ids)) == 64
    assert set(roster["before_learning_memory_ids"]) == {
        key for key, value in memories.items() if value["stage"] == "before_learning"}
    assert set(roster["after_learning_memory_ids"]) == {
        key for key, value in memories.items() if value["stage"] == "after_learning"}
    assert roster["train_task_ids"] == [f"T{i:02d}" for i in range(32)]
    assert Counter(roster["schedule"]) == {task_id: 8 for task_id in roster["train_task_ids"]}
    for repeat in range(8):
        assert roster["schedule"][repeat * 32:(repeat + 1) * 32] == [
            f"T{(i + 4 * repeat) % 32:02d}" for i in range(32)]
    assert roster["cost_task_ids"] == ["T00", "T08", "T16", "T24"]
    assert roster["fresh_task_ids"] == [f"{group}{r}-{member}" for group in "PLVO"
                                        for r in range(4) for member in range(2)]
    assert roster["retention_task_ids"] == [f"T{i:02d}" for base in (0, 8, 16, 24)
                                            for i in range(base, base + 4)]
    assert Counter(tasks[key]["required_operation"] for key in roster["train_task_ids"]) == {
        "SPLICE": 24, "STOP": 8}
    assert Counter(tasks[key]["required_operation"] for key in roster["fresh_task_ids"]) == {
        "SPLICE": 28, "STOP": 4}
    assert Counter(tasks[key]["required_operation"] for key in roster["retention_task_ids"]) == {
        "SPLICE": 12, "STOP": 4}
    payload_offers, action_labels = 0, 0
    for task_id in roster["schedule"]:
        teacher = tasks[task_id]["teacher"]
        if teacher["operation"] == "SPLICE":
            payload_offers += len(bytes.fromhex(teacher["replacement_hex"])) + 1
            action_labels += 3
        else:
            action_labels += 1
    assert (payload_offers, action_labels) == (480, 640)

    for task in tasks.values():
        before = {item["path"]: bytes.fromhex(item["content_hex"]) for item in task["files"]}
        assert len(before) == 3 and max(map(len, before.values())) <= 128
        assert max(len(path.encode()) for path in before) <= 31
        assert len(bytes.fromhex(task["request_hex"])) <= 31
        teacher = task["teacher"]
        actual = dict(before)
        if teacher["operation"] == "SPLICE":
            path, start, end = teacher["path"], teacher["start"], teacher["end"]
            raw = before[path]
            assert 0 <= start <= end <= len(raw)
            assert teacher["stop_reason"] == "eos"
            actual[path] = raw[:start] + bytes.fromhex(teacher["replacement_hex"]) + raw[end:]
        assert actual == {item["path"]: bytes.fromhex(item["content_hex"])
                          for item in task["expected_after_files"]}
        # Independent Python literal interpretation, never exec/eval/runpy/subprocess.
        module_values = {}
        for path, raw in actual.items():
            if path == "main.py":
                continue
            module_values[path[:-3]] = {}
            body = ast.parse(raw.decode("ascii"), filename=path).body
            assert len(body) == 2
            for assignment in body:
                assert isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                assert isinstance(assignment.targets[0], ast.Name)
                value = ast.literal_eval(assignment.value)
                assert isinstance(value, str)
                module_values[path[:-3]][assignment.targets[0].id] = value.encode("ascii")
            assert set(module_values[path[:-3]]) == {"x", "y"}
        main = ast.parse(actual["main.py"].decode("ascii")).body
        assert isinstance(main[0], ast.Import) and len(main[0].names) == 2
        names = [alias.name for alias in main[0].names]
        stdout = b"|".join(module_values[name][variable] for name in names
                             for variable in ("x", "y")) + b"\n"
        assert stdout.hex() == task["expected_stdout_hex"]
        observation = data.task_observation(task)
        assert set(observation) == {"task_id", "memory_id", "request_hex", "files",
                                    "workspace_sha256"}
        assert set(observation["files"][0]) == {"path", "content_hex"}
        observation["files"][0]["content_hex"] = ""
        assert task["files"][0]["content_hex"] != ""

    for group in "PLVO":
        for r in range(4):
            left, right = tasks[f"{group}{r}-0"], tasks[f"{group}{r}-1"]
            assert left["pair_id"] == right["pair_id"] == f"{group}{r}"
            assert (left["member"], right["member"]) == (0, 1)
            assert left["expected_after_files"] != right["expected_after_files"]
            if group == "P":
                assert left["memory_id"] == right["memory_id"]
                assert left["workspace_sha256"] == right["workspace_sha256"]
                assert left["files"] == list(reversed(right["files"]))
                assert left["teacher"]["path"] != right["teacher"]["path"]
                assert left["teacher"]["replacement_hex"] == right["teacher"]["replacement_hex"]
            elif group == "L":
                assert left["memory_id"] == right["memory_id"]
                assert left["teacher"]["start"] != right["teacher"]["start"]
                assert (left["request_hex"] == right["request_hex"]) is (r < 2)
                assert (left["workspace_sha256"] == right["workspace_sha256"]) is (r >= 2)
            elif group == "V":
                assert left["request_hex"] == right["request_hex"]
                assert left["files"] == right["files"]
                assert left["memory_id"] != right["memory_id"]
                a, b = (memories[item["memory_id"]] for item in (left, right))
                assert a["stage"] == b["stage"] == (
                    "before_learning" if r < 2 else "after_learning")
                assert len(a["document_hex"]) == len(b["document_hex"]) == (2 if r < 2 else 5)
                assert left["teacher"]["replacement_hex"] != right["teacher"]["replacement_hex"]
                target_prefix = f"Record k{40 + r} has value ".encode()
                other_prefix = f"Record k{60 + r} has value ".encode()
                assert bytes.fromhex(a["document_hex"][0])[len(target_prefix):] == (
                    bytes.fromhex(b["document_hex"][1])[len(other_prefix):])
                assert bytes.fromhex(a["document_hex"][1])[len(other_prefix):] == (
                    bytes.fromhex(b["document_hex"][0])[len(target_prefix):])
            else:
                assert left["files"] == right["files"] and left["memory_id"] == right["memory_id"]
                assert left["required_operation"] == "SPLICE"
                assert right["required_operation"] == "STOP"
                assert bytes.fromhex(left["request_hex"])[4:] == (
                    bytes.fromhex(right["request_hex"])[5:])
                if r == 1:
                    assert left["teacher"]["start"] == left["teacher"]["end"]
                if r == 2:
                    assert left["teacher"]["replacement_hex"] == ""
                if r == 3:
                    assert left["teacher"]["replacement_hex"] == "625c2763"

    old = json.loads(old_path.read_bytes())["checks"]
    assert sha256(old_path.read_bytes()).hexdigest() == data.OLD_ROSTER_SHA256
    for row, prior in zip(roster["old_panel"], old, strict=True):
        assert row["case_id"] == prior["id"]
        assert row["query_hex"] == prior["query_hex"]
        assert row["expected_hex"] == prior["expected_hex"]
        assert row["max_bytes"] == 2 and "delay_reads" not in row
        assert row["originally_correct"] is (prior["id"] not in (16, 18))
        assert memories[row["memory_id"]]["document_hex"] == prior["document_hex"]
    malformed = tmp_path / "wrong-old-roster.json"
    malformed.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact authenticated v6 roster"):
        data.build_roster(malformed)
    with passed_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps({"schema": "neuronal-splice-roster-data-qa-v1", "status": "passed",
            "counts": roster["counts"], "generated_roster_sha256": sha256(
                generated_path.read_bytes()).hexdigest(), "source_sha256": sha256(
                Path(data.__file__).read_bytes()).hexdigest(), "test_sha256": sha256(
                Path(__file__).read_bytes()).hexdigest(), "completed_roster_generations": 1,
            "rejected_old_json": 1, "models": 0, "children": 0, "teachers": 0},
            sort_keys=True, allow_nan=False) + "\n")


def test_splice_online_saved_cost_decision_custody(tmp_path, monkeypatch):
    """One synthetic admission and three authenticated-but-inconsistent variants."""
    import copy
    import hashlib
    import json

    from experiments.r4_integrated import check_neuronal_splice_online_v1 as checker

    # Only source identity and pathname context are synthetic. The production
    # readers, hashes, monitor checks and projection validator run unchanged.
    fake_checker = tmp_path / "experiments/r4_integrated/check_neuronal_splice_online_v1.py"
    fake_checker.parent.mkdir(parents=True)
    fake_checker.write_text("# Authored source identity; never imported.\n", encoding="utf-8")
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "__file__", str(fake_checker))

    def save(path, value):
        raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
        path.write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    config = checker.expected_config()
    sources = {"experiments/r4_integrated/check_neuronal_splice_online_v1.py":
               hashlib.sha256(fake_checker.read_bytes()).hexdigest()}
    roster_path = tmp_path / "authored-roster.json"
    roster_sha = save(roster_path, {"schema": "authored-no-neural-roster", "tasks": []})
    cost_dir = tmp_path / "cost"
    cost_dir.mkdir()
    paths = {
        "cost_report": cost_dir / "report.json",
        "cost_completion": cost_dir / "completion.json",
        "cost_launch": tmp_path / "cost.launch.json",
        "cost_admission": tmp_path / "cost-admission.json",
        "cost_audit": tmp_path / "cost-audit.json",
        "cost_audit_launch": tmp_path / "cost-audit.launch.json",
    }
    values, hashes = {}, {}
    values["cost_admission"] = {
        "schema": "neuronal-splice-admission-v1", "status": "passed",
        "authorized_phase": "cost", "qa_passed": True, "config": config,
        "source_sha256": sources, "roster_path": str(roster_path), "roster_sha256": roster_sha,
    }
    hashes["cost_admission"] = save(paths["cost_admission"], values["cost_admission"])
    summary = {}
    for arm in checker.ARMS:
        summary[arm] = {}
        for phase, count, durations in (("regular", 4, (.001, .002, .0005)),
                                         ("cap", 1, (.003, .004, .001))):
            row = {"rows": count}
            for stage, maximum in zip(("model", "action", "journal"), durations, strict=True):
                row[stage + "_seconds"] = count * maximum
                row[stage + "_max_seconds"] = maximum
            summary[arm][phase] = row
    values["cost_report"] = {
        "schema": "neuronal-splice-online-v1", "status": "completed", "mode": "cost",
        "config": config, "source_sha256": sources, "source_sha256_after": sources,
        "admission_sha256": hashes["cost_admission"], "roster_sha256": roster_sha,
        "pid": 101, "deadline_unix": 1900.0, "logs": {}, "timing_summary": summary,
        "category_probes": [{"action_seconds": value} for value in (.002, .003, .004, .005)],
        "memory": {"peak_working_set_mib": 100.0},
        "worker": {"worker": {"memory": {"peak_working_set_mib": 150.0}}},
    }
    hashes["cost_report"] = save(paths["cost_report"], values["cost_report"])
    values["cost_completion"] = {
        "schema": "neuronal-splice-completion-v1", "status": "completed", "mode": "cost",
        "report_sha256": hashes["cost_report"], "admission_sha256": hashes["cost_admission"], "logs": {},
    }
    hashes["cost_completion"] = save(paths["cost_completion"], values["cost_completion"])

    def monitor(pid, command, seconds, peak):
        # Actual watchdog schema: .NET DateTime UTC ticks, not FILETIME.
        ticks = 621355968000000000 + 1001 * 10_000_000
        return {"status": "passed", "exit_code": 0, "seconds_limit": 900,
                "tree_peak_sum_mib_limit": 2048, "started_unix": 1000.0,
                "deadline_unix": 1900.0, "elapsed_seconds": seconds,
                "observed_tree_peak_sum_mib": peak, "command": command,
                "child_pid": pid, "observed_process_peaks_mib": {f"{pid}/{ticks}": peak},
                "launcher_sha256": "a" * 64}

    prefix = [str(tmp_path / ".venv/Scripts/python.exe"), "-X", "utf8", "-m"]
    cost_command = prefix + ["experiments.r4_integrated.neuronal_splice_online_v1", "cost",
                             "--output", str(cost_dir), "--admission", str(paths["cost_admission"]),
                             "--admission-sha", hashes["cost_admission"]]
    values["cost_launch"] = monitor(101, cost_command, 5.0, 300.0)
    hashes["cost_launch"] = save(paths["cost_launch"], values["cost_launch"])
    preliminary = checker.project_cost(values["cost_report"], values["cost_launch"])
    assert preliminary["projected_run_seconds"] == pytest.approx(58.32)
    assert preliminary["audit_peak_pending"] and not preliminary["admissible"]
    bound_inputs = {str(paths[name]): hashes[name] for name in
                    ("cost_report", "cost_completion", "cost_launch", "cost_admission")}
    bound_inputs.update({str(fake_checker): sources[next(iter(sources))], str(roster_path): roster_sha})
    values["cost_audit"] = {
        "schema": "neuronal-splice-independent-audit-v1", "status": "passed", "mode": "cost",
        "pid": 202, "source_sha256": sources, "roster_sha256": roster_sha,
        "report_sha256": hashes["cost_report"], "completion_sha256": hashes["cost_completion"],
        "launch_sha256": hashes["cost_launch"], "admission_sha256": hashes["cost_admission"],
        "input_sha256": bound_inputs, "projection": preliminary,
        "learning_pair_timing": {arm: {
            "regular": {"rows": 4, "seconds": .0004, "max_seconds": .0001},
            "cap": {"rows": 1, "seconds": .0003, "max_seconds": .0003},
        } for arm in checker.ARMS},
    }
    hashes["cost_audit"] = save(paths["cost_audit"], values["cost_audit"])
    audit_command = prefix + ["experiments.r4_integrated.check_neuronal_splice_online_v1",
                              "--output", str(paths["cost_audit"]), "--cost-dir", str(cost_dir),
                              "--launch", str(paths["cost_launch"]),
                              "--admission", str(paths["cost_admission"]),
                              "--admission-sha", hashes["cost_admission"]]
    values["cost_audit_launch"] = monitor(202, audit_command, 2.0, 120.0)
    hashes["cost_audit_launch"] = save(paths["cost_audit_launch"], values["cost_audit_launch"])
    final = checker.project_cost(values["cost_report"], values["cost_launch"],
                                 audit=values["cost_audit"], audit_outer=values["cost_audit_launch"])
    assert final["projected_audit_seconds"] == pytest.approx(4.6112)
    assert final["projected_seconds"] == pytest.approx(62.9312)
    assert final["projected_tree_mib"] == 440.0
    decision = {
        "schema": "neuronal-splice-cost-decision-v1", "status": "passed", "config": config,
        "source_sha256": sources, "roster_sha256": roster_sha, "completed_cost_audit": True,
        "inputs": {name: {"path": str(paths[name]), "sha256": hashes[name]} for name in paths},
        "projection": final, "projected_seconds": final["projected_seconds"],
        "projected_tree_mib": final["projected_tree_mib"],
    }
    decision_path = tmp_path / "decision.json"
    decision_sha = save(decision_path, decision)
    assert checker.validate_cost_decision(decision_path, decision_sha, config, sources, roster_sha) == final

    for mutation, message in (("pid", "actual process absent"),
                               ("input", "audit did not authenticate selected input"),
                               ("projection", "final admission projection mismatch")):
        altered_audit, altered_decision = copy.deepcopy(values["cost_audit"]), copy.deepcopy(decision)
        if mutation == "pid":
            altered_audit["pid"] = 999
        elif mutation == "input":
            altered_audit["input_sha256"][str(paths["cost_report"])] = "b" * 64
        else:
            altered_decision["projection"]["projected_seconds"] -= 1.0
            altered_decision["projected_seconds"] -= 1.0
        # Reauthenticate altered bytes so rejection exercises inner custody/
        # independent arithmetic, rather than merely an outer file-hash mismatch.
        altered_decision["inputs"]["cost_audit"]["sha256"] = save(paths["cost_audit"], altered_audit)
        altered_sha = save(decision_path, altered_decision)
        with pytest.raises(ValueError, match=message):
            checker.validate_cost_decision(decision_path, altered_sha, config, sources, roster_sha)


def test_address_competition_scalar_circuit_contract(tmp_path):
    import json

    import torch

    from humanbrain.neuronal_address_competition import compete

    maximum = torch.finfo(torch.float32).max
    fixtures = (
        ([-3], 0), ([1, -2], 0), ([-3, 0, 5, 2], 2), ([3, 3, 1], 0),
        ([0] * 128, 0), ([maximum, -maximum, 0], 0),
        ([1, 1 + 2**-23], 1), (list(range(128)), 127),
    )
    for ordinal, (values, expected_winner) in enumerate(fixtures):
        scores = torch.tensor(values, dtype=torch.float32)
        before = scores.clone()
        result = compete(scores)
        raw = {
            "fixture": ordinal, "scope": "authored circuit only",
            "scores": result.scores.tolist(), "pair_currents": result.pair_currents.tolist(),
            "comparators": result.comparator_activity.tolist(),
            "winner_currents": result.winner_currents.tolist(),
            "winners": result.winner_activity.tolist(), "index": result.winner_index,
            "work": result.work,
        }
        with (tmp_path / f"competition-scalar-{ordinal}.json").open("x") as stream:
            json.dump(raw, stream, allow_nan=False)
        numbers = [float(value) for value in before.tolist()]
        n, pairs = len(numbers), len(numbers) * (len(numbers) - 1) // 2
        currents, activity = [], []
        winner_currents = [i - (n - 1) for i in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                current = numbers[i] - numbers[j]
                firing = int(current >= 0)
                currents.append(current)
                activity.append(bool(firing))
                winner_currents[i] += firing
                winner_currents[j] -= firing
        expected_work = {
            "competition_candidates": n, "competition_pairs": pairs,
            "competition_threshold_layers": 2,
            "competition_comparator_neurons": pairs, "competition_winner_neurons": n,
            "competition_comparator_synapse_applications": 2 * pairs,
            "competition_winner_synapse_applications": 2 * pairs,
            "competition_comparator_subtractions": pairs,
            "competition_winner_accumulations": 2 * pairs,
            "competition_winner_bias_values": n,
            "competition_winner_threshold_subtractions": n,
            "competition_threshold_evaluations": pairs + n,
            "competition_identity_synapse_applications": n,
            "competition_identity_reductions": n - 1,
            "competition_winner_count_reductions": n - 1,
            "competition_index_values_constructed": 2 * pairs + n,
            "competition_receipt_tensor_bytes": 13 * n + 9 * pairs,
            "competition_named_allocated_tensor_bytes": 45 * n + 49 * pairs + 16,
            "competition_pair_index_bytes": 16 * pairs,
            "competition_score_gather_bytes": 16 * pairs,
        }
        assert torch.equal(scores, before)
        assert result.scores.data_ptr() != scores.data_ptr()
        assert result.scores.dtype == torch.float32
        assert result.pair_currents.dtype == result.winner_currents.dtype == torch.float64
        assert result.comparator_activity.dtype == result.winner_activity.dtype == torch.bool
        assert result.pair_currents.tolist() == currents
        assert result.comparator_activity.tolist() == activity
        assert result.winner_currents.tolist() == winner_currents
        assert result.winner_activity.tolist() == [i == expected_winner for i in range(n)]
        assert result.winner_index == expected_winner
        assert result.work == expected_work
        arrays = (result.scores, result.pair_currents, result.comparator_activity,
                  result.winner_currents, result.winner_activity)
        assert sum(item.numel() * item.element_size() for item in arrays) == 13 * n + 9 * pairs
        assert all(item.device.type == "cpu" and not item.requires_grad for item in arrays)
        if n == 128:
            assert pairs + n == 8256
            assert 4 * pairs == 32512


def test_address_competition_input_guards(tmp_path):
    import json

    import torch

    from humanbrain.neuronal_address_competition import compete

    fixtures = (
        torch.empty(0, dtype=torch.float32),
        torch.zeros(129, dtype=torch.float32),
        torch.zeros((1, 1), dtype=torch.float32),
        torch.zeros(1, dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float32),
        torch.tensor([float("inf")], dtype=torch.float32),
        torch.zeros(1, dtype=torch.float32, requires_grad=True),
    )
    for ordinal, scores in enumerate(fixtures):
        before = scores.detach().clone()
        error = None
        try:
            compete(scores)
        except ValueError as caught:
            error = str(caught)
        raw = {
            "scope": "authored invalid circuit input", "fixture": ordinal,
            "dtype": str(scores.dtype), "shape": list(scores.shape),
            "requires_grad": scores.requires_grad, "input_repr": repr(before.tolist()),
            "error": error,
        }
        with (tmp_path / f"competition-invalid-{ordinal}.json").open("x") as stream:
            json.dump(raw, stream, allow_nan=False)
        assert error == "address scores must be finite detached CPU float32 [1..128]"
        assert torch.equal(torch.isnan(scores), torch.isnan(before))
        assert torch.equal(scores[~torch.isnan(scores)], before[~torch.isnan(before)])
        assert scores.grad is None


def test_competitive_reader_single_cursor_and_full_bank(tmp_path):
    import json
    import math
    from dataclasses import fields, is_dataclass

    import pytest
    import torch

    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner

    def plain(value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if is_dataclass(value):
            return {field.name: plain(getattr(value, field.name)) for field in fields(value)}
        if isinstance(value, dict):
            return {key: plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [plain(item) for item in value]
        if isinstance(value, bytes):
            return {"hex": value.hex()}
        return value

    model = CompetitiveLocalMemoryLearner(mode="learned", seed=733)
    with torch.no_grad():
        model.address_weight.zero_()
    state = model.ingest([b"ab", b"c"])
    empty_document = model.ingest([b""])
    before_parameters = {name: value.detach().clone()
                         for name, value in model.named_parameters()}
    for ordinal, (tape, cap, expected, reason, tokens) in enumerate((
        (state, 16, b"ab", "eos", [97, 98, 256]),
        (state, 1, b"a", "limit", [97, 98]),
        (empty_document, 16, b"", "eos", [256]),
    )):
        before_tape = tape.snapshot()
        result = model.predict_with_receipt(b"a", tape, cap)
        with (tmp_path / f"competitive-free-{ordinal}.json").open("x") as stream:
            json.dump({"scope": "authored fresh reader", "prediction": plain(result)},
                      stream, allow_nan=False)
        n, decisions = tape.used, len(tokens)
        assert n == (1 if ordinal == 2 else 5)
        assert result.generation.data == expected
        assert result.generation.stop_reason == reason
        assert [step.observed_token for step in result.steps] == tokens
        assert [step.output_step for step in result.steps] == list(range(decisions))
        assert [step.branch_cursors.tolist() for step in result.steps] == [
            [i] for i in range(decisions)
        ]
        assert result.initial_log_belief.tolist() == result.final_log_belief.tolist() == [0.0]
        assert result.diagnostic_log_prior.shape == (n,)
        assert result.diagnostic_log_prior.tolist() == pytest.approx([-math.log(n)] * n)
        assert result.competition.winner_index == 0
        assert result.competition.scores.tolist() == [0.0] * n
        for step in result.steps:
            assert step.prior_log_belief.tolist() == step.posterior_log_belief.tolist() == [0.0]
            assert step.mixture_probabilities.shape == (257,)
            assert float(step.mixture_probabilities[step.observed_token]) == pytest.approx(.5)
            assert step.nll == pytest.approx(math.log(2))
        expected_work = dict.fromkeys(model.WORK_KEYS, 0)
        expected_work.update({
            "query_raw_bytes": 1, "query_eos_symbols": 1, "query_lag_shift_steps": 2,
            "query_population_normalizations": 1, "address_feature_slots": n,
            "address_feature_products": n * 32768,
            "address_feature_channel_reductions": n * 31744,
            "address_score_synapse_evaluations": n * 1024,
            "address_score_reductions": n * 1023, "address_score_evaluations": n,
            "address_normalizations": 1, "codec_slot_evaluations": n,
            "codec_output_potentials": n * 257, "codec_normalizations": n,
            "codec_response_gathers": decisions,
            "codec_response_values_gathered": decisions * 257 * 4,
            "codec_response_bytes_gathered": decisions * 257 * 28,
            "mixture_output_reductions": decisions * 257,
            "prefix_belief_updates": decisions, "branch_cursor_initializations": 1,
            "branch_cursor_advances": decisions - 1, "codec_bank_slot_indices": n,
        })
        expected_work.update(result.competition.work)
        assert result.work == expected_work
        assert all(torch.equal(value, before_parameters[name])
                   for name, value in model.named_parameters())
        assert all(torch.equal(value, before_tape[name])
                   for name, value in tape.snapshot().items())


def test_competitive_reader_inherited_publication_modes(tmp_path):
    import json

    import torch

    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner

    for mode in ("learned", "address_frozen", "slow_frozen", "writes_disabled"):
        model = CompetitiveLocalMemoryLearner(mode=mode, seed=733)
        with torch.no_grad():
            model.address_weight.zero_()
        state = model.ingest([b"ab"])
        before_tape = state.snapshot()
        enabled_calls = (True, False) if mode == "learned" else (True,)
        for learning_enabled in enabled_calls:
            before = {name: value.detach().clone() for name, value in model.named_parameters()}
            result = model.teach(b"a", b"b", state, .01, learning_enabled=learning_enabled)
            changed = {name: not torch.equal(value, before[name])
                       for name, value in model.named_parameters()}
            raw = {
                "scope": "authored inherited local teacher", "mode": mode,
                "learning_enabled": learning_enabled, "teacher_count": result.teacher_count,
                "total_loss": result.total_loss, "mean_loss": result.mean_loss,
                "codec_updated": result.codec_updated, "address_updated": result.address_updated,
                "parameter_changed": changed, "work": result.work,
                "initial_log_belief": result.initial_log_belief.tolist(),
                "final_log_belief": result.final_log_belief.tolist(),
                "address_error": result.address_error.tolist(),
                "steps": [{"token": step.observed_token,
                           "prior": step.prior_log_belief.tolist(),
                           "posterior": step.posterior_log_belief.tolist(),
                           "cursors": step.branch_cursors.tolist(),
                           "mixture": step.mixture_probabilities.tolist(), "nll": step.nll}
                          for step in result.steps],
            }
            suffix = "enabled" if learning_enabled else "disabled"
            with (tmp_path / f"competitive-teacher-{mode}-{suffix}.json").open("x") as stream:
                json.dump(raw, stream, allow_nan=False)
            codec_enabled = learning_enabled and mode != "slow_frozen"
            address_enabled = learning_enabled and mode not in {"address_frozen", "slow_frozen"}
            codec_changed = changed["codec.weight"] or changed["codec.bias"]
            assert result.teacher_count == 2
            assert [step.observed_token for step in result.steps] == [98, 256]
            assert [step.branch_cursors.tolist() for step in result.steps] == [[0, 1, 2], [1, 2, 2]]
            assert all(step.prior_log_belief.shape == step.posterior_log_belief.shape == (3,)
                       for step in result.steps)
            assert result.codec_updated == codec_changed == codec_enabled
            assert result.address_updated == changed["address_weight"]
            assert changed["address_weight"] == (address_enabled and mode != "writes_disabled")
            if mode == "writes_disabled":
                assert not state.keys.count_nonzero() and not state.sensory.count_nonzero()
                assert not changed["codec.weight"] and changed["codec.bias"]
            expected_work = dict.fromkeys(model.WORK_KEYS, 0)
            expected_work.update({
                "query_raw_bytes": 1, "query_eos_symbols": 1, "query_lag_shift_steps": 2,
                "query_population_normalizations": 1, "address_feature_slots": 3,
                "address_feature_products": 3 * 32768,
                "address_feature_channel_reductions": 3 * 31744,
                "address_score_synapse_evaluations": 3 * 1024,
                "address_score_reductions": 3 * 1023, "address_score_evaluations": 3,
                "address_normalizations": 1, "codec_slot_evaluations": 3,
                "codec_output_potentials": 3 * 257, "codec_normalizations": 3,
                "codec_response_gathers": 3, "codec_response_values_gathered": 3 * 257 * 4,
                "codec_response_bytes_gathered": 3 * 257 * 28,
                "mixture_output_reductions": 2 * 257, "prefix_belief_updates": 6,
                "branch_cursor_initializations": 3, "branch_cursor_advances": 3,
                "teacher_offers": 2, "codec_update_group_offers": 1,
                "address_update_group_offers": 1,
                "enabled_codec_update_groups": int(codec_enabled),
                "enabled_address_update_groups": int(address_enabled),
                "address_local_synapse_contributions": 3 * 1024,
                "address_local_slot_reductions": 2 * 1024,
            })
            assert result.work == expected_work
            assert not any(name.startswith("competition_") for name in result.work)
            assert all(torch.equal(value, before_tape[name])
                       for name, value in state.snapshot().items())
            assert all(torch.equal(value, before_tape[name])
                       for name, value in result.state.snapshot().items())
            assert all(not value.requires_grad and value.grad is None and value.grad_fn is None
                       for value in model.parameters())


def test_competitive_reader_metadata_and_snapshot(tmp_path):
    import copy
    import json

    import torch

    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = CompetitiveLocalMemoryLearner(mode="learned", seed=733)
    state = model.ingest([b"ab"])
    snapshot = model.snapshot(state)
    pristine = copy.deepcopy(snapshot)
    damaged_copy = copy.deepcopy(snapshot)
    with torch.no_grad():
        model.address_weight.add_(1)
        damaged_copy["model"]["address_weight"].zero_()
        damaged_copy["memory"]["keys"].zero_()
    restored = model.restore_snapshot(snapshot)
    metadata, accounting = model.get_extra_state(), model.accounting()
    parameter_equal = {
        name: torch.equal(value, pristine["model"][name])
        for name, value in model.state_dict().items() if isinstance(value, torch.Tensor)
    }
    tape_equal = {name: torch.equal(value, pristine["memory"][name])
                  for name, value in restored.snapshot().items()}
    raw = {
        "scope": "fresh snapshot only; no checkpoint file or neural query",
        "metadata": metadata, "accounting": accounting,
        "parameter_equal": parameter_equal, "tape_equal": tape_equal,
        "teacher_inherited": CompetitiveLocalMemoryLearner.teach is LocalMemoryLearner.teach,
        "query_inherited": CompetitiveLocalMemoryLearner._query is LocalMemoryLearner._query,
    }
    with (tmp_path / "competitive-metadata-snapshot.json").open("x") as stream:
        json.dump(raw, stream, allow_nan=False)
    assert metadata["schema"] == "local-neuronal-competitive-address-learner-v1"
    assert "one actual cursor" in metadata["recall"]
    assert "diagnostic only" in metadata["free_belief"]
    assert "every occupied sensory slot once" in metadata["codec_activity"]
    assert "reopen all address candidates on the same synapses" in metadata["teacher_phase"]
    assert "no competition pass" in metadata["teacher_phase"]
    assert CompetitiveLocalMemoryLearner.teach is LocalMemoryLearner.teach
    assert CompetitiveLocalMemoryLearner._query is LocalMemoryLearner._query
    assert CompetitiveLocalMemoryLearner.predict_with_receipt is not (
        LocalMemoryLearner.predict_with_receipt)
    assert all(parameter_equal.values()) and all(tape_equal.values())
    assert accounting["allocated_address_competition_neurons"] == 8256
    assert accounting["maximum_competition_signed_synapse_applications"] == 32512
    assert accounting["competition_slow_synapses"] == 0
    assert accounting["free_active_cursor_count"] == 1
    assert accounting["free_log_belief_bytes"] == 8
    assert accounting["maximum_free_diagnostic_log_prior_bytes"] == 1024
    assert accounting["free_codec_response_gather_bytes"] == 257 * 28
    assert accounting["maximum_free_overlapping_codec_response_bytes"] == 130 * 257 * 28
    assert all(not value.requires_grad and value.grad is None and value.grad_fn is None
               for value in model.parameters())
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(snapshot["model"][name], pristine["model"][name])
            assert value.data_ptr() != snapshot["model"][name].data_ptr()
    for name, value in vars(restored).items():
        assert value.data_ptr() != snapshot["memory"][name].data_ptr()
        assert value.data_ptr() != vars(state)[name].data_ptr()
        assert torch.equal(snapshot["memory"][name], pristine["memory"][name])


def test_competitive_splice_wrong_action_then_teacher_parity(tmp_path):
    """One actual wrong edit precedes correction on the same persistent reader."""
    import dataclasses
    import json
    from hashlib import sha256
    from pathlib import Path
    from time import perf_counter

    from experiments.r4_integrated.neuronal_splice_io_v1 import JsonlJournal
    from experiments.r4_integrated.neuronal_splice_online_v1 import (
        SPEC,
        execute_action,
        qualified_launch,
    )
    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
    from humanbrain.local_memory_learner import LocalMemoryLearner
    from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
    from humanbrain.neuronal_splice_action import SpliceProposal, SpliceWorkspace
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest

    def numerical(model):
        return {name: value.clone() for name, value in model.state_dict().items()
                if isinstance(value, torch.Tensor)}

    def serial(value):
        if isinstance(value, torch.Tensor):
            return value.tolist()
        if isinstance(value, bytes):
            return value.hex()
        if dataclasses.is_dataclass(value):
            return {field.name: serial(getattr(value, field.name))
                    for field in dataclasses.fields(value)}
        if isinstance(value, dict):
            return {key: serial(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [serial(item) for item in value]
        return value

    def save(name, value):
        path = tmp_path / name
        with path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        return sha256(path.read_bytes()).hexdigest()

    root = Path(__file__).resolve().parents[1]
    launch = qualified_launch(root)
    reader = CompetitiveLocalMemoryLearner(seed=733)
    base = LocalMemoryLearner(seed=733)
    with torch.no_grad():
        reader.address_weight.zero_()
        base.address_weight.zero_()
    model = NeuronalSpliceLearner(seed=733, reader=reader)
    state, reference_state = model.ingest([b"ab"]), base.ingest([b"ab"])
    with torch.no_grad():
        for parameter in model.actions.parameters():
            parameter.zero_()
        model.actions.operation_bias[1] = 1
    workspace = SpliceWorkspace((OpaqueWorkspaceFile("a.py", b"x = 'old'\n"),
                                 OpaqueWorkspaceFile("main.py", b"import a\nprint(a.x)\n")))
    before = numerical(reader)
    _assert_nested_equal(before, numerical(base))
    _assert_nested_equal(state.snapshot(), reference_state.snapshot())
    tape_before = state.snapshot()
    prediction = model.propose(b"a", workspace, state)
    journal = JsonlJournal(tmp_path / "action.jsonl", SPEC)
    envelope, action_sha, _, grade = execute_action(
        task=None, workspace=workspace, proposal=prediction.proposal,
        attempt_id=prediction.attempt_id, phase="competitive-qa", arm="learned", index=0,
        output=tmp_path, launch=launch, journal=journal,
    )
    journal_summary = journal.close()
    completed = perf_counter()
    raw_sha = save("raw.json", {
        "schema": "competitive-splice-component-raw-v1", "status": "pending_checks",
        "model_instance": id(model), "reader_instance": id(reader), "state_instance": id(state),
        "proposal": prediction.proposal.to_record(), "payload": serial(prediction.payload),
        "envelope": envelope, "action_sha256": action_sha, "journal": journal_summary,
        "action_completed_perf": completed, "model_before_sha256": prediction.model_sha256,
        "memory_before_sha256": prediction.memory_sha256,
    })
    # These outcome checks deliberately follow the immutable raw action journal.
    action = json.loads((tmp_path / "action.jsonl").read_bytes())
    assert grade is None and action["rejection"] is None
    assert prediction.proposal == SpliceProposal("SPLICE", "a.py", 0, 0, b"ab", "eos", workspace.sha256)
    assert envelope["execution"]["returncode"] != 0
    assert type(envelope["execution"]["returncode"]) is int
    assert b"AttributeError" in bytes.fromhex(envelope["execution"]["stderr_hex"])
    assert (tmp_path / "executions/competitive-qa-learned-000000/a.py").read_bytes() == b"abx = 'old'\n"
    assert prediction.attempt_id == f"{id(model)}:1" and model.reader is reader
    assert prediction.payload.competition.winner_index == 0
    assert len(prediction.payload.steps) == 3
    _assert_nested_equal(before, numerical(reader))
    _assert_nested_equal(tape_before, state.snapshot())
    teacher = SpliceProposal("SPLICE", "a.py", 5, 8, b"b", "eos", workspace.sha256)
    teacher_started = perf_counter()
    feedback = model.teach(prediction, workspace, state, teacher, envelope)
    reference = base.teach(b"a", b"b", reference_state, .01)
    actual = feedback.reader_feedback
    save("teacher-raw.json", {
        "actual": {field.name: serial(getattr(actual, field.name))
                   for field in dataclasses.fields(actual) if field.name != "state"},
        "reference": {field.name: serial(getattr(reference, field.name))
                      for field in dataclasses.fields(reference) if field.name != "state"},
        "teacher_started_perf": teacher_started, "teacher_completed_perf": perf_counter(),
        "reader_after_sha256": _digest(numerical(reader)),
        "reference_after_sha256": _digest(numerical(base)),
        "action_work": feedback.work["action"],
    })
    for field in dataclasses.fields(actual):
        if field.name == "state":
            _assert_nested_equal(actual.state.snapshot(), reference.state.snapshot())
        elif field.name == "steps":
            for left, right in zip(actual.steps, reference.steps, strict=True):
                for step_field in dataclasses.fields(left):
                    _assert_nested_equal(getattr(left, step_field.name), getattr(right, step_field.name))
        else:
            _assert_nested_equal(getattr(actual, field.name), getattr(reference, field.name))
    _assert_nested_equal(numerical(reader), numerical(base))
    _assert_nested_equal(state.snapshot(), tape_before)
    _assert_nested_equal(reference_state.snapshot(), tape_before)
    assert not torch.equal(before["address_weight"], reader.address_weight)
    assert actual.address_updated and actual.teacher_count == 2
    assert actual.work["branch_cursor_initializations"] == 3
    assert not any(key.startswith("competition_") for key in actual.work)
    assert set(feedback.teacher_choices) == {"start", "end", "operation"}
    assert sum(feedback.work["action"][f"{head}_teacher_offers"]
               for head in ("start", "end", "operation")) == 3
    assert model._pending is None and model.reader is reader
    assert teacher_started >= completed
    assert all(not p.requires_grad and p.grad is None and p.grad_fn is None
               for p in model.parameters())
    save("passed.json", {
        "schema": "competitive-splice-component-qa-v1", "status": "passed",
        "raw_sha256": raw_sha, "action_sha256": action_sha, "real_children": 1,
        "free_calls": 1, "free_decisions": 3, "teacher_calls": 2, "teacher_offers": 4,
        "action_labels": 3, "same_reader_instance": True, "teacher_numerical_parity": True,
        "address_changed": True, "task_success": False, "historical_checkpoint_loads": 0,
    })


def _competitive_receipt_qa_save(path, value):
    import json
    from types import SimpleNamespace

    def plain(item):
        if isinstance(item, SimpleNamespace):
            return {key: plain(child) for key, child in vars(item).items() if not callable(child)}
        if isinstance(item, bytes):
            return item.hex()
        if isinstance(item, dict):
            return {key: plain(child) for key, child in item.items()}
        if isinstance(item, (list, tuple)):
            return [plain(child) for child in item]
        return item

    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(plain(value), sort_keys=True, allow_nan=False) + "\n")


def _competitive_receipt_qa_trace(scores):
    from types import SimpleNamespace as Value

    n = len(scores)
    p = n * (n - 1) // 2
    currents = [float(scores[i]) - float(scores[j]) for i in range(n) for j in range(i + 1, n)]
    active = [current >= 0 for current in currents]
    winner_currents = [i - (n - 1) for i in range(n)]
    cursor = 0
    for i in range(n):
        for j in range(i + 1, n):
            winner_currents[i] += active[cursor]
            winner_currents[j] -= active[cursor]
            cursor += 1
    winners = [current >= 0 for current in winner_currents]
    work = {
        "competition_candidates": n,
        "competition_pairs": p,
        "competition_threshold_layers": 2,
        "competition_comparator_neurons": p,
        "competition_winner_neurons": n,
        "competition_comparator_synapse_applications": 2 * p,
        "competition_winner_synapse_applications": 2 * p,
        "competition_comparator_subtractions": p,
        "competition_winner_accumulations": 2 * p,
        "competition_winner_bias_values": n,
        "competition_winner_threshold_subtractions": n,
        "competition_threshold_evaluations": p + n,
        "competition_identity_synapse_applications": n,
        "competition_identity_reductions": n - 1,
        "competition_winner_count_reductions": n - 1,
        "competition_index_values_constructed": 2 * p + n,
        "competition_receipt_tensor_bytes": 13 * n + 9 * p,
        "competition_named_allocated_tensor_bytes": 45 * n + 49 * p + 16,
        "competition_pair_index_bytes": 16 * p,
        "competition_score_gather_bytes": 16 * p,
    }
    return Value(
        scores=list(scores),
        pair_currents=currents,
        comparator_activity=active,
        winner_currents=[float(value) for value in winner_currents],
        winner_activity=winners,
        winner_index=sum(i * bit for i, bit in enumerate(winners)),
        work=work,
    )


def _competitive_receipt_qa_work(n, decisions, *, trace=None, teacher=False):
    branches = 1 if trace is not None else n
    gathers = decisions if trace is not None else n * (decisions - 1)
    result = {
        "query_raw_bytes": 1,
        "query_eos_symbols": 1,
        "query_lag_shift_steps": 2,
        "query_population_normalizations": 1,
        "address_feature_slots": n,
        "address_feature_products": n * 32768,
        "address_feature_channel_reductions": n * 31744,
        "address_score_synapse_evaluations": n * 1024,
        "address_score_reductions": n * 1023,
        "address_score_evaluations": n,
        "address_normalizations": 1,
        "address_local_synapse_contributions": n * 1024 if teacher else 0,
        "address_local_slot_reductions": (n - 1) * 1024 if teacher else 0,
        "codec_slot_evaluations": n,
        "codec_output_potentials": n * 257,
        "codec_normalizations": n,
        "codec_response_gathers": gathers,
        "codec_response_values_gathered": gathers * 1028,
        "codec_response_bytes_gathered": gathers * 7196,
        "mixture_output_reductions": 257 * decisions,
        "prefix_belief_updates": branches * decisions,
        "branch_cursor_initializations": branches,
        "branch_cursor_advances": branches * (decisions - 1),
        "teacher_offers": decisions if teacher else 0,
        "codec_update_group_offers": int(teacher),
        "address_update_group_offers": int(teacher),
        "enabled_codec_update_groups": int(teacher),
        "enabled_address_update_groups": int(teacher),
    }
    if trace is not None:
        result.update(trace.work)
        result["codec_bank_slot_indices"] = n
    return result


def _competitive_receipt_qa_reader(n, decisions, *, kind, scores=None):
    import math
    from types import SimpleNamespace as Value

    successor = [*range(1, n), n - 1]
    state = Value(used=n, successor=successor)
    competitive = kind == "competitive"
    teacher = kind == "teacher"
    trace = _competitive_receipt_qa_trace(scores or [0.0] * n) if competitive else None
    belief = [0.0] if competitive else [-math.log(n)] * n
    cursors = [trace.winner_index] if competitive else list(range(n))
    steps = []
    for index in range(decisions):
        token = 65 if index < decisions - 1 else 256
        probabilities = [1 / 257] * 257 if teacher else [0.0] * 257
        if not teacher:
            probabilities[token] = 1.0
        steps.append(
            Value(
                output_step=index,
                observed_token=token,
                nll=math.log(257) if teacher else 0.0,
                mixture_probabilities=probabilities,
                prior_log_belief=list(belief),
                posterior_log_belief=list(belief),
                branch_cursors=list(cursors),
            )
        )
        cursors = [successor[cursor] for cursor in cursors]
    common = dict(
        steps=steps,
        initial_log_belief=list(belief),
        final_log_belief=list(belief),
        work=_competitive_receipt_qa_work(n, decisions, trace=trace, teacher=teacher),
    )
    if teacher:
        common.update(
            teacher_count=decisions,
            total_loss=sum(step.nll for step in steps),
            mean_loss=sum(step.nll for step in steps) / decisions,
            address_error=[0.0] * n,
            codec_updated=False,
            address_updated=False,
        )
    else:
        common["generation"] = Value(data=b"A" * (decisions - 1), stop_reason="eos")
    if competitive:
        centered = [value - max(trace.scores) for value in trace.scores]
        norm = math.log(math.fsum(math.exp(value) for value in centered))
        common.update(competition=trace, diagnostic_log_prior=[value - norm for value in centered])
    return Value(**common), state


def test_competitive_receipt_maximum_complete_row(tmp_path):
    import json
    import math
    from hashlib import sha256
    from types import SimpleNamespace as Value

    from experiments.r4_integrated import check_neuronal_address_competition_v1 as checker
    from experiments.r4_integrated import neuronal_splice_receipts_v2 as writer
    from experiments.r4_integrated.neuronal_splice_receipts_v1 import proposal_digest, tree_bytes

    free, state = _competitive_receipt_qa_reader(128, 17, kind="competitive")
    teacher, _ = _competitive_receipt_qa_reader(128, 17, kind="teacher")
    heads = {
        head: Value(logits=[0.0] * n, probabilities=[1 / n] * n, selected_index=0)
        for head, n in (("start", 516), ("end", 129), ("operation", 2))
    }
    proposal = dict(
        operation="STOP",
        path="a.py",
        start=0,
        end=0,
        replacement_hex=(b"A" * 16).hex(),
        stop_reason="eos",
        workspace_sha256="a" * 64,
    )
    prediction = Value(
        proposal=Value(to_record=lambda: dict(proposal)),
        attempt_id="authored:1",
        proposal_sha256=proposal_digest(proposal),
        model_sha256="b" * 64,
        memory_sha256="c" * 64,
        choices=heads,
        payload=free,
        work={"action": {}, "reader": free.work},
    )
    feedback = Value(
        attempt_id="authored:1",
        action_updates={head: False for head in heads},
        action_losses={head: math.log(len(value.logits)) for head, value in heads.items()},
        teacher_choices=heads,
        reader_feedback=teacher,
        work={"action": {}, "reader": teacher.work},
    )
    _competitive_receipt_qa_save(
        tmp_path / "maximum-input-raw.json",
        {"free": free, "teacher": teacher, "proposal": proposal, "heads": heads},
    )
    row = {
        "prediction": writer.compact_prediction(
            prediction, state, expected_policy="competitive_free"
        ),
        "feedback": writer.compact_feedback(feedback, state),
        "metadata": "x" * 32768,
        "host_path": "p" * 1024,
        "executable_path": "e" * 1024,
        "bootstrap": "b" * 4096,
        "stdout_hex": "ff" * 16384,
        "stderr_hex": "ff" * 16384,
        "diagnostics": ["d" * 1024, "e" * 1024],
        "ids": ["i" * 128 for _ in range(12)],
        "hashes": ["a" * 64 for _ in range(16)],
        "work": [{f"counter_{i:03d}".ljust(96, "x"): 10**12 for i in range(160)} for _ in range(4)],
    }
    _competitive_receipt_qa_save(tmp_path / "maximum-result-raw.json", row)
    encoded = writer.bounded_record(row)
    with (tmp_path / "maximum.jsonl").open("xb") as stream:
        stream.write(encoded)
    parsed = json.loads(encoded)
    for compact, policy in (
        (parsed["prediction"]["payload"], "competitive_free"),
        (parsed["feedback"]["reader"], "all_address_teacher"),
    ):
        writer.validate_reader(compact, state.successor, expected_policy=policy)
        checker.check_reader(compact, state.successor, policy)
    assert parsed == row
    assert writer.LINE_LIMIT == writer.TREE_LIMIT == 2**21
    assert len(encoded) <= writer.LINE_LIMIT
    assert tree_bytes(row) <= writer.TREE_LIMIT
    assert tree_bytes(parsed) <= writer.TREE_LIMIT
    compact = parsed["prediction"]["payload"]
    assert len(compact["competition"]["pair_currents"]) == 8128
    assert len(compact["competition"]["comparator_activity"]) == 8128
    assert compact["initial_log_belief"] == [0.0]
    assert len(parsed["feedback"]["reader"]["initial_log_belief"]) == 128
    assert all(
        set(step) == {"token", "nll", "mixture", "posterior"}
        for part in (compact, parsed["feedback"]["reader"])
        for step in part["steps"]
    )
    _competitive_receipt_qa_save(
        tmp_path / "maximum-passed.json",
        {
            "status": "passed",
            "line_bytes": len(encoded),
            "writer_tree_bytes": tree_bytes(row),
            "parsed_tree_bytes": tree_bytes(parsed),
            "sha256": sha256(encoded).hexdigest(),
            "models": 0,
            "actions": 0,
            "synthetic_step_records": 34,
        },
    )


def test_competitive_receipt_policy_and_stable_prior(tmp_path):
    import copy
    import math

    import pytest

    from experiments.r4_integrated import check_neuronal_address_competition_v1 as checker
    from experiments.r4_integrated import neuronal_splice_receipts_v2 as writer

    maximum = float.fromhex("0x1.fffffep+127")
    free, state = _competitive_receipt_qa_reader(2, 2, kind="competitive", scores=[maximum] * 2)
    marginal, _ = _competitive_receipt_qa_reader(2, 2, kind="marginal")
    teacher, _ = _competitive_receipt_qa_reader(2, 2, kind="teacher")
    _competitive_receipt_qa_save(tmp_path / "policy-input-raw.json", [free, marginal, teacher])
    comp = writer.compact_reader(free, state, expected_policy="competitive_free")
    marg = writer.compact_reader(marginal, state, expected_policy="marginal_free")
    taught = writer.compact_reader(teacher, state, expected_policy="all_address_teacher")
    _competitive_receipt_qa_save(tmp_path / "policy-result-raw.json", [comp, marg, taught])
    for row, policy in (
        (comp, "competitive_free"),
        (marg, "marginal_free"),
        (taught, "all_address_teacher"),
    ):
        writer.validate_reader(row, state.successor, expected_policy=policy)
        checker.check_reader(row, state.successor, policy)
    assert comp["diagnostic_log_prior"] == [-math.log(2)] * 2
    assert comp["competition"]["pair_currents"] == [0.0]
    assert comp["competition"]["winner_index"] == 0
    assert "schema" in comp and "schema" not in marg and "schema" not in taught
    extra = copy.deepcopy(comp)
    extra["unexpected"] = 1
    negatives = [
        ("competitive_as_marginal", comp, "marginal_free"),
        ("marginal_as_competitive", marg, "competitive_free"),
        ("teacher_as_free", taught, "competitive_free"),
        ("free_as_teacher", comp, "all_address_teacher"),
        ("extra_field", extra, "competitive_free"),
    ]
    _competitive_receipt_qa_save(tmp_path / "policy-negative-raw.json", negatives)
    for _, row, policy in negatives:
        with pytest.raises(ValueError):
            writer.validate_reader(row, state.successor, expected_policy=policy)
        with pytest.raises(ValueError):
            checker.check_reader(row, state.successor, policy)
    _competitive_receipt_qa_save(
        tmp_path / "policy-passed.json",
        {
            "status": "passed",
            "valid_policy_checks_per_validator": 3,
            "expected_rejections_per_validator": 5,
            "models": 0,
            "actions": 0,
        },
    )


def test_competitive_receipt_trace_tampering_and_bounds(tmp_path):
    import copy
    import json
    import math
    import sys

    import pytest

    from experiments.r4_integrated import check_neuronal_address_competition_v1 as checker
    from experiments.r4_integrated import neuronal_splice_receipts_v2 as writer
    from experiments.r4_integrated.neuronal_splice_receipts_v1 import tree_bytes

    free, state = _competitive_receipt_qa_reader(2, 2, kind="competitive", scores=[1.0, 0.0])
    _competitive_receipt_qa_save(tmp_path / "tamper-input-raw.json", free)
    row = writer.compact_reader(free, state, expected_policy="competitive_free")
    _competitive_receipt_qa_save(tmp_path / "tamper-result-raw.json", row)
    writer.validate_reader(row, state.successor, expected_policy="competitive_free")
    checker.check_reader(row, state.successor, "competitive_free")
    changes = []
    for field, index, replacement in (
        ("pair_currents", 0, 2.0),
        ("comparator_activity", 0, False),
        ("winner_currents", 0, -1.0),
        ("winner_activity", 0, False),
    ):
        changed = copy.deepcopy(row)
        changed["competition"][field][index] = replacement
        changes.append((field, changed))
    changed = copy.deepcopy(row)
    changed["competition"]["winner_index"] = 1
    changes.append(("winner_index", changed))
    changed = copy.deepcopy(row)
    changed["diagnostic_log_prior"][0] += 0.1
    changes.append(("diagnostic_prior", changed))
    changed = copy.deepcopy(row)
    del changed["competition"]
    changes.append(("missing_trace", changed))
    changed = copy.deepcopy(row)
    changed["competition"]["work"]["competition_pairs"] += 1
    changes.append(("wrong_work", changed))
    changed = copy.deepcopy(row)
    changed["initial_log_belief"] = [-math.log(2)] * 2
    changes.append(("wrong_branch_width", changed))
    changed = copy.deepcopy(row)
    changed["steps"][0]["token"] = 66
    changes.append(("nonmaximum_output", changed))
    changed = copy.deepcopy(row)
    changed["steps"][0]["cursor"] = 0
    changes.append(("extra_compact_cursor", changed))
    _competitive_receipt_qa_save(tmp_path / "tamper-negative-raw.json", changes)
    for _, changed in changes:
        with pytest.raises(ValueError):
            writer.validate_reader(changed, state.successor, expected_policy="competitive_free")
        with pytest.raises(ValueError):
            checker.check_reader(changed, state.successor, "competitive_free")
    raw_changes = []
    changed = copy.deepcopy(free)
    changed.steps[1].prior_log_belief[0] = 0.1
    raw_changes.append(("omitted_prior", changed))
    changed = copy.deepcopy(free)
    changed.steps[1].branch_cursors[0] = 0
    raw_changes.append(("omitted_cursor", changed))
    changed = copy.deepcopy(free)
    changed.final_log_belief[0] = 0.1
    raw_changes.append(("omitted_final", changed))
    _competitive_receipt_qa_save(tmp_path / "omission-negative-raw.json", raw_changes)
    for _, changed in raw_changes:
        with pytest.raises(ValueError):
            writer.compact_reader(changed, state, expected_policy="competitive_free")
    expanded = {"text": "\u0080" * (writer.LINE_LIMIT // 6 + 1)}
    tree_heavy = {"values": [0] * (writer.TREE_LIMIT // sys.getsizeof(0) + 1)}
    sizes = {
        "expanded_tree": tree_bytes(expanded),
        "expanded_line": len(json.dumps(expanded, ensure_ascii=True).encode("ascii")) + 1,
        "heavy_tree": tree_bytes(tree_heavy),
        "heavy_line": len(json.dumps(tree_heavy, ensure_ascii=True).encode("ascii")) + 1,
    }
    _competitive_receipt_qa_save(tmp_path / "guard-preconditions-raw.json", sizes)
    assert sizes["expanded_tree"] <= writer.TREE_LIMIT < sizes["heavy_tree"]
    assert sizes["heavy_line"] <= writer.LINE_LIMIT < sizes["expanded_line"]
    for oversized in (expanded, tree_heavy):
        with pytest.raises(ValueError):
            writer.bounded_record(oversized)
    assert len(changes) == 11 and len(raw_changes) == 3
    _competitive_receipt_qa_save(
        tmp_path / "tamper-passed.json",
        {
            "status": "passed",
            "compact_rejections_per_validator": len(changes),
            "raw_compactor_rejections": len(raw_changes),
            "guard_rejections": 2,
            "models": 0,
            "actions": 0,
        },
    )


def test_competitive_splice_prior_worker_exchange():
    """Five strict reader continuations; one competitive worker action, no teaching."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from experiments.r4_integrated.check_neuronal_address_competition_v1 import check_reader
    from experiments.r4_integrated.check_neuronal_splice_online_v1 import initial_action_tensors
    from experiments.r4_integrated.neuronal_splice_endpoint_v2 import (
        check_acknowledgement,
        check_proposal,
    )
    from experiments.r4_integrated.neuronal_splice_io_v1 import (
        JsonlJournal,
        process_identity,
        read_row,
        write_message,
    )
    from experiments.r4_integrated.neuronal_splice_models_v2 import ARMS, reader_policy
    from experiments.r4_integrated.neuronal_splice_online_v1 import (
        execute_action,
        qualified_launch,
        workspace_from_records,
    )
    from experiments.r4_integrated.neuronal_splice_prior_v2 import initialize_models, tensor_bytes
    from experiments.r4_integrated.neuronal_splice_receipts_v2 import validate_prediction
    from experiments.r4_integrated.neuronal_splice_worker_v2 import (
        LIMITS,
        SOURCE_MODULES,
        WORKER_SOURCE,
        check_requests,
        file_hash,
        write_json,
    )
    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
    from humanbrain.local_memory_learner import LocalMemoryLearner
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest

    root = Path(__file__).resolve().parents[1]
    output = root / ".pytest-tmp/competitive-splice-prior-worker-qa-v1"
    output.mkdir(exist_ok=False)
    deadline = float(os.environ["HUMANBRAIN_RUN_DEADLINE_UNIX"])
    assert 0 < deadline - time.time() <= 60
    spec = SimpleNamespace(config={"receipt_limits": LIMITS})
    baseline_paths = {
        *SOURCE_MODULES.values(),
        WORKER_SOURCE,
        "experiments/r4_integrated/neuronal_splice_prior_v2.py",
        "experiments/r4_integrated/neuronal_splice_prior_v1.py",
        "experiments/r4_integrated/neuronal_splice_endpoint_v2.py",
        "experiments/r4_integrated/check_neuronal_address_competition_v1.py",
        "experiments/r4_integrated/check_neuronal_splice_online_v1.py",
        "experiments/r4_integrated/neuronal_splice_online_v1.py",
    }
    baseline = {
        "source_sha256": {name: file_hash(root / name) for name in sorted(baseline_paths)},
        "test_sha256": file_hash(Path(__file__)),
        "parent_identity": process_identity(),
    }
    write_json(output / "before-initialization.json", baseline)
    models, custody = initialize_models(root)
    write_json(output / "prior.json", custody)
    assert (
        tuple(models)
        == ARMS
        == ("learned", "action_frozen", "slow_frozen", "memory_disabled", "marginal_reference")
    )
    assert custody["schema"] == "neuronal-splice-prior-custody-v2"
    assert custody["actual_loaded_payload_tensor_bytes"] == 33801664
    assert custody["historical_payload_tensor_bytes"] == 33801664
    assert custody["new_model_tensor_bytes"] == 3172160
    assert custody["selected_entry"] == "learned.weights"
    assert custody["metadata_fields_changed"]["marginal_reference"] == []
    assert custody["reader_policy"] == {arm: reader_policy(arm) for arm in ARMS}
    initial_actions = initial_action_tensors(torch)
    marginal_metadata = models["marginal_reference"].reader.get_extra_state()
    reader_tensors = {
        key: value
        for key, value in models["marginal_reference"].reader.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    assert _digest(reader_tensors) == custody["initial_reader_tensor_sha256"]
    allowed_metadata = {
        "mode",
        "schema",
        "recall",
        "competition",
        "free_belief",
        "codec_activity",
        "teacher_phase",
    }
    initial_model_hashes = {}
    for arm, current in models.items():
        mode = "learned" if arm == "marginal_reference" else arm
        expected_type = (
            LocalMemoryLearner if arm == "marginal_reference" else CompetitiveLocalMemoryLearner
        )
        assert type(current.reader) is expected_type
        assert current.mode == mode and current.reader.mode == NeuronalSpliceLearner.MODES[mode]
        assert current._pending is None and current._experience == 0
        assert all(
            not value.requires_grad
            and value.grad is None
            and value.grad_fn is None
            and torch.isfinite(value).all()
            for value in current.parameters()
        )
        assert all(
            torch.equal(current.state_dict()[key], value) for key, value in initial_actions.items()
        )
        assert all(
            torch.equal(current.reader.state_dict()[key], value)
            for key, value in reader_tensors.items()
        )
        metadata = current.reader.get_extra_state()
        changed = sorted(
            key
            for key in set(metadata) | set(marginal_metadata)
            if metadata.get(key) != marginal_metadata.get(key)
        )
        assert changed == custody["metadata_fields_changed"][arm]
        assert set(changed) <= allowed_metadata
        assert metadata["mode"] == current.reader.mode
        if arm != "marginal_reference":
            assert metadata["schema"] == "local-neuronal-competitive-address-learner-v1"
        initial_model_hashes[arm] = _digest(current.state_dict())
    assert initial_model_hashes == custody["initial_model_sha256"]
    for key in reader_tensors:
        assert (
            len({current.reader.state_dict()[key].data_ptr() for current in models.values()}) == 5
        )
    for key in initial_actions:
        assert len({current.state_dict()[key].data_ptr() for current in models.values()}) == 5
    model = models["learned"]
    document = b"Record k00 has value ab."
    state = model.ingest([document])
    assert len(document) == 24 and state.used == 25
    successor = state.successor[: state.used].tolist()
    model_hash, memory_hash = _digest(model.state_dict()), _digest(state.snapshot())
    workspace = workspace_from_records(
        [
            {"path": "a.py", "content_hex": b"x = 'old'\n".hex()},
            {"path": "main.py", "content_hex": b"import a\nprint(a.x)\n".hex()},
        ]
    )
    task = {
        "task_id": "endpoint-0",
        "memory_id": "qa-memory",
        "request_hex": b"SET a.py x k00".hex(),
        "workspace_sha256": workspace.sha256,
        "files": [
            {"path": item.path, "content_hex": item.content.hex()} for item in workspace.files
        ],
    }
    package = output / "package"
    package.mkdir()
    sources = {}
    for relative in (*SOURCE_MODULES.values(), WORKER_SOURCE):
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
        assert file_hash(destination) == file_hash(root / relative)
        sources[relative] = file_hash(destination)
    assert len(sources) == 16
    assert {path.relative_to(package).as_posix() for path in package.rglob("*.py")} == set(sources)
    assert all(digest == baseline["source_sha256"][name] for name, digest in sources.items())
    checkpoint = output / "source_free.pt"
    payload = {
        "schema": "neuronal-splice-single-exchange-qa-state-v2",
        "arms": {
            "learned": {
                "weights": model.state_dict(),
                "memories": {"qa-memory": state.snapshot()},
                "experience": 0,
            }
        },
    }
    assert tensor_bytes(payload) == 1291336
    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)
    del payload
    checkpoint_hash = file_hash(checkpoint)
    requests = {
        "schema": "neuronal-splice-single-exchange-qa-request-v2",
        "checkpoint_sha256": checkpoint_hash,
        "source_sha256": sources,
        "receipt_limits": LIMITS,
        "arms": ["learned"],
        "reading_cases": [],
        "cases": [
            {
                "id": task["task_id"],
                **{key: task[key] for key in ("memory_id", "request_hex", "files")},
            }
        ],
    }
    check_requests(requests, single_exchange_qa=True)
    with pytest.raises(ValueError):
        check_requests(requests)
    request_path = output / "query_only.json"
    write_json(request_path, requests)
    requests_hash = file_hash(request_path)
    qualification_paths = {
        ".pytest-tmp/temporary-module-bootstrap-v1-qualification.json": (
            "0bb809fd33aee145f169776a03a884fd62daf381ff4e67ed333a2a378c46d92b"
        ),
        ".pytest-tmp/stdlib-python-action-v1-qualification.json": (
            "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c"
        ),
    }
    assert all(file_hash(root / name) == digest for name, digest in qualification_paths.items())
    launch = qualified_launch(root)
    write_json(
        output / "qualification.json",
        {
            "qualification_sha256": qualification_paths,
            "launch_record": launch.to_record(),
            "launch_record_sha256": launch.sha256,
        },
    )
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(package / WORKER_SOURCE),
        "--package-root",
        str(package),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha",
        checkpoint_hash,
        "--requests",
        str(request_path),
        "--requests-sha",
        requests_hash,
        "--output",
        str(output / "worker"),
        "--deadline-unix",
        repr(deadline),
        "--single-exchange-qa",
    ]
    action_journal = JsonlJournal(output / "actions.jsonl", spec)
    try:
        with (output / "worker-stderr.bin").open("xb") as stderr:
            worker_started_unix = time.time()
            process = subprocess.Popen(
                command, cwd=output, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr
            )
            try:
                launcher_identity = process_identity(process.pid)
                write_json(
                    output / "worker-launch.json",
                    {
                        "command": command,
                        "cwd": str(output),
                        "pid": process.pid,
                        "process_identity": launcher_identity,
                        "parent_pid": os.getpid(),
                        "baseline_sha256": file_hash(output / "before-initialization.json"),
                        "parent_identity": baseline["parent_identity"],
                        "started_unix": worker_started_unix,
                    },
                )
                proposal_row = read_row(process.stdout, spec)
                write_json(output / "proposal.json", proposal_row)
                proposal = check_proposal(
                    proposal_row, "learned", 0, task, model_hash, memory_hash, successor=successor
                )
                validate_prediction(
                    proposal_row["prediction"],
                    successor,
                    expected_policy="competitive_free",
                    boundaries=workspace.boundaries,
                )
                payload_row = proposal_row["prediction"]["payload"]
                summary = check_reader(payload_row, successor, "competitive_free")
                assert payload_row["work"]["query_raw_bytes"] == 14
                assert len(payload_row["competition"]["scores"]) == 25
                assert len(payload_row["competition"]["pair_currents"]) == 300
                assert len(summary["final_cursors"]) == 1
                assert payload_row["work"]["codec_slot_evaluations"] == 25
                assert payload_row["work"]["codec_response_gathers"] == summary["decisions"]
                identity = {
                    key: proposal_row[key]
                    for key in ("schema", "type", "arm", "index", "id", "attempt_id")
                }
                envelope, action_hash, _, _ = execute_action(
                    task=None,
                    workspace=workspace,
                    proposal=proposal,
                    attempt_id=identity["attempt_id"],
                    phase="endpoint",
                    arm="learned",
                    index=0,
                    output=output,
                    launch=launch,
                    journal=action_journal,
                )
                write_message(
                    process.stdin,
                    {
                        **identity,
                        "type": "executed",
                        "action_sha256": action_hash,
                        "outcome": envelope,
                    },
                    spec,
                )
                acknowledgement = read_row(process.stdout, spec)
                write_json(output / "acknowledgement.json", acknowledgement)
                check_acknowledgement(
                    acknowledgement, identity, action_hash, model_hash, memory_hash
                )
                with pytest.raises(ValueError):
                    check_acknowledgement(
                        {**acknowledgement, "action_sha256": "0" * 64},
                        identity,
                        action_hash,
                        model_hash,
                        memory_hash,
                    )
                process.stdin.close()
                process.wait(timeout=max(0.001, deadline - time.time()))
                assert process.returncode == 0 and read_row(process.stdout, spec) is None
                worker_finished_unix = time.time()
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
                if not process.stdin.closed:
                    process.stdin.close()
    finally:
        action_log = action_journal.close()
    completion = json.loads((output / "worker/completion.json").read_bytes())
    assert completion["schema"] == "neuronal-splice-endpoint-completion-v2"
    assert completion["scope"] == "single_exchange_qa" and completion["status"] == "passed"
    assert completion["pid"] == process.pid or completion["parent_pid"] == process.pid
    worker_identity = completion["process_identity"]
    assert worker_identity["pid"] == completion["pid"]
    assert baseline["parent_identity"]["creation_ticks"] <= launcher_identity["creation_ticks"]
    assert launcher_identity["creation_ticks"] <= worker_identity["creation_ticks"]
    if completion["pid"] == process.pid:
        assert worker_identity == launcher_identity and completion["parent_pid"] == os.getpid()
    else:
        assert completion["parent_pid"] == launcher_identity["pid"]
    worker_created_unix = (worker_identity["creation_ticks"] - 621355968000000000) / 10**7
    assert worker_started_unix - 0.001 <= worker_created_unix <= worker_finished_unix + 0.001
    assert completion["checkpoint_sha256"] == file_hash(checkpoint) == checkpoint_hash
    assert completion["requests_sha256"] == file_hash(request_path) == requests_hash
    assert completion["source_sha256"] == sources
    assert completion["source_origins"] == {
        key: str(package / path) for key, path in SOURCE_MODULES.items()
    }
    assert completion["final"] == {
        "learned": {
            "model_sha256": model_hash,
            "memories_sha256": {"qa-memory": memory_hash},
            "experience": 1,
        }
    }
    assert completion["endpoint_log"]["rows"] == 2 and completion["reading_log"]["rows"] == 0
    for name, filename in (("endpoint_log", "endpoints.jsonl"), ("reading_log", "reading.jsonl")):
        log = completion[name]
        log_path = output / "worker" / filename
        assert set(log) == {
            "path",
            "rows",
            "bytes",
            "sha256",
            "max_line_bytes",
            "max_row_tree_bytes",
        }
        assert log["path"] == filename and log["bytes"] == log_path.stat().st_size
        assert log["sha256"] == file_hash(log_path)
        assert 0 <= log["max_line_bytes"] <= LIMITS["max_line_bytes"]
        assert 0 <= log["max_row_tree_bytes"] <= LIMITS["max_row_tree_bytes"]
        if log["rows"]:
            assert log["bytes"] > 0 and log["max_line_bytes"] > 0 and log["max_row_tree_bytes"] > 0
        else:
            assert log["bytes"] == log["max_line_bytes"] == log["max_row_tree_bytes"] == 0
    assert (output / "worker-stderr.bin").stat().st_size <= 1024
    assert _digest(model.state_dict()) == model_hash and _digest(state.snapshot()) == memory_hash
    assert {
        arm: _digest(current.state_dict()) for arm, current in models.items()
    } == initial_model_hashes
    assert all(current._pending is None and current._experience == 0 for current in models.values())
    assert all(
        file_hash(root / relative) == file_hash(package / relative) == digest
        for relative, digest in sources.items()
    )
    assert all(file_hash(root / name) == digest for name, digest in qualification_paths.items())
    assert {name: file_hash(root / name) for name in baseline["source_sha256"]} == baseline[
        "source_sha256"
    ]
    assert file_hash(Path(__file__)) == baseline["test_sha256"]
    assert action_log["rows"] == 1
    assert acknowledgement["work"] == {
        "action": {"custody_tensor_bytes_hashed": 1291336},
        "reader": {},
    }
    steps = proposal_row["prediction"]["payload"]["steps"]
    assert 1 <= len(steps) <= 17
    write_json(
        output / "passed.json",
        {
            "schema": "competitive-splice-prior-worker-qa-v1",
            "status": "passed",
            "parent_pid": os.getpid(),
            "baseline_sha256": file_hash(output / "before-initialization.json"),
            "launcher_identity": launcher_identity,
            "worker_identity": completion["process_identity"],
            "action_log": action_log,
            "prior": custody,
            "worker_completion_sha256": file_hash(output / "worker/completion.json"),
            "source_sha256": sources,
            "test_sha256": file_hash(Path(__file__)),
            "combined_model_constructions": 6,
            "parent_models": 5,
            "reader_prior_copies": 5,
            "prior_loads": 1,
            "worker_checkpoint_loads": 1,
            "ingests": 1,
            "document_raw_bytes": len(document),
            "document_eos": 1,
            "free_proposals": 1,
            "actual_payload_decisions": len(steps),
            "action_children": 1,
            "worker_processes": 1,
            "teachers": 0,
            "reading_panel_calls": 0,
            "checkpoint_output_probes": 0,
            "query_raw_bytes": 14,
            "query_eos": 1,
            "qualification_sha256": qualification_paths,
            "launch_record_sha256": launch.sha256,
            "reader_policy": reader_policy("learned"),
            "initial_model_sha256": initial_model_hashes,
            "proposal_sha256": proposal.sha256,
            "action_line_sha256": action_hash,
            "qualification_scope": (
                "Same pending neuronal worker through actual execution; no acquired task skill."
            ),
        },
    )


def test_competitive_splice_roster_data_contract():
    """One new roster construction; saved v1 comparison; no model or program execution."""
    import ast
    import json
    from collections import Counter
    from hashlib import sha256
    from pathlib import Path

    from experiments.r4_integrated import neuronal_splice_roster_v2 as data

    root = Path(__file__).resolve().parents[1]
    old_path = root / ".pytest-tmp/local-joint-address-online-v6-run/offline_teacher_roster.json"
    previous_path = root / ".pytest-tmp/neuronal-splice-roster-qa-generated-v1.json"
    old_source = root / "experiments/r4_integrated/neuronal_splice_roster_v1.py"
    new_source = root / "experiments/r4_integrated/neuronal_splice_roster_v2.py"
    bindings = {
        old_path: "009b42e22976733bba207e46337ec23d5f6de5ab15ab061bc6fd08539e44a2b1",
        previous_path: "245a0d908ded5d4bbf4216028d062a9609597561d4c63c67c50f69fd164de5e9",
        old_source: "b814edae2d3bc0da5d6be85ef1adb36f85b411522b189ee376617c192f194d77",
        new_source: "4b7f7fd5203462d039a1867d6c4ab23761431e6c85aa81da898d43fbc1eab413",
    }
    for path, expected in bindings.items():
        assert sha256(path.read_bytes()).hexdigest() == expected
    previous = json.loads(previous_path.read_bytes())
    output = root / ".pytest-tmp/neuronal-splice-roster-qa-generated-v2.json"
    raw_receipt = root / ".pytest-tmp/neuronal-splice-roster-data-qa-v2-raw.json"
    passed = root / ".pytest-tmp/neuronal-splice-roster-data-qa-v2.json"
    assert not any(path.exists() for path in (output, raw_receipt, passed))
    roster = data.build_roster(old_path)  # The sole constructor call.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(roster, sort_keys=True, allow_nan=False) + "\n")
    receipt = {
        "schema": "competitive-splice-roster-data-qa-v1", "status": "pending_checks",
        "source_inputs": {str(path): value for path, value in bindings.items()},
        "tested_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_path": str(output), "generated_sha256": sha256(output.read_bytes()).hexdigest(),
        "builder_calls": 1, "v1_builder_calls": 0, "models": 0, "children": 0,
        "checkpoint_loads": 0, "program_executions": 0,
    }
    with raw_receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)

    assert roster["schema"] == "neuronal-splice-roster-v2"
    assert roster["source_inputs"] == previous["source_inputs"]
    for name in ("train_task_ids", "schedule", "cost_task_ids", "retention_task_ids", "old_panel"):
        assert roster[name] == previous[name]
    tasks, memories = roster["tasks"], roster["memory_packages"]
    for task_id in roster["train_task_ids"]:
        assert tasks[task_id] == previous["tasks"][task_id]
        memory_id = tasks[task_id]["memory_id"]
        assert memories[memory_id] == previous["memory_packages"][memory_id]
    old_ids = {row["memory_id"] for row in roster["old_panel"]}
    assert len(old_ids) == 12
    for memory_id in old_ids:
        assert memories[memory_id] == previous["memory_packages"][memory_id]
    assert Counter(roster["schedule"]) == {f"T{i:02d}": 8 for i in range(32)}
    assert roster["fresh_task_ids"] == [f"{g}{r}-{m}" for g in "DCVO"
                                         for r in range(4) for m in range(2)]
    assert set(tasks) == set(roster["train_task_ids"] + roster["fresh_task_ids"])
    assert Counter(tasks[key]["required_operation"] for key in roster["fresh_task_ids"]) == {
        "SPLICE": 28, "STOP": 4}
    assert roster["counts"] == {
        "tasks": 64, "train_tasks": 32, "fresh_tasks": 32, "retention_tasks": 16,
        "lessons_per_arm": 256, "memory_packages_per_arm": 72,
        "before_learning_packages": 48, "after_learning_packages": 24,
        "documents_per_arm": 168, "document_raw_bytes_per_arm": 4022,
        "document_eos_per_arm": 168, "ingestion_symbols_per_arm": 4190,
        "maximum_memory_slots": 125, "reader_teacher_offers_per_arm": 480,
        "action_teacher_labels_per_arm": 640, "full_action_verifications": 1520,
        "old_panel_reader_calls": 200, "full_reader_teacher_offers": 2400,
        "full_action_teacher_labels": 3200,
    }
    docs = [bytes.fromhex(raw) for package in memories.values() for raw in package["document_hex"]]
    assert (len(docs), sum(map(len, docs))) == (168, 4022)
    before_ids, after_ids = (roster[f"{stage}_memory_ids"] for stage in
                             ("before_learning", "after_learning"))
    assert len(before_ids) == 48 and len(after_ids) == 24
    assert not set(before_ids) & set(after_ids)
    assert set(before_ids + after_ids) == set(memories)
    for stage, ids in (("before_learning", before_ids), ("after_learning", after_ids)):
        assert {key for key, value in memories.items() if value["stage"] == stage} == set(ids)
    fresh_memory_ids = {tasks[key]["memory_id"] for key in roster["fresh_task_ids"]}
    assert len(fresh_memory_ids) == 28
    assert fresh_memory_ids & set(before_ids) == {f"M-V{r}-{m}" for r in range(2) for m in range(2)}
    assert all(sum(len(bytes.fromhex(raw)) + 1 for raw in memory["document_hex"]) <= 128
               for memory in memories.values())

    syntax_parses = literal_reads = 0
    for task in tasks.values():
        before = {f["path"]: bytes.fromhex(f["content_hex"]) for f in task["files"]}
        expected_after = {f["path"]: bytes.fromhex(f["content_hex"])
                          for f in task["expected_after_files"]}
        assert len(before) == 3 and set(before) == set(expected_after)
        assert max(len(path.encode()) for path in before) <= 31
        assert 0 < len(bytes.fromhex(task["request_hex"])) <= 31
        teacher, actual = task["teacher"], dict(before)
        # Independent generic digest, not the constructor's helper.
        digest = sha256(b"humanbrain.raw-splice-workspace.v1\0")
        digest.update(len(before).to_bytes(4, "big"))
        for path in sorted(before, key=lambda value: value.encode()):
            for raw in (path.encode(), before[path]):
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
        assert task["workspace_sha256"] == teacher["workspace_sha256"] == digest.hexdigest()
        assert teacher["operation"] == task["required_operation"]
        assert teacher["stop_reason"] == "eos"
        if teacher["operation"] == "SPLICE":
            path, start, end = teacher["path"], teacher["start"], teacher["end"]
            replacement = bytes.fromhex(teacher["replacement_hex"])
            assert 0 <= start <= end <= len(before[path]) and len(replacement) <= 16
            actual[path] = before[path][:start] + replacement + before[path][end:]
        else:
            assert teacher["operation"] == "STOP" and teacher["replacement_hex"] == ""
        assert actual == expected_after
        after_trees = {}
        for workspace in (before, actual):
            for path, raw in workspace.items():
                assert len(raw) <= 128
                tree = ast.parse(raw.decode("ascii"), filename=path)
                syntax_parses += 1
                if workspace is actual:
                    after_trees[path] = tree
        values = {}
        for path, tree in after_trees.items():
            if path == "main.py":
                continue
            values[path[:-3]] = {}
            assert len(tree.body) == 2
            for assignment in tree.body:
                assert isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                assert isinstance(assignment.targets[0], ast.Name)
                value = ast.literal_eval(assignment.value)
                literal_reads += 1
                assert isinstance(value, str)
                values[path[:-3]][assignment.targets[0].id] = value.encode("ascii")
            assert set(values[path[:-3]]) == {"x", "y"}
        main = after_trees["main.py"].body
        assert len(main) == 2 and isinstance(main[0], ast.Import)
        assert len(main[0].names) == 2 and all(alias.asname is None for alias in main[0].names)
        names = [alias.name for alias in main[0].names]
        assert set(names) == set(values)
        assert isinstance(main[1], ast.Expr) and isinstance(main[1].value, ast.Call)
        call = main[1].value
        assert isinstance(call.func, ast.Name) and call.func.id == "print"
        assert len(call.args) == 4 and len(call.keywords) == 1
        assert call.keywords[0].arg == "sep" and ast.literal_eval(call.keywords[0].value) == "|"
        argument_names = [(n, v) for n in names for v in ("x", "y")]
        for arg, (name, variable) in zip(call.args, argument_names, strict=True):
            assert isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
            assert (arg.value.id, arg.attr) == (name, variable)
        stdout = b"|".join(values[name][var] for name in names for var in ("x", "y")) + b"\n"
        assert stdout.hex() == task["expected_stdout_hex"]
        observation = data.task_observation(task)
        assert set(observation) == {
            "task_id", "memory_id", "request_hex", "files", "workspace_sha256"}
        assert all(set(item) == {"path", "content_hex"} for item in observation["files"])
        assert all(observation[key] == task[key] for key in observation)
        observation["files"][0]["content_hex"] = ""
        assert task["files"][0]["content_hex"] != ""
    assert (syntax_parses, literal_reads) == (384, 256)

    continuation = ((b"ab", b"ababab"), (b"ba", b"bababa"),
                    (b"cd", b"cdcdcd"), (b"dc", b"dcdcdc"))
    role_values = ((b"ab", b"ba"), (b"cd", b"dc"), (b"ac", b"ca"), (b"bd", b"db"))
    for group in "DCVO":
        for r in range(4):
            left, right = (tasks[f"{group}{r}-{m}"] for m in range(2))
            assert left["pair_id"] == right["pair_id"] == f"{group}{r}"
            assert (left["member"], right["member"]) == (0, 1)
            assert left["files"] == right["files"]
            documents = [[bytes.fromhex(raw) for raw in memories[t["memory_id"]]["document_hex"]]
                         for t in (left, right)]
            payloads = tuple(bytes.fromhex(t["teacher"]["replacement_hex"]) for t in (left, right))
            if group != "O":
                assert left["request_hex"] == right["request_hex"]
                assert left["memory_id"] != right["memory_id"]
                for field in ("path", "start", "end", "operation"):
                    assert left["teacher"][field] == right["teacher"][field]
            if group == "D":
                assert len(documents[0]) == 2 and documents[0] == documents[1][::-1]
                assert left["teacher"] == right["teacher"]
                assert left["expected_after_files"] == right["expected_after_files"]
            elif group == "C":
                assert payloads == continuation[r] and payloads[0] == payloads[1][:2]
                assert documents[0][1] == documents[1][1]
                assert len(documents[0]) == len(documents[1]) == 2
                assert documents[1][0] == documents[0][0][:-3] + payloads[1] + b"."
                assert left["expected_after_files"] != right["expected_after_files"]
            elif group == "V":
                assert payloads == role_values[r]
                expected_slots = 50 if r < 2 else 125
                for member, rows in enumerate(documents):
                    assert sum(len(raw) + 1 for raw in rows) == expected_slots
                    prefix = f"Record k{88 + r} has value ".encode()
                    assert rows[0] == prefix + payloads[member] + b"."
                    assert rows[1] == b"Record k96 has value " + payloads[1 - member] + b"."
                assert documents[0][2:] == documents[1][2:]
                assert left["expected_after_files"] != right["expected_after_files"]
            else:
                assert left["memory_id"] == right["memory_id"] and documents[0] == documents[1]
                assert bytes.fromhex(left["request_hex"]).startswith(b"SET ")
                assert bytes.fromhex(right["request_hex"]) == (
                    b"KEEP " + bytes.fromhex(left["request_hex"])[4:])
                raw = next(bytes.fromhex(f["content_hex"]) for f in left["files"]
                           if f["path"] == left["teacher"]["path"])
                if r < 2:
                    assert b"\t" in raw
                    if r == 0:
                        assert b"  " in raw
                    assert left["teacher"]["end"] - left["teacher"]["start"] == 3
                elif r == 2:
                    assert raw.startswith(b"\n# insertion\n\n")
                    assert left["teacher"]["start"] == left["teacher"]["end"] and payloads[0]
                else:
                    assert raw.startswith(b"# deletion\n\n") and payloads[0] == b""
                    assert left["teacher"]["end"] > left["teacher"]["start"]
    exposed_paths = {f["path"] for task in previous["tasks"].values() for f in task["files"]}
    fresh_paths = {f["path"] for key in roster["fresh_task_ids"] for f in tasks[key]["files"]}
    assert exposed_paths & fresh_paths == {"main.py"}
    fresh_keys = {bytes.fromhex(raw)[7:10] for key in fresh_memory_ids
                  for raw in memories[key]["document_hex"]}
    exposed_keys = {bytes.fromhex(raw)[7:10] for key, value in previous["memory_packages"].items()
                    if not key.startswith("M-old-") for raw in value["document_hex"]}
    assert not fresh_keys & exposed_keys
    assert fresh_keys == {f"k{i}".encode() for i in range(80, 100)}
    for path, expected in bindings.items():
        assert sha256(path.read_bytes()).hexdigest() == expected
    receipt.update(status="passed", syntax_parses=syntax_parses,
                   assignment_literal_reads=literal_reads, paired_contrasts=16,
                   raw_receipt_sha256=sha256(raw_receipt.read_bytes()).hexdigest())
    with passed.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)


def _competitive_cost_synthetic_inputs():
    """Plain authored durations; no production call or experiment artifact."""
    arms = ("learned", "action_frozen", "slow_frozen", "memory_disabled", "marginal_reference")
    timing, pairs = {}, {}
    for arm in arms:
        timing[arm] = {
            "regular": {
                "rows": 4,
                "model_seconds": 0.04,
                "model_max_seconds": 0.01,
                "action_seconds": 0.08,
                "action_max_seconds": 0.02,
                "journal_seconds": 0.004,
                "journal_max_seconds": 0.001,
            },
            "cap": {
                "rows": 1,
                "model_seconds": 0.04 if arm == arms[-1] else 0.03,
                "model_max_seconds": 0.04 if arm == arms[-1] else 0.03,
                "action_seconds": 0.015,
                "action_max_seconds": 0.015,
                "journal_seconds": 0.002,
                "journal_max_seconds": 0.002,
            },
        }
        pairs[arm] = {
            "regular": {"rows": 4, "seconds": 0.004, "max_seconds": 0.001},
            "cap": {
                "rows": 1,
                "seconds": 0.003 if arm == arms[-1] else 0.002,
                "max_seconds": 0.003 if arm == arms[-1] else 0.002,
            },
        }
    report = {
        "mode": "cost",
        "status": "completed",
        "timing_summary": timing,
        "category_probes": [{"action_seconds": 0.025} for _ in range(4)],
        "memory": {"peak_working_set_mib": 300.0},
        "worker": {"worker": {"memory": {"peak_working_set_mib": 200.0}}},
    }
    outer = _competitive_cost_monitor(101, [], 20.0, 550.0)
    audit = {"status": "passed", "pid": 202, "learning_pair_timing": pairs}
    audit_outer = _competitive_cost_monitor(202, [], 5.0, 250.0)
    return report, outer, audit, audit_outer


def _competitive_cost_monitor(pid, command, seconds, peak):
    return {
        "status": "passed",
        "exit_code": 0,
        "seconds_limit": 900,
        "tree_peak_sum_mib_limit": 2048,
        "started_unix": 1000.0,
        "deadline_unix": 1900.0,
        "elapsed_seconds": seconds,
        "observed_tree_peak_sum_mib": peak,
        "command": command,
        "child_pid": pid,
        "launcher_sha256": "a" * 64,
        "observed_process_peaks_mib": {f"{pid}/{621355968000000000 + 1001 * 10_000_000}": peak},
    }


def _competitive_cost_authored_projection(report, outer, audit, audit_outer, *, finished):
    """Authored expected JSON, retaining float operation order for exact custody."""
    maxima = {
        arm: {
            "model": 0.04 if arm == "marginal_reference" else 0.03,
            "action": 0.025,
            "journal": 0.002,
        }
        for arm in report["timing_summary"]
    }
    sampled = sum(
        row["regular"][stage + "_seconds"]
        for row in report["timing_summary"].values()
        for stage in ("model", "action", "journal")
    )
    run = (
        512 * sum(sum(row.values()) for row in maxima.values())
        + 2 * (outer["elapsed_seconds"] - sampled)
        + 30
    )
    sampled_audit = sum(row["regular"]["seconds"] for row in audit["learning_pair_timing"].values())
    audit_maxima = sum(row["cap"]["max_seconds"] for row in audit["learning_pair_timing"].values())
    audited = 2 * (256 * audit_maxima + (audit_outer["elapsed_seconds"] - sampled_audit))
    return {
        "per_arm": maxima,
        "category_action_seconds": 0.025,
        "projected_run_seconds": run,
        "projected_audit_seconds": audited if finished else None,
        "projected_seconds": run + audited if finished else run,
        "run_projected_peak_mib": 702.0,
        "audit_projected_peak_mib": 306.0 if finished else None,
        "projected_tree_mib": 702.0,
        "audit_peak_pending": not finished,
        "admissible": finished,
    }


def _competitive_cost_save(path, value):
    import hashlib
    import json

    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)  # Disposable authored inputs may be restored between variants.
    return hashlib.sha256(raw).hexdigest()


def _competitive_cost_observe(log, label, inputs, call, *args, **kwargs):
    """Persist actual synthetic inputs/outcome before the caller asserts its grade."""
    import json

    log.write(
        json.dumps(
            {"label": label, "phase": "before", "inputs": inputs}, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    log.flush()
    try:
        result = call(*args, **kwargs)
    except Exception as error:
        log.write(
            json.dumps(
                {
                    "label": label,
                    "phase": "after",
                    "exception": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            )
            + "\n"
        )
        log.flush()
        if not isinstance(error, ValueError):
            raise
        return {"exception": "ValueError", "message": str(error)}
    outcome = {"result": result}
    log.write(
        json.dumps({"label": label, "phase": "after", **outcome}, sort_keys=True, allow_nan=False)
        + "\n"
    )
    log.flush()
    return outcome


def test_competitive_splice_saved_audit_arithmetic():
    import copy
    import hashlib
    import json
    from pathlib import Path

    from experiments.r4_integrated import check_neuronal_splice_online_v2 as checker

    report, outer, audit, audit_outer = _competitive_cost_synthetic_inputs()
    root = Path(__file__).resolve().parents[1]
    raw = root / ".pytest-tmp/competitive-cost-arithmetic-qa-v1-raw.jsonl"
    variants = [("preliminary", report, None, None), ("completed", report, audit, audit_outer)]
    costly = copy.deepcopy(report)
    costly["category_probes"][0]["action_seconds"] = 0.5
    large_audit = copy.deepcopy(audit_outer)
    large_audit["observed_tree_peak_sum_mib"] = 2000.0
    short = copy.deepcopy(report)
    short["timing_summary"]["marginal_reference"]["regular"]["rows"] = 3
    failed = copy.deepcopy(audit_outer)
    failed["exit_code"] = 1
    variants += [
        ("time_budget", costly, audit, audit_outer),
        ("memory_budget", report, audit, large_audit),
        ("fifth_incomplete", short, None, None),
        ("missing_monitor", report, audit, None),
        ("failed_monitor", report, audit, failed),
    ]
    outcomes = {}
    with raw.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"schema": "competitive-cost-qa-raw-v1",
                              "checker_sha256": hashlib.sha256(
                                  Path(checker.__file__).read_bytes()).hexdigest(),
                              "test_source_sha256": hashlib.sha256(
                                  Path(__file__).read_bytes()).hexdigest()}) + "\n")
        log.flush()
        for label, candidate, finished, monitor in variants:
            inputs = {
                "report": candidate,
                "outer": outer,
                "audit": finished,
                "audit_outer": monitor,
            }
            outcomes[label] = _competitive_cost_observe(
                log,
                label,
                inputs,
                checker.project_cost,
                candidate,
                outer,
                audit=finished,
                audit_outer=monitor,
            )
    preliminary, complete = (outcomes[name]["result"] for name in ("preliminary", "completed"))
    assert preliminary["projected_run_seconds"] == pytest.approx(219.8)
    assert preliminary["run_projected_peak_mib"] == 702
    assert preliminary["audit_peak_pending"] and not preliminary["admissible"]
    assert complete["projected_audit_seconds"] == pytest.approx(15.592)
    assert complete["projected_seconds"] == pytest.approx(235.392)
    assert complete["projected_tree_mib"] == 702 and complete["audit_projected_peak_mib"] == 306
    assert complete["admissible"] and not complete["audit_peak_pending"]
    assert list(complete["per_arm"]) == list(checker.ARMS)
    assert complete["per_arm"]["marginal_reference"]["model"] == 0.04
    assert not outcomes["time_budget"]["result"]["admissible"]
    assert outcomes["time_budget"]["result"]["projected_seconds"] > 900
    assert not outcomes["memory_budget"]["result"]["admissible"]
    assert outcomes["memory_budget"]["result"]["projected_tree_mib"] == 2056
    for label in ("fifth_incomplete", "missing_monitor", "failed_monitor"):
        assert outcomes[label]["exception"] == "ValueError"
    config = checker.expected_config()
    assert config["arms"] == list(report["timing_summary"])
    assert config["reader_policy"] == {
        arm: "marginal_free" if arm == "marginal_reference" else "competitive_free"
        for arm in checker.ARMS
    }
    assert (config["retained_memories_per_arm"], config["retained_tensor_bytes"]) == (72, 239657600)
    assert config["receipt_limits"] == {
        "max_line_bytes": 2**21,
        "max_row_tree_bytes": 2**21,
        "max_live_rows": 2,
        "max_line_copies": 2,
        "allocation_factor": 2,
    }
    assert tuple(
        config[k]
        for k in (
            "buffer_mib",
            "run_reserve_mib",
            "audit_reserve_mib",
            "full_only_tensor_bytes",
            "seconds",
            "tree_mib",
        )
    ) == (24, 128, 32, 0, 900, 2048)
    with (root / ".pytest-tmp/competitive-cost-arithmetic-qa-v1.json").open("x") as stream:
        json.dump(
            {
                "status": "passed",
                "direct_project_cost_calls": 7,
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "checker_sha256": hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest(),
                "models": 0,
                "children": 0,
            },
            stream,
        )


def test_competitive_splice_saved_cost_decision_custody(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    from pathlib import Path

    from experiments.r4_integrated import check_neuronal_splice_online_v2 as checker

    root = Path(__file__).resolve().parents[1]
    checker_sha = hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest()
    relative = "experiments/r4_integrated/check_neuronal_splice_online_v2.py"
    fake_checker = tmp_path / relative
    fake_checker.parent.mkdir(parents=True)
    source = b"# Authored source identity only; never imported.\n"
    fake_checker.write_bytes(source)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "__file__", str(fake_checker))
    config = checker.expected_config()
    sources = {relative: hashlib.sha256(source).hexdigest()}
    roster_path = tmp_path / "authored-roster.json"
    roster_sha = _competitive_cost_save(roster_path, {"schema": "synthetic-only", "tasks": []})
    cost_dir = tmp_path / "cost"
    cost_dir.mkdir()
    paths = {
        "cost_report": cost_dir / "report.json",
        "cost_completion": cost_dir / "completion.json",
        "cost_launch": tmp_path / "cost.launch.json",
        "cost_admission": tmp_path / "cost-admission.json",
        "cost_audit": tmp_path / "cost-audit.json",
        "cost_audit_launch": tmp_path / "cost-audit.launch.json",
    }
    report, outer, audit, audit_outer = _competitive_cost_synthetic_inputs()
    preliminary = _competitive_cost_authored_projection(
        report, outer, audit, audit_outer, finished=False
    )
    final = _competitive_cost_authored_projection(report, outer, audit, audit_outer, finished=True)
    values, hashes = {}, {}
    values["cost_admission"] = {
        "schema": "neuronal-splice-admission-v2",
        "status": "passed",
        "authorized_phase": "cost",
        "qa_passed": True,
        "config": config,
        "source_sha256": sources,
        "roster_path": str(roster_path),
        "roster_sha256": roster_sha,
    }
    hashes["cost_admission"] = _competitive_cost_save(
        paths["cost_admission"], values["cost_admission"]
    )
    report.update(
        schema="neuronal-splice-online-v2",
        config=config,
        source_sha256=sources,
        source_sha256_after=sources,
        admission_sha256=hashes["cost_admission"],
        roster_sha256=roster_sha,
        pid=101,
        deadline_unix=1900.0,
        logs={},
    )
    values["cost_report"] = report
    hashes["cost_report"] = _competitive_cost_save(paths["cost_report"], report)
    values["cost_completion"] = {
        "schema": "neuronal-splice-completion-v2",
        "status": "completed",
        "mode": "cost",
        "report_sha256": hashes["cost_report"],
        "admission_sha256": hashes["cost_admission"],
        "logs": {},
    }
    hashes["cost_completion"] = _competitive_cost_save(
        paths["cost_completion"], values["cost_completion"]
    )
    prefix = [str(tmp_path / ".venv/Scripts/python.exe"), "-X", "utf8", "-m"]
    outer["command"] = [
        *prefix,
        "experiments.r4_integrated.neuronal_splice_online_v2",
        "cost",
        "--output",
        str(cost_dir),
        "--admission",
        str(paths["cost_admission"]),
        "--admission-sha",
        hashes["cost_admission"],
    ]
    values["cost_launch"] = outer
    hashes["cost_launch"] = _competitive_cost_save(paths["cost_launch"], outer)
    bound = {
        str(paths[name]): hashes[name]
        for name in ("cost_report", "cost_completion", "cost_launch", "cost_admission")
    }
    bound.update({str(fake_checker): sources[relative], str(roster_path): roster_sha})
    audit.update(
        schema="neuronal-splice-independent-audit-v2",
        mode="cost",
        source_sha256=sources,
        roster_sha256=roster_sha,
        report_sha256=hashes["cost_report"],
        completion_sha256=hashes["cost_completion"],
        launch_sha256=hashes["cost_launch"],
        admission_sha256=hashes["cost_admission"],
        input_sha256=bound,
        projection=preliminary,
    )
    values["cost_audit"] = audit
    hashes["cost_audit"] = _competitive_cost_save(paths["cost_audit"], audit)
    audit_outer["command"] = [
        *prefix,
        "experiments.r4_integrated.check_neuronal_splice_online_v2",
        "--output",
        str(paths["cost_audit"]),
        "--cost-dir",
        str(cost_dir),
        "--launch",
        str(paths["cost_launch"]),
        "--admission",
        str(paths["cost_admission"]),
        "--admission-sha",
        hashes["cost_admission"],
    ]
    values["cost_audit_launch"] = audit_outer
    hashes["cost_audit_launch"] = _competitive_cost_save(paths["cost_audit_launch"], audit_outer)
    decision = {
        "schema": "neuronal-splice-cost-decision-v2",
        "status": "passed",
        "config": config,
        "source_sha256": sources,
        "roster_sha256": roster_sha,
        "completed_cost_audit": True,
        "inputs": {key: {"path": str(path), "sha256": hashes[key]} for key, path in paths.items()},
        "projection": final,
        "projected_seconds": final["projected_seconds"],
        "projected_tree_mib": final["projected_tree_mib"],
    }
    raw = root / ".pytest-tmp/competitive-cost-decision-qa-v1-raw.jsonl"
    outcomes = {}
    with raw.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"schema": "competitive-cost-qa-raw-v1", "checker_sha256": checker_sha,
                              "test_source_sha256": hashlib.sha256(
                                  Path(__file__).read_bytes()).hexdigest()}) + "\n")
        log.flush()
        for label in ("valid", "input", "command", "projection", "source"):
            altered, selected = copy.deepcopy(values), copy.deepcopy(decision)
            fake_checker.write_bytes(source + (b"# Changed.\n" if label == "source" else b""))
            if label == "input":
                altered["cost_audit"]["input_sha256"][str(paths["cost_report"])] = "b" * 64
            elif label == "command":
                altered["cost_audit_launch"]["command"][4] = (
                    "experiments.r4_integrated.check_neuronal_splice_online_v1"
                )
            elif label == "projection":
                selected["projection"]["projected_seconds"] -= 1
                selected["projected_seconds"] -= 1
            for name, value in altered.items():
                selected["inputs"][name]["sha256"] = _competitive_cost_save(paths[name], value)
            decision_path = tmp_path / "decision.json"
            decision_sha = _competitive_cost_save(decision_path, selected)
            inputs = {
                "decision": selected,
                "evidence": altered,
                "actual_source_sha256": hashlib.sha256(fake_checker.read_bytes()).hexdigest(),
            }
            outcomes[label] = _competitive_cost_observe(
                log,
                label,
                inputs,
                checker.validate_cost_decision,
                decision_path,
                decision_sha,
                config,
                sources,
                roster_sha,
            )
    assert outcomes["valid"]["result"] == final
    for label, fragment in (
        ("input", "audit did not authenticate selected input"),
        ("command", "different checker"),
        ("projection", "final admission projection mismatch"),
        ("source", "not the held source"),
    ):
        assert outcomes[label]["exception"] == "ValueError"
        assert fragment in outcomes[label]["message"]
    with (root / ".pytest-tmp/competitive-cost-decision-qa-v1.json").open("x") as stream:
        json.dump(
            {
                "status": "passed",
                "direct_decision_calls": 5,
                "checker_sha256": checker_sha,
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "models": 0,
                "children": 0,
            },
            stream,
        )


def test_competitive_splice_worker_saved_custody(tmp_path, monkeypatch):
    import copy
    import hashlib
    import json
    from pathlib import Path

    from experiments.r4_integrated import check_neuronal_splice_online_v2 as checker

    root = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    output = tmp_path / "cost"
    isolated = output / "endpoint"
    package = isolated / "package"
    package.mkdir(parents=True)
    stderr = isolated / "worker-stderr.bin"
    stderr.write_bytes(b"")

    def ticks(seconds):
        return 621355968000000000 + seconds * 10_000_000

    report = {
        "pid": 101,
        "deadline_unix": 1900.0,
        "mode": "cost",
        "worker": {
            "checkpoint_sha256": "c" * 64,
            "requests_sha256": "d" * 64,
            "worker_stderr_sha256": hashlib.sha256(b"").hexdigest(),
            "initial_hashes": {arm: {"experience": 4} for arm in checker.ARMS},
        },
    }
    command = [
        str(tmp_path / ".venv/Scripts/python.exe"),
        "-I",
        "-X",
        "utf8",
        str(package / checker.WORKER_SOURCE),
        "--package-root",
        str(package),
        "--checkpoint",
        str(isolated / "source_free.pt"),
        "--checkpoint-sha",
        "c" * 64,
        "--requests",
        str(isolated / "query_only.json"),
        "--requests-sha",
        "d" * 64,
        "--output",
        str(isolated / "worker"),
        "--deadline-unix",
        repr(1900.0),
    ]
    launch = {
        "parent_pid": 101,
        "cwd": str(isolated),
        "started_unix": 1002.0,
        "command": command,
        "pid": 301,
        "process_identity": {"pid": 301, "creation_ticks": ticks(1003)},
        "checkpoint_sha256": "c" * 64,
        "requests_sha256": "d" * 64,
    }
    lineage = {
        "final_model_sha256": {arm: "e" * 64 for arm in checker.ARMS},
        "memory_sha256": {arm: "f" * 64 for arm in checker.ARMS},
    }
    worker = {
        "schema": "neuronal-splice-endpoint-completion-v2",
        "scope": "integrated",
        "pid": 302,
        "parent_pid": 301,
        "process_identity": {"pid": 302, "creation_ticks": ticks(1004)},
        "checkpoint_sha256": "c" * 64,
        "requests_sha256": "d" * 64,
        "source_origins": {
            name: str(package / relative) for name, relative in checker.WORKER_MODULES.items()
        },
        "final": {
            arm: {"model_sha256": "e" * 64, "memories_sha256": "f" * 64, "experience": 52}
            for arm in checker.ARMS
        },
        "memory": {"peak_working_set_mib": 200.0},
    }
    requests = {
        "schema": "neuronal-splice-endpoint-request-v2",
        "checkpoint_sha256": "c" * 64,
        "source_sha256": {
            relative: "b" * 64
            for relative in (*checker.WORKER_MODULES.values(), checker.WORKER_SOURCE)
        },
    }
    outer = _competitive_cost_monitor(101, [], 20.0, 550.0)
    outer["observed_process_peaks_mib"].update(
        {f"301/{ticks(1003)}": 5.0, f"302/{ticks(1004)}": 200.0}
    )
    raw = root / ".pytest-tmp/competitive-worker-custody-qa-v1-raw.jsonl"
    outcomes = {}
    with raw.open("x", encoding="utf-8") as log:
        log.write(json.dumps({"schema": "competitive-cost-qa-raw-v1",
                              "checker_sha256": hashlib.sha256(
                                  Path(checker.__file__).read_bytes()).hexdigest(),
                              "test_source_sha256": hashlib.sha256(
                                  Path(__file__).read_bytes()).hexdigest()}) + "\n")
        log.flush()
        for label in ("redirector", "direct", "qa_scope", "origin", "creation", "fifth", "command"):
            candidate, selected_launch = copy.deepcopy(worker), copy.deepcopy(launch)
            if label == "direct":
                candidate.update(
                    pid=301, parent_pid=101, process_identity=launch["process_identity"]
                )
            elif label == "qa_scope":
                candidate["scope"] = "single_exchange_qa"
            elif label == "origin":
                candidate["source_origins"]["humanbrain"] = str(tmp_path / "outside.py")
            elif label == "creation":
                candidate["process_identity"]["creation_ticks"] = ticks(1001)
            elif label == "fifth":
                candidate["final"]["marginal_reference"]["experience"] = 51
            elif label == "command":
                selected_launch["command"][
                    selected_launch["command"].index("--requests-sha") + 1
                ] = "0" * 64
            selected_outer = copy.deepcopy(outer)
            if label == "creation":
                del selected_outer["observed_process_peaks_mib"][f"302/{ticks(1004)}"]
                selected_outer["observed_process_peaks_mib"][f"302/{ticks(1001)}"] = 200.0
            report["worker"]["worker_launch_sha256"] = _competitive_cost_save(
                isolated / "worker-launch.json", selected_launch
            )
            inputs = {
                "report": report,
                "worker": candidate,
                "requests": requests,
                "lineage": lineage,
                "outer": selected_outer,
                "launch": selected_launch,
            }
            outcomes[label] = _competitive_cost_observe(
                log,
                label,
                inputs,
                checker.check_worker_custody,
                output,
                report,
                candidate,
                requests,
                lineage,
                selected_outer,
            )
    assert outcomes["redirector"] == outcomes["direct"] == {"result": None}
    assert len(requests["source_sha256"]) == 16 and len(worker["source_origins"]) == 15
    for label, fragment in (
        ("qa_scope", "QA worker scope"),
        ("origin", "module origins"),
        ("creation", "worker creation outside"),
        ("fifth", "final state/experience"),
        ("command", "actual worker command differs"),
    ):
        assert outcomes[label]["exception"] == "ValueError"
        assert fragment in outcomes[label]["message"]
    with (root / ".pytest-tmp/competitive-worker-custody-qa-v1.json").open("x") as stream:
        json.dump(
            {
                "status": "passed",
                "direct_worker_custody_calls": 7,
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "checker_sha256": hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest(),
                "models": 0,
                "children": 0,
            },
            stream,
        )


def _coincidence_qa_plain(value):
    import dataclasses

    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, bytes):
        return {"hex": value.hex()}
    if dataclasses.is_dataclass(value):
        return {field.name: _coincidence_qa_plain(getattr(value, field.name))
                for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return {key: _coincidence_qa_plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_coincidence_qa_plain(item) for item in value]
    return value


def _coincidence_qa_save(path, value):
    import hashlib
    import json

    with path.open("x", encoding="utf-8") as stream:
        json.dump(_coincidence_qa_plain(value), stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _coincidence_qa_authored_rails(symbols):
    """Independent fixture construction; never calls the production encoder."""
    rows = []
    for symbol in symbols:
        rows.append([False] * 18 if symbol is None else [
            bool(((symbol >> bit) & 1) == rail) for bit in range(9) for rail in (0, 1)
        ])
    return torch.tensor(rows, dtype=torch.bool)


def test_coincidence_all_symbol_truth(tmp_path):
    from humanbrain.neuronal_byte_coincidence import coincidence_features, encode_symbol_rails

    encode_work = {}
    encoded = encode_symbol_rails(torch.arange(258, dtype=torch.int64), work=encode_work)
    populations = torch.zeros(288, 18, dtype=torch.bool)
    populations[:258] = encoded
    keys = populations.reshape(9, 32, 18)
    identities = list(range(258)) + [None] * 30
    kernel_work = {}
    for block in range(9):
        query = populations[block * 32:(block + 1) * 32]
        actual = coincidence_features(query, keys, work=kernel_work)
        query_ids = identities[block * 32:(block + 1) * 32]
        qn = sum(symbol is not None for symbol in query_ids)
        expected = []
        for key_block in range(9):
            key_ids = identities[key_block * 32:(key_block + 1) * 32]
            kn = sum(symbol is not None for symbol in key_ids)
            gain = math.sqrt(32) / math.sqrt(qn * kn)
            expected.append([[gain if q is not None and q == k else 0.0
                              for k in key_ids] for q in query_ids])
        target = torch.tensor(expected, dtype=torch.float32)
        _coincidence_qa_save(tmp_path / f"truth-{block}.json", {
            "scope": "authored complete symbol truth", "block": block,
            "actual": actual, "expected": target, "work": kernel_work,
        })
        assert torch.equal(actual != 0, target != 0)
        assert torch.allclose(actual, target, rtol=1e-6, atol=1e-7)
    assert torch.equal(encoded, _coincidence_qa_authored_rails(range(258)))
    assert encode_work["rail_encode_symbols"] == 258
    assert kernel_work["coincidence_feature_calls"] == 9
    assert kernel_work["coincidence_lag_pairs"] == 82_944
    assert kernel_work["coincidence_threshold_evaluations"] == 19 * 82_944
    assert kernel_work["coincidence_input_applications"] == 54 * 82_944
    assert int((encoded.sum(1) == 9).sum()) == 258
    assert not populations[258:].any()


def test_coincidence_occupancy_and_collision_arithmetic(tmp_path):
    from humanbrain.neuronal_byte_coincidence import coincidence_features

    query = _coincidence_qa_authored_rails([65, 66] + [None] * 30)
    keys = torch.stack([_coincidence_qa_authored_rails(items + [None] * (32 - len(items)))
                        for items in ([65], [65, 66], [65, 66], [])])
    work = {}
    actual = coincidence_features(query, keys, work=work)
    repeated = _coincidence_qa_authored_rails([65, 65] + [None] * 30)
    saturated = coincidence_features(repeated, repeated[None], work=work)
    outcomes = {}
    for name in ("partial", "dtype"):
        candidate = query.clone()
        if name == "partial":
            candidate[0, 0:2] = False
        else:
            candidate = candidate.to(torch.float32)
        try:
            coincidence_features(candidate, keys)
        except ValueError as exc:
            outcomes[name] = str(exc)
        else:
            outcomes[name] = None
    _coincidence_qa_save(tmp_path / "occupancy.json", {
        "actual": actual, "saturated": saturated, "rejections": outcomes, "work": work,
    })
    expected = torch.zeros(4, 32, 32)
    expected[0, 0, 0] = 4
    expected[1, 0, 0] = expected[1, 1, 1] = math.sqrt(8)
    expected[2] = expected[1]
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-6)
    assert torch.equal(actual[1], actual[2]) and not actual[3].any()
    assert float(saturated.square().sum()) == pytest.approx(32, rel=1e-6)
    assert torch.count_nonzero(saturated) == 4
    assert "one active rail per bit" in outcomes["partial"]
    assert "torch.bool" in outcomes["dtype"]
    assert work["coincidence_feature_calls"] == 2
    assert work["coincidence_lag_pairs"] == 5 * 1024


def test_coincidence_raw_tape_publication(tmp_path):
    from humanbrain.coincidence_sequence_memory import TAPE_BYTES, CoincidenceSequenceMemory
    from humanbrain.neuronal_sequence_memory import TapeState

    enabled, disabled = CoincidenceSequenceMemory(), CoincidenceSequenceMemory()
    work_on, work_off = {}, {}
    documents = (b"\x00A", b"", b"B")
    state = enabled.ingest(documents, work=work_on)
    blank = disabled.ingest(documents, writes_enabled=False, work=work_off)
    original = state.snapshot()
    restored = enabled.restore_state(original)
    legacy = TapeState(torch.zeros(128, 1024), state.sensory.clone(),
                       state.successor.clone(), state.write_index.clone())
    overflow_work, outcomes = {}, {}
    for name in ("overflow", "legacy"):
        try:
            if name == "overflow":
                enabled.ingest([b"x" * 128], work=overflow_work)
            else:
                enabled.check_state(legacy)
        except ValueError as exc:
            outcomes[name] = str(exc)
        else:
            outcomes[name] = None
    _coincidence_qa_save(tmp_path / "tape.json", {
        "state": state, "disabled": blank, "work_on": work_on, "work_off": work_off,
        "overflow_work": overflow_work, "rejections": outcomes,
        "accounting": enabled.accounting(state),
    })
    assert state.used == blank.used == 6
    assert state.keys.shape == (128, 32, 18) and state.keys.dtype == torch.bool
    assert state.successor[:6].tolist() == [1, 2, 2, 3, 5, 5]
    assert torch.equal(state.successor, blank.successor)
    assert state.sensory[:6].argmax(1).tolist() == [0, 65, 256, 256, 66, 256]
    for slot, prefix in enumerate(([], [0], [65, 0], [], [], [66])):
        expected = _coincidence_qa_authored_rails(prefix + [None] * (32 - len(prefix)))
        assert torch.equal(state.keys[slot], expected)
    assert not state.keys[..., 17].any()  # Raw prefixes never store EOS/BOS.
    assert not blank.keys.any() and not blank.sensory.any()
    assert sum(v.numel() * v.element_size() for v in vars(state).values()) == TAPE_BYTES == 206344
    assert work_on["coincidence_ingest_allocation_steps"] == 6
    assert work_off["coincidence_ingest_allocation_steps"] == 6
    assert work_on["coincidence_ingest_key_publications"] == 6 * 576
    assert work_on["coincidence_ingest_sensory_publications"] == 6
    assert work_off.get("coincidence_ingest_key_publications", 0) == 0
    assert work_on["rail_encode_symbols"] == work_off["rail_encode_symbols"] == 3
    assert overflow_work == {}
    assert "capacity" in outcomes["overflow"] and "geometry" in outcomes["legacy"]
    _assert_nested_equal(state.snapshot(), original)
    _assert_nested_equal(restored.snapshot(), original)
    assert all(value.data_ptr() != original[name].data_ptr()
               and value.data_ptr() != vars(state)[name].data_ptr()
               for name, value in vars(restored).items())


def test_coincidence_direct_local_delta(tmp_path, monkeypatch):
    from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner

    model = CoincidenceLocalMemoryLearner(seed=733)
    state = model.ingest([b"A"])
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    before_tape = state.snapshot()
    events = []
    original_query = model._query

    def capture(query, tape):
        event = original_query(query, tape)
        events.append(event)
        return event

    monkeypatch.setattr(model, "_query", capture)
    result = model.teach(b"A", b"A", state, .01)
    event = events[0]
    features = event.address_features.tolist()
    errors = result.address_error.tolist()
    local = torch.tensor([[sum(features[s][i][j] * errors[s] for s in range(2))
                           for j in range(32)] for i in range(32)], dtype=torch.float64)
    address_expected = before["address_weight"] - .01 * local.to(torch.float32)

    # Scalar held-codec reference: no second codec/query/teacher invocation.
    weight_before = before["codec.weight"].tolist()
    bias_before = before["codec.bias"].tolist()
    probabilities = {}
    for token in (65, 256):
        logits = [weight_before[row][token] + bias_before[row] for row in range(257)]
        largest = max(logits)
        unnormalized = [math.exp(value - largest) for value in logits]
        total = sum(unnormalized)
        probabilities[token] = [value / total for value in unnormalized]
    weight_error = [[0.0] * 257 for _ in range(257)]
    bias_error = [0.0] * 257
    prior = [math.exp(value) for value in event.log_prior.tolist()]
    expected_steps = []
    for position, target in enumerate((65, 256)):
        sensory_tokens = (65, 256) if position == 0 else (256, 256)
        mixture = [sum(prior[s] * probabilities[token][row]
                       for s, token in enumerate(sensory_tokens)) for row in range(257)]
        posterior = [prior[s] * probabilities[token][target] / mixture[target]
                     for s, token in enumerate(sensory_tokens)]
        for s, token in enumerate(sensory_tokens):
            for row in range(257):
                error = posterior[s] * (probabilities[token][row] - int(row == target))
                weight_error[row][token] += error
                bias_error[row] += error
        expected_steps.append({"mixture": mixture, "posterior": posterior})
        prior = posterior
    codec_expected = before["codec.weight"] - .01 * torch.tensor(weight_error, dtype=torch.float64)
    bias_expected = before["codec.bias"] - .01 * torch.tensor(bias_error, dtype=torch.float64)
    _coincidence_qa_save(tmp_path / "local-delta.json", {
        "teacher": result, "query": event, "scalar_address_delta": local,
        "expected_steps": expected_steps,
        "address_expected": address_expected, "codec_expected": codec_expected,
        "bias_expected": bias_expected, "after": dict(model.named_parameters()),
    })
    assert len(events) == 1 and result.teacher_count == 2
    assert torch.equal(model.address_weight, address_expected)
    assert torch.allclose(model.codec.weight, codec_expected, rtol=1e-12, atol=1e-12)
    assert torch.allclose(model.codec.bias, bias_expected, rtol=1e-12, atol=1e-12)
    for actual, expected in zip(result.steps, expected_steps, strict=True):
        assert actual.mixture_probabilities.tolist() == pytest.approx(expected["mixture"])
        assert actual.posterior_log_belief.exp().tolist() == pytest.approx(expected["posterior"])
    assert event.activity.shape == (1, 576) and event.activity.dtype == torch.bool
    assert not event.address_features[:, 0].any()
    assert torch.equal(before["address_weight"][0], model.address_weight[0])
    assert not torch.equal(before["address_weight"], model.address_weight)
    assert result.work["address_local_synapse_contributions"] == 2 * 1024
    assert result.work["address_local_slot_reductions"] == 1024
    assert result.work["coincidence_feature_calls"] == 1
    assert result.work["coincidence_threshold_evaluations"] == 19 * 2 * 1024
    assert result.work["address_feature_products"] == 0
    assert result.work["address_feature_channel_reductions"] == 0
    assert result.work["query_population_normalizations"] == 0
    assert not any(key.startswith("competition_") for key in result.work)
    _assert_nested_equal(state.snapshot(), before_tape)


def test_coincidence_controls_and_atomicity(tmp_path, monkeypatch):
    import humanbrain.local_memory_learner as local_module
    from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner
    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learned = learned_state = None
    for mode in ("learned", "address_frozen", "slow_frozen", "writes_disabled"):
        model = CoincidenceLocalMemoryLearner(mode=mode, seed=733)
        state = model.ingest([b"A"])
        before = {name: value.detach().clone() for name, value in model.named_parameters()}
        tape_before = state.snapshot()
        result = model.teach(b"A", b"A", state, .01)
        changed = {name: not torch.equal(value, before[name])
                   for name, value in model.named_parameters()}
        _coincidence_qa_save(tmp_path / f"control-{mode}.json", {
            "mode": mode, "teacher": result, "changed": changed,
            "metadata": model.get_extra_state(), "accounting": model.accounting(),
        })
        assert changed["address_weight"] == (mode == "learned")
        assert (changed["codec.weight"] or changed["codec.bias"]) == (mode != "slow_frozen")
        if mode == "writes_disabled":
            assert not state.keys.any() and not state.sensory.any()
            assert not changed["codec.weight"] and changed["codec.bias"]
        assert result.teacher_count == 2
        assert result.work["coincidence_feature_calls"] == 1
        assert result.work["coincidence_threshold_evaluations"] == 19 * 2 * 1024
        _assert_nested_equal(state.snapshot(), tape_before)
        assert all(not p.requires_grad and p.grad is None and p.grad_fn is None
                   for p in model.parameters())
        if mode == "learned":
            learned, learned_state = model, state

    pristine = learned.snapshot(learned_state)
    with torch.no_grad():
        learned.address_weight.add_(1)
    restored = learned.restore_snapshot(pristine)
    _assert_nested_equal(learned.snapshot(restored), pristine)
    assert all(v.data_ptr() != pristine["memory"][name].data_ptr()
               for name, v in vars(restored).items())
    invalid = {}
    for name in ("metadata", "fixed_codes"):
        snapshot = copy.deepcopy(pristine)
        if name == "metadata":
            snapshot["model"]["_extra_state"]["schema"] = "legacy-graded-reader"
        else:
            snapshot["model"]["input_codes"][0, 0] += 1
        try:
            learned.restore_snapshot(snapshot)
        except ValueError as exc:
            invalid[name] = str(exc)
        else:
            invalid[name] = None
    calls = []
    actual_finite = local_module._finite

    def fail_late(value, name):
        actual_finite(value, name)
        if name == "codec parameter proposal":
            calls.append(name)
        if name == "address parameter proposal":
            calls.append(name)
            raise ValueError("authored late address validation fault")

    with monkeypatch.context() as context:
        context.setattr(local_module, "_finite", fail_late)
        try:
            learned.teach(b"A", b"A", learned_state, .01)
        except ValueError as exc:
            fault = str(exc)
        else:
            fault = None
    _coincidence_qa_save(tmp_path / "atomicity.json", {
        "fault": fault, "validation_calls": calls, "invalid": invalid,
        "before": pristine, "after": learned.snapshot(learned_state),
        "metadata": learned.get_extra_state(),
    })
    assert fault == "authored late address validation fault"
    assert calls == ["codec parameter proposal", "codec parameter proposal",
                     "address parameter proposal"]
    assert "configuration mismatch" in invalid["metadata"]
    assert "fixed input codes" in invalid["fixed_codes"]
    _assert_nested_equal(learned.snapshot(learned_state), pristine)
    assert CoincidenceLocalMemoryLearner.teach is LocalMemoryLearner.teach
    assert CoincidenceLocalMemoryLearner.predict_with_receipt is (
        CompetitiveLocalMemoryLearner.predict_with_receipt)
    assert all(not p.requires_grad and p.grad is None and p.grad_fn is None
               for p in learned.parameters())


def test_coincidence_same_instance_action_then_correction(tmp_path):
    import json
    from pathlib import Path
    from time import get_clock_info, perf_counter

    from experiments.r4_integrated.neuronal_splice_io_v1 import JsonlJournal
    from experiments.r4_integrated.neuronal_splice_online_v1 import (
        SPEC,
        execute_action,
        qualified_launch,
    )
    from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner
    from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
    from humanbrain.neuronal_splice_action import SpliceProposal, SpliceWorkspace
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest

    root = Path(__file__).resolve().parents[1]
    launch = qualified_launch(root)
    reader = CoincidenceLocalMemoryLearner(seed=733)
    model = NeuronalSpliceLearner(seed=733, reader=reader)
    state = model.ingest([b"ab"])
    with torch.no_grad():
        reader.address_weight.zero_()
        model.actions.operation_weight.zero_()
        model.actions.operation_bias.zero_()
        model.actions.operation_bias[0] = 1
    workspace = SpliceWorkspace((OpaqueWorkspaceFile("a.py", b"x = 'old'\n"),
                                 OpaqueWorkspaceFile("main.py", b"import a\nprint(a.x)\n")))
    before = model.snapshot(state)
    prediction = model.propose(b"a", workspace, state)
    journal = JsonlJournal(tmp_path / "action.jsonl", SPEC)
    try:
        envelope, action_sha, _, grade = execute_action(
            task=None, workspace=workspace, proposal=prediction.proposal,
            attempt_id=prediction.attempt_id, phase="coincidence-qa", arm="learned", index=0,
            output=tmp_path, launch=launch, journal=journal,
        )
    finally:
        journal_summary = journal.close()
    completed = perf_counter()
    raw_sha = _coincidence_qa_save(tmp_path / "action-raw.json", {
        "status": "pending_checks", "prediction": prediction, "envelope": envelope,
        "action_sha256": action_sha, "journal": journal_summary,
        "model_instance": id(model), "reader_instance": id(reader), "state_instance": id(state),
        "completed_perf": completed, "clock": vars(get_clock_info("perf_counter")),
    })
    action = json.loads((tmp_path / "action.jsonl").read_bytes())
    assert grade is None and action["rejection"] is None
    assert prediction.proposal.operation == "STOP"
    assert prediction.payload.generation.data == b"ab"
    assert prediction.payload.generation.stop_reason == "eos"
    assert len(prediction.payload.steps) == 3
    assert prediction.payload.competition.winner_index == 0
    assert envelope["execution"]["returncode"] == 0
    assert bytes.fromhex(envelope["execution"]["stdout_hex"]) == b"old\n"
    for item in workspace.files:
        assert (tmp_path / "executions/coincidence-qa-learned-000000" / item.path).read_bytes() == (
            item.content)
    assert prediction.model_sha256 == _digest(model.state_dict())
    assert prediction.memory_sha256 == _digest(state.snapshot())
    _assert_nested_equal(model.snapshot(state)["model"], before["model"])
    _assert_nested_equal(state.snapshot(), before["memory"])
    teacher = SpliceProposal("SPLICE", "a.py", 5, 8, b"ab", "eos", workspace.sha256)
    started = perf_counter()
    feedback = model.teach(prediction, workspace, state, teacher, envelope)
    _coincidence_qa_save(tmp_path / "teacher-raw.json", {
        "feedback": feedback, "before": before, "after": model.snapshot(state),
        "teacher_started_perf": started, "action_completed_perf": completed,
        "same_model": id(model), "same_reader": id(reader), "same_state": id(state),
    })
    assert started >= completed and model.reader is reader and model._pending is None
    assert feedback.reader_feedback.teacher_count == 3
    assert [step.observed_token for step in feedback.reader_feedback.steps] == [97, 98, 256]
    assert feedback.reader_feedback.work["branch_cursor_initializations"] == 3
    assert feedback.reader_feedback.work["coincidence_feature_calls"] == 1
    assert not any(key.startswith("competition_") for key in feedback.reader_feedback.work)
    assert set(feedback.teacher_choices) == {"start", "end", "operation"}
    assert sum(feedback.work["action"][f"{head}_teacher_offers"]
               for head in ("start", "end", "operation")) == 3
    _assert_nested_equal(state.snapshot(), before["memory"])
    assert all(not p.requires_grad and p.grad is None and p.grad_fn is None
               for p in model.parameters())
    _coincidence_qa_save(tmp_path / "passed.json", {
        "status": "passed", "raw_sha256": raw_sha, "action_sha256": action_sha,
        "reader_constructions": 1, "combined_constructions": 1, "free_calls": 1,
        "free_decisions": 3, "teacher_calls": 1, "teacher_offers": 3,
        "action_labels": 3, "actual_children": 1, "historical_checkpoint_loads": 0,
        "claim": "authored STOP execution then same-instance local correction, not learned skill",
    })


def test_coincidence_splice_roster_data_contract():
    """One new roster construction; saved v2 comparison; no model or program execution."""
    import ast
    import json
    from collections import Counter
    from hashlib import sha256
    from pathlib import Path

    from experiments.r4_integrated import neuronal_splice_roster_v3 as data

    root = Path(__file__).resolve().parents[1]
    old_path = root / ".pytest-tmp/local-joint-address-online-v6-run/offline_teacher_roster.json"
    previous_path = root / ".pytest-tmp/neuronal-splice-roster-qa-generated-v2.json"
    old_source = root / "experiments/r4_integrated/neuronal_splice_roster_v1.py"
    new_source = root / "experiments/r4_integrated/neuronal_splice_roster_v3.py"
    bindings = {
        old_path: "009b42e22976733bba207e46337ec23d5f6de5ab15ab061bc6fd08539e44a2b1",
        previous_path: "1f76de378e238ddebd6ebcc8dd62f9dffc3187cada1923e794f86e51dfb4440e",
        old_source: "b814edae2d3bc0da5d6be85ef1adb36f85b411522b189ee376617c192f194d77",
        new_source: "c091fc8814683cb5bc3dda89d9451853582e9bcd2f42ad6925c5c58f46e37b39",
    }
    for path, expected in bindings.items():
        assert sha256(path.read_bytes()).hexdigest() == expected
    previous = json.loads(previous_path.read_bytes())
    output = root / ".pytest-tmp/neuronal-splice-roster-qa-generated-v3.json"
    raw_receipt = root / ".pytest-tmp/neuronal-splice-roster-data-qa-v3-raw.json"
    passed = root / ".pytest-tmp/neuronal-splice-roster-data-qa-v3.json"
    assert not any(path.exists() for path in (output, raw_receipt, passed))
    roster = data.build_roster(old_path)  # The sole constructor call.
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(roster, sort_keys=True, allow_nan=False) + "\n")
    receipt = {
        "schema": "coincidence-splice-roster-data-qa-v1", "status": "pending_checks",
        "source_inputs": {str(path): value for path, value in bindings.items()},
        "tested_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "generated_path": str(output), "generated_sha256": sha256(output.read_bytes()).hexdigest(),
        "builder_calls": 1, "historical_builder_calls": 0, "models": 0, "children": 0,
        "checkpoint_loads": 0, "program_executions": 0,
    }
    with raw_receipt.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)

    assert roster["schema"] == "neuronal-splice-roster-v3"
    assert roster["source_inputs"] == previous["source_inputs"]
    for name in ("train_task_ids", "schedule", "cost_task_ids", "retention_task_ids", "old_panel"):
        assert roster[name] == previous[name]
    tasks, memories = roster["tasks"], roster["memory_packages"]
    for task_id in roster["train_task_ids"]:
        assert tasks[task_id] == previous["tasks"][task_id]
        memory_id = tasks[task_id]["memory_id"]
        assert memories[memory_id] == previous["memory_packages"][memory_id]
    old_ids = {row["memory_id"] for row in roster["old_panel"]}
    assert len(old_ids) == 12
    for memory_id in old_ids:
        assert memories[memory_id] == previous["memory_packages"][memory_id]
    assert Counter(roster["schedule"]) == {f"T{i:02d}": 8 for i in range(32)}
    assert roster["fresh_task_ids"] == [f"{g}{r}-{m}" for g in "DCVO"
                                         for r in range(4) for m in range(2)]
    assert set(tasks) == set(roster["train_task_ids"] + roster["fresh_task_ids"])
    assert Counter(tasks[key]["required_operation"] for key in roster["fresh_task_ids"]) == {
        "SPLICE": 28, "STOP": 4}
    assert roster["counts"] == {
        "tasks": 64, "train_tasks": 32, "fresh_tasks": 32, "retention_tasks": 16,
        "lessons_per_arm": 256, "memory_packages_per_arm": 72,
        "before_learning_packages": 48, "after_learning_packages": 24,
        "documents_per_arm": 168, "document_raw_bytes_per_arm": 4022,
        "document_eos_per_arm": 168, "ingestion_symbols_per_arm": 4190,
        "maximum_memory_slots": 125, "reader_teacher_offers_per_arm": 480,
        "action_teacher_labels_per_arm": 640, "full_action_verifications": 1520,
        "old_panel_reader_calls": 200, "full_reader_teacher_offers": 2400,
        "full_action_teacher_labels": 3200,
    }
    docs = [bytes.fromhex(raw) for package in memories.values() for raw in package["document_hex"]]
    assert (len(docs), sum(map(len, docs))) == (168, 4022)
    before_ids, after_ids = (roster[f"{stage}_memory_ids"] for stage in
                             ("before_learning", "after_learning"))
    assert len(before_ids) == 48 and len(after_ids) == 24
    assert not set(before_ids) & set(after_ids)
    assert set(before_ids + after_ids) == set(memories)
    for stage, ids in (("before_learning", before_ids), ("after_learning", after_ids)):
        assert {key for key, value in memories.items() if value["stage"] == stage} == set(ids)
    fresh_memory_ids = {tasks[key]["memory_id"] for key in roster["fresh_task_ids"]}
    assert len(fresh_memory_ids) == 28
    assert fresh_memory_ids & set(before_ids) == {f"M-V{r}-{m}" for r in range(2) for m in range(2)}
    assert all(sum(len(bytes.fromhex(raw)) + 1 for raw in memory["document_hex"]) <= 128
               for memory in memories.values())

    syntax_parses = literal_reads = 0
    for task in tasks.values():
        before = {f["path"]: bytes.fromhex(f["content_hex"]) for f in task["files"]}
        expected_after = {f["path"]: bytes.fromhex(f["content_hex"])
                          for f in task["expected_after_files"]}
        assert len(before) == 3 and set(before) == set(expected_after)
        assert max(len(path.encode()) for path in before) <= 31
        assert 0 < len(bytes.fromhex(task["request_hex"])) <= 31
        teacher, actual = task["teacher"], dict(before)
        # Independent generic digest, not the constructor's helper.
        digest = sha256(b"humanbrain.raw-splice-workspace.v1\0")
        digest.update(len(before).to_bytes(4, "big"))
        for path in sorted(before, key=lambda value: value.encode()):
            for raw in (path.encode(), before[path]):
                digest.update(len(raw).to_bytes(8, "big"))
                digest.update(raw)
        assert task["workspace_sha256"] == teacher["workspace_sha256"] == digest.hexdigest()
        assert teacher["operation"] == task["required_operation"]
        assert teacher["stop_reason"] == "eos"
        if teacher["operation"] == "SPLICE":
            path, start, end = teacher["path"], teacher["start"], teacher["end"]
            replacement = bytes.fromhex(teacher["replacement_hex"])
            assert 0 <= start <= end <= len(before[path]) and len(replacement) <= 16
            actual[path] = before[path][:start] + replacement + before[path][end:]
        else:
            assert teacher["operation"] == "STOP" and teacher["replacement_hex"] == ""
        assert actual == expected_after
        after_trees = {}
        for workspace in (before, actual):
            for path, raw in workspace.items():
                assert len(raw) <= 128
                tree = ast.parse(raw.decode("ascii"), filename=path)
                syntax_parses += 1
                if workspace is actual:
                    after_trees[path] = tree
        values = {}
        for path, tree in after_trees.items():
            if path == "main.py":
                continue
            values[path[:-3]] = {}
            assert len(tree.body) == 2
            for assignment in tree.body:
                assert isinstance(assignment, ast.Assign) and len(assignment.targets) == 1
                assert isinstance(assignment.targets[0], ast.Name)
                value = ast.literal_eval(assignment.value)
                literal_reads += 1
                assert isinstance(value, str)
                values[path[:-3]][assignment.targets[0].id] = value.encode("ascii")
            assert set(values[path[:-3]]) == {"x", "y"}
        main = after_trees["main.py"].body
        assert len(main) == 2 and isinstance(main[0], ast.Import)
        assert len(main[0].names) == 2 and all(alias.asname is None for alias in main[0].names)
        names = [alias.name for alias in main[0].names]
        assert set(names) == set(values)
        assert isinstance(main[1], ast.Expr) and isinstance(main[1].value, ast.Call)
        call = main[1].value
        assert isinstance(call.func, ast.Name) and call.func.id == "print"
        assert len(call.args) == 4 and len(call.keywords) == 1
        assert call.keywords[0].arg == "sep"
        assert isinstance(call.keywords[0].value, ast.Constant)
        assert call.keywords[0].value.value == "|"
        argument_names = [(n, v) for n in names for v in ("x", "y")]
        for arg, (name, variable) in zip(call.args, argument_names, strict=True):
            assert isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name)
            assert (arg.value.id, arg.attr) == (name, variable)
        stdout = b"|".join(values[name][var] for name in names for var in ("x", "y")) + b"\n"
        assert stdout.hex() == task["expected_stdout_hex"]
        observation = data.task_observation(task)
        assert set(observation) == {
            "task_id", "memory_id", "request_hex", "files", "workspace_sha256"}
        assert all(set(item) == {"path", "content_hex"} for item in observation["files"])
        assert all(observation[key] == task[key] for key in observation)
        observation["files"][0]["content_hex"] = ""
        assert task["files"][0]["content_hex"] != ""
    assert (syntax_parses, literal_reads) == (384, 256)

    continuation = ((b"ac", b"acacac"), (b"ca", b"cacaca"),
                    (b"bd", b"bdbdbd"), (b"db", b"dbdbdb"))
    role_values = ((b"ad", b"da"), (b"bc", b"cb"), (b"ab", b"ba"), (b"cd", b"dc"))
    for group in "DCVO":
        for r in range(4):
            left, right = (tasks[f"{group}{r}-{m}"] for m in range(2))
            assert left["pair_id"] == right["pair_id"] == f"{group}{r}"
            assert (left["member"], right["member"]) == (0, 1)
            assert left["files"] == right["files"]
            stems = {"D": ("od", "at"), "C": ("cq", "cp"),
                     "V": ("vr", "vd"), "O": ("ox", "oy")}[group]
            pair_paths = tuple(f"{stem}{r}.py" for stem in stems)
            assert {item["path"] for item in left["files"]} == {*pair_paths, "main.py"}
            key_id = 4 * "DCVO".index(group) + r
            expected_request = (
                f"SET {pair_paths[r % 2]} {'x' if r % 2 == 0 else 'y'} z{key_id:02d}"
            ).encode("ascii")
            assert bytes.fromhex(left["request_hex"]) == expected_request
            documents = [[bytes.fromhex(raw) for raw in memories[t["memory_id"]]["document_hex"]]
                         for t in (left, right)]
            payloads = tuple(bytes.fromhex(t["teacher"]["replacement_hex"]) for t in (left, right))
            if group != "O":
                assert left["request_hex"] == right["request_hex"]
                assert left["memory_id"] != right["memory_id"]
                for field in ("path", "start", "end", "operation"):
                    assert left["teacher"][field] == right["teacher"][field]
            if group == "D":
                assert payloads == ((b"ad", b"da", b"bc", b"cb")[r],) * 2
                assert len(documents[0]) == 2 and documents[0] == documents[1][::-1]
                assert left["teacher"] == right["teacher"]
                assert left["expected_after_files"] == right["expected_after_files"]
            elif group == "C":
                assert payloads == continuation[r] and payloads[0] == payloads[1][:2]
                assert documents[0][1] == documents[1][1]
                assert len(documents[0]) == len(documents[1]) == 2
                assert documents[1][0] == documents[0][0][:-3] + payloads[1] + b"."
                assert left["expected_after_files"] != right["expected_after_files"]
            elif group == "V":
                assert payloads == role_values[r]
                expected_slots = 50 if r < 2 else 125
                for member, rows in enumerate(documents):
                    assert sum(len(raw) + 1 for raw in rows) == expected_slots
                    prefix = f"Record z{8 + r:02d} has value ".encode()
                    assert rows[0] == prefix + payloads[member] + b"."
                    assert rows[1] == b"Record z16 has value " + payloads[1 - member] + b"."
                assert documents[0][2:] == documents[1][2:]
                assert left["expected_after_files"] != right["expected_after_files"]
            else:
                assert payloads == ((b"bc", b"cb", b"cd", b"")[r], b"")
                assert documents[0][0] == (
                    f"Record z{12 + r:02d} has value ".encode() + payloads[0] + b".")
                assert left["memory_id"] == right["memory_id"] and documents[0] == documents[1]
                assert bytes.fromhex(left["request_hex"]).startswith(b"SET ")
                assert bytes.fromhex(right["request_hex"]) == (
                    b"KEEP " + bytes.fromhex(left["request_hex"])[4:])
                raw = next(bytes.fromhex(f["content_hex"]) for f in left["files"]
                           if f["path"] == left["teacher"]["path"])
                if r < 2:
                    assert b"\t" in raw
                    assert raw == (
                        b"x \t= 'old'\ny= \t'old'\n" if r == 0 else
                        b"x\t= \t'old'\ny  = 'old'\n")
                    assert left["teacher"]["end"] - left["teacher"]["start"] == 3
                elif r == 2:
                    assert raw.startswith(b"# new gap\n\n")
                    assert left["teacher"]["start"] == left["teacher"]["end"] and payloads[0]
                else:
                    assert raw.startswith(b"\n# remove\n") and payloads[0] == b""
                    assert left["teacher"]["end"] > left["teacher"]["start"]
    exposed_paths = {f["path"] for task in previous["tasks"].values() for f in task["files"]}
    fresh_paths = {f["path"] for key in roster["fresh_task_ids"] for f in tasks[key]["files"]}
    assert exposed_paths & fresh_paths == {"main.py"}
    fresh_keys = {bytes.fromhex(raw)[7:10] for key in fresh_memory_ids
                  for raw in memories[key]["document_hex"]}
    exposed_keys = {bytes.fromhex(raw)[7:10] for key, value in previous["memory_packages"].items()
                    if not key.startswith("M-old-") for raw in value["document_hex"]}
    assert not fresh_keys & exposed_keys
    assert fresh_keys == {f"z{i:02d}".encode() for i in range(20)}
    for path, expected in bindings.items():
        assert sha256(path.read_bytes()).hexdigest() == expected
    receipt.update(status="passed", syntax_parses=syntax_parses,
                   assignment_literal_reads=literal_reads, paired_contrasts=16,
                   raw_receipt_sha256=sha256(raw_receipt.read_bytes()).hexdigest())
    with passed.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True)


# Prospective append fragment; relies on the existing qualified
# _competitive_receipt_qa_reader/_save helpers in test_byte_sequence_decoder.py.
# Source only until root reviews, appends, freezes, doctors and selects execution.


def _coincidence_receipt_qa_reader(n, decisions, *, kind, frontend, raw):
    receipt, state = _competitive_receipt_qa_reader(n, decisions, kind=kind)
    work = receipt.work
    work["query_raw_bytes"] = raw
    work["query_lag_shift_steps"] = raw + 1
    if frontend == "graded_random_code":
        return receipt, state
    # Authored scalar table: no production formula helper or kernel invocation.
    q, f, rows = raw + 1, n * 1024, (n + 1) * 32
    work.update(
        {
            "query_population_normalizations": 0,
            "address_feature_products": 0,
            "address_feature_channel_reductions": 0,
            "rail_encode_symbols": q,
            "rail_encode_bit_positions": 9,
            "rail_encode_bit_shifts": 9 * q,
            "rail_encode_bit_masks": 9 * q,
            "rail_encode_boolean_cast_values": 9 * q,
            "rail_encode_complements": 9 * q,
            "rail_encode_output_values": 18 * q,
            "rail_encode_named_allocated_bytes": 108 * q + 72,
            "rail_validation_rows": rows,
            "rail_validation_complementary_pair_checks": 9 * rows,
            "rail_validation_special_policy_rows": rows,
            "coincidence_feature_calls": 1,
            "coincidence_slots": n,
            "coincidence_lag_pairs": f,
            "coincidence_and_cell_evaluations": 18 * f,
            "coincidence_match_cell_evaluations": f,
            "coincidence_threshold_evaluations": 19 * f,
            "coincidence_and_input_applications": 36 * f,
            "coincidence_match_input_applications": 18 * f,
            "coincidence_input_applications": 54 * f,
            "coincidence_pair_additions": 18 * f,
            "coincidence_rail_sum_additions": 17 * f,
            "coincidence_operand_cast_values": 18 * rows,
            "coincidence_match_input_cast_values": 18 * f,
            "coincidence_occupancy_rail_reductions": 17 * rows,
            "coincidence_occupancy_count_additions": 31 * (n + 1),
            "coincidence_population_products": n,
            "coincidence_population_square_roots": n,
            "coincidence_population_clamps": n,
            "coincidence_gain_fills": n,
            "coincidence_gain_divisions": n,
            "coincidence_feature_cast_values": f,
            "coincidence_feature_scalings": f,
            "coincidence_named_operand_cast_bytes": 18 * rows,
            "coincidence_named_rail_current_bytes": 18 * f,
            "coincidence_named_and_activity_bytes": 18 * f,
            "coincidence_named_match_input_bytes": 18 * f,
            "coincidence_named_match_current_bytes": f,
            "coincidence_named_match_activity_bytes": f,
            "coincidence_named_feature_bytes": 4 * f,
            "coincidence_named_occupancy_bytes": rows,
            "coincidence_named_occupied_count_bytes": 4 * (n + 1),
            "coincidence_named_denominator_bytes": 4 * n,
            "coincidence_named_gain_bytes": 4 * n,
            "coincidence_named_allocated_bytes": 60 * f + 612 * (n + 1) + 8 * n,
            "query_lag_shift_values": 558 * q,
            "query_lag_shift_copy_values": 558 * q,
            "query_input_rail_values": 18 * q,
            "query_token_bytes": 8 * q,
            "query_raw_lag_bytes": 576,
        }
    )
    return receipt, state


def test_coincidence_receipt_maximum_complete_row(tmp_path):
    import json
    import math
    from hashlib import sha256
    from types import SimpleNamespace as Value

    from experiments.r4_integrated import check_neuronal_byte_coincidence_v1 as checker
    from experiments.r4_integrated import neuronal_splice_receipts_v3 as writer
    from experiments.r4_integrated.neuronal_splice_receipts_v1 import proposal_digest, tree_bytes

    free, state = _coincidence_receipt_qa_reader(
        128, 17, kind="competitive", frontend="coincidence", raw=31
    )
    teacher, _ = _coincidence_receipt_qa_reader(
        128, 17, kind="teacher", frontend="coincidence", raw=31
    )
    heads = {
        head: Value(logits=[0.0] * n, probabilities=[1 / n] * n, selected_index=0)
        for head, n in (("start", 516), ("end", 129), ("operation", 2))
    }
    proposal = dict(
        operation="STOP",
        path="a.py",
        start=0,
        end=0,
        replacement_hex=(b"A" * 16).hex(),
        stop_reason="eos",
        workspace_sha256="a" * 64,
    )
    prediction = Value(
        proposal=Value(to_record=lambda: dict(proposal)),
        attempt_id="authored:1",
        proposal_sha256=proposal_digest(proposal),
        model_sha256="b" * 64,
        memory_sha256="c" * 64,
        choices=heads,
        payload=free,
        work={"action": {}, "reader": free.work},
    )
    feedback = Value(
        attempt_id="authored:1",
        action_updates={head: False for head in heads},
        action_losses={head: math.log(len(value.logits)) for head, value in heads.items()},
        teacher_choices=heads,
        reader_feedback=teacher,
        work={"action": {}, "reader": teacher.work},
    )
    _competitive_receipt_qa_save(
        tmp_path / "maximum-input-raw.json",
        {"free": free, "teacher": teacher, "proposal": proposal, "heads": heads},
    )
    row = {
        "prediction": writer.compact_prediction(
            prediction,
            state,
            expected_policy="competitive_free",
            expected_frontend="coincidence",
            query_raw_bytes=31,
        ),
        "feedback": writer.compact_feedback(
            feedback, state, expected_frontend="coincidence", query_raw_bytes=31
        ),
        "metadata": "x" * 32768,
        "host_path": "p" * 1024,
        "executable_path": "e" * 1024,
        "bootstrap": "b" * 4096,
        "stdout_hex": "ff" * 16384,
        "stderr_hex": "ff" * 16384,
        "diagnostics": ["d" * 1024, "e" * 1024],
        "ids": ["i" * 128 for _ in range(12)],
        "hashes": ["a" * 64 for _ in range(16)],
        "work": [{f"counter_{i:03d}".ljust(96, "x"): 10**12 for i in range(160)} for _ in range(4)],
    }
    _competitive_receipt_qa_save(tmp_path / "maximum-result-raw.json", row)
    encoded = writer.bounded_record(row)
    with (tmp_path / "maximum.jsonl").open("xb") as stream:
        stream.write(encoded)
    parsed = json.loads(encoded)
    for compact, policy in (
        (parsed["prediction"]["payload"], "competitive_free"),
        (parsed["feedback"]["reader"], "all_address_teacher"),
    ):
        writer.validate_reader(
            compact,
            state.successor,
            expected_policy=policy,
            expected_frontend="coincidence",
            query_raw_bytes=31,
        )
        checker.check_reader(
            compact, state.successor, policy, expected_frontend="coincidence", query_raw_bytes=31
        )
    assert parsed == row
    assert writer.LINE_LIMIT == writer.TREE_LIMIT == 2**21
    assert len(encoded) <= writer.LINE_LIMIT
    assert tree_bytes(row) <= writer.TREE_LIMIT
    assert tree_bytes(parsed) <= writer.TREE_LIMIT
    compact = parsed["prediction"]["payload"]
    assert len(compact["competition"]["pair_currents"]) == 8128
    assert len(compact["competition"]["comparator_activity"]) == 8128
    assert compact["initial_log_belief"] == [0.0]
    assert len(parsed["feedback"]["reader"]["initial_log_belief"]) == 128
    assert all(
        set(step) == {"token", "nll", "mixture", "posterior"}
        for part in (compact, parsed["feedback"]["reader"])
        for step in part["steps"]
    )
    _competitive_receipt_qa_save(
        tmp_path / "maximum-passed.json",
        {
            "status": "passed",
            "line_bytes": len(encoded),
            "writer_tree_bytes": tree_bytes(row),
            "parsed_tree_bytes": tree_bytes(parsed),
            "sha256": sha256(encoded).hexdigest(),
            "models": 0,
            "actions": 0,
            "synthetic_step_records": 34,
            "direct_calls": {
                "compact_prediction": 1,
                "compact_feedback": 1,
                "bounded_record": 1,
                "validate_reader": 2,
                "check_reader": 2,
            },
        },
    )


def test_coincidence_receipt_frontend_work_and_query_bindings(tmp_path):
    import copy

    import pytest

    from experiments.r4_integrated import check_neuronal_byte_coincidence_v1 as checker
    from experiments.r4_integrated import neuronal_splice_receipts_v3 as writer

    fixtures, rows = {}, {}
    for frontend in ("coincidence", "graded_random_code"):
        for phase, kind in (
            ("competitive_free", "competitive"),
            ("all_address_teacher", "teacher"),
        ):
            raw, state = _coincidence_receipt_qa_reader(2, 2, kind=kind, frontend=frontend, raw=1)
            fixtures[frontend + "/" + phase] = raw
    _competitive_receipt_qa_save(tmp_path / "dispatch-input-raw.json", fixtures)
    for key, raw in fixtures.items():
        frontend, phase = key.split("/")
        row = writer.compact_reader(
            raw, state, expected_policy=phase, expected_frontend=frontend, query_raw_bytes=1
        )
        rows[key] = row
        writer.validate_reader(
            row,
            state.successor,
            expected_policy=phase,
            expected_frontend=frontend,
            query_raw_bytes=1,
        )
        checker.check_reader(
            row, state.successor, phase, expected_frontend=frontend, query_raw_bytes=1
        )
    _competitive_receipt_qa_save(tmp_path / "dispatch-result-raw.json", rows)

    negatives = []

    def negative(
        name,
        *,
        frontend="coincidence",
        phase="competitive_free",
        expected_frontend=None,
        expected_policy=None,
        query=1,
        work_key=None,
        value=None,
        remove=False,
    ):
        row = copy.deepcopy(rows[frontend + "/" + phase])
        if work_key is not None:
            if remove:
                del row["work"][work_key]
            else:
                row["work"][work_key] = value
        negatives.append(
            {
                "name": name,
                "row": row,
                "frontend": expected_frontend or frontend,
                "policy": expected_policy or phase,
                "query": query,
            }
        )

    negative("candidate_as_graded", expected_frontend="graded_random_code")
    negative("graded_as_candidate", frontend="graded_random_code", expected_frontend="coincidence")
    negative("teacher_as_free", phase="all_address_teacher", expected_policy="competitive_free")
    negative("free_as_teacher", expected_policy="all_address_teacher")
    negative("query_length", query=2)
    negative("missing_gate_counter", work_key="coincidence_threshold_evaluations", remove=True)
    negative("retired_dense_products", work_key="address_feature_products", value=1)
    negative("current_storage", work_key="coincidence_named_rail_current_bytes", value=36865)
    negative("threshold_work", work_key="coincidence_threshold_evaluations", value=38913)
    negative("extra_counter", work_key="coincidence_undeclared", value=0)
    negative("boolean_counter", work_key="coincidence_feature_calls", value=True)
    negative(
        "teacher_local_reduction",
        phase="all_address_teacher",
        work_key="address_local_slot_reductions",
        value=1025,
    )
    negative("missing_shift_copy", work_key="query_lag_shift_copy_values", remove=True)
    negative(
        "graded_extra_gate",
        frontend="graded_random_code",
        work_key="coincidence_feature_calls",
        value=0,
    )
    negative(
        "teacher_enabled_flag",
        phase="all_address_teacher",
        work_key="enabled_codec_update_groups",
        value=2,
    )
    negative("teacher_query_length", phase="all_address_teacher", query=2)
    _competitive_receipt_qa_save(tmp_path / "dispatch-negative-raw.json", negatives)
    assert len(negatives) == 16
    for item in negatives:
        with pytest.raises(ValueError):
            writer.validate_reader(
                item["row"],
                state.successor,
                expected_policy=item["policy"],
                expected_frontend=item["frontend"],
                query_raw_bytes=item["query"],
            )
        with pytest.raises(ValueError):
            checker.check_reader(
                item["row"],
                state.successor,
                item["policy"],
                expected_frontend=item["frontend"],
                query_raw_bytes=item["query"],
            )
    _competitive_receipt_qa_save(
        tmp_path / "dispatch-passed.json",
        {
            "status": "passed",
            "synthetic_readers": 4,
            "synthetic_step_records": 8,
            "negative_fixtures": 16,
            "expected_rejections": 32,
            "direct_calls": {"compact_reader": 4, "validate_reader": 20, "check_reader": 20},
            "models": 0,
            "actions": 0,
            "gate_evaluations": 0,
            "teacher_executions": 0,
            "dtype_scope": (
                "trusted frontend dispatch only; checkpoint dtype is an enclosing audit check"
            ),
        },
    )


def test_coincidence_splice_prior_worker_exchange():
    """Five typed/graded prior continuations; one coincidence worker action, no teaching."""
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from pathlib import Path
    from types import SimpleNamespace

    from experiments.r4_integrated.check_neuronal_byte_coincidence_v1 import check_reader
    from experiments.r4_integrated.check_neuronal_splice_online_v3 import (
        expected_reader_metadata,
        initial_action_tensors,
    )
    from experiments.r4_integrated.neuronal_splice_endpoint_v3 import (
        check_acknowledgement,
        check_proposal,
    )
    from experiments.r4_integrated.neuronal_splice_io_v1 import (
        JsonlJournal,
        process_identity,
        read_row,
        write_message,
    )
    from experiments.r4_integrated.neuronal_splice_models_v3 import (
        ARMS,
        address_frontend,
        reader_policy,
    )
    from experiments.r4_integrated.neuronal_splice_online_v1 import (
        execute_action,
        qualified_launch,
        workspace_from_records,
    )
    from experiments.r4_integrated.neuronal_splice_prior_v3 import initialize_models, tensor_bytes
    from experiments.r4_integrated.neuronal_splice_receipts_v3 import validate_prediction
    from experiments.r4_integrated.neuronal_splice_worker_v3 import (
        LIMITS,
        SOURCE_MODULES,
        WORKER_SOURCE,
        check_requests,
        file_hash,
        write_json,
    )
    from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner
    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
    from humanbrain.local_memory_learner import LocalMemoryLearner
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner, _digest

    root = Path(__file__).resolve().parents[1]
    output = root / ".pytest-tmp/coincidence-splice-prior-worker-qa-v1"
    output.mkdir(exist_ok=False)
    deadline = float(os.environ["HUMANBRAIN_RUN_DEADLINE_UNIX"])
    assert 0 < deadline - time.time() <= 60
    spec = SimpleNamespace(config={"receipt_limits": LIMITS})
    baseline_paths = {
        *SOURCE_MODULES.values(),
        WORKER_SOURCE,
        "experiments/r4_integrated/neuronal_splice_prior_v3.py",
        "experiments/r4_integrated/neuronal_splice_prior_v1.py",
        "experiments/r4_integrated/neuronal_splice_endpoint_v3.py",
        "experiments/r4_integrated/check_neuronal_address_competition_v1.py",
        "experiments/r4_integrated/check_neuronal_byte_coincidence_v1.py",
        "experiments/r4_integrated/check_neuronal_splice_online_v3.py",
        "experiments/r4_integrated/check_neuronal_splice_online_v1.py",
        "experiments/r4_integrated/neuronal_splice_online_v1.py",
    }
    baseline = {
        "source_sha256": {name: file_hash(root / name) for name in sorted(baseline_paths)},
        "test_sha256": file_hash(Path(__file__)),
        "parent_identity": process_identity(),
    }
    write_json(output / "before-initialization.json", baseline)
    models, custody = initialize_models(root)
    write_json(output / "prior.json", custody)
    assert (
        tuple(models)
        == ARMS
        == ("learned", "action_frozen", "slow_frozen", "memory_disabled", "graded_reference")
    )
    assert custody["schema"] == "neuronal-splice-prior-custody-v3"
    assert custody["actual_loaded_payload_tensor_bytes"] == 33801664
    assert custody["historical_payload_tensor_bytes"] == 33801664
    assert custody["new_model_tensor_bytes"] == 3172160
    assert custody["selected_entry"] == "learned.weights"
    assert custody["reader_policy"] == {arm: reader_policy(arm) for arm in ARMS}
    initial_actions = initial_action_tensors(torch)
    original_metadata = LocalMemoryLearner.get_extra_state(models["graded_reference"].reader)
    assert _digest(original_metadata) == custody["original_reader_metadata_sha256"]
    assert custody["address_frontend"] == {arm: address_frontend(arm) for arm in ARMS}
    reader_tensors = {
        key: value
        for key, value in models["graded_reference"].reader.state_dict().items()
        if isinstance(value, torch.Tensor)
    }
    assert _digest(reader_tensors) == custody["initial_reader_tensor_sha256"]
    initial_model_hashes = {}
    for arm, current in models.items():
        mode = "learned" if arm == "graded_reference" else arm
        expected_type = (
            CompetitiveLocalMemoryLearner
            if arm == "graded_reference"
            else CoincidenceLocalMemoryLearner
        )
        assert type(current.reader) is expected_type
        assert current.mode == mode and current.reader.mode == NeuronalSpliceLearner.MODES[mode]
        assert current._pending is None and current._experience == 0
        assert all(
            not value.requires_grad
            and value.grad is None
            and value.grad_fn is None
            and torch.isfinite(value).all()
            for value in current.parameters()
        )
        assert all(
            torch.equal(current.state_dict()[key], value) for key, value in initial_actions.items()
        )
        assert all(
            torch.equal(current.reader.state_dict()[key], value)
            for key, value in reader_tensors.items()
        )
        metadata = current.reader.get_extra_state()
        changed = sorted(
            key
            for key in set(metadata) | set(original_metadata)
            if metadata.get(key) != original_metadata.get(key)
        )
        assert changed == custody["metadata_fields_changed"][arm]
        assert metadata == expected_reader_metadata(original_metadata, arm)
        assert _digest(metadata) == custody["initial_reader_metadata_sha256"][arm]
        assert _digest(current.actions.state_dict()) == custody["initial_action_tensor_sha256"][arm]
        assert metadata["mode"] == current.reader.mode
        assert reader_policy(arm) == "competitive_free"
        initial_model_hashes[arm] = _digest(current.state_dict())
    assert initial_model_hashes == custody["initial_model_sha256"]
    for key in reader_tensors:
        assert (
            len({current.reader.state_dict()[key].data_ptr() for current in models.values()}) == 5
        )
    for key in initial_actions:
        assert len({current.state_dict()[key].data_ptr() for current in models.values()}) == 5
    model = models["learned"]
    document = b"Record k00 has value ab."
    state = model.ingest([document])
    assert len(document) == 24 and state.used == 25
    assert state.keys.dtype == torch.bool and tuple(state.keys.shape) == (128, 32, 18)
    assert tensor_bytes(state.snapshot()) == 206344
    expected_keys = torch.zeros(128, 32, 18, dtype=torch.bool)
    for position in range(25):
        for lag, token in enumerate(reversed(document[:position][-32:])):
            for bit in range(9):
                expected_keys[position, lag, bit * 2 + ((token >> bit) & 1)] = True
    assert torch.equal(state.keys, expected_keys)
    del expected_keys
    successor = state.successor[: state.used].tolist()
    model_hash, memory_hash = _digest(model.state_dict()), _digest(state.snapshot())
    workspace = workspace_from_records(
        [
            {"path": "a.py", "content_hex": b"x = 'old'\n".hex()},
            {"path": "main.py", "content_hex": b"import a\nprint(a.x)\n".hex()},
        ]
    )
    task = {
        "task_id": "endpoint-0",
        "memory_id": "qa-memory",
        "request_hex": b"SET a.py x k00".hex(),
        "workspace_sha256": workspace.sha256,
        "files": [
            {"path": item.path, "content_hex": item.content.hex()} for item in workspace.files
        ],
    }
    package = output / "package"
    package.mkdir()
    sources = {}
    for relative in (*SOURCE_MODULES.values(), WORKER_SOURCE):
        destination = package / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / relative, destination)
        assert file_hash(destination) == file_hash(root / relative)
        sources[relative] = file_hash(destination)
    assert len(sources) == 20
    assert {path.relative_to(package).as_posix() for path in package.rglob("*.py")} == set(sources)
    assert all(digest == baseline["source_sha256"][name] for name, digest in sources.items())
    checkpoint = output / "source_free.pt"
    payload = {
        "schema": "neuronal-splice-single-exchange-qa-state-v3",
        "arms": {
            "learned": {
                "weights": model.state_dict(),
                "memories": {"qa-memory": state.snapshot()},
                "experience": 0,
            }
        },
    }
    assert tensor_bytes(payload) == 840776
    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)
    del payload
    checkpoint_hash = file_hash(checkpoint)
    requests = {
        "schema": "neuronal-splice-single-exchange-qa-request-v3",
        "checkpoint_sha256": checkpoint_hash,
        "source_sha256": sources,
        "receipt_limits": LIMITS,
        "arms": ["learned"],
        "reading_cases": [],
        "cases": [
            {
                "id": task["task_id"],
                **{key: task[key] for key in ("memory_id", "request_hex", "files")},
            }
        ],
    }
    check_requests(requests, single_exchange_qa=True)
    with pytest.raises(ValueError):
        check_requests(requests)
    request_path = output / "query_only.json"
    write_json(request_path, requests)
    requests_hash = file_hash(request_path)
    qualification_paths = {
        ".pytest-tmp/temporary-module-bootstrap-v1-qualification.json": (
            "0bb809fd33aee145f169776a03a884fd62daf381ff4e67ed333a2a378c46d92b"
        ),
        ".pytest-tmp/stdlib-python-action-v1-qualification.json": (
            "4c3a6438549ce1ecd4c8d92509f1a60753692b8aab1ab9155df00b12db10941c"
        ),
    }
    assert all(file_hash(root / name) == digest for name, digest in qualification_paths.items())
    launch = qualified_launch(root)
    write_json(
        output / "qualification.json",
        {
            "qualification_sha256": qualification_paths,
            "launch_record": launch.to_record(),
            "launch_record_sha256": launch.sha256,
        },
    )
    command = [
        sys.executable,
        "-I",
        "-X",
        "utf8",
        str(package / WORKER_SOURCE),
        "--package-root",
        str(package),
        "--checkpoint",
        str(checkpoint),
        "--checkpoint-sha",
        checkpoint_hash,
        "--requests",
        str(request_path),
        "--requests-sha",
        requests_hash,
        "--output",
        str(output / "worker"),
        "--deadline-unix",
        repr(deadline),
        "--single-exchange-qa",
    ]
    action_journal = JsonlJournal(output / "actions.jsonl", spec)
    try:
        with (output / "worker-stderr.bin").open("xb") as stderr:
            worker_started_unix = time.time()
            process = subprocess.Popen(
                command, cwd=output, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr
            )
            try:
                launcher_identity = process_identity(process.pid)
                write_json(
                    output / "worker-launch.json",
                    {
                        "command": command,
                        "cwd": str(output),
                        "pid": process.pid,
                        "process_identity": launcher_identity,
                        "parent_pid": os.getpid(),
                        "baseline_sha256": file_hash(output / "before-initialization.json"),
                        "parent_identity": baseline["parent_identity"],
                        "started_unix": worker_started_unix,
                    },
                )
                proposal_row = read_row(process.stdout, spec)
                write_json(output / "proposal.json", proposal_row)
                proposal = check_proposal(
                    proposal_row, "learned", 0, task, model_hash, memory_hash, successor=successor
                )
                validate_prediction(
                    proposal_row["prediction"],
                    successor,
                    expected_policy="competitive_free",
                    expected_frontend="coincidence",
                    query_raw_bytes=14,
                    boundaries=workspace.boundaries,
                )
                payload_row = proposal_row["prediction"]["payload"]
                summary = check_reader(
                    payload_row,
                    successor,
                    "competitive_free",
                    expected_frontend="coincidence",
                    query_raw_bytes=14,
                )
                assert payload_row["work"]["query_raw_bytes"] == 14
                assert payload_row["work"]["coincidence_threshold_evaluations"] == 19 * 25 * 1024
                assert payload_row["work"]["coincidence_input_applications"] == 54 * 25 * 1024
                assert payload_row["work"]["address_feature_products"] == 0
                assert payload_row["work"]["address_feature_channel_reductions"] == 0
                assert payload_row["work"]["query_population_normalizations"] == 0
                assert len(payload_row["competition"]["scores"]) == 25
                assert len(payload_row["competition"]["pair_currents"]) == 300
                assert len(summary["final_cursors"]) == 1
                assert payload_row["work"]["codec_slot_evaluations"] == 25
                assert payload_row["work"]["codec_response_gathers"] == summary["decisions"]
                identity = {
                    key: proposal_row[key]
                    for key in ("schema", "type", "arm", "index", "id", "attempt_id")
                }
                envelope, action_hash, _, _ = execute_action(
                    task=None,
                    workspace=workspace,
                    proposal=proposal,
                    attempt_id=identity["attempt_id"],
                    phase="endpoint",
                    arm="learned",
                    index=0,
                    output=output,
                    launch=launch,
                    journal=action_journal,
                )
                write_message(
                    process.stdin,
                    {
                        **identity,
                        "type": "executed",
                        "action_sha256": action_hash,
                        "outcome": envelope,
                    },
                    spec,
                )
                acknowledgement = read_row(process.stdout, spec)
                write_json(output / "acknowledgement.json", acknowledgement)
                check_acknowledgement(
                    acknowledgement, identity, action_hash, model_hash, memory_hash
                )
                with pytest.raises(ValueError):
                    check_acknowledgement(
                        {**acknowledgement, "action_sha256": "0" * 64},
                        identity,
                        action_hash,
                        model_hash,
                        memory_hash,
                    )
                process.stdin.close()
                process.wait(timeout=max(0.001, deadline - time.time()))
                assert process.returncode == 0 and read_row(process.stdout, spec) is None
                worker_finished_unix = time.time()
            except BaseException:
                if process.poll() is None:
                    process.kill()
                    process.wait()
                raise
            finally:
                process.stdout.close()
                if not process.stdin.closed:
                    process.stdin.close()
    finally:
        action_log = action_journal.close()
    completion = json.loads((output / "worker/completion.json").read_bytes())
    assert completion["schema"] == "neuronal-splice-endpoint-completion-v3"
    assert completion["scope"] == "single_exchange_qa" and completion["status"] == "passed"
    assert completion["pid"] == process.pid or completion["parent_pid"] == process.pid
    worker_identity = completion["process_identity"]
    assert worker_identity["pid"] == completion["pid"]
    assert baseline["parent_identity"]["creation_ticks"] <= launcher_identity["creation_ticks"]
    assert launcher_identity["creation_ticks"] <= worker_identity["creation_ticks"]
    if completion["pid"] == process.pid:
        assert worker_identity == launcher_identity and completion["parent_pid"] == os.getpid()
    else:
        assert completion["parent_pid"] == launcher_identity["pid"]
    worker_created_unix = (worker_identity["creation_ticks"] - 621355968000000000) / 10**7
    assert worker_started_unix - 0.001 <= worker_created_unix <= worker_finished_unix + 0.001
    assert completion["checkpoint_sha256"] == file_hash(checkpoint) == checkpoint_hash
    assert completion["requests_sha256"] == file_hash(request_path) == requests_hash
    assert completion["source_sha256"] == sources
    assert completion["source_origins"] == {
        key: str(package / path) for key, path in SOURCE_MODULES.items()
    }
    assert completion["final"] == {
        "learned": {
            "model_sha256": model_hash,
            "memories_sha256": {"qa-memory": memory_hash},
            "experience": 1,
        }
    }
    assert completion["endpoint_log"]["rows"] == 2 and completion["reading_log"]["rows"] == 0
    for name, filename in (("endpoint_log", "endpoints.jsonl"), ("reading_log", "reading.jsonl")):
        log = completion[name]
        log_path = output / "worker" / filename
        assert set(log) == {
            "path",
            "rows",
            "bytes",
            "sha256",
            "max_line_bytes",
            "max_row_tree_bytes",
        }
        assert log["path"] == filename and log["bytes"] == log_path.stat().st_size
        assert log["sha256"] == file_hash(log_path)
        assert 0 <= log["max_line_bytes"] <= LIMITS["max_line_bytes"]
        assert 0 <= log["max_row_tree_bytes"] <= LIMITS["max_row_tree_bytes"]
        if log["rows"]:
            assert log["bytes"] > 0 and log["max_line_bytes"] > 0 and log["max_row_tree_bytes"] > 0
        else:
            assert log["bytes"] == log["max_line_bytes"] == log["max_row_tree_bytes"] == 0
    assert (output / "worker-stderr.bin").stat().st_size <= 1024
    assert _digest(model.state_dict()) == model_hash and _digest(state.snapshot()) == memory_hash
    assert {
        arm: _digest(current.state_dict()) for arm, current in models.items()
    } == initial_model_hashes
    assert all(current._pending is None and current._experience == 0 for current in models.values())
    assert all(
        file_hash(root / relative) == file_hash(package / relative) == digest
        for relative, digest in sources.items()
    )
    assert all(file_hash(root / name) == digest for name, digest in qualification_paths.items())
    assert {name: file_hash(root / name) for name in baseline["source_sha256"]} == baseline[
        "source_sha256"
    ]
    assert file_hash(Path(__file__)) == baseline["test_sha256"]
    assert action_log["rows"] == 1
    assert acknowledgement["work"] == {
        "action": {"custody_tensor_bytes_hashed": 840776},
        "reader": {},
    }
    steps = proposal_row["prediction"]["payload"]["steps"]
    assert 1 <= len(steps) <= 17
    write_json(
        output / "passed.json",
        {
            "schema": "coincidence-splice-prior-worker-qa-v1",
            "status": "passed",
            "parent_pid": os.getpid(),
            "baseline_sha256": file_hash(output / "before-initialization.json"),
            "launcher_identity": launcher_identity,
            "worker_identity": completion["process_identity"],
            "action_log": action_log,
            "prior": custody,
            "worker_completion_sha256": file_hash(output / "worker/completion.json"),
            "source_sha256": sources,
            "test_sha256": file_hash(Path(__file__)),
            "combined_model_constructions": 6,
            "parent_models": 5,
            "reader_prior_copies": 5,
            "prior_loads": 1,
            "worker_checkpoint_loads": 1,
            "ingests": 1,
            "document_raw_bytes": len(document),
            "document_eos": 1,
            "free_proposals": 1,
            "actual_payload_decisions": len(steps),
            "action_children": 1,
            "worker_processes": 1,
            "teachers": 0,
            "reading_panel_calls": 0,
            "checkpoint_output_probes": 0,
            "query_raw_bytes": 14,
            "query_eos": 1,
            "qualification_sha256": qualification_paths,
            "launch_record_sha256": launch.sha256,
            "reader_policy": reader_policy("learned"),
            "address_frontend": {arm: address_frontend(arm) for arm in ARMS},
            "tape_tensor_bytes": 206344,
            "worker_payload_tensor_bytes": 840776,
            "actual_reader_work": payload_row["work"],
            "initial_model_sha256": initial_model_hashes,
            "proposal_sha256": proposal.sha256,
            "action_line_sha256": action_hash,
            "qualification_scope": (
                "Same pending neuronal worker through actual execution; no acquired task skill."
            ),
        },
    )


def test_coincidence_splice_saved_cost_contract(tmp_path, monkeypatch):
    """Authored JSON only; six admission calls, no genuine runtime evidence."""
    import ast
    import copy
    import hashlib
    import json
    from pathlib import Path

    from experiments.r4_integrated import check_neuronal_splice_online_v3 as checker

    root = Path(__file__).resolve().parents[1]
    raw = root / ".pytest-tmp/coincidence-cost-qa-v1-raw.jsonl"
    passed = root / ".pytest-tmp/coincidence-cost-qa-v1.json"
    assert not raw.exists() and not passed.exists()
    old_source = root / "experiments/r4_integrated/check_neuronal_splice_online_v2.py"
    new_source = Path(checker.__file__)
    trees = [ast.parse(path.read_text(encoding="utf-8")) for path in (old_source, new_source)]
    definitions = [
        {node.name: ast.dump(node, include_attributes=False)
         for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
        for tree in trees
    ]
    unchanged = sorted(
        name for name in definitions[0]
        if definitions[0][name] == definitions[1].get(name)
    )
    assert len(unchanged) == 29
    assert {"project_cost", "check_monitor", "Stream"} <= set(unchanged)
    checker_sha = hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest()
    relative = "experiments/r4_integrated/check_neuronal_splice_online_v3.py"
    fake_checker = tmp_path / relative
    fake_checker.parent.mkdir(parents=True)
    source = b"# Authored source identity only; never imported.\n"
    fake_checker.write_bytes(source)
    monkeypatch.setattr(checker, "ROOT", tmp_path)
    monkeypatch.setattr(checker, "__file__", str(fake_checker))
    config = checker.expected_config()
    arms = ["learned", "action_frozen", "slow_frozen", "memory_disabled", "graded_reference"]
    assert config == {
        "schema": "neuronal-splice-online-v3", "arms": arms, "seed": 733,
        "reader_policy": dict.fromkeys(arms, "competitive_free"),
        "address_frontend": {arm: "graded_random_code" if arm == "graded_reference"
                             else "coincidence" for arm in arms},
        "learning_rate": 0.01, "lessons_per_arm": 256,
        "cost_task_ids": ["T00", "T08", "T16", "T24"],
        "fresh_per_arm": 32, "retention_per_arm": 16, "panel_per_pass_per_arm": 20,
        "retained_memories_per_arm": 72,
        "retained_tensor_bytes": 5 * 634432 + 72 * (4 * 206344 + 656904),
        "receipt_limits": {"max_line_bytes": 2**21, "max_row_tree_bytes": 2**21,
                           "max_live_rows": 2, "max_line_copies": 2, "allocation_factor": 2},
        "seconds": 900, "tree_mib": 2048, "run_reserve_mib": 128,
        "audit_reserve_mib": 32, "buffer_mib": 24, "duration_clock": "perf_counter",
        "full_only_tensor_bytes": 0,
    }
    assert config["retained_tensor_bytes"] == 109896320
    sources = {relative: hashlib.sha256(source).hexdigest()}
    roster_path = tmp_path / "authored-roster.json"
    roster_sha = _competitive_cost_save(roster_path, {"schema": "synthetic-only", "tasks": []})
    cost_dir = tmp_path / "cost"
    cost_dir.mkdir()
    paths = {
        "cost_report": cost_dir / "report.json",
        "cost_completion": cost_dir / "completion.json",
        "cost_launch": tmp_path / "cost.launch.json",
        "cost_admission": tmp_path / "cost-admission.json",
        "cost_audit": tmp_path / "cost-audit.json",
        "cost_audit_launch": tmp_path / "cost-audit.launch.json",
    }
    report, outer, audit, audit_outer = _competitive_cost_synthetic_inputs()
    preliminary = _competitive_cost_authored_projection(
        report, outer, audit, audit_outer, finished=False
    )
    final = _competitive_cost_authored_projection(report, outer, audit, audit_outer, finished=True)
    # Preserve the old authored durations and their independent scalar oracle;
    # rename only the fifth arm after that oracle has made its asymmetric row.
    for mapping in (report["timing_summary"], audit["learning_pair_timing"],
                    preliminary["per_arm"], final["per_arm"]):
        mapping["graded_reference"] = mapping.pop("marginal_reference")
    values, hashes = {}, {}
    values["cost_admission"] = {
        "schema": "neuronal-splice-admission-v3",
        "status": "passed",
        "authorized_phase": "cost",
        "qa_passed": True,
        "config": config,
        "source_sha256": sources,
        "roster_path": str(roster_path),
        "roster_sha256": roster_sha,
    }
    hashes["cost_admission"] = _competitive_cost_save(
        paths["cost_admission"], values["cost_admission"]
    )
    report.update(
        schema="neuronal-splice-online-v3",
        config=config,
        source_sha256=sources,
        source_sha256_after=sources,
        admission_sha256=hashes["cost_admission"],
        roster_sha256=roster_sha,
        pid=101,
        deadline_unix=1900.0,
        logs={},
    )
    values["cost_report"] = report
    hashes["cost_report"] = _competitive_cost_save(paths["cost_report"], report)
    values["cost_completion"] = {
        "schema": "neuronal-splice-completion-v3",
        "status": "completed",
        "mode": "cost",
        "report_sha256": hashes["cost_report"],
        "admission_sha256": hashes["cost_admission"],
        "logs": {},
    }
    hashes["cost_completion"] = _competitive_cost_save(
        paths["cost_completion"], values["cost_completion"]
    )
    prefix = [str(tmp_path / ".venv/Scripts/python.exe"), "-X", "utf8", "-m"]
    outer["command"] = [
        *prefix,
        "experiments.r4_integrated.neuronal_splice_online_v3",
        "cost",
        "--output",
        str(cost_dir),
        "--admission",
        str(paths["cost_admission"]),
        "--admission-sha",
        hashes["cost_admission"],
    ]
    values["cost_launch"] = outer
    hashes["cost_launch"] = _competitive_cost_save(paths["cost_launch"], outer)
    bound = {
        str(paths[name]): hashes[name]
        for name in ("cost_report", "cost_completion", "cost_launch", "cost_admission")
    }
    bound.update({str(fake_checker): sources[relative], str(roster_path): roster_sha})
    audit.update(
        schema="neuronal-splice-independent-audit-v3",
        mode="cost",
        source_sha256=sources,
        roster_sha256=roster_sha,
        report_sha256=hashes["cost_report"],
        completion_sha256=hashes["cost_completion"],
        launch_sha256=hashes["cost_launch"],
        admission_sha256=hashes["cost_admission"],
        input_sha256=bound,
        projection=preliminary,
    )
    values["cost_audit"] = audit
    hashes["cost_audit"] = _competitive_cost_save(paths["cost_audit"], audit)
    audit_outer["command"] = [
        *prefix,
        "experiments.r4_integrated.check_neuronal_splice_online_v3",
        "--output",
        str(paths["cost_audit"]),
        "--cost-dir",
        str(cost_dir),
        "--launch",
        str(paths["cost_launch"]),
        "--admission",
        str(paths["cost_admission"]),
        "--admission-sha",
        hashes["cost_admission"],
    ]
    values["cost_audit_launch"] = audit_outer
    hashes["cost_audit_launch"] = _competitive_cost_save(paths["cost_audit_launch"], audit_outer)
    decision = {
        "schema": "neuronal-splice-cost-decision-v3",
        "status": "passed",
        "config": config,
        "source_sha256": sources,
        "roster_sha256": roster_sha,
        "completed_cost_audit": True,
        "inputs": {key: {"path": str(path), "sha256": hashes[key]} for key, path in paths.items()},
        "projection": final,
        "projected_seconds": final["projected_seconds"],
        "projected_tree_mib": final["projected_tree_mib"],
    }
    outcomes = {}
    with raw.open("x", encoding="utf-8") as log:
        log.write(json.dumps({
            "schema": "coincidence-cost-qa-raw-v1", "checker_sha256": checker_sha,
            "old_checker_sha256": hashlib.sha256(old_source.read_bytes()).hexdigest(),
            "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "unchanged_functions_and_classes": unchanged, "config": config,
        }) + "\n")
        log.flush()
        for label in ("valid", "frontend", "mixed_payload", "schema", "command", "source"):
            altered, selected = copy.deepcopy(values), copy.deepcopy(decision)
            fake_checker.write_bytes(source + (b"# Changed.\n" if label == "source" else b""))
            expected = copy.deepcopy(config)
            if label == "frontend":
                expected["address_frontend"]["graded_reference"] = "coincidence"
                selected["config"] = expected
            elif label == "mixed_payload":
                expected["retained_tensor_bytes"] = 239657600
                selected["config"] = expected
            elif label == "schema":
                selected["schema"] = "neuronal-splice-cost-decision-v2"
            elif label == "command":
                altered["cost_audit_launch"]["command"][4] = (
                    "experiments.r4_integrated.check_neuronal_splice_online_v2"
                )
            for name, value in altered.items():
                selected["inputs"][name]["sha256"] = _competitive_cost_save(paths[name], value)
            decision_path = tmp_path / "decision.json"
            decision_sha = _competitive_cost_save(decision_path, selected)
            inputs = {
                "decision": selected,
                "evidence": altered,
                "actual_source_sha256": hashlib.sha256(fake_checker.read_bytes()).hexdigest(),
                "expected_config": expected,
            }
            outcomes[label] = _competitive_cost_observe(
                log,
                label,
                inputs,
                checker.validate_cost_decision,
                decision_path,
                decision_sha,
                expected,
                sources,
                roster_sha,
            )
    assert outcomes["valid"]["result"] == final
    for label, fragment in (
        ("frontend", "not this complete protocol"),
        ("mixed_payload", "not this complete protocol"),
        ("schema", "not this complete protocol"),
        ("command", "different checker"),
        ("source", "not the held source"),
    ):
        assert outcomes[label]["exception"] == "ValueError"
        assert fragment in outcomes[label]["message"]
    with passed.open("x") as stream:
        json.dump(
            {
                "status": "passed",
                "direct_decision_calls": 6,
                "direct_expected_config_calls": 1,
                "valid_decisions": 1,
                "expected_value_errors": 5,
                "unchanged_functions_and_classes": unchanged,
                "old_checker_sha256": hashlib.sha256(old_source.read_bytes()).hexdigest(),
                "checker_sha256": checker_sha,
                "test_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "models": 0,
                "children": 0,
            },
            stream,
        )

# Prospective one-node append to tests/test_byte_sequence_decoder.py; unexecuted.


def test_persistent_coincidence_512_append_and_reader_regression(tmp_path):
    """One ordinary capacity regression, no training stream or actual action."""
    from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner
    from humanbrain.persistent_coincidence_reader import PersistentCoincidenceReader512

    old = CoincidenceLocalMemoryLearner(seed=733)
    reader = PersistentCoincidenceReader512(seed=733)
    for name, value in old.state_dict().items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(reader.state_dict()[name], value)
    state = reader.ingest([b"ab"])
    old_state = old.ingest([b"ab"])
    event = reader._query(b"a", state)
    old_event = old._query(b"a", old_state)
    assert torch.equal(event.activity, old_event.activity)
    assert torch.equal(event.address_features, old_event.address_features)
    assert torch.equal(event.address_scores, old_event.address_scores)
    assert torch.equal(event.log_prior, old_event.log_prior)
    for key, value in old_event.work.items():
        assert event.work[key] == value
    assert event.work["persistent_feature_block_count"] == 1
    assert event.work["persistent_feature_assembly_bytes"] == 0
    # Independent scalar raw coincidence and occupied-count normalization.
    q = [256, 97]
    prefixes = ([], [97], [98, 97])
    for slot, prefix in enumerate(prefixes):
        gain = math.sqrt(32) / math.sqrt(max(1, len(q) * len(prefix)))
        for i in range(32):
            for j in range(32):
                expected = gain if i < len(q) and j < len(prefix) and q[i] == prefix[j] else 0
                assert float(event.address_features[slot, i, j]) == pytest.approx(
                    expected, rel=2e-6, abs=2e-7)
    del event, old_event
    initial = state.snapshot()
    state = reader.ingest([b"a" * 125], state)
    assert state.used == 129
    for name in ("keys", "sensory", "successor"):
        assert torch.equal(getattr(state, name)[:3], initial[name][:3])
    crossed = reader.memory.advance(state, torch.tensor([127], dtype=torch.int64))
    assert crossed.tolist() == [128]
    assert reader.memory.content(state, crossed)[0, 256] == 1
    middle = state.snapshot()
    state = reader.ingest([b"b" * 382], state)
    assert state.used == 512
    for name in ("keys", "sensory", "successor"):
        assert torch.equal(getattr(state, name)[:129], middle[name][:129])
    full = state.snapshot()
    with pytest.raises(ValueError, match="capacity exceeded"):
        reader.ingest([b""], state)
    _assert_nested_equal(state.snapshot(), full)
    all_cursors = reader.memory.initial_recall(state)
    assert all_cursors.tolist() == list(range(512))
    advanced = reader.memory.advance(state, all_cursors)
    assert torch.equal(advanced, state.successor)
    assert reader.memory.accounting(state)["state_tensor_bytes"] == 825352
    with pytest.raises(ValueError):
        reader.memory.restore_state(old_state.snapshot())
    restored = reader.memory.restore_state(full)
    _assert_nested_equal(restored.snapshot(), full)
    # Fresh authored zero scores exercise full512 competition without a teacher.
    with torch.no_grad():
        reader.address_weight.zero_()
    weights = copy.deepcopy(reader.state_dict())
    prediction = reader.predict_with_receipt(b"a", state, max_bytes=0)
    assert len(prediction.steps) == 1
    assert prediction.competition.winner_index == 0
    assert prediction.competition.winner_activity.sum() == 1
    assert prediction.competition.pair_currents.numel() == 130816
    assert torch.count_nonzero(prediction.competition.pair_currents) == 0
    assert prediction.work["competition_candidates"] == 512
    assert prediction.work["competition_pairs"] == 130816
    assert prediction.work["coincidence_feature_calls"] == 4
    assert prediction.work["persistent_feature_block_count"] == 4
    assert prediction.work["coincidence_slots"] == 512
    assert prediction.work["coincidence_threshold_evaluations"] == 19 * 512 * 1024
    assert prediction.work["persistent_feature_assembly_bytes"] == 512 * 1024 * 4
    assert prediction.work["persistent_feature_copy_values"] == 512 * 1024
    assert prediction.work["branch_cursor_initializations"] == 1
    assert prediction.work["codec_bank_slot_indices"] == 512
    _assert_nested_equal(reader.state_dict(), weights)
    _assert_nested_equal(state.snapshot(), full)
    # Exercise the inherited all-address teacher at full capacity without publication.
    feedback = reader.teach(b"a", b"b", state, learning_rate=0.01, learning_enabled=False)
    assert feedback.teacher_count == 2
    assert [step.observed_token for step in feedback.steps] == [98, 256]
    assert all(step.branch_cursors.numel() == 512 for step in feedback.steps)
    assert feedback.work["branch_cursor_initializations"] == 512
    assert feedback.work["branch_cursor_advances"] == 512
    assert feedback.work["address_feature_slots"] == 512
    assert feedback.work["address_local_synapse_contributions"] == 512 * 1024
    assert feedback.work["address_local_slot_reductions"] == 511 * 1024
    assert feedback.work["coincidence_feature_calls"] == 4
    assert feedback.work["coincidence_threshold_evaluations"] == 19 * 512 * 1024
    assert feedback.work["persistent_feature_copy_values"] == 512 * 1024
    assert feedback.work["codec_slot_evaluations"] == 512
    assert feedback.work["codec_output_potentials"] == 512 * 257
    assert feedback.work["codec_response_gathers"] == 512
    assert feedback.work["codec_response_bytes_gathered"] == 512 * 257 * 28
    assert feedback.work["prefix_belief_updates"] == 2 * 512
    assert feedback.work["enabled_codec_update_groups"] == 0
    assert feedback.work["enabled_address_update_groups"] == 0
    assert not feedback.codec_updated and not feedback.address_updated
    _assert_nested_equal(reader.state_dict(), weights)
    _assert_nested_equal(state.snapshot(), full)
    _assert_nested_equal(feedback.state.snapshot(), full)
    _coincidence_qa_save(tmp_path / "persistent-capacity.json", {
        "status": "passed", "reader_constructions": 2, "ingest_calls": 4,
        "successful_ingested_raw_bytes": 511, "successful_ingested_eos": 4,
        "rejected_ingests": 1, "direct_query_calls": 2, "free_prediction_calls": 1,
        "free_decisions": 1, "teacher_calls": 1, "teacher_offers": 2,
        "teacher_work": feedback.work, "learning_enabled": False, "actual_children": 0,
        "file_checkpoint_loads": 0, "memory_restores": 2, "rejected_memory_restores": 1,
        "capacity": 512, "state_tensor_bytes": 825352, "work": prediction.work,
        "winner_index": prediction.competition.winner_index,
        "metadata": reader.get_extra_state(),
        "claim": "append/storage/forward regression, not an acquired-skill gate",
    })


def test_public_language_action_execution_contract(tmp_path):
    """One real child; two private grades, one baseline prediction, no neurons."""
    import copy
    import hashlib
    import json
    import time
    from pathlib import Path

    from experiments.r4_integrated.persistent_language_action_execution_v1 import (
        execute_proposal,
        grade_execution,
        proposal_from_record,
        qualified_launch,
        workspace_from_records,
    )
    from experiments.r4_integrated.persistent_language_action_tasks_v1 import (
        SimpleBaseline,
        build_stream,
        observation,
    )

    root = Path(__file__).resolve().parents[1]
    manifest = build_stream()
    with (tmp_path / "full-manifest.json").open("x") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    launch = qualified_launch(root)
    files = [
        {"path": "cfg.py", "content_hex": b"value = 'old'\n".hex()},
        {"path": "main.py", "content_hex": b"import cfg\nprint(cfg.value)\n".hex()},
    ]
    workspace = workspace_from_records(files)
    proposal = proposal_from_record(
        {
            "operation": "SPLICE",
            "path": "cfg.py",
            "start": 8,
            "end": 13,
            "replacement_hex": b'"blue"'.hex(),
            "stop_reason": "eos",
        },
        workspace,
    )
    after, receipt = execute_proposal(
        workspace,
        proposal,
        tmp_path / "actual-action",
        launch=launch,
        deadline=time.perf_counter() + 10,
    )
    with (tmp_path / "actual-execution.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
    # The canonical single-quoted spelling differs from the actual correct edit.
    grade = {
        "expected_returncode": 0,
        "expected_stdout_hex": b"blue\n".hex(),
        "protected_paths": ["main.py"],
        "expected_after_files": [
            {"path": "cfg.py", "content_hex": b"value = 'blue'\n".hex()},
            files[1],
        ],
    }
    positive = grade_execution(receipt, grade, files)
    # Pure scorer negative: explicitly synthetic, not a second execution receipt.
    protected_counterexample = copy.deepcopy(receipt)
    next(row for row in protected_counterexample["after_files"] if row["path"] == "main.py")[
        "content_hex"
    ] = b"print('blue')\n".hex()
    negative = grade_execution(protected_counterexample, grade, files)
    baseline = SimpleBaseline()
    document = b"Harbor's code is 'Q7'.\n"
    baseline.append_documents([document])
    answer = baseline.predict(b"What is Harbor's code?", [], kind="reading")
    public = observation(
        {
            "kind": "reading",
            "request_hex": b"What is Harbor's code?".hex(),
            "files": [],
            "teacher": {"target_hex": b"private-answer".hex()},
            "grade": {"secret": True},
            "episode_id": "PRIVATE-ID",
            "split_annotation": {"target_note": 3},
        }
    )
    with (tmp_path / "contract-checks.json").open("x") as handle:
        json.dump(
            {
                "actual_after": after,
                "alternate_patch_grade": positive,
                "synthetic_protected_grade": negative,
                "baseline": answer,
                "public_observation": public,
                "actual_child_executions": 1,
                "neuronal_calls": 0,
            },
            handle,
            indent=2,
        )
    assert receipt["returncode"] == 0 and receipt["stdout_hex"] == b"blue\n".hex()
    assert positive["task_success"] and not negative["task_success"]
    assert after != grade["expected_after_files"]
    assert answer["answer_hex"] == b"Q7".hex()
    assert answer["attribution"]["document_sha256"] == hashlib.sha256(document).hexdigest()
    assert answer["attribution"]["literal_span"] == [18, 20]
    assert set(public) == {"kind", "request_hex", "files"}
    assert len(manifest["episodes"]) == 56 and len(manifest["documents"]) == 16
    assert sum(len(bytes.fromhex(row["text_hex"])) + 1 for row in manifest["documents"]) == 453
    assert sum(row["teacher"] is not None for row in manifest["episodes"]) == 24
    assert sum(row["attempt_limit"] for row in manifest["episodes"]) == 72
    assert sum(len(bytes.fromhex(row["request_hex"])) > 31 for row in manifest["episodes"]) == 4
    for episode in manifest["episodes"]:
        assert set(observation(episode)) == {"kind", "request_hex", "files"}
        assert all(len(bytes.fromhex(file["content_hex"])) <= 128 for file in episode["files"])
        assert episode["teacher"] is None or episode["phase"] == "acquisition"


# Prospective ordinary QA append to tests/test_byte_sequence_decoder.py.
# UNEXECUTED. Two nodes, two new readers total (one inside one combined model),
# two fixed-cell modules, one action circuit, two ingests (3raw+2EOS), no prior
# or file-checkpoint loads. Three free calls / five decisions; four teacher
# invocations / seven actual offers, including the second offer rejected by a
# declared late finite-check fault. Three successful teachers / five offers.
# Four extra fixed-helper calls over two authored rows each; sixteen helper
# calls including reader activity / thirty-three candidate populations total.
# One actual qualified child, one same-pending action correction. No task fit,
# public stream, acquired-skill threshold or repeated model experiment.


def test_request_conditioned_cells_and_weighted_local_update(tmp_path):
    import copy
    import math
    from unittest.mock import patch

    import torch

    from humanbrain.request_conditioned_memory_learner import RequestConditionedMemoryLearner

    def rails(tokens, width):
        result = torch.zeros(width, 18, dtype=torch.bool)
        for row, token in enumerate(tokens):
            for bit in range(9):
                result[row, 2 * bit + ((token >> bit) & 1)] = True
        return result

    reader = RequestConditionedMemoryLearner(seed=733)
    cells = reader.context
    contexts = torch.zeros(2, 64, 18, dtype=torch.bool)
    contexts[0, 32:] = rails([97, *([256] * 31)], 32)
    contexts[1, 32:] = rails([256] * 32, 32)
    outputs = []
    fixtures = ((b"a" * 31, b"a" * 16), (b"z" * 31, b"a" * 16),
                (b"a" * 31, b"z" * 16), (b"", b""))
    for query, prefix in fixtures:
        q, p = rails([*query, 256], 32), rails([257, *prefix], 17)
        work = {}
        actual = cells(q, p, contexts, contexts[:, 32], len(prefix), work=work)
        expected = []
        for candidate in range(2):
            position = [int(i == len(prefix)) for i in range(17)]
            raw = [*q.flatten().tolist(), *p.flatten().tolist(),
                   *contexts[candidate].flatten().tolist(), *position]
            thresholds = [
                int(sum(int(raw[i]) * sign for i, sign in zip(indices, signs, strict=True)) >= 0)
                for indices, signs in zip(cells.indices.tolist(), cells.signs.tolist(), strict=True)
            ]
            feature = [*thresholds, *contexts[candidate, 32].tolist(), *position, 1]
            scale = math.sqrt(32) / math.sqrt(sum(float(v) ** 2 for v in feature))
            expected.append([float(v) * scale for v in feature])
        torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64),
                                   rtol=2e-15, atol=2e-15)
        assert work["output_context_threshold_evaluations"] == 512
        assert work["output_context_signed_input_applications"] == 2 * 256 * 32
        outputs.append(actual)
    # These authored broad contrasts exercise the fixed mixed cells; collisions
    # for arbitrary unseen inputs remain permitted by the architecture.
    assert not torch.equal(outputs[0], outputs[1])
    assert not torch.equal(outputs[0], outputs[2])
    fixed_before = copy.deepcopy(cells.state_dict())
    changed = copy.deepcopy(fixed_before)
    changed["signs"][0, 0] *= -1
    with pytest.raises(ValueError, match="wiring identity"):
        cells.load_state_dict(changed)
    _assert_nested_equal(cells.state_dict(), fixed_before)

    state = reader.ingest([b"a"])
    before, tape_before = copy.deepcopy(reader.state_dict()), state.snapshot()
    captured = {}
    original_address, original_conditional = reader._address, reader._conditional

    def address(*args, **kwargs):
        result = original_address(*args, **kwargs)
        captured["xq"] = args[2].clone()
        captured["xp"] = result[1].clone()
        captured["prior"] = result[3].clone()
        return result

    def conditional(*args, **kwargs):
        result = original_conditional(*args, **kwargs)
        captured["x"], captured["logits"] = result[0].clone(), result[1].clone()
        return result

    with patch.object(reader, "_address", side_effect=address), patch.object(
        reader, "_conditional", side_effect=conditional
    ):
        feedback = reader.teach(b"a", b"", state, learning_rate=0.01)
    assert feedback.teacher_count == 1 and feedback.steps[0].prefix == b""
    assert feedback.steps[0].observed_token == 256
    x, z = captured["x"].tolist(), captured["logits"].tolist()
    prior_log = captured["prior"].tolist()
    prior = [math.exp(v) for v in prior_log]
    likelihood = []
    for row in z:
        peak = max(row)
        likelihood.append(math.exp(row[256] - peak) / sum(math.exp(v - peak) for v in row))
    normalizer = sum(p * c for p, c in zip(prior, likelihood, strict=True))
    responsibility = [p * c / normalizer for p, c in zip(prior, likelihood, strict=True)]
    energy = [sum(v * v for v in row) for row in x]
    target = [math.log(256) if c == 256 else 0.0 for c in range(257)]
    delta = [[sum(responsibility[s] * (target[c] - z[s][c]) * x[s][j] / energy[s]
                  for s in range(2)) for j in range(292)] for c in range(257)]
    expected_u = before["output_weight"] + torch.tensor(delta, dtype=torch.float64)
    torch.testing.assert_close(reader.output_weight, expected_u, rtol=2e-13, atol=2e-13)
    for name, field in (("address_weight", "xq"), ("prefix_weight", "xp")):
        feature = captured[field].tolist()
        change = [[sum((prior[s] - responsibility[s]) * feature[s][i][j]
                       for s in range(2)) for j in range(32)] for i in range(32)]
        expected = before[name] - 0.01 * torch.tensor(change, dtype=torch.float32)
        torch.testing.assert_close(reader.state_dict()[name], expected, rtol=0, atol=2e-7)
    # The isolated single-feature rule is an algebraic oracle, not another
    # teacher call or a claim that the weighted two-address update clamps both.
    isolated = [[(target[c] - z[0][c]) * v / energy[0] for v in x[0]] for c in range(257)]
    corrected = [z[0][c] + sum(a * b for a, b in zip(isolated[c], x[0], strict=True))
                 for c in range(257)]
    assert corrected == pytest.approx(target, rel=2e-13, abs=2e-13)
    for name in ("codec.weight", "codec.bias", "input_codes", "context.indices", "context.signs"):
        assert torch.equal(reader.state_dict()[name], before[name])
    assert feedback.output_updated and not feedback.legacy_codec_updated
    _assert_nested_equal(state.snapshot(), tape_before)
    _coincidence_qa_save(tmp_path / "conditioned-cells-local.json", {
        "status": "passed", "extra_helper_calls": 4, "helper_candidate_populations": 8,
        "reader_constructions": 1, "ingests": 1, "teacher_calls": 1, "teacher_offers": 1,
        "actual_children": 0, "checkpoint_loads": 0,
        "responsibility": responsibility, "work": feedback.work,
        "claim": "fixed currents and direct local arithmetic; no acquired-skill threshold",
    })


def test_request_conditioned_prefix_atomicity_and_actual_action(tmp_path):
    import copy
    import json
    import time
    from pathlib import Path
    from unittest.mock import patch

    import torch

    from experiments.r4_integrated.persistent_language_action_execution_v1 import (
        execute_proposal,
        qualified_launch,
        workspace_from_records,
    )
    from humanbrain import request_conditioned_memory_learner as implementation
    from humanbrain.neuronal_splice_action import SpliceProposal
    from humanbrain.request_conditioned_splice_learner import RequestConditionedSpliceLearner

    model = RequestConditionedSpliceLearner(seed=733)
    reader = model.reader
    state = model.ingest([b"ab"])
    with torch.no_grad():
        reader.address_weight.zero_()
        reader.prefix_weight.zero_()
        reader.prefix_weight[0, 1] = 1
        reader.output_weight.zero_()
        model.actions.operation_weight.zero_()
        model.actions.operation_bias.copy_(torch.tensor([1.0, 0.0]))
    before, tape_before = copy.deepcopy(model.state_dict()), state.snapshot()
    free = reader.predict_with_receipt(b"", state, max_bytes=16)
    assert free.generation.data == b"a" and free.generation.stop_reason == "eos"
    assert [s.selected_index for s in free.steps] == [0, 2]
    assert [s.prefix for s in free.steps] == [b"", b"a"]
    assert [s.observed_token for s in free.steps] == [97, 256]
    assert all(s.competition is not None and s.posterior_log_belief is None for s in free.steps)
    limited = reader.predict_with_receipt(b"", state, max_bytes=0)
    assert limited.generation.data == b"" and limited.generation.stop_reason == "limit"
    assert len(limited.steps) == 1 and limited.steps[0].observed_token == 97
    disabled = reader.teach(b"", b"b", state, learning_rate=0.01, learning_enabled=False)
    assert [s.prefix for s in disabled.steps] == [b"", b"b"]
    assert all(s.selected_index is None and s.logits.shape == (3, 257) for s in disabled.steps)
    assert all(s.posterior_log_belief.shape == (3,) for s in disabled.steps)
    assert not disabled.output_updated and not disabled.address_updated
    assert disabled.work["output_staged_update_steps"] == 0
    _assert_nested_equal(model.state_dict(), before)
    _assert_nested_equal(state.snapshot(), tape_before)

    finite, offers = implementation._finite, []

    def reject_late(value, name):
        if name == "conditioned output parameter proposal":
            offers.append(name)
            if len(offers) == 2:
                raise ValueError("authored late finite rejection")
        return finite(value, name)

    with patch.object(implementation, "_finite", side_effect=reject_late), pytest.raises(
        ValueError, match="authored late finite rejection"
    ):
        reader.teach(b"", b"b", state, learning_rate=0.01)
    assert len(offers) == 2
    _assert_nested_equal(model.state_dict(), before)
    _assert_nested_equal(state.snapshot(), tape_before)
    tampered = copy.deepcopy(before)
    tampered["reader.context.indices"][0, 0] ^= 1
    with pytest.raises(ValueError, match="fixed reader wiring"):
        model.load_state_dict(tampered)
    _assert_nested_equal(model.state_dict(), before)

    workspace = workspace_from_records([{"path": "main.py", "content_hex": b"print('a')\n".hex()}])
    prediction = model.propose(b"", workspace, state)
    assert prediction.proposal.operation == "STOP"
    assert model._pending[0] is prediction
    _assert_nested_equal(model.state_dict(), before)
    root = Path(__file__).resolve().parents[1]
    launch = qualified_launch(root)
    _, execution = execute_proposal(
        workspace, prediction.proposal, tmp_path / "actual-action", launch=launch,
        deadline=time.perf_counter() + 10,
    )
    completed = time.perf_counter()
    raw = tmp_path / "actual-action.json"
    _coincidence_qa_save(raw, execution)
    assert execution["returncode"] == 0 and execution["stdout_hex"] == b"a\n".hex()
    assert execution["capture_complete"] and not execution["timeout"]
    envelope = {
        "attempt_id": prediction.attempt_id, "proposal_sha256": prediction.proposal_sha256,
        "workspace_sha256": workspace.sha256, "execution": execution,
    }
    teacher = SpliceProposal("SPLICE", "main.py", 7, 8, b"b", "eos", workspace.sha256)
    actual_teach, calls = reader.teach, []

    def observed_teach(query, target, tape, *args, **kwargs):
        assert json.loads(raw.read_text()) == execution
        assert model._pending[0] is prediction and tape is state
        assert query == b"" and target == b"b" and time.perf_counter() >= completed
        _assert_nested_equal(model.state_dict(), before)
        calls.append(1)
        return actual_teach(query, target, tape, *args, **kwargs)

    with patch.object(reader, "teach", side_effect=observed_teach):
        feedback = model.teach(prediction, workspace, state, teacher, envelope)
    assert calls == [1] and feedback.reader_feedback.teacher_count == 2
    assert model._pending is None and feedback.reader_feedback.output_updated
    assert not feedback.reader_feedback.legacy_codec_updated
    for name in ("reader.codec.weight", "reader.codec.bias"):
        assert torch.equal(model.state_dict()[name], before[name])
    _assert_nested_equal(state.snapshot(), tape_before)
    _coincidence_qa_save(tmp_path / "conditioned-prefix-action.json", {
        "status": "passed", "combined_models": 1, "readers": 1, "ingests": 1,
        "free_calls_including_action": 3, "free_decisions": 5,
        "teacher_invocations": 3, "actual_offers_including_rejected": 6,
        "successful_teacher_calls": 2, "successful_offers": 4,
        "late_failed_offers": len(offers), "actual_children": 1, "checkpoint_loads": 0,
        "free_selected_addresses": [s.selected_index for s in free.steps],
        "disabled_work": disabled.work, "action_feedback_work": feedback.work,
        "claim": "causal prefixes, atomic local publication and real execution custody only",
    })


def test_incremental_prefix_full_capacity_exact_activity(tmp_path):
    """One cache, no learner: all17 positions match the frozen full circuit."""
    import json

    import pytest
    import torch

    from humanbrain.incremental_prefix_coincidence import IncrementalPrefixCoincidence
    from humanbrain.neuronal_byte_coincidence import coincidence_features, encode_symbol_rails

    n = 512
    symbols = torch.arange(n * 32, dtype=torch.int64).reshape(n, 32) % 256
    keys = encode_symbol_rails(symbols)
    lengths = torch.arange(n, dtype=torch.int64) % 33
    keys &= (torch.arange(32)[None, :] < lengths[:, None])[:, :, None]
    reference_keys = keys.clone()
    work = {}
    cache = IncrementalPrefixCoincidence(keys, work=work)
    keys.zero_()  # Caller changes cannot alter this call's owned key population.
    raw = bytes([0, 255, 0, 17, 17, 128, 1, 254, 4, 5, 6, 7, 8, 9, 10, 11])
    b = torch.arange(1024, dtype=torch.float32).reshape(32, 32) / 1024
    c = b.flip(0) - 0.5
    held_query = torch.ones(n, 32, 32, dtype=torch.float32) / 32
    full_calls = 0
    for position in range(17):
        if position:
            cache.advance(raw[position - 1])
        encoded = encode_symbol_rails(torch.tensor([257, *raw[:position]], dtype=torch.int64))
        lags = torch.zeros(32, 18, dtype=torch.bool)
        lags[: position + 1].copy_(encoded.flip(0))
        expected = torch.empty(n, 32, 32, dtype=torch.float32)
        for first in range(0, n, 128):
            expected[first:first + 128].copy_(
                coincidence_features(lags, reference_keys[first:first + 128])
            )
            full_calls += 1
        actual = cache.features()
        assert actual.is_contiguous()
        assert torch.equal(actual, expected)
        actual_scores = (held_query * b).sum((1, 2)) + (actual * c).sum((1, 2))
        expected_scores = (held_query * b).sum((1, 2)) + (expected * c).sum((1, 2))
        assert torch.equal(actual_scores, expected_scores)
    before_work = dict(work)
    for invalid in [True, 256, 1]:
        with pytest.raises(ValueError):
            cache.advance(invalid)
        assert work == before_work
    assert torch.equal(cache.features(), actual)
    pairs = n * 1024 + 16 * n * 32
    assert work['prefix_cache_constructions'] == 1
    assert work['prefix_cache_advances'] == 16
    assert work['prefix_cache_feature_calls'] == 18
    assert work['prefix_cache_lag_pairs'] == pairs
    assert work['prefix_cache_threshold_evaluations'] == 19 * pairs
    assert work['prefix_cache_input_applications'] == 54 * pairs
    assert work['prefix_cache_reused_activity_values'] == 16 * n * 31 * 32
    assert work['prefix_cache_shift_copy_values'] == 32 * n * 31 * 32
    assert work['prefix_cache_feature_scalings'] == 18 * n * 1024
    assert work['prefix_cache_held_key_bytes'] == n * 32 * 18
    assert work['prefix_cache_held_match_bytes'] == n * 32 * 32
    assert work['prefix_cache_held_count_bytes'] == n * 4
    (tmp_path / 'incremental-prefix-parity.json').write_text(json.dumps({
        'status': 'passed', 'positions': 17, 'candidates': n,
        'cache_constructions': 1, 'cache_feature_calls': 18,
        'valid_advances': 16, 'rejected_advances': 3,
        'frozen_full_kernel_calls': full_calls,
        'model_constructions': 0, 'teachers': 0, 'checkpoint_loads': 0,
        'actions': 0, 'work': work,
    }, sort_keys=True) + '\n', encoding='utf-8')


def test_active_path_combined_execution_stop_and_ack(tmp_path):
    import copy
    import json
    import time
    from pathlib import Path
    from unittest.mock import patch

    import pytest
    import torch

    from experiments.r4_integrated.persistent_language_action_execution_v1 import (
        execute_proposal,
        qualified_launch,
        workspace_from_records,
    )
    from humanbrain.active_path_splice_learner import ActivePathSpliceLearner
    from humanbrain.neuronal_splice_action import SpliceProposal

    model = ActivePathSpliceLearner(seed=733)
    reader = model.reader
    state = model.ingest([b'ab'])
    with torch.no_grad():
        reader.address_weight.zero_()
        reader.prefix_weight.zero_()
        reader.prefix_weight[0, 1] = 1
        reader.output_weight.zero_()
        model.actions.operation_weight.zero_()
        model.actions.operation_bias.copy_(torch.tensor([1.0, 0.0]))
    original = copy.deepcopy(model.state_dict())
    original_tape = state.snapshot()
    workspace = workspace_from_records([
        {'path': 'main.py', 'content_hex': b"print('a')\n".hex()},
    ])
    launch = qualified_launch(Path(__file__).resolve().parents[1])
    receipts, feedback_work = [], []
    for phase in ['splice', 'stop', 'ack']:
        before = copy.deepcopy(model.state_dict())
        prediction = model.propose(b'', workspace, state)
        assert prediction.proposal.operation == 'STOP'
        assert prediction.payload.generation.data == b'a'
        assert prediction.payload.generation.stop_reason == 'eos'
        assert len(prediction.payload.steps) == 2
        assert model._pending[0] is prediction and reader.pending_prediction is prediction.payload
        teacher = SpliceProposal('SPLICE', 'main.py', 7, 8, b'b', 'eos', workspace.sha256)
        if phase == 'splice':
            with pytest.raises(ValueError):
                model.teach(prediction, workspace, state, teacher, {})
            with pytest.raises(ValueError):
                model.snapshot(state)
            with pytest.raises(ValueError):
                model.load_state_dict(original)
            _assert_nested_equal(model.state_dict(), before)
        _, execution = execute_proposal(
            workspace, prediction.proposal, tmp_path / phase,
            launch=launch, deadline=time.perf_counter() + 10,
        )
        completed = time.perf_counter()
        raw = tmp_path / (phase + '-execution.json')
        _coincidence_qa_save(raw, execution)
        receipts.append(execution)
        assert execution['returncode'] == 0 and execution['stdout_hex'] == b'a\n'.hex()
        assert execution['capture_complete'] and not execution['timeout']
        envelope = {
            'attempt_id': prediction.attempt_id,
            'proposal_sha256': prediction.proposal_sha256,
            'workspace_sha256': workspace.sha256, 'execution': execution,
        }
        if phase == 'splice':
            retained = prediction.payload.selected_features[0]
            saved = retained.clone()
            retained[0] += 1
            with patch.object(model.actions, 'prepare_update', side_effect=AssertionError(
                'stale nested reader must fail before action proposals'
            )), pytest.raises(ValueError, match='trace'):
                model.teach(prediction, workspace, state, teacher, envelope)
            retained.copy_(saved)
            _assert_nested_equal(model.state_dict(), before)
            actual_teach = reader.teach
            calls = []

            def observed_teach(
                query, target, tape, *args,
                _bound=(raw, execution, completed, prediction, before, calls, actual_teach),
                **kwargs,
            ):
                raw, execution, completed, prediction, before, calls, actual_teach = _bound
                assert json.loads(raw.read_text()) == execution
                assert time.perf_counter() >= completed
                assert kwargs['prediction'] is prediction.payload and tape is state
                assert query == b'' and target == b'b'
                _assert_nested_equal(model.state_dict(), before)
                calls.append(1)
                return actual_teach(query, target, tape, *args, **kwargs)

            with patch.object(reader, 'teach', side_effect=observed_teach):
                feedback = model.teach(prediction, workspace, state, teacher, envelope)
            assert calls == [1]
            assert feedback.reader_feedback.teacher_count == 2
            assert feedback.reader_feedback.active_output_offers == 1
            assert feedback.reader_feedback.output_updated
            assert feedback.work['reader']['active_trace_validations'] == 3
            feedback_work.append(feedback.work)
        elif phase == 'stop':
            teacher = SpliceProposal('STOP', 'main.py', 0, 0, b'', 'eos', workspace.sha256)
            with patch.object(
                reader, 'teach', side_effect=AssertionError('STOP cannot teach payload')
            ):
                feedback = model.teach(prediction, workspace, state, teacher, envelope)
            assert feedback.reader_feedback is None
            assert set(feedback.action_losses) == {'operation'}
            assert feedback.work['reader']['teacher_offers'] == 0
            assert feedback.work['reader']['active_trace_validations'] == 2
            assert feedback.work['reader']['active_response_acknowledgements'] == 1
            for key, value in before.items():
                if key.startswith('reader.') and isinstance(value, torch.Tensor):
                    assert torch.equal(model.state_dict()[key], value)
            feedback_work.append(feedback.work)
        else:
            with patch.object(
                reader, 'teach', side_effect=AssertionError('ack cannot teach payload')
            ):
                work = model.finish_without_teaching(prediction, workspace, state, envelope)
            assert work['reader']['teacher_offers'] == 0
            assert work['reader']['active_trace_validations'] == 2
            assert work['reader']['active_response_acknowledgements'] == 1
            _assert_nested_equal(model.state_dict(), before)
            feedback_work.append(work)
        assert model._pending is None and reader.pending_prediction is None
        _assert_nested_equal(state.snapshot(), original_tape)
        if phase == 'splice':
            # Same instance; one explicit in-memory fixture reload keeps the
            # following two lifecycle checks independent of learned behavior.
            model.load_state_dict(original)

    reading = reader.predict_with_receipt(b'', state, max_bytes=0)
    assert len(reading.steps) == 1 and model._pending is None
    for operation in [lambda: model.snapshot(state), lambda: model.load_state_dict(original)]:
        with pytest.raises(ValueError):
            operation()
    reader.acknowledge(reading, state)
    assert reader.pending_prediction is None
    _coincidence_qa_save(tmp_path / 'active-combined-lifecycle.json', {
        'status': 'passed', 'combined_models': 1, 'readers': 1, 'action_circuits': 1,
        'ingests': 1, 'raw_ingest_bytes': 2, 'eos_ingest_symbols': 1,
        'action_proposals': 3, 'standalone_reading_calls': 1, 'free_decisions': 7,
        'successful_combined_teach_calls': 2, 'rejected_combined_teach_calls': 2,
        'reader_teacher_calls': 1, 'reader_teacher_offers': 2,
        'action_teacher_labels': 4, 'active_output_clamps': 1,
        'actual_children': len(receipts), 'checkpoint_loads': 0,
        'in_memory_fixture_reloads': 1, 'feedback_work': feedback_work,
    })


# Source-only append fragment. One reader/one ingest (2 raw bytes + EOS),
# four actual free calls/eight expected decisions, four entered teacher loops/
# nine offered symbols including two before a final authored failure. One extra
# over-cap teach call rejects before forward/offers. Three same-instance restores.
# Three offered active clamps, one enabled publication; zero children/checkpoints.


def test_active_path_reader_clamp_custody_and_atomicity(tmp_path):
    import copy
    import hashlib
    import json
    import math
    from dataclasses import replace
    from pathlib import Path
    from unittest.mock import patch

    import torch

    from humanbrain import active_path_memory_learner as implementation
    from humanbrain.neuronal_splice_learner import _digest

    reader = implementation.ActivePathMemoryLearner(seed=733)
    state = reader.ingest([b"ab"])
    with torch.no_grad():
        reader.address_weight.zero_()
        reader.prefix_weight.zero_()
        reader.prefix_weight[0, 1] = 1
        reader.output_weight.zero_()
    snapshot = reader.snapshot(state)
    original_model = copy.deepcopy(reader.state_dict())
    original_tape = state.snapshot()
    raw_path = tmp_path / "active-path-reader-raw.jsonl"

    def save_raw(record):
        with raw_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")

    save_raw({
        "phase": "initial", "source_sha256": hashlib.sha256(
            Path(implementation.__file__).read_bytes()).hexdigest(),
        "model_sha256": _digest(original_model), "memory_sha256": _digest(original_tape),
        "reader_constructions": 1, "ingests": 1, "occupied_slots": state.used,
        "raw_bytes": 2, "eos": 1, "checkpoint_loads": 0,
    })
    predictions = []

    def predict():
        prediction = reader.predict_with_receipt(b"", state, max_bytes=2)
        predictions.append(prediction)
        save_raw({
            "phase": "free", "attempt_id": prediction.attempt_id,
            "trace_sha256": prediction.trace_sha256,
            "tokens": [s.observed_token for s in prediction.steps],
            "prefix_hex": [s.prefix.hex() for s in prediction.steps],
            "selected": [s.selected_index for s in prediction.steps],
            "answer_hex": prediction.generation.data.hex(),
            "stop_reason": prediction.generation.stop_reason, "work": prediction.work,
        })
        assert prediction.generation.data == b"a"
        assert prediction.generation.stop_reason == "eos"
        assert [s.selected_index for s in prediction.steps] == [0, 2]
        assert [s.prefix for s in prediction.steps] == [b"", b"a"]
        assert [s.observed_token for s in prediction.steps] == [97, 256]
        assert reader.pending_prediction is prediction
        assert prediction.work["prefix_cache_feature_calls"] == 2
        assert prediction.work["prefix_cache_advances"] == 1
        assert prediction.work["prefix_cache_lag_pairs"] == 3 * (1024 + 32)
        assert prediction.work["active_retained_feature_values"] == 2 * 292
        assert prediction.work["active_retained_feature_bytes"] == 2 * 292 * 8
        assert all(x.dtype == torch.float64 and tuple(x.shape) == (292,)
                   and not x.requires_grad for x in prediction.selected_features)
        return prediction

    actual = predict()
    rejections = []

    def reject(label, call):
        before_model, before_tape = _digest(reader.state_dict()), _digest(vars(state))
        error = None
        try:
            call()
        except ValueError as caught:
            error = str(caught)
        record = {
            "phase": "rejected", "label": label, "error": error,
            "model_before": before_model, "model_after": _digest(reader.state_dict()),
            "memory_before": before_tape, "memory_after": _digest(vars(state)),
            "same_pending": reader.pending_prediction is actual,
        }
        save_raw(record)
        rejections.append(label)
        assert error is not None
        assert record["model_before"] == record["model_after"]
        assert record["memory_before"] == record["memory_after"]
        assert record["same_pending"]

    with patch.object(reader, "_conditional", side_effect=AssertionError("unexpected forward")):
        reject("substituted_prediction", lambda: reader.validate_pending(replace(actual), state))
        reject("different_request", lambda: reader.validate_pending(actual, state, query=b"x"))
        feature = actual.selected_features[1]
        held_feature = feature.clone()
        feature[0] += 1
        try:
            reject("changed_selected_feature", lambda: reader.validate_pending(actual, state))
        finally:
            feature.copy_(held_feature)
        logits = actual.steps[1].logits
        held_logits = logits.clone()
        logits[0, 0] += 1
        try:
            reject("changed_actual_current", lambda: reader.validate_pending(actual, state))
        finally:
            logits.copy_(held_logits)
        held_steps = actual.steps
        object.__setattr__(actual, "steps", held_steps[:-1])
        try:
            reject("missing_step", lambda: reader.validate_pending(actual, state))
        finally:
            object.__setattr__(actual, "steps", held_steps)
        object.__setattr__(actual, "steps", (*held_steps, held_steps[-1]))
        try:
            reject("extra_step", lambda: reader.validate_pending(actual, state))
        finally:
            object.__setattr__(actual, "steps", held_steps)
        held_key = state.keys[2].clone()
        state.keys[2].copy_(state.keys[1])
        try:
            reject("changed_tape", lambda: reader.validate_pending(actual, state))
        finally:
            state.keys[2].copy_(held_key)
        reject("snapshot_pending", lambda: reader.snapshot(state))
        reject("ingest_pending", lambda: reader.ingest([b"x"], state))
        reject("load_pending", lambda: reader.load_state_dict(snapshot["model"]))
        reject("over_cap_teacher", lambda: reader.teach(
            b"", b"abc", state, .01, prediction=actual))
    _assert_nested_equal(reader.state_dict(), original_model)
    _assert_nested_equal(state.snapshot(), original_tape)

    conditional = reader._conditional
    finite = implementation._finite
    entered_teachers = []

    def teach(prediction, target, *, enabled=True, late=False):
        held_u = reader.output_weight.detach().clone()
        captures = []

        def observed(*args, **kwargs):
            result = conditional(*args, **kwargs)
            captures.append({
                "position": args[4], "held_u": torch.equal(args[5], held_u),
                "logit_shape": list(result[1].shape), "work": args[6],
            })
            return result

        def checked(value, name):
            if late and name == "final active output parameter proposal":
                raise ValueError("authored final active proposal rejection")
            return finite(value, name)

        feedback, error = None, None
        with patch.object(reader, "_conditional", side_effect=observed), patch.object(
            implementation, "_finite", side_effect=checked
        ):
            try:
                feedback = reader.teach(
                    b"", target, state, .01, prediction=prediction, learning_enabled=enabled)
            except ValueError as caught:
                error = str(caught)
        work = feedback.work if feedback is not None else dict(captures[-1]["work"])
        record = {
            "phase": "teacher", "target_hex": target.hex(), "enabled": enabled,
            "late_failure": late, "error": error,
            "conditional_positions": [c["position"] for c in captures],
            "held_u_all_steps": all(c["held_u"] for c in captures),
            "conditional_shapes": [c["logit_shape"] for c in captures],
            "model_after_sha256": _digest(reader.state_dict()), "work": work,
            "pending_retained": reader.pending_prediction is prediction,
        }
        if feedback is not None:
            record.update(correction_position=feedback.correction_position,
                          correction_target=feedback.correction_target,
                          active_output_offers=feedback.active_output_offers,
                          query_address_updated=feedback.query_address_updated,
                          prefix_address_updated=feedback.prefix_address_updated,
                          output_updated=feedback.output_updated)
        save_raw(record)
        entered_teachers.append(record)
        assert record["held_u_all_steps"]
        assert record["conditional_positions"] == list(range(len(target) + 1))
        assert all(shape == [3, 257] for shape in record["conditional_shapes"])
        assert work["teacher_offers"] == len(target) + 1
        assert work["output_local_synapse_products"] == 0
        assert work["output_local_slot_reductions"] == 0
        assert work["output_address_local_products"] == (len(target) + 1) * 3 * 2048
        assert work["prefix_cache_feature_calls"] == len(target) + 1
        assert work["prefix_cache_advances"] == len(target)
        assert work["prefix_cache_lag_pairs"] == 3 * (1024 + len(target) * 32)
        if late:
            assert error == "authored final active proposal rejection"
            assert feedback is None and reader.pending_prediction is prediction
        else:
            assert error is None and reader.pending_prediction is None
        return feedback, work

    # Independent scalar current-error times actual own-feature oracle. No model
    # forward is used to form the expected update or same-feature clamp result.
    x = actual.selected_features[1].tolist()
    z = actual.steps[1].logits[0].tolist()
    energy = sum(v * v for v in x)
    target_currents = [math.log(256) if c == 98 else 0.0 for c in range(257)]
    delta = [[(target_currents[c] - z[c]) * v / energy for v in x] for c in range(257)]
    expected_u = original_model["output_weight"] + torch.tensor(delta, dtype=torch.float64)
    feedback, work = teach(actual, b"ab")
    save_raw({"phase": "clamp_oracle", "feature": x, "actual_currents": z,
              "expected_u_sha256": _digest(expected_u),
              "actual_u_sha256": _digest(reader.output_weight)})
    torch.testing.assert_close(reader.output_weight, expected_u, rtol=2e-13, atol=2e-13)
    actual_delta = (reader.output_weight - original_model["output_weight"]).tolist()
    clamped = [z[c] + sum(a * b for a, b in zip(actual_delta[c], x, strict=True))
               for c in range(257)]
    assert clamped == pytest.approx(target_currents, rel=2e-13, abs=2e-13)
    assert feedback.correction_position == 1 and feedback.correction_target == 98
    assert feedback.active_output_offers == 1 and feedback.output_updated
    assert work["active_output_event_comparisons"] == 2
    assert work["active_output_clamp_offers"] == 1
    assert work["active_output_local_products"] == 257 * 292
    assert work["active_output_local_divisions"] == 257 * 292
    assert work["active_output_energy_products"] == 292
    assert work["active_output_energy_reductions"] == 291
    assert work["output_staged_update_steps"] == 3
    assert work["output_enabled_groups"] == 1
    for key in ("codec.weight", "codec.bias", "input_codes", "context.indices", "context.signs"):
        assert torch.equal(reader.state_dict()[key], original_model[key])
    _assert_nested_equal(state.snapshot(), original_tape)

    state = reader.restore_snapshot(snapshot)
    disabled_prediction = predict()
    disabled, disabled_work = teach(disabled_prediction, b"b", enabled=False)
    assert disabled.correction_position == 0 and disabled.active_output_offers == 1
    assert not disabled.output_updated and not disabled.address_updated
    assert disabled_work["output_staged_update_steps"] == 0
    assert disabled_work["output_enabled_groups"] == 0
    _assert_nested_equal(reader.state_dict(), original_model)
    _assert_nested_equal(state.snapshot(), original_tape)

    state = reader.restore_snapshot(snapshot)
    failed_prediction = predict()
    _, failed_work = teach(failed_prediction, b"b", late=True)
    assert failed_work["active_output_clamp_offers"] == 1
    assert failed_work["output_staged_update_steps"] == 2
    _assert_nested_equal(reader.state_dict(), original_model)
    _assert_nested_equal(state.snapshot(), original_tape)
    with patch.object(reader, "_conditional", side_effect=AssertionError("ack forward")):
        ack = reader.acknowledge(failed_prediction, state)
    save_raw({"phase": "acknowledged_failed_response", "work": ack})
    assert reader.pending_prediction is None
    assert ack["active_response_acknowledgements"] == 1
    assert ack["teacher_offers"] == 0 and ack["output_decisions"] == 0
    assert ack["active_output_clamp_offers"] == 0

    state = reader.restore_snapshot(snapshot)
    exact_prediction = predict()
    exact, exact_work = teach(exact_prediction, b"a")
    assert exact.correction_position is None and exact.correction_target is None
    assert exact.active_output_offers == 0 and not exact.output_updated
    assert exact_work["active_output_clamp_offers"] == 0
    assert exact_work["active_output_local_products"] == 0
    assert exact_work["output_enabled_groups"] == 0
    assert torch.equal(reader.output_weight, original_model["output_weight"])
    assert not exact.legacy_codec_updated
    _assert_nested_equal(state.snapshot(), original_tape)
    assert len({p.attempt_id for p in predictions}) == 4
    assert reader.pending_prediction is None
    (tmp_path / "active-path-reader-passed.json").write_text(json.dumps({
        "status": "passed", "reader_constructions": 1, "ingests": 1,
        "raw_bytes": 2, "eos": 1, "free_calls": 4,
        "actual_free_decisions": sum(len(p.steps) for p in predictions),
        "teacher_loops_entered": 4, "teacher_offers": 9,
        "over_cap_teacher_rejected_before_forward": 1,
        "validation_only_rejections": len(rejections) - 1,
        "offered_active_clamps": 3, "published_active_clamps": 1,
        "same_instance_snapshot_restores": 3, "actual_children": 0, "checkpoint_loads": 0,
        "teacher_records": entered_teachers,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "claim": "actual-path arithmetic, custody and atomicity; no acquired skill",
    }, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def test_sequence_assembly_cells_local_staging_contract(tmp_path):
    """One small standalone bank: local arithmetic and custody, not task learning."""
    import hashlib
    import json

    from humanbrain import sequence_assembly_cells as module

    raw_path = tmp_path / "assembly-cells-raw.jsonl"

    def save(row):
        with raw_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def snapshot_equal(actual, expected):
        assert actual["_extra_state"] == expected["_extra_state"]
        for key in ("prototypes", "outgoing", "used"):
            assert torch.equal(actual[key], expected[key])

    bank = module.SequenceAssemblyCells(input_width=10)
    fallback = torch.linspace(-0.5, 0.5, 257, dtype=torch.float64)
    a = torch.zeros(10, dtype=torch.bool)
    b = a.clone()
    b[:2] = True
    c = a.clone()
    c[2] = True
    tie = a.clone()
    tie[0] = True
    work = {}
    initial = bank.snapshot()
    stage = bank.begin_stage(work)
    empty = bank.evaluate(a, fallback, work=work)
    save({"phase": "empty", "winner": empty.winner_index,
          "currents": empty.currents.tolist(), "work": dict(work)})
    assert empty.winner_index is None and not empty.exact
    assert torch.equal(empty.currents, fallback)

    for inputs, target in ((a, 65), (b, 66), (a, 67)):
        context = bank.prepare_context(inputs, fallback, stage=stage, work=work)
        before = context.initial_currents.tolist()
        held_others = stage.outgoing.clone()
        update = bank.prepare_teach(context, target, stage=stage, work=work)
        expected = [old + ((math.log(256) if i == target else 0.0) - old) * 1.0 / 1.0
                    for i, old in enumerate(before)]
        save({"phase": "local_offer", "target": target, "selected": update.selected_index,
              "recruited": update.recruited, "before": before,
              "actual": update.currents_after.tolist(), "expected": expected,
              "used": int(stage.used), "work": dict(work)})
        assert update.staged and not update.exhausted
        assert update.currents_after.tolist() == expected
        assert update.error.tolist() == [
            (math.log(256) if i == target else 0.0) - old for i, old in enumerate(before)]
        for index in range(512):
            if index != update.selected_index:
                assert torch.equal(stage.outgoing[index], held_others[index])
        snapshot_equal(bank.snapshot(), initial)
    assert int(stage.used) == 2 and stage.revision == 3
    assert torch.equal(stage.prototypes[0], a) and torch.equal(stage.prototypes[1], b)
    bank.publish(stage, work)
    published = bank.snapshot()
    assert int(bank.used) == 2
    tied = bank.evaluate(tie, fallback, work=work)
    oracle_counts = [sum(int(x == y) for x, y in zip(tie.tolist(), row, strict=True))
                     for row in bank.prototypes[:2].tolist()]
    save({"phase": "tie", "counts": tied.agreement_counts.tolist(),
          "oracle_counts": oracle_counts, "winner": tied.winner_index,
          "pair_currents": tied.competition.pair_currents.tolist(), "work": dict(work)})
    assert oracle_counts == [9, 9] and tied.agreement_counts.tolist() == oracle_counts
    assert tied.winner_index == 0 and not tied.exact
    assert tied.competition.comparator_activity.tolist() == [True]
    assert tied.competition.winner_activity.tolist() == [True, False]
    assert torch.equal(tied.currents, bank.outgoing[0])

    stage = bank.begin_stage(work)
    context = bank.prepare_context(c, fallback, stage=stage, work=work)
    frozen_stage = (stage.prototypes.clone(), stage.outgoing.clone(), stage.used.clone())
    disabled = bank.prepare_teach(context, 68, stage=stage, learning_enabled=False, work=work)
    save({"phase": "disabled", "staged": disabled.staged,
          "recruited": disabled.recruited, "work": dict(work)})
    assert disabled.recruited and not disabled.staged and stage.revision == 0
    for actual, expected in zip(
            (stage.prototypes, stage.outgoing, stage.used), frozen_stage, strict=True):
        assert torch.equal(actual, expected)

    original_finite = module._finite

    def fail_final(value, label):
        if label == "assembly output proposal":
            raise ValueError("authored late finite failure")
        return original_finite(value, label)

    with (patch.object(module, "_finite", fail_final),
          pytest.raises(ValueError, match="authored late finite failure")):
        bank.prepare_teach(context, 69, stage=stage, work=work)
    save({"phase": "late_failure", "used": int(stage.used),
          "revision": stage.revision, "work": dict(work)})
    for actual, expected in zip(
            (stage.prototypes, stage.outgoing, stage.used), frozen_stage, strict=True):
        assert torch.equal(actual, expected)
    snapshot_equal(bank.snapshot(), published)
    context.inputs[0] = ~context.inputs[0]
    with pytest.raises(ValueError, match="prepared assembly context"):
        bank.prepare_teach(context, 70, stage=stage, work=work)
    context.inputs[0] = ~context.inputs[0]
    stage.outgoing[0, 0] = torch.nan
    with pytest.raises(ValueError, match="assembly outgoing"):
        bank.publish(stage, work)
    snapshot_equal(bank.snapshot(), published)

    malformed = bank.snapshot()
    malformed["prototypes"][0, 0] = True
    malformed["outgoing"][0, 0] = torch.nan
    with pytest.raises(ValueError, match="assembly outgoing"):
        bank.load_state_dict(malformed)
    snapshot_equal(bank.snapshot(), published)
    save({"phase": "custody_rejections", "mutated_context": True,
          "nonfinite_stage": True, "nonfinite_load": True})

    # Authored full bank:512 distinct9-bit inputs, not512 teacher calls or a fit.
    full = bank.snapshot()
    full["used"].fill_(512)
    full["prototypes"].zero_()
    for index in range(512):
        for bit in range(9):
            full["prototypes"][index, bit] = bool((index >> bit) & 1)
    full["outgoing"].zero_()
    full["outgoing"][:, 65] = 1.0
    bank.load_state_dict(full)
    stage = bank.begin_stage(work)
    novel = a.clone()
    novel[9] = True
    exhausted_context = bank.prepare_context(novel, fallback, stage=stage, work=work)
    exhausted = bank.prepare_teach(exhausted_context, 71, stage=stage, work=work)
    exact_context = bank.prepare_context(a, fallback, stage=stage, work=work)
    exact_disabled = bank.prepare_teach(
        exact_context, 72, stage=stage, learning_enabled=False, work=work)
    save({"phase": "full", "exhausted": exhausted.exhausted,
          "novel_counts": exhausted_context.evaluation.agreement_counts.tolist(),
          "exact_selected": exact_disabled.selected_index, "work": dict(work)})
    assert exhausted.exhausted and not exhausted.staged and exhausted.selected_index is None
    assert exact_context.evaluation.exact and exact_disabled.selected_index == 0
    assert not exact_disabled.exhausted and not exact_disabled.staged
    assert stage.revision == 0
    snapshot_equal(bank.snapshot(), full)
    assert torch.equal(stage.prototypes, full["prototypes"])
    assert torch.equal(stage.outgoing, full["outgoing"])
    assert int(stage.used) == 512
    assert work["assembly_evaluations"] == 8
    assert work["assembly_candidates"] == 1031
    assert work["assembly_agreement_cells"] == 10310
    assert work["assembly_agreement_thresholds"] == 30930
    assert work["assembly_agreement_input_applications"] == 41240
    assert work["assembly_output_offers"] == 7
    assert work["assembly_staged_output_updates"] == 3
    assert work["assembly_staged_recruitments"] == 2
    assert work["assembly_exhausted_offers"] == 1
    assert work["assembly_publications"] == 1
    assert work["assembly_output_local_products"] == 6 * 257
    assert all(name in module.WORK_KEYS for name in work)
    assert not list(bank.parameters())
    (tmp_path / "assembly-cells-passed.json").write_text(json.dumps({
        "status": "passed", "standalone_banks": 1, "input_width": 10,
        "evaluations": 8, "teacher_calls": 8, "offers_entered": 7,
        "completed_offers": 6, "staged_updates": 3, "recruits": 2,
        "publications": 1, "late_failed_offers": 1, "disabled_offers": 2,
        "exhausted_skips": 1, "reader_models": 0, "actions": 0,
        "checkpoint_loads": 0, "authored_full_bank_loads": 1,
        "work": work, "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "claim": "local assembly mechanics; no acquired task skill",
    }, sort_keys=True) + "\n", encoding="utf-8")


def test_sequence_assembly_rolling_prefix_exact_boundaries(tmp_path):
    """One129-slot cache reaches128 outputs, comparing six declarative lag banks."""
    import hashlib
    import json

    from humanbrain.neuronal_byte_coincidence import coincidence_features, encode_symbol_rails
    from humanbrain.sequence_assembly_prefix import RollingPrefixCoincidence

    n = 129
    symbols = torch.arange(n * 32, dtype=torch.int64).reshape(n, 32) % 256
    keys = encode_symbol_rails(symbols)
    lengths = torch.arange(n, dtype=torch.int64) % 33
    keys &= (torch.arange(32)[None, :] < lengths[:, None])[:, :, None]
    reference_keys = keys.clone()
    work = {}
    cache = RollingPrefixCoincidence(keys, work=work)
    keys.zero_()
    raw = bytes((index * 37 + 11) % 256 for index in range(128))
    boundaries = {0, 16, 31, 32, 127, 128}
    records = []
    full_calls = 0
    raw_path = tmp_path / "assembly-prefix-raw.jsonl"
    weights = torch.arange(1024, dtype=torch.float32).reshape(32, 32) / 1024
    for position in range(129):
        if position:
            cache.advance(raw[position - 1])
        if position not in boundaries:
            continue
        prefix_symbols = [257, *raw[:position]][-32:]
        encoded = encode_symbol_rails(torch.tensor(prefix_symbols, dtype=torch.int64))
        lags = torch.zeros(32, 18, dtype=torch.bool)
        lags[:len(prefix_symbols)].copy_(encoded.flip(0))
        expected = torch.empty(n, 32, 32, dtype=torch.float32)
        for first in range(0, n, 128):
            expected[first:first + 128].copy_(
                coincidence_features(lags, reference_keys[first:first + 128]))
            full_calls += 1
        actual = cache.features()
        row = {"position": position, "occupied": cache.occupied_lags,
               "raw_count": cache.raw_count, "equal": torch.equal(actual, expected),
               "oldest_symbol": prefix_symbols[0], "newest_symbol": prefix_symbols[-1],
               "score_equal": torch.equal((actual * weights).sum((1, 2)),
                                           (expected * weights).sum((1, 2))),
               "actual_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
               "expected_sha256": hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
               "work": dict(work)}
        with raw_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True) + "\n")
        records.append(row)
        assert row["equal"] and row["score_equal"]
        assert cache.raw_count == position and cache.occupied_lags == min(position + 1, 32)
        assert (257 in prefix_symbols) == (position < 32)
    before = dict(work)
    for invalid in (True, 256, 0):
        with pytest.raises(ValueError):
            cache.advance(invalid)
        assert work == before and cache.raw_count == 128
    actual.zero_()
    final = cache.features()
    assert torch.equal(final, expected)
    pairs = n * 1024 + 128 * n * 32
    assert work["prefix_cache_constructions"] == 1
    assert work["prefix_cache_advances"] == 128
    assert work["prefix_cache_rolling_drops"] == 97
    assert work["prefix_cache_feature_calls"] == 7
    assert work["prefix_cache_lag_pairs"] == pairs
    assert work["prefix_cache_threshold_evaluations"] == 19 * pairs
    assert work["prefix_cache_input_applications"] == 54 * pairs
    assert work["prefix_cache_gate_blocks"] == 2 * 129
    assert work["prefix_cache_reused_activity_values"] == 128 * n * 31 * 32
    assert work["prefix_cache_shift_copy_values"] == 256 * n * 31 * 32
    assert work["prefix_cache_feature_scalings"] == 7 * n * 1024
    assert work["prefix_cache_held_key_bytes"] == n * 32 * 18
    assert work["prefix_cache_held_match_bytes"] == n * 32 * 32
    assert work["prefix_cache_held_count_bytes"] == n * 4
    assert full_calls == 12
    (tmp_path / "assembly-prefix-passed.json").write_text(json.dumps({
        "status": "passed", "candidates": n, "cache_constructions": 1,
        "valid_advances": 128, "rejected_advances": 3, "feature_calls": 7,
        "full_reference_calls": full_calls, "boundaries": sorted(boundaries),
        "records": records, "work": work, "reader_models": 0, "teachers": 0,
        "actions": 0, "checkpoint_loads": 0,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
    }, sort_keys=True) + "\n", encoding="utf-8")


def test_sequence_assembly_raw_observation_and_execution(tmp_path):
    """Two actual children, one pure framing contract; zero neuronal calls."""
    import json
    import time
    from pathlib import Path

    import pytest

    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        feedback_from_receipt,
        grade_execution,
        qualified_launch,
        workspace_from_records,
    )
    from humanbrain.sequence_assembly_observation import (
        RawObservation,
        SpliceProposal,
        ToolFeedback,
        apply_splice,
    )

    def save(name, value):
        with (tmp_path / name).open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)

    all_bytes = bytes(range(256))
    maximal_files = [
        {"path": ("a" * 28 + ".py") if i == 0 else f"f{i}.py", "content_hex": (all_bytes * 2).hex()}
        for i in range(4)
    ]
    maximal = workspace_from_records(maximal_files)
    feedback = ToolFeedback(
        0xC0000005, all_bytes * 2, all_bytes[::-1] * 2, True, False, True, True, False
    )
    observation = RawObservation(all_bytes, maximal, feedback)
    frame = observation.sensor_frame()
    absent = RawObservation(b"", None).query_frame()
    empty = RawObservation(
        b"", None, ToolFeedback(0, b"", b"", True, False, True, True, False)
    ).query_frame()
    save(
        "framing-raw.json",
        {
            "observation": observation.to_record(),
            "symbols": frame.symbols,
            "flags": frame.flags,
            "absent": vars(absent),
            "empty": vars(empty),
        },
    )
    assert len(frame.symbols) == 3471 and len(frame.flags) == 12
    assert frame.symbols[:257] == (*all_bytes, 256)
    assert frame.symbols[257:770] == (*all_bytes, *all_bytes, 256)
    assert frame.symbols[770:1283] == (*all_bytes[::-1], *all_bytes[::-1], 256)
    assert frame.symbols[1283:1291] == tuple((0xC0000005).to_bytes(8, "little", signed=True))
    assert absent.symbols[257] == -1 and empty.symbols[257] == 256
    assert absent.flags[0] is False and empty.flags[0] is True
    assert len(maximal.boundaries) == 2052
    stop = SpliceProposal("STOP", "", 999, -1, b"x" * 128, "limit", maximal.sha256)
    assert apply_splice(maximal, stop) == maximal.files
    with pytest.raises(ValueError):
        apply_splice(maximal, SpliceProposal("SPLICE", "f1.py", 0, 1, b"x", "eos", "0" * 64))
    with pytest.raises(ValueError):
        workspace_from_records([{"path": "CON/main.py", "content_hex": ""}])

    original = b"def value():\n    return b'old'\n" + b"#" + b"x" * 150 + b"\n"
    main = (
        b"import app,sys\nsys.stdout.buffer.write(app.value())\n"
        b"sys.stderr.buffer.write(b'\\x80'*512)\n"
    )
    records = [
        {"path": "app.py", "content_hex": original.hex()},
        {"path": "main.py", "content_hex": main.hex()},
    ]
    workspace = workspace_from_records(records)
    expression = b"(b'\\xff\\x00'+b'a'*510)"
    replacement = expression + b" " * (128 - len(expression))
    start = original.index(b"b'old'")
    proposal = SpliceProposal(
        "SPLICE", "app.py", start, start + 6, replacement, "eos", workspace.sha256
    )
    save("execution-input.json", {"files": records, "proposal": proposal.to_record()})
    launch = qualified_launch(Path(__file__).resolve().parents[1])
    after, receipt = execute_proposal(
        workspace,
        proposal,
        tmp_path / "exact-byte-child",
        launch=launch,
        deadline=time.perf_counter() + 10,
    )
    save("exact-byte-execution.json", receipt)
    public = feedback_from_receipt(receipt)
    grade = {
        "expected_returncode": 0,
        "expected_stdout_hex": (b"\xff\x00" + b"a" * 510).hex(),
        "protected_paths": ["main.py"],
    }
    graded = grade_execution(receipt, grade, records)
    save(
        "exact-byte-outcome.json",
        {"feedback": public.to_record(), "grade": graded, "after_files": after},
    )
    assert graded["task_success"] and receipt["accepted"]
    assert public.stdout == b"\xff\x00" + b"a" * 510 and public.stderr == b"\x80" * 512
    assert public.capture_complete and not public.output_limit_exceeded
    assert public.to_record().keys() == {
        "returncode",
        "stdout_hex",
        "stderr_hex",
        "accepted",
        "timeout",
        "capture_complete",
        "files_complete",
        "output_limit_exceeded",
    }
    assert receipt["workspace_sha256"] == workspace.sha256
    assert receipt["proposal_sha256"] == proposal.sha256

    oversized = workspace_from_records(
        [
            {
                "path": "main.py",
                "content_hex": b"import sys\nsys.stdout.buffer.write(b'Z'*513)\n".hex(),
            }
        ]
    )
    _, overflow = execute_proposal(
        oversized,
        None,
        tmp_path / "overflow-child",
        launch=launch,
        deadline=time.perf_counter() + 10,
    )
    save("overflow-execution.json", overflow)
    capped = feedback_from_receipt(overflow)
    save(
        "overflow-outcome.json",
        {"feedback": capped.to_record(), "actual_children": 2, "models": 0, "teachers": 0},
    )
    assert capped.stdout == b"Z" * 512 and capped.output_limit_exceeded
    assert not capped.capture_complete and overflow["stdout_observed_bytes"] == 513
    assert not grade_execution(
        overflow,
        {
            "expected_returncode": 0,
            "expected_stdout_hex": (b"Z" * 512).hex(),
            "protected_paths": [],
        },
        [],
    )["task_success"]


def test_sequence_assembly_reader_causal_prefix_and_atomicity(tmp_path):
    """One authored reader lifecycle fixture; no external prior or task-skill claim."""
    import hashlib
    import json

    from humanbrain import sequence_assembly_cells as assembly_module
    from humanbrain.sequence_assembly_memory_learner import (
        INPUT_WIDTH,
        SequenceAssemblyMemoryLearner,
    )
    from humanbrain.sequence_assembly_observation import RawObservation, ToolFeedback

    raw_path = tmp_path / "sequence-assembly-reader-raw.jsonl"

    def save(row):
        with raw_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")

    def state_equal(actual, expected):
        assert set(actual) == set(expected)
        for key, value in expected.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(actual[key], value), key
            else:
                assert actual[key] == value, key

    reader = SequenceAssemblyMemoryLearner(mode="learned", seed=733)
    with torch.no_grad():
        reader.address_weight.zero_()
        reader.prefix_weight.zero_()
        reader.codec.weight.zero_()
        reader.codec.bias.zero_()
        reader.codec.bias[63] = math.log(256)
    state = reader.ingest([b"abc"])
    assert state.used == 4
    held_tape = state.snapshot()
    held_model = copy.deepcopy(reader.state_dict())
    # Hand-authored public diagnostic fields; there is no child execution here.
    feedback = ToolFeedback(
        returncode=-7, stdout=b"\x00\xff", stderr=b"\xfeerror\x80",
        accepted=False, timeout=False, capture_complete=True, files_complete=True,
        output_limit_exceeded=False)
    observation = RawObservation(
        b"Read the complete raw value after this longer instruction: \xff", None, feedback)
    frame = observation.sensor_frame()
    assert len(observation.request) > 31
    assert len(frame.symbols) == 3471 and len(frame.flags) == 12
    assert INPUT_WIDTH == 66093
    predictions, teacher_events = [], []
    checked_event_work = []
    original_kernel = reader.assemblies._evaluate_validated
    in_reference = False

    def compare_checked(inputs, fallback, prototypes, outgoing, count, *, work=None):
        nonlocal in_reference
        before_work = dict(work or {})
        actual = original_kernel(inputs, fallback, prototypes, outgoing, count, work=work)
        if not in_reference:
            reference_work = {}
            in_reference = True
            try:
                expected = reader.assemblies.evaluate(inputs, fallback, work=reference_work)
            finally:
                in_reference = False
            assert actual.winner_index == expected.winner_index
            assert actual.exact == expected.exact
            assert torch.equal(actual.currents, expected.currents)
            assert torch.equal(actual.agreement_counts, expected.agreement_counts)
            assert reference_work["assembly_validation_values"] == (
                512 * 257 + (512 - count) * (INPUT_WIDTH + 257) + 1)
            expected_work = {key: value for key, value in reference_work.items()
                             if key != "assembly_validation_values"}
            # Both entry paths evaluate identical cells; only bank validation differs.
            actual_work = {key: work.get(key, 0) - before_work.get(key, 0)
                           for key in expected_work}
            assert actual_work == expected_work
            checked_event_work.append(reference_work)
        return actual

    def predict():
        with patch.object(reader.assemblies, "_evaluate_validated", compare_checked):
            prediction = reader.predict_with_receipt(observation, state, max_bytes=33)
        count = int(reader.assemblies.used.item())
        assert prediction.work["assembly_validation_values"] == (
            512 * 257 + (512 - count) * (INPUT_WIDTH + 257) + 1)
        predictions.append(prediction)
        save({"phase": "free", "attempt_id": prediction.attempt_id,
              "output_hex": prediction.generation.data.hex(),
              "stop_reason": prediction.generation.stop_reason,
              "prefixes": [step.prefix.hex() for step in prediction.steps],
              "addresses": [step.selected_index for step in prediction.steps],
              "assemblies": [step.assembly_index for step in prediction.steps],
              "work": prediction.work})
        return prediction

    original_teach = reader.assemblies.prepare_teach

    def capture_offer(context, target, **kwargs):
        row = {"target": target, "selected": context.selected_index,
               "recruited": context.recruited, "enabled": kwargs["learning_enabled"],
               "completed": False}
        teacher_events.append(row)
        try:
            result = original_teach(context, target, **kwargs)
            row.update(completed=True, staged=result.staged)
            return result
        finally:
            row["work"] = dict(kwargs["work"])
            save({"phase": "teacher_offer", **row})

    first = predict()
    assert first.generation.data == b"?" * 33 and first.generation.stop_reason == "limit"
    assert len(first.steps) == 34 and all(step.selected_index == 0 for step in first.steps)
    assert first.work["prefix_cache_advances"] == 33
    assert first.work["prefix_cache_rolling_drops"] == 2
    assert first.work.get("sequence_conditioned_fallbacks", 0) == 0
    assert first.work["sequence_codec_fallbacks"] == 34
    assert first.work["query_raw_bytes"] == len(observation.request)
    assert first.work["sequence_query_chunks"] == (len(observation.request) + 30) // 31
    with patch.object(reader.assemblies, "prepare_teach", capture_offer):
        taught = reader.teach(observation, b"ab", state, prediction=first)
    save({"phase": "enabled_feedback", "count": taught.teacher_count,
          "prefixes": [step["prefix_hex"] for step in taught.teacher_steps],
          "targets": [step["target"] for step in taught.teacher_steps], "work": taught.work})
    assert taught.teacher_count == 3 and taught.output_updated
    assert not taught.address_updated and not taught.legacy_codec_updated
    assert [step["prefix_hex"] for step in taught.teacher_steps] == ["", "61", "6162"]
    assert [step["target"] for step in taught.teacher_steps] == [97, 98, 256]
    assert [step["selected_index"] for step in taught.teacher_steps] == [0, 1, 2]
    assert all(step["recruited"] and step["staged"] for step in taught.teacher_steps)
    assert int(reader.assemblies.used) == 3
    assert taught.work["assembly_output_offers"] == 3
    assert taught.work["assembly_staged_output_updates"] == 3
    assert taught.work["assembly_publications"] == 1
    assert reader.pending_prediction is None

    # Independent scalar IO oracle: preserve every symbol, including raw invalid
    # UTF8 diagnostics, in each learned receptive prototype. No encoder call.
    sensor_bits = []
    for symbol in frame.symbols:
        if symbol == -1:
            sensor_bits.extend([False] * 18)
        else:
            for bit in range(9):
                one = bool((symbol >> bit) & 1)
                sensor_bits.extend((not one, one))
    sensor_bits.extend(frame.flags)
    expected_sensor = torch.tensor(sensor_bits, dtype=torch.bool)
    assert expected_sensor.numel() == 62490
    for index in range(3):
        assert torch.equal(reader.assemblies.prototypes[index, :62490], expected_sensor)
    for name, value in held_model.items():
        if not name.startswith("assemblies."):
            actual = reader.state_dict()[name]
            assert (torch.equal(actual, value)
                    if isinstance(value, torch.Tensor) else actual == value)
    _assert_nested_equal(state.snapshot(), held_tape)
    trained = copy.deepcopy(reader.state_dict())

    second = predict()
    assert second.generation.data == b"ab" and second.generation.stop_reason == "eos"
    assert len(second.steps) == 3
    assert [step.assembly_index for step in second.steps] == [0, 1, 2]
    assert [step.prefix for step in second.steps] == [b"", b"a", b"ab"]
    assert all(step.exact_context for step in second.steps)
    original_logits = second.steps[0].logits.clone()
    second.steps[0].logits[0] += 1
    with pytest.raises(ValueError, match="response or work changed"):
        reader.validate_pending(second, state, observation=observation)
    second.steps[0].logits.copy_(original_logits)
    stale = RawObservation(observation.request + b"!", None, feedback)
    with pytest.raises(ValueError, match="observation changed"):
        reader.validate_pending(second, state, observation=stale)
    with pytest.raises(ValueError, match="pending"):
        reader.load_state_dict(trained)
    assert reader.pending_prediction is second
    state_equal(reader.state_dict(), trained)
    with patch.object(reader.assemblies, "prepare_teach", capture_offer):
        disabled = reader.teach(
            observation, b"c", state, prediction=second, learning_enabled=False)
    save({"phase": "disabled_feedback", "count": disabled.teacher_count,
          "prefixes": [step["prefix_hex"] for step in disabled.teacher_steps],
          "targets": [step["target"] for step in disabled.teacher_steps], "work": disabled.work})
    assert disabled.teacher_count == 2 and not disabled.output_updated
    assert [step["prefix_hex"] for step in disabled.teacher_steps] == ["", "63"]
    assert [step["target"] for step in disabled.teacher_steps] == [99, 256]
    assert all(not step["staged"] for step in disabled.teacher_steps)
    assert disabled.work.get("assembly_staged_output_updates", 0) == 0
    state_equal(reader.state_dict(), trained)
    _assert_nested_equal(state.snapshot(), held_tape)

    third = predict()
    assert third.generation.data == b"ab" and third.generation.stop_reason == "eos"
    finite_calls = 0
    original_finite = assembly_module._finite

    def fail_second_proposal(value, label):
        nonlocal finite_calls
        if label == "assembly output proposal":
            finite_calls += 1
            if finite_calls == 2:
                raise ValueError("authored final sequence proposal failure")
        return original_finite(value, label)

    with (patch.object(reader.assemblies, "prepare_teach", capture_offer),
          patch.object(assembly_module, "_finite", fail_second_proposal),
          pytest.raises(ValueError, match="authored final sequence proposal failure")):
        reader.teach(observation, b"b", state, prediction=third)
    assert finite_calls == 2 and reader.pending_prediction is third
    state_equal(reader.state_dict(), trained)
    _assert_nested_equal(state.snapshot(), held_tape)
    with patch.object(reader, "_context_forward", side_effect=AssertionError("ack forward")):
        ack = reader.acknowledge(third, state)
    save({"phase": "late_failure_ack", "work": ack,
          "used": int(reader.assemblies.used), "teacher_events": len(teacher_events)})
    assert ack["sequence_acknowledgements"] == 1 and reader.pending_prediction is None
    assert ack.get("assembly_output_offers", 0) == 0
    assert len(teacher_events) == 7 and sum(row["completed"] for row in teacher_events) == 6
    assert sum(len(prediction.steps) for prediction in predictions) == 40
    assert len({prediction.attempt_id for prediction in predictions}) == 3
    state_equal(reader.state_dict(), trained)
    _assert_nested_equal(state.snapshot(), held_tape)
    assert len(checked_event_work) == 40
    # Mutations during an owned call must fail before a response becomes pending.
    # Equal-valued rebinding is invisible to byte hashes and needs the identity guard.
    failed_call_events = []
    for mode in ("mutation", "rebinding"):
        original_buffer = reader.assemblies.outgoing
        original_values = original_buffer.clone()
        experience = reader._active_experience
        calls = 0

        def corrupt_after_event(inputs, fallback, prototypes, outgoing, count, *, work=None):
            nonlocal calls
            result = original_kernel(inputs, fallback, prototypes, outgoing, count, work=work)
            calls += 1
            if calls == 1:
                if mode == "mutation":
                    outgoing[0, 0] += 1
                else:
                    reader.assemblies.outgoing = outgoing.clone()
            return result

        try:
            with (patch.object(reader.assemblies, "_evaluate_validated", corrupt_after_event),
                  pytest.raises(RuntimeError, match="mutated model|rebound assembly")):
                reader.predict_with_receipt(observation, state, max_bytes=2)
            assert reader.pending_prediction is None
            assert reader._active_experience == experience
            failed_call_events.append(calls)
        finally:
            reader.assemblies.outgoing = original_buffer
            original_buffer.copy_(original_values)
        state_equal(reader.state_dict(), trained)
        _assert_nested_equal(state.snapshot(), held_tape)
    (tmp_path / "sequence-assembly-reader-passed.json").write_text(json.dumps({
        "status": "passed", "readers": 1, "assembly_banks": 1,
        "ingests": 1, "raw_ingest_bytes": 3, "ingest_eos": 1, "slots": 4,
        "free_calls": 3, "free_decisions": 40, "teacher_loops": 3,
        "offered_events": 7, "completed_events": 6, "published_output_clamps": 3,
        "disabled_events": 2, "late_failed_events": 1, "recruited_units": 3,
        "validation_rejections": 3, "checkpoint_loads": 0, "actions": 0,
        "teacher_events": teacher_events,
        "checked_reference_events": len(checked_event_work),
        "removed_event_validation_values": sum(
            item["assembly_validation_values"] for item in checked_event_work),
        "in_call_mutation_rejections": 2, "failed_free_call_events": failed_call_events,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "claim": "authored causal sequence mechanics; no broad acquisition or external prior",
    }, sort_keys=True) + "\n", encoding="utf-8")


def test_sequence_assembly_combined_custody_and_atomicity(tmp_path):
    """One combined learner, three real children, four reader offers incl failed staging."""
    import copy
    import json
    import time
    from pathlib import Path
    from unittest.mock import patch

    import pytest
    import torch

    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        feedback_from_receipt,
        qualified_launch,
        workspace_from_records,
    )
    from humanbrain.sequence_assembly_observation import RawObservation, SpliceProposal
    from humanbrain.sequence_assembly_splice_learner import SequenceAssemblySpliceLearner

    def save(name, value):
        with (tmp_path / name).open("x", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                default=lambda item: item.tolist() if isinstance(item, torch.Tensor) else str(item),
            )

    model = SequenceAssemblySpliceLearner(seed=733)
    reader = model.reader
    state = model.ingest([b"ab"])
    with torch.no_grad():
        reader.address_weight.zero_()
        reader.prefix_weight.zero_()
        reader.output_weight.zero_()
        reader.codec.weight.zero_()
        reader.codec.bias.zero_()
        reader.codec.bias[256] = 10
        model.actions.operation_weight.zero_()
        model.actions.operation_bias.copy_(torch.tensor([10.0, 0.0]))
    initial = copy.deepcopy(model.state_dict())
    tape = state.snapshot()
    # An earlier otherwise-valid action change must never publish before a later
    # invalid assembly or fixed-context tensor is detected.
    load_rejections = []
    for fault in ("assembly_capacity", "fixed_context"):
        incoming = copy.deepcopy(initial)
        incoming["actions.operation_bias"].add_(1)
        if fault == "assembly_capacity":
            incoming["reader.assemblies.used"].fill_(513)
        else:
            incoming["reader.context.signs"][0, 0] *= -1
        save(
            f"load-{fault}-input.json",
            {
                "fault": fault,
                "earlier_action_bias": incoming["actions.operation_bias"],
                "incoming_used": incoming["reader.assemblies.used"],
                "first_sign": incoming["reader.context.signs"][0, 0],
            },
        )
        with pytest.raises(ValueError) as rejected:
            model.load_state_dict(incoming)
        load_rejections.append({"fault": fault, "error": str(rejected.value)})
        _assert_nested_equal(model.state_dict(), initial)  # noqa: F821 -- existing test-file helper
        del incoming
    save("load-rejections.json", load_rejections)
    del initial

    workspace = workspace_from_records(
        [
            {"path": "main.py", "content_hex": b"print(0)\n".hex()},
        ]
    )
    request = b"Change the output to one while keeping the surrounding statement."
    observation = RawObservation(request, workspace)
    launch = qualified_launch(Path(__file__).resolve().parents[1])
    total_free_decisions = 0
    feedback_records = []
    for index, phase in enumerate(("splice", "stop", "ack")):
        before = copy.deepcopy(model.state_dict())
        seen = []
        actual_prepare = reader._prepare_observation

        def inspect_observation(value, tape_state, work, _original=actual_prepare, _seen=seen):
            _seen.append(value.to_record())
            return _original(value, tape_state, work)

        with patch.object(reader, "_prepare_observation", side_effect=inspect_observation):
            prediction = model.propose(observation, state)
        total_free_decisions += len(prediction.payload.steps)
        save(
            f"{index}-proposal.json",
            {
                "observation": observation.to_record(),
                "observed_preparation": seen,
                "proposal": prediction.proposal.to_record(),
                "attempt_id": prediction.attempt_id,
                "reader_attempt_id": prediction.payload.attempt_id,
                "observation_sha256": prediction.observation_sha256,
                "model_sha256": prediction.model_sha256,
                "memory_sha256": prediction.memory_sha256,
                "reader_trace_sha256": prediction.payload.trace_sha256,
                "decisions": len(prediction.payload.steps),
                "work": prediction.work,
            },
        )
        assert seen == [observation.to_record()]
        assert prediction.proposal.operation == "STOP"
        assert prediction.payload.observation is observation
        if index:
            assert observation.feedback.stdout == b"0\n"
        assert model._pending is not None and reader.pending_prediction is prediction.payload
        after, execution = execute_proposal(
            workspace,
            prediction.proposal,
            tmp_path / f"actual-{phase}",
            launch=launch,
            deadline=time.perf_counter() + 10,
        )
        completed = time.perf_counter()
        path = tmp_path / f"{index}-execution.json"
        save(path.name, execution)
        envelope = {
            "attempt_id": prediction.attempt_id,
            "proposal_sha256": prediction.proposal_sha256,
            "workspace_sha256": workspace.sha256,
            "execution": execution,
        }
        assert execution["returncode"] == 0 and execution["stdout_hex"] == b"0\n".hex()

        if phase == "splice":
            teacher = SpliceProposal("SPLICE", "main.py", 6, 7, b"1", "eos", workspace.sha256)
            prepared_heads = []
            actual_update = model.actions.prepare_update

            def observed_update(head, *args, _bound=(actual_update, prepared_heads), **kwargs):
                original_update, heads = _bound
                update = original_update(head, *args, **kwargs)
                heads.append(head)
                return update

            failed = {}

            def late_failure(
                stage,
                work=None,
                _bound=(failed, prepared_heads, path, execution, completed, before),
            ):
                failed, prepared_heads, path, execution, completed, before = _bound
                failed.update(
                    revision=stage.revision, work=dict(work), prepared_heads=list(prepared_heads)
                )
                assert json.loads(path.read_text()) == execution
                assert time.perf_counter() >= completed
                _assert_nested_equal(model.state_dict(), before)  # noqa: F821 -- existing test-file helper
                raise ValueError("authored final assembly publication rejection")

            with (
                patch.object(model.actions, "prepare_update", side_effect=observed_update),
                patch.object(reader.assemblies, "publish", side_effect=late_failure),
                pytest.raises(ValueError, match="authored final"),
            ):
                model.teach(prediction, observation, state, teacher, envelope)
            save("failed-teacher-stage.json", failed)
            assert prepared_heads == ["start", "end", "operation"]
            assert failed["work"]["assembly_output_offers"] == 2
            _assert_nested_equal(model.state_dict(), before)  # noqa: F821 -- existing test-file helper
            assert reader.pending_prediction is prediction.payload and model._pending is not None
            feedback = model.teach(prediction, observation, state, teacher, envelope)
            save(
                "successful-teacher.json",
                {
                    "reader_count": feedback.reader_feedback.teacher_count,
                    "teacher_steps": feedback.reader_feedback.teacher_steps,
                    "action_losses": feedback.action_losses,
                    "work": feedback.work,
                },
            )
            assert feedback.reader_feedback.teacher_count == 2
            assert feedback.reader_feedback.output_updated
            assert set(feedback.action_losses) == {"start", "end", "operation"}
            feedback_records.append(feedback.work)
        elif phase == "stop":
            teacher = SpliceProposal("STOP", "", 0, 0, b"", "eos", workspace.sha256)
            with patch.object(reader, "teach", side_effect=AssertionError("STOP payload teacher")):
                feedback = model.teach(prediction, observation, state, teacher, envelope)
            assert feedback.reader_feedback is None
            assert set(feedback.action_losses) == {"operation"}
            assert feedback.work["reader"]["sequence_acknowledgements"] == 1
            for name, value in before.items():
                if name.startswith("reader.") and isinstance(value, torch.Tensor):
                    assert torch.equal(value, model.state_dict()[name])
            feedback_records.append(feedback.work)
        else:
            with patch.object(reader, "teach", side_effect=AssertionError("ack payload teacher")):
                work = model.finish_without_teaching(prediction, observation, state, envelope)
            assert work["reader"]["sequence_acknowledgements"] == 1
            _assert_nested_equal(model.state_dict(), before)  # noqa: F821 -- existing test-file helper
            feedback_records.append(work)
        assert model._pending is None and reader.pending_prediction is None
        _assert_nested_equal(state.snapshot(), tape)  # noqa: F821 -- existing test-file helper
        workspace = workspace_from_records(after)
        observation = RawObservation(request, workspace, feedback_from_receipt(execution))
        del before
    save(
        "combined-qa-outcome.json",
        {
            "actual_child_executions": 3,
            "combined_model_constructions": 1,
            "ingestions": 1,
            "raw_memory_bytes": 2,
            "memory_eos": 1,
            "free_queries": 3,
            "actual_free_decisions": total_free_decisions,
            "reader_teacher_calls": 2,
            "reader_target_offers_including_failed_stage": 4,
            "action_labels_including_failed_stage": 7,
            "checkpoint_loads": 0,
            "feedback_work": feedback_records,
        },
    )


def test_sequence_assembly_action_blocks_embedding_and_atomic_update(tmp_path):
    """One authored action circuit; no reader, tape, child, checkpoint or task."""
    import json
    import math
    from unittest.mock import patch

    import pytest
    import torch

    import humanbrain.sequence_assembly_action_cells as cells
    from humanbrain.multiscale_opaque_event_tape import OpaqueWorkspaceFile
    from humanbrain.sequence_assembly_observation import (
        RawObservation,
        SpliceWorkspace,
        ToolFeedback,
    )

    model = cells.SequenceAssemblyActionCircuit(seed=733)
    widths = {"start": 748947, "end": 42561, "operation": 149252}
    assert widths == cells.WIDTHS and cells.BLOCK_SIZE == 16
    old_widths = {"start": 6177, "end": 2049, "operation": 4244}
    legacy = {}
    for head, width in old_widths.items():
        shape = (2, width) if head == "operation" else (width,)
        legacy[head + "_weight"] = (
            torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape) + 1
        ) / 10000
    legacy["operation_bias"] = torch.tensor([-0.125, 0.25])
    model.load_legacy_state(legacy)

    # Independent flat-coordinate inverse: identify every old location and then
    # require the entire complementary new population to be exactly zero.
    def destination(head, old):
        if head == "start":
            if old < 1024:
                return old
            if old < 5152:
                row, column = divmod(old - 1024, 129)
                return 1298 * 32 + row * 513 + column
            if old < 6176:
                return 1298 * (32 + 513) + old - 5152
            return widths[head] - 1
        if head == "end":
            return old if old < 1024 else 1298 * 32 + old - 1024
        if old < 1024:
            return old
        if old < 2048:
            return 1298 * 32 + old - 1024
        if old < 4241:
            row, column = divmod(old - 2048, 17)
            return 1298 * 64 + row * 129 + column
        return widths[head] - 3 + old - 4241

    for head, width in old_widths.items():
        actual = getattr(model, head + "_weight")
        indices = torch.tensor([destination(head, i) for i in range(width)])
        assert torch.equal(actual.index_select(-1, indices), legacy[head + "_weight"])
        complement = torch.ones(widths[head], dtype=torch.bool)
        complement[indices] = False
        assert torch.count_nonzero(actual[..., complement]) == 0
    assert torch.equal(model.operation_bias, legacy["operation_bias"])

    raw = b"abcdefghijklmnop"
    workspace = SpliceWorkspace((OpaqueWorkspaceFile("main.py", raw),))
    feedback = ToolFeedback(-2, b"out", b"err", True, False, True, True, False)
    observation = RawObservation(b"query", workspace, feedback)
    # Exact binary-fraction authored input activities, not a fitted codec.
    codes = ((torch.arange(258 * 32).reshape(258, 32) % 17) - 8).float() / 16
    encoded = cells.encode(observation, codes)

    def rows(values, size):
        result = torch.zeros(size, 32)
        for index, token in enumerate(values):
            result[index] = codes[token]
        return result

    query = torch.zeros(1298, 32)
    query[:6] = rows([256, *reversed(b"query")], 6)
    query[257:261] = rows([*b"out", 256], 4)
    query[770:774] = rows([*b"err", 256], 4)
    query[1283:1291] = rows(list((-2).to_bytes(8, "little", signed=True)), 8)
    query[1291:] = rows([1, 1, 1, 0, 1, 1, 0], 7)
    assert torch.equal(encoded.query, query)
    path = rows([256, *reversed(b"main.py")], 32)
    whole = rows([*raw, 256], 513)
    contexts = torch.zeros(17, 32, 32)
    for boundary in range(17):
        positions = [boundary - j for j in range(1, 17)] + list(range(boundary, boundary + 16))
        for lag, position in enumerate(positions):
            if 0 <= position <= 16:
                contexts[boundary, lag] = codes[raw[position] if position < 16 else 256]

    def oracle(head):
        if head == "start":
            raw_features = torch.cat((
                (query @ path.T).flatten().expand(17, -1),
                (query @ whole.T).flatten().expand(17, -1),
                torch.matmul(query[None], contexts.transpose(1, 2)).flatten(1),
                torch.arange(17).float()[:, None] / 128,
            ), dim=1)
        elif head == "end":
            raw_features = torch.cat((
                torch.matmul(query[None], contexts.transpose(1, 2)).flatten(1),
                torch.matmul(contexts[0:1], contexts.transpose(1, 2)).flatten(1),
                torch.arange(17).float()[:, None] / 128,
            ), dim=1)
        else:
            span, payload = rows([*raw[2:7], 256], 513), rows([*b"xy", 256], 129)
            raw_features = torch.cat((
                (query @ contexts[2].T).flatten(), (query @ path.T).flatten(),
                (span @ payload.T).flatten(), torch.tensor([5 / 128, 2 / 16, 1.0]),
            ))[None]
        return raw_features / torch.linalg.vector_norm(
            raw_features, dim=1, keepdim=True
        ).clamp_min(1e-8) * math.sqrt(32)

    before = {name: value.clone() for name, value in model.named_parameters()}
    summaries = {}
    for head in widths:
        coordinates = {} if head == "start" else {"path": "main.py", "start": 0}
        if head == "operation":
            coordinates.update(start=2, end=7, payload=b"xy", stop_reason="eos")
        full = oracle(head)
        chunks = list(cells.feature_blocks(head, encoded, **coordinates))
        assert [chunk.shape[0] for chunk in chunks] == ([1] if head == "operation" else [16, 1])
        actual = torch.cat(chunks)
        torch.testing.assert_close(actual, full, rtol=1e-6, atol=1e-7)
        del actual, chunks
        parameter = getattr(model, head + "_weight")
        logits = (full * parameter).sum(1)
        if head == "operation":
            logits += model.operation_bias
        probabilities = torch.softmax(logits.double(), dim=0)
        decision = model.score(head, encoded, **coordinates)
        torch.testing.assert_close(decision.logits, logits, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(decision.probabilities, probabilities, rtol=1e-6, atol=1e-7)
        target = 1 if head == "operation" else 16
        work = {}
        update = model.prepare_update(head, encoded, target, work=work, **coordinates)
        error = probabilities.clone()
        error[target] -= 1
        # Whole-bank direct affine-output rule; no reverse feature derivative.
        weighted = error[:, None] * full.double()
        signal = weighted if head == "operation" else weighted.sum(0)
        expected = parameter - 0.01 * signal.float()
        torch.testing.assert_close(update.parameters[0][1], expected, rtol=1e-6, atol=2e-7)
        if head == "operation":
            torch.testing.assert_close(
                update.parameters[1][1], model.operation_bias - 0.01 * error.float()
            )
        assert update.target == target and update.nll == pytest.approx(
            float(-torch.log_softmax(logits.double(), dim=0)[target]), rel=1e-6, abs=1e-6
        )
        assert work[head + "_feature_blocks"] == (2 if head == "operation" else 4)
        assert work[head + "_feature_normalizations"] == (2 if head == "operation" else 34)
        assert work[head + "_teacher_offers"] == 1
        summaries[head] = work
        del full, weighted, signal, expected, update
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])

    real_finite = cells._finite
    late_proposals = 0

    def reject_last_proposal(value, name):
        nonlocal late_proposals
        if name == "extended action parameter proposal":
            late_proposals += 1
            if late_proposals == 2:
                value.fill_(float("nan"))
        return real_finite(value, name)

    with (
        patch.object(cells, "_finite", side_effect=reject_last_proposal),
        pytest.raises(ValueError, match="extended action parameter proposal"),
    ):
        model.prepare_update("operation", encoded, 1, path="main.py", start=2,
                             end=7, payload=b"xy", stop_reason="eos")
    assert late_proposals == 2
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
    receipt = {
        "status": "passed", "standalone_action_circuits": 1, "legacy_tensor_loads": 1,
        "checkpoint_loads": 0, "reader_calls": 0, "children": 0,
        "source_files": 1, "raw_file_bytes": 16, "boundaries": 17,
        "direct_scores": 3, "successful_staged_updates": 3, "rejected_staged_updates": 1,
        "offered_action_labels": 4, "published_updates": 0,
        "module_feature_passes": 14, "module_feature_blocks": 22,
        "module_candidate_features": 142, "independent_full_oracle_passes": 3,
        "late_proposals_checked": late_proposals, "teacher_work": summaries,
        "accounting": model.accounting(),
    }
    with (tmp_path / "sequence-assembly-action-qa.json").open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, allow_nan=False)


def test_persistent_coincidence_1024_promotion_and_append(tmp_path):
    """Authored capacity fixture; no experiment checkpoint, query or teacher."""
    from humanbrain.persistent_coincidence_memory import (
        PersistentCoincidenceMemory512,
        PersistentCoincidenceMemory1024,
    )

    old, memory = PersistentCoincidenceMemory512(), PersistentCoincidenceMemory1024()
    source = old.ingest([b"a" * 504])
    before = source.snapshot()
    work = {}
    promoted = memory.promote_512_state(source, work=work)
    assert promoted.used == source.used == 505
    for name, value in vars(source).items():
        actual = getattr(promoted, name)
        assert torch.equal(actual if name == "write_index" else actual[:512], value)
        assert all(actual.untyped_storage().data_ptr() != other.untyped_storage().data_ptr()
                   for other in vars(source).values())
    assert not promoted.keys[512:].any() and not promoted.sensory[512:].any()
    assert (promoted.successor[512:] == -1).all()
    assert work["promotion_ingest_calls"] == 0
    assert work["promotion_copied_bytes"] == 825352
    assert work["promotion_initialized_bytes"] == 1650696
    assert memory.accounting(promoted)["state_tensor_bytes"] == 1650696
    with pytest.raises(ValueError, match="geometry"):
        old.check_state(promoted)

    documents = [b"b" * 64] * 4 + [b"c" * 128] * 2
    append_work = {}
    expanded = memory.ingest(documents, promoted, work=append_work)
    assert expanded.used == 1023 and promoted.used == 505
    for name in ("keys", "sensory", "successor"):
        assert torch.equal(getattr(expanded, name)[:505], before[name][:505])
    cursor = 505
    for document in documents:
        assert not expanded.keys[cursor].any()
        for offset, token in enumerate((*document, 256)):
            slot = cursor + offset
            assert expanded.sensory[slot, token] == 1
            expected_next = slot if token == 256 else slot + 1
            assert expanded.successor[slot] == expected_next
        cursor += len(document) + 1
    assert expanded.successor[504] == 504
    assert append_work["coincidence_ingest_allocation_steps"] == 518
    assert append_work["coincidence_ingest_sensory_publications"] == 518
    assert append_work["coincidence_ingest_document_resets"] == 6
    occupied = expanded.snapshot()
    with pytest.raises(ValueError, match="capacity exceeded"):
        memory.ingest([b"x"], expanded)
    _assert_nested_equal(expanded.snapshot(), occupied)
    full = memory.ingest([b""], expanded)
    assert full.used == 1024 and full.sensory[1023, 256] == 1
    all_cursors = memory.initial_recall(full)
    assert all_cursors.tolist() == list(range(1024))
    assert torch.equal(memory.advance(full, all_cursors), full.successor)
    with pytest.raises(ValueError, match="capacity exceeded"):
        memory.ingest([b""], full)
    _assert_nested_equal(source.snapshot(), before)
    _assert_nested_equal(expanded.snapshot(), occupied)
    _coincidence_qa_save(tmp_path / "persistent-1024-promotion.json", {
        "status": "passed", "memory_objects": 2, "promotions": 1,
        "successful_ingests": 3, "raw_ingested_bytes": 1016, "ingest_eos": 8,
        "rejected_ingests": 2, "final_slots": 1024, "promotion_work": work,
        "append_work": append_work, "readers": 0, "teachers": 0,
        "actual_children": 0, "checkpoint_loads": 0,
    })


def test_persistent_1024_competition_and_prefix(tmp_path):
    """Full candidate coverage, extreme/tied scores and exact rowwise prefix activity."""
    from humanbrain.neuronal_byte_coincidence import encode_symbol_rails
    from humanbrain.persistent_coincidence_reader import compete512, compete1024
    from humanbrain.sequence_assembly_prefix import (
        RollingPrefixCoincidence,
        RollingPrefixCoincidence1024,
    )

    limit = torch.finfo(torch.float32).max
    extremes = torch.full((1024,), -limit)
    extremes[511:513] = limit
    vectors = [torch.tensor([-7.0]), torch.arange(512, dtype=torch.float32),
               torch.arange(513, dtype=torch.float32), torch.zeros(1024), extremes]
    receipts = []
    for values, winner in zip(vectors, (0, 511, 512, 0, 511), strict=True):
        result = compete1024(values)
        assert result.winner_index == winner
        assert result.winner_activity.sum() == 1 and result.winner_activity[winner]
        assert torch.isfinite(result.pair_currents).all()
        n = values.numel()
        assert result.work["competition_candidates"] == n
        assert result.work["competition_pairs"] == n * (n - 1) // 2
        assert result.work["competition_threshold_layers"] == 2
        if n == 512:
            old = compete512(values)
            assert torch.equal(result.pair_currents, old.pair_currents)
            assert torch.equal(result.winner_currents, old.winner_currents)
            assert result.work == old.work
        receipts.append(result.work)
    with pytest.raises(ValueError, match="1..512"):
        compete512(vectors[2])
    with pytest.raises(ValueError, match="finite"):
        compete1024(torch.tensor([float("nan")]))

    keys = torch.zeros(1024, 32, 18, dtype=torch.bool)
    symbols = torch.arange(1024, dtype=torch.int64) % 256
    keys[:, 0] = encode_symbol_rails(symbols)
    new_work, old_work = {}, {}
    cache = RollingPrefixCoincidence1024(keys, work=new_work)
    old_cache = RollingPrefixCoincidence(keys[:512], work=old_work)
    with pytest.raises(ValueError, match="1..512"):
        RollingPrefixCoincidence(keys)
    for prefix in (b"", b"a", b"ab"):
        if prefix:
            cache.advance(prefix[-1])
            old_cache.advance(prefix[-1])
        actual = cache.features()
        assert actual.shape == (1024, 32, 32)
        assert torch.equal(actual[:512], old_cache.features())
        assert torch.equal(actual[:512], actual[512:])
        # Only one key lag is occupied. Independently check all nonzero matches.
        wanted = torch.zeros_like(actual)
        gain = math.sqrt(32) / math.sqrt(len(prefix) + 1)
        for lag, token in enumerate(reversed(prefix)):
            wanted[symbols == token, lag, 0] = gain
        assert torch.allclose(actual, wanted, rtol=2e-6, atol=2e-7)
    assert new_work["prefix_cache_advances"] == 2
    _coincidence_qa_save(tmp_path / "persistent-1024-competition-prefix.json", {
        "status": "passed", "competition_calls_1024": 5, "competition_calls_512": 1,
        "rejected_competitions": 2, "prefix_caches": 2, "rejected_prefix_caches": 1,
        "prefix_feature_calls": 6, "prefix_advances": 4, "full_prefix_scalar_checks": 3,
        "competition_work": receipts, "prefix_work": new_work,
        "readers": 0, "teachers": 0, "actual_children": 0, "checkpoint_loads": 0,
    })


def test_factorized_router_local_outcome_contract(tmp_path):
    """Authored routing mechanics only: no reader, old state, workload or child."""
    from humanbrain.factorized_sequence_features import (
        INPUT_WIDTH,
        SENSOR_SYMBOLS,
        build_candidate_view,
        build_context,
    )
    from humanbrain.factorized_sequence_router import (
        FactorizedSequenceRouter,
        compete_float64,
    )

    def pattern(request):
        # Independent fixed bit fan-out constructs an actual canonical layout.
        result = torch.zeros(INPUT_WIDTH, dtype=torch.bool)
        for position, symbol in enumerate([*request, 256]):
            for bit in range(9):
                result[position * 18 + 2 * bit + ((symbol >> bit) & 1)] = True
        prefix_start = SENSOR_SYMBOLS * 18 + 12
        for bit in range(9):
            result[prefix_start + 2 * bit + ((257 >> bit) & 1)] = True
        result[-129] = True  # Output position zero; absent files/feedback/memory.
        return result

    vectors = [torch.tensor([1.0, 1.0 + 2**-40, -3.0], dtype=torch.float64),
               torch.tensor([7.0, 7.0, -1.0], dtype=torch.float64),
               torch.tensor([-4.0], dtype=torch.float64)]
    competitions = []
    for values, expected in zip(vectors, (1, 0, 0), strict=True):
        receipt = compete_float64(values)
        n = len(values)
        pairs = n * (n - 1) // 2
        scalar_pairs = [float(values[i]) - float(values[j])
                        for i in range(n) for j in range(i + 1, n)]
        assert receipt.scores.dtype == torch.float64
        assert receipt.winner_index == expected
        assert receipt.pair_currents.tolist() == scalar_pairs
        assert receipt.comparator_activity.tolist() == [x >= 0 for x in scalar_pairs]
        assert receipt.winner_activity.tolist() == [i == expected for i in range(n)]
        assert receipt.work["competition_pairs"] == pairs
        assert receipt.work["competition_threshold_evaluations"] == pairs + n
        assert receipt.work["competition_receipt_tensor_bytes"] == 17 * n + 9 * pairs
        competitions.append(receipt.work)
    # The genuinely distinct winning scores collapse if illicitly cast to f32.
    assert vectors[0].float()[0] == vectors[0].float()[1]
    maximum = torch.finfo(torch.float64).max
    with pytest.raises(ValueError, match="comparator currents must be finite"):
        compete_float64(torch.tensor([maximum, -maximum], dtype=torch.float64))
    with pytest.raises(ValueError, match="scores must be finite"):
        compete_float64(torch.tensor([float("nan")], dtype=torch.float64))

    raw_patterns = [pattern(b"ab"), pattern(b"ac")]
    feature_work, work = {}, {}
    views = [build_candidate_view(raw, work=feature_work) for raw in raw_patterns]
    current = build_context(raw_patterns[0], work=feature_work)
    router = FactorizedSequenceRouter()
    initial = router.snapshot()
    stage = router.begin_stage(work=work)
    decision = router.evaluate(current, views, stage=stage, work=work)
    # Scalar arithmetic over independently constructed fixed population columns.
    baseline = [math.fsum(float(current.populations[int(index)]) * float(value)
                          for index, value in zip(view.indices, view.values, strict=True)) / 45
                for view in views]
    assert decision.selected_index == 0
    assert decision.scores.tolist() == pytest.approx(baseline, rel=0, abs=2e-15)
    assert baseline[0] > baseline[1]
    assert decision.energy == pytest.approx(0.5, rel=0, abs=2e-15)
    assert work["router_candidates"] == 2
    assert work["factorized_activity_calls"] == 3  # All owners, then selected-row custody.
    assert work["router_weight_products"] == sum(view.indices.numel() + 14 for view in views)
    assert router.tensor_bytes == 8266432

    # A nested tensor mutation must fail before even a disabled outcome is accepted.
    held_value = decision.selected_activity[0].item()
    decision.selected_activity[0] = held_value + 1
    with pytest.raises(ValueError, match="decision, trace or weights changed"):
        router.prepare_outcome(decision, 65, 66, stage=stage, learning_enabled=False, work=work)
    decision.selected_activity[0] = held_value
    before = stage.theta.clone()
    frozen = router.prepare_outcome(decision, 65, 66, stage=stage,
                                    learning_enabled=False, work=work)
    assert not frozen.staged and not frozen.skipped
    assert stage.revision == 0 and torch.equal(stage.theta, before)
    assert torch.equal(router.theta, before)

    indices = decision.selected_indices.tolist()
    own_inputs = decision.selected_activity.tolist()
    scalar_energy = math.fsum(x * x for x in own_inputs)
    scalar_error = -1.0 - baseline[0]
    scalar_delta = [scalar_error * x / scalar_energy for x in own_inputs]
    update = router.prepare_outcome(decision, 65, 66, stage=stage, work=work)
    raw_receipt = {
        "status": "observed_before_final_assertions", "scores": decision.scores.tolist(),
        "selected": decision.selected_index, "indices": indices, "activity": own_inputs,
        "scalar_energy": scalar_energy, "scalar_error": scalar_error,
        "scalar_delta": scalar_delta, "actual_delta": update.delta.tolist(),
        "actual_score_after": update.score_after, "disabled_score_after": frozen.score_after,
        "competition_work": competitions,
    }
    _coincidence_qa_save(tmp_path / "factorized-router-raw.json", raw_receipt)
    assert update.staged and not update.skipped and update.desired == -1
    assert update.error == pytest.approx(scalar_error, abs=2e-15)
    assert update.delta.tolist() == pytest.approx(scalar_delta, rel=2e-14, abs=2e-15)
    expected = torch.zeros_like(stage.theta)
    expected[torch.tensor(indices, dtype=torch.int64)] = torch.tensor(
        scalar_delta, dtype=torch.float64)
    assert torch.allclose(stage.theta, expected, rtol=2e-14, atol=2e-15)
    assert stage.revision == 1 and torch.equal(router.theta, before)
    assert update.score_after == pytest.approx(-1, rel=0, abs=2e-14)
    assert frozen.delta.tolist() == update.delta.tolist()
    with pytest.raises(ValueError, match="decision, trace or weights changed"):
        router.prepare_outcome(decision, 65, 66, stage=stage, work=work)
    router.validate_stage(stage, work=work)
    router.commit_validated(stage)
    assert torch.equal(router.theta, stage.theta)
    assert router.theta.untyped_storage().data_ptr() != stage.theta.untyped_storage().data_ptr()
    router.load_state_dict(initial)
    assert torch.equal(router.theta, initial["theta"])

    # Canonical contexts share EOS/BOS, so a genuine nonempty zero lexical row
    # is not promised. The actual empty-bank decision has zero energy and skips.
    empty_stage = router.begin_stage(work=work)
    empty = router.evaluate(current, [], stage=empty_stage, work=work)
    skipped = router.prepare_outcome(empty, 65, 65, stage=empty_stage, work=work)
    assert empty.selected_index is None and empty.energy == 0 and empty.competition is None
    assert skipped.skipped and not skipped.staged and skipped.desired == 1
    assert empty_stage.revision == 0 and torch.equal(empty_stage.theta, initial["theta"])
    assert torch.equal(router.theta, initial["theta"])
    assert work["router_evaluations"] == 2 and work["router_outcome_offers"] == 3
    assert work["router_staged_updates"] == 1 and work["router_skipped_outcomes"] == 1
    _coincidence_qa_save(tmp_path / "factorized-router-passed.json", {
        "status": "passed", "routers": 1, "reader_models": 0,
        "candidate_builds": 2, "current_builds": 1, "router_evaluations": 2,
        "activity_calls": 3, "direct_valid_competitions": 3,
        "direct_rejected_competitions": 2, "internal_competitions": 1,
        "outcome_attempts": 5, "accepted_outcome_offers": 3,
        "rejected_tampered_outcomes": 1, "rejected_stale_outcomes": 1,
        "staged_router_updates": 1, "router_commits": 1,
        "snapshots": 1, "same_instance_state_restores": 1,
        "reader_teachers": 0, "ingests": 0, "actual_children": 0, "checkpoint_loads": 0,
        "feature_work": feature_work, "router_work": work,
    })


def test_factorized_reader_action_persistence_and_atomicity(tmp_path):
    """One authored persistent actor; full tape, direct corrections and one child."""
    import copy
    import time
    from pathlib import Path
    from unittest.mock import patch

    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        qualified_launch,
        workspace_from_records,
    )
    from humanbrain import factorized_sequence_memory_learner as module
    from humanbrain.factorized_sequence_splice_learner import FactorizedSequenceSpliceLearner
    from humanbrain.sequence_assembly_observation import (
        RawObservation,
        SpliceProposal,
        ToolFeedback,
    )

    model = FactorizedSequenceSpliceLearner(seed=733)
    reader = model.reader
    original_owned = reader.router._evaluate_owned_free
    checked_router_events = []

    def checked_owned(context, candidates, owned, *, work=None):
        before = dict(work)
        actual = original_owned(context, candidates, owned, work=work)
        actual_work = {key: value - before.get(key, 0) for key, value in work.items()}
        reference_work = {}
        reference = reader.router.evaluate(context, candidates, work=reference_work)
        for decision in (actual, reference):
            assert decision._stage is None
        actual_fields, reference_fields = dict(vars(actual)), dict(vars(reference))
        for fields in (actual_fields, reference_fields):
            competition = fields["competition"]
            fields["competition"] = None if competition is None else vars(competition)
        _assert_nested_equal(actual_fields, reference_fields)
        # Same arithmetic/receipt work; only the redundant checked entry is absent.
        omitted = {
            "factorized_validation_values": 1_033_304,
            "factorized_validation_norm_products": 1_033_290,
            "factorized_validation_norm_reductions": 1_033_290 - 45,
            "factorized_validation_cast_values": 1_033_290,
            "factorized_named_tensor_bytes": 16 * 1_033_290,
            "router_validation_values": 1_033_304,
            "router_custody_bytes_hashed": 8 * 1_033_304,
        }
        for key in actual_work.keys() | reference_work.keys() | omitted.keys():
            assert reference_work.get(key, 0) - actual_work.get(key, 0) == omitted.get(key, 0)
        checked_router_events.append(len(candidates))
        return actual

    with torch.no_grad():
        reader.address_weight.zero_()
        reader.prefix_weight.zero_()
        reader.output_weight.zero_()
        reader.codec.weight.zero_()
        reader.codec.bias.zero_()
        reader.codec.bias[256] = 10
        model.actions.operation_weight.zero_()
        model.actions.operation_bias.copy_(torch.tensor([10.0, 0.0]))
    state = model.ingest([b"a" * 512])
    assert state.used == 513 and state.keys.shape == (1024, 32, 18)
    tape = state.snapshot()
    held = copy.deepcopy(model.state_dict())
    observation = RawObservation(
        b"Read this complete retained value after the long instruction: \xff", None,
        ToolFeedback(-7, b"\x00\xff", b"failure\x80", False, False, True, True, False))
    with patch.object(reader.router, "_evaluate_owned_free", side_effect=checked_owned):
        first = reader.predict_with_receipt(observation, state, max_bytes=2)
    assert first.generation.data == b"" and first.generation.stop_reason == "eos"
    assert len(first.steps) == 1 and first.steps[0].address_scores.numel() == 513
    assert first.steps[0].exact_context is None
    assert first.work.get("assembly_agreement_cells", 0) == 0
    taught = reader.teach(observation, b"a", state, prediction=first)
    assert taught.teacher_count == 2 and taught.output_updated and taught.routing_updated
    assert int(reader.assemblies.used) == len(reader._candidate_views) == 2
    assert [r["prefix_hex"] for r in taught.teacher_steps] == ["", "61"]
    assert [r["target"] for r in taught.teacher_steps] == [97, 256]
    assert taught.teacher_steps[0]["routing_outcome"]["skipped"]
    for row in taught.teacher_steps:
        desired = 1 if row["emitted_byte"] == row["target"] else -1
        assert row["routing_outcome"]["desired"] == desired
        if not row["routing_outcome"]["skipped"]:
            assert row["routing_outcome"]["score_after"] == pytest.approx(
                row["routing_outcome"]["desired"], abs=1e-12)
    for key, value in held.items():
        if not key.startswith(("reader.assemblies.", "reader.router.")):
            actual = model.state_dict()[key]
            assert (
                torch.equal(actual, value) if isinstance(value, torch.Tensor) else actual == value
            )
    _assert_nested_equal(state.snapshot(), tape)
    trained = copy.deepcopy(model.state_dict())
    views = reader._candidate_views
    with patch.object(reader.router, "_evaluate_owned_free", side_effect=checked_owned):
        second = reader.predict_with_receipt(observation, state, max_bytes=2)
    assert all(step.address_scores.numel() == 513 for step in second.steps)
    assert second.work.get("assembly_agreement_cells", 0) == 0
    original_score = second.steps[0].routing["selected_score"]
    second.steps[0].routing["selected_score"] += 1
    with pytest.raises(ValueError, match="response or work changed"):
        reader.validate_pending(second, state, observation=observation)
    second.steps[0].routing["selected_score"] = original_score
    disabled = reader.teach(observation, b"b", state, prediction=second, learning_enabled=False)
    assert not disabled.routing_updated and not disabled.output_updated
    assert all(not row["routing_outcome"]["staged"] and not row["staged"]
               for row in disabled.teacher_steps)
    _assert_nested_equal(model.state_dict(), trained)
    assert reader._candidate_views is views
    # Strict restoration derives views from actual incoming prototype bytes.
    model.load_state_dict(trained)
    assert reader._candidate_views is not views
    _assert_nested_equal(model.state_dict(), trained)
    for fault in ("router", "prototype", "fixed"):
        incoming = copy.deepcopy(trained)
        incoming["actions.operation_bias"].add_(1)
        if fault == "router":
            incoming["reader.router.theta"][0] = float("nan")
        elif fault == "prototype":
            incoming["reader.assemblies.prototypes"][0].zero_()
        else:
            incoming["reader.context.signs"][0, 0] *= -1
        with pytest.raises(ValueError):
            model.load_state_dict(incoming)
        _assert_nested_equal(model.state_dict(), trained)
        del incoming
    del held
    # Force three raw output events without fitting: exercise digest reuse across
    # changing prefixes. Restore the same live tensors after teacher-free ack.
    borrowed_currents = reader.assemblies.outgoing.clone()
    with torch.no_grad():
        reader.assemblies.outgoing[:2].zero_()
    with patch.object(reader.router, "_evaluate_owned_free", side_effect=checked_owned):
        bounded = reader.predict_with_receipt(observation, state, max_bytes=2)
    assert len(bounded.steps) == 3 and bounded.generation.data == b"\x00\x00"
    assert bounded.generation.stop_reason == "limit"
    reader.acknowledge(bounded, state)
    with torch.no_grad():
        reader.assemblies.outgoing.copy_(borrowed_currents)
    _assert_nested_equal(model.state_dict(), trained)
    _assert_nested_equal(state.snapshot(), tape)
    assert checked_router_events == [0] + [2] * (len(second.steps) + 3)
    # A borrowed-buffer replacement must be detected even with identical values.
    original_route = reader._route
    borrowed = reader.assemblies.outgoing
    experience = reader._active_experience
    failed_events = []

    def replace_after_route(*args, **kwargs):
        result = original_route(*args, **kwargs)
        failed_events.append(1)
        reader.assemblies.outgoing = borrowed.clone()
        return result

    with (patch.object(reader, "_route", side_effect=replace_after_route),
          pytest.raises(RuntimeError)):
        reader.predict_with_receipt(observation, state, max_bytes=0)
    reader.assemblies.outgoing = borrowed
    assert failed_events == [1] and reader._active_experience == experience
    assert reader.pending_prediction is None
    routing_failures = []
    for fault in ("theta_rebind", "theta_numpy", "theta_version", "owner_rebind"):
        borrowed_theta = reader.router.theta
        original_theta = borrowed_theta.clone()
        original_owner = reader.router._owner
        original_version = borrowed_theta._version
        events = []

        def corrupt_after_route(*args, fault=fault, tensor=borrowed_theta,
                                version=original_version, events=events, **kwargs):
            result = original_route(*args, **kwargs)
            events.append(1)
            if fault == "theta_rebind":
                reader.router.theta = tensor.clone()
            elif fault == "theta_numpy":
                tensor.numpy()[0] += 1
                assert tensor._version == version
            elif fault == "theta_version":
                with torch.no_grad():
                    tensor.add_(0)
                assert tensor._version != version
            else:
                reader.router._owner = object()
            return result

        try:
            with patch.object(reader, "_route", side_effect=corrupt_after_route), pytest.raises(
                    RuntimeError):
                reader.predict_with_receipt(observation, state, max_bytes=0)
        finally:
            reader.router.theta = borrowed_theta
            reader.router._owner = original_owner
            with torch.no_grad():
                borrowed_theta.copy_(original_theta)
        assert events == [1] and reader._active_experience == experience
        assert reader.pending_prediction is None
        _assert_nested_equal(model.state_dict(), trained)
        _assert_nested_equal(state.snapshot(), tape)
        routing_failures.append(fault)
    workspace = workspace_from_records([{"path": "main.py", "content_hex": b"print(0)\n".hex()}])
    action_observation = RawObservation(
        b"Change the printed value to one and retain the statement.", workspace)
    prediction = model.propose(action_observation, state)
    assert prediction.payload.observation is action_observation
    assert prediction.proposal.operation == "STOP"
    _, execution = execute_proposal(
        workspace, prediction.proposal, tmp_path / "actual-action",
        launch=qualified_launch(Path(__file__).resolve().parents[1]),
        deadline=time.perf_counter() + 10)
    assert execution["returncode"] == 0 and execution["stdout_hex"] == b"0\n".hex()
    envelope = {"attempt_id": prediction.attempt_id, "proposal_sha256": prediction.proposal_sha256,
                "workspace_sha256": workspace.sha256, "execution": execution}
    teacher = SpliceProposal("SPLICE", "main.py", 6, 7, b"1", "eos", workspace.sha256)
    failed_validation = []

    def reject_final_views(*args, **kwargs):
        failed_validation.append(dict(kwargs["work"]))
        raise ValueError("authored final joint view rejection")

    with (patch.object(module, "validate_candidate_view", side_effect=reject_final_views),
          pytest.raises(ValueError, match="authored final joint")):
        model.teach(prediction, action_observation, state, teacher, envelope)
    assert len(failed_validation) == 1
    assert failed_validation[0]["router_outcome_offers"] == 2
    assert failed_validation[0]["assembly_output_offers"] == 2
    _assert_nested_equal(model.state_dict(), trained)
    assert model._pending is not None and reader.pending_prediction is prediction.payload
    feedback = model.teach(prediction, action_observation, state, teacher, envelope)
    assert feedback.reader_feedback.routing_updated and feedback.reader_feedback.output_updated
    assert model._pending is None and reader.pending_prediction is None
    assert int(reader.assemblies.used) == len(reader._candidate_views) == 4
    _assert_nested_equal(state.snapshot(), tape)
    _coincidence_qa_save(tmp_path / "factorized-integration-passed.json", {
        "status": "passed", "actors": 1, "readers": 1, "ingests": 1,
        "ingested_raw_bytes": 512, "ingested_eos": 1, "memory_used": state.used,
        "reader_predictions": 3, "action_predictions": 1,
        "completed_free_events": (len(first.steps) + len(second.steps)
                                  + len(bounded.steps) + len(prediction.payload.steps)),
        "checked_router_calls": len(checked_router_events),
        "checked_router_owner_counts": checked_router_events,
        "teacher_free_acknowledgements": 1,
        "failed_free_calls": 1 + len(routing_failures),
        "failed_free_events": 1 + len(routing_failures),
        "routing_custody_failures": routing_failures,
        "teacher_calls": 4, "teacher_symbols_including_failed": 8,
        "successful_enabled_teacher_calls": 2, "disabled_teacher_calls": 1,
        "failed_teacher_calls": 1, "actual_children": 1,
        "same_instance_state_restores": 1, "rejected_state_loads": 3,
        "experimental_checkpoint_loads": 0, "router_updates": {
            "first": taught.work, "disabled": disabled.work,
            "failed": failed_validation[0], "action": feedback.work},
    })


def test_persistent_v5_summary_contract(tmp_path):
    """Authored scalar rows only: no stream builder, learner, prior or child."""
    from copy import deepcopy

    from experiments.r4_integrated.persistent_language_action_v5 import summarize
    from experiments.r4_integrated.review_persistent_language_action_v5 import check_metrics

    episodes = []
    for index in range(24):
        episode = {
            "episode_id": f"A{index:02d}",
            "phase": "acquisition",
            "kind": "reading" if index % 2 == 0 else "edit",
            "encounter": ("corrected_first", "changed_form", "exact_delayed")[index // 8],
        }
        if index >= 16:
            episode["same_session_origin_id"] = f"A{index - 16:02d}"
        episodes.append(episode)
    for index in range(16):
        episodes.append(
            {
                "episode_id": f"F{index:02d}",
                "phase": "transfer",
                "kind": "reading" if index % 2 == 0 else "edit",
                "encounter": "fresh",
            }
        )
    for index in range(8):
        episodes.append(
            {
                "episode_id": f"C{index:02d}",
                "phase": "recovery",
                "kind": "edit",
                "encounter": "teacher_free",
            }
        )
    for index in range(16):
        episodes.append(
            {
                "episode_id": f"R{index:02d}",
                "phase": "retention",
                "kind": "reading" if index % 2 == 0 else "edit",
                "encounter": "old_retained" if index < 8 else "old_regression",
                "previous_session_episode_id": f"P{index:02d}",
            }
        )
    manifest = {"episodes": episodes}
    recovery_correct = (
        (False, True, False),
        (False, False, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (False, True, True),
        (False, False, False),
        (False, False, False),
    )
    recovery_proposals = (
        ("x", "x", "x"),
        ("x", "y", "y"),
        (None, None, None),
        ("a", "b", "c"),
        ("x", "x", "x"),
        ("x", "x", "y"),
        ("a", "b", "a"),
        ("x", "x", None),
    )
    routes = ("online", "learning_disabled", "simple")
    attempts = []
    for route in routes:
        for episode in episodes:
            episode_id = episode["episode_id"]
            index = int(episode_id[1:])
            recovery = episode["phase"] == "recovery"
            for attempt_index in range(1, 4 if recovery else 2):
                unsupported = route == "online" and episode_id == "C07" and attempt_index == 3
                correct = (
                    recovery_correct[index][attempt_index - 1]
                    if recovery
                    else index in {0, 3, 9, 10, 16, 17}
                    if episode_id.startswith("A")
                    else index in {0, 1, 4, 5}
                    if episode_id.startswith("R")
                    else index in {0, 1}
                )
                observation = episode.get("same_session_origin_id", episode_id)
                if episode_id.startswith("R") and index < 8:
                    observation = f"prior-{index}"
                project = episode_id in {"A01", "A09"}
                attempts.append(
                    {
                        "route": route,
                        "episode_id": episode_id,
                        "phase": episode["phase"],
                        "kind": episode["kind"],
                        "attempt_index": attempt_index,
                        "correct": correct,
                        "status": "unsupported" if unsupported else "completed",
                        "observation_sha256": observation,
                        "memory_used": 505 if episode_id.startswith("A") and index < 8 else 864,
                        "feedback_consumed": recovery
                        and attempt_index > 1
                        and route != "learning_disabled"
                        and not unsupported,
                        "proposal_signature": recovery_proposals[index][attempt_index - 1]
                        if recovery
                        else None,
                        "project_id": "project" if project else None,
                        "predecessor_episode_id": "A01" if episode_id == "A09" else None,
                        "workspace_lifecycle": "carry_actual_project"
                        if episode_id == "A09"
                        else "reset",
                    }
                )
    prior = {}
    for route in routes:
        prior[route] = {
            f"P{index:02d}": {
                "attempt_index": 3 if index >= 8 else 1,
                "route_ordinal": 64 + index if index < 8 else index - 8,
                "correct": index in ({0, 2, 4, 6} if route == "simple" else {0, 1, 2, 3})
                if route != "learning_disabled"
                else True,
                "observation_sha256": f"prior-{index}",
            }
            for index in range(16)
        }
    _coincidence_qa_save(
        tmp_path / "persistent-v5-summary-raw.json",
        {
            "manifest": manifest,
            "attempts": attempts,
            "prior_outcomes": prior,
            "scope": {
                "authored_episodes": 64,
                "rows_per_route": 80,
                "rows": 240,
                "summary_calls": 3,
                "expected_rejections": 2,
                "models": 0,
                "builders": 0,
                "checkpoint_loads": 0,
                "children": 0,
            },
        },
    )
    result = summarize(attempts, manifest, prior)
    assert check_metrics({"metrics": result}, attempts, manifest, prior) == result
    wrong_report = deepcopy(result)
    wrong_report["online"]["recovery"]["later_correct"] += 1
    with pytest.raises(ValueError, match="independent outcomes"):
        check_metrics({"metrics": wrong_report}, attempts, manifest, prior)
    _coincidence_qa_save(tmp_path / "persistent-v5-summary-result.json", result)
    assert len(episodes) == 64 and len(attempts) == 240
    for route in routes:
        actual = result[route]
        assert sum(row["route"] == route for row in attempts) == 80
        for encounter in ("corrected_first", "changed_form", "exact_delayed"):
            assert actual["acquisition_encounters"][encounter] == {
                "reading": {"episodes": 4, "first_correct": 1},
                "edit": {"episodes": 4, "first_correct": 1},
            }
        expected_phase = {
            "acquisition": ((12, 12, 3), (12, 12, 3)),
            "transfer": ((8, 8, 1), (8, 8, 1)),
            "recovery": ((0, 0, 0), (8, 24, 2)),
            "retention": ((8, 8, 2), (8, 8, 2)),
        }
        for phase, counts in expected_phase.items():
            for kind, (count, row_count, correct) in zip(("reading", "edit"), counts, strict=True):
                assert actual["phases"][phase][kind] == {
                    "episodes": count,
                    "attempts": row_count,
                    "first_correct": correct,
                    "unsupported": int(
                        route == "online" and phase == "recovery" and kind == "edit"
                    ),
                }
        assert actual["recovery"] == {
            "first_failures": 6,
            "first_correct": 2,
            "later_correct": 3,
            "final_attempt_correct": 3,
            "final_recovered_after_first_failure": 2,
            "policy": "three fixed attempts; private grade never controls retries",
            "tool_feedback_consumed": route != "learning_disabled",
            "feedback_consuming_attempts": {"online": 15, "learning_disabled": 0, "simple": 16}[
                route
            ],
            "identical_consecutive_proposals": 7,
        }
        assert [
            actual[key]
            for key in ("same_session_retained", "same_session_forgotten", "same_session_improved")
        ] == [1, 1, 1]
        assert len(actual["same_session_retention"]) == 8
        for row in actual["same_session_retention"]:
            assert row["intervening_attempts"] == 15
            assert row["initial_memory_slots"] == 505 and row["memory_slots"] == 864
        assert [actual[key] for key in ("retained", "forgotten", "improved")] == [2, 2, 2]
        assert len(actual["retention"]) == 16
        for index, row in enumerate(actual["retention"]):
            assert row["prior_route"] == ("simple" if route == "simple" else "online")
            assert row["initial_correct"] == (
                index in ({0, 2, 4, 6} if route == "simple" else {0, 1, 2, 3})
            )
            assert row["intervening_current_attempts"] == 64 + index
            assert row["intervening_prior_attempts"] == (7 - index if index < 8 else 79 - index)
            assert row["intervening_attempts"] == (71 if index < 8 else 143)
            assert row["prior_attempt_index"] == (1 if index < 8 else 3)
            assert row["same_raw_observation_as_prior"] == (index < 8)
            assert (
                row["observation_scope"]
                == "current authored reset vs previous actual last-attempt input"
            )
        assert [row["workspace_from_actual_predecessor"] for row in actual["project_chains"]] == [
            False,
            True,
        ]
        assert actual["project_chains"][1]["predecessor_episode_id"] == "A01"

    changed_input = deepcopy(attempts)
    changed_input[16]["observation_sha256"] = "different actual input"
    with pytest.raises(ValueError, match="same-session exact repeat input or age changed"):
        summarize(changed_input, manifest, prior)
    changed_age = list(attempts)
    changed_age[0], changed_age[1] = changed_age[1], changed_age[0]
    with pytest.raises(ValueError, match="same-session exact repeat input or age changed"):
        summarize(changed_age, manifest, prior)
    _coincidence_qa_save(
        tmp_path / "persistent-v5-summary-passed.json",
        {
            "status": "passed",
            "summary_calls": 3,
            "expected_rejections": 2,
            "independent_metric_audits": 2,
            "rejected_forged_metric_reports": 1,
            "authored_episodes": 64,
            "rows_per_route": 80,
            "rows": 240,
            "models": 0,
            "builders": 0,
            "checkpoint_loads": 0,
            "children": 0,
        },
    )


def test_input_defined_recognition_local_contract(tmp_path):
    """Two authored contexts; no reader, task stream, checkpoint or execution."""
    from dataclasses import replace
    from types import SimpleNamespace
    from unittest.mock import patch

    from humanbrain import input_defined_recognition as module
    from humanbrain.factorized_sequence_features import (
        INPUT_WIDTH,
        SENSOR_SYMBOLS,
        build_candidate_view,
        build_context,
    )
    from humanbrain.factorized_sequence_router import (
        FactorizedSequenceRouter,
        compete_float64,
        compete_float64_1024,
    )

    def pattern(request):
        raw = torch.zeros(INPUT_WIDTH, dtype=torch.bool)
        for position, symbol in enumerate([*request, 256]):
            for bit in range(9):
                raw[position * 18 + 2 * bit + ((symbol >> bit) & 1)] = True
        start = SENSOR_SYMBOLS * 18 + 12
        for bit in range(9):
            raw[start + 2 * bit + ((257 >> bit) & 1)] = True
        raw[-129] = True
        return raw

    # Exercise the expanded, genuinely f64 competition while preserving old512.
    scores = torch.zeros(1024, dtype=torch.float64)
    scores[-2], scores[-1] = 1.0, 1.0 + 2**-40
    competition = compete_float64_1024(scores)
    assert scores.float()[-2] == scores.float()[-1]
    assert competition.winner_index == 1023
    assert competition.pair_currents[-1].item() == -(2**-40)
    assert int(competition.comparator_activity.sum()) == 1022 * 1021 // 2
    assert competition.winner_activity.tolist() == [False] * 1023 + [True]
    assert competition.work["competition_pairs"] == 1024 * 1023 // 2
    with pytest.raises(ValueError, match=r"1\.\.512"):
        compete_float64(scores)
    maximum = torch.finfo(torch.float64).max
    with pytest.raises(ValueError, match="comparator currents must be finite"):
        compete_float64_1024(torch.tensor([maximum, -maximum], dtype=torch.float64))

    patterns = [pattern(b"ab"), pattern(b"ac")]
    feature_work, work, legacy_work = {}, {}, {}
    views = [build_candidate_view(raw, work=feature_work) for raw in patterns]
    currents = [build_context(raw, work=feature_work) for raw in patterns]
    wirings = [
        module.build_self_activity(view, raw, work=feature_work)
        for view, raw in zip(views, patterns, strict=True)
    ]
    oracle_h, oracle_z, oracle_kappa, oracle_baseline = [], [], [], []
    for view, wiring in zip(views, wirings, strict=True):
        values = view.values.tolist()
        squared = [value * value for value in values]
        own_norm = math.sqrt(math.fsum(value * value for value in squared))
        h = [value / own_norm / math.sqrt(2) for value in squared] + [0.0] * 14
        products = [
            float(currents[0].populations[int(index)]) * value
            for index, value in zip(view.indices, values, strict=True)
        ]
        norm = math.sqrt(math.fsum(value * value for value in products))
        z = [value / norm / math.sqrt(2) for value in products] + [0.0] * 14
        assert wiring.values.tolist() == pytest.approx(h, rel=2e-14, abs=2e-15)
        assert wiring.indices[: len(values)].tolist() == view.indices.tolist()
        assert wiring.values[-14:].tolist() == [0.0] * 14
        oracle_h.append(h)
        oracle_z.append(z)
        oracle_kappa.append(math.fsum(a * b for a, b in zip(h, z, strict=True)))
        oracle_baseline.append(math.fsum(products) / 45)
    assert feature_work["recognition_self_builds"] == 2
    assert feature_work["recognition_self_tensor_bytes"] == sum(
        12 * (view.indices.numel() + 14) for view in views
    )

    prior = FactorizedSequenceRouter()
    prior.theta[views[0].indices.long()] = 0.03
    theta_before = prior.theta.clone()
    recognition = module.InputDefinedRecognition()
    initial = recognition.snapshot()
    assert recognition.capacity == 1024 and recognition.tensor_bytes == 8192
    assert recognition.gains.shape == (1024,) and not bool(recognition.gains.any())
    legacy = prior.evaluate(currents[0], views, work=legacy_work)
    actual = recognition.evaluate(currents[0], views, wirings, prior.theta, work=work)
    assert torch.equal(actual.prior_scores, legacy.scores)
    assert torch.equal(actual.scores, legacy.scores)
    assert actual.kappas.tolist() == pytest.approx(oracle_kappa, rel=2e-14, abs=2e-15)
    assert actual.selected_index == legacy.selected_index
    scalar_prior = [
        math.fsum(
            float(prior.theta[int(index)]) * value
            for index, value in zip(wiring.indices, z, strict=True)
        )
        + baseline
        for wiring, z, baseline in zip(wirings, oracle_z, oracle_baseline, strict=True)
    ]
    assert actual.prior_scores.tolist() == pytest.approx(scalar_prior, rel=2e-14, abs=2e-15)
    owned = recognition._begin_validated_free_response(prior.theta, work)
    fast = recognition._evaluate_owned_free(currents[0], views, wirings, owned, work=work)
    for field in ("scores", "prior_scores", "kappas"):
        assert torch.equal(getattr(fast, field), getattr(actual, field))
    assert fast.selected_index == actual.selected_index

    # Cached fixed wiring and a prepared local context each reject mutation.
    held = wirings[0].values[0].item()
    wirings[0].values[0] = held + 1
    with pytest.raises(ValueError, match="self activity or its canonical view changed"):
        module.check_self_activity(wirings[0], views[0], work)
    wirings[0].values[0] = held
    stage = recognition.begin_stage(work)
    prepared = recognition.prepare_context(
        currents[0], views[0], wirings[0], 0, prior.theta, stage=stage, work=work
    )
    tampered = replace(prepared, score_before=prepared.score_before + 1)
    with pytest.raises(ValueError, match="changed after preparation"):
        recognition.prepare_recognition(tampered, stage=stage, work=work)
    disabled = recognition.prepare_recognition(
        prepared, stage=stage, learning_enabled=False, work=work
    )
    assert not disabled.staged and not disabled.skipped
    assert stage.revision == 0 and torch.equal(stage.gains, initial["gains"])
    assert torch.equal(recognition.gains, initial["gains"])
    assert disabled.score_after == pytest.approx(1.0, abs=2e-14)
    update = recognition.prepare_recognition(prepared, stage=stage, work=work)
    expected_gain = (1 - scalar_prior[0]) / oracle_kappa[0]
    assert update.delta_gain == pytest.approx(expected_gain, rel=2e-14, abs=2e-15)
    assert update.score_after == pytest.approx(1.0, rel=0, abs=2e-14)
    assert update.delta_gain == disabled.delta_gain
    assert stage.revision == 1 and update.staged
    assert stage.gains[0] == pytest.approx(expected_gain, rel=2e-14, abs=2e-15)
    assert not bool(stage.gains[1:].any())
    assert torch.equal(recognition.gains, initial["gains"])
    with pytest.raises(ValueError, match="changed after preparation"):
        recognition.prepare_recognition(prepared, stage=stage, work=work)
    changed = recognition.evaluate(currents[0], views, wirings, prior.theta, stage=stage, work=work)
    assert changed.scores[0] == pytest.approx(1.0, rel=0, abs=2e-14)
    assert changed.scores[1].item() == actual.scores[1].item()
    assert torch.equal(changed.prior_scores, actual.prior_scores)
    recognition.validate_stage(stage, work)
    recognition.commit_validated(stage)
    assert torch.equal(recognition.gains, stage.gains)
    assert recognition.gains.data_ptr() != stage.gains.data_ptr()
    with pytest.raises(RuntimeError, match="owned recognition gains or theta changed"):
        recognition._check_owned_free_response(owned)

    live = recognition.gains.clone()
    failed_stage = recognition.begin_stage(work)
    with pytest.raises(ValueError, match="exact context"):
        recognition.prepare_context(
            currents[0], views[1], wirings[1], 1, prior.theta, stage=failed_stage, work=work
        )
    later = recognition.prepare_context(
        currents[1], views[1], wirings[1], 1, prior.theta, stage=failed_stage, work=work
    )
    # Reject the final finite validation after genuine local scalar arithmetic.
    # Replace only this module's math binding, never Python's shared math module.
    with (
        patch.object(module, "math", SimpleNamespace(isfinite=lambda value: False)),
        pytest.raises(ValueError, match="local proposal must be finite"),
    ):
        recognition.prepare_recognition(later, stage=failed_stage, work=work)
    assert failed_stage.revision == 0 and torch.equal(failed_stage.gains, live)
    assert torch.equal(recognition.gains, live)
    _coincidence_qa_save(
        tmp_path / "input-defined-recognition-raw.json",
        {
            "prior_scores": actual.prior_scores.tolist(),
            "self_activity": oracle_h,
            "actual_activity_oracle": oracle_z,
            "kappa_oracle": oracle_kappa,
            "actual_kappas": actual.kappas.tolist(),
            "expected_gain": expected_gain,
            "actual_gain": update.gain_after,
            "corrected_scores": changed.scores.tolist(),
            "expanded_competition_work": competition.work,
            "work": work,
            "feature_work": feature_work,
        },
    )
    invalid = {"gains": live.clone(), "_extra_state": recognition.get_extra_state()}
    invalid["gains"][0], invalid["gains"][4] = 77.0, float("nan")
    with pytest.raises(ValueError, match="gains must be finite"):
        recognition.load_state_dict(invalid)
    assert torch.equal(recognition.gains, live)
    invalid = {
        "gains": live.clone(),
        "_extra_state": recognition.get_extra_state() | {"capacity": 512},
    }
    with pytest.raises(ValueError, match="state metadata mismatch"):
        recognition.load_state_dict(invalid)
    assert torch.equal(recognition.gains, live)
    recognition.load_state_dict(initial)
    empty = recognition.evaluate(currents[0], [], [], prior.theta, work=work)
    assert empty.selected_index is None and empty.energy == empty.selected_kappa == 0
    assert empty.competition is None and empty.scores.numel() == 0
    assert torch.equal(prior.theta, theta_before)
    assert torch.equal(recognition.gains, initial["gains"])
    assert work["recognition_evaluations"] == 4 and work["recognition_candidates"] == 6
    assert work["recognition_prepared_contexts"] == 2
    assert work["recognition_offers"] == 3 and work["recognition_staged_updates"] == 1
    assert work["recognition_local_errors"] == work["recognition_local_divisions"] == 3
    assert work["recognition_stage_copy_bytes"] == 2 * 8192
    assert work["factorized_activity_calls"] == 11
    _coincidence_qa_save(
        tmp_path / "input-defined-recognition-passed.json",
        {
            "status": "passed",
            "recognition_modules": 1,
            "legacy_routers": 1,
            "reader_models": 0,
            "task_builders": 0,
            "checkpoint_loads": 0,
            "children": 0,
            "authored_contexts": 2,
            "candidate_builds_including_canonical_checks": 4,
            "current_builds": 2,
            "self_activity_builds": 2,
            "recognition_evaluations": 4,
            "legacy_evaluations": 1,
            "prepared_context_attempts": 3,
            "prepared_contexts": 2,
            "recognition_update_attempts": 5,
            "recognition_offers": 3,
            "enabled_staged_updates": 1,
            "disabled_proposals": 1,
            "late_failed_proposals": 1,
            "rejected_tampered_contexts": 1,
            "rejected_stale_contexts": 1,
            "recognition_commits": 1,
            "snapshots": 1,
            "same_instance_state_restores": 1,
            "rejected_state_loads": 2,
            "expanded_competitions": 1,
            "direct_rejected_competitions": 2,
            "work": work,
            "feature_work": feature_work,
        },
    )


def test_recognition_reader_growth_and_executed_atomicity(tmp_path):
    """One actor, one child, causal recognition and lossless independent growth."""
    import copy
    import time
    from pathlib import Path
    from unittest.mock import patch

    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        qualified_launch,
        workspace_from_records,
    )
    from humanbrain import recognition_sequence_memory_learner as module
    from humanbrain.persistent_coincidence_memory import PersistentCoincidenceMemory1024
    from humanbrain.persistent_coincidence_reader import compete2048
    from humanbrain.recognition_sequence_splice_learner import RecognitionSequenceSpliceLearner
    from humanbrain.sequence_assembly_cells import SequenceAssemblyCells1024
    from humanbrain.sequence_assembly_observation import (
        RawObservation,
        SpliceProposal,
        ToolFeedback,
    )
    from humanbrain.sequence_assembly_prefix import RollingPrefixCoincidence2048

    model = RecognitionSequenceSpliceLearner(seed=733)
    reader = model.reader
    with torch.no_grad():
        reader.address_weight.zero_()
        reader.prefix_weight.zero_()
        reader.output_weight.zero_()
        reader.codec.weight.zero_()
        reader.codec.bias.zero_()
        reader.codec.bias[256] = 10
        model.actions.operation_weight.zero_()
        model.actions.operation_bias.copy_(torch.tensor([10.0, 0.0]))
    state = model.ingest([b"a"])
    tape = state.snapshot()
    before = copy.deepcopy(model.state_dict())
    workspace = workspace_from_records([{"path": "main.py", "content_hex": b"print(0)\n".hex()}])
    observation = RawObservation(b"Change the printed value to one.", workspace)
    prediction = model.propose(observation, state)
    assert prediction.proposal.operation == "STOP" and len(prediction.payload.steps) == 1
    _, execution = execute_proposal(
        workspace, prediction.proposal, tmp_path / "actual-action",
        launch=qualified_launch(Path(__file__).resolve().parents[1]),
        deadline=time.perf_counter() + 10)
    _coincidence_qa_save(tmp_path / "recognition-actual-action.json", {
        "proposal": prediction.proposal.to_record(), "execution": execution})
    assert execution["returncode"] == 0 and execution["stdout_hex"] == b"0\n".hex()
    envelope = {"attempt_id": prediction.attempt_id, "proposal_sha256": prediction.proposal_sha256,
                "workspace_sha256": workspace.sha256, "execution": execution}
    teacher = SpliceProposal("SPLICE", "main.py", 6, 7, b"1", "eos", workspace.sha256)
    failed = []

    def reject_final_wiring(wiring, view, pattern, work):
        failed.append(dict(work))
        raise ValueError("authored final recognition wiring rejection")

    with (patch.object(module, "validate_self_activity", side_effect=reject_final_wiring),
          pytest.raises(ValueError, match="authored final recognition")):
        model.teach(prediction, observation, state, teacher, envelope)
    assert len(failed) == 1 and failed[0]["recognition_offers"] == 2
    assert failed[0]["assembly_output_offers"] == 2
    _assert_nested_equal(model.state_dict(), before)
    assert model._pending is not None and reader.pending_prediction is prediction.payload
    feedback = model.teach(prediction, observation, state, teacher, envelope)
    taught = feedback.reader_feedback
    assert taught.recognition_updated and taught.output_updated and not taught.routing_updated
    assert int(reader.assemblies.used) == len(reader._candidate_views) == 2
    assert len(reader._self_wirings) == 2
    assert model._pending is None and reader.pending_prediction is None
    assert [row["target"] for row in taught.teacher_steps] == [49, 256]
    for row in taught.teacher_steps:
        association = row["recognition_update"]
        assert association["selected_index"] == row["selected_index"]
        assert association["context_sha256"] == row["routing"]["context_sha256"]
        assert association["score_after"] == pytest.approx(1, abs=1e-12)
        assert association["staged"] and not association["skipped"]
    for name, value in before.items():
        if not name.startswith(("reader.assemblies.", "reader.recognition.", "actions.")):
            _assert_nested_equal(model.state_dict()[name], value)
    trained = copy.deepcopy(model.state_dict())
    old_views, old_wirings = reader._candidate_views, reader._self_wirings
    load_work = {}
    model.load_state_dict(trained, work=load_work)
    assert reader._candidate_views is not old_views and reader._self_wirings is not old_wirings
    for old, new in zip(old_wirings, reader._self_wirings, strict=True):
        assert old.indices.data_ptr() != new.indices.data_ptr()
        assert old.values.data_ptr() != new.values.data_ptr()
    _assert_nested_equal(model.state_dict(), trained)
    incoming = copy.deepcopy(trained)
    incoming["actions.operation_bias"].add_(1)
    incoming["reader.recognition.gains"][2] = 1
    with pytest.raises(ValueError, match="outside occupied"):
        model.load_state_dict(incoming)
    _assert_nested_equal(model.state_dict(), trained)
    del incoming, before
    feedback_observation = RawObservation(
        observation.request, workspace,
        ToolFeedback(execution["returncode"], bytes.fromhex(execution["stdout_hex"]),
                     bytes.fromhex(execution["stderr_hex"]), execution["timeout"],
                     execution["output_limit_exceeded"], execution["capture_complete"],
                     execution["files_complete"], False))
    second = reader.predict_with_receipt(feedback_observation, state, max_bytes=1)
    assert second.observation is feedback_observation
    assert second.observation.sha256 != prediction.payload.observation.sha256
    assert all(step.exact_context is None for step in second.steps)
    disabled = reader.teach(feedback_observation, b"2", state, prediction=second,
                            learning_enabled=False)
    assert not disabled.recognition_updated and not disabled.output_updated
    assert all(not row["staged"] and not row["recognition_update"]["staged"]
               for row in disabled.teacher_steps)
    _assert_nested_equal(model.state_dict(), trained)
    _assert_nested_equal(state.snapshot(), tape)
    # A borrowed gain replacement cannot hide behind equal numerical values.
    original = reader._recognize
    gains = reader.recognition.gains
    experience = reader._active_experience
    failed_free = []

    def rebind_after_read(*args, **kwargs):
        result = original(*args, **kwargs)
        failed_free.append(1)
        reader.recognition.gains = gains.clone()
        return result

    try:
        with patch.object(reader, "_recognize", side_effect=rebind_after_read), pytest.raises(
                RuntimeError):
            reader.predict_with_receipt(feedback_observation, state, max_bytes=0)
    finally:
        reader.recognition.gains = gains
    assert failed_free == [1] and reader.pending_prediction is None
    assert reader._active_experience == experience
    _assert_nested_equal(model.state_dict(), trained)
    # Full old tape is retained, then append crosses the old1024-slot boundary.
    prior_memory = PersistentCoincidenceMemory1024()
    prior_state = prior_memory.ingest([b"a" * 1023])
    prior_snapshot = prior_state.snapshot()
    growth_work = {}
    grown = reader.memory.promote_1024_state(prior_state, work=growth_work)
    for name, value in vars(prior_state).items():
        actual = getattr(grown, name)
        assert actual.data_ptr() != value.data_ptr()
        assert torch.equal(actual if name == "write_index" else actual[:1024], value)
    assert not bool(grown.keys[1024:].any()) and not bool(grown.sensory[1024:].any())
    assert bool(grown.successor[1024:].eq(-1).all()) and grown.used == 1024
    appended = reader.memory.ingest([b"b"], grown)
    assert appended.used == 1026 and appended.successor[1025].item() == 1025
    _assert_nested_equal(prior_state.snapshot(), prior_snapshot)
    assert grown.used == 1024
    prefix_work = {}
    cache = RollingPrefixCoincidence2048(appended.keys[:appended.used], work=prefix_work)
    initial_prefix = cache.features()
    cache.advance(98)
    advanced_prefix = cache.features()
    assert initial_prefix.shape == advanced_prefix.shape == (1026, 32, 32)
    scores = torch.zeros(1026, dtype=torch.float32)
    scores[1024:] = 1
    competition = compete2048(scores)
    assert competition.winner_index == 1024
    # A small generic bank proves lossless old-owner growth and new owner512.
    bank = SequenceAssemblyCells1024(10)
    metadata = bank.get_extra_state()
    metadata.update(schema="sequence-assembly-cells-v1", capacity=512)
    old_patterns = torch.tensor(
        [[bool((index >> bit) & 1) for bit in range(10)] for index in range(512)])
    old_bank = {"prototypes": old_patterns, "outgoing": torch.zeros(512, 257, dtype=torch.float64),
                "used": torch.tensor(512, dtype=torch.int64), "_extra_state": metadata}
    bank_work = {}
    promoted = bank.promote_512_state(old_bank, work=bank_work)
    for name in ("prototypes", "outgoing", "used"):
        assert promoted[name].data_ptr() != old_bank[name].data_ptr()
        retained = promoted[name] if name == "used" else promoted[name][:512]
        assert torch.equal(retained, old_bank[name])
    bank.load_state_dict(promoted)
    stage = bank.begin_stage(bank_work)
    new_pattern = torch.tensor([False] * 9 + [True])
    prepared = bank.prepare_context(new_pattern, torch.zeros(257, dtype=torch.float64),
                                    stage=stage, work=bank_work)
    assert prepared.recruited and prepared.selected_index == 512
    offered = bank.prepare_teach(prepared, 65, stage=stage, work=bank_work)
    bank.validate_stage(stage, bank_work)
    bank.commit_validated(stage)
    assert offered.staged and int(bank.used) == 513
    assert torch.equal(bank.prototypes[512], new_pattern)
    assert torch.equal(bank.prototypes[:512], old_bank["prototypes"])
    _coincidence_qa_save(tmp_path / "recognition-integration-passed.json", {
        "status": "passed", "combined_models": 1, "ingests": 3,
        "ingested_raw_bytes": 1025, "ingested_eos": 3,
        "free_calls": 2, "free_decisions": 1 + len(second.steps), "failed_free_calls": 1,
        "failed_free_decisions": 1, "actual_children": 1,
        "reader_teacher_calls": 3, "reader_teacher_offers": 6,
        "recognition_offers": 6, "action_teacher_calls": 2, "action_labels": 6,
        "same_instance_loads": 1, "rejected_loads": 1, "checkpoint_loads": 0,
        "standalone_assembly_constructions": 1, "authored_prior_owners": 512,
        "standalone_assembly_offers": 1, "final_standalone_owners": 513,
        "fact_growth": growth_work, "fact_competition": competition.work,
        "prefix_growth": prefix_work, "event_growth": bank_work,
        "load": load_work, "failed_teacher": failed[0],
        "enabled_teacher": taught.work, "disabled_teacher": disabled.work,
    })


def test_persistent_v6_summary_contract(tmp_path):
    """Authored scalar outcomes only; no task builder, learner, prior or child."""
    from copy import deepcopy

    from experiments.r4_integrated.persistent_language_action_v6 import summarize
    from experiments.r4_integrated.persistent_language_action_v6_review import check_metrics

    episodes = []
    retained_indices = [*range(36, 48), *range(54, 60)]
    starters, dependents = (19, 21, 23), (27, 31, 35)
    for index in range(72):
        phase = ("acquisition" if index < 6 or 18 <= index < 24 or 48 <= index < 54
                 else "transfer" if 6 <= index < 18 or 24 <= index < 36
                 else "retention" if index < 60 else "recovery")
        encounter = ("corrected_first" if index < 6 else "changed_form" if 18 <= index < 24
                     else "exact_delayed" if 48 <= index < 54
                     else "fresh" if phase == "transfer"
                     else "prior_regression" if phase == "retention" else "teacher_free")
        project = (starters.index(index) if index in starters
                   else dependents.index(index) if index in dependents else None)
        episode = {
            "episode_id": f"E{index:02d}", "phase": phase,
            "kind": "edit" if phase == "recovery" or index % 2 else "reading",
            "encounter": encounter, "requirement_changed": encounter == "changed_form",
            "project_id": None if project is None else f"project-{project}",
            "predecessor_episode_id": (
                f"E{starters[project]:02d}" if index in dependents else None),
            "workspace_lifecycle": "carry_actual_project" if index in dependents else "reset",
        }
        if phase == "transfer":
            episode["fresh_stratum"] = (
                "instruction_construction" if index < 18 else "structural_combination")
        if encounter == "exact_delayed":
            episode["same_session_origin_id"] = f"E{index - 48:02d}"
        if phase == "retention":
            episode["previous_session_episode_id"] = f"P{retained_indices.index(index):02d}"
        episodes.append(episode)
    manifest = {"episodes": episodes}
    recovery_correct = (
        (False, True, False), (False, False, True), (True, False, True),
        (True, True, False), (False, False, False), (False, True, True),
        (False, False, False), (False, False, False), (False, False, True),
        (True, True, True), (False, False, False), (False, False, False),
    )
    signatures = (
        ("x", "x", "x"), ("x", "y", "y"), (None, None, None), ("a", "b", "c"),
        ("x", "x", "x"), ("x", "x", "y"), ("a", "b", "a"), ("x", "x", None),
        ("x", "y", "z"), ("m", "m", "m"), (None, None, None), ("q", "q", "q"),
    )
    routes = ("online", "learning_disabled", "simple")
    attempts = []
    for route in routes:
        for index, episode in enumerate(episodes):
            recovery = index >= 60
            retained = retained_indices.index(index) if index in retained_indices else None
            for attempt_index in range(1, 4 if recovery else 2):
                unsupported = route == "online" and (
                    index == 16 or (index == 71 and attempt_index == 3))
                correct = (recovery_correct[index - 60][attempt_index - 1] if recovery
                           else retained in {0, 1, 4, 5, 12, 15} if retained is not None
                           else index in {0, 3, 6, 9, 18, 21, 24, 25, 27, 48, 49})
                observation = episode.get("same_session_origin_id", episode["episode_id"])
                if retained is not None and retained < 9:
                    observation = f"prior-{retained}"
                attempts.append({
                    "route": route, "episode_id": episode["episode_id"],
                    "phase": episode["phase"], "kind": episode["kind"],
                    "attempt_index": attempt_index, "correct": correct,
                    "status": "unsupported" if unsupported else "completed",
                    "observation_sha256": observation,
                    "memory_used": 864 if index < 6 else 1836,
                    "feedback_consumed": recovery and attempt_index > 1
                    and route != "learning_disabled" and not unsupported,
                    "proposal_signature": signatures[index - 60][attempt_index - 1]
                    if recovery else None,
                    **{name: episode[name] for name in (
                        "project_id", "predecessor_episode_id", "workspace_lifecycle")},
                })
    prior = {
        route: {
            f"P{index:02d}": {
                "attempt_index": 1 if index < 9 else 3,
                "route_ordinal": 60 + index if index < 9 else index - 9,
                "correct": (True if route == "learning_disabled"
                            else index in ({0, 2, 4, 6, 12, 16} if route == "simple"
                                           else {0, 1, 2, 3, 12, 14})),
                "observation_sha256": f"prior-{index}",
            } for index in range(18)
        } for route in routes
    }
    _coincidence_qa_save(tmp_path / "persistent-v6-summary-raw.json", {
        "manifest": manifest, "attempts": attempts, "prior_outcomes": prior,
        "scope": {"episodes": 72, "rows_per_route": 96, "rows": 288,
                  "summary_calls": 3, "independent_metric_calls": 2,
                  "models": 0, "builders": 0, "checkpoint_loads": 0, "children": 0},
    })
    result = summarize(attempts, manifest, prior)
    _coincidence_qa_save(tmp_path / "persistent-v6-summary-result.json", result)
    assert check_metrics({"metrics": result}, attempts, manifest, prior) == result
    wrong = deepcopy(result)
    wrong["online"]["recovery"]["later_correct"] += 1
    with pytest.raises(ValueError, match="independent outcomes"):
        check_metrics({"metrics": wrong}, attempts, manifest, prior)
    assert len(episodes) == 72 and len(attempts) == 288
    for route in routes:
        actual = result[route]
        assert sum(row["route"] == route for row in attempts) == 96
        for encounter in ("corrected_first", "changed_form", "exact_delayed"):
            assert actual["acquisition_encounters"][encounter] == {
                "reading": {"episodes": 3, "first_correct": 1},
                "edit": {"episodes": 3, "first_correct": 1},
                "changed_requirements": 6 if encounter == "changed_form" else 0,
            }
        phase_counts = {
            "acquisition": ((9, 9, 3), (9, 9, 3)),
            "transfer": ((12, 12, 2), (12, 12, 3)),
            "retention": ((9, 9, 3), (9, 9, 3)),
            "recovery": ((0, 0, 0), (12, 36, 3)),
        }
        for phase, counts in phase_counts.items():
            for kind, (count, rows, correct) in zip(("reading", "edit"), counts, strict=True):
                assert actual["phases"][phase][kind] == {
                    "episodes": count, "attempts": rows, "first_correct": correct,
                    "unsupported": int(route == "online" and (
                        (phase == "transfer" and kind == "reading")
                        or (phase == "recovery" and kind == "edit"))),
                }
        for stratum, groups in {
            "instruction_construction": ((6, 1, 6, 1), (0, 0, 0, 0)),
            "structural_combination": ((6, 1, 3, 1), (0, 0, 3, 1)),
        }.items():
            for dependency, counts in zip(("independent", "dependent_project"), groups,
                                           strict=True):
                for kind, (count, correct) in zip(("reading", "edit"),
                                                 (counts[:2], counts[2:]), strict=True):
                    assert actual["fresh_strata"][stratum][dependency][kind] == {
                        "episodes": count, "first_correct": correct,
                        "unsupported": int(route == "online" and kind == "reading"
                                           and stratum == "instruction_construction"
                                           and dependency == "independent"),
                    }
        assert actual["recovery"] == {
            "first_failures": 9, "first_correct": 3, "later_correct": 4,
            "final_attempt_correct": 5, "final_recovered_after_first_failure": 3,
            "policy": "three fixed attempts; private grade never controls retries",
            "tool_feedback_consumed": route != "learning_disabled",
            "feedback_consuming_attempts": {"online": 23, "learning_disabled": 0, "simple": 24}[
                route],
            "identical_consecutive_proposals": 11,
        }
        retention_keys = (
            "same_session_retained", "same_session_forgotten", "same_session_improved")
        assert [actual[name] for name in retention_keys] == [1, 1, 1]
        assert len(actual["same_session_retention"]) == 6
        for row in actual["same_session_retention"]:
            assert row["intervening_attempts"] == 47
            assert row["initial_memory_slots"] == 864 and row["memory_slots"] == 1836
        assert [actual[name] for name in ("retained", "forgotten", "improved")] == [3, 3, 3]
        assert len(actual["retention"]) == 18
        for index, row in enumerate(actual["retention"]):
            assert row["prior_route"] == ("simple" if route == "simple" else "online")
            assert row["initial_correct"] == (index in (
                {0, 2, 4, 6, 12, 16} if route == "simple" else {0, 1, 2, 3, 12, 14}))
            assert row["intervening_current_attempts"] == retained_indices[index]
            assert row["intervening_prior_attempts"] == (19 - index if index < 9 else 88 - index)
            assert row["intervening_attempts"] == (55 if index < 9 else 124 if index < 12 else 130)
            assert row["prior_attempt_index"] == (1 if index < 9 else 3)
            assert row["same_raw_observation_as_prior"] == (index < 9)
        assert [row["workspace_from_actual_predecessor"] for row in actual["project_chains"]] == [
            False, False, False, True, True, True]
        assert [row["predecessor_episode_id"] for row in actual["project_chains"]][3:] == [
            "E19", "E21", "E23"]
    changed_input = deepcopy(attempts)
    changed_input[48]["observation_sha256"] = "different public input"
    with pytest.raises(ValueError, match="same-session exact repeat input or age changed"):
        summarize(changed_input, manifest, prior)
    changed_age = list(attempts)
    changed_age[0], changed_age[1] = changed_age[1], changed_age[0]
    with pytest.raises(ValueError, match="same-session exact repeat input or age changed"):
        summarize(changed_age, manifest, prior)
    _coincidence_qa_save(tmp_path / "persistent-v6-summary-passed.json", {
        "status": "passed", "summary_calls": 3, "expected_summary_rejections": 2,
        "independent_metric_calls": 2, "rejected_forged_metric_reports": 1,
        "authored_episodes": 72, "rows_per_route": 96, "rows": 288,
        "models": 0, "builders": 0, "checkpoint_loads": 0, "children": 0,
    })


def test_factual_tournament_exact_circuit(tmp_path):
    """All-address winner parity, actual relay traces and strict input contract."""
    from humanbrain.factual_address_tournament import compete_factual
    from humanbrain.persistent_coincidence_reader import compete2048

    widths = (1, 2, 3, 127, 128, 129, 511, 512, 513, 1760, 2017, 2047, 2048)
    cases = [((torch.arange(n, dtype=torch.float32) * 37).remainder(19) - 9)
             for n in widths]
    extreme = torch.finfo(torch.float32).max
    cases.extend([
        torch.tensor([-0.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, -0.0], dtype=torch.float32),
        torch.tensor([extreme, -extreme, extreme], dtype=torch.float32),
        torch.cat((torch.zeros(511), torch.ones(2))),
        torch.ones(513),
    ])
    records = []
    for scores in cases:
        original = scores.clone()
        old = compete2048(scores)
        actual = compete_factual(scores)
        n = scores.numel()
        assert actual.winner_index == old.winner_index
        expected_score = float(scores[old.winner_index].item())
        assert actual.winner_score == expected_score
        assert math.copysign(1, actual.winner_score) == math.copysign(1, expected_score)
        assert torch.equal(actual.scores.view(torch.int32), scores.view(torch.int32))
        assert actual.scores.data_ptr() != scores.data_ptr()
        assert torch.equal(scores.view(torch.int32), original.view(torch.int32))
        assert len(actual.levels) == (n - 1).bit_length()
        ids = list(range(n))
        retained = actual.scores.numel() * actual.scores.element_size()
        for level in actual.levels:
            assert level.input_count == len(ids)
            expected_ids, expected_currents, expected_activity = [], [], []
            for offset in range(0, len(ids) - 1, 2):
                left, right = ids[offset:offset + 2]
                difference = float(original[left]) - float(original[right])
                expected_currents.append(difference)
                expected_activity.append(difference >= 0)
                expected_ids.append(left if difference >= 0 else right)
            if len(ids) % 2:
                expected_ids.append(ids[-1])
            assert level.pair_currents.tolist() == expected_currents
            assert level.left_wins.tolist() == expected_activity
            assert level.identity_relays.tolist() == expected_ids
            expected_potentials = original[expected_ids].to(torch.float64)
            assert torch.equal(level.score_relays.view(torch.int64),
                               expected_potentials.view(torch.int64))
            assert all(not tensor.requires_grad for tensor in (
                level.pair_currents, level.left_wins, level.score_relays, level.identity_relays))
            retained += sum(tensor.numel() * tensor.element_size() for tensor in (
                level.pair_currents, level.left_wins, level.score_relays, level.identity_relays))
            ids = expected_ids
        assert ids == [old.winner_index]
        work = actual.work
        assert work["factual_tournament_candidates"] == n
        assert work["factual_tournament_comparators"] == n - 1
        assert work["factual_tournament_logical_gated_payload_branches"] == 4 * (n - 1)
        assert work["factual_tournament_tensor_where_outputs"] == 2 * (n - 1)
        assert work["factual_tournament_retained_tensor_bytes"] == retained
        assert retained <= work["factual_tournament_peak_named_live_bytes"]
        assert work["factual_tournament_peak_named_live_bytes"] <= 82_899
        assert work["factual_tournament_named_allocated_bytes"] <= 124_086
        exact_bounds = {
            1: (22, 4, 20), 3: (146, 78, 117), 1760: (81_448, 51_047, 57_200),
            2048: (94_184, 59_367, 66_560),
        }
        if n in exact_bounds:
            assert tuple(work["factual_tournament_" + name] for name in (
                "named_allocated_bytes", "retained_tensor_bytes", "peak_named_live_bytes"
            )) == exact_bounds[n]
        if n == 2017:
            assert work["factual_tournament_named_allocated_bytes"] == 124_086
        if n == 2047:
            assert work["factual_tournament_peak_named_live_bytes"] == 82_899
        records.append({"scores": original, "winner": actual.winner_index,
                        "winner_score": actual.winner_score, "old_work": old.work,
                        "new_work": work, "levels": [vars(level) for level in actual.levels]})
        scores.fill_(float("nan"))
        assert torch.equal(actual.scores.view(torch.int32), original.view(torch.int32))
    rejected = [torch.empty(0), torch.zeros(1, 1), torch.zeros(2049),
                torch.zeros(2, dtype=torch.float64), torch.zeros(2, dtype=torch.int64),
                torch.tensor([float("nan")]), torch.tensor([float("inf")]),
                torch.zeros(2, requires_grad=True)]
    for scores in rejected:
        with pytest.raises(ValueError, match="finite detached CPU float32"):
            compete_factual(scores)
    _coincidence_qa_save(tmp_path / "factual-tournament-passed.json", {
        "status": "passed", "valid_cases": len(cases), "invalid_cases": len(rejected),
        "new_competition_calls": len(cases) + len(rejected),
        "old_competition_calls": len(cases), "models": 0, "teacher_calls": 0,
        "checkpoint_loads": 0, "children": 0, "records": records,
    })


def test_tournament_recognition_executed_learning_parity(tmp_path):
    """Two fresh actors compare actual actions, local corrections and strict state."""
    import time
    from pathlib import Path

    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        feedback_from_receipt,
        qualified_launch,
        workspace_from_records,
    )
    from humanbrain.recognition_sequence_splice_learner import RecognitionSequenceSpliceLearner
    from humanbrain.sequence_assembly_observation import (
        RawObservation,
        SpliceProposal,
    )
    from humanbrain.tournament_recognition_learner import TournamentRecognitionSpliceLearner

    models = (RecognitionSequenceSpliceLearner(seed=733),
              TournamentRecognitionSpliceLearner(seed=733))
    workspace = workspace_from_records([{"path": "main.py", "content_hex": b"print(0)\n".hex()}])
    observation = RawObservation(b"Change the printed value to one.", workspace)
    teacher = SpliceProposal("SPLICE", "main.py", 6, 7, b"1", "eos", workspace.sha256)
    predictions, corrections, later, disabled, executions, states = [], [], [], [], [], []
    for index, model in enumerate(models):
        reader = model.reader
        with torch.no_grad():
            reader.address_weight.zero_()
            reader.prefix_weight.zero_()
            reader.output_weight.zero_()
            reader.codec.weight.zero_()
            reader.codec.bias.zero_()
            reader.codec.bias[256] = 10
            model.actions.operation_weight.zero_()
            model.actions.operation_bias.copy_(torch.tensor([10.0, 0.0]))
        state = model.ingest([b"ab"])
        states.append(state)
        prediction = model.propose(observation, state)
        predictions.append(prediction)
        assert prediction.proposal.operation == "STOP" and len(prediction.payload.steps) == 1
        _, execution = execute_proposal(
            workspace, prediction.proposal, tmp_path / f"actual-action-{index}",
            launch=qualified_launch(Path(__file__).resolve().parents[1]),
            deadline=time.perf_counter() + 10)
        executions.append(execution)
        assert execution["returncode"] == 0 and execution["stdout_hex"] == b"0\n".hex()
        envelope = {"attempt_id": prediction.attempt_id,
                    "proposal_sha256": prediction.proposal_sha256,
                    "workspace_sha256": workspace.sha256, "execution": execution}
        correction = model.teach(prediction, observation, state, teacher, envelope)
        corrections.append(correction.reader_feedback)
        assert correction.reader_feedback.recognition_updated
        assert correction.reader_feedback.output_updated and not correction.reader_feedback.routing_updated
        assert model._pending is None and reader.pending_prediction is None
        follow_observation = RawObservation(
            observation.request, workspace, feedback_from_receipt(execution))
        next_prediction = reader.predict_with_receipt(follow_observation, state, max_bytes=1)
        later.append(next_prediction)
        before = copy.deepcopy(model.state_dict())
        disabled_feedback = reader.teach(follow_observation, b"2", state,
                                         prediction=next_prediction, learning_enabled=False)
        disabled.append(disabled_feedback)
        assert not disabled_feedback.output_updated and not disabled_feedback.recognition_updated
        _assert_nested_equal(model.state_dict(), before)
        del before
    _assert_nested_equal(states[0].snapshot(), states[1].snapshot())
    old_state, new_state = models[0].state_dict(), models[1].state_dict()
    assert set(old_state) == set(new_state)
    for key in old_state:
        if isinstance(old_state[key], torch.Tensor):
            assert torch.equal(old_state[key], new_state[key]), key
    assert predictions[0].proposal.to_record() == predictions[1].proposal.to_record()
    for left, right in ((predictions[0].payload, predictions[1].payload), (later[0], later[1])):
        assert left.generation == right.generation and len(left.steps) == len(right.steps)
        for a, b in zip(left.steps, right.steps, strict=True):
            for key in vars(a):
                _assert_nested_equal(getattr(a, key), getattr(b, key))
    for left, right in ((corrections[0], corrections[1]), (disabled[0], disabled[1])):
        assert len(left.teacher_steps) == len(right.teacher_steps)
        for a, b in zip(left.teacher_steps, right.teacher_steps, strict=True):
            _assert_nested_equal(a, b)
    old_work, new_work = predictions[0].payload.work, predictions[1].payload.work
    assert old_work["competition_pairs"] == 3
    assert "competition_pairs" not in new_work
    assert new_work["factual_tournament_candidates"] == 3
    assert new_work["factual_tournament_comparators"] == 2
    assert corrections[1].work["factual_tournament_calls"] == 2
    assert disabled[1].work["factual_tournament_calls"] == 2
    trained = copy.deepcopy(new_state)
    old_wirings = models[1].reader._self_wirings
    load_work = {}
    models[1].load_state_dict(trained, work=load_work)
    _assert_nested_equal(models[1].state_dict(), trained)
    assert models[1].reader._self_wirings is not old_wirings
    assert int(models[1].reader.assemblies.used) == 2
    accounting = models[1].reader.accounting()
    assert "maximum_address_comparator_neurons" not in accounting
    assert "maximum_address_winner_neurons" not in accounting
    assert accounting["factual_competition"]["checkpoint_tensor_bytes"] == 0
    _coincidence_qa_save(tmp_path / "tournament-recognition-parity-passed.json", {
        "status": "passed", "models": 2, "ingests": 2, "ingested_raw_bytes": 4,
        "ingested_eos": 2, "free_calls": 4,
        "free_decisions": sum(len(p.steps) for p in [*(r.payload for r in predictions), *later]),
        "teacher_calls": 4, "teacher_symbols": 8, "enabled_symbols": 4,
        "disabled_symbols": 4, "action_labels": 6, "children": 2,
        "same_instance_state_loads": 1, "checkpoint_file_loads": 0,
        "experimental_checkpoint_loads": 0, "executions": executions,
        "old_free_work": old_work, "new_free_work": new_work,
        "teacher_work": [feedback.work for feedback in corrections],
        "disabled_work": [feedback.work for feedback in disabled],
        "load_work": load_work, "accounting": accounting,
    })


def _persistent_checkpoint_fixture(root):
    """Parent-only fixed evidence; opaque payloads are never tensor-decoded."""
    import hashlib

    from experiments.r4_integrated.persistent_language_action_v1 import encode
    from experiments.r4_integrated.persistent_session_checkpoint import (
        COUNTERS,
        ROUTES,
        PersistentSessionCheckpoint,
    )

    root.mkdir()
    source = root / "source"
    source.mkdir()
    (source / "fixture.txt").write_bytes(b"fixed checkpoint fixture\n")
    evidence = root / "evidence"
    evidence.mkdir()
    journals = {}
    for kind in ("attempts", "teachers", "ingests"):
        name = kind + ".jsonl"
        (evidence / name).write_bytes(b"")
        journals[kind] = [name]
    manifest = root / "manifest.json"
    manifest.write_bytes(encode({"episodes": [
        {"episode_id": "fixture", "attempt_limit": 1, "append_document_ids": [],
         "teacher": {"kind": "edit", "operation": "SPLICE", "path": "main.py",
                     "start": 6, "end": 7, "replacement_hex": b"1".hex(),
                     "stop_reason": "eos"}}]}))
    store = PersistentSessionCheckpoint.create(
        root / "checkpoint", source_root=source, evidence_root=evidence,
        manifest_path=manifest, manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        source_revision="a" * 40,
        source_sha256={"fixture.txt": hashlib.sha256(
            (source / "fixture.txt").read_bytes()).hexdigest()}, session_id="ordinary-qa")
    host = {
        "cursor": {"episode_index": 0, "route_index": 0,
                   "attempt_index": 1, "append_done": False},
        "projects": {route: {} for route in ROUTES}, "current": None,
        "counters": {route: dict.fromkeys(COUNTERS, 0) for route in ROUTES},
        "elapsed_seconds": 0, "budget_seconds": 180, "budget_mib": 2048,
        "initial_bindings": {"fixture": hashlib.sha256(b"ordinary QA").hexdigest()},
    }
    return store, host, evidence, journals


def test_persistent_checkpoint_interrupted_frontiers(tmp_path):
    """Publication and dirty-tail custody without models or tensor decoding."""
    import hashlib

    from experiments.r4_integrated.persistent_language_action_v1 import encode
    from experiments.r4_integrated.persistent_session_checkpoint import (
        DirtyCheckpoint,
        PersistentSessionCheckpoint,
    )

    cases = ("clean", "intent", "partial_journal", "attempt_before_teacher",
             "payload", "manifest", "published_head", "source_change", "payload_corruption",
             "empty_execution_directory")
    records = []
    for case in cases:
        store, host, evidence, journals = _persistent_checkpoint_fixture(tmp_path / case)
        payload = b"opaque parent-only QA payload; never torch-loaded"
        payload_name = "route-fixture.pt"
        (store.directory / payload_name).write_bytes(payload)
        row = {"payload": payload_name, "sha256": hashlib.sha256(payload).hexdigest(),
               "bytes": len(payload), "model_sha256": "b" * 64, "memory_sha256": "c" * 64,
               "experience": 0, "reader_experience": 0, "mode": "learned", "seed": 733,
               "actor_class": "parent_only_opaque_fixture"}
        boundary = {
            "schema": "persistent-session-checkpoint-boundary-v1", "sequence": 0,
            "config_sha256": store.config_sha256, "previous_sha256": None,
            "intent_sha256": hashlib.sha256(
                (store.directory / "intent-000000000.json").read_bytes()).hexdigest(),
            "routes": {route: dict(row) for route in ("online", "learning_disabled")},
            "baseline": {"documents_hex": [], "demonstrations": [],
                         "new_teacher_offers": {"reading": 0, "STOP": 0, "SPLICE": 0}},
            "host": host,
            "frontiers": store.capture_frontiers(journals, execution_root="executions"),
            "work": {},
        }

        def publish(value, *, directory=store.directory):
            name = f"boundary-{value['sequence']:09d}.json"
            raw = encode(value)
            (directory / name).write_bytes(raw)
            (directory / "HEAD.json").write_bytes(encode({
                "schema": "persistent-session-checkpoint-head-v1", "sequence": value["sequence"],
                "manifest": name, "sha256": hashlib.sha256(raw).hexdigest()}))

        publish(boundary)
        store = PersistentSessionCheckpoint.open(
            store.directory, config_sha256=store.config_sha256)
        assert store.inspect()["status"] == "clean"
        if case in {"intent", "partial_journal", "attempt_before_teacher", "payload",
                    "manifest", "published_head"}:
            digest = store.begin("attempt", host["cursor"], observation_sha256="e" * 64)
            if case == "partial_journal":
                (evidence / "attempts.jsonl").write_bytes(b'{"interrupted":')
            elif case == "attempt_before_teacher":
                (evidence / "attempts.jsonl").write_bytes(encode({"attempt": "actual"}))
            elif case in {"payload", "manifest", "published_head"}:
                (store.directory / "route-uncommitted.pt").write_bytes(b"uncommitted")
                if case in {"manifest", "published_head"}:
                    successor = copy.deepcopy(boundary)
                    successor.update(sequence=1, intent_sha256=digest)
                    successor["host"]["cursor"]["route_index"] = 1
                    successor["previous_sha256"] = store.inspect(
                        require_clean=False)["boundary_sha256"]
                    if case == "manifest":
                        (store.directory / "boundary-000000001.json").write_bytes(encode(successor))
                    else:
                        publish(successor)
        elif case == "source_change":
            (tmp_path / case / "source" / "fixture.txt").write_bytes(b"changed\n")
        elif case == "payload_corruption":
            (store.directory / payload_name).write_bytes(b"changed payload")
        elif case == "empty_execution_directory":
            (evidence / "executions" / "unexplained-empty").mkdir(parents=True)
        reopened = PersistentSessionCheckpoint.open(
            store.directory, config_sha256=store.config_sha256) if case != "source_change" else None
        if case in {"clean", "published_head"}:
            assert reopened.inspect()["status"] == "clean"
        elif case == "source_change":
            with pytest.raises(ValueError, match="source changed"):
                PersistentSessionCheckpoint.open(store.directory, config_sha256=store.config_sha256)
        elif case == "payload_corruption":
            with pytest.raises(ValueError, match="payload changed"):
                reopened.inspect()
        else:
            with pytest.raises(DirtyCheckpoint):
                reopened.inspect()
            partial = reopened.inspect(require_clean=False)
            assert partial["status"] == "dirty" and partial["boundary"]["sequence"] == 0
        records.append({"case": case, "work": store.work})
    _coincidence_qa_save(tmp_path / "persistent-checkpoint-frontiers-passed.json", {
        "status": "passed", "cases": records, "models": 0, "teacher_symbols": 0,
        "tensor_loads": 0, "children": 0,
        "scope": "constructed parent-only publication fixtures; no neuronal continuation claim",
    })


def test_persistent_checkpoint_actual_actor_continuation(tmp_path):
    """Exact learned state, counters, actual files and independent restored actors."""
    import hashlib
    import time
    from pathlib import Path

    from experiments.r4_integrated.persistent_language_action_baseline_v2 import SimpleReference
    from experiments.r4_integrated.persistent_language_action_execution_v2 import (
        execute_proposal,
        feedback_from_receipt,
        qualified_launch,
        workspace_from_records,
    )
    from experiments.r4_integrated.persistent_language_action_v1 import encode
    from experiments.r4_integrated.persistent_session_checkpoint import PersistentSessionCheckpoint
    from humanbrain.neuronal_splice_learner import _digest
    from humanbrain.sequence_assembly_observation import RawObservation, SpliceProposal
    from humanbrain.tournament_recognition_learner import TournamentRecognitionSpliceLearner

    store, host, evidence, journals = _persistent_checkpoint_fixture(tmp_path / "actual")
    routes = ("online", "learning_disabled")
    models = {route: TournamentRecognitionSpliceLearner(seed=733) for route in routes}
    states = {}
    for route, model in models.items():
        with torch.no_grad():
            model.reader.address_weight.zero_()
            model.reader.prefix_weight.zero_()
            model.reader.output_weight.zero_()
            model.reader.codec.weight.zero_()
            model.reader.codec.bias.zero_()
            model.reader.codec.bias[256] = 10
            model.actions.operation_weight.zero_()
            model.actions.operation_bias.copy_(torch.tensor([10.0, 0.0]))
        states[route] = model.ingest([b"a"])
    baseline = SimpleReference()
    baseline.documents = [b"a"]
    # Transport an authentic-shaped prior demo directly; this is not new teaching.
    baseline.demonstrations = [{"query": b"prior raw request", "old": b"0", "replacement": b"1",
                                "source_sha256": hashlib.sha256(b"print(0)\n").hexdigest(),
                                "query_sha256": hashlib.sha256(b"prior raw request").hexdigest()}]
    baseline.new_teacher_offers = {"reading": 3, "STOP": 2, "SPLICE": 1}
    frontiers = store.capture_frontiers(journals, execution_root="executions")
    first = store.commit(models, states, baseline, host, frontiers)
    assert first["sequence"] == 0 and first["work"]["checkpoint_payload_writes"] == 2
    first_boundary = store.inspect()["boundary"]
    workspace = workspace_from_records([{"path": "main.py", "content_hex": b"print(0)\n".hex()}])
    observation = RawObservation(b"Change the printed value to one.", workspace)
    teacher = SpliceProposal("SPLICE", "main.py", 6, 7, b"1", "eos", workspace.sha256)
    observation_hash = hashlib.sha256(encode(observation.to_record())).hexdigest()
    executions, commits = [], []
    for route_index, route in enumerate(routes):
        store.begin("attempt", host["cursor"], observation_sha256=observation_hash)
        model, state = models[route], states[route]
        prediction = model.propose(observation, state)
        assert prediction.proposal.operation == "STOP" and len(prediction.payload.steps) == 1
        actual_files, execution = execute_proposal(
            workspace, prediction.proposal, evidence / "executions" / f"{route}-1",
            launch=qualified_launch(Path(__file__).resolve().parents[1]),
            deadline=time.perf_counter() + 10)
        executions.append(execution)
        assert execution["returncode"] == 0 and execution["stdout_hex"] == b"0\n".hex()
        envelope = {"attempt_id": prediction.attempt_id,
                    "proposal_sha256": prediction.proposal_sha256,
                    "workspace_sha256": workspace.sha256, "execution": execution}
        # Actual runner fields needed by the boundary contract, including the
        # actual observation, execution and pre-correction state. No fake outcome.
        row = {"schema": "persistent-language-action-attempt-checkpoint-qa-v1",
               "route": route, "episode_id": "fixture", "kind": "edit", "attempt_index": 1,
               "observation": observation.to_record(),
               "observation_record_sha256": observation_hash,
               "prediction": {"attempt_id": prediction.attempt_id},
               "execution": execution, "after_files": actual_files, "project_id": "actual-project",
               "model_after_sha256": _digest(model.state_dict()),
               "memory_after_sha256": _digest(vars(state))}
        raw_attempt = encode(row)
        with (evidence / "attempts.jsonl").open("ab") as handle:
            handle.write(raw_attempt)
        correction = model.teach(
            prediction, observation, state, teacher, envelope, learning_enabled=route == "online")
        assert correction.reader_feedback.recognition_updated == (route == "online")
        with (evidence / "teachers.jsonl").open("ab") as handle:
            handle.write(encode({
                "schema": "persistent-language-action-teacher-checkpoint-qa-v1", "route": route,
                "episode_id": "fixture", "attempt_sha256": hashlib.sha256(raw_attempt).hexdigest(),
                "teacher": store.manifest["episodes"][0]["teacher"],
                "learning_enabled": route == "online", "reader_offers": 2, "action_labels": 3,
                "model_after_sha256": _digest(model.state_dict()),
                "memory_after_sha256": _digest(vars(state))}))
        host["cursor"]["route_index"] = route_index + 1
        host["elapsed_seconds"] = route_index + 1
        host["counters"][route].update(attempts=1, executions=1, teacher_occasions=1,
                                       reader_offers=2, action_labels=3)
        feedback = feedback_from_receipt(execution).to_record()
        host["projects"][route]["actual-project"] = {
            "episode_id": "fixture", "files": actual_files, "feedback": feedback}
        host["current"] = {"route": route, "episode_id": "fixture",
                           "files": actual_files, "feedback": feedback}
        commits.append(store.commit(models, states, baseline, host,
                       store.capture_frontiers(journals, execution_root="executions")))
        if route == "online":
            second_boundary = store.inspect()["boundary"]
            assert (second_boundary["routes"]["online"]["payload"]
                    != first_boundary["routes"]["online"]["payload"])
            assert (second_boundary["routes"]["learning_disabled"]["payload"]
                    == first_boundary["routes"]["learning_disabled"]["payload"])
    second, third = commits
    assert second["sequence"] == 1 and third["sequence"] == 2
    final_boundary = store.inspect()["boundary"]
    assert third["work"]["checkpoint_payload_writes"] == 3
    assert third["work"]["checkpoint_payload_reuses"] == 3
    assert final_boundary["routes"]["learning_disabled"]["reader_experience"] > 0
    assert final_boundary["routes"]["learning_disabled"]["experience"] == 1
    fresh = {route: TournamentRecognitionSpliceLearner(seed=733) for route in routes}
    reopened = PersistentSessionCheckpoint.open(store.directory, config_sha256=store.config_sha256)
    restored = reopened.restore_into(fresh)
    assert restored["host"] == host
    assert restored["baseline"].documents == baseline.documents
    assert restored["baseline"].demonstrations == baseline.demonstrations
    assert restored["baseline"].new_teacher_offers == baseline.new_teacher_offers
    carried = restored["host"]["projects"]["online"]["actual-project"]
    next_observation = RawObservation(observation.request, workspace_from_records(carried["files"]),
                                      feedback_from_receipt(executions[0]))
    assert next_observation.workspace.sha256 == workspace.sha256
    comparisons = []
    for route in routes:
        _assert_nested_equal(models[route].state_dict(), fresh[route].state_dict())
        _assert_nested_equal(states[route].snapshot(), restored["states"][route].snapshot())
        assert models[route]._experience == fresh[route]._experience
        assert models[route].reader._active_experience == fresh[route].reader._active_experience
        # Empty tuples may be shared; independence concerns actual cached tensors.
        for original, loaded in zip(models[route].reader._candidate_views,
                                    fresh[route].reader._candidate_views, strict=True):
            for name in ("indices", "values", "flags"):
                left, right = getattr(original, name), getattr(loaded, name)
                if left.numel():
                    assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
        for original, loaded in zip(models[route].reader._self_wirings,
                                    fresh[route].reader._self_wirings, strict=True):
            for name in ("indices", "values"):
                left, right = getattr(original, name), getattr(loaded, name)
                if left.numel():
                    assert left.untyped_storage().data_ptr() != right.untyped_storage().data_ptr()
        old_prediction = models[route].reader.predict_with_receipt(
            next_observation, states[route], max_bytes=1)
        new_prediction = fresh[route].reader.predict_with_receipt(
            next_observation, restored["states"][route], max_bytes=1)
        assert old_prediction.generation == new_prediction.generation
        assert len(old_prediction.steps) == len(new_prediction.steps)
        for left, right in zip(old_prediction.steps, new_prediction.steps, strict=True):
            _assert_nested_equal(vars(left), vars(right))
        models[route].reader.acknowledge(old_prediction, states[route])
        fresh[route].reader.acknowledge(new_prediction, restored["states"][route])
        comparisons.append({"route": route, "steps": len(old_prediction.steps),
                            "generation": vars(old_prediction.generation)})
    _coincidence_qa_save(tmp_path / "persistent-checkpoint-continuation-passed.json", {
        "status": "passed", "models": 4, "ingests": 2, "source_raw_bytes": 2,
        "source_eos": 2, "free_calls": 6, "teacher_calls": 2, "teacher_symbols": 4,
        "enabled_teacher_symbols": 2, "disabled_teacher_symbols": 2,
        "action_labels": 6, "children": 2, "checkpoint_commits": 3,
        "restore_calls": 1, "tensor_loads": restored["work"]["checkpoint_torch_load_calls"],
        "experimental_checkpoint_loads": 0, "payload_writes": 3, "payload_reuses": 3,
        "executions": executions, "comparisons": comparisons,
        "commit_work": third["work"], "restore_work": restored["work"],
    })
