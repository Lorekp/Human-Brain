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
    assert len(teacher_queries) == 1 and held_codec_calls == [4, 4, 4]
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
    assert feedback.work["codec_slot_evaluations"] == 12
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
    assert cap["work"]["codec_slot_evaluations"] == 3 * 128
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
