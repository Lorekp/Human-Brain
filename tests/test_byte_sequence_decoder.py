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
        if mode in {"writer_maps_frozen", "frozen"}:
            assert all(torch.count_nonzero(credit) == 0 for credit in state.credit.values())
        else:
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


@pytest.fixture(params=(2, 3, 4, 5))
def local_cost_runner(request):
    if request.param == 2:
        from experiments.r4_integrated import local_memory_byte_v2 as runner

        return runner, "experiments.r4_integrated.local_memory_byte_v2"
    from functools import partial
    from types import SimpleNamespace

    from experiments.r4_integrated import local_memory_byte_harness as harness
    if request.param == 3:
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
        "status": "completed", "mode": "cost", "config": runner.CONFIG,
        "selected_lesson_indices": [3, 4, 5, 6],
        "lessons": [
            {"arm": arm, "index": index, "seconds": 0.01}
            for index in [3, 4, 5, 6] for arm in runner.ARMS
        ],
        "cap_generation": {
            arm: {"seconds": 0.02, "stop_reason": "limit", "decisions": 33,
                  "generated_hex": bytes(32).hex()} for arm in runner.ARMS
        },
        "actions": [{}] * 100, "motor_demonstrations": [{}, {}],
        "counts": {"first_attempts": 20, "teacher_byte_offers": 200,
                   "enabled_decoder_step_opportunities": 160},
        "memory": {"peak_working_set_mib": 300},
        "worker": {"memory": {"peak_working_set_mib": 400}},
    }
    launch = {"elapsed_seconds": 10.0, "observed_tree_peak_sum_mib": 750}
    if hasattr(runner, "duration_clock"):
        report["duration_clock"] = runner.duration_clock()
    result = runner.project_cost(report, launch)
    assert result["projected_seconds"] == pytest.approx(2 * 1024 * 5 * 0.03 + 2 * 9.8 + 30)
    assert result["peak_upper_mib"] == 750 and result["admissible"]
    report["cap_generation"]["frozen"]["seconds"] = 0.0
    with pytest.raises(AssertionError):
        runner.project_cost(report, launch)
    report["cap_generation"]["frozen"]["seconds"] = 0.02
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
    candidate = harness.new_model(SPEC, "tagged")
    baseline = previous.new_model("tagged")
    _assert_nested_equal(candidate.state_dict(), baseline.state_dict())


def test_local_harness_capped_prediction_measures_subtick_duration(monkeypatch):
    from experiments.r4_integrated import local_memory_byte_harness as harness
    from experiments.r4_integrated.local_memory_byte_v4 import SPEC

    model = harness.new_model(SPEC, "frozen")
    monkeypatch.setattr(harness.time, "monotonic", lambda: 1000.0)
    ticks = iter((2000.0, 2000.003))
    monkeypatch.setattr(harness.time, "perf_counter", lambda: next(ticks))
    with torch.no_grad():
        sample = harness.capped_prediction(SPEC, "frozen", model)
    assert sample["decisions"] == 33 and sample["stop_reason"] == "limit"
    assert sample["seconds"] == pytest.approx(0.003)


def test_local_harness_export_uses_same_high_resolution_clock(tmp_path, monkeypatch):
    from experiments.r4_integrated import local_memory_byte_harness as harness
    from humanbrain.local_memory_learner import LocalMemoryLearner

    model = LocalMemoryLearner(
        mode="frozen", embedding_size=2, hidden_size=3, key_size=2, value_size=3
    )
    checks = [
        {"id": index, "memory_id": str(index), "base_memory_id": None,
         "document_hex": [b"x".hex()], "query_hex": b"q".hex(), "delay_reads": 0}
        for index in range(8)
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
    from humanbrain.local_memory_learner import LocalMemoryLearner

    candidate = LocalMemoryLearner(
        mode=mode, embedding_size=2, hidden_size=3, key_size=2, value_size=3, seed=19
    ).to(dtype)
    reference = copy.deepcopy(candidate)
    for documents, query, target in (([b"x=a", b"y=b"], b"x?", b"a"),
                                     ([b"x=b", b"y=a"], b"y?", b"a!")):
        state = candidate.ingest(documents)
        baseline = reference.ingest(documents)
        before = candidate.snapshot(state)
        with patch.object(
            candidate.query_cell, "local_step", side_effect=AssertionError("no traces")
        ):
            actual_context = candidate._prediction_context(query, state)
            actual = candidate.predict(query, state, 3)
        with torch.no_grad():
            context, _, _ = reference._query(query, baseline)
            expected = reference.decoder.generate(context, 3)[0]
        assert torch.equal(actual_context, context)
        assert actual == expected
        _assert_nested_equal(before, candidate.snapshot(state))
        feedback = candidate.teach(query, target, state, 0.001)
        expected_feedback = reference.teach(query, target, baseline, 0.001)
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
    from humanbrain.local_memory_learner import LocalMemoryLearner

    # Authored transport witness: this tests causal routing, not learned capability.
    learner = LocalMemoryLearner(embedding_size=1, hidden_size=1, key_size=1, value_size=1)
    with torch.no_grad():
        learner.query_cell.weight.zero_()
        learner.query_cell.recurrent.zero_()
        learner.query_cell.bias.fill_(1)
        learner.memory.query_weight.fill_(1)
        learner.memory.query_bias.zero_()
        learner.decoder.embedding.weight.zero_()
        learner.decoder.cell.weight.zero_()
        learner.decoder.cell.weight[0, 1] = 1
        learner.decoder.cell.recurrent.zero_()
        learner.decoder.cell.bias.zero_()
        learner.decoder.output.weight.zero_()
        learner.decoder.output.bias.fill_(-10)
        learner.decoder.output.bias[65:67] = 0
        learner.decoder.output.weight[65, 0] = 1
        learner.decoder.output.weight[66, 0] = -1
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
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(embedding_size=1, hidden_size=1, key_size=1, value_size=1)
    state = learner.memory.initial_local_state(1)
    with torch.no_grad():
        if corrupt == "synapses":
            state.synapses.fill_(torch.nan)
        elif corrupt == "credit":
            next(iter(state.credit.values())).fill_(torch.nan)
        elif corrupt == "cell":
            state.cell.activity.fill_(torch.nan)
        elif corrupt == "input":
            learner.decoder.embedding.weight[ord("q")].fill_(torch.nan)
        elif corrupt == "activity":
            learner.query_cell.bias.fill_(torch.nan)
        elif corrupt == "read":
            learner.memory.query_weight.fill_(torch.nan)
        else:
            learner.decoder.output.bias.fill_(torch.nan)
    before = digest(learner.snapshot(state))
    with pytest.raises(ValueError, match="finite"):
        learner.predict(b"q", state, 1)
    assert digest(learner.snapshot(state)) == before


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_local_forward_prediction_omits_unused_eligibility_overflow_but_teaching_rejects_it(dtype):
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(
        embedding_size=1, hidden_size=1, key_size=1, value_size=1
    ).to(dtype)
    with torch.no_grad():
        learner.decoder.embedding.weight.zero_()
        learner.query_cell.weight.zero_()
        learner.query_cell.bias.zero_()
        learner.query_cell.recurrent.fill_(3)
    state = learner.memory.initial_local_state(1)
    before = learner.snapshot(state)
    # h stays zero while unused bias eligibility doubles per step and overflows.
    query = b"x" * (130 if dtype == torch.float32 else 1030)
    assert isinstance(learner.predict(query, state, 1), ByteGeneration)
    _assert_nested_equal(before, learner.snapshot(state))
    with pytest.raises(ValueError, match="finite"):
        learner.teach(query, b"a", state, 0.001)
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
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(
        mode=mode, embedding_size=4, hidden_size=6, key_size=4, value_size=6, seed=39
    ).to(dtype=dtype)
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
        assert learner.predict(b"which?", state, 7) == reference.predict(
            b"which?", reference_state, 7
        )
        feedback = learner.teach(b"which?", b"blue", state, 0.003)
        reference_feedback = reference.teach(b"which?", b"blue", reference_state, 0.003)
        assert feedback.mean_loss == reference_feedback.mean_loss
        assert torch.equal(feedback.mean_context_signal, reference_feedback.mean_context_signal)
        assert feedback.teacher_count == reference_feedback.teacher_count == 5
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
    from humanbrain.local_memory_learner import LocalMemoryLearner

    learner = LocalMemoryLearner(
        mode="writer_maps_frozen", embedding_size=2, hidden_size=3, key_size=2, value_size=3
    )
    state = learner.ingest([b"a"])
    learner.memory.key_weight.requires_grad_(True)
    before = copy.deepcopy(learner.state_dict())
    with pytest.raises(ValueError, match="remain frozen"):
        learner.ingest([b"b"], state)
    with pytest.raises(ValueError, match="remain frozen"):
        learner.teach(b"q", b"a", state, 0.01)
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


def test_direct_context_manifest_binds_seed_rule_and_rejects_wrong_source_before_allocation(tmp_path):
    from experiments.r4_integrated import local_context_use_v1 as runner

    sources = runner.sources()
    assert "src/humanbrain/local_memory_learner.py" in sources
    assert "experiments/r4_integrated/local_context_use_v1.md" in sources
    assert "experiments/r4_integrated/run_source_absent_memory_byte_v1.ps1" in sources
    output = tmp_path / "no-run"
    with patch.object(runner.subprocess, "check_output", return_value="revision\n"), patch.object(
        runner, "new_decoder", side_effect=AssertionError("must not allocate")
    ), pytest.raises(ValueError, match="source manifest"):
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
    with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")), patch.object(
        torch.autograd, "grad", side_effect=AssertionError("no reverse gradients")
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

    assert runner.ARMS == tuple(f"{mode}_{condition}" for mode in ("random", "symmetric")
                               for condition in ("correct", "zero", "shuffled", "frozen"))
    assert runner.schedule() == previous.schedule()
    assert runner.lesson is previous.lesson and runner.generation is previous.generation
    for key, value in previous.CONFIG.items():
        if key != "schema":
            assert runner.CONFIG[key] == value
    states = {arm: runner.new_decoder(mode).state_dict()
              for arm, (mode, _) in runner.ARM_SPECS.items()}
    assert runner.matched_initial_tensors(states)
    baseline = previous.new_decoder().state_dict()
    assert all(torch.equal(value, states["symmetric_correct"][key])
               for key, value in baseline.items() if isinstance(value, torch.Tensor))
    assert states["symmetric_correct"]["_extra_state"]["hidden_feedback_mode"] == "symmetric"
    states["symmetric_correct"]["output.bias"][0] += 1
    assert not runner.matched_initial_tensors(states)
    scores = {arm: {"train": 0, "heldout": 0} for arm in runner.ARMS}
    scores["symmetric_correct"] = {"train": 12, "heldout": 4}
    assert runner.qualifies(scores)
    for comparator in ("random_correct", "symmetric_zero", "symmetric_shuffled", "symmetric_frozen"):
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
    expected_arms = ("decoder_lexical", "decoder_permuted", "reference_lexical",
                     "reference_permuted", "reference_zero", "reference_frozen")
    assert runner.ARMS == expected_arms
    rng = random.Random(734)
    for cycle in range(64):
        permuted = list(runner.TRAIN)
        rng.shuffle(permuted)
        group = rows[cycle * 12:(cycle + 1) * 12]
        assert [row["permuted_target"] for row in group] == permuted
        assert [row["lexical_target"] for row in group] == list(runner.TRAIN)
        assert Counter(permuted) == Counter(runner.TRAIN)
        assert set(permuted).isdisjoint(runner.HELDOUT)
        for row in group:
            assert runner.assignment("decoder_permuted", row) == runner.assignment("reference_permuted", row)
            target, supplied = runner.assignment("reference_zero", row)
            assert target == row["lexical_target"] and supplied is None
    fake = [{"teacher_losses": [int(index % 12 == 0 and index // 12 < 54), 0, 0]}
            for index in range(768)]
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
    assert runner.same_state(decoder.state_dict(), runner.new_model("decoder_permuted").state_dict())
    initial = runner.new_model("reference_lexical").state_dict()
    for arm in ("reference_permuted", "reference_zero", "reference_frozen"):
        assert runner.same_state(initial, runner.new_model(arm).state_dict())
    fixture = [{"loss_kind": "reference_teacher_mistake", "first": {"exact": False, "stop_reason": "limit"},
                "mean_loss": 1 / 3, "teacher_losses": [1, 0, 0]}]
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
    with patch.object(torch.autograd, "backward", side_effect=AssertionError("no backward")), patch.object(
        torch.autograd, "grad", side_effect=AssertionError("no gradients")
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
    assert model.accounting()["serialized_tensor_bytes"] == sum(
        value.numel() * value.element_size() for value in model.buffers()
    ) == 302652
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
    with pytest.raises(ValueError, match="0.01 tick"):
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
