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
