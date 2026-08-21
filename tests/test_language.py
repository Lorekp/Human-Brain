from humanbrain import CellLanguageModel, ModelConfig


CORPUS = (
    "the membrane predicts the next word.\n"
    "the membrane predicts the next word.\n"
    "the cortex predicts the next token.\n"
)


def test_predict_next_is_a_spike_selected_token() -> None:
    model = CellLanguageModel.grow([CORPUS], ModelConfig(max_order=8))
    prediction = model.predict_next("the membrane predicts the next ")
    assert prediction.text == "word"
    assert prediction.spike_time_ms > 0.0
    assert prediction.active_cells >= 1


def test_autoregressive_completion_uses_its_own_emitted_tokens() -> None:
    model = CellLanguageModel.grow([CORPUS])
    text = model.complete("the membrane ", max_new_tokens=8)
    assert text.startswith("predicts")
    assert "word" in text


def test_completion_stops_after_requested_text() -> None:
    model = CellLanguageModel.grow(["prompt first STOP ignored"])
    assert model.complete("prompt", max_new_tokens=16, stop_text="STOP") == (
        " first STOP"
    )


def test_model_grows_a_separate_cell_vocabulary() -> None:
    model = CellLanguageModel.grow([CORPUS])
    assert model.vocabulary_size >= 256
    assert model.grown_branches > 0


def test_held_out_next_token_evaluation_uses_cell_predictions() -> None:
    model = CellLanguageModel.grow([CORPUS])
    result = model.evaluate("the membrane predicts the next word.\n")
    assert result.predictions > 4
    assert result.accuracy > 0.5
    assert result.tokens_per_second > 0.0


def test_checkpoint_round_trip_preserves_cell_prediction(tmp_path) -> None:
    model = CellLanguageModel.grow([CORPUS])
    before = model.predict_next("the membrane predicts the next ")
    path = tmp_path / "cell-model.npz"
    artifact_sha = model.save(path)
    restored = CellLanguageModel.load(path)
    after = restored.predict_next("the membrane predicts the next ")
    assert len(artifact_sha) == 64
    assert restored.training_digest == model.training_digest
    assert restored.grown_branches == model.grown_branches
    assert after == before
