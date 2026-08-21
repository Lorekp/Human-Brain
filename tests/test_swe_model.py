from humanbrain import swe_model


class FakeCellModel:
    def complete(
        self, prompt: str, max_new_tokens: int, stop_text: str | None = None
    ) -> str:
        assert '<message role="user">' in prompt
        assert max_new_tokens == 64
        assert stop_text == "</bash>"
        return '[THOUGHT]\nInspect files.\n<bash>{"command":"find . -maxdepth 2 -type f"}</bash>'


def test_mini_swe_agent_adapter_returns_one_bash_action(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "cell.npz"
    checkpoint.touch()
    monkeypatch.setattr(
        swe_model.CellLanguageModel,
        "load",
        lambda path: FakeCellModel(),
    )
    model = swe_model.HumanCellModel(checkpoint=checkpoint, max_new_tokens=64)
    result = model.query([{"role": "user", "content": "Inspect the repository."}])
    assert result["extra"]["actions"] == [
        {"command": "find . -maxdepth 2 -type f"}
    ]
    assert model.n_calls == 1


def test_mini_swe_agent_adapter_formats_shell_observation(monkeypatch, tmp_path) -> None:
    checkpoint = tmp_path / "cell.npz"
    checkpoint.touch()
    monkeypatch.setattr(
        swe_model.CellLanguageModel,
        "load",
        lambda path: FakeCellModel(),
    )
    model = swe_model.HumanCellModel(checkpoint=checkpoint)
    messages = model.format_observation_messages(
        {},
        [{"returncode": 0, "output": "file.py", "exception_info": None}],
    )
    assert messages[0]["role"] == "user"
    assert "<returncode>0</returncode>" in messages[0]["content"]
    assert "file.py" in messages[0]["content"]
