import json

from humanbrain.swe_growth import growth_examples_from_trajectory


def test_teacher_trajectory_becomes_exact_runtime_cell_prompts() -> None:
    trajectory = {
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {
                "role": "assistant",
                "content": "[THOUGHT]\nInspect.",
                "extra": {"actions": [{"command": "sed -n '1,5p' file.py"}]},
            },
            {"role": "user", "content": "<output>source</output>"},
            {
                "role": "assistant",
                "content": "[THOUGHT]\nEdit.",
                "extra": {"actions": [{"command": "printf 'x\\ny'"}]},
            },
            {"role": "exit", "content": "done"},
        ]
    }

    examples = growth_examples_from_trajectory(trajectory)

    assert len(examples) == 2
    assert examples[0].command == "sed -n '1,5p' file.py"
    assert json.loads(
        examples[0].response.split("<bash>", 1)[1].split("</bash>", 1)[0]
    ) == {"command": "sed -n '1,5p' file.py"}
    assert examples[0].response in examples[1].prompt
    assert "<output>source</output>" in examples[1].prompt
