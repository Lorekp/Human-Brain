"""Convert a verified mini-swe teacher trajectory into cell-growth sequences."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from typing import Any

from humanbrain.swe_model import serialize_messages


@dataclass(frozen=True)
class SWEGrowthExample:
    prompt: str
    response: str
    command: str

    @property
    def sequence(self) -> str:
        return self.prompt + self.response


def growth_examples_from_trajectory(
    trajectory: dict[str, Any],
) -> tuple[SWEGrowthExample, ...]:
    """Rebuild the exact cell-agent history for every teacher action."""
    messages = trajectory.get("messages")
    if not isinstance(messages, list):
        raise ValueError("teacher trajectory must contain a messages list")

    runtime_messages: list[dict[str, Any]] = []
    examples: list[SWEGrowthExample] = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("teacher messages must be objects")
        role = str(message.get("role", ""))
        if role == "exit":
            break
        if role != "assistant":
            runtime_messages.append(deepcopy(message))
            continue

        actions = message.get("extra", {}).get("actions", [])
        if len(actions) != 1 or not isinstance(actions[0], dict):
            raise ValueError("each teacher response must contain exactly one action")
        command = actions[0].get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("teacher action must contain a non-empty command")

        thought = str(message.get("content", "")).rstrip()
        action_json = json.dumps(
            {"command": command},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        response = f"{thought}\n<bash>{action_json}</bash>"
        prompt = serialize_messages(runtime_messages)
        examples.append(
            SWEGrowthExample(prompt=prompt, response=response, command=command)
        )
        runtime_messages.append(
            {
                "role": "assistant",
                "content": response,
                "extra": deepcopy(message.get("extra", {})),
            }
        )

    if not examples:
        raise ValueError("teacher trajectory contains no actions")
    return tuple(examples)
