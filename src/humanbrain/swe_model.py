"""mini-swe-agent v2 model adapter for a frozen human-cell circuit."""

from __future__ import annotations

import json
from pathlib import Path
import re
import time
from typing import Any

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from humanbrain.language import CellLanguageModel
from minisweagent.exceptions import FormatError
from minisweagent.models.utils.actions_text import format_observation_messages
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content


DEFAULT_ACTION_REGEX = r"<bash>\s*\{\s*\"command\"\s*:\s*\"((?:\\.|[^\"\\])*)\"\s*\}\s*</bash>"


class HumanCellModelConfig(BaseModel):
    model_name: str = "human-cell-dendritic"
    checkpoint: Path
    max_new_tokens: int = 256
    action_regex: str = DEFAULT_ACTION_REGEX
    format_error_template: str = "Cell output did not contain exactly one valid <bash> action: {{ error }}"
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    multimodal_regex: str = ""


def serialize_messages(messages: list[dict[str, Any]]) -> str:
    """Create the fixed transcript ingested as currents by the cell tokenizer."""
    chunks: list[str] = []
    for message in messages:
        role = str(message.get("role", "user"))
        content = message.get("content", "")
        if isinstance(content, list):
            content = "\n".join(
                str(part.get("text", "")) if isinstance(part, dict) else str(part)
                for part in content
            )
        chunks.append(f"<message role=\"{role}\">\n{content}\n</message>\n")
    chunks.append('<message role="assistant">\n')
    return "".join(chunks)


class HumanCellModel:
    """Duck-typed mini-swe-agent model whose query is a cell simulation."""

    def __init__(self, **kwargs) -> None:
        self.config = HumanCellModelConfig(**kwargs)
        self._model = CellLanguageModel.load(self.config.checkpoint)
        self._calls = 0

    @property
    def n_calls(self) -> int:
        return self._calls

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        transcript = serialize_messages(messages)
        generated = self._model.complete(
            transcript,
            max_new_tokens=int(kwargs.get("max_new_tokens", self.config.max_new_tokens)),
            stop_text="</bash>",
        )
        matches = re.findall(self.config.action_regex, generated, re.DOTALL)
        if len(matches) != 1:
            error = f"expected exactly one action, found {len(matches)}"
            content = Template(
                self.config.format_error_template,
                undefined=StrictUndefined,
            ).render(error=error, actions=matches)
            raise FormatError(
                {
                    "role": "user",
                    "content": content,
                    "extra": {
                        "interrupt_type": "FormatError",
                        "n_actions": len(matches),
                        "model_response": generated,
                    },
                }
            )
        command = json.loads(f'"{matches[0]}"')
        self._calls += 1
        return {
            "role": "assistant",
            "content": generated,
            "extra": {
                "actions": [{"command": command}],
                "cost": 0.0,
                "timestamp": time.time(),
                "cell_checkpoint": str(self.config.checkpoint.resolve()),
            },
        }

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self,
        message: dict,
        outputs: list[dict],
        template_vars: dict | None = None,
    ) -> list[dict]:
        return format_observation_messages(
            outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }
