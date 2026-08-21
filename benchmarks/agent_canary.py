"""Train and run one exact mini-swe-agent protocol canary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanbrain.language import CellLanguageModel, ModelConfig
from humanbrain.swe_model import HumanCellModel, serialize_messages
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment


ROOT = Path(__file__).resolve().parents[1]
SYSTEM = "You are a helpful software engineer."
TASK = "Inspect the repository files."
EXPECTED_COMMAND = "echo HUMAN_CELL_AGENT_CANARY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "artifacts" / "agent_canary.npz",
    )
    args = parser.parse_args()
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TASK},
    ]
    prompt = serialize_messages(messages)
    response = (
        "[THOUGHT]\nI will inspect the files.\n"
        f'<bash>{{"command":"{EXPECTED_COMMAND}"}}</bash>'
    )
    model = CellLanguageModel.grow(
        [prompt + response],
        ModelConfig(max_order=64),
    )
    checkpoint_sha256 = model.save(args.checkpoint)
    adapter = HumanCellModel(
        checkpoint=args.checkpoint,
        max_new_tokens=len(response),
    )
    try:
        environment = LocalEnvironment(cwd=str(ROOT), timeout=5)
        agent = DefaultAgent(
            adapter,
            environment,
            system_template=SYSTEM,
            instance_template="{{task}}",
            step_limit=1,
            cost_limit=0.0,
        )
        agent.extra_template_vars = {"task": TASK}
        agent.add_messages(
            adapter.format_message(role="system", content=SYSTEM),
            adapter.format_message(role="user", content=TASK),
        )
        observations = agent.step()
        message = agent.messages[-2]
        commands = [action["command"] for action in message["extra"]["actions"]]
        observed = "\n".join(str(item.get("content", "")) for item in observations)
        output = {
            "pass": commands == [EXPECTED_COMMAND] and "HUMAN_CELL_AGENT_CANARY" in observed,
            "commands": commands,
            "content": message["content"],
            "observation": observed,
            "mini_swe_agent_calls": agent.n_calls,
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        }
    except Exception as error:
        output = {
            "pass": False,
            "error": f"{type(error).__name__}: {error}",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": checkpoint_sha256,
        }
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
