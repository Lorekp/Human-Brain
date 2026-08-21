"""Verify Gemma native tool calls through mini-swe-agent's production model class."""

from __future__ import annotations

import argparse
import json
import time

from minisweagent.models.litellm_model import LitellmModel


EXPECTED_COMMAND = "echo GEMMA_MINI_SWE_CANARY"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-base", default="http://127.0.0.1:8081/v1")
    args = parser.parse_args()

    model = LitellmModel(
        model_name="openai/gemma-4-12B-it-qat-q4_0",
        cost_tracking="ignore_errors",
        model_kwargs={
            "api_base": args.api_base,
            "api_key": "none",
            "custom_llm_provider": "openai",
            "drop_params": True,
            "parallel_tool_calls": True,
            "max_tokens": 4096,
            "temperature": 1.0,
            "top_p": 0.95,
            "seed": 42,
            "extra_body": {"top_k": 64},
        },
    )
    started = time.perf_counter()
    result = model.query(
        [
            {
                "role": "system",
                "content": "You are a coding agent. Use the provided bash tool for shell actions.",
            },
            {
                "role": "user",
                "content": (
                    "Call bash exactly once with command: "
                    f"{EXPECTED_COMMAND}. Do not explain."
                ),
            },
        ]
    )
    elapsed = time.perf_counter() - started
    actions = result["extra"]["actions"]
    commands = [str(action.get("command", "")).strip() for action in actions]
    output = {
        "pass": commands == [EXPECTED_COMMAND],
        "actions": actions,
        "commands": commands,
        "elapsed_seconds": elapsed,
        "finish_reason": result["extra"]["response"]["choices"][0]["finish_reason"],
        "usage": result["extra"]["response"].get("usage"),
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    if not output["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
