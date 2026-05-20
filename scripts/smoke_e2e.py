"""End-to-end smoke test for Wattle's provider plugins.

For each provider plugin, runs the agent loop with a minimal task that
exercises the Bash/Write tools, then verifies the resulting filesystem state.
Reads API keys from ~/.wattle/auth.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

from wattle import run_agent

SYSTEM = (
    "You are a coding assistant with access to file-system tools. "
    "Use them to complete the user's task, then briefly confirm what you did."
)

CASES = [
    ("anthropic", "claude-haiku-4-5-20251001"),
    ("deepseek", "deepseek-v4-flash"),
    ("kimi", "kimi-k2.6"),
    ("minimax", "MiniMax-M2.7"),
    ("openai_completions", "gpt-4.1-mini"),
    ("openai_responses", "gpt-4.1-mini"),
]


def run_case(provider_name: str, model: str) -> bool:
    folder = Path(f"./test_{provider_name}_agent")
    user_prompt = (
        f"Create the folder {folder} (if it doesn't exist) and put a single "
        ".txt file inside it containing a one-line random sentence."
    )

    print(f"\n=== {provider_name} ({model}) ===", flush=True)
    response = run_agent(provider_name, model, user_prompt, system=SYSTEM)

    print(f"stop_reason: {response.stop_reason}")
    for block in response.content:
        if block.type == "text":
            print(f"assistant: {block.text}")

    if not folder.is_dir():
        print(f"FAIL: {folder} was not created")
        return False
    files = [f for f in folder.iterdir() if f.is_file()]
    if not files:
        print(f"FAIL: {folder} is empty")
        return False
    print(f"PASS: {folder}/ contains {[f.name for f in files]}")
    return True


def main() -> int:
    results = [(p, run_case(p, m)) for p, m in CASES]
    print("\n=== summary ===")
    for p, ok in results:
        print(f"  {p}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(ok for _, ok in results) else 1


if __name__ == "__main__":
    sys.exit(main())
