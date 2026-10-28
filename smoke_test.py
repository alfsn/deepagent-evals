"""Smoke test: checks the API key, model names, tool calling and structured output.

    uv run python smoke_test.py                 # default candidate models
    uv run python smoke_test.py kimi-k3 gemma4:31b
"""

import sys
import time

from langchain_core.tools import tool
from pydantic import BaseModel

from llm import get_llm

CANDIDATES = ["gemma4:31b", "kimi-k3", "deepseek-v4.1-flash"]


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


class Capital(BaseModel):
    country: str
    capital: str


def check(model: str) -> dict:
    row = {"model": model}
    llm = get_llm(model)

    t = time.perf_counter()
    try:
        llm.invoke("Reply with exactly: pong")
        row["chat"] = f"ok {time.perf_counter() - t:.1f}s"
    except Exception as e:
        row["chat"] = f"FAIL {type(e).__name__}: {str(e)[:80]}"
        return row

    try:
        msg = llm.bind_tools([multiply]).invoke("What is 37 * 91? Use the tool.")
        calls = [c["name"] for c in msg.tool_calls]
        row["tools"] = "ok" if calls == ["multiply"] else f"no tool call ({calls})"
    except Exception as e:
        row["tools"] = f"FAIL {type(e).__name__}"

    try:
        out = llm.with_structured_output(Capital, method="function_calling").invoke("Capital of Peru?")
        row["structured"] = "ok" if out and out.capital.lower() == "lima" else f"wrong: {out}"
    except Exception as e:
        row["structured"] = f"FAIL {type(e).__name__}"
    return row


if __name__ == "__main__":
    for m in sys.argv[1:] or CANDIDATES:
        r = check(m)
        print(f"{r['model']:<24} chat={r.get('chat')}  tools={r.get('tools', '-')}  structured={r.get('structured', '-')}")
