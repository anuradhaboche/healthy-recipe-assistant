"""The agent loop, shared by every step that gives Claude tools.

    send conversation + tools  ->  Claude replies
    reply has tool calls?      ->  run them, append results, go again
    no tool calls?             ->  Claude is done: return its structured answer

Step 2 introduced this loop inside nutrition.py. Step 3 needs the same thing
with different tools, so it lives here now.
"""

import json
from typing import Callable, TypeVar

import anthropic
from pydantic import BaseModel

from .vision import MODEL, RefusalError, ask_claude

T = TypeVar("T", bound=BaseModel)


def run_agent(
    client: anthropic.Anthropic,
    *,
    system: str,
    tools: list[dict],
    run_tool: Callable[[str, dict], str],
    output_format: type[T],
    first_message: str,
    effort: str = "low",
    max_turns: int = 25,
    trace: Callable[[str], None] = lambda _: None,
) -> T:
    messages: list = [{"role": "user", "content": first_message}]

    for turn in range(1, max_turns + 1):
        response = ask_claude(
            client,
            model=MODEL,
            max_tokens=16000,
            system=system,
            tools=tools,
            output_config={"effort": effort},
            output_format=output_format,
            messages=messages,
        )

        if response.stop_reason == "refusal":
            raise RefusalError("Claude declined the request")

        # Always append the full content (including thinking blocks) so the
        # next request carries Claude's reasoning forward.
        messages.append({"role": "assistant", "content": response.content})

        tool_calls = [b for b in response.content if b.type == "tool_use"]
        if not tool_calls:
            if response.parsed_output is None:
                raise RuntimeError(f"No structured result (stop_reason={response.stop_reason})")
            trace(f"turn {turn}: done")
            return response.parsed_output

        results = []
        for call in tool_calls:
            trace(f"turn {turn}: {call.name}({json.dumps(call.input)})")
            try:
                content, is_error = run_tool(call.name, call.input), False
            except Exception as e:  # tell Claude what went wrong instead of crashing
                content, is_error = f"Error: {e}", True
                trace(f"  -> error: {e}")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error})
        # All results for one turn go back in a single user message.
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Agent did not finish within {max_turns} turns")
