"""Step 2: match each ingredient to real USDA nutrition data.

Agent concept: tool use and the agent loop.

Step 1 was one call: photo in, data out. Here Claude gets *tools* (functions
we wrote) and decides for itself which to call and with what arguments:
search for "greek yogurt", look at the matches, maybe search again for
"yogurt, greek, plain, nonfat", then pick one. We run each tool call, hand the
result back, and loop until Claude says it's done.

The loop below is written out by hand on purpose so you can see every step.
(The SDK also has a tool runner that does this for you; we'll switch to it
once the pattern is familiar.)

Grounding: Claude only decides *which* USDA food matches and *how many grams*.
The protein and calorie numbers are computed in Python from USDA data, so the
model can't make them up or get the arithmetic wrong.
"""

import json
from typing import Callable

import anthropic
from pydantic import BaseModel, Field

from . import usda
from .models import FridgeInventory
from .vision import MODEL, RefusalError, ask_claude, make_client

MAX_TURNS = 25

SYSTEM_PROMPT = """You match a list of fridge ingredients to entries in the USDA FoodData Central database.

For each ingredient:
1. Call search_foods with a short, plain query (e.g. "egg whole raw", "chicken breast raw").
   If no result fits, try a different wording once or twice.
2. Pick the closest match. Prefer raw/plain versions unless the item is clearly cooked or prepared.
3. Convert the quantity to grams using typical sizes (1 large egg ≈ 50 g, 1 chicken breast ≈ 175 g,
   half a bag of spinach ≈ 150 g). If the quantity is 'unknown', estimate a typical household amount
   and say so in the note.

Skip condiments, drinks and anything under ~10 g that doesn't matter for protein.
Search for several ingredients at once when you can. When every ingredient is matched or skipped,
return the final result."""

TOOLS = [
    {
        "name": "search_foods",
        "description": (
            "Search USDA FoodData Central for generic foods. Returns up to 5 matches, each with "
            "fdc_id, description and protein/fat/carbs/calories per 100 g."
        ),
        "strict": True,
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Plain food name, e.g. 'greek yogurt plain'"}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
]


class MatchedIngredient(BaseModel):
    ingredient: str = Field(description="Name from the input list")
    fdc_id: int | None = Field(description="Chosen USDA fdc_id, or null if skipped / no match")
    grams: float = Field(description="Estimated amount in grams")
    note: str | None = Field(default=None, description="Assumptions, e.g. 'assumed 1 bag = 300 g'")


class MatchResult(BaseModel):
    items: list[MatchedIngredient]


class NutritionLine(BaseModel):
    ingredient: str
    matched_food: str | None
    fdc_id: int | None
    grams: float
    protein_g: float | None
    calories_kcal: float | None
    carbs_g: float | None
    fat_g: float | None
    note: str | None = None


def _run_tool(name: str, args: dict) -> str:
    if name == "search_foods":
        return json.dumps(usda.search_foods(args["query"]))
    raise ValueError(f"Unknown tool {name}")


def _scale(per_100g: dict, grams: float) -> dict:
    return {k: (round(v * grams / 100, 1) if v is not None else None) for k, v in per_100g.items()}


def match_ingredients(
    inventory: FridgeInventory,
    client: anthropic.Anthropic | None = None,
    trace: Callable[[str], None] = lambda _: None,
) -> MatchResult:
    """The agent loop: let Claude search USDA until every ingredient is matched."""
    client = client or make_client()
    lines = [f"- {i.name}: {i.quantity}" for i in inventory.ingredients]
    messages: list = [{"role": "user", "content": "Ingredients:\n" + "\n".join(lines)}]

    for turn in range(1, MAX_TURNS + 1):
        response = ask_claude(
            client,
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            output_config={"effort": "low"},
            output_format=MatchResult,
            messages=messages,
        )

        if response.stop_reason == "refusal":
            raise RefusalError("Claude declined the nutrition lookup")

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
                content, is_error = _run_tool(call.name, call.input), False
            except Exception as e:  # tell Claude what went wrong instead of crashing
                content, is_error = f"Error: {e}", True
                trace(f"  -> error: {e}")
            results.append({"type": "tool_result", "tool_use_id": call.id, "content": content, "is_error": is_error})
        # All results for one turn go back in a single user message.
        messages.append({"role": "user", "content": results})

    raise RuntimeError(f"Agent did not finish within {MAX_TURNS} turns")


def compute_nutrition(match: MatchResult) -> list[NutritionLine]:
    """Turn Claude's matches into numbers, using USDA data only."""
    out = []
    for m in match.items:
        if m.fdc_id is None:
            out.append(NutritionLine(ingredient=m.ingredient, matched_food=None, fdc_id=None, grams=m.grams,
                                     protein_g=None, calories_kcal=None, carbs_g=None, fat_g=None, note=m.note))
            continue
        food = usda.get_food(m.fdc_id)
        out.append(NutritionLine(ingredient=m.ingredient, matched_food=food["description"], fdc_id=m.fdc_id,
                                 grams=m.grams, note=m.note, **_scale(food["per_100g"], m.grams)))
    return out
