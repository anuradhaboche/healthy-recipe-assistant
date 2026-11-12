"""Tests for the step 2 agent loop and USDA parsing. No network, no API key."""

import json
from types import SimpleNamespace

import pytest

from fridge_agent import nutrition, usda
from fridge_agent.models import FridgeInventory
from fridge_agent.nutrition import MatchedIngredient, MatchResult, compute_nutrition, match_ingredients

INVENTORY = FridgeInventory.model_validate(
    {"ingredients": [{"name": "egg", "quantity": "6", "category": "protein", "confidence": 0.9}]}
)

EGG_SEARCH = [{"fdc_id": 171287, "description": "Egg, whole, raw, fresh",
               "per_100g": {"protein_g": 12.6, "fat_g": 9.5, "carbs_g": 0.7, "calories_kcal": 143.0}}]


def tool_use(id_, name, input_):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=input_)


def text(t):
    return SimpleNamespace(type="text", text=t)


class FakeStream:
    """Stands in for client.messages.stream(...) as a context manager."""

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class ScriptedMessages:
    """Plays back a fixed sequence of responses, recording each request."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def stream(self, **kwargs):
        # Snapshot messages: the loop keeps appending to the same list.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        return FakeStream(self.responses.pop(0))


def test_agent_loop_runs_tools_and_returns_result(monkeypatch):
    searched = []
    monkeypatch.setattr(usda, "search_foods", lambda q: searched.append(q) or EGG_SEARCH)

    final = MatchResult(items=[MatchedIngredient(ingredient="egg", fdc_id=171287, grams=300)])
    msgs = ScriptedMessages([
        SimpleNamespace(stop_reason="tool_use", parsed_output=None,
                        content=[tool_use("t1", "search_foods", {"query": "egg whole raw"})]),
        SimpleNamespace(stop_reason="end_turn", parsed_output=final, content=[text("{...}")]),
    ])
    trace = []
    result = match_ingredients(INVENTORY, client=SimpleNamespace(messages=msgs), trace=trace.append)

    assert result == final
    assert searched == ["egg whole raw"]
    assert len(msgs.calls) == 2
    # Second request carries: user, assistant (tool_use), user (tool_result)
    second = msgs.calls[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    tool_result = second[2]["content"][0]
    assert tool_result["tool_use_id"] == "t1"
    assert json.loads(tool_result["content"])[0]["fdc_id"] == 171287
    assert any("search_foods" in t for t in trace)


def test_tool_error_is_reported_to_claude_not_raised(monkeypatch):
    def boom(q):
        raise usda.USDAError("rate limited")

    monkeypatch.setattr(usda, "search_foods", boom)
    final = MatchResult(items=[MatchedIngredient(ingredient="egg", fdc_id=None, grams=300)])
    msgs = ScriptedMessages([
        SimpleNamespace(stop_reason="tool_use", parsed_output=None,
                        content=[tool_use("t1", "search_foods", {"query": "egg"})]),
        SimpleNamespace(stop_reason="end_turn", parsed_output=final, content=[text("{}")]),
    ])
    match_ingredients(INVENTORY, client=SimpleNamespace(messages=msgs))
    result_block = msgs.calls[1]["messages"][2]["content"][0]
    assert result_block["is_error"] is True
    assert "rate limited" in result_block["content"]


def test_loop_gives_up_after_max_turns(monkeypatch):
    monkeypatch.setattr(usda, "search_foods", lambda q: [])
    monkeypatch.setattr(nutrition, "MAX_TURNS", 3)
    looping = SimpleNamespace(stop_reason="tool_use", parsed_output=None,
                              content=[tool_use("t", "search_foods", {"query": "x"})])
    msgs = ScriptedMessages([looping] * 3)
    with pytest.raises(RuntimeError, match="did not finish"):
        match_ingredients(INVENTORY, client=SimpleNamespace(messages=msgs))


def test_compute_nutrition_scales_usda_values(monkeypatch):
    monkeypatch.setattr(usda, "get_food", lambda fid: EGG_SEARCH[0])
    lines = compute_nutrition(MatchResult(items=[
        MatchedIngredient(ingredient="egg", fdc_id=171287, grams=300),
        MatchedIngredient(ingredient="ketchup", fdc_id=None, grams=20, note="skipped condiment"),
    ]))
    egg, ketchup = lines
    assert egg.protein_g == pytest.approx(37.8)
    assert egg.calories_kcal == pytest.approx(429.0)
    assert ketchup.protein_g is None and ketchup.note == "skipped condiment"


def test_macros_per_100g_handles_both_response_shapes():
    search_shape = {"foodNutrients": [{"nutrientId": 1003, "value": 10.0}, {"nutrientId": 1008, "value": 200}]}
    detail_shape = {"foodNutrients": [{"nutrient": {"id": 1003}, "amount": 10.0},
                                      {"nutrient": {"id": 2047}, "amount": 199}]}
    assert usda.macros_per_100g(search_shape)["protein_g"] == 10.0
    assert usda.macros_per_100g(search_shape)["calories_kcal"] == 200
    assert usda.macros_per_100g(detail_shape)["calories_kcal"] == 199  # Atwater fallback
