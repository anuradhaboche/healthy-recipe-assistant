"""Tests for step 3: the recipe checker and the recipe agent. No network, no API key."""

import json
from types import SimpleNamespace

import pytest

from fridge_agent import usda
from fridge_agent.nutrition import NutritionLine
from fridge_agent.profile import Profile, load_profile
from fridge_agent.recipes import Pantry, Recipe, RecipeIngredient, RecipeSuggestions, suggest_recipes
from tests.test_nutrition import ScriptedMessages, text, tool_use

PER_100G = {
    1: {"protein_g": 22.5, "calories_kcal": 120.0, "carbs_g": 0.0, "fat_g": 2.6},  # chicken breast
    2: {"protein_g": 2.9, "calories_kcal": 23.0, "carbs_g": 3.6, "fat_g": 0.4},    # spinach
}

NUTRITION = [
    NutritionLine(ingredient="chicken breast", matched_food="Chicken", fdc_id=1, grams=525,
                  protein_g=None, calories_kcal=None, carbs_g=None, fat_g=None),
    NutritionLine(ingredient="spinach", matched_food="Spinach", fdc_id=2, grams=300,
                  protein_g=None, calories_kcal=None, carbs_g=None, fat_g=None),
]

PROFILE = Profile(body_weight_kg=65, daily_protein_g=120, daily_calories=2000, meals_per_day=3,
                  avoid=["peanut"], max_cook_minutes=30)


@pytest.fixture(autouse=True)
def fake_usda(monkeypatch):
    monkeypatch.setattr(usda, "get_food", lambda fid: {"fdc_id": fid, "description": "", "per_100g": PER_100G[fid]})


def recipe(**overrides):
    base = dict(title="Chicken & spinach", why="uses the spinach", servings=2, cook_minutes=20,
                ingredients=[RecipeIngredient(ingredient="chicken breast", grams=350),
                             RecipeIngredient(ingredient="spinach", grams=150),
                             RecipeIngredient(ingredient="olive oil", grams=10)],
                steps=["Cook it"])
    base.update(overrides)
    return Recipe(**base)


def test_good_recipe_passes_and_macros_come_from_usda():
    result = Pantry(NUTRITION, PROFILE).check(recipe())
    assert result.problems == []
    assert result.meets_protein_target
    # (350 * 22.5 + 150 * 2.9) / 100 / 2 servings
    assert result.per_serving.protein_g == pytest.approx(41.55, abs=0.1)


def test_flags_missing_ingredient_overuse_avoid_list_and_time():
    bad = recipe(cook_minutes=45, ingredients=[
        RecipeIngredient(ingredient="chicken breast", grams=900),   # only 525 g
        RecipeIngredient(ingredient="salmon", grams=200),          # not in fridge
        RecipeIngredient(ingredient="peanut butter", grams=30),    # avoid list
    ])
    problems = " | ".join(Pantry(NUTRITION, PROFILE).check(bad).problems)
    assert "only 525 g available" in problems
    assert "salmon: not in the fridge" in problems
    assert "avoid list" in problems
    assert "over the 30 min limit" in problems


def test_low_protein_fails_target():
    light = recipe(servings=4, ingredients=[RecipeIngredient(ingredient="spinach", grams=300)])
    result = Pantry(NUTRITION, PROFILE).check(light)
    assert not result.meets_protein_target
    assert any("target is 40 g" in p for p in result.problems)


def test_agent_checks_then_returns_recipes():
    draft = recipe().model_dump()
    final = RecipeSuggestions(recipes=[recipe()])
    msgs = ScriptedMessages([
        SimpleNamespace(stop_reason="tool_use", parsed_output=None,
                        content=[tool_use("t1", "check_recipe", draft)]),
        SimpleNamespace(stop_reason="end_turn", parsed_output=final, content=[text("{}")]),
    ])
    results = suggest_recipes(NUTRITION, PROFILE, n=1, client=SimpleNamespace(messages=msgs))

    tool_result = json.loads(msgs.calls[1]["messages"][2]["content"][0]["content"])
    assert tool_result["problems"] == []
    assert "40 g protein" in msgs.calls[0]["system"]          # target reached the prompt
    assert "spinach: 300 g available" in msgs.calls[0]["messages"][0]["content"]
    [(r, check)] = results
    assert r.title == "Chicken & spinach" and check.meets_protein_target


def test_profile_falls_back_to_example(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "profile.example.json").write_text(PROFILE.model_dump_json())
    assert load_profile().daily_protein_g == 120
