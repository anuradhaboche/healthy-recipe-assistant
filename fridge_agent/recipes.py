"""Step 3: suggest high-protein recipes from what's in the fridge.

Agent concept: self-checking with a tool.

Claude is good at inventing recipes and bad at adding up grams of protein in
its head. So we give it a calculator tool, `check_recipe`, that:
  - computes protein / calories / carbs / fat per serving from USDA data
  - rejects ingredients that aren't in the fridge or pantry basics
  - rejects using more of something than you have
  - says whether the recipe hits your protein-per-meal target

The prompt tells Claude to check every recipe and fix any that fail before
answering. That's a draft -> check -> revise loop, the same thing a careful
person does.

Then, as a final guardrail, Python re-checks the recipes Claude returns. The
numbers you see are always computed by our code, never typed by the model.
"""

import json
from typing import Callable

import anthropic
from pydantic import BaseModel, Field

from . import usda
from .agent_loop import run_agent
from .nutrition import NutritionLine
from .profile import Profile
from .vision import make_client

MAX_TURNS = 20


class RecipeIngredient(BaseModel):
    ingredient: str = Field(description="Exact ingredient name from the fridge list, or a pantry basic")
    grams: float = Field(ge=0, description="Total grams for the whole recipe (all servings)")


class Recipe(BaseModel):
    title: str
    why: str = Field(description="One line on why this fits, e.g. 'uses the spinach before it wilts'")
    servings: int = Field(ge=1)
    cook_minutes: int = Field(ge=1)
    ingredients: list[RecipeIngredient]
    steps: list[str]


class RecipeSuggestions(BaseModel):
    recipes: list[Recipe]


class Macros(BaseModel):
    protein_g: float
    calories_kcal: float
    carbs_g: float
    fat_g: float


class CheckResult(BaseModel):
    per_serving: Macros
    problems: list[str]
    meets_protein_target: bool


class Pantry:
    """What's available: fridge items with USDA data, plus pantry basics."""

    def __init__(self, nutrition: list[NutritionLine], profile: Profile):
        self.profile = profile
        self.items: dict[str, dict] = {}
        for n in nutrition:
            per_100g = usda.get_food(n.fdc_id)["per_100g"] if n.fdc_id else None
            self.items[n.ingredient.lower()] = {"name": n.ingredient, "available_g": n.grams, "per_100g": per_100g}
        self.basics = {b.lower() for b in profile.pantry_basics}

    def describe(self) -> str:
        lines = []
        for item in self.items.values():
            p = item["per_100g"]
            macros = (f"{p['protein_g'] or 0:g} g protein, {p['calories_kcal'] or 0:g} kcal per 100 g"
                      if p else "no nutrition data")
            lines.append(f"- {item['name']}: {item['available_g']:g} g available ({macros})")
        return "\n".join(lines)

    def check(self, recipe: Recipe) -> CheckResult:
        totals = {"protein_g": 0.0, "calories_kcal": 0.0, "carbs_g": 0.0, "fat_g": 0.0}
        problems = []
        for ing in recipe.ingredients:
            key = ing.ingredient.lower()
            if key in self.items:
                item = self.items[key]
                if ing.grams > item["available_g"] * 1.05:
                    problems.append(f"{ing.ingredient}: uses {ing.grams:g} g but only {item['available_g']:g} g available")
                if item["per_100g"]:
                    for k in totals:
                        totals[k] += (item["per_100g"].get(k) or 0) * ing.grams / 100
            elif key in self.basics:
                pass  # small amounts of staples; not counted
            else:
                problems.append(f"{ing.ingredient}: not in the fridge or pantry basics")
            if any(a.lower() in key for a in self.profile.avoid):
                problems.append(f"{ing.ingredient}: on your avoid list")
        if recipe.cook_minutes > self.profile.max_cook_minutes:
            problems.append(f"takes {recipe.cook_minutes} min, over the {self.profile.max_cook_minutes} min limit")

        per_serving = Macros(**{k: round(v / recipe.servings, 1) for k, v in totals.items()})
        meets = per_serving.protein_g >= self.profile.protein_per_meal_g * 0.9
        if not meets:
            problems.append(f"{per_serving.protein_g:g} g protein per serving, target is "
                            f"{self.profile.protein_per_meal_g:g} g")
        return CheckResult(per_serving=per_serving, problems=problems, meets_protein_target=meets)


SYSTEM_PROMPT = """You are a sports-nutrition-minded home cook. Suggest {n} recipes the user can cook
right now from what's in their fridge, aimed at someone strength training.

Rules:
- Use only fridge ingredients (exact names from the list) plus these pantry basics: {basics}.
- Never use: {avoid}.
- Each serving should give about {protein} g protein and roughly {calories} kcal or less.
- Each recipe takes at most {minutes} minutes.
- Make the recipes different from each other (different main protein or style).
- Prefer using ingredients that spoil first (fresh vegetables, opened dairy, raw meat).
{cuisines}
Before answering, call check_recipe on EVERY recipe. If it reports problems, change the
recipe (adjust grams, servings or ingredients) and check again. Only return recipes that
passed. Grams are totals for the whole recipe, not per serving. Steps should be short and clear."""

TOOLS = [
    {
        "name": "check_recipe",
        "description": (
            "Check a draft recipe against the fridge contents and the user's goals. Returns protein, calories, "
            "carbs and fat per serving computed from USDA data, plus a list of problems (missing ingredients, "
            "using more than available, protein below target, too long to cook). An empty problems list means it passes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "why": {"type": "string"},
                "servings": {"type": "integer"},
                "cook_minutes": {"type": "integer"},
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"ingredient": {"type": "string"}, "grams": {"type": "number"}},
                        "required": ["ingredient", "grams"],
                    },
                },
                "steps": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["title", "why", "servings", "cook_minutes", "ingredients", "steps"],
        },
    },
]


def suggest_recipes(
    nutrition: list[NutritionLine],
    profile: Profile,
    n: int = 3,
    client: anthropic.Anthropic | None = None,
    trace: Callable[[str], None] = lambda _: None,
) -> list[tuple[Recipe, CheckResult]]:
    client = client or make_client()
    pantry = Pantry(nutrition, profile)

    def run_tool(name: str, args: dict) -> str:
        if name != "check_recipe":
            raise ValueError(f"Unknown tool {name}")
        result = pantry.check(Recipe.model_validate(args))
        trace(f"  -> {result.per_serving.protein_g:g} g protein/serving, "
              f"{'OK' if not result.problems else '; '.join(result.problems)}")
        return result.model_dump_json()

    system = SYSTEM_PROMPT.format(
        n=n,
        basics=", ".join(profile.pantry_basics),
        avoid=", ".join(profile.avoid) or "nothing",
        protein=profile.protein_per_meal_g,
        calories=profile.calories_per_meal,
        minutes=profile.max_cook_minutes,
        cuisines=(f"- The user enjoys: {', '.join(profile.cuisine_preferences)}.\n"
                  if profile.cuisine_preferences else ""),
    )
    suggestions = run_agent(
        client,
        system=system,
        tools=TOOLS,
        run_tool=run_tool,
        output_format=RecipeSuggestions,
        first_message="What's in my fridge:\n" + pantry.describe(),
        effort="medium",  # creative + arithmetic trade-offs: worth a bit more thinking
        max_turns=MAX_TURNS,
        trace=trace,
    )
    # Final guardrail: re-check in code; the printed numbers come from here.
    return [(r, pantry.check(r)) for r in suggestions.recipes]
