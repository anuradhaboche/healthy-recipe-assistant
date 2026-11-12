"""Your goals and preferences, loaded from profile.json.

Agent concept: context. The same fridge should give different recipes to
different people. Instead of baking preferences into the prompt, we keep
them in a file the agent reads every run.
"""

from pathlib import Path

from pydantic import BaseModel, Field

DEFAULT_PATH = Path("profile.json")
EXAMPLE_PATH = Path("profile.example.json")


class Profile(BaseModel):
    body_weight_kg: float = Field(gt=0)
    daily_protein_g: float = Field(gt=0, description="Total protein to eat per day")
    daily_calories: float = Field(gt=0)
    meals_per_day: int = Field(default=3, ge=1, le=8)
    avoid: list[str] = Field(default_factory=list, description="Foods to never use (allergies, dislikes)")
    max_cook_minutes: int = Field(default=40, gt=0)
    cuisine_preferences: list[str] = Field(default_factory=list)
    pantry_basics: list[str] = Field(
        default_factory=lambda: ["olive oil", "salt", "pepper", "garlic", "onion", "dried herbs", "spices", "soy sauce"],
        description="Staples you always have, so recipes can use them without them being in the photo",
    )

    @property
    def protein_per_meal_g(self) -> float:
        return round(self.daily_protein_g / self.meals_per_day)

    @property
    def calories_per_meal(self) -> float:
        return round(self.daily_calories / self.meals_per_day)


def load_profile(path: Path | None = None) -> Profile:
    """Load profile.json, falling back to the example so first runs still work."""
    for candidate in [path or DEFAULT_PATH, EXAMPLE_PATH]:
        if candidate.exists():
            return Profile.model_validate_json(candidate.read_text())
    raise FileNotFoundError("No profile.json found. Copy profile.example.json to profile.json and edit it.")
