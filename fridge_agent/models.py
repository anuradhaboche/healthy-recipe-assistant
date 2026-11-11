"""Data shapes the agent passes around.

Agent concept: structured output. Instead of asking Claude for free text and
parsing it ourselves, we describe the exact shape we want as a Pydantic model.
The SDK turns it into a JSON schema, the API constrains Claude's answer to that
schema, and we get back a validated Python object.
"""

from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    protein = "protein"
    dairy = "dairy"
    vegetable = "vegetable"
    fruit = "fruit"
    grain = "grain"
    condiment = "condiment"
    beverage = "beverage"
    other = "other"


class Ingredient(BaseModel):
    name: str = Field(description="Common name, singular, lowercase, e.g. 'egg', 'greek yogurt'")
    quantity: str = Field(description="Rough amount visible, e.g. '6', 'half a bag', '500 g', 'unknown'")
    category: Category
    confidence: float = Field(ge=0, le=1, description="How sure you are this item is really there")
    note: str | None = Field(default=None, description="Anything useful, e.g. 'looks close to expiring'")


class FridgeInventory(BaseModel):
    ingredients: list[Ingredient]
    unclear_items: list[str] = Field(
        default_factory=list,
        description="Things you can see but can't identify, e.g. 'white container, top shelf'",
    )
