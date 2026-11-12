"""Thin client for USDA FoodData Central (https://fdc.nal.usda.gov/api-guide).

Free API. Get a key at https://fdc.nal.usda.gov/api-key-signup (instant),
or leave USDA_API_KEY unset to use DEMO_KEY (heavily rate limited).

Responses are cached on disk so repeat runs don't hit the API again.
"""

import hashlib
import json
import os
from pathlib import Path

import httpx

BASE_URL = "https://api.nal.usda.gov/fdc/v1"
CACHE_DIR = Path(os.environ.get("FRIDGE_AGENT_CACHE", ".cache/usda"))

# FoodData Central nutrient ids
PROTEIN = 1003
FAT = 1004
CARBS = 1005
ENERGY_KCAL = 1008
ENERGY_ATWATER = (2047, 2048)  # Foundation foods often report energy under these instead


class USDAError(RuntimeError):
    pass


def _api_key() -> str:
    return os.environ.get("USDA_API_KEY") or "DEMO_KEY"


def _cached_get(path: str, params: dict) -> dict:
    key = hashlib.sha256(json.dumps([path, params], sort_keys=True).encode()).hexdigest()[:16]
    cache_file = CACHE_DIR / f"{key}.json"
    if cache_file.exists():
        return json.loads(cache_file.read_text())

    try:
        resp = httpx.get(f"{BASE_URL}{path}", params={**params, "api_key": _api_key()}, timeout=20)
        resp.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 429:
            raise USDAError("USDA rate limit hit. Set USDA_API_KEY to your own free key.") from e
        raise USDAError(f"USDA API error {e.response.status_code}") from e
    except httpx.HTTPError as e:
        raise USDAError(f"Could not reach USDA API: {e}") from e

    data = resp.json()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(data))
    return data


def macros_per_100g(food: dict) -> dict:
    """Pull protein/fat/carbs/kcal per 100 g out of a search result or food detail record."""
    values: dict[int, float] = {}
    for n in food.get("foodNutrients", []):
        # Search results and detail records use slightly different shapes.
        nid = n.get("nutrientId") or n.get("nutrient", {}).get("id")
        amount = n.get("value", n.get("amount"))
        if nid is not None and amount is not None:
            values[int(nid)] = float(amount)

    kcal = values.get(ENERGY_KCAL)
    if kcal is None:
        kcal = next((values[i] for i in ENERGY_ATWATER if i in values), None)
    return {
        "protein_g": values.get(PROTEIN),
        "fat_g": values.get(FAT),
        "carbs_g": values.get(CARBS),
        "calories_kcal": kcal,
    }


def _food_cache(fdc_id: int) -> Path:
    return CACHE_DIR / "foods" / f"{fdc_id}.json"


def search_foods(query: str, limit: int = 5) -> list[dict]:
    """Search generic foods (Foundation + SR Legacy, which report per 100 g)."""
    data = _cached_get(
        "/foods/search",
        {"query": query, "dataType": "Foundation,SR Legacy", "pageSize": limit},
    )
    results = [
        {"fdc_id": f["fdcId"], "description": f["description"], "per_100g": macros_per_100g(f)}
        for f in data.get("foods", [])
    ]
    # Remember each food we've seen, so get_food() doesn't need another API call.
    (CACHE_DIR / "foods").mkdir(parents=True, exist_ok=True)
    for r in results:
        _food_cache(r["fdc_id"]).write_text(json.dumps(r))
    return results


def get_food(fdc_id: int) -> dict:
    cached = _food_cache(fdc_id)
    if cached.exists():
        return json.loads(cached.read_text())
    data = _cached_get(f"/food/{fdc_id}", {})
    return {"fdc_id": fdc_id, "description": data.get("description", ""), "per_100g": macros_per_100g(data)}
