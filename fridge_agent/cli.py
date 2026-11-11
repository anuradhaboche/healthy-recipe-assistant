"""Command line entry point.

    python -m fridge_agent photo1.jpg [photo2.jpg ...]           # list ingredients
    python -m fridge_agent photo.jpg --save-inventory inv.json    # ...and save them
    python -m fridge_agent --inventory inv.json --nutrition -v    # skip the photo, look up nutrition
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .models import FridgeInventory
from .nutrition import NutritionLine, compute_nutrition, match_ingredients
from .vision import detect_ingredients


def print_inventory(inventory: FridgeInventory) -> None:
    by_category: dict[str, list] = {}
    for item in inventory.ingredients:
        by_category.setdefault(item.category.value, []).append(item)

    for category, items in sorted(by_category.items()):
        print(f"\n{category.upper()}")
        for i in sorted(items, key=lambda x: -x.confidence):
            flag = "" if i.confidence >= 0.7 else "  (not sure)"
            note = f"  - {i.note}" if i.note else ""
            print(f"  {i.name:<24} {i.quantity:<14}{flag}{note}")

    if inventory.unclear_items:
        print("\nCOULDN'T IDENTIFY")
        for u in inventory.unclear_items:
            print(f"  {u}")


def _fmt(v: float | None) -> str:
    return "-" if v is None else f"{v:g}"


def print_nutrition(lines: list[NutritionLine]) -> None:
    print(f"\n{'INGREDIENT':<22}{'GRAMS':>7}{'PROTEIN g':>11}{'KCAL':>7}{'CARBS g':>9}{'FAT g':>7}  USDA MATCH")
    for n in sorted(lines, key=lambda x: -(x.protein_g or 0)):
        print(f"{n.ingredient[:21]:<22}{n.grams:>7g}{_fmt(n.protein_g):>11}{_fmt(n.calories_kcal):>7}"
              f"{_fmt(n.carbs_g):>9}{_fmt(n.fat_g):>7}  {n.matched_food or '(skipped)'}")
    total_protein = sum(n.protein_g or 0 for n in lines)
    total_kcal = sum(n.calories_kcal or 0 for n in lines)
    print(f"\nTotal in fridge: {total_protein:.0f} g protein, {total_kcal:.0f} kcal")
    notes = [n for n in lines if n.note]
    if notes:
        print("\nAssumptions:")
        for n in notes:
            print(f"  {n.ingredient}: {n.note}")


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # picks up ANTHROPIC_API_KEY / USDA_API_KEY from a local .env file
    parser = argparse.ArgumentParser(description="See what's in your fridge and what it's worth nutritionally.")
    parser.add_argument("photos", nargs="*", type=Path)
    parser.add_argument("--inventory", type=Path, help="load a saved ingredient list instead of photos")
    parser.add_argument("--save-inventory", type=Path, help="save the ingredient list as JSON")
    parser.add_argument("--nutrition", action="store_true", help="look up USDA nutrition for each ingredient")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of tables")
    parser.add_argument("-v", "--verbose", action="store_true", help="show each step the agent takes")
    args = parser.parse_args(argv)

    if bool(args.photos) == bool(args.inventory):
        parser.error("give either photos or --inventory")

    if args.inventory:
        inventory = FridgeInventory.model_validate_json(args.inventory.read_text())
    else:
        missing = [p for p in args.photos if not p.exists()]
        if missing:
            print(f"Not found: {', '.join(map(str, missing))}", file=sys.stderr)
            return 1
        inventory = detect_ingredients(args.photos)

    if args.save_inventory:
        args.save_inventory.write_text(inventory.model_dump_json(indent=2))

    if not args.nutrition:
        if args.json:
            print(json.dumps(inventory.model_dump(mode="json"), indent=2))
        else:
            print_inventory(inventory)
        return 0

    trace = (lambda msg: print(f"[agent] {msg}", file=sys.stderr)) if args.verbose else (lambda _: None)
    lines = compute_nutrition(match_ingredients(inventory, trace=trace))
    if args.json:
        print(json.dumps([n.model_dump() for n in lines], indent=2))
    else:
        print_nutrition(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
