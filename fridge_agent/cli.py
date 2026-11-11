"""Command line entry point.

    python -m fridge_agent photo1.jpg [photo2.jpg ...] [--json]
"""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .vision import detect_ingredients


def main(argv: list[str] | None = None) -> int:
    load_dotenv()  # picks up ANTHROPIC_API_KEY from a local .env file
    parser = argparse.ArgumentParser(description="List the ingredients in fridge photos.")
    parser.add_argument("photos", nargs="+", type=Path)
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of a table")
    args = parser.parse_args(argv)

    missing = [p for p in args.photos if not p.exists()]
    if missing:
        print(f"Not found: {', '.join(map(str, missing))}", file=sys.stderr)
        return 1

    inventory = detect_ingredients(args.photos)

    if args.json:
        print(json.dumps(inventory.model_dump(mode="json"), indent=2))
        return 0

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
