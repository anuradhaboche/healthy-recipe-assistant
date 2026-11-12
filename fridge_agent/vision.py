"""Step 1: turn fridge photos into a structured ingredient list.

Agent concept: multimodal input. Claude can read images, so the "eyes" of
the agent are just an image content block in the message. Combined with
structured output (see models.py), one call turns pixels into data the rest
of the agent can reason over.
"""

import base64
import mimetypes
from pathlib import Path

import anthropic

from .models import FridgeInventory

MODEL = "claude-opus-5-5"

SYSTEM_PROMPT = """You catalogue the contents of a home fridge, freezer or pantry from photos.
The list feeds a recipe planner for someone strength training, so protein sources matter most.

- List only food you can actually see. Do not guess at items hidden behind others.
- Merge duplicates across photos into one entry.
- Read labels when they're legible (e.g. 'greek yogurt' rather than 'yogurt').
- Estimate quantities roughly; say 'unknown' rather than inventing a number.
- Put anything you can see but cannot identify in unclear_items instead of guessing."""

# Give up after 2 minutes instead of the SDK default of 10, so a network
# problem shows up as an error rather than a silent hang.
TIMEOUT_SECONDS = 120


def make_client() -> anthropic.Anthropic:
    return anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=1)


SUPPORTED_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}


class RefusalError(RuntimeError):
    pass


def _image_block(path: Path) -> dict:
    media_type, _ = mimetypes.guess_type(path.name)
    if media_type not in SUPPORTED_TYPES:
        raise ValueError(f"{path.name}: unsupported image type {media_type!r} (use JPEG, PNG, GIF or WebP)")
    data = base64.standard_b64encode(path.read_bytes()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


def detect_ingredients(
    image_paths: list[Path],
    client: anthropic.Anthropic | None = None,
) -> FridgeInventory:
    """Send one or more photos to Claude and get back a validated FridgeInventory."""
    if not image_paths:
        raise ValueError("Need at least one photo")
    client = client or make_client()

    content = [_image_block(Path(p)) for p in image_paths]
    content.append({"type": "text", "text": "List the ingredients in these photos."})

    response = client.messages.parse(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        # Low effort: this is a perception task, not a hard reasoning one.
        output_config={"effort": "low"},
        messages=[{"role": "user", "content": content}],
        output_format=FridgeInventory,
    )

    if response.stop_reason == "refusal":
        raise RefusalError("Claude declined to analyse these images")
    if response.parsed_output is None:
        raise RuntimeError(f"No structured output (stop_reason={response.stop_reason})")
    return response.parsed_output
