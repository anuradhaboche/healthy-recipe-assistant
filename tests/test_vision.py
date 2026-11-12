"""Unit tests that don't call the API (a fake client stands in for Claude)."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from fridge_agent.models import FridgeInventory
from fridge_agent.vision import RefusalError, detect_ingredients

SAMPLE = FridgeInventory.model_validate(
    {
        "ingredients": [
            {"name": "egg", "quantity": "6", "category": "protein", "confidence": 0.95},
            {"name": "spinach", "quantity": "half a bag", "category": "vegetable", "confidence": 0.8},
        ],
        "unclear_items": ["white tub, top shelf"],
    }
)


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


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def stream(self, **kwargs):
        self.calls.append(kwargs)
        return FakeStream(self.response)


def fake_client(stop_reason="end_turn", parsed=SAMPLE):
    return SimpleNamespace(messages=FakeMessages(SimpleNamespace(stop_reason=stop_reason, parsed_output=parsed)))


@pytest.fixture
def photo(tmp_path: Path) -> Path:
    p = tmp_path / "fridge.jpg"
    p.write_bytes(b"\xff\xd8\xff fake jpeg")
    return p


def test_returns_parsed_inventory(photo):
    client = fake_client()
    result = detect_ingredients([photo], client=client)
    assert [i.name for i in result.ingredients] == ["egg", "spinach"]

    call = client.messages.calls[0]
    assert call["output_format"] is FridgeInventory
    blocks = call["messages"][0]["content"]
    assert blocks[0]["type"] == "image"
    assert blocks[0]["source"]["media_type"] == "image/jpeg"
    assert blocks[-1]["type"] == "text"


def test_multiple_photos_become_multiple_image_blocks(photo, tmp_path):
    second = tmp_path / "freezer.png"
    second.write_bytes(b"\x89PNG fake")
    client = fake_client()
    detect_ingredients([photo, second], client=client)
    blocks = client.messages.calls[0]["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "image", "text"]


def test_refusal_raises(photo):
    with pytest.raises(RefusalError):
        detect_ingredients([photo], client=fake_client(stop_reason="refusal", parsed=None))


def test_rejects_unsupported_file(tmp_path):
    bad = tmp_path / "fridge.heic"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match="unsupported"):
        detect_ingredients([bad], client=fake_client())
