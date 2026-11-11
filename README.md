# Healthy Recipe Assistant

An AI agent that looks at photos of your fridge and suggests healthy, high-protein recipes you can cook right now. Built step by step as a way to learn how agents work.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # paste your key from console.anthropic.com (and optionally a free USDA key)
```

## Use

```bash
python -m fridge_agent photos/fridge.jpg photos/freezer.jpg
python -m fridge_agent photos/fridge.jpg --json   # raw structured output

# Step 2: nutrition. Save the ingredient list once, then reuse it (saves API calls)
python -m fridge_agent photos/fridge.jpg --save-inventory inventory.json
python -m fridge_agent --inventory inventory.json --nutrition -v

# No photo handy? Try the sample fridge image or the sample list
python -m fridge_agent examples/sample_fridge.jpg
python -m fridge_agent --inventory examples/sample_inventory.json --nutrition -v
```

Run the tests (no API key needed): `pytest`

## Roadmap and what each step teaches

| Step | What it does | Agent concept |
| --- | --- | --- |
| 1 ✅ | Photo to ingredient list | Multimodal input, structured output |
| 2 ✅ | Real nutrition data from USDA FoodData Central | Tool use and the agent loop |
| 3 | 2 or 3 high-protein recipes with macros | Grounding, retrieval (RAG) |
| 4 | Weekly plan around training days + shopping list | Planning and subtasks |
| 5 | Chef, Nutritionist and Critic agents | Multi-agent orchestration, self-checking |
| 6 | Remembers ratings and what you ate | Long-term memory, feedback loops |
| 7 | Phone-friendly web page (Streamlit) | Putting an agent behind a UI |
| Throughout | Test set of fridge photos, step-by-step logs | Evals and tracing |

## Step 1: how it works

`fridge_agent/vision.py` sends your photos to Claude as image blocks, alongside a short system prompt. Instead of asking for free text, it passes a Pydantic model (`FridgeInventory` in `models.py`) as `output_format`. The API constrains Claude's reply to that JSON schema, and the SDK hands back a validated Python object. That's the pattern every later step builds on: the model's output is data the next step can trust.

This step is a single call, not yet an agent.

## Step 2: how it works

`fridge_agent/nutrition.py` is the first real agent. Claude gets a `search_foods` tool (backed by the free [USDA FoodData Central](https://fdc.nal.usda.gov/) API in `usda.py`) and a list of ingredients. Then a loop runs:

1. Send the conversation to Claude, along with the tool definitions.
2. If Claude's reply contains tool calls, run them, append the results, and go back to 1.
3. If it doesn't, Claude is done: read the structured `MatchResult`.

Claude decides what to search for, retries with different wording when results look wrong, and estimates grams from quantities like "half a bag". Run with `-v` to watch each step.

Two things worth noticing:

- **Grounding.** Claude only picks *which* USDA food matches and *how many grams*. The protein and calorie numbers are computed in Python from USDA data (`compute_nutrition`), so the model never invents or miscalculates them.
- **Errors go back to the model.** If a tool fails (rate limit, bad query), the loop returns the error as a tool result instead of crashing, so Claude can try something else. `MAX_TURNS` stops a runaway loop.

The loop is hand-written so every step is visible. The Anthropic SDK also has a tool runner that does this for you; later steps may switch to it.
