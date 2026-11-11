# Healthy Recipe Assistant

An AI agent that looks at photos of your fridge and suggests healthy, high-protein recipes you can cook right now. Built step by step as a way to learn how agents work.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your key from console.anthropic.com
```

## Use

```bash
python -m fridge_agent photos/fridge.jpg photos/freezer.jpg
python -m fridge_agent photos/fridge.jpg --json   # raw structured output
```

Run the tests (no API key needed): `pytest`

## Roadmap and what each step teaches

| Step | What it does | Agent concept |
| --- | --- | --- |
| 1 ✅ | Photo to ingredient list | Multimodal input, structured output |
| 2 | Real nutrition data from USDA FoodData Central | Tool use and the agent loop |
| 3 | 2 or 3 high-protein recipes with macros | Grounding, retrieval (RAG) |
| 4 | Weekly plan around training days + shopping list | Planning and subtasks |
| 5 | Chef, Nutritionist and Critic agents | Multi-agent orchestration, self-checking |
| 6 | Remembers ratings and what you ate | Long-term memory, feedback loops |
| 7 | Phone-friendly web page (Streamlit) | Putting an agent behind a UI |
| Throughout | Test set of fridge photos, step-by-step logs | Evals and tracing |

## Step 1: how it works

`fridge_agent/vision.py` sends your photos to Claude as image blocks, alongside a short system prompt. Instead of asking for free text, it passes a Pydantic model (`FridgeInventory` in `models.py`) as `output_format`. The API constrains Claude's reply to that JSON schema, and the SDK hands back a validated Python object. That's the pattern every later step builds on: the model's output is data the next step can trust.

This step is a single call, not yet an agent. The agent loop (Claude choosing which tools to call, looking at results, and deciding what to do next) arrives in step 2.
