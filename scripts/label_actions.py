"""Offline step: label every (PROBLEM, ACTION) pair with an action type and the
part acted on, for the fix leaderboard.

    python -m scripts.label_actions

Runs once over ~5,000 unique pairs in batches (about 50 Gemini calls) and
resumes from data/action_labels.csv if interrupted. At query time the app
only does a pandas groupby over these labels: no LLM calls, no rate limits.
Pairs the LLM fails to label keep the rule-based label from build_index.
"""
import json
import time

import pandas as pd
from google import genai
from google.genai import types

from copilot import config
from copilot.labels import ACTION_TYPE_NAMES

BATCH = 100
MODELS = [config.GEMINI_FALLBACK_MODEL, config.GEMINI_MODEL]  # Lite first: cheaper, and it's a simple task

PROMPT = """You label aircraft engine maintenance logbook entries.
For each item, read the reported PROBLEM and the ACTION taken, and return:
- action_type: exactly one of {types}
  ("Replaced" = removed and replaced or installed a new/serviceable part;
   "Installed missing hardware" = fitted a screw/bolt/nut/rivet/clamp that was missing;
   "Inspected - no fault found" = checked/ran and found nothing wrong or could not duplicate;
   "Inspected / troubleshot" = investigation step without a final fix in this entry).
- part: the main part acted on, as a short generic lower-case noun phrase. Drop cylinder numbers,
  left/right and positions. Use consistent names, e.g. "intake gasket", "rocker cover gasket",
  "baffle screw", "baffle", "spark plug", "magneto", "cylinder assembly", "pushrod tube seal",
  "exhaust gasket", "oil cooler". Use "engine" for whole-engine checks and "none" if no part.

Items (JSON):
{items}
"""

SCHEMA = {
    "type": "ARRAY",
    "items": {
        "type": "OBJECT",
        "properties": {
            "i": {"type": "INTEGER"},
            "action_type": {"type": "STRING", "enum": ACTION_TYPE_NAMES},
            "part": {"type": "STRING"},
        },
        "required": ["i", "action_type", "part"],
    },
}


def label_batch(client, batch: pd.DataFrame) -> list[dict]:
    items = [{"i": int(i), "problem": p, "action": a}
             for i, p, a in zip(batch.index, batch["problem"], batch["action"])]
    prompt = PROMPT.format(types=ACTION_TYPE_NAMES, items=json.dumps(items))
    last_err = None
    for model in MODELS:
        for attempt in range(3):
            try:
                resp = client.models.generate_content(
                    model=model, contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0, response_mime_type="application/json", response_schema=SCHEMA))
                return json.loads(resp.text)
            except Exception as e:  # 429 rate limit / 503 overload: back off, then try next model
                last_err = e
                time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"all models failed: {last_err}")


def main():
    df = pd.read_csv(config.RECORDS_CSV, dtype={"ident": str}).fillna("")
    pairs = df[["problem", "action"]].drop_duplicates().reset_index(drop=True)

    done = pd.DataFrame(columns=["problem", "action", "action_type", "part"])
    if config.ACTION_LABELS_CSV.exists():
        done = pd.read_csv(config.ACTION_LABELS_CSV).fillna("")
    seen = set(zip(done["problem"], done["action"]))
    todo = pairs[[(p, a) not in seen for p, a in zip(pairs["problem"], pairs["action"])]]
    print(f"{len(pairs)} unique pairs, {len(todo)} still to label")

    client = genai.Client()
    for start in range(0, len(todo), BATCH):
        batch = todo.iloc[start:start + BATCH]
        try:
            out = label_batch(client, batch)
        except RuntimeError as e:
            print(f"batch at {start} skipped: {e}")
            continue
        by_i = {o["i"]: o for o in out if o["i"] in batch.index}
        rows = [{"problem": batch.at[i, "problem"], "action": batch.at[i, "action"],
                 "action_type": o["action_type"], "part": o["part"].strip().lower()}
                for i, o in by_i.items()]
        done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
        done.to_csv(config.ACTION_LABELS_CSV, index=False)
        print(f"labelled {len(done)}/{len(pairs)}")
    print(done["action_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
