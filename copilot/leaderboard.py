"""Fix leaderboard: group the actions taken in similar past records and rank them
by how often each occurred. Pure pandas over offline labels, so it costs no LLM calls."""
from collections import Counter

import pandas as pd

from copilot import config

INSPECTION_TYPES = {"Inspected - no fault found", "Inspected / troubleshot"}


class ActionLabels:
    def __init__(self, records: pd.DataFrame):
        df = records[["ident", "problem", "action", "component", "action_type_rule"]].copy()
        if config.ACTION_LABELS_CSV.exists():
            llm = pd.read_csv(config.ACTION_LABELS_CSV).fillna("").drop_duplicates(["problem", "action"])
            df = df.merge(llm, on=["problem", "action"], how="left")
        else:
            df["action_type"], df["part"] = None, None
        # Fall back to the rule label (and the problem's component) where the LLM gave none.
        missing = df["action_type"].isna() | (df["action_type"] == "")
        df.loc[missing, "action_type"] = df.loc[missing, "action_type_rule"]
        df.loc[missing, "part"] = df.loc[missing, "component"]
        df["label"] = [label(t, p) for t, p in zip(df["action_type"], df["part"])]
        self.df = df.set_index("ident")
        self.llm_coverage = float((~missing).mean())

    def parts(self, idents: list[str]) -> Counter:
        """How many of these records acted on each (canonical) part, e.g. {'baffle plug': 4, 'spark plug': 1}.
        Deterministic, so even a weak local LLM gets a clear signal when one word covers two parts."""
        sub = self.df.loc[[i for i in idents if i in self.df.index]]
        return Counter(p for p in map(canonical_part, sub["part"]) if p and p not in ("none", "engine"))


PART_SYNONYMS = [
    ("rocker box cover", "rocker cover"), ("rocker box", "rocker cover"),
    ("push rod", "pushrod"), ("dip stick", "dipstick"), ("spark plugs", "spark plug"),
    ("induction tube", "intake tube"), ("induction", "intake"), ("baffling", "baffle"),
]


def canonical_part(part: str) -> str:
    """Merge naming variants so counts don't split: 'rocker box cover gaskets' -> 'rocker cover gasket'."""
    p = " ".join(str(part or "").lower().split())
    if "/" in p:  # a rule-fallback component name such as "intake / induction": keep as is
        return p
    for a, b in PART_SYNONYMS:
        p = p.replace(a, b)
    words = p.split()
    if words and len(words[-1]) > 3 and words[-1].endswith("s") and not words[-1].endswith("ss"):
        words[-1] = words[-1][:-1]  # last-word plural: gaskets -> gasket
    return " ".join(words)


def label(action_type: str, part: str) -> str:
    part = canonical_part(part)
    if action_type in INSPECTION_TYPES or part in ("", "none", "engine"):
        return action_type
    return f"{action_type}: {part}"


def build_leaderboard(labels: ActionLabels, idents: list[str], top: int = 8) -> dict:
    """{'n_records': 9, 'rows': [{'label': 'Replaced: spark plug', 'count': 5, 'idents': [...]}, ...]}"""
    idents = [i for i in dict.fromkeys(idents) if i in labels.df.index]
    if not idents:
        return {"n_records": 0, "rows": []}
    sub = labels.df.loc[idents]
    rows = []
    for lab, grp in sub.groupby("label", sort=False):
        rows.append({"label": lab, "count": len(grp), "idents": list(grp.index)})
    rows.sort(key=lambda r: (-r["count"], r["label"]))
    return {"n_records": len(idents), "rows": rows[:top]}
