"""Offline step: raw CSV -> normalised records + labels -> embedding matrix.

Run once (about a minute on a laptop CPU):
    python -m scripts.build_index
BM25 is rebuilt in memory at app start-up from data/records.csv (it takes <1 s).
"""
import time

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from copilot import config
from copilot.labels import action_type, component, fault_type
from copilot.normalize import SpellFixer, expand


def main():
    t0 = time.time()
    df = pd.read_csv(config.RAW_CSV, encoding="utf-8-sig", dtype={"IDENT": str})
    df = df.rename(columns={"IDENT": "ident", "PROBLEM": "problem", "ACTION": "action"})
    df["problem"] = df["problem"].fillna("").str.strip()
    df["action"] = df["action"].fillna("").str.strip()

    # Abbreviations first, then spelling, using a dictionary built from this corpus.
    exp_p, exp_a = df["problem"].map(expand), df["action"].map(expand)
    spell = SpellFixer(pd.concat([exp_p, exp_a]))
    df["problem_norm"] = exp_p.map(spell.fix)
    df["action_norm"] = exp_a.map(spell.fix)

    df["component"] = df["problem_norm"].map(component)
    df["fault_type"] = df["problem_norm"].map(fault_type)
    df["action_type_rule"] = df["action_norm"].map(action_type)

    config.DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(config.RECORDS_CSV, index=False)
    fixes = {k: v for k, v in spell.corrections.items() if k != v}
    print(f"Saved {len(df)} records; {len(fixes)} spelling fixes applied")
    print("Components:\n", df["component"].value_counts().to_string())
    print("Fault types:\n", df["fault_type"].value_counts().to_string())

    # Dense index over the PROBLEM text: we match on the reported fault, then
    # read the ACTION to see what fixed it.
    model = SentenceTransformer(config.EMBED_MODEL, device=config.DEVICE)
    texts = df["problem_norm"].str.lower().tolist()
    emb = model.encode(texts, batch_size=128, normalize_embeddings=True, show_progress_bar=True)

    config.EMBEDDINGS_NPY.parent.mkdir(exist_ok=True)
    np.save(config.EMBEDDINGS_NPY, emb.astype(np.float32))
    print(f"Saved {emb.shape[0]} x {emb.shape[1]} embeddings to {config.EMBEDDINGS_NPY}; "
          f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
