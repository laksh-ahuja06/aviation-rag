"""Retrieval evaluation: hit@5 per retriever, plus the score distributions used to
calibrate the evidence-gate threshold.

    python -m eval.run_retrieval_eval [--split tune|test|all]
"""
import argparse
import json
import re
from pathlib import Path

from copilot import config
from copilot.retrieval import Retriever

QUESTIONS = json.loads((Path(__file__).parent / "test_questions.json").read_text())


def is_relevant(problem: str, patterns: list[str]) -> bool:
    return all(re.search(p, problem) for p in patterns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all", choices=["tune", "test", "all"])
    args = ap.parse_args()
    keep = lambda q: args.split == "all" or q["split"] == args.split
    answerable = [q for q in QUESTIONS["answerable"] if keep(q)]
    unanswerable = [q for q in QUESTIONS["unanswerable"] if keep(q)]

    r = Retriever()
    configs = {"bm25": dict(mode="bm25", rerank=False), "dense": dict(mode="dense", rerank=False),
               "hybrid (RRF)": dict(mode="hybrid", rerank=False), "hybrid + rerank": dict(mode="hybrid")}
    print(f"split={args.split}  answerable={len(answerable)}  unanswerable={len(unanswerable)}\n")
    for name, kw in configs.items():
        hits = 0
        misses = []
        for q in answerable:
            top = r.search(q["question"], k=5, **kw).hits
            ok = any(is_relevant(h["problem"], q["relevant"]) for h in top)
            hits += ok
            if not ok:
                misses.append(q["id"])
        print(f"{name:16s} hit@5 = {hits}/{len(answerable)}   misses: {misses}")

    print("\nBest cross-encoder score per question (for the refusal threshold):")
    for q in answerable + unanswerable:
        res = r.search(q["question"], k=5)
        top = res.hits[0] if res.hits else {}
        print(f"  {q['id']}  best={res.best_score:6.2f}  q='{q['question']}'  ->  {top.get('problem', '')[:50]}")
    print(f"\nCurrent EVIDENCE_THRESHOLD = {config.EVIDENCE_THRESHOLD}")


if __name__ == "__main__":
    main()
