"""End-to-end evaluation of the full copilot (agent + gate + verifier), per the PS 1.3 brief:
- answered rate on answerable questions
- citation precision: share of cited records whose PROBLEM matches the question's relevance rule
- correct refusals on unanswerable questions

    python -m eval.run_agent_eval     (about 20 questions x 3-5 Gemini calls)
"""
import json
import re
import time
from pathlib import Path

from copilot.agent import CITE_RE, Copilot
from eval.run_retrieval_eval import QUESTIONS, is_relevant

OUT = Path(__file__).parent / "results_agent.json"


def main():
    c = Copilot()
    rows = []
    for q in QUESTIONS["answerable"] + QUESTIONS["unanswerable"]:
        t0 = time.time()
        a = c.ask(q["question"])
        cited = sorted(set(CITE_RE.findall(a.answer)))
        row = {"id": q["id"], "split": q["split"], "question": q["question"], "answerable": "relevant" in q,
               "refused": a.refused, "cited": cited, "seconds": round(time.time() - t0, 1),
               "fallback_summary": any(s["tool"] == "fallback_summary" for s in a.trace),
               "llm_error": a.llm_error,
               "asked_back": bool(a.clarification),
               "answer": a.answer}
        if row["answerable"] and cited:
            rel = [i for i in cited if is_relevant(c.retriever.df.loc[i, "problem"], q["relevant"])]
            row["cited_relevant"] = len(rel)
        rows.append(row)
        print(f"{q['id']} {'LLM-ERROR ' if a.llm_error else ''}{'ASKED-BACK ' if a.clarification else ''}refused={a.refused!s:5} cited={len(cited):2d} "
              f"relevant={row.get('cited_relevant', '-')!s:>2} {row['seconds']:5.1f}s  {q['question']}")

    # Rows where the LLM was unavailable measure search-only mode, not the agent: exclude them.
    errored = [r["id"] for r in rows if r["llm_error"]]
    ans = [r for r in rows if r["answerable"] and not r["llm_error"]]
    una = [r for r in rows if not r["answerable"] and not r["llm_error"]]
    n_cited = sum(len(r["cited"]) for r in ans)
    n_rel = sum(r.get("cited_relevant", 0) for r in ans)
    summary = {
        "answered": f"{sum(not r['refused'] and not r['asked_back'] for r in ans)}/{len(ans)}",
        "asked_back": [r["id"] for r in rows if r["asked_back"]],
        "citation_precision": f"{n_rel}/{n_cited} = {n_rel / n_cited:.2f}" if n_cited else "n/a",
        "correct_refusals": f"{sum(r['refused'] for r in una)}/{len(una)}",
        "fallback_summaries": sum(r["fallback_summary"] for r in rows),
        "median_seconds": sorted(r["seconds"] for r in rows)[len(rows) // 2],
        "excluded_llm_errors": errored,
    }
    print("\n", json.dumps(summary, indent=2))
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
