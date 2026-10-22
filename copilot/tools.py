"""Agent tools. `make_tools` builds a fresh set per question, bound to a RunState
that records every retrieved record and every call for the UI's trace panel."""
from dataclasses import dataclass, field
from typing import Literal, Optional

from langchain_core.tools import tool

from copilot import config
from copilot.labels import COMPONENT_NAMES, FAULT_TYPE_NAMES
from copilot.leaderboard import ActionLabels, build_leaderboard
from copilot.retrieval import Retriever

Component = Literal[tuple(COMPONENT_NAMES)]
FaultType = Literal[tuple(FAULT_TYPE_NAMES)]


@dataclass
class RunState:
    question: str = ""
    llm_error: str | None = None
    seen: dict[str, dict] = field(default_factory=dict)   # ident -> best hit seen by any tool
    trace: list[dict] = field(default_factory=list)
    clarification: dict | None = None                       # {'question': ..., 'options': [...]}
    allow_clarify: bool = True                              # False once the technician has clarified

    def remember(self, hits: list[dict]):
        for h in hits:
            if h["ident"] not in self.seen or h["score"] > self.seen[h["ident"]]["score"]:
                self.seen[h["ident"]] = h


def _fmt_hit(h: dict) -> str:
    tag = "STRONG" if h["strong"] else "weak"
    return f"[rec {h['ident']}] ({tag}, score {h['score']:.1f}) PROBLEM: {h['problem']} | ACTION: {h['action']}"


def make_tools(retriever: Retriever, labels: ActionLabels, state: RunState, allow_clarify: bool = True):

    @tool
    def search_records(query: str, component: Optional[Component] = None,
                       fault_type: Optional[FaultType] = None, k: int = 10) -> str:
        """Hybrid keyword + semantic search over the aviation maintenance logbook.
        Abbreviations and spelling in the query are normalised automatically.
        Optionally filter by component and/or fault_type. Returns records marked STRONG
        (relevant enough to cite) or weak (do not cite)."""
        res = retriever.search(query, k=min(max(k, 1), 20), component=component, fault_type=fault_type)
        state.remember(res.hits)
        n_strong = len(res.strong_hits)
        state.trace.append({"tool": "search_records",
                            "args": {"query": query, "component": component, "fault_type": fault_type, "k": k},
                            "normalized_query": res.normalized_query,
                            "summary": f"{len(res.hits)} hits, {n_strong} strong, best score {res.best_score:.2f}"})
        if not res.hits:
            return f"No records found for normalised query '{res.normalized_query}'. Try different wording or remove filters."
        verdict = (f"{n_strong} of {len(res.hits)} records are STRONG evidence." if n_strong else
                   "NO strong evidence: none of these records match well. Rephrase and search again, "
                   "or conclude INSUFFICIENT_EVIDENCE.")
        lines = [f"Normalised query: {res.normalized_query}", verdict]
        parts = labels.parts([h["ident"] for h in res.strong_hits])
        if parts:
            lines.append("Parts acted on in the STRONG records: "
                         + ", ".join(f"{p} x{n}" for p, n in parts.most_common(6)))
        return "\n".join(lines + [_fmt_hit(h) for h in res.hits])

    @tool
    def get_record(ident: str) -> str:
        """Fetch one full logbook record by its record ID (e.g. '100004')."""
        r = retriever.get(ident)
        state.trace.append({"tool": "get_record", "args": {"ident": ident},
                            "summary": "found" if r else "not found"})
        if not r:
            return f"No record with ID {ident}."
        return (f"[rec {r['ident']}] PROBLEM: {r['problem']} | ACTION: {r['action']} | "
                f"component: {r['component']} | fault type: {r['fault_type']}")

    @tool
    def count_similar(query: str, component: Optional[Component] = None,
                      fault_type: Optional[FaultType] = None) -> str:
        """Count how many logbook records report a fault similar to `query` and break the
        fixes down by frequency. Use for 'how often / how many times / does this recur' questions."""
        res = retriever.count_matches(query, component=component, fault_type=fault_type)
        strong = res.strong_hits
        state.remember(strong)
        board = build_leaderboard(labels, [h["ident"] for h in strong])
        state.trace.append({"tool": "count_similar",
                            "args": {"query": query, "component": component, "fault_type": fault_type},
                            "normalized_query": res.normalized_query,
                            "summary": f"{len(strong)} of {len(retriever.df)} records match (exact count)"})
        if not strong:
            return "No records strongly match this fault, so it cannot be counted."
        lines = [f"{len(strong)} records strongly match '{res.normalized_query}'"
                 + " (exact count over the whole logbook).",
                 f"Distinct problem descriptions: {len({h['problem'] for h in strong})}.",
                 "Most frequent fixes:"]
        for row in board["rows"]:
            ids = ", ".join(row["idents"][:5]) + (", ..." if row["count"] > 5 else "")
            lines.append(f"- {row['label']}: {row['count']} records [rec {ids}]")
        return "\n".join(lines)

    @tool
    def ask_clarification(question: str, options: list[str]) -> str:
        """Ask the technician ONE short question instead of answering, when their wording fits
        two or more different parts or faults in the STRONG records (e.g. 'plugs' could mean
        spark plugs or baffle hole plugs). Give 2-4 short options, each naming one meaning."""
        options = [o.strip() for o in options if o.strip()][:4]
        state.clarification = {"question": question.strip(), "options": options}
        state.trace.append({"tool": "ask_clarification", "args": {"question": question, "options": options},
                            "summary": "asked the technician to clarify; stopped before answering"})
        return "Question sent to the technician. Stop now and wait for their reply."

    tools = [search_records, get_record, count_similar]
    return tools + [ask_clarification] if allow_clarify else tools
