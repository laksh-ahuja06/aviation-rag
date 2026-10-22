"""The copilot: a LangGraph tool-calling loop, followed by a deterministic trust layer.

    agent <-> tools   (rewrite, search, read, count; at most MAX_AGENT_STEPS rounds)
       |
    finalize: evidence gate -> citation verifier -> fix leaderboard
"""
import json
import time
import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from copilot import config
from copilot.labels import COMPONENT_NAMES, FAULT_TYPE_NAMES
from copilot.leaderboard import ActionLabels, build_leaderboard
from copilot.llm import chat_model, with_fallback
from copilot.retrieval import Retriever
from copilot.tools import RunState, make_tools

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
CITE_RE = re.compile(r"\b(1\d{5})\b")

SYSTEM_PROMPT = f"""You are a maintenance copilot for piston aircraft engines. You answer ONLY from
past logbook records that your tools return. A confident wrong answer is worse than no answer.

How to work:
1. Call search_records with the technician's fault description. If evidence is weak, search again
   with different wording: technical terms, synonyms, or the shorthand a mechanic would write
   (e.g. "redline" -> "overspeed rpm", "shaking" -> "rough vibration"). Try at most 3 searches.
2. If the user names a part or fault type, you may filter.
   component is one of: {COMPONENT_NAMES}
   fault_type is one of: {FAULT_TYPE_NAMES}
3. For "how often / how many / recurring" questions, call count_similar.
4. Use get_record only if you need to re-read a specific record.

Answer rules:
- Use ONLY records marked STRONG. If after your searches no record is STRONG, or the STRONG
  records are about a different fault, reply with exactly: {INSUFFICIENT}
- Format: one summary sentence, then "Likely causes:" bullets, then "Actions that fixed it:" bullets.
  Every bullet ends with its sources, e.g. [rec 100123, rec 100456]. Cite only record IDs that a tool returned.
- For count questions, the summary sentence gives the exact count from count_similar, and the
  bullets list the most frequent fixes with example record IDs. Bullets are still required.
- Do not add steps, torque values, part numbers or procedures that are not in the cited records.
- Keep it under 150 words. The UI shows a ranked fix table and the records separately."""

VERIFY_PROMPT = """You check a maintenance answer against its cited logbook records.
QUESTION: {question}

For each numbered line, decide if it is supported: the cited records must (a) directly support what
the line claims and (b) describe the same kind of fault the QUESTION asks about.
A line is unsupported if it states a cause, action, number or procedure not present in its cited
records, or if its records are about a different fault than the question.

{lines}

Return JSON: a list of {{"line": <number>, "supported": true|false, "reason": "<short>"}}."""


@dataclass
class CopilotAnswer:
    question: str
    answer: str
    refused: bool
    leaderboard: dict
    evidence: list[dict]           # gated records, cited ones first
    trace: list[dict] = field(default_factory=list)
    verifier: list[dict] = field(default_factory=list)
    llm_error: str | None = None   # set when the LLM was unavailable (search-only mode)


def _text(msg) -> str:
    """Gemini 3 models may return content as a list of parts; .text joins them."""
    t = getattr(msg, "text", None)
    t = t() if callable(t) else t
    return (t if isinstance(t, str) else str(msg.content)).strip()


class Copilot:
    def __init__(self, retriever: Retriever | None = None):
        self.retriever = retriever or Retriever()
        self.labels = ActionLabels(self.retriever.df.reset_index(drop=True))
        self.primary, self.fallbacks = chat_model()
        self.verifier_llm = with_fallback(self.primary, self.fallbacks)

    # --- agent loop -------------------------------------------------------------

    def _graph(self, state: RunState):
        tools = make_tools(self.retriever, self.labels, state)
        llm = with_fallback(self.primary, self.fallbacks, tools)

        def agent(s: MessagesState):
            t0 = time.time()
            msg = llm.invoke([SystemMessage(SYSTEM_PROMPT)] + s["messages"])
            model = (msg.response_metadata or {}).get("model_name", "?")
            calls = ", ".join(c["name"] for c in msg.tool_calls) or "final answer"
            state.trace.append({"tool": "llm", "summary": f"{model} -> {calls} ({time.time() - t0:.1f}s)"})
            return {"messages": [msg]}

        def route(s: MessagesState):
            last = s["messages"][-1]
            rounds = sum(1 for m in s["messages"] if isinstance(m, AIMessage) and m.tool_calls)
            if getattr(last, "tool_calls", None) and rounds <= config.MAX_AGENT_STEPS:
                return "tools"
            return END

        g = StateGraph(MessagesState)
        g.add_node("agent", agent)
        g.add_node("tools", ToolNode(tools))
        g.add_edge(START, "agent")
        g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
        g.add_edge("tools", "agent")
        return g.compile()

    def ask(self, question: str) -> CopilotAnswer:
        state = RunState(question=question)
        try:
            out = self._graph(state).invoke({"messages": [HumanMessage(question)]},
                                            {"recursion_limit": 2 * config.MAX_AGENT_STEPS + 4})
            last = out["messages"][-1]
            if getattr(last, "tool_calls", None):  # step budget ran out mid-search: force an answer
                last = self.verifier_llm.invoke([SystemMessage(SYSTEM_PROMPT)] + out["messages"][:-1]
                                                + [HumanMessage("Stop searching. Write the final answer now.")])
            draft = _text(last)
            state.trace.append({"tool": "agent", "summary": f"draft answer ({len(draft)} chars)"})
        except Exception as e:
            # LLM quota/outage. Never invent: fall back to search-only mode, where retrieval,
            # the evidence gate and the leaderboard still work with zero LLM calls.
            state.llm_error = f"{type(e).__name__}: {str(e)[:200]}"
            state.trace.append({"tool": "agent", "summary": f"LLM unavailable ({state.llm_error})"})
            res = self.retriever.search(question, k=20)
            state.remember(res.hits)
            state.trace.append({"tool": "search_records", "args": {"query": question},
                                "normalized_query": res.normalized_query,
                                "summary": f"search-only mode: {len(res.hits)} hits, {len(res.strong_hits)} strong"})
            draft = None
        return self._finalize(question, draft, state)

    # --- trust layer ------------------------------------------------------------

    def _finalize(self, question: str, draft: str, state: RunState) -> CopilotAnswer:
        # 1. Evidence gate. A record is evidence if it cleared the threshold for the original
        #    question, OR it was STRONG for one of the agent's rewritten queries and is still
        #    loosely related to the original (the rewrite "redline" -> "overspeed" needs this;
        #    a rewrite that wanders off-topic doesn't pass the floor).
        scores = self.retriever.rescore(question, list(state.seen))
        gated = {}
        for i, s in scores.items():
            h = state.seen[i]
            if s >= config.EVIDENCE_THRESHOLD or (h["strong"] and s >= config.ORIGINAL_QUESTION_FLOOR):
                gated[i] = {**h, "score": round(max(s, h["score"]), 3), "score_vs_question": round(s, 3)}
        state.trace.append({"tool": "evidence_gate",
                            "summary": f"{len(gated)} of {len(state.seen)} retrieved records pass "
                                       f"(threshold {config.EVIDENCE_THRESHOLD}, "
                                       f"floor vs question {config.ORIGINAL_QUESTION_FLOOR})"})
        evidence = sorted(gated.values(), key=lambda h: -h["score"])
        board = build_leaderboard(self.labels, [h["ident"] for h in evidence])

        if not gated or (draft is not None and INSUFFICIENT in draft):
            reason = "no record passed the evidence gate" if not gated else "the agent found no matching records"
            return CopilotAnswer(question, f"Insufficient evidence: {reason}. The logbook has no "
                                 "sufficiently similar past fault, so no fix is suggested.",
                                 True, {"n_records": 0, "rows": []}, [], state.trace,
                                 llm_error=state.llm_error)

        if draft is None:  # search-only mode
            answer = self._records_summary(board, "The AI assistant is unavailable (Gemini quota or outage), "
                                                  "so this is a search-only result")
            return CopilotAnswer(question, answer, False, board, evidence, state.trace, llm_error=state.llm_error)

        # 2. Citation verifier: deterministic ID check, then an LLM support check.
        answer, report = self._verify(draft, gated, state)
        if not answer:
            # Strong evidence exists but the LLM's wording couldn't be verified: answer straight
            # from the records rather than refusing. Every line is a count over cited records.
            answer = self._records_summary(board)
            state.trace.append({"tool": "fallback_summary",
                                "summary": "draft failed verification; answered from leaderboard records"})
        cited = set(CITE_RE.findall(answer))
        evidence.sort(key=lambda h: (h["ident"] not in cited, -h["score"]))
        return CopilotAnswer(question, answer, False, board, evidence, state.trace, report)

    @staticmethod
    def _records_summary(board: dict, why: str = "The AI-written answer could not be verified") -> str:
        lines = [f"{why}: here is what {board['n_records']} similar past records show.",
                 "Actions recorded for this fault:"]
        for row in board["rows"][:5]:
            ids = ", ".join(f"rec {i}" for i in row["idents"][:3])
            lines.append(f"- {row['label']}: {row['count']} record{'s' if row['count'] != 1 else ''} [{ids}]")
        return "\n".join(lines)

    def _verify(self, draft: str, gated: dict, state: RunState) -> tuple[str, list[dict]]:
        lines = [l for l in draft.splitlines() if l.strip()]
        is_bullet = lambda l: bool(re.match(r"\s*([-*•]|\d+[.)])\s", l))
        report, kept, dropped_ids = [], [], set()
        for idx, line in enumerate(lines):
            ids = CITE_RE.findall(line)
            bad = [i for i in ids if i not in gated]
            if bad:  # strip citations to records that never passed the gate
                dropped_ids.update(bad)
                for b in bad:
                    line = re.sub(rf",?\s*(rec\s*)?{b}", "", line)
                line = re.sub(r"\[\s*,?\s*\]", "", line)
            good = [i for i in ids if i in gated]
            if is_bullet(line) and not good:
                report.append({"line": line.strip(), "kept": False, "reason": "no valid citation"})
                continue
            kept.append((idx, line, good))
        if dropped_ids:
            state.trace.append({"tool": "citation_check",
                                "summary": f"removed citations to ungated/unknown records: {sorted(dropped_ids)}"})

        claims = [(n, line, ids) for n, (_, line, ids) in enumerate(kept) if ids]
        verdicts = {}
        if claims:
            block = "\n\n".join(
                f"Line {n}: {line.strip()}\nCited records:\n" + "\n".join(
                    f"  [rec {i}] PROBLEM: {gated[i]['problem']} | ACTION: {gated[i]['action']}" for i in ids)
                for n, line, ids in claims)
            try:
                raw = _text(self.verifier_llm.invoke(VERIFY_PROMPT.format(question=state.question, lines=block)))
                raw = raw[raw.find("["): raw.rfind("]") + 1]
                verdicts = {int(v["line"]): v for v in json.loads(raw)}
                state.trace.append({"tool": "citation_verifier",
                                    "summary": f"checked {len(claims)} cited lines, "
                                               f"{sum(not v['supported'] for v in verdicts.values())} unsupported"})
            except Exception as e:
                state.trace.append({"tool": "citation_verifier",
                                    "summary": f"LLM check skipped ({type(e).__name__}); ID check still applied"})

        final = []
        for n, (_, line, ids) in enumerate(kept):
            v = verdicts.get(n)
            if v and not v.get("supported", True):
                report.append({"line": line.strip(), "kept": False, "reason": v.get("reason", "unsupported")})
                continue
            report.append({"line": line.strip(), "kept": True, "reason": v.get("reason", "") if v else ""})
            final.append(line)
        if not any(CITE_RE.search(l) for l in final):
            return "", report
        return "\n".join(final), report
