"""Maintenance Copilot: Streamlit chat UI.
    run the application:
    ./venv/bin/python -m streamlit run app.py
"""
import os
import re

import streamlit as st

# Streamlit Community Cloud passes the API key via its Secrets setting (st.secrets), not a .env
# file. Copy string secrets into the environment before copilot/config.py reads them, including
# ones nested under a [section], and accept GEMINI_API_KEY as another name for the key.
try:
    for _k, _v in st.secrets.items():
        for _name, _val in (_v.items() if hasattr(_v, "items") else [(_k, _v)]):
            if isinstance(_val, str) and _val.strip():
                os.environ.setdefault(_name, _val.strip())
except Exception:  # no secrets file locally: .env is used instead
    pass
if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

from copilot import config
from copilot.agent import Copilot, CopilotAnswer

st.set_page_config(page_title="Maintenance Copilot", page_icon="🛠️", layout="wide")

DISCLAIMER = ("**Decision support only.** Answers summarise past logbook entries and are **not approved "
              "maintenance procedures**. Always follow the aircraft maintenance manual and your "
              "organisation's approved data before acting.")
EXAMPLES = [
    "#3 rocker cover leaking, what usually fixes it?",
    "How often does the intake gasket leak?",
    "Big mag drop during run-up",
    "Flight data shows the engine went past redline rpm",
    "GPS display is blank after power up",
]
CITE_RE = re.compile(r"\b(1\d{5})\b")


@st.cache_resource(show_spinner="Loading models and index (first run only)…")
def load_copilot() -> Copilot:
    if not config.EMBEDDINGS_NPY.exists():  # fresh deployment: the index isn't committed, build it (~1 min)
        from scripts.build_index import main as build_index
        build_index()
    return Copilot()


def render_leaderboard(board: dict):
    if not board["rows"]:
        return
    with st.container(border=True):
        st.markdown(f"**🏆 Fix leaderboard**: based on **{board['n_records']}** similar past records")
        top = board["rows"][0]["count"]
        for row in board["rows"]:
            c1, c2, c3 = st.columns([4, 3, 5])
            c1.markdown(row["label"])
            c2.progress(row["count"] / top, text=f"{row['count']} case{'s' if row['count'] != 1 else ''}")
            ids = row["idents"]
            c3.caption("rec " + ", ".join(ids[:6]) + (f" … +{len(ids) - 6} more" if len(ids) > 6 else ""))


def render_records(ans: CopilotAnswer, limit: int = 12):
    cited = set(CITE_RE.findall(ans.answer))
    shown = ans.evidence[:limit]
    st.markdown(f"**📄 Evidence records** ({len(cited)} cited, {len(ans.evidence)} passed the gate)")
    for h in shown:
        tag = "✅ cited" if h["ident"] in cited else "supporting"
        with st.container(border=True):
            st.markdown(f"**rec {h['ident']}** · {tag} · relevance {h['score']:.1f}")
            st.markdown(f"**Problem:** {h['problem']}  \n**Action:** {h['action']}")
            st.caption(f"{h['component']} · {h['fault_type']}")
    if len(ans.evidence) > limit:
        st.caption(f"+ {len(ans.evidence) - limit} more records in the leaderboard counts.")


def render_trace(ans: CopilotAnswer):
    with st.expander("🔎 Agent trace (tool calls, evidence gate, citation check)"):
        for i, step in enumerate(ans.trace, 1):
            line = f"{i}. `{step['tool']}`"
            if step.get("args"):
                args = {k: v for k, v in step["args"].items() if v not in (None, "")}
                line += f" {args}"
            st.markdown(line)
            if step.get("normalized_query"):
                st.caption(f"normalised query → {step['normalized_query']}")
            st.caption(step.get("summary", ""))
        dropped = [v for v in ans.verifier if not v["kept"]]
        if dropped:
            st.markdown("**Lines removed by the citation verifier:**")
            for v in dropped:
                st.caption(f"✂️ {v['line']}: {v['reason']}")


def to_markdown(answer: str) -> str:
    """Put blank lines around paragraphs so Markdown doesn't merge headings into sentences."""
    out, prev_bullet = [], False
    for line in answer.splitlines():
        bullet = bool(re.match(r"\s*([-*•]|\d+[.)])\s", line))
        if out and not (bullet and prev_bullet):
            out.append("")
        out.append(re.sub(r"^\s*•", "-", line))
        prev_bullet = bullet
    return "\n".join(out)


def render_clarification(ans: CopilotAnswer, key: int):
    """The agent asked back instead of guessing: one button per meaning."""
    st.info(f"**Before I answer:** {ans.clarification['question']}", icon="❓")
    cols = st.columns(len(ans.clarification["options"]) or 1)
    for i, (col, opt) in enumerate(zip(cols, ans.clarification["options"])):
        if col.button(opt, key=f"clarify-{key}-{i}", use_container_width=True):
            # Re-ask with the clarified wording; ask-back is disabled for it, so no loops.
            st.session_state.pending = f"{ans.question} (I mean: {opt})"
            st.session_state.pending_clarified = True
    if ans.evidence:
        with st.expander(f"Why I'm asking: {len(ans.evidence)} matching records cover different parts"):
            for h in ans.evidence[:8]:
                st.markdown(f"**rec {h['ident']}**: {h['problem']}  \n→ {h['action']}")
    render_trace(ans)


def render_answer(ans: CopilotAnswer, key: int):
    if ans.clarification:
        render_clarification(ans, key)
        return
    if ans.llm_error:
        st.info("AI assistant unavailable (Gemini quota or outage). Showing search-only results: "
                "retrieval, evidence gate and fix leaderboard still work without the LLM.", icon="📴")
    if ans.refused:
        st.error(ans.answer, icon="🛑")
        render_trace(ans)
        return
    left, right = st.columns([3, 2], gap="large")
    with left:
        render_leaderboard(ans.leaderboard)
        st.markdown(to_markdown(ans.answer))
    with right:
        render_records(ans)
    render_trace(ans)


# --- page -------------------------------------------------------------------------

st.title("🛠️ Maintenance Copilot")
st.caption("Agentic RAG over 6,169 aviation maintenance logbook records (MaintNet, UND Aviation). "
           "Every answer cites record IDs; weak evidence means no answer.")
st.warning(DISCLAIMER, icon="⚠️")

with st.sidebar:
    st.header("Try an example")
    for ex in EXAMPLES:
        if st.button(ex, use_container_width=True):
            st.session_state.pending = ex
    st.divider()
    st.markdown("**How it works**")
    st.markdown("1. Normalise abbreviations and spelling\n2. Hybrid BM25 + embedding search (RRF)\n"
                "3. Gemini agent rewrites and re-searches\n4. Cross-encoder evidence gate\n"
                "5. Citation verifier drops unsupported lines\n6. Fix leaderboard counts past actions")
    st.caption(f"LLM: {config.GEMINI_MODEL} (fallback {config.GEMINI_FALLBACK_MODEL}) · "
               f"evidence threshold {config.EVIDENCE_THRESHOLD}")
    if st.button("Clear chat"):
        st.session_state.history = []

copilot = load_copilot()
if copilot.llm_setup_error:
    st.sidebar.error("No Gemini API key found, so answers are search-only. Set GOOGLE_API_KEY in "
                     "the app's Secrets (Streamlit Cloud) or in .env (local).", icon="🔑")
st.session_state.setdefault("history", [])

for i, (q, ans) in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.markdown(q)
    with st.chat_message("assistant"):
        render_answer(ans, key=i)

question = st.chat_input("Describe the fault, e.g. 'R/H eng #2 cyl intake leaking'")
question = question or st.session_state.pop("pending", None)
clarified = st.session_state.pop("pending_clarified", False)
if question:
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("Searching the logbook…"):
            ans = copilot.ask(question, allow_clarify=not clarified)
        render_answer(ans, key=len(st.session_state.history))
    st.session_state.history.append((question, ans))
