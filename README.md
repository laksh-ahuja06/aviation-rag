# Maintenance Copilot: Agentic RAG over Aviation Logbooks

**NexAI Hack 2026 · Track 1 (Agentic AI & Intelligent Automation) · Problem Statement 1.3**

> ⚠️ **Disclaimer:** This is a decision-support prototype built for a hackathon. Its answers are **not approved maintenance procedures** and must not be used for airworthiness decisions. Always follow the aircraft maintenance manual and your organisation's approved procedures.

- **Live demo:** **[maintainance-copilot.streamlit.app](https://maintainance-copilot.streamlit.app/)** (Streamlit Community Cloud). The app sleeps after 12 hours without visitors; the first visit after that takes about a minute to wake it.
- **Demo video:** _TBD_
- **Development log:** [DEV_LOG.md](DEV_LOG.md)

---

## Problem statement

When a technician sees a fault, the fastest fix is often buried in years of past logbook entries. Logbook text is terse, abbreviation-heavy and misspelled (`CYL 2 LOWER PLUG FOULED`, `R&R`, `A/C`), so plain embedding search misses obvious matches. In maintenance, a confident wrong answer is worse than none.

**Goal:** a copilot that takes a fault description, finds similar past records, summarises the likely causes and the actions that fixed them, **cites every record ID it relied on**, and says **"insufficient evidence"** instead of inventing a procedure.

## Solution overview

- **Text normalisation** expands abbreviations and fixes spelling, while the raw text is kept for display.
- **Hybrid retrieval** combines BM25 keyword search with dense embeddings, merged by Reciprocal Rank Fusion (RRF).
- **An agentic loop** (LangGraph + Gemini) rewrites the query, runs several searches, reads records and decides whether it has enough evidence.
- **Trust layer:**
  - An evidence gate (cross-encoder) refuses when matches are weak.
  - A citation verifier drops claims that the cited records don't support.
- **Fix leaderboard (standout feature):** an evidence card above the answer that groups the actions taken in similar past cases and ranks them by frequency:
  ```
  Based on 9 similar records:
  • Replaced spark plugs        5 cases [rec 100112, 100487, …]
  • Cleaned / regapped plugs    3 cases [rec 100903, …]
  • Replaced ignition lead      1 case  [rec 102201]
  ```
- **Chat UI** shows the answer, the leaderboard, the cited records side by side, the agent's tool-call trace and the disclaimer.

## Demo walkthrough

Five questions that show each behaviour of the copilot (screenshots from the running app).

### 1. Answer with citations and a fix leaderboard
**"#3 rocker cover leaking, what usually fixes it?"**
- The leaderboard ranks past fixes: **"Replaced: rocker cover gasket", 9 of 10 similar records**.
- Every bullet in the answer cites its records, and the cited records appear on the right with their original logbook text.

![Answer with citations and fix leaderboard](docs/screenshots/2-answer-rocker-cover.png)

### 2. Refusal when the logbook has no evidence
**"GPS display is blank after power up"**
- The logbook covers piston engines only, so no record passes the evidence gate.
- The copilot says **"Insufficient evidence"** instead of inventing a fix.

![Refusal on an unanswerable question](docs/screenshots/1-refusal-gps.png)

### 3. Ask back when the question is ambiguous
**"plugs are worn"**
- "Plugs" matches two different parts in the logbook: **spark plugs** (ignition) and **baffle hole / access plugs** (cooling baffle).
- Instead of merging them into one answer, the copilot asks which one is meant, with one button per option.

![Ask-back with option buttons](docs/screenshots/4-ask-back-plugs.png)

### 4. The clarified answer
**"plugs are worn (I mean: Baffle hole / access plugs)"**
- After the technician picks an option, the answer covers only baffle plugs.
- Causes (wear, chafing that enlarges the holes) and fixes (replace the plug, or patch/rivet the hole) each cite their records.

![Answer after clarification](docs/screenshots/5-clarified-baffle-plugs.png)

### 5. Verifier fallback: never show an unverified answer
**"pipeline leak"**
- The search found 3 intake-pipe gasket records, but the AI-written answer failed the citation verifier.
- Instead of showing it, the copilot shows an **auto-summary built directly from the records**: "Replaced: intake gasket, 3 records".

![Auto-summary when the AI answer can't be verified](docs/screenshots/3-auto-summary-pipeline-leak.png)

Every answer also has an expandable **agent trace** showing each tool call, the normalised query, the evidence gate result and the citation verifier's decisions.

## Architecture

![Maintenance Copilot architecture](docs/architecture.svg)

### Agent capabilities
- Rephrase or expand the query (e.g. "cylinder" → "CYL")
- Run several searches and merge the results
- Filter by part, component or fault type
- Decide the evidence is insufficient and search again
- Count or summarise across many records ("how often does this fault recur?")
- **Ask back instead of guessing** when the wording fits two different parts. For example, "plugs are worn" matches both *spark plugs* and *baffle hole plugs*, so the agent asks which one you mean and shows option buttons.

### Agent tools
| Tool | Purpose |
|---|---|
| `search_records(query, k, filters)` | Hybrid BM25 + dense search fused with RRF; also lists which **parts** the strong records acted on (an ambiguity hint) |
| `get_record(ident)` | Fetch a full record by its `IDENT` |
| `count_similar(query, filters)` | Exact count of matching records across the whole logbook, plus the most frequent fixes |
| `ask_clarification(question, options)` | Stop and ask the technician which meaning they intended (2–4 options) |

## Technology stack

| Layer | Tools |
|---|---|
| Language | Python 3.14 |
| Data | pandas, numpy |
| Normalisation | symspellpy, MaintNet abbreviation list |
| Keyword search | rank-bm25 |
| Embeddings and re-ranking | sentence-transformers (`BAAI/bge-small-en-v1.5`, `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| Vector search | numpy exact cosine search over a saved embedding matrix (replaced Chroma, see DEV_LOG) |
| Agent | LangGraph, langchain-core |
| LLM | Google Gemini (free tier) via langchain-google-genai |
| UI | Streamlit |
| Hosting | Hugging Face Spaces |

All pinned versions are in [requirements.txt](requirements.txt).

## Setup and running

**Prerequisites:** Python 3.14 and a free Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).

1. Create and activate the virtual environment:
   ```bash
   python3 -m venv venv
   ```
   ```bash
   source venv/bin/activate
   ```
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Configure your credentials:
   ```bash
   cp .env.example .env
   ```
   Then edit `.env` and set `GOOGLE_API_KEY`. `GEMINI_MODEL` defaults to `gemini-3.5-flash`, with `gemini-flash-lite-latest` as an automatic fallback when the main model is overloaded.
4. Put the dataset at `Aircraft_Annotation_DataFile.csv` in the project root.
5. Build the normalised records and the vector index (about 15 s):
   ```bash
   python -m scripts.build_index
   ```
6. *(Optional, already committed)* Re-label the actions for the fix leaderboard (about 50 Gemini calls):
   ```bash
   python -m scripts.label_actions
   ```
7. Run the app:
   ```bash
   streamlit run app.py
   ```
   If the index is missing, the app builds it on first start.
8. Run the evaluations:
   ```bash
   python -m eval.run_retrieval_eval --split test
   ```
   ```bash
   python -m eval.run_agent_eval
   ```

### Deployment
**Streamlit Community Cloud (free; the live demo runs here):**
- Deploy from this GitHub repo, with `app.py` as the entry file and Python 3.14 selected under *Advanced settings*.
- Put `GOOGLE_API_KEY = "..."` in the app's **Secrets** box. `app.py` copies Streamlit secrets into environment variables, so no `.env` file is needed. Without a key, the app runs in search-only mode.
- The embedding index isn't committed: the app builds it on first start (about 20 s), and the models download from Hugging Face. Peak memory use is about 610 MB, within the platform's limit.

**Docker (any container host, or Hugging Face Spaces with a paid plan):**
- Since July 2026, Hugging Face requires a PRO plan for Docker and Gradio Spaces on CPU, so this route isn't free there.
- The `Dockerfile` uses Python 3.14 and CPU-only PyTorch, and builds the index and downloads the models at build time.
- For a Hugging Face Space, add the Space header block (`sdk: docker`, `app_port: 8501`) to the top of the README, and set `GOOGLE_API_KEY` as a Space secret.
- Test the same image locally (with your key in `.env`):
  ```bash
  docker build -t maintenance-copilot .
  ```
  ```bash
  docker run --rm -p 8501:8501 --env-file .env maintenance-copilot
  ```

## Project structure

```
app.py                     Streamlit chat UI (answer, leaderboard, records, trace)
copilot/
  config.py                paths, model names, thresholds
  normalize.py             abbreviation expansion + corpus spell-fixer + BM25 tokens
  labels.py                rule-based component / fault-type / action-type labels
  retrieval.py             BM25 + dense (numpy) search, RRF fusion, cross-encoder rerank
  tools.py                 agent tools: search_records, get_record, count_similar, ask_clarification
  agent.py                 LangGraph loop + evidence gate + citation verifier
  leaderboard.py           fix leaderboard (pandas group-by over offline labels)
  llm.py                   Gemini with automatic fallback model
  resources/english_words.txt   words the spell-fixer must never "correct"
scripts/
  build_index.py           offline: CSV -> data/records.csv + index/embeddings.npy
  label_actions.py         offline: Gemini labels each action -> data/action_labels.csv
eval/
  test_questions.json      15 answerable + 5 unanswerable questions (written before tuning)
  run_retrieval_eval.py    hit@5 per retriever + score distributions
  run_agent_eval.py        end-to-end: answered, citation precision, refusals
data/                      generated records + action labels (committed)
```

## Dataset attribution and licence

- **Dataset:** MaintNet aviation maintenance logbook: 6,169 records (`IDENT`, `PROBLEM`, `ACTION`) of free-text problems and the actions taken, from the University of North Dakota Aviation Program.
- **Source:** [MaintNet](https://people.rit.edu/fa3019/MaintNet/). A Kaggle mirror, "Aircraft Historical Maintenance Dataset (2012–2017)", also exists.
- **Citation:** Akhbardeh, F., Desell, T., & Zampieri, M. (2020). *MaintNet: A Collaborative Open-Source Library for Predictive Maintenance Language Resources.* Proceedings of COLING 2020: System Demonstrations.
- **Licence:** open research library. See the MaintNet page for the terms of use.

## Evaluation

Following the PS 1.3 brief, the test questions were written **before** any tuning:
- **Retrieval hit@5** on 15 team-written question → record pairs
- **Citation precision:** the share of cited records that actually support the answer
- **Correct refusals** on 5 questions the data cannot answer

**Retrieval hit@5** (`python -m eval.run_retrieval_eval`). Thresholds were chosen on the tune split only:

| Split | BM25 | Dense | Hybrid (RRF) | Hybrid + rerank |
|---|---|---|---|---|
| tune (8 questions) | 7/8 | 7/8 | 7/8 | 7/8 |
| test (7 questions) | 6/7 | 6/7 | 6/7 | 6/7 |

**Refusals at the retrieval level:** all 5 unanswerable questions fall below the evidence threshold of 0.0 (best cross-encoder scores −8.67, −9.30, −2.63, −3.89, −1.10).

**End-to-end agent evaluation** (`python -m eval.run_agent_eval`, full agent with live Gemini, 22 Oct 2026). Raw results are in [eval/results_agent.json](eval/results_agent.json).

| Metric | Result |
|---|---|
| **Citation precision**: cited records whose problem matches the question's relevance rule | **66 / 66 (100%)** |
| **Correct refusals** on unanswerable questions | **3 / 3** (2 more excluded, see below) |
| Answerable questions answered with citations | **9 / 13** |
| ↳ asked back instead (A07 "big magneto drop", A14 "engine shaking badly") | 2 |
| ↳ refused (A06 "idles too fast", A15 "alternator not charging") | 2 |
| Median time per question | 5.5 s |

- **Excluded:** 4 of the 20 questions (A10, A11, U02, U03) hit a Gemini **rate limit** (`429 RESOURCE_EXHAUSTED`) during the run. They are left out of every metric above rather than counted as answers or refusals.
- **How to read the misses:**
  - The two refusals are counted as failures under the pre-written test definition. The data partly supports them, though: the closest records describe an alternator *cooling hose* chafing and an engine *quitting* at idle, which is not the same fault as the question.
  - Both ask-backs were the agent's own choice. "Engine shaking badly" really does have several different causes in the logbook.

**Limitations of this evaluation:**
- 20 questions is a small set, and 4 of them were lost to rate limits.
- On this set the four retrievers perform identically, so the benefit of hybrid search is **not demonstrated** here.
- Citation precision is measured with the same regex relevance rules as hit@5, not by human judgement.
- More detail on the misses is in [DEV_LOG.md](DEV_LOG.md).

## Responsible AI and limitations
- Answers are grounded only in the logbook records and always cite `IDENT`s; when evidence is weak, the copilot refuses.
- The dataset covers one training fleet (2012–2017), so its coverage of aircraft types and faults is limited.
- Short records make similarity scores bunch together. The refusal threshold is tuned on a held-out test set and documented in [DEV_LOG.md](DEV_LOG.md).

## AI tools and dependencies declared
- **AI coding assistant:** Claude Code (Anthropic), used for architecture review, scaffolding and documentation. Details are in [DEV_LOG.md](DEV_LOG.md).
- **Runtime LLM:** Google Gemini (free tier).
- **Open-source libraries:** see [requirements.txt](requirements.txt).

## Team

| Name | Contact |
|---|---|
| Laksh Ahuja (Solo) | ahuja.laksh@gmail.com |
