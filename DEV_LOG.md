# Development Log

Running record of key decisions, AI assistance, experiments and rejected approaches, as encouraged by the NexAI Hack 2026 instructions.

---

## 21 Oct 2026: Kickoff and setup

### Problem selection
- Chose **PS 1.3, Maintenance Copilot: agentic RAG over logbooks** (Track 1, difficulty ★★★).
- Must-haves from the brief:
  1. Ingestion that normalises text (abbreviation expansion) and builds a hybrid BM25 + embedding index.
  2. An agent with at least two tools (`search_records`, `get_record`) that cites record IDs and refuses when matches are weak.
  3. A deployed chat UI showing the answer, the cited records side by side and the tool-call trace.

### Architecture designed
Pipeline (see the diagram in the README):

1. **Raw logbook CSV**: loaded with pandas; keep raw and normalised text.
2. **Normalise text**: expand abbreviations (MaintNet list), fix spelling with symspell.
3. **Hybrid index**: `rank_bm25` + Chroma with `bge-small-en-v1.5` (fallback: `all-MiniLM-L6-v2`).
4. **Agent loop**: LangGraph + Gemini tool calling; rewrite, search, read records; search tool fuses results with RRF in plain Python.
5. **Evidence gate**: cross-encoder `ms-marco-MiniLM-L-6-v2` on CPU; refuse if the evidence is weak.
6. **Citation verifier**: checks every claim against the cited records; drops unsupported lines.
7. **Fix leaderboard** (standout feature): groups the actions taken in similar past records and ranks them by count, e.g. "Replaced spark plugs: 5 cases [rec 112, 487, …]".
8. **Chat UI**: Streamlit, deployed on Hugging Face Spaces, with a disclaimer.

Agent behaviours to support: query expansion ("cylinder" → "CYL"), several searches with merged results, filters by part/component/fault type, re-searching when evidence is thin, and counting questions ("how often does this fault recur?").

### Dataset
- Put `Aircraft_Annotation_DataFile.csv` (MaintNet aviation logbook, University of North Dakota Aviation Program) in the project root.
- Profiled it:
  - 6,169 records, columns `IDENT`, `PROBLEM`, `ACTION`.
  - `IDENT` is unique and runs from 100001 to 106169; this is the ID we cite.
  - No empty `PROBLEM` or `ACTION` fields.
  - The file starts with a UTF-8 BOM, so read it with `encoding="utf-8-sig"`.
  - Only **3,595 unique `PROBLEM` texts**: the same fault often appears on several rows with different actions (e.g. 100001–100003). This matters for the leaderboard (counts per record vs per distinct event) and for retrieval (near-duplicate hits).
  - Text is upper-case, terse and abbreviation-heavy (`CYL 2 LOWER PLUG FOULED`, `A/C`, `R&R`).
- No part, component or fault-type columns exist, so filters will need derived labels.

### LLM choice
- **Gemini free tier** via `langchain-google-genai`, chosen for its tool-calling support and a generous free quota.
- Risk: requests-per-minute and requests-per-day limits. Mitigations planned: cap agent iterations, do deterministic work (abbreviation expansion, citation ID checks, leaderboard counting) in Python, and label actions offline once rather than per query.
- The API key goes in `.env` (git-ignored); `.env.example` is committed.

### Environment
- Created `venv/` with **Python 3.14.7** (Homebrew).
- Wrote `requirements.txt` with pinned versions. Ran `pip install --dry-run` to confirm that every package has a Python 3.14 wheel (torch 2.14.1, onnxruntime 1.31.0 and chromadb 1.5.9 all resolve).
- Added `.gitignore` (venv, `.env`, caches) and `.env.example`.

- Installed `requirements.txt` into the venv; `pip check` reports no broken requirements.
- Smoke test passed: Chroma, BM25, symspell, LangGraph and the Gemini imports all work. Downloaded and ran `bge-small-en-v1.5` (384-dim embeddings) and `ms-marco-MiniLM-L-6-v2`.
- Observations from the smoke test:
  - The cross-encoder returned **5.06** for ("spark plug fouled", "CYL 2 LOWER PLUG FOULED"). This confirms it outputs raw logits, not probabilities, so the evidence-gate threshold must be calibrated, not assumed to lie in 0–1.
  - `BM25Okapi` gave a score of 0 on a 2-document toy corpus. Its IDF formula goes to zero or negative for terms that appear in half or more of the documents. That's harmless on the full 6,169 records, but very common words (e.g. ENGINE) will contribute almost nothing.

### AI assistance
- Used **Claude Code** to review the architecture and list pitfalls before coding, read the hackathon brief, profile the dataset, and generate `requirements.txt`, `README.md`, this log, `.env.example` and `.gitignore`.

### Decisions and rejected approaches
| Decision | Why | Rejected alternative |
|---|---|---|
| Hybrid BM25 + dense search with RRF | Terse, abbreviated text defeats embeddings alone; RRF needs no score calibration | Embeddings only |
| Deterministic abbreviation expansion for query rewriting | Free, fast and reproducible | LLM-only query rewriting (costs quota, can hallucinate terms) |
| Corpus-built symspell dictionary | A general English dictionary "corrects" domain terms such as CYL and MAG | symspell's default English dictionary |
| Label actions offline once; count with pandas at runtime | Fast, deterministic, no rate limits during the demo | Asking the LLM to label actions on every query |
| Evidence gate threshold set on a held-out test set | The brief warns that scores bunch together on short records | A fixed, untested threshold |

---

## 21–22 Oct 2026: First build (MVP)

### Gemini setup
- `gemini-2.5-flash` (my first default) returned 404 "no longer available to new users". Listed the models the key can use instead.
- `gemini-3.8-flash` and `gemini-flash-latest` returned **503 "high demand"** on every call during the event. With retries, each agent step burned 30s+, and early answers took 45–97s.
- **Decision:** primary `gemini-3.5-flash` (about 3s per call), with an automatic fallback to `gemini-flash-lite-latest` (`copilot/llm.py`), `max_retries=1` and a 15s timeout, so an overload costs seconds rather than minutes.

### Test questions written first
- `eval/test_questions.json` has 15 answerable and 5 unanswerable questions. They were written **before any retrieval code existed**, as the brief asks.
- A record counts as relevant if its PROBLEM text matches the question's regex. This was defined using plain pandas, independent of the retriever.
- Unanswerable topics (GPS, pressurisation, pitot, hydraulics, windshield) were checked to have 0 matching records.
- The questions are split into **tune** (8 + 2), used to pick thresholds, and **test** (7 + 3), used only for reporting.

### Normalisation experiments
- The abbreviation map (`copilot/normalize.py`) was built by reading the 400 most frequent tokens and every short or slashed token, e.g. CYL, ENG, R/H, A/C, R&R, CK'D, IAW, ASSY, MAG and CHT. `#3` is reduced to `3`, so "cyl 3" and "#3 CYL" match.
- **Rejected: a corpus-only symspell.** The first version made 175 "corrections", including dangerous ones: CRANK→CRACK, **UNSERVICEABLE→SERVICEABLE** (a meaning flip), INJECTION→INDUCTION and LUBRICATED→FABRICATED. There was also a bug: `Verbosity.CLOSEST` returned the word itself, so the first run made zero corrections.
- **Fix:** never correct a real English word. The allowlist is the set of corpus words found in the system word list (base forms included, so TAPPED counts) and is saved to `copilot/resources/english_words.txt`. This leaves 87 corrections, all genuine typos (BAFFEL, REMOEVD, OVESPEED, ...).
- Queries hit the same problem: "blank" was corrected to "BLACK". Queries now only correct words of 6+ letters.

### Derived labels
- The dataset is entirely piston-engine records, and its text is cut off at about 60 characters.
- Rule-based component and fault-type labels (`copilot/labels.py`) drive the search filters. Only 309 of the 6,169 records are labelled "other" component.

### Retrieval results (`python -m eval.run_retrieval_eval`)
| Split | BM25 | Dense | Hybrid (RRF) | Hybrid + rerank |
|---|---|---|---|---|
| tune hit@5 | 7/8 | 7/8 | 7/8 | 7/8 |
| test hit@5 | 6/7 | 6/7 | 6/7 | 6/7 |

- Honest finding: all four retrievers score the same on this small set. After normalisation, most questions share vocabulary with the records, so BM25 alone already finds them. The hybrid setup is kept for robustness on free-text paraphrases, but **this test set does not prove it helps**.
- **A03 (tune) miss:** the top record "ENGINE RPM WAS 2740 FOR 3 SECONDS" *is* an overspeed, but the relevance rule written before tuning only accepted the word OVERSPEED. I left the gold label unchanged rather than editing the test after seeing results.
- **A10 (test) miss:** "exhaust pipe cracked" returns "#1 CYLINDER CRACKED". This is a genuine miss.

### Evidence gate threshold
- Cross-encoder logits on the tune split: answerable best scores ≥ **2.68**, unanswerable ≤ **−7.98**. I chose `EVIDENCE_THRESHOLD = 0.0` from inside this gap, before looking at the test split.
- Test split: all 3 unanswerable questions were refused (best scores −0.54, −1.10, −3.89). **U03 at −0.54 is uncomfortably close**, which is why the LLM verifier is a second check.
- **Rejected: gating only on the user's original wording.** "flight data shows the engine went past redline" was refused even though the agent's rewrite ("overspeed") found records scoring 1–3. The gate now accepts a record that was STRONG for an agent rewrite **and** scores ≥ −4.0 against the original question (`ORIGINAL_QUESTION_FLOOR`; tune-split unanswerable questions scored ≤ −7.98).
- The verifier now also checks that the cited records are about the **same fault as the question**, so an off-topic rewrite can't produce an answer.

### Fix leaderboard
- `scripts/label_actions.py` labelled all 5,122 unique (problem, action) pairs in 52 batched Gemini Flash-Lite calls, with 0 failures. Each pair gets an `action_type` from a fixed list plus a generic `part`.
- At runtime the leaderboard is a pandas group-by, so it makes no LLM calls.
- Part names are canonicalised ("rocker box cover gaskets" → "rocker cover gasket"). "Replaced: rocker cover" stays separate from "...gasket" on purpose: I checked, and in those records the whole cover really was replaced.
- `count_similar` originally capped at 100, and then 1000, retrieved records. Scoring all 3,437 unique problem texts with the cross-encoder takes about 1.2s, so counts are now **exact over the whole logbook** (e.g. 1,813 intake-leak records).

### Trust layer behaviour
- The agent's draft passes three checks: citations to records that never passed the gate are stripped, bullets left with no valid citation are dropped, and an LLM checks each cited line against its records.
- **Rejected: refusing whenever the draft fails verification.** Flash-Lite once answered a count question with one uncited sentence, which caused a false refusal despite 1,805 gated records. Now, if strong evidence exists but the draft fails, the app shows a deterministic **auto-summary** built from the leaderboard records. It still refuses when nothing passes the gate or the agent itself reports insufficient evidence.
- If the LLM errors, the app fails closed and refuses rather than inventing an answer.

### Gemini free-tier quota (a demo risk)
- The first end-to-end evaluation went wrong after 5 questions. Every later question "refused" in 0.7s because Gemini returned **429 RESOURCE_EXHAUSTED**, with quota ID `GenerateRequestsPerDayPerProjectPerModel-FreeTier` and **quotaValue 20**.
- So the free tier allows **20 requests per model per day**. The agent uses 3–5 calls per question, which means only about 5 questions per model per day. That run's results were discarded rather than reported: its "5/5 correct refusals" were LLM errors, not evidence-based refusals.
- Changes made:
  1. **Search-only mode:** if the LLM is unavailable, the app runs retrieval, the evidence gate and the leaderboard with zero LLM calls. It answers from records with a visible "AI offline" banner and still refuses when nothing passes the gate. I tested this with a deliberately invalid key.
  2. **Fallback chain across models,** each with its own daily quota: `gemini-3.5-flash` → `gemini-3.5-flash-lite` → `gemini-flash-lite-latest` → `gemini-3.1-flash-lite`.
  3. `eval/run_agent_eval.py` flags rows with LLM errors and excludes them from the metrics.
- After the changes, a real run (with Flash-Lite answering via the chain) answered "big magneto drop during run-up" from an exact count of **26 matching records**, with all 6 cited lines verified. It refused "pitot tube blocked" after 3 searches.

### Next steps
- [ ] Create a public GitHub repo and commit at least hourly.
- [ ] Re-run `python -m eval.run_agent_eval` after the quota resets, and report the results in the README.
- [x] `pip install -r requirements.txt` in the venv.
- [x] Write test questions before tuning.
- [x] Build ingestion, normalisation, hybrid index, agent, trust layer, leaderboard and UI.
- [ ] Deploy to Hugging Face Spaces.
