# Development Log

Running record of key decisions, AI assistance, experiments and rejected approaches, as encouraged by the NexAI Hack 2026 instructions.

---

## 21 Oct 2026, 9:00 PM: Kickoff and setup

**9:00 PM:** Looked at all the problem statements, studied the tech stacks, and decided which packages/SDKs best suit the idea.

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

## 22 Oct 2026, 6:00 AM: First build (version 1.0)

**6:00 AM:** Started implementing the first version (version 1.0) of aviation-rag. Committed as "Version 1.0" at 7:04 AM.

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

---

## 22 Oct 2026, 8:30 AM: Startup crash and dropping Chroma

### The bug
- Running `streamlit run app.py` crashed the whole process during model loading with a native error: `malloc: *** error for object ...: pointer being freed was not allocated`, followed by `zsh: abort`. This is memory corruption inside a C/C++/Rust extension, not a Python exception.
- The crash came right after both models had loaded (199/199, then 105/105 weights). In `Retriever.__init__` the next step was `chromadb.PersistentClient(...)`, which runs Chroma's native Rust engine. The likely cause is that engine sharing a process with PyTorch on the very new Python 3.14.

### Investigation
- I couldn't reproduce it in 13 attempts: 5 plain-script loads, 6 loads in a worker thread (how Streamlit runs scripts), and 2 real Streamlit starts, including two browser sessions at once. So it's intermittent, which is the worst kind of bug to discover during a live demo.

### Decision: drop Chroma and use numpy exact search
- At 6,169 vectors × 384 dimensions, the index is a 9.5 MB matrix. One matrix product does an **exact** cosine search in under a millisecond, while Chroma's HNSW search is approximate. Component/fault-type filters become a boolean mask, the same one BM25 already used.
- Removing the native library removes the crash candidate and simplifies deployment.
- Rejected alternatives:
  - Keeping Chroma and hoping: the crash is unexplained and intermittent.
  - Pinning an older Python: there's no time to rebuild the environment mid-hackathon.
  - FAISS: another native library, and overkill at this size.
- Verified after the change: retrieval hit@5 is unchanged at **13/15 overall**; all unanswerable questions remain below the threshold; and the app passed a two-session Streamlit start plus a full agent answer.

### Correction: Chroma was NOT the cause
- The app crashed again without Chroma (`zsh: segmentation fault`), so the diagnosis above was wrong. It was based on timing ("the next step after loading"), not evidence.
- **The evidence came from macOS crash reports** (`~/Library/Logs/DiagnosticReports/Python-*.ips`, one for each crash, at 07:03 and 08:59). Both faulting threads were named `ThreadPoolExecutor-9_1` and crashed inside **PyTorch's MPS (Apple GPU) backend**: `Tensor.to(device)` → `mps_copy_` → `MetalShaderLibrary::exec_unary_kernel` → a hash-table insert. One run aborted with "pointer being freed was not allocated", the other segfaulted.
- **Root cause:**
  1. `sentence-transformers` puts models on `mps` by default on Apple Silicon.
  2. LangGraph's `ToolNode` runs parallel tool calls in a thread pool.
  3. Two threads using the MPS backend at once race on PyTorch's shared Metal kernel cache.
  4. It only happens when the agent makes parallel tool calls at the right moment, which is why it was intermittent.
- **Proof:** a stress script (8 threads, models on MPS, no lock) segfaulted **3/3 times** (exit 139). After the fix, the same stress test through the real `Retriever` passed **3/3 times**, 240 searches each.
- **Fix** (`copilot/retrieval.py`, `config.DEVICE`):
  - Models run on **CPU**, which is fast enough for these small models and matches the CPU-only Hugging Face Spaces deployment.
  - A `threading.Lock` serialises every embedder and cross-encoder call, so parallel tool calls can't run the models at the same time.
  - Embeddings were rebuilt on CPU. hit@5 is unchanged at 13/15, and the unanswerable scores are identical.
- **Chroma stays out anyway:** numpy exact search is simpler and exact, but Chroma was innocent. Lesson: read the crash report before changing code.

---

## 22 Oct 2026, 9:30 AM: Ask back on ambiguous questions

### The trigger
- A user test of "plugs are worn" produced a summary and one cause bullet, then an **empty "Actions that fixed it:" heading**.
- Cause: "plugs" matched two different parts. Record 105445 is "ALL SPARK PLUGS WORN" (ignition); records 101073, 101156, 100331 and others are "SPARK PLUG BAFFLE HOLE PLUGS ARE WORN" (cooling-baffle hardware).
- The agent merged both kinds into its action bullets, and the citation verifier (which checks "same fault as the question") removed them. The trust layer worked, but the answer looked broken.

### Design
- **`ask_clarification(question, options)` tool:** the graph stops after this tool runs, and the UI shows the question with one button per option. Clicking a button re-asks "<question> (I mean: <option>)" with ask-back **disabled**, so it can never loop.
- **Deterministic ambiguity hint:** `search_records` now lists which parts the STRONG records acted on, using the offline leaderboard labels (e.g. "baffle plug x4, spark plug x1"). The agent decides whether to ask, but the signal comes from data rather than LLM intuition, which matters for the planned local-model branch, where a weaker model needs a clearer signal.
- Prompt rule: ask only when the words fit different parts or faults, not when the records agree or the extra parts are sub-parts of the same fix.
- `eval/run_agent_eval.py` reports ask-backs separately, so they're counted neither as answers nor as refusals.

### A bug the test exposed: wrong component labels
- After clarifying "Baffle plugs", the agent filtered by `component='baffle / cooling'` and then refused. **All 214 records** mentioning a baffle *and* a plug were labelled **ignition**, because the rule for "PLUG" ran before the rule for "BAFFLE". The filter therefore hid exactly the right records.
- **Fix:** a new rule, checked before ignition: a problem that mentions both a baffle and a plug, or a "button plug", is labelled baffle/cooling. I checked by hand that the 214 records are baffle access/hole/snap/button plugs, caps and covers, never spark plugs themselves. 215 records changed label; ignition went from 281 to 66.
- **Safety net:** the prompt now says filters come from automatic labels and can be wrong, so the agent must search again without filters before concluding anything.
- Retrieval hit@5 is unchanged at 13/15.

### Results (live Gemini, Flash-Lite via the fallback chain)
| Question | Behaviour |
|---|---|
| "plugs are worn" | Asks "Which plugs are worn?" with [Baffle plugs] [Spark plugs] (2 LLM calls, 3.8s) |
| "... (I mean: Spark plugs)" | Answers from rec 105445; 0 lines removed by the verifier |
| "... (I mean: Baffle plugs)" | Before the label fix: wrongly refused. After: "Replaced: baffle plug, 10 cases", all citations verified |
| "#3 rocker cover leaking" | Answers directly with no unnecessary ask-back |

### 10:00 AM: Version 2.0 started: local model instead of an API key
- Started version 2.0 on a separate branch. It uses a local model instead of the Gemini API.
- Why:
  - The Gemini free tier allows only 20 requests per model per day (see "Gemini free-tier quota" above).
  - The venue network intercepts HTTPS, which can block the API during the demo.
  - A local model removes both risks and needs no API key.
- Plan: the LLM is created only in `copilot/llm.py`, so that should be the main file to change. The deterministic parts work with any model: retrieval, evidence gate, ambiguity check, the verifier's ID check, the leaderboard and search-only mode. The main risk is tool-calling reliability with a smaller model.

### 10:15 AM: Follow-up: making ask-back deterministic
- In a user test, the same "plugs are worn" prompt **didn't** ask back. Flash-Lite answered directly, mixed spark plugs with baffle plugs, and the verifier removed the cause bullet, leaving an **empty "Likely causes:" heading**. Whether the LLM calls `ask_clarification` varies from run to run, so the trust behaviour can't depend on it alone.
- **Deterministic rule** (`Copilot._ambiguity`): after the evidence gate, take the gated records within 1.0 of the best score. If they span 2+ components (ignoring "other") and the best score is ≥ 5.0, ask back. Each option names its part and component, e.g. "Baffle plug (baffle / cooling)" vs "Spark plug (ignition)". The first version showed "Spark plug hole plug" vs "Spark plug", which was confusing.
- **Tuning, offline with no LLM:** over the 15 test questions the rule fires on **none**. A10 ("exhaust pipe cracked") would have fired at band-only, but its best score is 3.34, below the 5.0 minimum. It does fire on genuinely vague inputs: "seal leaking" (pushrod vs baffle) and "screw missing" (baffle vs rocker cover). On the user's actual evidence (spark plug 8.0, baffle plugs 7.9) it fires.
- **Empty headings:** after verification, a heading line ending in ":" with no bullet under it is dropped.
- The LLM tool stays as well: the model can still ask back for reasons the rule can't see.

### 10:15 AM: Network: TLS interception and offline model loading
- On the event Wi-Fi, Hugging Face Hub requests failed with "Trustwave Organization Validation CA, Level 2 certificate is revoked", a TLS-inspection proxy. The models still loaded from the local cache, but only after about a minute of retries at each start. Gemini calls failed too, so the app correctly dropped to search-only mode.
- **Fix:** `config.py` sets `HF_HUB_OFFLINE=1` when both models are already in the local cache, so startup no longer contacts the Hub. A fresh machine (or Hugging Face Spaces) still downloads them on first run.
- **Demo risk:** if the venue network blocks or intercepts `generativelanguage.googleapis.com`, Gemini won't work. Search-only mode still answers. Have a phone hotspot ready as a backup.

### Housekeeping
- `.env.example` had been deleted locally, and the committed copy was missing its `GOOGLE_API_KEY` / `GEMINI_MODEL` lines. I restored it with placeholders (the rules require it). I scanned every commit of it: no real API key has ever been committed, and `.env` was never tracked.

---

## 22 Oct 2026, 10:40 AM: Learning the architecture and codebase
- Went through the architecture and the codebase so every part can be explained to the judges, as the rules require for AI-generated code.
- Created `docs/architecture.svg`, an updated diagram of the current pipeline. It replaces the original sketch and the outdated Mermaid diagram in the README. Changes include numpy search instead of Chroma, the 4 agent tools, the offline labels, search-only mode and the ambiguity check.
- Mapped each box in the diagram to the file and function that implements it.
- Reviewed the generated data files:
  - `data/records.csv`: the original 3 columns plus `problem_norm`, `action_norm`, `component`, `fault_type` and `action_type_rule`.
  - `data/action_labels.csv`: Gemini's `action_type` + `part` for each unique problem/action pair.
- Confirmed that `action_labels.csv` is derived only from the dataset: Gemini labelled the existing ACTION text and added no outside information. The evaluation doesn't use it, since test relevance is defined on PROBLEM text written before the labels existed.
- Open question: the Gemini labels haven't been checked for accuracy. Plan to hand-check a random sample of 50 and report the result.

---

## 22 Oct 2026, 12:30 PM: Deployment (live on Streamlit Community Cloud)

**Live app:** https://maintainance-copilot.streamlit.app/

### First plan: Hugging Face Spaces with Docker (rejected)
- Wrote a `Dockerfile`: Python 3.14 (the same as local) and CPU-only PyTorch (`torch==2.14.1` from the PyTorch CPU index instead of the multi-GB CUDA build). It builds the index and downloads the models at build time.
- Checked that all 101 packages resolve for Linux x86_64 on Python 3.14 (`pip install --dry-run --platform ...`), including `torch 2.14.1+cpu`.
- **Rejected:** since July 2026 the Hugging Face docs say *"Gradio and Docker Spaces run on compute and require a paid plan to create"* (PRO, $9/month, for personal accounts). Free accounts get only static Spaces, or Gradio on ZeroGPU, and Streamlit is no longer an SDK option.
- The `Dockerfile` and `.dockerignore` are kept for any Docker host, or for a paid Space.

### Chosen: Streamlit Community Cloud (free)
- It hosts Streamlit apps directly from the public GitHub repo, using `app.py` with Python 3.14 selected in Advanced settings.
- Measured the app's peak memory locally at **about 610 MB** (both models loaded, searches and exact counts run), within the platform's limit of up to 2.7 GB.
- The embedding index isn't committed; the app builds it on first start, and the models download from Hugging Face.
- **API key:** stored in the app's **Secrets** setting, never in the repo. `app.py` copies Streamlit secrets into environment variables, so `copilot/config.py` reads the key the same way it reads `.env` locally. I checked that `.env` has never been committed and isn't on GitHub.

### Bug found on first deploy: crash when the key is missing
- The first deploy showed a red `pydantic ValidationError` on load: *"API key required for Gemini Developer API."* The secret wasn't set yet, so `ChatGoogleGenerativeAI(google_api_key=None)` failed inside `Copilot.__init__` and took the whole app down. I reproduced it locally with the key unset.
- This contradicted the README's claim that the app runs search-only without a key. **Fixes:**
  - `Copilot.__init__` catches the setup error. Every question then goes to **search-only mode** (search, evidence gate, leaderboard), and the sidebar shows a 🔑 "No Gemini API key found" warning.
  - The secrets reader also accepts keys nested under a `[section]` and the name `GEMINI_API_KEY`, and trims whitespace.
  - `.streamlit/secrets.toml` is git-ignored, so a local secrets file can never be pushed to the public repo.
- Tested in a clean copy with no key: the app starts, answers from records, and still refuses "GPS display is blank". Locally with `.env`, the key loads and all 4 Gemini models set up as before.
- After the key was added to the Secrets setting, the deployed app answered with the full agent.

### Next steps
- [ ] Commit at least hourly.
- [ ] Hand-check 50 random action labels and report the accuracy in the README.
- [ ] Re-run `python -m eval.run_agent_eval` after the quota resets, and report the results in the README.
- [ ] Local-model branch: keep the deterministic parts hint and the search-only fallback; tool-calling reliability is the main risk.
- [ ] Demo video (≤ 5 min), pitch deck (PowerPoint), LinkedIn + X posts, and team details in the README.
- [x] `pip install -r requirements.txt` in the venv.
- [x] Write test questions before tuning.
- [x] Build ingestion, normalisation, hybrid index, agent, trust layer, leaderboard and UI.
- [x] Deploy: live at https://maintainance-copilot.streamlit.app/ (Streamlit Community Cloud).
