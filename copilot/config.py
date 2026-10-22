"""Paths, model names and tunable thresholds in one place."""
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Once both models are in the local Hugging Face cache, load them without contacting the Hub.
# On a network with TLS interception (a "certificate is revoked" proxy), every Hub check
# otherwise retries for about a minute before falling back to the cache.
_HF_CACHE = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
if all((_HF_CACHE / f"models--{m.replace('/', '--')}").exists()
       for m in ("BAAI/bge-small-en-v1.5", "cross-encoder/ms-marco-MiniLM-L-6-v2")):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

RAW_CSV = ROOT / "Aircraft_Annotation_DataFile.csv"
DATA_DIR = ROOT / "data"
RECORDS_CSV = DATA_DIR / "records.csv"            # raw + normalised text + labels
ACTION_LABELS_CSV = DATA_DIR / "action_labels.csv"  # offline LLM labels for the leaderboard
# Dense index: one normalised bge vector per record, rows in records.csv order.
# (Exact numpy search replaced Chroma: simpler and exact at this size, see DEV_LOG.md.)
EMBEDDINGS_NPY = ROOT / "index" / "embeddings.npy"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# bge models expect this prefix on queries only, never on documents.
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# Never "mps": PyTorch's Apple-GPU backend crashed under parallel tool calls (DEV_LOG.md).
DEVICE = "cpu"

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
GEMINI_FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite,gemini-flash-lite-latest,gemini-3.1-flash-lite")

# Retrieval
CANDIDATES_PER_RETRIEVER = 50  # BM25 and dense each return this many before fusion
RRF_K = 60
RERANK_POOL = 40               # fused candidates passed to the cross-encoder
DEFAULT_TOP_K = 10

# Evidence gate: cross-encoder logit below which a record is not counted as evidence.
# Calibrated on the 'tune' split of eval/test_questions.json (see DEV_LOG.md).
EVIDENCE_THRESHOLD = float(os.getenv("EVIDENCE_THRESHOLD", "0.0"))
# A record found by an agent-rewritten query must still score at least this against the
# user's original wording. Tune-split unanswerable questions scored <= -7.98.
ORIGINAL_QUESTION_FLOOR = float(os.getenv("ORIGINAL_QUESTION_FLOOR", "-4.0"))

# Deterministic ask-back: if the evidence within AMBIGUITY_BAND of the best score covers 2+
# components, and the best score is at least AMBIGUITY_MIN_SCORE, ask which part is meant.
AMBIGUITY_MIN_SCORE = 5.0
AMBIGUITY_BAND = 1.0

# Agent
MAX_AGENT_STEPS = 6
