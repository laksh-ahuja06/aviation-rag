"""Text normalisation for terse, abbreviation-heavy logbook entries.

The same `normalize()` runs on records and on user queries, so "cylinder",
"CYL" and "cyl" all end up as the same token.
"""
import re
from collections import Counter
from pathlib import Path

from symspellpy import SymSpell, Verbosity

# Multi-character patterns that need regex (slashes, ampersands, '#3').
PATTERNS = [
    (r"\bR\s*&\s*R\b", " REMOVED AND REPLACED "),
    (r"\bA/C\b", " AIRCRAFT "),
    (r"\bR/H\b|\bRH\b|\bRH/", " RIGHT "),
    (r"\bL/H\b|\bLH\b|\bLH/", " LEFT "),
    (r"\bW/O\b", " WITHOUT "),
    (r"\bW/IN\b", " WITHIN "),
    (r"\bW/", " WITH "),
    (r"\bI/B\b", " INBOARD "),
    (r"\bO/B\b", " OUTBOARD "),
    (r"\bC/W\b", " COMPLIED WITH "),
    (r"\bP/N\b", " PART NUMBER "),
    (r"\bS/N\b", " SERIAL NUMBER "),
    (r"\bPUSH\s?RODS?\b", " PUSHROD "),
    (r"\bDIP\s?STICK\b", " DIPSTICK "),
    (r"\bOVER\s?SPEED\b", " OVERSPEED "),
    (r"#\s*(\d)", r" \1 "),
    (r"&", " AND "),
]

# Single-token abbreviations seen in the corpus (see DEV_LOG.md for how this list was built).
ABBREVIATIONS = {
    "CYL": "CYLINDER", "CYLS": "CYLINDERS", "ENG": "ENGINE", "ENGS": "ENGINES",
    "CK": "CHECK", "CKS": "CHECKS", "CKD": "CHECKED", "CK'D": "CHECKED", "CK'ED": "CHECKED", "CKED": "CHECKED",
    "FWD": "FORWARD", "IAW": "IN ACCORDANCE WITH", "ASSY": "ASSEMBLY", "ASSYS": "ASSEMBLIES",
    "LWR": "LOWER", "UPR": "UPPER", "MX": "MAINTENANCE", "CHT": "CYLINDER HEAD TEMPERATURE",
    "EGT": "EXHAUST GAS TEMPERATURE", "INSP": "INSPECTION", "INSPS": "INSPECTIONS",
    "FOD": "FOREIGN OBJECT DEBRIS", "OPS": "OPERATIONS", "EX": "EXHAUST", "EXH": "EXHAUST",
    "CARB": "CARBURETOR", "MAG": "MAGNETO", "MAGS": "MAGNETOS", "COMP": "COMPRESSION",
    "INJ": "INJECTOR", "GOV": "GOVERNOR", "GEN": "GENERATOR", "REF": "REFERENCE",
    "PROP": "PROPELLER", "PROPS": "PROPELLERS", "TEMP": "TEMPERATURE", "TEMPS": "TEMPERATURES",
    "QTS": "QUARTS", "QT": "QUART", "LBS": "POUNDS", "DISCREP": "DISCREPANCY", "REPL": "REPLACE",
    "PWR": "POWER", "PRESS": "PRESSURE", "ALT": "ALTERNATOR", "STBY": "STANDBY", "TACH": "TACHOMETER",
    "ACCY": "ACCESSORY", "HRS": "HOURS", "BTWN": "BETWEEN", "INSTL": "INSTALL", "REQ": "REQUIRED",
    "FDM": "FLIGHT DATA MONITORING", "SRM": "STRUCTURAL REPAIR MANUAL", "SVC": "SERVICE",
    "SERV": "SERVICEABLE", "RPLCD": "REPLACED", "RPLC": "REPLACE",
}

TOKEN_RE = re.compile(r"[A-Z0-9][A-Z0-9'.\-]*")


def expand(text: str) -> str:
    """Upper-case the text and expand abbreviations. Spelling is left alone."""
    t = f" {str(text).upper()} "
    for pat, rep in PATTERNS:
        t = re.sub(pat, rep, t)
    out = []
    for tok in TOKEN_RE.findall(t):
        core = tok.rstrip(".-'")
        out.append(ABBREVIATIONS.get(core, core))
    return " ".join(w for w in out if w)


ENGLISH_WORDS_FILE = Path(__file__).parent / "resources" / "english_words.txt"
SYSTEM_DICT = Path("/usr/share/dict/words")


def _base_forms(w: str) -> list[str]:
    """LUBRICATED -> LUBRICATE, TAPPED -> TAP, GOVERNORS -> GOVERNOR, ..."""
    forms = [w]
    for suf, add in (("IED", "Y"), ("IES", "Y"), ("ED", ""), ("ED", "E"), ("ING", ""), ("ING", "E"),
                     ("ES", ""), ("S", ""), ("ER", ""), ("ERS", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            b = w[: -len(suf)] + add
            forms.append(b)
            if len(b) >= 2 and b[-1] == b[-2]:  # TAPP -> TAP
                forms.append(b[:-1])
    return forms


def build_english_allowlist(corpus_words) -> set[str]:
    """Corpus words that are real English words (or inflections of one).

    Built once from the system word list and saved to resources/english_words.txt.
    The committed file always wins, so every machine applies identical corrections.
    """
    if ENGLISH_WORDS_FILE.exists():
        return set(ENGLISH_WORDS_FILE.read_text().split())
    if SYSTEM_DICT.exists():
        english = {w.strip().upper() for w in SYSTEM_DICT.open()}
        keep = sorted(w for w in set(corpus_words)
                      if w.isalpha() and any(b in english for b in _base_forms(w)))
        ENGLISH_WORDS_FILE.parent.mkdir(exist_ok=True)
        ENGLISH_WORDS_FILE.write_text("\n".join(keep))
        return set(keep)
    return set()


class SpellFixer:
    """Conservative spelling correction using a dictionary built from the corpus.

    Only rare words (seen <= 2 times) that are NOT real English words are
    corrected, and only towards a word at least 10x more frequent. Without the
    English guard it turned CRANK into CRACK and UNSERVICEABLE into SERVICEABLE.
    """

    def __init__(self, texts, protected: set[str] | None = None):
        self.freq = Counter(w for t in texts for w in t.split())
        self.protected = protected if protected is not None else build_english_allowlist(self.freq)
        self.sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        for w, c in self.freq.items():
            if w.isalpha():
                self.sym.create_dictionary_entry(w, c)
        self.corrections: dict[str, str] = {}

    def fix_word(self, w: str, min_len: int = 5) -> str:
        if not w.isalpha() or len(w) < min_len or self.freq.get(w, 0) > 2 or w in self.protected:
            return w
        if w in self.corrections and min_len == 5:
            return self.corrections[w]
        max_d = 2 if len(w) >= 8 else 1
        # ALL, not CLOSEST: CLOSEST stops at the word itself (distance 0).
        cands = [s for s in self.sym.lookup(w, Verbosity.ALL, max_edit_distance=max_d)
                 if s.term != w and s.count >= 10 * max(1, self.freq.get(w, 0))]
        cands.sort(key=lambda s: (s.distance, -s.count))
        best = cands[0].term if cands else w
        if min_len == 5:
            self.corrections[w] = best
        return best

    def fix(self, text: str, min_len: int = 5) -> str:
        return " ".join(self.fix_word(w, min_len) for w in text.split())


_spell: SpellFixer | None = None


def set_spellfixer(sf: SpellFixer) -> None:
    global _spell
    _spell = sf


def normalize(text: str, query: bool = True) -> str:
    """Expand abbreviations, then fix spelling if a corpus spell-fixer is loaded.

    Queries contain everyday words that aren't in the corpus (and so aren't in the
    English allowlist), e.g. BLANK, which was being "fixed" to BLACK. For queries
    we therefore only correct words of 6+ letters (BAFFEL -> BAFFLE still works).
    """
    t = expand(text)
    return _spell.fix(t, min_len=6 if query else 5) if _spell else t


# --- tokenisation for BM25 -------------------------------------------------

STOPWORDS = set("""A AN THE AND OR OF ON IN AT TO FOR FROM BY WITH IS ARE WAS WERE BE BEEN
HAS HAD HAVE IT ITS THIS THAT AS NO NOT WHAT WHATS WHAT'S HOW WHY WHEN WHICH DO DOES DID
USUAL FIX ONE""".split())


def stem(w: str) -> str:
    """Tiny suffix stripper so LEAKING / LEAKS / LEAKED all become LEAK."""
    for suf in ("ING", "ED", "ES", "S"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def bm25_tokens(normalized: str) -> list[str]:
    return [stem(w) for w in normalized.split() if w not in STOPWORDS]
