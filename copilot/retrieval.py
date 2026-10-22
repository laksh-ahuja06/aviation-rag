"""Hybrid retrieval: BM25 + dense embeddings, fused with Reciprocal Rank Fusion,
then re-ranked by a cross-encoder whose score drives the evidence gate."""
from dataclasses import dataclass, field

import chromadb
import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from copilot import config
from copilot.normalize import SpellFixer, bm25_tokens, expand, normalize, set_spellfixer


def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K) -> dict[str, float]:
    """score(doc) = sum over rankings of 1 / (k + rank). Rank starts at 1."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class SearchResult:
    query: str
    normalized_query: str
    filters: dict
    hits: list[dict] = field(default_factory=list)

    @property
    def strong_hits(self) -> list[dict]:
        return [h for h in self.hits if h["strong"]]

    @property
    def best_score(self) -> float:
        return max((h["score"] for h in self.hits), default=float("-inf"))


class Retriever:
    def __init__(self):
        self.df = pd.read_csv(config.RECORDS_CSV, dtype={"ident": str}).fillna("")
        self.df = self.df.set_index("ident", drop=False)
        self.ids = self.df["ident"].tolist()

        # Same spell-fixer as ingestion, so queries are normalised identically.
        corpus = pd.concat([self.df["problem"].map(expand), self.df["action"].map(expand)])
        set_spellfixer(SpellFixer(corpus))

        self.bm25 = BM25Okapi([bm25_tokens(t) for t in self.df["problem_norm"]])
        self.embedder = SentenceTransformer(config.EMBED_MODEL)
        self.reranker = CrossEncoder(config.RERANK_MODEL)
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        self.col = client.get_collection(config.CHROMA_COLLECTION)

    # --- individual retrievers ------------------------------------------------

    def _mask(self, component: str | None, fault_type: str | None) -> np.ndarray:
        m = np.ones(len(self.df), dtype=bool)
        if component:
            m &= (self.df["component"] == component).to_numpy()
        if fault_type:
            m &= (self.df["fault_type"] == fault_type).to_numpy()
        return m

    def bm25_search(self, qnorm: str, n: int, mask: np.ndarray) -> list[str]:
        scores = self.bm25.get_scores(bm25_tokens(qnorm))
        scores = np.where(mask, scores, -np.inf)
        order = np.argsort(-scores)[:n]
        return [self.ids[i] for i in order if scores[i] > 0]

    def dense_search(self, qnorm: str, n: int, component=None, fault_type=None) -> list[str]:
        q = self.embedder.encode(config.EMBED_QUERY_PREFIX + qnorm.lower(), normalize_embeddings=True)
        conds = [{"component": component}] if component else []
        conds += [{"fault_type": fault_type}] if fault_type else []
        where = conds[0] if len(conds) == 1 else ({"$and": conds} if conds else None)
        res = self.col.query(query_embeddings=[q.tolist()], n_results=n, where=where)
        return res["ids"][0]

    # --- public API -------------------------------------------------------------

    def search(self, query: str, k: int = config.DEFAULT_TOP_K, component: str | None = None,
               fault_type: str | None = None, mode: str = "hybrid", rerank: bool = True,
               pool_size: int = config.RERANK_POOL) -> SearchResult:
        qnorm = normalize(query)
        n = max(config.CANDIDATES_PER_RETRIEVER, pool_size)
        mask = self._mask(component, fault_type)
        rankings = []
        if mode in ("hybrid", "bm25"):
            rankings.append(self.bm25_search(qnorm, n, mask))
        if mode in ("hybrid", "dense") and mask.any():
            rankings.append(self.dense_search(qnorm, min(n, int(mask.sum())), component, fault_type))
        fused = rrf_fuse(rankings)
        pool = sorted(fused, key=fused.get, reverse=True)[: pool_size if rerank else k]

        result = SearchResult(query, qnorm, {"component": component, "fault_type": fault_type})
        if not pool:
            return result
        rows = self.df.loc[pool]
        if rerank:
            # Many records share the same PROBLEM text, so score each unique text once.
            uniq = list(dict.fromkeys(rows["problem_norm"]))
            s = self.reranker.predict([(qnorm.lower(), t.lower()) for t in uniq])
            text_score = dict(zip(uniq, map(float, s)))
            scores = [text_score[t] for t in rows["problem_norm"]]
        else:
            scores = [fused[i] for i in pool]
        for ident, score in sorted(zip(pool, scores), key=lambda x: -x[1])[:k]:
            r = self.df.loc[ident]
            result.hits.append({
                "ident": ident, "problem": r["problem"], "action": r["action"],
                "component": r["component"], "fault_type": r["fault_type"],
                "score": round(score, 3), "rrf": round(fused[ident], 4),
                "strong": (not rerank) or score >= config.EVIDENCE_THRESHOLD,
            })
        return result

    def count_matches(self, query: str, component: str | None = None,
                      fault_type: str | None = None) -> SearchResult:
        """Score EVERY record (within filters) with the cross-encoder, for exact counts.
        Cheap because only ~3,400 problem texts are unique: about 1 s on a laptop CPU."""
        qnorm = normalize(query)
        sub = self.df[self._mask(component, fault_type)]
        result = SearchResult(query, qnorm, {"component": component, "fault_type": fault_type})
        if sub.empty:
            return result
        uniq = sub["problem_norm"].unique().tolist()
        s = self.reranker.predict([(qnorm.lower(), t.lower()) for t in uniq], batch_size=256)
        text_score = dict(zip(uniq, map(float, s)))
        scored = sorted(((text_score[t], i) for i, t in zip(sub["ident"], sub["problem_norm"])), reverse=True)
        for score, ident in scored:
            if score < config.EVIDENCE_THRESHOLD:
                break
            r = self.df.loc[ident]
            result.hits.append({"ident": ident, "problem": r["problem"], "action": r["action"],
                                "component": r["component"], "fault_type": r["fault_type"],
                                "score": round(score, 3), "rrf": 0.0, "strong": True})
        return result

    def rescore(self, query: str, idents: list[str]) -> dict[str, float]:
        """Cross-encoder score of each record against `query` (used by the final evidence gate)."""
        if not idents:
            return {}
        qnorm = normalize(query).lower()
        texts = self.df.loc[idents, "problem_norm"].str.lower().tolist()
        uniq = list(dict.fromkeys(texts))
        s = dict(zip(uniq, map(float, self.reranker.predict([(qnorm, t) for t in uniq]))))
        return {i: s[t] for i, t in zip(idents, texts)}

    def get(self, ident: str) -> dict | None:
        ident = str(ident).strip().lstrip("#")
        if ident not in self.df.index:
            return None
        return self.df.loc[ident].to_dict()
