"""Hybrid retrieval over the tariff prose.

This is the only retrieval in the system, and it is worth being precise about why
it exists, because "we added RAG" is the answer to a question nobody asked.

    The rate table is a join. The tariff rules are prose.
    Only the second one is a retrieval problem.

Rates, weight breaks, class factors and accessorial prices are keyed data and are
looked up in SQL by `pricing.py`. Nothing is embedded, nothing is retrieved, and a
model never sees them. What IS retrieved is the rules text -- when a liftgate
charge is valid, how a reweigh tolerance is computed, what documentation the
carrier owes on a dispute. That text is genuinely unstructured, it is what a human
auditor argues from, and it is what a reviewer needs quoted in front of them
before they press Dispute.

Three stages, each of which earns its place and each of which is measured
separately in `eval/run_eval.py`:

  BM25 (SQLite FTS5, Porter-stemmed)
      Carries most of the recall here, and that is not a surprise: the queries are
      full of exact tokens -- "reweigh", "liftgate", "NMFC", accessorial codes, rule
      numbers like MFS-110. Lexical retrieval is *better* than dense at exact
      tokens, and logistics vocabulary is mostly exact tokens. Anyone who reaches
      for embeddings first in this domain has not looked at the queries.

  Dense (cosine over stored vectors)
      Earns its place on the paraphrased queries -- an ops question like "they
      charged us for a tailgate we never asked for" contains none of the tariff's
      words. Pluggable behind `Embedder`: a Voyage API backend when VOYAGE_API_KEY
      is set, otherwise a deterministic local hashing embedder so the pipeline runs
      offline. The hashing embedder is a BASELINE, not a semantic model -- see its
      docstring. The eval reports both so the difference is visible rather than
      claimed.

  Reciprocal rank fusion, then optional LLM rerank
      RRF because the two scorers are on incomparable scales (BM25 is a negative
      log-odds-ish score, cosine is [-1,1]) and rank fusion needs no calibration
      and no tuned weight. The rerank stage is a cross-encoder in spirit: the model
      sees the query and each candidate together and orders them. It is off by
      default because it costs a call per query and the eval shows what it buys.

Production swap: the same two indexes on Postgres, where it is already the system of record --
`tsvector`/`pg_search` for the lexical half, `pgvector` for the dense half. That is
a build-versus-buy call, and at this corpus size it is not close. A separate vector
database would be a third system to operate for fourteen clause chunks. That
argument gets weaker as the corpus grows, and it is the corpus size -- not a
preference -- that should decide it.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import sqlite3
import struct
from dataclasses import dataclass

import numpy as np

from . import db

DIM = 384
RRF_K = 60


# --------------------------------------------------------------------------
# Embedding backends
# --------------------------------------------------------------------------


class Embedder:
    name = "base"
    dim = DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    """Deterministic local embedder. No model, no network, no download.

    Hashes word unigrams, word bigrams and character 4-grams into a fixed space
    with sublinear term frequency and L2 normalisation -- a signed random-projection
    bag-of-ngrams.

    Be clear about what this is: it captures *lexical and morphological* overlap,
    so it degrades gracefully on typos, inflections and word order, but it has no
    semantic knowledge. "Tailgate" and "liftgate" are close under this embedder
    only because they share characters, not because it knows they are the same
    thing. It exists so the whole pipeline runs and can be evaluated with zero
    setup and zero cost; set VOYAGE_API_KEY for a real semantic backend and the
    eval will show you what the upgrade is worth.

    Claiming this is "semantic search" would be the exact kind of overstatement
    this project is built to argue against.
    """

    name = "hashing-local"

    def _features(self, text: str) -> dict[str, float]:
        t = re.sub(r"[^a-z0-9 ]+", " ", text.lower())
        words = [w for w in t.split() if w]
        feats: dict[str, float] = {}
        for w in words:
            feats[f"w:{w}"] = feats.get(f"w:{w}", 0.0) + 1.0
        for a, b in zip(words, words[1:]):
            feats[f"b:{a}_{b}"] = feats.get(f"b:{a}_{b}", 0.0) + 1.0
        packed = " ".join(words)
        for i in range(len(packed) - 3):
            g = packed[i:i + 4]
            if " " not in g:
                feats[f"c:{g}"] = feats.get(f"c:{g}", 0.0) + 0.5
        return feats

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, tf in self._features(text).items():
                h = hashlib.blake2b(feat.encode(), digest_size=8).digest()
                idx = struct.unpack("<Q", h)[0] % self.dim
                sign = 1.0 if (idx + h[0]) % 2 == 0 else -1.0
                out[row, idx] += sign * (1.0 + math.log(tf))  # sublinear tf
            n = np.linalg.norm(out[row])
            if n > 0:
                out[row] /= n
        return out


class VoyageEmbedder(Embedder):
    """Real semantic embeddings via Voyage. Used when VOYAGE_API_KEY is set.

    Voyage rather than a local sentence-transformer because this machine has no
    torch wheels for its Python version and a 2 GB download is not a dependency a
    reviewer should have to accept to run an eval.
    """

    name = "voyage-3.5-lite"
    dim = 512

    def __init__(self, api_key: str):
        import httpx  # noqa: PLC0415  -- optional dependency, only on this path

        self._key = api_key
        self._http = httpx.Client(timeout=60.0)

    def encode(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for i in range(0, len(texts), 96):
            r = self._http.post(
                "https://api.voyageai.com/v1/embeddings",
                headers={"Authorization": f"Bearer {self._key}"},
                json={"model": "voyage-3.5-lite", "input": texts[i:i + 96],
                      "output_dimension": self.dim},
            )
            r.raise_for_status()
            vecs.extend(d["embedding"] for d in r.json()["data"])
        arr = np.asarray(vecs, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, 1e-9)


_embedder: Embedder | None = None


def embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        key = os.environ.get("VOYAGE_API_KEY")
        if key:
            try:
                _embedder = VoyageEmbedder(key)
            except Exception:
                _embedder = HashingEmbedder()
        else:
            _embedder = HashingEmbedder()
    return _embedder


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def index_clauses(con: sqlite3.Connection) -> int:
    """Embed every clause and store the vectors. FTS5 indexed itself at insert."""
    emb = embedder()
    rows = db.rows(con, "SELECT id, title, text FROM clauses ORDER BY id")
    if not rows:
        return 0
    # Prepend the rule title to the chunk before embedding. The title carries the
    # topic ("Liftgate Service at Delivery") and the body often does not repeat it,
    # so without this a chunk about liftgates may contain the word once.
    texts = [f"{r['title']}. {r['text']}" for r in rows]
    vecs = emb.encode(texts)
    con.execute("DELETE FROM clause_vectors")
    con.executemany(
        "INSERT INTO clause_vectors (id, dim, vec) VALUES (?,?,?)",
        [(r["id"], emb.dim, v.astype(np.float32).tobytes()) for r, v in zip(rows, vecs)],
    )
    con.commit()
    return len(rows)


def _load_vectors(con: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    rows = db.rows(con, "SELECT id, dim, vec FROM clause_vectors ORDER BY id")
    if not rows:
        return [], np.zeros((0, embedder().dim), dtype=np.float32)
    ids = [r["id"] for r in rows]
    mat = np.vstack([np.frombuffer(r["vec"], dtype=np.float32) for r in rows])
    return ids, mat


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


@dataclass
class Hit:
    id: str
    rule_id: str
    title: str
    text: str
    score: float
    source: str


_FTS_SAFE = re.compile(r"[^A-Za-z0-9 ]+")


def _fts_query(q: str) -> str:
    """Turn a natural-language query into a safe FTS5 OR-query.

    FTS5's MATCH grammar treats quotes, hyphens and parentheses as syntax, so a raw
    user string is both a syntax error waiting to happen and an injection surface.
    Tokenising and rejoining with OR is the deterministic option; single-character
    tokens are dropped as noise.
    """
    tokens = [t for t in _FTS_SAFE.sub(" ", q).split() if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in tokens) if tokens else '"tariff"'


def bm25_search(con: sqlite3.Connection, query: str, k: int = 8) -> list[Hit]:
    sql = """
        SELECT c.id, c.rule_id, c.title, c.text, bm25(clauses_fts, 2.0, 1.0) AS score
        FROM clauses_fts JOIN clauses c ON c.id = clauses_fts.id
        WHERE clauses_fts MATCH ? ORDER BY score LIMIT ?
    """
    # bm25() is more negative for better matches; negate so higher is better.
    return [
        Hit(r["id"], r["rule_id"], r["title"], r["text"], -float(r["score"]), "bm25")
        for r in db.rows(con, sql, (_fts_query(query), k))
    ]


def dense_search(con: sqlite3.Connection, query: str, k: int = 8) -> list[Hit]:
    ids, mat = _load_vectors(con)
    if not ids:
        return []
    qv = embedder().encode([query])[0]
    sims = mat @ qv  # both sides are L2-normalised, so this is cosine
    order = np.argsort(-sims)[:k]
    by_id = {r["id"]: r for r in db.rows(con, "SELECT id, rule_id, title, text FROM clauses")}
    out = []
    for i in order:
        r = by_id[ids[i]]
        out.append(Hit(r["id"], r["rule_id"], r["title"], r["text"], float(sims[i]), "dense"))
    return out


def rrf(runs: list[list[Hit]], k: int = 8) -> list[Hit]:
    """Reciprocal rank fusion.

    score(d) = sum over runs of 1 / (RRF_K + rank(d)). No score normalisation and
    no tuned alpha, which is the point -- BM25 and cosine are on scales that cannot
    be compared, and any weighted blend of them is a hyperparameter somebody has to
    own. Rank fusion needs neither.
    """
    scores: dict[str, float] = {}
    best: dict[str, Hit] = {}
    for run in runs:
        for rank, hit in enumerate(run, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (RRF_K + rank)
            if hit.id not in best:
                best[hit.id] = hit
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:k]
    out = []
    for cid, s in ranked:
        h = best[cid]
        out.append(Hit(h.id, h.rule_id, h.title, h.text, s, "hybrid"))
    return out


def search(
    con: sqlite3.Connection,
    query: str,
    k: int = 4,
    strategy: str = "hybrid",
    candidates: int = 10,
) -> list[Hit]:
    """Public entry point. `strategy` is one of bm25 | dense | hybrid | hybrid_rerank."""
    if strategy == "bm25":
        return bm25_search(con, query, k)
    if strategy == "dense":
        return dense_search(con, query, k)

    fused = rrf([bm25_search(con, query, candidates), dense_search(con, query, candidates)],
                k=candidates)
    if strategy == "hybrid_rerank":
        return llm_rerank(query, fused, k)
    return fused[:k]


def llm_rerank(query: str, cands: list[Hit], k: int = 4) -> list[Hit]:
    """Reorder fused candidates with the model, cross-encoder style.

    The model sees the query and every candidate together and returns an ordering,
    which is strictly more information than scoring each candidate alone. Falls
    back to the fused order on any failure -- a rerank stage that can take down
    retrieval is worse than no rerank stage.
    """
    from pydantic import BaseModel, Field

    from . import llm

    if not cands or not llm.available():
        return cands[:k]

    class Ranking(BaseModel):
        ordered_ids: list[str] = Field(description="Candidate ids, most relevant first.")

    listing = "\n\n".join(f"[{h.id}] {h.title}\n{h.text[:600]}" for h in cands)
    try:
        out = llm.structured(
            Ranking,
            system=(
                "You rank tariff clauses by how directly they govern a specific freight "
                "billing question. A clause that states the rule being applied outranks one "
                "that merely mentions the same words. Return every candidate id exactly once."
            ),
            content=f"QUESTION: {query}\n\nCANDIDATES:\n{listing}",
            stage="rerank",
            max_tokens=1000,
            effort="low",
        )
    except Exception:
        return cands[:k]

    by_id = {h.id: h for h in cands}
    ordered = [by_id[i] for i in out.ordered_ids if i in by_id]
    for h in cands:  # anything the model dropped keeps its fused position
        if h not in ordered:
            ordered.append(h)
    return ordered[:k]
