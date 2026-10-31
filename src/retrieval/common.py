# src/retrieval/common.py
"""
Shared pieces for step 7 (retrieval experiments): paths, loaders, BM25,
OpenAI embeddings with a disk cache, token estimates, metrics, bootstrap.

Every system writes one results file per (system, tree) into
    data/processed/subset/eval/results/<system>__<tree>__<split>.json
with the same per-question record shape, so evaluate.py can compare them:

    {
      "qid": "dev-003", "question": ..., "gold": [doc ids], "n_gold": 2,
      "opened": [doc ids in the order they were opened/ranked],
      "slots": [{"type": "doc"|"summary", "id": ...}, ...]   # optional (RAPTOR collapsed)
      "summaries_read": int, "doc_rows_read": int,
      "llm_calls": int, "input_tokens": int, "output_tokens": int,
      "summary_tokens_read": int,                            # tokens of summaries shown/scored
      "trace": [...]                                         # optional
    }
"""
import hashlib
import json
import math
import os
import random
import re
import threading
from collections import Counter, defaultdict
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data/processed/subset"
GRAPH_DIR = DATA_DIR / "graph"
TREE_DIR = GRAPH_DIR / "trees"
SUMMARY_DIR = TREE_DIR / "summaries"
SHORTCUT_DIR = TREE_DIR / "shortcuts"
EVAL_DIR = DATA_DIR / "eval"
RESULTS_DIR = EVAL_DIR / "results"
CACHE_DIR = EVAL_DIR / "cache"

CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTION_PATHS = {"dev": DATA_DIR / "questions.jsonl",
                  "heldout": DATA_DIR / "questions_heldout.jsonl"}

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
MAX_EMBED_CHARS = 20000          # ~5k tokens; text-embedding-3-small limit is 8191 tokens
EMBED_BATCH = 64

TREES = ("leiden", "tfidf-kmeans")
SHORTCUT_POLICIES = ("typed", "random_typed", "strongest", "random")

_LOCK = threading.Lock()


# ------------------------------------------------------------
# io
# ------------------------------------------------------------

def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------
# data loaders
# ------------------------------------------------------------

def load_corpus():
    docs = read_jsonl(CORPUS_PATH)
    if not docs:
        raise FileNotFoundError(f"Missing corpus: {CORPUS_PATH}")
    return docs


def load_questions(split="dev"):
    path = QUESTION_PATHS[split]
    qs = read_jsonl(path)
    if not qs:
        raise FileNotFoundError(f"Missing questions: {path}")
    out = []
    for i, q in enumerate(qs):
        out.append({"qid": f"{split}-{i:03d}", "question": q["question"],
                    "gold": list(dict.fromkeys(q["article_ids"])), "n_gold": len(set(q["article_ids"]))})
    return out


def load_tree(method):
    path = TREE_DIR / f"{method}_tree.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing tree: {path} (run build_tree_v2.py --method {method})")
    tree = json.load(path.open(encoding="utf-8"))
    tree["by_id"] = {n["id"]: n for n in tree["nodes"]}
    return tree


def load_summaries(method, require_complete=True):
    path = SUMMARY_DIR / f"{method}_summaries.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing summaries: {path} (run summarize_tree_v2.py --tree {method})")
    data = json.load(path.open(encoding="utf-8"))
    tree = load_tree(method)
    missing = [n for n in tree["by_id"] if n not in data["summaries"]]
    if missing and require_complete:
        raise RuntimeError(f"{len(missing)} tree nodes have no summary yet; finish summarize_tree_v2.py --tree {method}")
    data["tree"] = tree
    return data


def load_shortcuts(method, policy):
    path = SHORTCUT_DIR / f"{method}_shortcuts_{policy}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing shortcuts: {path} (run select_shortcuts.py --tree {method})")
    return json.load(path.open(encoding="utf-8"))["shortcuts"]


def node_text(summary):
    """Routing text of a node: what the navigator reads, what the embedding/BM25 routers score."""
    parts = [summary["title"], summary["summary"],
             " ".join(summary.get("question_types", [])), ", ".join(summary.get("key_terms", []))]
    return "\n".join(p for p in parts if p)


def doc_text(doc):
    return doc["contents"] if doc["contents"].startswith(doc["title"]) else doc["title"] + "\n" + doc["contents"]


# ------------------------------------------------------------
# tokens
# ------------------------------------------------------------

try:  # exact counts if tiktoken is installed, otherwise the usual 4-chars heuristic
    import tiktoken
    _ENC = tiktoken.get_encoding("o200k_base")

    def num_tokens(text):
        return len(_ENC.encode(text))
    TOKEN_COUNTER = "tiktoken/o200k_base"
except Exception:  # pragma: no cover
    def num_tokens(text):
        return max(1, len(text) // 4)
    TOKEN_COUNTER = "chars/4"


# ------------------------------------------------------------
# BM25 (same tokenizer and parameters as the subset gate: k1=1.5, b=0.75)
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    def __init__(self, ids, texts, k1=1.5, b=0.75):
        self.ids = list(ids)
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(t)) for t in texts]
        self.dl = [sum(tf.values()) for tf in self.tfs]
        self.n = len(self.ids)
        self.avgdl = (sum(self.dl) / self.n) if self.n else 1.0
        self.df = Counter()
        for tf in self.tfs:
            self.df.update(tf.keys())
        self.idf = {w: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for w, d in self.df.items()}
        self.index = {i: k for k, i in enumerate(self.ids)}

    def score(self, query, idx):
        tf, dl = self.tfs[idx], self.dl[idx]
        s = 0.0
        for w in tokenize(query):
            if w in tf:
                s += self.idf[w] * tf[w] * (self.k1 + 1) / (tf[w] + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def rank(self, query, candidates=None):
        idxs = range(self.n) if candidates is None else [self.index[c] for c in candidates]
        scored = [(self.score(query, i), self.ids[i]) for i in idxs]
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored


# ------------------------------------------------------------
# embeddings (OpenAI) with a content-addressed disk cache
# ------------------------------------------------------------

class Embedder:
    """embed(texts) -> list of vectors; identical text is embedded once (cache keyed by model+sha256)."""

    def __init__(self, model=EMBED_MODEL, cache_path=None, client=None):
        self.model = model
        self.cache_path = cache_path or (CACHE_DIR / f"embeddings_{model}.jsonl")
        self.cache = {}
        for r in read_jsonl(self.cache_path):
            if r.get("model") == model:
                self.cache[r["key"]] = r["vector"]
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("Missing OPENAI_API_KEY in .env (needed to embed texts not yet cached)")
            self._client = OpenAI(timeout=120.0, max_retries=3)
        return self._client

    def key(self, text):
        return sha256_text(self.model + "\x00" + text)

    def embed(self, texts, label=""):
        texts = [t[:MAX_EMBED_CHARS] for t in texts]
        keys = [self.key(t) for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache]
        if todo:
            print(f"  embedding {len(todo)} new texts with {self.model} {label}".rstrip())
        for start in range(0, len(todo), EMBED_BATCH):
            batch = todo[start:start + EMBED_BATCH]
            response = self.client.embeddings.create(model=self.model, input=[t for _, t in batch])
            for (k, _), item in zip(batch, response.data):
                self.cache[k] = item.embedding
                append_jsonl(self.cache_path, {"model": self.model, "key": k, "vector": item.embedding})
        return [self.cache[k] for k in keys]


def normalize_rows(matrix):
    import numpy as np
    m = np.asarray(matrix, dtype="float32")
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return m / norms


# ------------------------------------------------------------
# metrics
# ------------------------------------------------------------

def opened_at(record, budget):
    """Documents credited within a budget of `budget` reads/slots."""
    if record.get("slots"):
        return [s["id"] for s in record["slots"][:budget] if s["type"] == "doc"]
    return record["opened"][:budget]


def question_metrics(record, budget):
    gold = set(record["gold"])
    got = opened_at(record, budget)
    got_set = set(got)
    hit = gold & got_set
    if record.get("slots"):   # rank = retrieval slot, so a summary ahead of the gold article counts
        first = next((i + 1 for i, s in enumerate(record["slots"][:budget]) if s["type"] == "doc" and s["id"] in gold), None)
    else:
        first = next((i + 1 for i, d in enumerate(got) if d in gold), None)
    return {
        "all_gold": float(gold <= got_set),
        "recall": len(hit) / len(gold),
        "mrr": (1.0 / first) if first else 0.0,
        "n_opened": len(got),
    }


def paired_bootstrap(a, b, n_boot=2000, seed=42):
    """Mean difference a-b with a 95% percentile CI and a two-sided bootstrap p-value."""
    assert len(a) == len(b) and len(a) > 0
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    observed = sum(diffs) / n
    samples = []
    for _ in range(n_boot):
        s = sum(diffs[rng.randrange(n)] for _ in range(n)) / n
        samples.append(s)
    samples.sort()
    lo, hi = samples[int(0.025 * n_boot)], samples[min(n_boot - 1, int(0.975 * n_boot))]
    # p-value: shift the bootstrap distribution to mean 0 and count |s| >= |observed|
    centered = [s - observed for s in samples]
    p = sum(abs(s) >= abs(observed) for s in centered) / n_boot
    return {"diff": observed, "ci_low": lo, "ci_high": hi, "p": p}


# ------------------------------------------------------------
# results files
# ------------------------------------------------------------

def results_path(system, tree, split):
    return RESULTS_DIR / f"{system}__{tree}__{split}.json"


def save_results(system, tree, split, config, records):
    path = results_path(system, tree, split)
    write_json(path, {"system": system, "tree": tree, "split": split, "config": config,
                      "n_questions": len(records), "per_question": records})
    return path


def quick_report(system, records, budgets=(5, 10)):
    n = len(records)
    parts = [f"{system:>26s} n={n}"]
    for B in budgets:
        m = [question_metrics(r, B) for r in records]
        parts.append(f"all-gold@{B} {sum(x['all_gold'] for x in m)/n:.3f} R@{B} {sum(x['recall'] for x in m)/n:.3f}")
    m = [question_metrics(r, max(budgets)) for r in records]
    parts.append(f"MRR {sum(x['mrr'] for x in m)/n:.3f}")
    reads = sum(r.get("summaries_read", 0) for r in records) / n
    calls = sum(r.get("llm_calls", 0) for r in records) / n
    toks = sum(r.get("input_tokens", 0) + r.get("output_tokens", 0) for r in records) / n
    parts.append(f"| summaries/q {reads:.1f} calls/q {calls:.1f} tokens/q {toks:.0f}")
    return " ".join(parts)
