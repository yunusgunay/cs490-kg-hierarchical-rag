# src/kg/build_tree_v2.py
"""
Recursive partitioning of the document graph into a tree (step 5).

Differences from recursive_partition.py (v1)
--------------------------------------------
* Nodes are documents (doc_graph.json), not passages.
* No bin-packing. Every internal node is a real partition: Leiden with
  the resolution chosen per node (bisection) so that each node has
  MIN_CHILDREN..MAX_CHILDREN children; communities smaller than
  MIN_LEAF are merged into the sibling they are most connected to.
  v1 had 57 of 62 internal nodes produced by size-balanced packing;
  here the "semantic internal node share" is 100% by construction.
* Cut edges are recorded per internal node (which child pair, weight,
  shared entities, bridge predicates): they are the shortcut
  candidates for step 6.
* --method tfidf-kmeans builds a tree of the same shape with recursive
  k-means on TF-IDF vectors (no KG): the partitioning ablation the
  interim report needs.

Outputs (data/processed/subset/graph/trees/)
  <method>_tree.json, <method>_tree_stats.json
"""
import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import igraph as ig
import leidenalg
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/subset"
GRAPH_DIR = DATA_DIR / "graph"
TREE_DIR = GRAPH_DIR / "trees"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
GRAPH_PATH = GRAPH_DIR / "doc_graph.json"

SCHEMA_VERSION = "doc-tree-v2"

MAX_LEAF = 8          # documents per leaf (upper bound)
MIN_LEAF = 3          # communities smaller than this are merged into a sibling
MIN_CHILDREN = 3
MAX_CHILDREN = 8
MAX_DEPTH = 6
SEED = 42


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def node_id(depth, doc_ids):
    h = hashlib.sha256(json.dumps(sorted(doc_ids)).encode()).hexdigest()[:16]
    return f"t{depth}_{h}"


# ------------------------------------------------------------
# Splitters: each returns a list of groups (lists of doc ids)
# ------------------------------------------------------------

class LeidenSplitter:
    def __init__(self, graph_json):
        self.w = defaultdict(float)
        self.meta = {}
        for e in graph_json["edges"]:
            a, b = sorted((e["source"], e["target"]))
            self.w[(a, b)] = e["weight"]
            self.meta[(a, b)] = e
        self.adj = defaultdict(dict)
        for (a, b), w in self.w.items():
            self.adj[a][b] = w
            self.adj[b][a] = w
        self.name = "leiden"

    def _partition(self, docs, resolution, seed):
        index = {d: i for i, d in enumerate(docs)}
        edges, weights = [], []
        for a in docs:
            for b, w in self.adj[a].items():
                if b in index and index[a] < index[b]:
                    edges.append((index[a], index[b]))
                    weights.append(w)
        g = ig.Graph(n=len(docs), edges=edges, directed=False)
        g.es["weight"] = weights
        part = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition, weights="weight",
            resolution_parameter=resolution, n_iterations=-1, seed=seed)
        return [[docs[i] for i in community] for community in part]

    def split(self, docs, seed):
        """Bisection on resolution to land in [MIN_CHILDREN, MAX_CHILDREN]."""
        docs = sorted(docs)
        lo, hi = 0.01, 64.0
        best = None
        for _ in range(24):
            res = math.sqrt(lo * hi)
            groups = self._merge_small(docs, self._partition(docs, res, seed))
            k = len(groups)
            if MIN_CHILDREN <= k <= MAX_CHILDREN:
                return groups, res
            if best is None or abs(k - (MIN_CHILDREN + MAX_CHILDREN) / 2) < abs(len(best[0]) - (MIN_CHILDREN + MAX_CHILDREN) / 2):
                best = (groups, res)
            if k < MIN_CHILDREN:
                lo = res
            else:
                hi = res
        return best

    def _merge_small(self, docs, groups):
        """Merge communities below MIN_LEAF into the sibling they are most connected to."""
        groups = [sorted(g) for g in groups]
        changed = True
        while changed and len(groups) > 1:
            changed = False
            groups.sort(key=len)
            small = groups[0]
            if len(small) >= MIN_LEAF:
                break
            best_j, best_w = None, -1.0
            for j in range(1, len(groups)):
                w = sum(self.adj[a].get(b, 0.0) for a in small for b in groups[j])
                if w > best_w:
                    best_j, best_w = j, w
            if best_w <= 0.0:          # no connection at all: smallest sibling
                best_j = 1
            groups[best_j] = sorted(groups[best_j] + small)
            groups.pop(0)
            changed = True
        return sorted(groups, key=lambda g: (-len(g), g[0]))

    def cut_summary(self, groups):
        """Edges crossing child groups, strongest first (shortcut candidates)."""
        member = {d: i for i, g in enumerate(groups) for d in g}
        cuts = []
        internal = cut = 0.0
        for (a, b), w in self.w.items():
            if a in member and b in member:
                if member[a] == member[b]:
                    internal += w
                else:
                    cut += w
                    e = self.meta[(a, b)]
                    cuts.append({
                        "doc_a": a, "doc_b": b, "child_a": member[a], "child_b": member[b],
                        "weight": round(w, 3),
                        "shared_entities": [s[0] for s in e.get("shared_entities", [])[:3]],
                        "bridge_predicates": sorted({x["predicate"] for x in e.get("bridges", [])}),
                    })
        cuts.sort(key=lambda c: -c["weight"])
        return {"cut_weight_fraction": round(cut / (cut + internal), 4) if cut + internal else 0.0,
                "cut_edges": len(cuts), "top_cut_edges": cuts[:30]}


class TfidfKMeansSplitter:
    """Recursive k-means on TF-IDF vectors: the no-KG ablation."""

    def __init__(self, docs):
        tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
        toks = {d["id"]: tok(d["title"] + " " + d["contents"][:4000]) for d in docs}
        df = Counter()
        for t in toks.values():
            df.update(set(t))
        n = len(docs)
        vocab = {w: i for i, w in enumerate(sorted(w for w, c in df.items() if 1 < c < 0.5 * n))}
        self.vec = {}
        for i, t in toks.items():
            v = np.zeros(len(vocab), dtype=np.float32)
            for w, c in Counter(t).items():
                if w in vocab:
                    v[vocab[w]] = (1 + math.log(c)) * math.log(n / df[w])
            norm = np.linalg.norm(v) or 1.0
            self.vec[i] = v / norm
        self.name = "tfidf-kmeans"

    def _kmeans(self, docs, k, seed):
        rng = np.random.default_rng(seed)
        X = np.stack([self.vec[d] for d in docs])
        centers = X[rng.choice(len(docs), size=k, replace=False)].copy()
        labels = np.zeros(len(docs), dtype=int)
        for _ in range(50):
            sims = X @ centers.T
            new = sims.argmax(axis=1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for c in range(k):
                members = X[labels == c]
                if len(members):
                    m = members.mean(axis=0)
                    centers[c] = m / (np.linalg.norm(m) or 1.0)
        groups = defaultdict(list)
        for d, l in zip(docs, labels):
            groups[int(l)].append(d)
        return [sorted(g) for g in groups.values()]

    def split(self, docs, seed):
        docs = sorted(docs)
        k = max(MIN_CHILDREN, min(MAX_CHILDREN, math.ceil(len(docs) / MAX_LEAF)))
        k = min(k, len(docs) // MIN_LEAF) if len(docs) >= 2 * MIN_LEAF else 2
        groups = self._kmeans(docs, max(2, k), seed)
        # merge tiny clusters into the nearest centroid cluster
        groups.sort(key=len)
        while len(groups) > 1 and len(groups[0]) < MIN_LEAF:
            small = groups.pop(0)
            cs = [np.mean([self.vec[d] for d in g], axis=0) for g in groups]
            sv = np.mean([self.vec[d] for d in small], axis=0)
            j = int(np.argmax([float(sv @ c) for c in cs]))
            groups[j] = sorted(groups[j] + small)
            groups.sort(key=len)
        return sorted(groups, key=lambda g: (-len(g), g[0])), None

    def cut_summary(self, groups):
        return {"cut_weight_fraction": None, "cut_edges": None, "top_cut_edges": []}


# ------------------------------------------------------------
# Recursive builder
# ------------------------------------------------------------

def build_tree(splitter, all_docs):
    nodes = {}
    doc_to_leaf = {}

    def recurse(docs, parent, depth):
        docs = sorted(docs)
        nid = node_id(depth, docs)
        node = {"id": nid, "parent_id": parent, "depth": depth, "size": len(docs),
                "doc_ids": docs, "children": [], "is_leaf": False,
                "split_method": None, "resolution": None, "split_statistics": None}
        nodes[nid] = node
        if len(docs) <= MAX_LEAF or depth >= MAX_DEPTH:
            node["is_leaf"] = True
            node["stop_reason"] = "leaf_size" if len(docs) <= MAX_LEAF else "max_depth"
            for d in docs:
                doc_to_leaf[d] = nid
            return nid
        groups, res = splitter.split(docs, SEED + depth)
        if len(groups) < 2:
            node["is_leaf"] = True
            node["stop_reason"] = "no_valid_split"
            for d in docs:
                doc_to_leaf[d] = nid
            return nid
        node["split_method"] = splitter.name
        node["resolution"] = round(res, 4) if res else None
        node["split_statistics"] = splitter.cut_summary(groups)
        for g in groups:
            node["children"].append(recurse(g, nid, depth + 1))
        return nid

    root = recurse(all_docs, None, 0)
    return nodes, root, doc_to_leaf


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------

def tree_statistics(nodes, root, doc_to_leaf, by_id, questions, splitter):
    internal = [n for n in nodes.values() if not n["is_leaf"]]
    leaves = [n for n in nodes.values() if n["is_leaf"]]
    product = lambda t: re.sub(r"\s+Request$", "", t.split(":", 1)[0]).strip().casefold() if ":" in t else None
    purity = []
    for leaf in leaves:
        prods = [product(by_id[d]["title"]) for d in leaf["doc_ids"]]
        prods = [p for p in prods if p]
        if prods:
            purity.append(Counter(prods).most_common(1)[0][1] / len(leaf["doc_ids"]))

    # ancestors for LCA depth
    parent = {n["id"]: n["parent_id"] for n in nodes.values()}
    depth = {n["id"]: n["depth"] for n in nodes.values()}

    def ancestors(leaf):
        out = []
        while leaf is not None:
            out.append(leaf)
            leaf = parent[leaf]
        return out

    lca_depths, same_leaf, same_parent, total = [], 0, 0, 0
    for q in questions:
        ids = [i for i in q["article_ids"] if i in doc_to_leaf]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                total += 1
                la, lb = doc_to_leaf[ids[i]], doc_to_leaf[ids[j]]
                if la == lb:
                    same_leaf += 1
                anc_a = ancestors(la)
                anc_b = set(ancestors(lb))
                lca = next(x for x in anc_a if x in anc_b)
                lca_depths.append(depth[lca])
                if parent[la] == parent[lb]:
                    same_parent += 1
    cross_leaf = None
    if isinstance(splitter, LeidenSplitter):
        tot = cut = 0.0
        for (a, b), w in splitter.w.items():
            tot += w
            if doc_to_leaf[a] != doc_to_leaf[b]:
                cut += w
        cross_leaf = round(cut / tot, 4)
    return {
        "method": splitter.name, "documents": len(doc_to_leaf),
        "tree_nodes": len(nodes), "internal_nodes": len(internal), "leaves": len(leaves),
        "max_depth": max(depth.values()),
        "root_children": len(nodes[root]["children"]),
        "branching": dict(Counter(len(n["children"]) for n in internal)),
        "leaf_sizes": dict(sorted(Counter(n["size"] for n in leaves).items())),
        "mean_leaf_size": round(sum(n["size"] for n in leaves) / len(leaves), 2),
        "semantic_internal_share": round(sum(1 for n in internal if n["split_method"] in ("leiden", "tfidf-kmeans")) / max(1, len(internal)), 3),
        "mean_leaf_product_purity": round(sum(purity) / len(purity), 3) if purity else None,
        "cross_leaf_weight_fraction": cross_leaf,
        "multi_gold_pairs": {"pairs": total, "same_leaf": same_leaf, "same_parent": same_parent,
                             "lca_depth_distribution": dict(sorted(Counter(lca_depths).items()))},
    }


def main():
    global MAX_LEAF
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["leiden", "tfidf-kmeans"], default="leiden")
    parser.add_argument("--max-leaf", type=int, default=MAX_LEAF)
    args = parser.parse_args()
    MAX_LEAF = args.max_leaf

    docs = read_jsonl(CORPUS_PATH)
    by_id = {d["id"]: d for d in docs}
    questions = read_jsonl(QUESTIONS_PATH) if QUESTIONS_PATH.exists() else []
    graph = read_json(GRAPH_PATH)
    doc_ids = [n["id"] for n in graph["nodes"]]

    splitter = LeidenSplitter(graph) if args.method == "leiden" else TfidfKMeansSplitter(docs)
    nodes, root, doc_to_leaf = build_tree(splitter, doc_ids)
    assert set(doc_to_leaf) == set(doc_ids), "every document must land in exactly one leaf"

    stats = tree_statistics(nodes, root, doc_to_leaf, by_id, questions, splitter)
    stats["parameters"] = {"MAX_LEAF": MAX_LEAF, "MIN_LEAF": MIN_LEAF, "MIN_CHILDREN": MIN_CHILDREN,
                           "MAX_CHILDREN": MAX_CHILDREN, "MAX_DEPTH": MAX_DEPTH, "SEED": SEED}
    out = TREE_DIR / f"{args.method}_tree.json"
    write_json(out, {"schema_version": SCHEMA_VERSION, "method": args.method,
                     "source_graph": GRAPH_PATH.name, "parameters": stats["parameters"],
                     "root_id": root, "nodes": sorted(nodes.values(), key=lambda n: (n["depth"], n["id"])),
                     "doc_to_leaf": dict(sorted(doc_to_leaf.items()))})
    write_json(TREE_DIR / f"{args.method}_tree_stats.json", stats)

    print(f"========== DOCUMENT TREE ({args.method}) ==========")
    for k in ("documents", "tree_nodes", "internal_nodes", "leaves", "max_depth", "root_children",
              "branching", "leaf_sizes", "mean_leaf_size", "semantic_internal_share",
              "mean_leaf_product_purity", "cross_leaf_weight_fraction", "multi_gold_pairs"):
        print(f"{k}: {stats[k]}")
    # show the root split
    print("\nroot children (size | top products):")
    for cid in nodes[root]["children"]:
        c = nodes[cid]
        prods = Counter(re.sub(r"\s+Request$", "", by_id[d]["title"].split(":", 1)[0]).strip()
                        for d in c["doc_ids"] if ":" in by_id[d]["title"]).most_common(3)
        print(f"   {c['size']:4d} | {prods}")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
