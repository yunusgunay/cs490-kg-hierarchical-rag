# src/retrieval/baselines.py
"""
Non-LLM retrieval systems for step 7 (all share the evaluation protocol of
evaluate.py: ranked/opened documents, read costs).

Flat baselines (tree independent):
  bm25               BM25 over full articles (k1=1.5, b=0.75; same as the subset gate)
  dense              cosine over OpenAI text-embedding-3-small article embeddings

Tree systems (need <tree>_summaries.json):
  raptor_collapsed   RAPTOR collapsed-tree retrieval (Sarthi et al. 2024): one pool of
                     articles + every node summary, ranked by cosine. A summary occupies
                     a retrieval slot but cannot contain a gold article, so under a slot
                     budget this measures what summaries cost at article-level recall.
  raptor_traversal_bK  RAPTOR tree traversal: top-down beam of width K by cosine between
                     the question and node summaries; articles of the reached leaves are
                     ranked by cosine. Same tree, same summaries as our navigator, but
                     embedding routing instead of an LLM and no shortcuts.
  tree_bm25_bK       same traversal with BM25 scoring (lexical router ablation).

Usage
-----
    python src/retrieval/baselines.py                         # all systems, leiden tree
    python src/retrieval/baselines.py --tree both             # + tfidf-kmeans ablation tree
    python src/retrieval/baselines.py --systems bm25,dense    # flat baselines only (no embeddings
                                                         #  needed for bm25)
Outputs: data/processed/subset/eval/results/<system>__<tree>.json
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.common import (BM25, Embedder, TREES, doc_text, load_corpus, load_questions,  # noqa: E402
                         load_summaries, node_text, normalize_rows, num_tokens, quick_report,
                         save_results)

TOP_N = 20              # ranked documents kept per question (evaluate.py cuts at the budget)
BEAMS = (1, 2, 3)
FLAT_SYSTEMS = ("bm25", "dense")
TREE_SYSTEMS = ("raptor_collapsed", "raptor_traversal", "tree_bm25")


def base_record(q):
    return {"qid": q["qid"], "question": q["question"], "gold": q["gold"], "n_gold": q["n_gold"],
            "opened": [], "summaries_read": 0, "doc_rows_read": 0, "llm_calls": 0,
            "input_tokens": 0, "output_tokens": 0, "summary_tokens_read": 0}


# ------------------------------------------------------------
# flat baselines
# ------------------------------------------------------------

def run_bm25(docs, questions):
    index = BM25([d["id"] for d in docs], [doc_text(d) for d in docs])
    records = []
    for q in questions:
        r = base_record(q)
        r["opened"] = [d for _, d in index.rank(q["question"])[:TOP_N]]
        records.append(r)
    return records, {"k1": 1.5, "b": 0.75, "unit": "full article"}


def run_dense(docs, questions, embedder):
    doc_ids = [d["id"] for d in docs]
    D = normalize_rows(embedder.embed([doc_text(d) for d in docs], "(articles)"))
    Q = normalize_rows(embedder.embed([q["question"] for q in questions], "(questions)"))
    sims = Q @ D.T
    records = []
    for qi, q in enumerate(questions):
        r = base_record(q)
        order = np.argsort(-sims[qi], kind="stable")[:TOP_N]
        r["opened"] = [doc_ids[i] for i in order]
        records.append(r)
    return records, {"embedding_model": embedder.model, "unit": "full article (<=20k chars)"}


# ------------------------------------------------------------
# tree systems
# ------------------------------------------------------------

class TreeIndex:
    def __init__(self, method, docs, embedder=None):
        self.method = method
        data = load_summaries(method)
        self.tree = data["tree"]
        self.nodes = self.tree["by_id"]
        self.root = self.tree["root_id"]
        self.summaries = data["summaries"]
        self.node_ids = sorted(self.summaries)
        self.node_texts = {n: node_text(self.summaries[n]) for n in self.node_ids}
        self.node_tokens = {n: num_tokens(t) for n, t in self.node_texts.items()}
        self.docs = docs
        self.doc_ids = [d["id"] for d in docs]
        self.doc_index = {d: i for i, d in enumerate(self.doc_ids)}
        self.node_bm25 = BM25(self.node_ids, [self.node_texts[n] for n in self.node_ids])
        self.doc_bm25 = BM25(self.doc_ids, [doc_text(d) for d in docs])
        self.embedder = embedder
        self._D = self._N = None

    def doc_matrix(self):
        if self._D is None:
            self._D = normalize_rows(self.embedder.embed([doc_text(d) for d in self.docs], "(articles)"))
        return self._D

    def node_matrix(self):
        if self._N is None:
            self._N = normalize_rows(self.embedder.embed([self.node_texts[n] for n in self.node_ids],
                                                         f"({self.method} node summaries)"))
        return self._N


def embed_questions(embedder, questions):
    return normalize_rows(embedder.embed([q["question"] for q in questions], "(questions)"))


def run_raptor_collapsed(ti, questions, Q):
    D, N = ti.doc_matrix(), ti.node_matrix()
    pool_ids = [("doc", d) for d in ti.doc_ids] + [("summary", n) for n in ti.node_ids]
    P = np.vstack([D, N])
    sims = Q @ P.T
    records = []
    for qi, q in enumerate(questions):
        r = base_record(q)
        order = np.argsort(-sims[qi], kind="stable")[:TOP_N]
        slots = []
        for i in order:
            kind, ident = pool_ids[i]
            slots.append({"type": kind, "id": ident, "score": round(float(sims[qi, i]), 4),
                          "tokens": ti.node_tokens[ident] if kind == "summary" else 0})
        r["slots"] = slots
        r["opened"] = [s["id"] for s in slots if s["type"] == "doc"]
        r["summaries_read"] = sum(s["type"] == "summary" for s in slots[:10])
        r["summary_tokens_read"] = sum(s["tokens"] for s in slots[:10])
        records.append(r)
    return records, {"embedding_model": ti.embedder.model, "pool": "articles + all node summaries",
                     "note": "summaries_read/summary_tokens_read counted within the first 10 slots; "
                             "evaluate.py recomputes them at the budget"}


def traverse(ti, q, beam, scorer, doc_ranker):
    """Top-down beam search over node summaries.

    The frontier holds at most `beam` nodes with their scores. Children of every
    internal frontier node are scored; leaves already in the frontier keep their
    score and compete with the new candidates. Stops when the frontier is all leaves.
    Returns (ranked docs of the reached leaves, set of nodes scored, reached leaves).
    """
    frontier = {ti.root: 0.0}
    scored_nodes = set()
    while any(not ti.nodes[n]["is_leaf"] for n in frontier):
        candidates = [c for n in frontier if not ti.nodes[n]["is_leaf"] for c in ti.nodes[n]["children"]]
        kept = {n: s for n, s in frontier.items() if ti.nodes[n]["is_leaf"]}
        new_scores = {n: s for s, n in scorer(q, candidates)}      # scorer -> [(score, node_id)]
        scored_nodes.update(candidates)
        merged = sorted({**kept, **new_scores}.items(), key=lambda x: (-x[1], x[0]))
        frontier = dict(merged[:beam])
    leaves = list(frontier)
    doc_pool = [d for n in leaves for d in ti.nodes[n]["doc_ids"]]
    ranked_docs = doc_ranker(q, doc_pool)[:TOP_N]
    return ranked_docs, scored_nodes, leaves


def run_traversal(ti, questions, beam, mode, Q=None):
    if mode == "emb":
        N, D = ti.node_matrix(), ti.doc_matrix()
        nid_index = {n: i for i, n in enumerate(ti.node_ids)}

        def make_scorer(qi):
            def scorer(_, candidates):
                return [(float(Q[qi] @ N[nid_index[c]]), c) for c in candidates]

            def doc_ranker(_, doc_pool):
                return [d for _, d in sorted(((float(Q[qi] @ D[ti.doc_index[d]]), d) for d in doc_pool),
                                             key=lambda x: (-x[0], x[1]))]
            return scorer, doc_ranker
    else:
        def make_scorer(qi):
            def scorer(question, candidates):
                return ti.node_bm25.rank(question, candidates)

            def doc_ranker(question, doc_pool):
                return [d for _, d in ti.doc_bm25.rank(question, doc_pool)]
            return scorer, doc_ranker

    records = []
    for qi, q in enumerate(questions):
        scorer, doc_ranker = make_scorer(qi)
        ranked, scored_nodes, frontier = traverse(ti, q["question"], beam, scorer, doc_ranker)
        r = base_record(q)
        r["opened"] = ranked
        r["summaries_read"] = len(set(scored_nodes))
        r["summary_tokens_read"] = sum(ti.node_tokens[n] for n in set(scored_nodes))
        r["doc_rows_read"] = sum(ti.nodes[n]["size"] for n in frontier)
        r["trace"] = {"leaves_reached": frontier}
        records.append(r)
    cfg = {"beam": beam, "router": "cosine over node summaries" if mode == "emb" else "BM25 over node summaries",
           "doc_ranker": "cosine" if mode == "emb" else "BM25"}
    if mode == "emb":
        cfg["embedding_model"] = ti.embedder.model
    return records, cfg


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--systems", default="all",
                        help="comma list of bm25,dense,raptor_collapsed,raptor_traversal,tree_bm25 or 'all'")
    parser.add_argument("--tree", choices=list(TREES) + ["both"], default="leiden")
    parser.add_argument("--beams", default=",".join(map(str, BEAMS)))
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    args = parser.parse_args()

    if args.split == "heldout":
        print("WARNING: held-out split is for the final run only; do not tune on it.")
    systems = list(FLAT_SYSTEMS + TREE_SYSTEMS) if args.systems == "all" else args.systems.split(",")
    trees = list(TREES) if args.tree == "both" else [args.tree]
    beams = [int(b) for b in args.beams.split(",")]

    docs = load_corpus()
    questions = load_questions(args.split)
    print(f"corpus {len(docs)} articles | questions {len(questions)} ({args.split}) | systems {systems} | trees {trees}")

    needs_embeddings = any(s in systems for s in ("dense", "raptor_collapsed", "raptor_traversal"))
    embedder = Embedder() if needs_embeddings else None

    if "bm25" in systems:
        records, cfg = run_bm25(docs, questions)
        save_results("bm25", "none", args.split, cfg, records)
        print(quick_report("bm25", records))
    if "dense" in systems:
        records, cfg = run_dense(docs, questions, embedder)
        save_results("dense", "none", args.split, cfg, records)
        print(quick_report("dense", records))

    tree_systems = [s for s in systems if s in TREE_SYSTEMS]
    for method in trees if tree_systems else []:
        try:
            ti = TreeIndex(method, docs, embedder)
        except (FileNotFoundError, RuntimeError) as error:
            print(f"\n--- tree {method}: skipped ({error})")
            continue
        Q = embed_questions(embedder, questions) if embedder else None
        print(f"\n--- tree {method}: {len(ti.node_ids)} nodes ---")
        if "raptor_collapsed" in systems:
            records, cfg = run_raptor_collapsed(ti, questions, Q)
            save_results("raptor_collapsed", method, args.split, cfg, records)
            print(quick_report("raptor_collapsed", records))
        for beam in beams:
            if "raptor_traversal" in systems:
                records, cfg = run_traversal(ti, questions, beam, "emb", Q)
                name = f"raptor_traversal_b{beam}"
                save_results(name, method, args.split, cfg, records)
                print(quick_report(name, records))
            if "tree_bm25" in systems:
                records, cfg = run_traversal(ti, questions, beam, "bm25")
                name = f"tree_bm25_b{beam}"
                save_results(name, method, args.split, cfg, records)
                print(quick_report(name, records))
    print("\nDone. Next: python src/retrieval/navigate_tree.py ; python src/retrieval/evaluate.py")


if __name__ == "__main__":
    main()
