# src/kg/select_shortcuts.py
"""
Cross-branch shortcuts from cut KG edges (step 6b) — the proposal's
core mechanism.

For every tree node X, take the document pairs that connect X to the
rest of the corpus and turn them into a small, budgeted set of typed
links to other tree nodes.

Candidate pools and selection policies
--------------------------------------
  typed         pool = typed KG bridges (bridge_candidates.jsonl: the
                dependency relations REQUIRES / RESOLVED_BY / CAUSES /
                APPLIES_TO / HAS_LIMITATION / LOCATED_IN through
                specific entities, counted once per (A->B, entity)).
                This pool is independent of the top-K pruning used for
                Leiden, so no cut KG relation is lost. Ranked by mass.
  random_typed  same pool as `typed`, destinations ranked at random
                (isolates "which relations are candidates" from "how
                candidates are ranked").
  strongest     pool = document-graph edges, value = total edge weight
                (shared entities + bridges). Ranked by mass.
  random        pool = document-graph edges, uniform value, random
                ranking (uninformed control).

Multi-resolution destination: for each ancestor A of X and each sibling
branch S under A that does not contain X, descend from S while one
child holds >= REFINE_SHARE of S's connection mass to X; the result is
the deepest node that still captures most of the evidence (A2 -> B11
rather than A2 -> B1). A destination is kept only if the refined target
holds <= MAX_TARGET_SIZE articles: a link whose evidence is spread over a
whole branch is not a shortcut (the navigator reaches that branch with
ordinary tree moves) and would only cost reading budget. Budget: top
BUDGET destinations per node.

Outputs (graph/trees/shortcuts/)
  <tree>_shortcuts_<policy>.json       node_id -> [shortcut, ...]
  <tree>_shortcuts_<policy>_stats.json structural coverage of multi-gold
      pairs that live in different leaves: whether a shortcut from the
      leaf itself / the leaf or its parent / any ancestor points to a
      node containing the other gold article. This is a STRUCTURAL
      upper bound on what navigation could exploit with one jump, not
      a navigation result.
"""
import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/subset"
GRAPH_DIR = DATA_DIR / "graph"
TREE_DIR = GRAPH_DIR / "trees"
OUT_DIR = TREE_DIR / "shortcuts"
GRAPH_PATH = GRAPH_DIR / "doc_graph.json"
BRIDGES_PATH = GRAPH_DIR / "bridge_candidates.jsonl"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"

SCHEMA_VERSION = "shortcuts-v2"
BUDGET = 3            # shortcuts kept per node
REFINE_SHARE = 0.6    # descend while a child holds this share of the mass
MIN_MASS = 0.5        # ignore destinations with negligible mass (ranked policies)
MAX_TARGET_SIZE = 24  # a shortcut must land in a small neighbourhood (<= 3 leaves' worth of
                      # articles); a diffuse link to a whole branch is reachable by ordinary
                      # tree moves and only costs the navigator reading budget
SEED = 42
POLICIES = ["typed", "random_typed", "strongest", "random"]


def read_json(p):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(p):
    if not p.exists():
        return []
    with p.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(p, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


class Tree:
    def __init__(self, tree_json):
        self.nodes = {n["id"]: n for n in tree_json["nodes"]}
        self.root = tree_json["root_id"]
        self.leaf_of = tree_json["doc_to_leaf"]
        self.parent = {n["id"]: n["parent_id"] for n in self.nodes.values()}
        self.docset = {n["id"]: set(n["doc_ids"]) for n in self.nodes.values()}

    def ancestors(self, node_id):
        out = []
        while node_id is not None:
            out.append(node_id)
            node_id = self.parent[node_id]
        return out                      # node itself first, root last


# ------------------------------------------------------------
# candidate pools: pair -> (value, support list)
# ------------------------------------------------------------

def typed_pool(bridges):
    """Typed bridges aggregated per unordered document pair; one count per
    (from -> to, entity) with the strongest predicate; full provenance kept."""
    per_key = {}
    for b in bridges:
        key = (b["from_doc"], b["to_doc"], b["via_entity"])
        if b["weight"] > per_key.get(key, (0.0, None))[0]:
            per_key[key] = (b["weight"], b)
    pool = {}
    for (a, b, e), (w, rec) in per_key.items():
        pair = (a, b) if a < b else (b, a)
        entry = pool.setdefault(pair, {"value": 0.0, "support": []})
        entry["value"] += w
        entry["support"].append({
            "from_doc": rec["from_doc"], "to_doc": rec["to_doc"], "via": rec["via_entity"],
            "predicate": rec["predicate"], "subject": rec["subject"], "object": rec["object"],
            "polarity": rec["polarity"], "certainty": rec["certainty"],
            "condition": rec.get("condition"), "evidence": (rec.get("evidence") or "")[:200],
            "weight": rec["weight"],
        })
    for entry in pool.values():
        entry["support"].sort(key=lambda s: -s["weight"])
    return pool


def graph_pool(graph, uniform):
    pool = {}
    for e in graph["edges"]:
        pair = (e["source"], e["target"]) if e["source"] < e["target"] else (e["target"], e["source"])
        pool[pair] = {
            "value": 1.0 if uniform else e["weight"],
            "support": [{
                "from_doc": e["source"], "to_doc": e["target"],
                "shared_entities": [s[0] for s in e.get("shared_entities", [])[:5]],
                "relations": [{"predicate": br["predicate"], "subject": br["subject"],
                               "object": br["object"], "via": br["via"]}
                              for br in e.get("bridges", [])[:3]],
                "weight": e["weight"],
            }],
        }
    return pool


# ------------------------------------------------------------
# shortcut construction
# ------------------------------------------------------------

def build_shortcuts(tree, pool, ranked, budget, rng, max_target_size=MAX_TARGET_SIZE):
    adj = defaultdict(list)
    for (a, b), entry in pool.items():
        adj[a].append((b, entry["value"], entry))
        adj[b].append((a, entry["value"], entry))

    shortcuts = {}
    for X in tree.nodes:
        inside = tree.docset[X]
        cross = [(b, v, entry, a) for a in inside for b, v, entry in adj.get(a, ()) if b not in inside]
        if not cross:
            shortcuts[X] = []
            continue
        candidates = []
        anc = tree.ancestors(X)
        for depth_up, A in enumerate(anc[1:], start=1):
            child_on_path = anc[depth_up - 1]
            for S in tree.nodes[A]["children"]:
                if S == child_on_path:
                    continue
                mass_by_doc = defaultdict(float)
                for b, v, entry, a in cross:
                    if b in tree.docset[S]:
                        mass_by_doc[b] += v
                total = sum(mass_by_doc.values())
                if total < (MIN_MASS if ranked else 1e-9):
                    continue
                target = S
                while not tree.nodes[target]["is_leaf"]:
                    child_mass = {c: sum(m for d, m in mass_by_doc.items() if d in tree.docset[c])
                                  for c in tree.nodes[target]["children"]}
                    best, best_mass = max(child_mass.items(), key=lambda kv: (kv[1], kv[0]))
                    if best_mass >= REFINE_SHARE * total:
                        target = best
                    else:
                        break
                if tree.nodes[target]["size"] > max_target_size:
                    continue                      # diffuse link, not a shortcut
                support = []
                for b, v, entry, a in sorted(cross, key=lambda t: -t[1]):
                    if b in tree.docset[target]:
                        support.append({"from_doc": a, "to_doc": b, "value": round(v, 3),
                                        "evidence": entry["support"][:3]})
                    if len(support) >= 3:
                        break
                candidates.append({
                    "target": target, "sibling_branch": S, "via_ancestor_depth": depth_up,
                    "mass": round(total, 3),
                    "n_pairs": sum(1 for b, *_ in cross if b in tree.docset[S]),
                    "support": support,
                })
        if ranked:
            chosen = sorted(candidates, key=lambda c: (-c["mass"], c["target"]))[:budget]
        else:
            rng.shuffle(candidates)
            chosen = candidates[:budget]
        shortcuts[X] = chosen
    return shortcuts


# ------------------------------------------------------------
# structural coverage
# ------------------------------------------------------------

def structural_coverage(tree, shortcuts, questions):
    cov = Counter()
    examples = {"reachable_from_leaf": [], "unreachable": []}

    def reach_level(src, dst):
        """0 = shortcut on src's leaf, 1 = on its parent, k = k levels up; None = none."""
        for level, node in enumerate(tree.ancestors(tree.leaf_of[src])):
            for sc in shortcuts.get(node, []):
                if dst in tree.docset[sc["target"]]:
                    return level, node, sc
        return None

    for q in questions:
        ids = [i for i in q["article_ids"] if i in tree.leaf_of]
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                if tree.leaf_of[a] == tree.leaf_of[b]:
                    cov["same_leaf"] += 1
                    continue
                cov["split_pairs"] += 1
                hits = [h for h in (reach_level(a, b), reach_level(b, a)) if h]
                if not hits:
                    if len(examples["unreachable"]) < 8:
                        examples["unreachable"].append(q["question"][:90])
                    continue
                level, node, sc = min(hits, key=lambda h: h[0])
                cov["from_any_ancestor"] += 1
                if level <= 1:
                    cov["from_leaf_or_parent"] += 1
                if level == 0:
                    cov["from_leaf"] += 1
                    if len(examples["reachable_from_leaf"]) < 6:
                        ev = sc["support"][0]["evidence"][0] if sc["support"] and sc["support"][0]["evidence"] else {}
                        examples["reachable_from_leaf"].append({
                            "question": q["question"][:90], "target_size": tree.nodes[sc["target"]]["size"],
                            "mass": sc["mass"], "relation": {k: ev.get(k) for k in ("predicate", "subject", "object", "via") if k in ev}})
                if tree.nodes[sc["target"]]["size"] <= 8:
                    cov["target_leaf_sized"] += 1
    return dict(cov), examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", choices=["leiden", "tfidf-kmeans"], default="leiden")
    parser.add_argument("--policy", choices=POLICIES + ["all"], default="all")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--max-target-size", type=int, default=MAX_TARGET_SIZE,
                        help="drop candidates whose refined target holds more articles than this")
    args = parser.parse_args()

    tree = Tree(read_json(TREE_DIR / f"{args.tree}_tree.json"))
    graph = read_json(GRAPH_PATH)
    bridges = read_jsonl(BRIDGES_PATH)
    questions = read_jsonl(QUESTIONS_PATH)
    pruned_pairs = {(e["source"], e["target"]) if e["source"] < e["target"] else (e["target"], e["source"])
                    for e in graph["edges"]}
    tpool = typed_pool(bridges)
    outside = sum(1 for p in tpool if p not in pruned_pairs)
    print(f"========== SHORTCUTS ({args.tree}, budget {args.budget}, max target size {args.max_target_size}) ==========")
    print(f"typed pool: {len(tpool)} document pairs from {len(bridges)} bridge records "
          f"({outside} pairs not in the pruned graph, kept) | graph pool: {len(pruned_pairs)} edges")

    policies = POLICIES if args.policy == "all" else [args.policy]
    for policy in policies:
        rng = random.Random(SEED)
        if policy == "typed":
            sc = build_shortcuts(tree, tpool, ranked=True, budget=args.budget, rng=rng, max_target_size=args.max_target_size)
        elif policy == "random_typed":
            sc = build_shortcuts(tree, tpool, ranked=False, budget=args.budget, rng=rng, max_target_size=args.max_target_size)
        elif policy == "strongest":
            sc = build_shortcuts(tree, graph_pool(graph, uniform=False), ranked=True, budget=args.budget, rng=rng, max_target_size=args.max_target_size)
        else:
            sc = build_shortcuts(tree, graph_pool(graph, uniform=True), ranked=False, budget=args.budget, rng=rng, max_target_size=args.max_target_size)
        n_links = sum(len(v) for v in sc.values())
        nodes_with = sum(1 for v in sc.values() if v)
        tdepth = Counter(tree.nodes[s["target"]]["depth"] for v in sc.values() for s in v)
        leaf_targets = sum(1 for v in sc.values() for s in v if tree.nodes[s["target"]]["is_leaf"])
        cov, examples = structural_coverage(tree, sc, questions)
        stats = {"schema_version": SCHEMA_VERSION, "tree": args.tree, "policy": policy, "budget": args.budget,
                 "refine_share": REFINE_SHARE, "max_target_size": args.max_target_size, "nodes": len(tree.nodes), "nodes_with_shortcuts": nodes_with,
                 "links": n_links, "links_to_leaves": leaf_targets,
                 "target_depth_distribution": dict(sorted(tdepth.items())),
                 "structural_coverage": cov, "examples": examples}
        write_json(OUT_DIR / f"{args.tree}_shortcuts_{policy}.json",
                   {"schema_version": SCHEMA_VERSION, "tree": args.tree, "policy": policy,
                    "budget": args.budget, "shortcuts": sc})
        write_json(OUT_DIR / f"{args.tree}_shortcuts_{policy}_stats.json", stats)
        print(f"\n[{policy}] links {n_links} on {nodes_with}/{len(tree.nodes)} nodes | to leaves {leaf_targets} "
              f"| target depth {dict(sorted(tdepth.items()))}")
        print(f"   structural coverage of split multi-gold pairs: {cov}")
        for ex in examples["reachable_from_leaf"][:3]:
            r = ex["relation"]
            print(f"   from leaf: '{ex['question'][:58]}' -> target({ex['target_size']} docs), mass {ex['mass']}"
                  + (f" | {r.get('predicate')}: {r.get('subject')} -> {r.get('object')}" if r.get("predicate") else ""))
    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
