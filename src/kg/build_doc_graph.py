# src/kg/build_doc_graph.py
"""
Document graph from the canonical KG (step 4).

Replaces the passage-level graph of the first pipeline. Nodes are
documents (gold labels are article-level, and a document must have one
canonical place in the tree). Edge weight between documents A and B:

    w(A,B) = w_shared + w_bridge + w_prefix + w_prior

  w_shared  sum over canonical entities shared by A and B of
            idf(e) * type_weight(type(e)); very frequent entities
            (df > MAX_DF_RATIO * N) and generic names are ignored
  w_bridge  typed KG relation in A whose endpoint is a *central*
            entity of B (title product or top-3 entity of B);
            weight = predicate_weight * idf(endpoint), counted once per
            (A -> B, entity). These bridges are also written out
            separately as POTENTIAL shortcut candidates: the actual
            shortcuts are chosen in step 6 among the bridges whose
            endpoints land in different branches of the tree.
  w_prefix  PREFIX_W when both titles carry the same "Product:" prefix
            (the proposal allows headings to seed partitions)
  w_prior   PRIOR_W * cosine for TF-IDF nearest neighbours (k=PRIOR_K);
            a weak lexical prior that improves connectivity for documents
            with few shared KG entities, without guaranteeing connectivity.

Order of operations matters (the earlier pipeline got it wrong):
filter generic entities -> weight -> THEN prune to each document's
top-K neighbours (union of kNN lists).

Outputs (data/processed/subset/graph/)
  doc_graph.json            nodes, weighted edges with explanations
  bridge_candidates.jsonl   typed cross-document relations (potential
                            shortcut candidates, selected in step 6)
  doc_graph_stats.json      connectivity, weight shares, gold-pair checks
"""
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/subset"
GRAPH_DIR = DATA_DIR / "graph"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"

MODEL_RUN = "gpt-6-luna_medium"
CANON_RECORDS_PATH = DATA_DIR / f"kg_v5_{MODEL_RUN}_canonical.jsonl"

SCHEMA_VERSION = "doc-graph-v1"

# ---- weights ----
TYPE_WEIGHT = {
    "FEATURE": 1.0, "SETTING": 1.0, "LOCATION": 1.0, "ISSUE": 1.0,
    "TASK": 0.8, "PLAN": 0.8, "EXTERNAL": 0.8,
    "CONCEPT": 0.6, "PRODUCT": 0.4,
}
PREDICATE_WEIGHT = {
    "REQUIRES": 1.0, "RESOLVED_BY": 1.0,
    "CAUSES": 0.8, "APPLIES_TO": 0.8, "HAS_LIMITATION": 0.8,
    "LOCATED_IN": 0.6, "PART_OF": 0.5, "SUPPORTS": 0.4, "RELATED_TO": 0.3,
}
MAX_DF_RATIO = 0.25       # entities in more than 25% of docs carry no topical signal
GENERIC_NAMES = {"wix", "site", "wix site", "website", "page", "user", "visitor",
                 "wix account", "dashboard", "site dashboard"}
PREFIX_W = 0.5
PRIOR_K = 5
PRIOR_W = 0.15
KNN_K = 20
CENTRAL_TOP = 3           # an entity is "central" to a doc if it is among its top-3
# Bridges = the proposal's shortcut candidates: dependency-type relations
# through specific (non-product, low-df) entities. Taxonomic relations
# (PART_OF / SUPPORTS to a product) are covered by prefix + shared weights.
BRIDGE_PREDICATES = {"REQUIRES", "RESOLVED_BY", "CAUSES", "APPLIES_TO",
                     "HAS_LIMITATION", "LOCATED_IN"}
BRIDGE_MAX_DF_RATIO = 0.05


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def title_product(title):
    if not title or ":" not in title:
        return None
    p = re.sub(r"\s+Request$", "", title.split(":", 1)[0].strip()).strip()
    if not p or p.casefold() == "request" or len(p.split()) > 5:
        return None
    return p.casefold()


# ------------------------------------------------------------
# TF-IDF prior
# ------------------------------------------------------------

def tfidf_neighbours(docs, k):
    tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    toks = {d["id"]: tok(d["title"] + " " + d["contents"][:4000]) for d in docs}
    df = Counter()
    for t in toks.values():
        df.update(set(t))
    n = len(docs)
    vec = {}
    for i, t in toks.items():
        c = Counter(t)
        v = {w: (1 + math.log(x)) * math.log(n / df[w]) for w, x in c.items() if df[w] < 0.5 * n}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vec[i] = {w: x / norm for w, x in v.items()}
    inv = defaultdict(list)
    for i, v in vec.items():
        for w, x in v.items():
            inv[w].append((i, x))
    out = {}
    for i, v in vec.items():
        s = defaultdict(float)
        for w, x in v.items():
            for j, y in inv[w]:
                if j != i:
                    s[j] += x * y
        out[i] = sorted(s.items(), key=lambda kv: (-kv[1], kv[0]))[:k]
    return out


# ------------------------------------------------------------
# main build
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--knn", type=int, default=KNN_K)
    parser.add_argument("--no-prior", action="store_true", help="Drop TF-IDF prior edges (ablation)")
    parser.add_argument("--no-prefix", action="store_true", help="Drop title-prefix edges (ablation)")
    args = parser.parse_args()

    docs = read_jsonl(CORPUS_PATH)
    by_id = {d["id"]: d for d in docs}
    records = read_jsonl(CANON_RECORDS_PATH)
    if not records:
        parser.error(f"Missing canonical records: {CANON_RECORDS_PATH}")
    n = len(docs)

    # ---- document -> entities (with per-doc mention weight) ----
    ent_name = {}
    ent_type_counts = defaultdict(Counter)   # entity id -> type -> mentions
    doc_ent_count = defaultdict(Counter)     # doc -> entity id -> mentions
    for r in records:
        for e in r["entities"]:
            ent_name[e["id"]] = e["name"]
            ent_type_counts[e["id"]][e["type"]] += 1
            doc_ent_count[r["document_id"]][e["id"]] += 1
        for x in r["relations"]:
            doc_ent_count[r["document_id"]][x["subject_id"]] += 1
            doc_ent_count[r["document_id"]][x["object_id"]] += 1
    # One type per canonical entity: majority vote, alphabetical tie-break
    # (deterministic; the same id may carry different types in different
    # articles, e.g. "Members Area" PRODUCT vs FEATURE).
    ent_type = {e: sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
                for e, c in ent_type_counts.items()}
    type_conflicts = {ent_name[e]: dict(c) for e, c in ent_type_counts.items() if len(c) > 1}

    ent_docs = defaultdict(set)
    for d, c in doc_ent_count.items():
        for e in c:
            ent_docs[e].add(d)
    idf = {e: math.log(n / len(ds)) for e, ds in ent_docs.items()}
    max_df = MAX_DF_RATIO * n
    usable = {e for e, ds in ent_docs.items()
              if 1 < len(ds) <= max_df and ent_name[e].casefold() not in GENERIC_NAMES}
    dropped_generic = sorted(
        ((len(ds), ent_name[e]) for e, ds in ent_docs.items()
         if len(ds) > max_df or ent_name[e].casefold() in GENERIC_NAMES), reverse=True)

    # central entities per doc: title product entity + top-3 by mentions
    product = {d["id"]: title_product(d["title"]) for d in docs}
    central = {}
    for d, c in doc_ent_count.items():
        top = {e for e, _ in c.most_common(CENTRAL_TOP)}
        for e in c:
            if product[d] and ent_name[e].casefold() == product[d]:
                top.add(e)
        central[d] = top

    # ---- 1. shared-entity weights ----
    edges = defaultdict(lambda: {"w_shared": 0.0, "w_bridge": 0.0, "w_prefix": 0.0,
                                 "w_prior": 0.0, "shared": [], "bridges": []})
    pair = lambda a, b: (a, b) if a < b else (b, a)
    for e in usable:
        ds = sorted(ent_docs[e])
        w = idf[e] * TYPE_WEIGHT.get(ent_type[e], 0.6)
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                ed = edges[pair(ds[i], ds[j])]
                ed["w_shared"] += w
                ed["shared"].append((ent_name[e], round(w, 3)))

    # ---- 2. typed relation bridges (shortcut candidates) ----
    bridges = []
    bridge_max_df = BRIDGE_MAX_DF_RATIO * n
    # One bridge per (source doc -> target doc, entity): a list-style
    # article with 16 "processing fee APPLIES_TO <country>" relations
    # must not count 16 times. The weight is the strongest predicate seen
    # through that entity; every relation is still kept as provenance.
    bridge_weight = {}          # (a, b, e) -> max predicate weight * idf
    for r in records:
        a = r["document_id"]
        for x in r["relations"]:
            if x["predicate"] not in BRIDGE_PREDICATES:
                continue
            for role, e in (("object", x["object_id"]), ("subject", x["subject_id"])):
                if (e not in usable or ent_type[e] == "PRODUCT"
                        or len(ent_docs[e]) > bridge_max_df):
                    continue
                for b in ent_docs[e]:
                    if b == a or e not in central.get(b, ()):
                        continue
                    w = PREDICATE_WEIGHT.get(x["predicate"], 0.3) * idf[e]
                    key = (a, b, e)
                    if w > bridge_weight.get(key, 0.0):
                        ed = edges[pair(a, b)]
                        ed["w_bridge"] += w - bridge_weight.get(key, 0.0)
                        bridge_weight[key] = w
                    ed = edges[pair(a, b)]
                    ed["bridges"].append({
                        "from": a, "to": b, "via": ent_name[e], "role": role,
                        "predicate": x["predicate"], "polarity": x["polarity"],
                        "subject": x["subject"], "object": x["object"],
                        "weight": round(w, 3),
                    })
                    bridges.append({
                        "from_doc": a, "to_doc": b, "via_entity": ent_name[e],
                        "endpoint_role": role, "predicate": x["predicate"],
                        "polarity": x["polarity"], "certainty": x["certainty"],
                        "subject": x["subject"], "object": x["object"],
                        "condition": x.get("condition"), "evidence": x["evidence"],
                        "weight": round(w, 3),
                    })

    # ---- 3. title-prefix edges ----
    if not args.no_prefix:
        by_prefix = defaultdict(list)
        for d in docs:
            if product[d["id"]]:
                by_prefix[product[d["id"]]].append(d["id"])
        for ids in by_prefix.values():
            ids = sorted(ids)
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    edges[pair(ids[i], ids[j])]["w_prefix"] = PREFIX_W

    # ---- 4. TF-IDF prior ----
    if not args.no_prior:
        for a, nbrs in tfidf_neighbours(docs, PRIOR_K).items():
            for b, cos in nbrs:
                ed = edges[pair(a, b)]
                ed["w_prior"] = max(ed["w_prior"], round(PRIOR_W * cos, 4))

    for ed in edges.values():
        ed["weight"] = ed["w_shared"] + ed["w_bridge"] + ed["w_prefix"] + ed["w_prior"]

    # ---- 5. prune: union of each document's top-K neighbours ----
    nbr = defaultdict(list)
    for (a, b), ed in edges.items():
        nbr[a].append((ed["weight"], b))
        nbr[b].append((ed["weight"], a))
    keep = set()
    for a, lst in nbr.items():
        for w, b in sorted(lst, key=lambda t: (-t[0], t[1]))[:args.knn]:
            keep.add(pair(a, b))
    pruned = {p: edges[p] for p in keep}

    # ---- stats ----
    parent = {d["id"]: d["id"] for d in docs}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in pruned:
        parent[find(a)] = find(b)
    sizes = sorted(Counter(find(d) for d in parent).values(), reverse=True)
    total = sum(ed["weight"] for ed in pruned.values()) or 1.0
    share = {k: round(sum(ed[k] for ed in pruned.values()) / total, 3)
             for k in ("w_shared", "w_bridge", "w_prefix", "w_prior")}
    glue = Counter()
    for ed in pruned.values():
        for name, _ in ed["shared"]:
            glue[name] += 1

    gold_stats = {}
    if QUESTIONS_PATH.exists():
        direct = comp = total_pairs = 0
        missing = []
        for q in read_jsonl(QUESTIONS_PATH):
            ids = [i for i in q["article_ids"] if i in by_id]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    total_pairs += 1
                    p = pair(ids[i], ids[j])
                    if p in pruned:
                        direct += 1
                    else:
                        missing.append([by_id[ids[i]]["title"][:50], by_id[ids[j]]["title"][:50],
                                        round(edges[p]["weight"], 3) if p in edges else 0.0])
                    comp += find(ids[i]) == find(ids[j])
        gold_stats = {"pairs": total_pairs, "direct_edge": direct, "same_component": comp,
                      "pairs_without_direct_edge": missing}

    stats = {
        "schema_version": SCHEMA_VERSION,
        "documents": n, "edges_before_prune": len(edges), "edges": len(pruned),
        "knn": args.knn, "prior": not args.no_prior, "prefix": not args.no_prefix,
        "usable_entities": len(usable), "generic_or_rare_entities_ignored": len(ent_docs) - len(usable),
        "top_ignored_generic": dropped_generic[:12],
        "entity_type_conflicts_resolved_by_majority": type_conflicts,
        "components": len(sizes), "largest_component": sizes[0], "isolated_docs": sum(s == 1 for s in sizes),
        "weight_share": share,
        "bridge_candidates": len(bridges),
        "bridge_predicates": dict(Counter(b["predicate"] for b in bridges).most_common()),
        "top_glue_entities": glue.most_common(20),
        "gold_pairs": gold_stats,
        "parameters": {"TYPE_WEIGHT": TYPE_WEIGHT, "PREDICATE_WEIGHT": PREDICATE_WEIGHT,
                       "MAX_DF_RATIO": MAX_DF_RATIO, "PREFIX_W": PREFIX_W,
                       "BRIDGE_PREDICATES": sorted(BRIDGE_PREDICATES),
                       "BRIDGE_MAX_DF_RATIO": BRIDGE_MAX_DF_RATIO,
                       "PRIOR_K": PRIOR_K, "PRIOR_W": PRIOR_W, "CENTRAL_TOP": CENTRAL_TOP},
    }

    # ---- export ----
    nodes = [{"id": d["id"], "title": d["title"], "product": product[d["id"]],
              "article_type": d["article_type"], "n_entities": len(doc_ent_count[d["id"]])}
             for d in docs]
    edge_rows = []
    for (a, b), ed in sorted(pruned.items(), key=lambda kv: -kv[1]["weight"]):
        edge_rows.append({
            "source": a, "target": b, "weight": round(ed["weight"], 4),
            "w_shared": round(ed["w_shared"], 4), "w_bridge": round(ed["w_bridge"], 4),
            "w_prefix": ed["w_prefix"], "w_prior": ed["w_prior"],
            "shared_entities": sorted(ed["shared"], key=lambda t: -t[1])[:10],
            "bridges": ed["bridges"][:10],
        })
    write_json(GRAPH_DIR / "doc_graph.json", {
        "schema_version": SCHEMA_VERSION, "source": CANON_RECORDS_PATH.name,
        "parameters": stats["parameters"], "nodes": nodes, "edges": edge_rows,
    })
    with (GRAPH_DIR / "bridge_candidates.jsonl").open("w", encoding="utf-8") as f:
        for b in sorted(bridges, key=lambda b: -b["weight"]):
            f.write(json.dumps(b, ensure_ascii=False) + "\n")
    write_json(GRAPH_DIR / "doc_graph_stats.json", stats)

    print("========== DOCUMENT GRAPH ==========")
    print(f"documents {n} | usable entities {len(usable)} (ignored {len(ent_docs) - len(usable)}: "
          f"generic/df>{int(max_df)} or single-doc)")
    print(f"edges {len(edges)} -> {len(pruned)} after top-{args.knn} pruning")
    print(f"components {len(sizes)} | largest {sizes[0]} | isolated docs {sum(s == 1 for s in sizes)}")
    print(f"weight share: {share}")
    print(f"potential shortcut candidates (typed bridges): {len(bridges)} | {stats['bridge_predicates']}")
    if type_conflicts:
        print(f"entity type conflicts resolved by majority vote: {type_conflicts}")
    print(f"top glue entities: {[f'{k} ({v})' for k, v in glue.most_common(12)]}")
    print(f"ignored generic: {[f'{nme} ({c})' for c, nme in dropped_generic[:8]]}")
    if gold_stats:
        print(f"multi-gold pairs: {gold_stats['pairs']} | direct edge {gold_stats['direct_edge']} "
              f"| same component {gold_stats['same_component']}")
        for m in gold_stats["pairs_without_direct_edge"][:12]:
            print("   no direct edge:", m)
    print("strongest edges:")
    for e in edge_rows[:6]:
        print(f"   {e['weight']:6.2f} | {by_id[e['source']]['title'][:38]} <-> "
              f"{by_id[e['target']]['title'][:38]} | {[s[0] for s in e['shared_entities'][:3]]}")
    print(f"\nSaved: {GRAPH_DIR / 'doc_graph.json'}")


if __name__ == "__main__":
    main()
