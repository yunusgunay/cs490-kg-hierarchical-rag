# src/kg/canonicalize_entities.py
"""
Entity canonicalization for KG v5.

Problem: the same concept appears under several names across articles
("site dashboard" / "dashboard" / "site's dashboard"), so documents that
talk about the same thing are not connected in the graph.

Pipeline
--------
1. Collect entity names from kg_v5_*_extracted.jsonl; rule-based
   normalization (case, articles, possessive, simple plurals, "&"/"and").
2. Candidate pairs: embedding cosine (OpenAI embeddings, cached) and
   lexical overlap; the model's own per-article aliases add candidates.
3. Candidate components are shown to an LLM that returns groups of
   names that denote the SAME concept (not parent/child), with a
   canonical name and type. Batched, checkpointed.
4. Outputs
   graph/entity_canon.json          raw name -> canonical name/type/id
   kg_v5_<run>_canonical.jsonl      extraction records with canonical names
   and a before/after report (singleton share, document connectivity).

Usage
-----
    python src/kg/canonicalize_entities.py --dry-run     # candidates only
    python src/kg/canonicalize_entities.py               # full
    python src/kg/canonicalize_entities.py --no-embeddings   # lexical only
"""
import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/subset"
GRAPH_DIR = DATA_DIR / "graph"

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv("LLM_MODEL", "gpt-6-luna").strip()
REASONING_EFFORT = os.getenv("LLM_REASONING_EFFORT", "medium").strip().lower()
IS_REASONING_MODEL = MODEL_NAME.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()

SAFE_MODEL = re.sub(r"[^a-zA-Z0-9_-]", "_", MODEL_NAME)
RUN_NAME = SAFE_MODEL + (f"_{REASONING_EFFORT}" if IS_REASONING_MODEL else "")

INPUT_PATH = DATA_DIR / f"kg_v5_{RUN_NAME}_extracted.jsonl"
CANON_PATH = GRAPH_DIR / "entity_canon.json"
CANON_RECORDS_PATH = DATA_DIR / f"kg_v5_{RUN_NAME}_canonical.jsonl"
EMBED_CACHE_PATH = GRAPH_DIR / "entity_embeddings_cache.jsonl"
DECISION_CHECKPOINT = GRAPH_DIR / "entity_merge_decisions.jsonl"

SCHEMA_VERSION = "entity-canon-v1"

EMBED_THRESHOLD = 0.80      # cosine between "name (TYPE)" strings
LEXICAL_JACCARD = 0.5       # token-set Jaccard after normalization
MAX_COMPONENT = 12          # larger candidate components are split
LLM_BATCH_GROUPS = 25       # candidate components per LLM call

GENERIC_NEVER_MERGE = {"wix", "site", "wix site", "website", "page", "user", "visitor"}


# ============================================================
# 2. SCHEMA FOR LLM DECISIONS
# ============================================================

EntityType = Literal["PRODUCT", "FEATURE", "SETTING", "LOCATION", "PLAN",
                     "TASK", "ISSUE", "EXTERNAL", "CONCEPT"]


class MergeGroup(BaseModel):
    group_id: int
    canonical_name: str
    type: EntityType
    members: list[str]


class MergeDecision(BaseModel):
    groups: list[MergeGroup]


MERGE_PROMPT = """
You resolve entity names extracted from Wix help-center articles.
Each candidate group lists names that MIGHT denote the same concept.
For every group decide which names are the SAME concept and return
one merge set per concept.

Rules
1. Merge only true synonyms / spelling variants / singular-plural /
   word-order variants: "site dashboard" = "dashboard" = "site's
   dashboard"; "domain transfer" = "transfer domain"; "Premium plan" =
   "premium subscription".
2. Never merge a concept with its parent, child, or a narrower variant:
   "SEO dashboard" is NOT "dashboard"; "connect domain" is NOT
   "domain"; "Wix Editor" is NOT "Studio Editor"; "business email" is
   NOT "email".
3. Never merge a product with a feature of that product.
   Different codes, numbers, products, plans or apps are different
   concepts: "401 error" is NOT "403 error"; "Premium plan" is NOT
   "Studio plan"; "Pricing Plans" (the app) is NOT "Premium plan"
   (Wix subscription); "Fit by Wix" is NOT "Wix app".
4. A merge set needs at least 2 members. Names you do not list stay
   separate. Only use names that appear in the group.
5. canonical_name: the most common, shortest unambiguous form, 1-4
   words. Keep official product names in their official casing.
6. type: the best type for the merged concept.
"""


# ============================================================
# 3. UTILITIES
# ============================================================

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def rule_normalize(name, keep_plural=False):
    """
    Deterministic normalization used as the merge key of step 1.

    keep_plural=True for PRODUCT names: "Pricing Plans" (the app) must
    not collapse onto "pricing plan" (one plan) before the LLM sees
    them; singular/plural product variants are left to step 3.
    """
    n = name.casefold().strip()
    n = n.replace("&", " and ").replace("’", "'")
    n = re.sub(r"[\"“”]", "", n)
    n = re.sub(r"\s+", " ", n).strip(" .,;:'")
    n = re.sub(r"^(the|a|an|your|my) ", "", n)
    n = re.sub(r"\b(\w+)'s\b", r"\1", n)          # site's -> site
    n = re.sub(r"\bwix's\b", "wix", n)
    if keep_plural:
        return n
    words = []
    for w in n.split():
        if len(w) > 4 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
            w = w[:-1]
        words.append(w)
    return " ".join(words)


def entity_key(name, entity_type):
    return rule_normalize(name, keep_plural=(entity_type == "PRODUCT"))


def digits_of(name):
    return tuple(sorted(re.findall(r"\d+", name)))


def safe_to_merge(a, b, ents):
    """
    Deterministic guards applied on top of the LLM decision.
    Returns None when the pair may merge, else a reason string.
    """
    if digits_of(a) != digits_of(b):
        return "different_numbers"            # 401 error vs 403 error
    ta = ents[a]["types"].most_common(1)[0][0]
    tb = ents[b]["types"].most_common(1)[0][0]
    if ta == "PRODUCT" and tb == "PRODUCT" and tokens(a) != tokens(b):
        return "distinct_products"            # Wix Editor vs Studio Editor
    return None


STOP = {"and", "of", "for", "to", "in", "on", "with", "or", "by", "from", "at", "as", "wix"}


def tokens(name):
    return {w for w in re.findall(r"[a-z0-9]+", rule_normalize(name)) if w not in STOP}


class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, a):
        self.parent.setdefault(a, a)
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]
            a = self.parent[a]
        return a

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


# ============================================================
# 4. COLLECT ENTITIES
# ============================================================

def collect_entities(records):
    """
    Returns dict key -> {names: Counter(surface), types: Counter,
    docs: set, aliases: Counter}, keyed by rule_normalize(name).
    Model-provided aliases are recorded for candidate generation.
    """
    ents = {}
    for r in records:
        for e in r["entities"]:
            key = entity_key(e["name"], e["type"])
            if not key:
                continue
            item = ents.setdefault(key, {"names": Counter(), "types": Counter(),
                                         "docs": set(), "aliases": Counter()})
            item["names"][e["name"]] += 1
            item["types"][e["type"]] += 1
            item["docs"].add(r["document_id"])
            for a in e.get("aliases", []):
                ak = rule_normalize(a)
                if ak and ak != key:
                    item["aliases"][ak] += 1
    return ents


# ============================================================
# 5. CANDIDATE PAIRS
# ============================================================

def lexical_candidates(ents):
    keys = list(ents)
    tok = {k: tokens(k) for k in keys}
    index = defaultdict(list)
    for k, t in tok.items():
        for w in t:
            index[w].append(k)
    pairs = {}
    for w, members in index.items():
        if len(members) > 60:
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = sorted((members[i], members[j]))
                if (a, b) in pairs:
                    continue
                ta, tb = tok[a], tok[b]
                if not ta or not tb:
                    continue
                jac = len(ta & tb) / len(ta | tb)
                if jac >= LEXICAL_JACCARD:
                    pairs[(a, b)] = ("lexical", round(jac, 3))
    # model aliases: alias string equal to another entity's key
    for k, item in ents.items():
        for ak in item["aliases"]:
            if ak in ents:
                a, b = sorted((k, ak))
                pairs.setdefault((a, b), ("alias", 1.0))
    return pairs


def embed_texts(client, texts):
    cache = {}
    for row in read_jsonl(EMBED_CACHE_PATH):
        if row.get("model") == EMBED_MODEL:
            cache[row["text"]] = row["vector"]
    missing = [t for t in texts if t not in cache]
    for start in range(0, len(missing), 256):
        batch = missing[start:start + 256]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        for text, item in zip(batch, response.data):
            cache[text] = item.embedding
            append_jsonl(EMBED_CACHE_PATH, {"model": EMBED_MODEL, "text": text,
                                            "vector": item.embedding})
    return [cache[t] for t in texts]


def embedding_candidates(client, ents):
    import numpy as np
    keys = list(ents)
    texts = [f"{ents[k]['names'].most_common(1)[0][0]} ({ents[k]['types'].most_common(1)[0][0]})"
             for k in keys]
    matrix = np.array(embed_texts(client, texts), dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9
    pairs = {}
    for start in range(0, len(keys), 512):
        block = matrix[start:start + 512] @ matrix.T
        rows, cols = np.where(block >= EMBED_THRESHOLD)
        for r, c in zip(rows, cols):
            i, j = start + int(r), int(c)
            if i < j:
                pairs[(keys[i], keys[j])] = ("embedding", round(float(block[r, c]), 3))
    return pairs


def build_components(ents, pairs):
    """
    Size-capped agglomeration of candidate pairs.

    Pairs are processed strongest first (model aliases = 1.0, then
    embedding cosine / lexical Jaccard); two groups are joined only if
    the result stays within MAX_COMPONENT. A giant component can never
    form, no name is dropped, and groups stay semantically local, so
    the LLM always sees a small set of genuinely similar names.
    """
    ranked = sorted(
        ((score, kind, a, b) for (a, b), (kind, score) in pairs.items()
         if a not in GENERIC_NEVER_MERGE and b not in GENERIC_NEVER_MERGE),
        key=lambda t: (-(t[0] + (1.0 if t[1] == "alias" else 0.0)), t[2], t[3]),
    )
    uf = UnionFind()
    size = {}
    for score, kind, a, b in ranked:
        ra, rb = uf.find(a), uf.find(b)
        if ra == rb:
            continue
        sa, sb = size.get(ra, 1), size.get(rb, 1)
        if sa + sb > MAX_COMPONENT:
            continue
        uf.union(ra, rb)
        size[uf.find(ra)] = sa + sb
    comps = defaultdict(list)
    for k in ents:
        if k in uf.parent:
            comps[uf.find(k)].append(k)
    out = [sorted(c) for c in comps.values() if len(c) > 1]
    return sorted(out, key=lambda c: (-len(c), c[0]))


# ============================================================
# 6. LLM MERGE DECISIONS
# ============================================================

def describe_group(gid, component, ents):
    lines = [f"Group {gid}:"]
    for k in component:
        item = ents[k]
        surface = item["names"].most_common(1)[0][0]
        typ = item["types"].most_common(1)[0][0]
        lines.append(f'  - "{surface}" [{typ}, {len(item["docs"])} articles]')
    return "\n".join(lines)


def decide_batch(client, batch, ents):
    user = "\n\n".join(describe_group(gid, comp, ents) for gid, comp in batch)
    request = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": MERGE_PROMPT},
                     {"role": "user", "content": user}],
        "response_format": MergeDecision,
    }
    if IS_REASONING_MODEL:
        request["reasoning_effort"] = REASONING_EFFORT
        request["max_completion_tokens"] = 8000
    response = client.chat.completions.parse(**request)
    choice = response.choices[0]
    if choice.message.parsed is None or choice.finish_reason == "length":
        raise RuntimeError("No structured merge decision returned.")
    return choice.message.parsed, response.usage


def apply_decisions(decisions, ents, components):
    """Validate LLM groups against their candidate component; union."""
    comp_by_id = {gid: set(comp) for gid, comp in components}
    surface_to_key = {}
    for k, item in ents.items():
        for name in item["names"]:
            surface_to_key[name.casefold()] = k
        surface_to_key[k] = k
    uf = UnionFind()
    canonical = {}
    dropped = Counter()
    claimed = set()          # a name may belong to one merge set only
    for dec in decisions:
        for g in dec["groups"]:
            allowed = comp_by_id.get(g["group_id"])
            if allowed is None:
                dropped["unknown_group"] += 1
                continue
            keys = []
            for m in g["members"]:
                k = (surface_to_key.get(m.casefold())
                     or surface_to_key.get(rule_normalize(m))
                     or surface_to_key.get(rule_normalize(m, keep_plural=True)))
                if k not in allowed:
                    dropped["member_outside_group"] += 1
                elif k in claimed:
                    dropped["member_in_multiple_groups"] += 1
                else:
                    keys.append(k)
            keys = list(dict.fromkeys(keys))
            # deterministic guards: drop members that conflict with the
            # first (anchor) member of the set
            safe = []
            for k in keys:
                reason = next((safe_to_merge(k, s, ents) for s in safe
                               if safe_to_merge(k, s, ents)), None)
                if reason:
                    dropped[reason] += 1
                else:
                    safe.append(k)
            keys = safe
            if len(keys) < 2:
                dropped["group_too_small"] += 1
                continue
            claimed.update(keys)
            for k in keys[1:]:
                uf.union(k, keys[0])
            root = uf.find(keys[0])
            canonical[root] = (g["canonical_name"].strip(), g["type"])
    mapping = {}
    for k in ents:
        root = uf.find(k) if k in uf.parent else k
        if root in canonical and root != k or (root == k and k in canonical):
            name, typ = canonical.get(root, (None, None))
        else:
            name, typ = None, None
        if name is None:
            name = ents[root]["names"].most_common(1)[0][0]
            typ = ents[root]["types"].most_common(1)[0][0]
        mapping[k] = {"canonical": name, "type": typ,
                      "id": "ent_" + hashlib.sha1(name.casefold().encode()).hexdigest()[:12]}
    return mapping, dropped


# ============================================================
# 7. REPORT
# ============================================================

def connectivity(records, key_fn):
    ent_docs = defaultdict(set)
    for r in records:
        for e in r["entities"]:
            ent_docs[key_fn(e["name"])].add(r["document_id"])
    docs = {r["document_id"] for r in records}
    n = len(ent_docs)
    single = sum(len(d) == 1 for d in ent_docs.values())
    cut = max(3, int(0.25 * len(docs)))
    uf = UnionFind()
    for d in docs:
        uf.find(d)
    for ds in ent_docs.values():
        ds = sorted(ds)
        if 1 < len(ds) <= cut:
            for d in ds[1:]:
                uf.union(ds[0], d)
    sizes = sorted(Counter(uf.find(d) for d in docs).values(), reverse=True)
    return {"entities": n, "singleton_share": round(single / max(1, n), 3),
            "doc_components": len(sizes), "largest": sizes[0] if sizes else 0,
            "isolated_docs": sum(s == 1 for s in sizes)}


# ============================================================
# 8. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Build candidates and report; no LLM merge calls")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="Lexical + alias candidates only")
    args = parser.parse_args()

    records = read_jsonl(INPUT_PATH)
    if not records:
        parser.error(f"No extraction records at {INPUT_PATH}")
    ents = collect_entities(records)
    print("========== ENTITY CANONICALIZATION ==========")
    raw_names = {e["name"].casefold() for r in records for e in r["entities"]}
    print(f"documents {len({r['document_id'] for r in records})} | raw names {len(raw_names)} "
          f"| after rule normalization {len(ents)}")

    client = None
    if not args.no_embeddings or not args.dry_run:
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("Missing OPENAI_API_KEY in .env")
        client = OpenAI(timeout=180.0, max_retries=3)

    pairs = lexical_candidates(ents)
    print(f"lexical/alias candidate pairs: {len(pairs)}")
    if not args.no_embeddings:
        emb = embedding_candidates(client, ents)
        for p, s in emb.items():
            pairs.setdefault(p, s)
        print(f"embedding candidate pairs (cos>={EMBED_THRESHOLD}): {len(emb)} | total {len(pairs)}")

    components = list(enumerate(build_components(ents, pairs), start=1))
    involved = sum(len(c) for _, c in components)
    print(f"candidate components: {len(components)} covering {involved} names "
          f"(largest {max((len(c) for _, c in components), default=0)})")
    for gid, comp in components[:8]:
        print("   e.g.", [ents[k]["names"].most_common(1)[0][0] for k in comp][:8])

    if args.dry_run:
        print("Dry run. No merge decisions requested.")
        return

    # ---- LLM decisions, checkpointed per batch ----
    done = {}
    for row in read_jsonl(DECISION_CHECKPOINT):
        if row.get("schema_version") == SCHEMA_VERSION and row.get("model") == MODEL_NAME:
            done[row["batch_key"]] = row
    decisions = []
    usage = Counter()
    batches = [components[i:i + LLM_BATCH_GROUPS]
               for i in range(0, len(components), LLM_BATCH_GROUPS)]
    for bi, batch in enumerate(batches, start=1):
        key = hashlib.sha1(json.dumps(batch, sort_keys=True).encode()).hexdigest()
        if key in done:
            decisions.append(done[key]["decision"])
            continue
        parsed, u = decide_batch(client, batch, ents)
        row = {"schema_version": SCHEMA_VERSION, "model": MODEL_NAME, "batch_key": key,
               "decision": parsed.model_dump(),
               "input_tokens": u.prompt_tokens if u else 0,
               "output_tokens": u.completion_tokens if u else 0}
        append_jsonl(DECISION_CHECKPOINT, row)
        decisions.append(row["decision"])
        usage["input"] += row["input_tokens"]
        usage["output"] += row["output_tokens"]
        print(f"[{bi}/{len(batches)}] merge sets returned: {len(parsed.groups)}")

    mapping, dropped = apply_decisions(decisions, ents, components)
    merged_into = Counter(v["canonical"].casefold() for v in mapping.values())
    n_merged = sum(c for c in merged_into.values() if c > 1)
    print(f"\nmerge sets applied; names merged into another name: "
          f"{n_merged - sum(1 for c in merged_into.values() if c > 1)} | dropped {dict(dropped)}")

    # ---- write mapping and canonical records ----
    raw_to_canon = {}
    for r in records:
        for e in r["entities"]:
            raw_to_canon[e["name"].casefold()] = mapping[entity_key(e["name"], e["type"])]
    write_json(CANON_PATH, {
        "schema_version": SCHEMA_VERSION, "model": MODEL_NAME, "embed_model": EMBED_MODEL,
        "thresholds": {"embedding": EMBED_THRESHOLD, "lexical_jaccard": LEXICAL_JACCARD},
        "raw_to_canonical": raw_to_canon,
    })
    canon_name = lambda raw: raw_to_canon[raw.casefold()]["canonical"]
    canonical_records = []
    for r in records:
        rec = dict(r)
        seen = {}
        for e in r["entities"]:
            c = raw_to_canon[e["name"].casefold()]
            item = seen.setdefault(c["id"], {"name": c["canonical"], "type": c["type"],
                                             "id": c["id"], "surface_forms": set()})
            item["surface_forms"].add(e["name"])
        rec["entities"] = [{**v, "surface_forms": sorted(v["surface_forms"])}
                           for v in seen.values()]
        rels = []
        for x in r["relations"]:
            y = dict(x)
            y["subject_raw"], y["object_raw"] = x["subject"], x["object"]
            y["subject"], y["object"] = canon_name(x["subject"]), canon_name(x["object"])
            y["subject_id"] = raw_to_canon[x["subject"].casefold()]["id"]
            y["object_id"] = raw_to_canon[x["object"].casefold()]["id"]
            if y["subject_id"] != y["object_id"]:
                rels.append(y)
        rec["relations"] = rels
        rec.pop("raw_outputs", None)
        canonical_records.append(rec)
    CANON_RECORDS_PATH.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in canonical_records),
        encoding="utf-8")

    before = connectivity(records, lambda n: n.casefold())
    after = connectivity(records, canon_name)
    print("\n--- before -> after ---")
    for k in before:
        print(f"{k:16s} {before[k]!s:>8}  ->  {after[k]!s:>8}")
    print(f"LLM tokens this run: {dict(usage)}")
    print(f"\nSaved: {CANON_PATH}\nSaved: {CANON_RECORDS_PATH}")


if __name__ == "__main__":
    main()
