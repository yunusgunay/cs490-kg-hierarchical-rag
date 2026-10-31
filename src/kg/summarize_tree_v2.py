# src/kg/summarize_tree_v2.py
"""
Routing-oriented summaries for the document tree (step 6a).

Differences from summarize_tree.py (v1)
---------------------------------------
* Nodes are document groups (build_tree_v2 output), not passages.
* No per-document LLM cards: a document row is title + first sentence
  + its most frequent canonical KG entities (zero LLM cost; WixQA
  titles are already routing-grade, and Corpus2Skill's ablation found
  LLM doc cards F1-neutral).
* Summary schema is built for a navigator, not for reading:
  title / summary / question_types / key_terms. Leaves are summarized
  from their documents, internal nodes from their children's summaries
  (bottom-up). Every level is processed concurrently.
* Per-node checkpoint keyed by (tree hash, prompt, schema, model).

Usage
-----
    python src/kg/summarize_tree_v2.py --dry-run
    python src/kg/summarize_tree_v2.py                      # leiden tree
    python src/kg/summarize_tree_v2.py --tree tfidf-kmeans  # ablation tree

Output: graph/trees/summaries/<tree>_summaries.json  (+ .jsonl checkpoint)
"""
import argparse
import hashlib
import json
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data/processed/subset"
GRAPH_DIR = DATA_DIR / "graph"
TREE_DIR = GRAPH_DIR / "trees"
SUMMARY_DIR = TREE_DIR / "summaries"
CORPUS_PATH = DATA_DIR / "corpus.jsonl"

load_dotenv(PROJECT_ROOT / ".env")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-6-luna").strip()
REASONING_EFFORT = os.getenv("LLM_SUMMARY_REASONING_EFFORT",
                             os.getenv("LLM_REASONING_EFFORT", "medium")).strip().lower()
IS_REASONING_MODEL = MODEL_NAME.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
MODEL_RUN = "gpt-6-luna_medium"
CANON_RECORDS_PATH = DATA_DIR / f"kg_v5_{MODEL_RUN}_canonical.jsonl"

SCHEMA_VERSION = "tree-summary-v2"
DOC_EXCERPT_CHARS = 600        # per document shown to the leaf summarizer
MAX_DOCS_SHOWN = 20        # >= largest leaf, so every article is shown with an excerpt
CHILD_SUMMARY_CHARS = 700      # pilot summaries were 460-600 chars; 500 cut the 3rd sentence
DOC_KEY_ENTITIES = 5
MAX_OUTPUT_TOKENS = 3000
DEFAULT_WORKERS = 6


# ------------------------------------------------------------
# schema + prompt
# ------------------------------------------------------------

class NodeSummary(BaseModel):
    title: str
    summary: str
    question_types: list[str]
    key_terms: list[str]


SUMMARY_PROMPT = """
You write navigation summaries for a tree of Wix help-center articles.
An agent will read your summary to decide whether to open this branch
for a user's question, so the summary must help ROUTING, not teach.

Return:
- title: 3-10 words naming the SPECIFIC scope of this branch. Several
  neighbouring branches may concern the same product, so the title must
  carry what is particular here (the providers, settings, tasks or
  problems covered), not only the product and a generic aspect.
  Good: "Connecting regional card providers (UAPay, Payzone, FasterPay)",
  "Switching, disconnecting and currency of payment providers".
  Weak: "Wix payment providers and settings".
- summary: 2-3 sentences. Sentence 1: what the articles in this branch
  cover (name the products, features, settings). Sentence 2: what
  distinguishes this branch from neighbouring topics (what it does
  NOT cover if that is a likely confusion). Sentence 3 (optional):
  notable prerequisites, limitations or troubleshooting content.
- question_types: 3-6 short examples of user questions this branch
  answers, phrased the way users ask ("my site isn't showing on
  Google", "how do I charge different shipping per region").
- key_terms: 5-15 terms a user might type: product names, feature
  names, settings, error messages, synonyms. Lowercase except proper
  names. No duplicates, no generic words (site, page, click).

Rules: use only the material given; do not invent features; be
specific and concrete; no marketing language.
"""


# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------

def read_jsonl(path):
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


_LOCK = threading.Lock()


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def sha256_file(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def first_sentence(text, title):
    body = text[len(title):].lstrip(" \n:-–") if text.startswith(title) else text
    body = re.sub(r"\s+", " ", body).strip()
    m = re.match(r"(.{20,220}?[.!?])(\s|$)", body)
    return (m.group(1) if m else body[:200]).strip()


# ------------------------------------------------------------
# document rows (no LLM)
# ------------------------------------------------------------

def build_doc_rows(docs, records):
    ent_by_doc = defaultdict(Counter)
    for r in records:
        for e in r["entities"]:
            ent_by_doc[r["document_id"]][e["name"]] += 1
        for x in r["relations"]:
            ent_by_doc[r["document_id"]][x["subject"]] += 1
            ent_by_doc[r["document_id"]][x["object"]] += 1
    rows = {}
    for d in docs:
        rows[d["id"]] = {
            "doc_id": d["id"],
            "title": d["title"],
            "one_line": first_sentence(d["contents"], d["title"]),
            "key_entities": [n for n, _ in ent_by_doc[d["id"]].most_common(DOC_KEY_ENTITIES)],
            "article_type": d["article_type"],
            "n_chars": len(d["contents"]),
        }
    return rows


# ------------------------------------------------------------
# LLM
# ------------------------------------------------------------

def call_llm(client, material):
    request = {
        "model": MODEL_NAME,
        "messages": [{"role": "system", "content": SUMMARY_PROMPT},
                     {"role": "user", "content": material}],
        "response_format": NodeSummary,
    }
    if IS_REASONING_MODEL:
        request["reasoning_effort"] = REASONING_EFFORT
        request["max_completion_tokens"] = MAX_OUTPUT_TOKENS
    response = client.chat.completions.parse(**request)
    choice = response.choices[0]
    if choice.finish_reason == "length" or choice.message.parsed is None:
        raise RuntimeError("No structured summary returned.")
    p = choice.message.parsed
    if not p.title.strip() or not p.summary.strip():
        raise RuntimeError("Empty title or summary.")
    u = response.usage
    details = getattr(u, "completion_tokens_details", None)
    return {
        "title": p.title.strip(),
        "summary": p.summary.strip(),
        "question_types": [q.strip() for q in p.question_types if q.strip()][:6],
        "key_terms": list(dict.fromkeys(t.strip() for t in p.key_terms if t.strip()))[:15],
        "input_tokens": u.prompt_tokens if u else 0,
        "output_tokens": u.completion_tokens if u else 0,
        "reasoning_tokens": (getattr(details, "reasoning_tokens", 0) or 0) if details else 0,
    }


def leaf_material(node, doc_rows, by_id):
    lines = [f"LEAF NODE with {node['size']} articles. Articles:"]
    for i, d in enumerate(node["doc_ids"][:MAX_DOCS_SHOWN], 1):
        row = doc_rows[d]
        excerpt = re.sub(r"\s+", " ", by_id[d]["contents"][:DOC_EXCERPT_CHARS])
        lines.append(f"\n--- Article {i}: {row['title']}\nkey entities: {', '.join(row['key_entities'])}\n{excerpt}")
    if node["size"] > MAX_DOCS_SHOWN:
        lines.append(f"\n(+{node['size'] - MAX_DOCS_SHOWN} more articles: "
                     + "; ".join(doc_rows[d]["title"] for d in node["doc_ids"][MAX_DOCS_SHOWN:]) + ")")
    return "\n".join(lines)


def parent_material(node, summaries, tree_nodes):
    lines = [f"INTERNAL NODE with {len(node['children'])} sub-branches "
             f"({node['size']} articles in total). Sub-branches:"]
    for i, cid in enumerate(node["children"], 1):
        s = summaries[cid]
        lines.append(f"\n--- Branch {i} ({tree_nodes[cid]['size']} articles): {s['title']}\n"
                     f"{s['summary'][:CHILD_SUMMARY_CHARS]}\n"
                     f"question types: {'; '.join(s['question_types'][:4])}\n"
                     f"key terms: {', '.join(s['key_terms'][:10])}")
    return "\n".join(lines)


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tree", choices=["leiden", "tfidf-kmeans"], default="leiden")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--limit", type=int, default=None, help="Max new LLM calls this run")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    tree_path = TREE_DIR / f"{args.tree}_tree.json"
    tree = read_json(tree_path)
    tree_nodes = {n["id"]: n for n in tree["nodes"]}
    docs = read_jsonl(CORPUS_PATH)
    by_id = {d["id"]: d for d in docs}
    records = read_jsonl(CANON_RECORDS_PATH)
    doc_rows = build_doc_rows(docs, records)

    fp = hashlib.sha256(json.dumps({
        "schema": SCHEMA_VERSION, "tree_hash": sha256_file(tree_path),
        "corpus_hash": sha256_file(CORPUS_PATH), "canonical_kg_hash": sha256_file(CANON_RECORDS_PATH),
        "model": MODEL_NAME, "reasoning": REASONING_EFFORT if IS_REASONING_MODEL else None,
        "prompt": SUMMARY_PROMPT, "output_schema": NodeSummary.model_json_schema(),
        "doc_excerpt": DOC_EXCERPT_CHARS, "max_docs_shown": MAX_DOCS_SHOWN,
        "child_summary_chars": CHILD_SUMMARY_CHARS, "doc_key_entities": DOC_KEY_ENTITIES,
    }, sort_keys=True).encode()).hexdigest()

    ckpt = SUMMARY_DIR / f"{args.tree}_summaries.jsonl"
    out = SUMMARY_DIR / f"{args.tree}_summaries.json"
    summaries = {}
    for row in read_jsonl(ckpt):
        if row.get("run_fingerprint") == fp and row["node_id"] in tree_nodes \
                and row["doc_ids"] == tree_nodes[row["node_id"]]["doc_ids"]:
            summaries[row["node_id"]] = row
    pending_total = len(tree_nodes) - len(summaries)

    print(f"========== TREE SUMMARIES ({args.tree}) ==========")
    print(f"tree nodes {len(tree_nodes)} | leaves {sum(n['is_leaf'] for n in tree_nodes.values())} "
          f"| already summarized {len(summaries)} | pending {pending_total}")
    print(f"model {MODEL_NAME} (reasoning {REASONING_EFFORT}) | checkpoint {ckpt}")
    if args.dry_run:
        sample = next(n for n in tree_nodes.values() if n["is_leaf"])
        print("\n--- example leaf material (truncated) ---")
        print(leaf_material(sample, doc_rows, by_id)[:1500])
        print("\nDry run. No API calls made.")
        return
    if pending_total and not os.getenv("OPENAI_API_KEY"):
        parser.error("Missing OPENAI_API_KEY in .env")
    client = OpenAI(timeout=180.0, max_retries=3) if pending_total else None

    # bottom-up by depth; each depth level runs concurrently
    calls = 0
    for depth in sorted({n["depth"] for n in tree_nodes.values()}, reverse=True):
        level = [n for n in tree_nodes.values() if n["depth"] == depth and n["id"] not in summaries]
        if args.limit is not None:
            level = level[: max(0, args.limit - calls)]
        if not level:
            continue
        missing_children = [n["id"] for n in level if not all(c in summaries for c in n["children"])]
        if missing_children:
            print(f"depth {depth}: {len(missing_children)} nodes wait for children (rerun to continue)")
            level = [n for n in level if n["id"] not in missing_children]

        def work(node):
            material = (leaf_material(node, doc_rows, by_id) if node["is_leaf"]
                        else parent_material(node, summaries, tree_nodes))
            result = call_llm(client, material)
            return node, result

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(work, n) for n in level]
            for fut in as_completed(futures):
                try:
                    node, result = fut.result()
                except Exception as error:
                    print(f"  FAILED: {type(error).__name__}: {error}")
                    continue
                record = {
                    "run_fingerprint": fp, "schema_version": SCHEMA_VERSION, "tree": args.tree,
                    "node_id": node["id"], "parent_id": node["parent_id"], "depth": node["depth"],
                    "is_leaf": node["is_leaf"], "size": node["size"], "doc_ids": node["doc_ids"],
                    "child_ids": node["children"], "model": MODEL_NAME, **result,
                }
                append_jsonl(ckpt, record)
                summaries[node["id"]] = record
                calls += 1
                print(f"  [{len(summaries)}/{len(tree_nodes)}] depth={node['depth']} size={node['size']:3d} "
                      f"{'leaf' if node['is_leaf'] else 'node'} | {result['title'][:70]}")

    # export (summaries + doc rows in one file for the navigator)
    write_json(out, {
        "schema_version": SCHEMA_VERSION, "tree": args.tree, "source_tree": tree_path.name,
        "root_id": tree["root_id"], "model": MODEL_NAME,
        "summaries": {k: {kk: vv for kk, vv in v.items() if kk != "run_fingerprint"}
                      for k, v in sorted(summaries.items())},
        "doc_rows": doc_rows,
    })
    tokens = {k: sum(v.get(k, 0) for v in summaries.values())
              for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
    print(f"\nsummarized {len(summaries)}/{len(tree_nodes)} nodes | LLM calls this run {calls} | tokens {tokens}")
    print(f"Saved: {out}")
    if len(summaries) < len(tree_nodes):
        print("Tree partially summarized. Rerun to continue from the checkpoint.")


if __name__ == "__main__":
    main()
