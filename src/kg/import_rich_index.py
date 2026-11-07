# src/kg/import_rich_index.py
"""
Register the mentor's document-grounded summary index (latest_beam_research/data/summaries/*.json)
as a second index on the SAME Leiden tree, under a new tree name (default: leiden-rich).

Why a new tree name and not a flag: every retrieval script (baselines.py, navigate_tree.py,
navigate_tree_beam.py, navigate_beam.py, evaluate_v2.py) already takes --tree and names its
result files <system>__<tree>__<split>.json, so "same partition, other summaries" becomes an
ablation axis exactly like tfidf-kmeans, with no code change beyond adding the name to TREES in
src/retrieval/common.py, and no result file is overwritten.

What is written (nothing from the package is executed, files are only read):
  graph/trees/<name>_tree.json                      copy of leiden_tree.json (identical partition)
  graph/trees/summaries/<name>_summaries.json       the new node summaries (+ coverage statements)
                                                    --doc-rows rich (default): the package's article rows
                                                    (one_line = 100-160-word LLM summary + qualifiers, key_entities
                                                    = LLM key terms); the simple beam's selector and our navigators'
                                                    leaf prompts read these rows
                                                    --doc-rows original: our article rows (title, first sentence,
                                                    top KG entities) -> only the node routing summaries change
  graph/trees/shortcuts/<name>_shortcuts_<policy>.json   typed: the package's file (same links as ours; exact
                                                    replay of its typed runs needs its support payload);
                                                    other policies: copies of our leiden files

Usage
-----
    python src/kg/import_rich_index.py --package <path>/latest_beam_research                       # leiden-rich (rich doc rows)
    python src/kg/import_rich_index.py --package <path>/latest_beam_research --doc-rows original --name leiden-richnodes
Then add the name to TREES in src/retrieval/common.py and run, e.g.
    python src/retrieval/baselines.py --tree leiden-rich --systems tree_bm25,raptor_traversal,raptor_collapsed --beams 1,2,3,4,6,8
    python src/retrieval/navigate_tree_beam.py --tree leiden-rich --router cosine --widths 1,2,3 --policy none,typed
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.common import SHORTCUT_DIR, SUMMARY_DIR, TREE_DIR, TREES, load_tree  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--package", required=True, type=Path, help="extracted latest_beam_research directory")
    parser.add_argument("--name", default="leiden-rich", help="new tree name (used in --tree and in result file names)")
    parser.add_argument("--doc-rows", choices=["rich", "original"], default="rich",
                        help="rich = the package's article rows (exact replay of its runs needs this); "
                             "original = our article rows, i.e. only the node summaries change")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    package = args.package.resolve()
    if args.name in ("leiden", "tfidf-kmeans"):
        sys.exit("choose a new name; the existing trees must not be overwritten")

    ours = load_tree("leiden")
    ptree = json.load((package / "data/graph/leiden_tree.json").open(encoding="utf-8"))
    if ptree["root_id"] != ours["root_id"] or {n["id"]: tuple(n["doc_ids"]) for n in ptree["nodes"]} != \
            {n: tuple(v["doc_ids"]) for n, v in ours["by_id"].items()}:
        sys.exit("the package tree is not our leiden_tree.json (different nodes or document partition)")

    source = package / "data/summaries" / ("rich_cards.json" if args.doc_rows == "rich" else "leaf_only.json")
    index = json.load(source.open(encoding="utf-8"))
    missing = set(ours["by_id"]) - set(index["summaries"])
    if missing:
        sys.exit(f"{len(missing)} tree nodes have no summary in {source.name}")
    for n, s in index["summaries"].items():
        if s["size"] != len(ours["by_id"][n]["doc_ids"]):
            sys.exit(f"summary size disagrees with the tree at node {n}")
    if set(index["doc_rows"]) != set(ours["by_id"][ours["root_id"]]["doc_ids"]):
        sys.exit("document rows do not cover exactly our corpus")

    targets = [TREE_DIR / f"{args.name}_tree.json", SUMMARY_DIR / f"{args.name}_summaries.json"]
    policies = sorted(p.name.split("_shortcuts_")[1][:-5] for p in SHORTCUT_DIR.glob("leiden_shortcuts_*.json")
                      if not p.name.endswith("_stats.json"))
    targets += [SHORTCUT_DIR / f"{args.name}_shortcuts_{p}.json" for p in policies]
    existing = [t for t in targets if t.exists()]
    if existing and not args.force:
        sys.exit("already present (use --force to overwrite): " + ", ".join(t.name for t in existing))

    shutil.copyfile(TREE_DIR / "leiden_tree.json", targets[0])
    out = {"schema_version": index["schema_version"], "tree": args.name, "source_tree": "leiden_tree.json",
           "root_id": index["root_id"], "model": index.get("model"),
           "source": f"{source.name} from latest_beam_research_2026-11-03 (document-grounded summaries, same Leiden tree)",
           "source_manifest": index.get("source_manifest"), "doc_rows_variant": args.doc_rows,
           "summaries": index["summaries"], "doc_rows": index["doc_rows"]}
    targets[1].write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    note = ""
    for p in policies:
        src_file = SHORTCUT_DIR / f"leiden_shortcuts_{p}.json"
        if p == "typed":
            # the package's typed file: same 429 links in the same order as ours, but equal-value support pairs
            # can be tie-broken differently by select_shortcuts.py; the routing label text (and therefore the
            # keyless replay of the package's typed runs) depends on the exact support payload, so take his.
            his = json.load((package / "data/graph/typed_shortcuts.json").open(encoding="utf-8"))
            ours = json.load(src_file.open(encoding="utf-8"))
            if {k: [l["target"] for l in v] for k, v in his["shortcuts"].items()} != \
                    {k: [l["target"] for l in v] for k, v in ours["shortcuts"].items()}:
                sys.exit("the package's typed shortcuts link different nodes than ours; refusing to import")
            if his["shortcuts"] != ours["shortcuts"]:
                note = " (typed file taken from the package: same links, different tie-order of support pairs)"
            shutil.copyfile(package / "data/graph/typed_shortcuts.json", SHORTCUT_DIR / f"{args.name}_shortcuts_{p}.json")
        else:
            shutil.copyfile(src_file, SHORTCUT_DIR / f"{args.name}_shortcuts_{p}.json")
    words = sum(len(s["summary"].split()) for s in index["summaries"].values()) / len(index["summaries"])
    print(f"wrote {len(targets)} files for tree '{args.name}': {len(index['summaries'])} node summaries "
          f"(mean {words:.0f} words, coverage statements included), doc rows = {args.doc_rows}, shortcut policies {policies}{note}")
    if args.name not in TREES:
        print(f"NEXT: add \"{args.name}\" to TREES in src/retrieval/common.py, e.g. TREES = (\"leiden\", \"tfidf-kmeans\", \"{args.name}\")")


if __name__ == "__main__":
    main()