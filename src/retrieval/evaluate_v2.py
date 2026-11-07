# src/retrieval/evaluate_v2.py
"""
Compare every results file in data/processed/subset/eval/results/ under the
same protocol (step 7c) - evaluate.py plus the beam-navigator analyses.

Metrics at an article budget B (default 5; also reported at 10 for ranked systems):
  all-gold@B   fraction of questions whose EVERY gold article is among the first
               B opened/ranked articles  (= complete evidence-chain recall)
  R@B          mean per-question recall of gold articles within B
  MRR@B        reciprocal rank of the first gold article
Cost columns: articles opened, node summaries read, summary tokens, LLM calls,
LLM tokens (prompt / completion incl. reasoning) per question.
Splits: all / single-gold / multi-gold questions.
Significance: paired bootstrap (2000 resamples) of all-gold@B against a
reference system (--ref, default bm25) and, for navigator shortcut policies,
against the tree-only navigator of the same variant and beam width.

New in v2 (for navigate_tree_v2.py results, systems beam[noev]_W<W>_<policy>_B<B>):
  * beam-width sweep: all-gold@B (all / single / multi) and cost per width, next
    to the cosine beam of the same width (raptor_traversal_bW) and the flat
    baselines; figure eval/beam_width_<tree>_<split>.png
  * shortcut ablation grouped by (variant, width): typed/random... vs none
    with how often a shortcut target entered the beam or was followed and
    whether a gold article was opened through it
  * beam...fill_* rows (--fill): how often the budget fill opened articles and
    whether a gold article came in through it
  * beamoracle_* rows (gold-aware scorer) are reported as structural ceilings
  * routing diagnostics and the budget-scaling curve cover beam traces
Earlier navigator results (nav_v4_...) are read and compared as before.

Usage
-----
    python src/retrieval/evaluate_v2.py                     # all systems, B=5, leiden tree
    python src/retrieval/evaluate_v2.py --ref dense
    python src/retrieval/evaluate_v2.py --strict --ref dense   # final table
    python src/retrieval/evaluate_v2.py --tree tfidf-kmeans
    python src/retrieval/evaluate_v2.py --ref dense --strict --only "bm25,dense,raptor_*,nav_v4_*,nav_v4b_*"
    python src/retrieval/evaluate_v2.py --ref dense --exclude "beamoracle*,nav_v3*"
--only / --exclude take comma-separated glob patterns on system names (the results files stay; only
the comparison changes). Navigator results are only included when they were run with the same --budget.
Outputs: eval/summary_v2_<tree>_<split>_B<budget>.md and .json
"""
import argparse
import fnmatch
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.common import (EVAL_DIR, RESULTS_DIR, TREES, load_questions, load_tree, opened_at,  # noqa: E402
                              paired_bootstrap, question_metrics, write_json)

ORDER = ["bm25", "dense", "raptor_collapsed", "raptor_traversal_b1", "raptor_traversal_b2", "raptor_traversal_b3"]
BEAM_RE = re.compile(r"^(.+?)_W(\d+)_([a-z_]+)_B(\d+)$")     # beam_W2_typed_B5, nav_v4b_W2_typed_B5, beamoracle_...


def is_nav(name):
    return name.startswith(("nav_", "beam", "simplebeam"))


def beam_parts(name):
    """(family, width, policy) of a beam system name, else None."""
    m = BEAM_RE.match(name)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else None


def system_rank(name):
    if name in ORDER:
        return (0, ORDER.index(name), 0, name)
    bp = beam_parts(name)
    if name.startswith("nav_") and not bp:
        return (1, 0 if "_none_" in name else 1, 0, name)         # single-path navigators, tree-only first
    if bp:
        family, width, policy = bp
        group = 2 if family.startswith("nav_") else 4 if family.startswith("beamoracle") else 3
        return (group, family, width, 0 if policy == "none" else 1, name)
    return (5, 0, 0, name)


def nav_key(results, s):
    """(variant, policy, width) of a navigator result, from its config."""
    cfg = results[s].get("config", {})
    return cfg.get("variant", "v2"), cfg.get("policy"), cfg.get("width")


def load_results(tree, split, budget):
    out, skipped = {}, []
    for path in sorted(RESULTS_DIR.glob("*.json")):
        data = json.load(path.open(encoding="utf-8"))
        if data["tree"] not in ("none", tree) or data.get("split", "dev") != split:
            continue
        nav_budget = data.get("config", {}).get("budget")
        if nav_budget is not None and nav_budget != budget:      # navigator run with another budget
            skipped.append(f"{data['system']} (budget {nav_budget})")
            continue
        out[data["system"]] = data
    if skipped:
        print(f"note: skipped navigator results run with a different budget than B={budget}: {skipped}")
    return out


def aggregate(records, budget):
    n = len(records)
    if n == 0:
        return {}
    m = [question_metrics(r, budget) for r in records]
    agg = {
        "n": n,
        "all_gold": sum(x["all_gold"] for x in m) / n,
        "recall": sum(x["recall"] for x in m) / n,
        "mrr": sum(x["mrr"] for x in m) / n,
        "opened": sum(x["n_opened"] for x in m) / n,
    }
    for key, out in (("summaries_read", "summaries"), ("summary_tokens_read", "summary_tokens"),
                     ("llm_calls", "calls"), ("input_tokens", "in_tokens"), ("output_tokens", "out_tokens")):
        vals = []
        for r in records:
            if r.get("slots"):                      # RAPTOR collapsed: count within the budget's slots
                if key == "summaries_read":
                    vals.append(sum(s["type"] == "summary" for s in r["slots"][:budget]))
                    continue
                if key == "summary_tokens_read":
                    vals.append(sum(s.get("tokens", 0) for s in r["slots"][:budget] if s["type"] == "summary"))
                    continue
            vals.append(r.get(key, 0))
        agg[out] = sum(vals) / n
    return agg


# ------------------------------------------------------------
# routing diagnostics (tree systems)
# ------------------------------------------------------------

def visited_nodes(record, tree):
    """Nodes whose children/articles the system looked at."""
    trace = record.get("trace")
    if isinstance(trace, list):                                   # navigator (v4: node per step; beam: nodes per step)
        nodes = set()
        for s in trace:
            if "node" in s:
                nodes.add(s["node"])
            nodes.update(s.get("nodes", []))
        return nodes
    if isinstance(trace, dict) and "leaves_reached" in trace:     # cosine / BM25 beam traversal
        nodes = set()
        for leaf in trace["leaves_reached"]:
            nodes.update(tree_ancestors(tree, leaf))
        return nodes
    return None


def tree_ancestors(tree, node_id):
    out = []
    while node_id is not None:
        out.append(node_id)
        node_id = tree["by_id"][node_id]["parent_id"]
    return out


def routing_diagnostics(records, tree, budget):
    """Per gold article: reached+opened / reached+skipped / left the gold path at depth k."""
    if tree is None:
        return None
    leaf_of = tree["doc_to_leaf"]
    buckets = Counter()
    for r in records:
        visited = visited_nodes(r, tree)
        if visited is None:
            return None
        opened = set(opened_at(r, budget))
        for g in r["gold"]:
            if g not in leaf_of:
                continue
            path = tree_ancestors(tree, leaf_of[g])[::-1]        # root ... leaf
            deepest = max((i for i, n in enumerate(path) if n in visited), default=-1)
            if deepest == len(path) - 1:
                buckets["reached_opened" if g in opened else "reached_skipped"] += 1
            else:
                buckets[f"left_path_at_depth_{deepest + 1}"] += 1
    return dict(sorted(buckets.items()))


def shortcut_usage(records):
    """How often a shortcut was used: questions where a shortcut target was kept in the beam, followed
    after a review (pure beam) or taken as a jump (v4/v4b). 'gold via shortcut' is reported only where
    the navigator tags the leaf an article was opened from (pure beam records); a gold article opened
    some steps after a jump cannot be attributed to the jump, so v4/v4b get None there."""
    used, gold_after, attribution = 0, 0, True
    for r in records:
        su = r.get("shortcut_use") or {}
        if "gold_via_shortcut" in su:                                   # pure beam navigator
            used += bool(su.get("kept_in_beam") or su.get("followed_in_review"))
            gold_after += bool(su["gold_via_shortcut"])
            continue
        trace = r.get("trace")
        if not isinstance(trace, list):
            return None
        shortcut_jump = any(step.get("action") == "jump" and step.get("jump_kind") == "shortcut" for step in trace)
        used += bool(shortcut_jump or su.get("kept_in_beam", 0) or su.get("jumps", 0))
        attribution = False
    return {"questions_with_jump": used, "jump_then_gold_opened": gold_after if attribution else None}


def evidence_usage(records):
    out = Counter()
    for r in records:
        eu = r.get("evidence_use")
        if eu is None:
            return None
        out["questions_following_evidence"] += bool(eu["followed"])
        out["needs_added"] += eu["needs_added"]
        out["gold_via_evidence"] += bool(eu["gold_via_evidence"])
    return dict(out)


# ------------------------------------------------------------
# report
# ------------------------------------------------------------

def fmt(x, nd=3):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


def delta_str(b):
    return f"{b['diff']:+.3f} [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] p={b['p']:.3f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", choices=list(TREES), default="leiden")
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--ref", default="bm25", help="reference system for the bootstrap test")
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--strict", action="store_true",
                        help="fail unless every system covers every question (use for the final table)")
    parser.add_argument("--only", default=None, help="comma-separated glob patterns: keep only matching systems")
    parser.add_argument("--exclude", default=None, help="comma-separated glob patterns: drop matching systems")
    args = parser.parse_args()
    B = args.budget

    results = load_results(args.tree, args.split, B)
    if args.only:
        pats = [p.strip() for p in args.only.split(",") if p.strip()]
        results = {s: r for s, r in results.items() if any(fnmatch.fnmatchcase(s, p) for p in pats)}
    if args.exclude:
        pats = [p.strip() for p in args.exclude.split(",") if p.strip()]
        results = {s: r for s, r in results.items() if not any(fnmatch.fnmatchcase(s, p) for p in pats)}
    if args.ref and args.ref not in results and results:
        print(f"note: reference system {args.ref} is not among the selected systems; no reference deltas")
    if not results:
        sys.exit(f"No results in {RESULTS_DIR} for tree {args.tree}, split {args.split}. "
                 "Run baselines.py / navigate_tree_v2.py first.")
    tree = load_tree(args.tree)
    systems = sorted(results, key=system_rank)

    # common question set (paired comparisons need the same questions everywhere)
    qid_sets = [set(r["qid"] for r in results[s]["per_question"]) for s in systems]
    common = set.intersection(*qid_sets)
    partial = {s: len(results[s]["per_question"]) for s in systems if len(results[s]["per_question"]) != len(common)}
    expected = {q["qid"] for q in load_questions(args.split)}
    n_total = len(expected)
    if partial or common != expected:
        msg = (f"compared on {len(common)} common questions; the {args.split} split has {n_total}; "
               f"systems with a different count: {partial}")
        if args.strict:
            sys.exit("STRICT: " + msg)
        print("note: " + msg)
    partial = partial or common != expected

    rows, boot, diag = {}, {}, {}
    per_q = {s: {r["qid"]: r for r in results[s]["per_question"] if r["qid"] in common} for s in systems}
    qids = sorted(common)
    scores_of = {s: [question_metrics(per_q[s][q], B)["all_gold"] for q in qids] for s in systems}
    for s in systems:
        recs = [per_q[s][q] for q in qids]
        run_budget = results[s].get("config", {}).get("budget")          # navigators open at most their budget
        rows[s] = {"all": aggregate(recs, B),
                   "single": aggregate([r for r in recs if r["n_gold"] == 1], B),
                   "multi": aggregate([r for r in recs if r["n_gold"] > 1], B),
                   "all@10": aggregate(recs, 10) if run_budget is None or run_budget >= 10 else None}
        rows[s]["stop_reasons"] = dict(Counter(r.get("stop_reason") for r in recs if r.get("stop_reason")))
        diag[s] = {"routing": routing_diagnostics(recs, tree, B) if results[s]["tree"] != "none" else None,
                   "shortcuts": shortcut_usage(recs) if is_nav(s) else None,
                   "evidence": evidence_usage(recs) if is_nav(s) else None}
    ref_scores = scores_of.get(args.ref)
    # tree-only navigator of each (variant, width) = reference for that group's shortcut policies
    none_of = {}
    for s in systems:
        if is_nav(s):
            variant, policy, width = nav_key(results, s)
            if policy == "none":
                none_of[(variant, width)] = s
    for s in systems:
        boot[s] = {}
        if ref_scores and s != args.ref:
            boot[s][f"vs_{args.ref}"] = paired_bootstrap(scores_of[s], ref_scores)
        if is_nav(s):
            variant, policy, width = nav_key(results, s)
            ref_none = none_of.get((variant, width))
            if ref_none and policy != "none":
                boot[s]["vs_nav_none"] = paired_bootstrap(scores_of[s], scores_of[ref_none])
                boot[s]["nav_none_ref"] = ref_none

    # ---- markdown ----
    lines = [f"# Retrieval results — tree `{args.tree}`, split `{args.split}`, budget B={B}, {len(qids)} questions "
             f"({sum(per_q[systems[0]][q]['n_gold'] == 1 for q in qids)} single-gold, "
             f"{sum(per_q[systems[0]][q]['n_gold'] > 1 for q in qids)} multi-gold)", ""]
    if partial:
        lines += [f"**Partial comparison:** {len(qids)} of {n_total} questions are covered by every system "
                  f"({partial}); not a final table.", ""]
    lines.append(f"| system | all-gold@{B} | R@{B} | MRR@{B} | all-gold@{B} single | all-gold@{B} multi | all-gold@10 | "
                 f"opened | summaries | summ. tokens | LLM calls | LLM tokens in/out | Δ all-gold vs {args.ref} [95% CI] p |")
    lines.append("|" + "---|" * 13)
    for s in systems:
        a, sg, mu, a10 = rows[s]["all"], rows[s]["single"], rows[s]["multi"], rows[s]["all@10"]
        b = boot[s].get(f"vs_{args.ref}")
        delta = delta_str(b) if b else "ref" if s == args.ref else "-"
        label = s + (" (oracle ceiling)" if s.startswith("beamoracle") else "")
        a10_value = fmt(a10["all_gold"]) if a10 else "-"
        lines.append(f"| {label} | {fmt(a['all_gold'])} | {fmt(a['recall'])} | {fmt(a['mrr'])} | {fmt(sg.get('all_gold'))} | "
                     f"{fmt(mu.get('all_gold'))} | {a10_value} | {a['opened']:.1f} | {a['summaries']:.1f} | "
                     f"{a['summary_tokens']:.0f} | {a['calls']:.1f} | {a['in_tokens']:.0f}/{a['out_tokens']:.0f} | {delta} |")

    # ---- beam width sweep ----
    beam_systems = [s for s in systems if beam_parts(s)]
    families = defaultdict(dict)                      # (family, policy) -> {width: system}
    for s in beam_systems:
        family, width, policy = beam_parts(s)
        families[(family, policy)][width] = s
    widths = sorted({beam_parts(s)[1] for s in beam_systems})
    sweep = {}
    if beam_systems:
        lines += ["", f"## Beam width sweep (all-gold@{B}; the cosine beam raptor_traversal_bW and the flat baselines for reference)", "",
                  f"| system | W | all-gold@{B} | single | multi | R@{B} | MRR@{B} | summaries | LLM calls | LLM tokens in | Δ vs {args.ref} p |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        def sweep_row(s, w):
            a, sg, mu = rows[s]["all"], rows[s]["single"], rows[s]["multi"]
            b = boot[s].get(f"vs_{args.ref}")
            delta = f"{b['diff']:+.3f} p={b['p']:.3f}" if b else ("ref" if s == args.ref else "-")
            return (f"| {s} | {w} | {fmt(a['all_gold'])} | {fmt(sg.get('all_gold'))} | {fmt(mu.get('all_gold'))} | "
                    f"{fmt(a['recall'])} | {fmt(a['mrr'])} | {a['summaries']:.1f} | {a['calls']:.1f} | {a['in_tokens']:.0f} | {delta} |")

        for (family, policy), by_w in sorted(families.items(),
                                             key=lambda kv: (kv[0][0].startswith("beamoracle"), kv[0][0], kv[0][1] != "none", kv[0][1])):
            for w in sorted(by_w):
                s = by_w[w]
                lines.append(sweep_row(s, w))
                a, sg, mu = rows[s]["all"], rows[s]["single"], rows[s]["multi"]
                sweep.setdefault(f"{family}_{policy}", {})[w] = {"all": a["all_gold"], "single": sg.get("all_gold"),
                                                                 "multi": mu.get("all_gold"), "calls": a["calls"],
                                                                 "summaries": a["summaries"], "in_tokens": a["in_tokens"]}
        for w in widths:
            for ref_name in (f"raptor_traversal_b{w}", f"tree_bm25_b{w}"):
                if ref_name in rows:
                    lines.append(sweep_row(ref_name, w))
        for ref_name in ("dense", "bm25"):
            if ref_name in rows:
                lines.append(sweep_row(ref_name, "-"))
        # pairwise: does W=k beat W=1 of the same family?
        pair_lines = []
        for (family, policy), by_w in sorted(families.items()):
            if 1 in by_w and len(by_w) > 1:
                for w in sorted(by_w):
                    if w == 1:
                        continue
                    b = paired_bootstrap(scores_of[by_w[w]], scores_of[by_w[1]])
                    pair_lines.append(f"| {by_w[w]} | {by_w[1]} | {delta_str(b)} |")
                    boot[by_w[w]]["vs_W1"] = b
        if pair_lines:
            lines += ["", "Width effect within a family (paired bootstrap, all-gold@%d):" % B, "",
                      "| system | vs | Δ [95% CI] p |", "|---|---|---|"] + pair_lines

    # ---- shortcut ablation ----
    nav_rows = [s for s in systems if is_nav(s) and boot[s].get("vs_nav_none")]
    if nav_rows:
        lines += ["", "## Shortcut ablation: policies vs the tree-only navigator of the same variant and width", "",
                  f"| policy | vs | all-gold@{B} | Δ vs tree-only [95% CI] p | multi all-gold@{B} | questions using a shortcut (kept in beam / followed / jumped) | gold opened from a shortcut leaf (pure beam only) |",
                  "|---|---|---|---|---|---|---|"]
        for s in nav_rows:
            b = boot[s]["vs_nav_none"]
            sc = diag[s]["shortcuts"] or {}
            gold_sc = sc.get("jump_then_gold_opened")
            lines.append(f"| {s} | {boot[s]['nav_none_ref']} | {fmt(rows[s]['all']['all_gold'])} | {delta_str(b)} | "
                         f"{fmt(rows[s]['multi'].get('all_gold'))} | {sc.get('questions_with_jump', '-')} | "
                         f"{'-' if gold_sc is None else gold_sc} |")
        lines += ["", "A shortcut is 'used' when a shortcut target was kept in the beam, followed after a review (pure beam) "
                  "or taken as a jump (v4/v4b). Gold opened from a shortcut leaf is counted only where the navigator tags "
                  "the leaf an article came from (pure beam); for v4/v4b a gold article opened after a jump is not "
                  "attributed to the jump. The paired Δ against the tree-only run is the shortcut effect."]
    ev_rows = [s for s in systems if diag[s]["evidence"]]
    if ev_rows:
        lines += ["", "## Evidence review (beam): follow-ups driven by KG facts of the opened articles", "",
                  "| system | questions following an evidence link | needs added from evidence | evidence link → gold opened | stop reasons |",
                  "|---|---|---|---|---|"]
        for s in ev_rows:
            e = diag[s]["evidence"]
            lines.append(f"| {s} | {e.get('questions_following_evidence', 0)} | {e.get('needs_added', 0)} | "
                         f"{e.get('gold_via_evidence', 0)} | {rows[s]['stop_reasons']} |")

    fill_rows = [s for s in systems if any(r.get("fill", {}).get("enabled") for r in per_q[s].values())]
    if fill_rows:
        lines += ["", "## Budget fill (beam --fill): articles opened from the reserve when the search ended with budget left", "",
                  "| system | opened/q | questions filled | articles filled | fill → gold opened | all-gold@%d |" % B,
                  "|---|---|---|---|---|---|"]
        for s in fill_rows:
            recs = [per_q[s][q] for q in qids]
            filled = [len(r["fill"]["filled"]) for r in recs]
            lines.append(f"| {s} | {rows[s]['all']['opened']:.2f} | {sum(f > 0 for f in filled)} | {sum(filled)} | "
                         f"{sum(bool(r['fill']['gold_via_fill']) for r in recs)} | {fmt(rows[s]['all']['all_gold'])} |")
    tree_rows = [s for s in systems if diag[s]["routing"]]
    if tree_rows:
        lines += ["", f"## Routing diagnostics (per gold article, budget {B})", "",
                  "| system | reached & opened | reached & skipped | left gold path at depth 1 | depth 2 | depth 3 | deeper |",
                  "|---|---|---|---|---|---|---|"]
        for s in tree_rows:
            d = diag[s]["routing"]
            deeper = sum(v for k, v in d.items() if k.startswith("left_path_at_depth_") and int(k.rsplit("_", 1)[1]) >= 4)
            lines.append(f"| {s} | {d.get('reached_opened', 0)} | {d.get('reached_skipped', 0)} | "
                         f"{d.get('left_path_at_depth_1', 0)} | {d.get('left_path_at_depth_2', 0)} | "
                         f"{d.get('left_path_at_depth_3', 0)} | {deeper} |")
        lines += ["", "'reached' = the system looked at the gold article's leaf (beam: the leaf's articles were listed; "
                  "cosine beam: the leaf was in the final frontier)."]

    # ---- budget scaling ----
    ks = list(range(1, 11))
    scaling = {}
    for s in systems:
        b = results[s].get("config", {}).get("budget")
        curve = []
        for k in ks:
            if b is not None and k > b:
                curve.append(None)
            else:
                curve.append(sum(question_metrics(per_q[s][q], k)["all_gold"] for q in qids) / len(qids))
        scaling[s] = curve
    lines += ["", "## Budget scaling: all-gold recall vs. articles read (k)", "",
              "| system | " + " | ".join(f"k={k}" for k in ks) + " |", "|---|" + "---|" * len(ks)]
    for s in systems:
        lines.append(f"| {s} | " + " | ".join("-" if v is None else f"{v:.2f}" for v in scaling[s]) + " |")
    lines.append("")
    lines.append("Navigator curves end at the run's article budget; ranked systems are truncated at k. "
                 "For the navigator at other budgets, rerun navigate_tree_v2.py with --budget.")

    # ---- figures ----
    tag = "_selected" if (args.only or args.exclude) else ""      # filtered runs get their own output files
    figs = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        png = EVAL_DIR / f"scaling_v2{tag}_{args.tree}_{args.split}.png"
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for s in systems:
            if s.startswith("beamoracle"):
                continue
            pts = [(k, v) for k, v in zip(ks, scaling[s]) if v is not None]
            style = "-o" if is_nav(s) else ("--" if s in ("bm25", "dense") else ":")
            ax.plot([k for k, _ in pts], [v for _, v in pts], style, label=s, markersize=3)
        ax.set_xlabel("articles read (k)")
        ax.set_ylabel("all-gold recall@k")
        ax.set_ylim(0, 1)
        ax.set_xticks(ks)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2)
        fig.tight_layout()
        fig.savefig(png, dpi=150)
        plt.close(fig)
        figs.append(png.name)
        if beam_systems and len(widths) > 1:
            png = EVAL_DIR / f"beam_width{tag}_{args.tree}_{args.split}.png"
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            panels = (("all", f"all-gold@{B} (all {len(qids)} questions)"),
                      ("multi", f"all-gold@{B} (multi-gold questions)"), ("calls", "LLM calls per question"))
            for ax, (key, title) in zip(axes, panels):
                for name, by_w in sorted(sweep.items()):
                    xs = sorted(by_w)
                    ys = [by_w[w][key] for w in xs]
                    if any(y is None for y in ys):
                        continue
                    oracle = name.startswith("beamoracle")
                    style = ":" if oracle else ("-o" if name.endswith(("typed", "strongest")) else "--s")
                    ax.plot(xs, ys, style, label=name + (" (ceiling)" if oracle else ""), markersize=4,
                            alpha=0.6 if oracle else 1.0)
                if key != "calls":
                    cos = [(w, rows[f"raptor_traversal_b{w}"][key]["all_gold"]) for w in widths
                           if f"raptor_traversal_b{w}" in rows and rows[f"raptor_traversal_b{w}"][key]]
                    if cos:
                        ax.plot([w for w, _ in cos], [v for _, v in cos], "-.^", color="gray", label="cosine beam (raptor_traversal_bW)", markersize=4)
                    for ref_name, color in (("dense", "black"), ("bm25", "brown")):
                        if ref_name in rows and rows[ref_name][key]:
                            ax.axhline(rows[ref_name][key]["all_gold"], color=color, linestyle="--", linewidth=1, label=ref_name)
                    v4 = [s for s in systems if s.startswith("nav_v4_") and "noev" not in s]
                    for s in v4:
                        if rows[s][key]:
                            ax.axhline(rows[s][key]["all_gold"], color="purple", linestyle=":", linewidth=1, label=s)
                    ax.set_ylim(0, 1)
                ax.set_xlabel("beam width W")
                ax.set_title(title, fontsize=10)
                ax.set_xticks(widths)
                ax.grid(alpha=0.3)
            axes[0].legend(fontsize=6)
            fig.tight_layout()
            fig.savefig(png, dpi=150)
            plt.close(fig)
            figs.append(png.name)
        lines.append("Figures: " + ", ".join(f"`{f}`" for f in figs))
    except Exception as error:  # matplotlib optional
        lines.append(f"(no figure: {type(error).__name__}: {error})")
    stops = {s: rows[s]["stop_reasons"] for s in systems if rows[s]["stop_reasons"]}
    if stops:
        lines += ["", "## Navigator stop reasons", ""]
        for s, c in stops.items():
            lines.append(f"- {s}: {c}")
    lines += ["", "Notes: all-gold@B = every gold article of the question is among the first B articles "
              "(complete evidence-chain recall). RAPTOR collapsed: a node summary occupies one of the B slots. "
              "Beam systems read the summaries of every candidate they rated (summaries column). "
              "beamoracle_* = the beam protocol with a gold-aware scorer: the ceiling the tree + shortcuts + evidence "
              "links allow at that width, not a retrieval system. "
              f"Bootstrap: paired, 2000 resamples, over the {len(qids)} common questions."]
    report = "\n".join(lines)
    out_md = EVAL_DIR / f"summary_v2{tag}_{args.tree}_{args.split}_B{B}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(report + "\n", encoding="utf-8")
    write_json(EVAL_DIR / f"summary_v2{tag}_{args.tree}_{args.split}_B{B}.json",
               {"tree": args.tree, "split": args.split, "budget": B, "n_questions": len(qids), "systems": rows, "bootstrap": boot,
                "diagnostics": diag, "scaling": {"k": ks, "all_gold": scaling}, "beam_sweep": sweep,
                "configs": {s: results[s]["config"] for s in systems}})
    print(report)
    print(f"\nSaved: {out_md}")


if __name__ == "__main__":
    main()