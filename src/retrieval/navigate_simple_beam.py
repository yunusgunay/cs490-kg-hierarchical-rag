# src/retrieval/navigate_simple_beam.py
"""
Graph-guided hierarchical retrieval - the mentor's "latest simplified beam" (2026-11-03) run on
OUR data and written in OUR result format (step 7, system family `simplebeam`).

Method (core vendored verbatim in retrieval/simplebeam/, see its __init__):
  1. plan      one LLM call -> 1-3 information needs (NavPlan)
  2. descent   ONE global LLM beam over the tree: every round the frontier's internal nodes are
               expanded, retained leaves and new children (policy typed: + the KG shortcut targets
               of the expanded nodes) are rated 0-100 in one call, the top W (score >= 10) survive;
               W = total frontier size; pruned nodes are never reopened; at most 8 rounds
  3. pool      the articles of the surviving leaves (disjoint leaves -> no duplicates)
  4. select    ONE LLM call picks and orders exactly min(B, |pool|) articles from title + key
               concepts; the plan is prefixed as "provisional needs" (question stays authoritative)
No agent loop, no up/jump moves, no evidence review, no second wave, no embeddings.
Model settings are the package's: plan/route reasoning_effort=none, temperature 0, 8000 output
tokens; select reasoning_effort=low, 4000 output tokens. Prompts are generic (not Wix-specific).

Compared with our navigators: navigate_tree_beam.py (v4c) keeps the v4 agent after the beam;
navigate_beam.py (pure beam v5) opens per leaf and runs an evidence review. This file is the
simplest member of the family: beam + one joint selection.

Replay. The package ships the model decisions of its W2/4/6/8 x none/typed dev runs
(recorded/decisions_*.jsonl). With --replay-from <package dir> those decisions act as a read-only
cache: the eight dev conditions on --tree leiden-rich rerun with ZERO API calls and are checked
against the package's recorded outputs. Any other condition (other tree/summaries, held-out, other
width) is a fresh paid run.

Usage
-----
    python src/retrieval/navigate_simple_beam.py --dry-run --tree leiden-rich
    python src/retrieval/navigate_simple_beam.py --tree leiden-rich --widths 2,4,6,8 --policy none,typed \
        --replay-from <path>/latest_beam_research                       # 0 API calls
    python src/retrieval/navigate_simple_beam.py --tree leiden --widths 6 --policy none          # old summaries (ablation)
    python src/retrieval/navigate_simple_beam.py --tree leiden-rich --widths 6 --policy none --split heldout
Outputs: data/processed/subset/eval/results/simplebeam_W<W>_<policy>_B<B>__<tree>__<split>.json
Cache:   data/processed/subset/eval/cache/simplebeam_cache.jsonl (package-compatible request keys)
Env (.env): OPENAI_API_KEY, LLM_MODEL (default gpt-6-luna; the recorded decisions are gpt-6-luna)
"""
import argparse
import hashlib
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.common import (CACHE_DIR, PROJECT_ROOT, SUMMARY_DIR, TREES, append_jsonl, load_questions,  # noqa: E402
                              load_shortcuts, load_summaries, num_tokens, quick_report, read_jsonl,
                              save_results, sha256_text)
from retrieval.simplebeam.simple_beam import PLAN, SCORE, _Candidate  # noqa: E402
from retrieval.simplebeam.simple_beam_plan_selection import (PROVISIONAL_NEEDS_INSTRUCTION,  # noqa: E402
                                                             PlanConditionedSelection)
from retrieval.simplebeam.simple_beam_shortcuts import SimpleBeamRetriever  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-6-luna").strip()
SCHEMA_VERSION = "simple-beam-r1"
KEY_VERSION = "research-beam-r1"            # the package's request-key version: keep for replay
CACHE_PATH = CACHE_DIR / "simplebeam_cache.jsonl"
WIDTHS = (6,)
MAX_ROUNDS = 8
MIN_CANDIDATE_SCORE = 10
POLICIES = ("none", "typed", "random_typed", "strongest", "random")

# exact text of the package's prompts/select.txt (stripped), as run_beam.py passes it
SELECT_PROMPT = (
    "Select evidence for the user's complete question from the candidate documents. Return exactly the requested "
    "number of distinct indices in decreasing usefulness, followed by one short comparative reason. Do not answer "
    "the question.\n\n"
    "Put the document with the strongest direct support for the requested information and explicit constraints "
    "first. Distinguish what the question states from causes, intentions, or conditions you would have to assume. "
    "Use the remaining slots for directly relevant and complementary evidence across the stated needs and "
    "reasonable unresolved interpretations. Prefer coverage explicitly supported by the metadata over speculative "
    "supporting material. A broader document whose key concepts explicitly cover the question can be more useful "
    "than several narrower documents for one unestablished interpretation. Do not exclude a relevant source merely "
    "because its topic or an action it covers is already mentioned in the question.\n\n"
    "Titles describe the primary subject; key concepts are incomplete evidence of coverage. Read both, without "
    "inventing unstated contents or treating missing words as proof of irrelevance. Overlapping sources are useful "
    "when they add evidence; neither topical variety nor repeated keywords alone makes a source useful. Choose one "
    "coherent evidence set and order it by relevance."
)


# ------------------------------------------------------------
# model settings and request keys (identical to the package's portable_api.py)
# ------------------------------------------------------------

def beam_settings(stage, model=MODEL_NAME):
    selection = stage in ("select", "select_retry")
    if not selection and stage != "plan" and not stage.startswith("score_round_"):
        raise ValueError(f"unknown beam stage {stage}")
    return {"model": model, "reasoning_effort": "low" if selection else "none",
            "temperature": None if selection else 0.0, "max_output_tokens": 4000 if selection else 8000}


def request_key(system, user, schema, settings):
    value = {"version": KEY_VERSION, "settings": settings,
             "schema": json.dumps(schema.model_json_schema(), sort_keys=True), "system": system, "user": user}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def router_text(summary):
    """The candidate text the router renders (title, summary, <=4 question types, <=10 key terms, coverage)."""
    parts = [summary["title"], summary["summary"], " | ".join(summary.get("question_types", [])[:4]),
             ", ".join(summary.get("key_terms", [])[:10]), " | ".join(summary.get("coverage", []))]
    return "\n".join(p for p in parts if p)


# ------------------------------------------------------------
# LLM access: recorded decisions -> our cache -> API
# ------------------------------------------------------------

class LLM:
    """ask(system, user, schema, stage) with the package's settings, keys and cache semantics."""

    def __init__(self, client=None, use_cache=True, recorded=None, allow_api=True):
        self.client, self.use_cache, self.allow_api = client, use_cache, allow_api
        self.recorded = recorded or {}
        self.lock = threading.Lock()
        self.local = threading.local()
        self.cache = {}
        if use_cache and CACHE_PATH.exists():
            for r in read_jsonl(CACHE_PATH):
                self.cache[r["key"]] = r

    def __call__(self, system, user, schema, stage):
        settings = beam_settings(stage)
        key = request_key(system, user, schema, settings)
        record = self.recorded.get(key)
        source = "recorded" if record else None
        if record is None and self.use_cache:
            record = self.cache.get(key)
            source = "cache" if record else None
        if record is None:
            record = self._api(system, user, schema, settings, key)
            source = "api"
        self.local.event = {"key": key, "stage": stage, "source": source, "usage": dict(record["usage"]),
                            "decision": record["decision"]}
        return schema.model_validate(record["decision"])

    def last_event(self):
        return dict(self.local.event)

    def _api(self, system, user, schema, settings, key):
        if not self.allow_api or self.client is None:
            raise RuntimeError(f"no recorded/cached decision for this request and API calls are disabled (stage {key[:8]})")
        request = {"model": settings["model"], "reasoning_effort": settings["reasoning_effort"],
                   "max_completion_tokens": settings["max_output_tokens"], "response_format": schema,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if settings["temperature"] is not None:
            request["temperature"] = settings["temperature"]
        response = self.client.chat.completions.parse(**request)
        choice = response.choices[0]
        if choice.finish_reason == "length" or choice.message.parsed is None:
            raise RuntimeError("No structured output returned.")
        u = response.usage
        details = getattr(u, "completion_tokens_details", None)
        usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                 "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0}
        record = {"key": key, "model": settings["model"], "settings": settings,
                  "decision": choice.message.parsed.model_dump(), "usage": usage}
        if self.use_cache:
            with self.lock:
                self.cache[key] = record
                append_jsonl(CACHE_PATH, record)
        return record


class Capture:
    """Per-question wrapper: collects the events of one retrieval (his CaptureAsk)."""

    def __init__(self, llm):
        self.llm, self.events = llm, []

    def __call__(self, system, user, schema, stage):
        value = self.llm(system, user, schema, stage)
        self.events.append(self.llm.last_event())
        return value


def load_recorded(package_dir):
    """Recorded decisions of the package, per condition: {"W6_none": {key: record}, ...}.

    The package keeps one decision file per condition. A few identical requests (same key) were
    answered differently in different conditions (the model is not exactly deterministic at
    temperature 0), so a replay must prefer the file of the condition being rerun.
    """
    root = Path(package_dir)
    files = sorted((root / "recorded").glob("decisions_*.jsonl")) or sorted(root.glob("decisions_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no recorded/decisions_*.jsonl under {root}")
    out = {}
    for f in files:
        condition = f.stem.removeprefix("decisions_")
        out[condition] = {}
        for r in read_jsonl(f):
            rec = {k: (json.loads(v) if isinstance(v, str) and k in ("decision", "usage", "settings") else v)
                   for k, v in r.items()}
            out[condition][rec["key"]] = rec
    return out


def recorded_for(recorded, condition):
    """The condition's own decisions win; the other conditions' decisions fill the gaps."""
    merged = {}
    for cond, recs in recorded.items():
        if cond != condition:
            merged.update(recs)
    merged.update(recorded.get(condition, {}))
    return merged


# ------------------------------------------------------------
# dataset adapter
# ------------------------------------------------------------

class SimpleBeamRunner:
    def __init__(self, tree_method, policy, width, budget, llm, max_rounds=MAX_ROUNDS):
        data = load_summaries(tree_method)
        self.tree_method, self.policy, self.width, self.budget, self.llm = tree_method, policy, width, budget, llm
        self.max_rounds = max_rounds
        self.root = data["tree"]["root_id"]
        self.nodes = data["tree"]["by_id"]
        self.tree = {n: {"children": list(v["children"]), "is_leaf": bool(v["is_leaf"]), "doc_ids": list(v["doc_ids"])}
                     for n, v in self.nodes.items()}
        self.summaries = data["summaries"]
        self.doc_rows = data["doc_rows"]
        self.index_info = {"file": str(SUMMARY_DIR / f"{tree_method}_summaries.json"),
                           "schema_version": data.get("schema_version"), "source": data.get("source")}
        self.shortcuts = load_shortcuts(tree_method, policy) if policy != "none" else {}
        self.leaf_of = {d: n for n, v in self.nodes.items() if v["is_leaf"] for d in v["doc_ids"]}

    def retriever(self, ask):
        return SimpleBeamRetriever(self.tree, self.summaries, self.doc_rows, self.root, ask,
                                   width=self.width, budget=self.budget, max_rounds=self.max_rounds,
                                   card_view="terms", selector_prompt=SELECT_PROMPT,
                                   policy="none" if self.policy == "none" else "typed", shortcuts=self.shortcuts)

    def run(self, q):
        capture = Capture(self.llm)
        result = self.retriever(PlanConditionedSelection(capture, q["question"])).retrieve(q["question"])
        events = capture.events
        # our trace convention: a list of steps; `nodes` = nodes whose children/articles were looked at
        trace, k = [], 0
        for step in result["trace"]:
            s = dict(step)
            if step["action"] == "plan":
                s["cached"] = events[k]["source"] != "api"; k += 1
            elif step["action"] == "score":
                s["cached"] = events[k]["source"] != "api"; k += 1
            elif step["action"] == "select":
                s["nodes"] = list(result["leaves_reached"])          # the selector lists these leaves' articles
                s["cached"] = events[k]["source"] != "api"; k += 1
            trace.append(s)
        scored = [c["node"] for s in result["trace"] if s["action"] == "score" for c in s["candidates"]]
        su = result.get("shortcut_use") or {"candidates_scored": 0, "kept_in_beam": 0,
                                            "leaves_via_shortcut": [], "opened_via_shortcut": []}
        gold = set(q["gold"])
        api = [e for e in events if e["source"] == "api"]
        return {
            "qid": q["qid"], "question": q["question"], "gold": q["gold"], "n_gold": q["n_gold"],
            "opened": list(result["opened"]), "candidate_pool": list(result["candidate_ids"]),
            "needs": list(result["needs"]), "rounds": result["rounds"], "leaves_reached": list(result["leaves_reached"]),
            "stop_reason": "beam_leaves",
            "summaries_read": len(scored),
            "summary_tokens_read": sum(num_tokens(router_text(self.summaries[n])) for n in scored),
            "doc_rows_read": len(result["candidate_ids"]),
            "llm_calls": len(events),
            "input_tokens": sum(e["usage"]["input_tokens"] for e in events),
            "output_tokens": sum(e["usage"]["output_tokens"] for e in events),      # includes reasoning tokens
            "reasoning_tokens": sum(e["usage"].get("reasoning_tokens", 0) for e in events),
            "api_calls_this_run": len(api),
            "api_input_tokens_this_run": sum(e["usage"]["input_tokens"] for e in api),
            "api_output_tokens_this_run": sum(e["usage"]["output_tokens"] for e in api),
            "decision_sources": dict(Counter(e["source"] for e in events)),
            "shortcut_use": {"candidates_scored": su["candidates_scored"], "kept_in_beam": su["kept_in_beam"],
                             "followed_in_review": 0,
                             "leaves_via_shortcut": list(su["leaves_via_shortcut"]),
                             "gold_via_shortcut": len(gold & set(su["opened_via_shortcut"]))},
            "trace": trace,
        }

    def dry_run(self, question):
        """Render the root-level routing prompt and a selection prompt without any model call."""
        r = self.retriever(lambda *a: (_ for _ in ()).throw(RuntimeError("dry run")))
        needs = [question]
        root_children = [_Candidate(c, "child", label=f"child of '{self.summaries[self.root]['title']}'")
                         for c in self.tree[self.root]["children"]]
        score_prompt = r._score_user(question, needs, root_children)
        first_leaf = next(n for n in self.nodes if self.nodes[n]["is_leaf"])
        select_prompt = r._selection_user(question, self.tree[first_leaf]["doc_ids"])
        return score_prompt, select_prompt


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", choices=list(TREES), default="leiden-rich" if "leiden-rich" in TREES else "leiden")
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)), help="comma list, e.g. 2,4,6,8")
    parser.add_argument("--policy", default="none", help="comma list of " + ",".join(POLICIES))
    parser.add_argument("--budget", type=int, default=5)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="first N questions (pilot)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=MAX_ROUNDS)
    parser.add_argument("--replay-from", default=None, metavar="DIR",
                        help="package directory: its recorded/decisions_*.jsonl serve as a read-only cache")
    parser.add_argument("--no-api", action="store_true", help="fail instead of calling the API on a cache miss")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    widths = [int(w) for w in args.widths.split(",")]
    policies = [p.strip() for p in args.policy.split(",")]
    bad = [p for p in policies if p not in POLICIES]
    if bad or any(w < 1 for w in widths):
        parser.error(f"bad policy {bad} or width {widths}")
    if args.split == "heldout":
        print("WARNING: held-out split is for the final run only; do not tune on it.")
    questions = load_questions(args.split)
    if args.limit:
        questions = questions[:args.limit]

    recorded = {}
    if args.replay_from:
        recorded = load_recorded(args.replay_from)
        print(f"recorded decisions: {sum(len(v) for v in recorded.values())} records in {len(recorded)} condition files "
              f"({args.replay_from})")

    if args.dry_run:
        runner = SimpleBeamRunner(args.tree, policies[0], widths[0], args.budget, LLM(allow_api=False), args.max_rounds)
        q = questions[0]
        score_prompt, select_prompt = runner.dry_run(q["question"])
        print("=" * 70 + "\n[PLAN system prompt]\n" + "=" * 70 + "\n" + PLAN)
        print("=" * 70 + "\n[ROUTE system prompt]\n" + "=" * 70 + "\n" + SCORE)
        print("=" * 70 + f"\n[routing prompt at the root for {q['qid']}: {len(runner.tree[runner.root]['children'])} candidates, "
              f"~{num_tokens(score_prompt)} tokens]\n" + "=" * 70 + "\n" + score_prompt)
        print("=" * 70 + "\n[SELECT system prompt]\n" + "=" * 70 + "\n" + SELECT_PROMPT)
        print("=" * 70 + f"\n[selection prompt example (one leaf's articles), ~{num_tokens(select_prompt)} tokens; "
              f"the plan is prefixed as provisional needs]\n" + "=" * 70 + "\n" + select_prompt)
        words = sum(len(s["summary"].split()) for s in runner.summaries.values()) / len(runner.summaries)
        print(f"\ntree {args.tree}: {len(runner.nodes)} nodes, summaries {words:.0f} words on average, "
              f"index {runner.index_info['file']} | settings plan/route {beam_settings('plan')} | select {beam_settings('select')} "
              f"| widths {widths} | policies {policies} | budget {args.budget} | questions {len(questions)} ({args.split})")
        return

    client = None
    if not args.no_api:
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("Missing OPENAI_API_KEY in .env (or use --no-api with --replay-from)")
        from openai import OpenAI
        client = OpenAI(timeout=180.0, max_retries=3)
    llm = LLM(client=client, use_cache=not args.no_cache, allow_api=not args.no_api)

    expected_dir = Path(args.replay_from) / "recorded" if args.replay_from else None
    for width, policy in [(w, p) for w in widths for p in policies]:
        llm.recorded = recorded_for(recorded, f"W{width}_{policy}") if recorded else {}
        runner = SimpleBeamRunner(args.tree, policy, width, args.budget, llm, args.max_rounds)
        system = f"simplebeam_W{width}_{policy}_B{args.budget}"
        print(f"\n========== SIMPLE BEAM W={width} policy={policy} tree={args.tree} budget={args.budget} "
              f"model={MODEL_NAME} | {len(questions)} questions ({args.split}) ==========")
        records = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(runner.run, q): q for q in questions}
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    r = fut.result()
                except Exception as error:
                    print(f"  FAILED {q['qid']}: {type(error).__name__}: {error}")
                    continue
                records[q["qid"]] = r
                got = len(set(r["gold"]) & set(r["opened"]))
                print(f"  {r['qid']} gold {got}/{r['n_gold']} | rounds {r['rounds']} leaves {len(r['leaves_reached'])} "
                      f"pool {r['doc_rows_read']} | calls {r['llm_calls']} {r['decision_sources']} "
                      f"| sc kept {r['shortcut_use']['kept_in_beam']} | {r['question'][:55]}")
        ordered = [records[q["qid"]] for q in questions if q["qid"] in records]
        if not ordered:
            print("no results")
            continue
        config = {"model": MODEL_NAME, "settings": {"plan": beam_settings("plan"), "route": beam_settings("score_round_1"),
                                                    "select": beam_settings("select")},
                  "budget": args.budget, "width": width, "policy": policy, "variant": "simplebeam",
                  "descent_shortcuts": policy != "none", "max_rounds": args.max_rounds,
                  "min_candidate_score": MIN_CANDIDATE_SCORE, "card_view": "terms",
                  "schema_version": SCHEMA_VERSION, "key_version": KEY_VERSION,
                  "prompt_sha256": sha256_text(PLAN + SCORE + SELECT_PROMPT + PROVISIONAL_NEEDS_INSTRUCTION),
                  "summary_index": runner.index_info, "replay_from": args.replay_from, "limit": args.limit,
                  "source": "mentor package latest_beam_research_2026-11-03 (core vendored in retrieval/simplebeam)"}
        path = save_results(system, args.tree, args.split, config, ordered)
        n = len(ordered)
        print(quick_report(system, ordered, budgets=(args.budget,)))
        sources = Counter()
        for r in ordered:
            sources.update(r["decision_sources"])
        print(f"  rounds/q {sum(r['rounds'] for r in ordered) / n:.2f} | leaves/q {sum(len(r['leaves_reached']) for r in ordered) / n:.2f} "
              f"| pool/q {sum(r['doc_rows_read'] for r in ordered) / n:.1f} | summaries/q {sum(r['summaries_read'] for r in ordered) / n:.1f} "
              f"| decisions {dict(sources)} | API calls this run {sum(r['api_calls_this_run'] for r in ordered)}")
        if expected_dir and args.split == "dev":
            exp = expected_dir / f"expected_W{width}_{policy}_B5.json".replace("_B5", "")
            if exp.exists():
                prior = {r["qid"]: r for r in json.load(exp.open(encoding="utf-8"))["per_question"]}
                same = sum(prior.get(r["qid"], {}).get("opened") == r["opened"] and
                           prior.get(r["qid"], {}).get("leaves_reached") == r["leaves_reached"] for r in ordered)
                print(f"  replay check vs {exp.name}: {same}/{n} questions identical (opened + leaves)")
        print(f"  saved {path}")


if __name__ == "__main__":
    main()