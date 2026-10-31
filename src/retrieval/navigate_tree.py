# src/retrieval/navigate_tree.py
"""
Graph-guided hierarchical retrieval - the online navigator (step 7).

End-to-end method (offline steps are the earlier scripts):
  corpus -> KG extraction (extract_relations_v5) -> entity canonicalisation
  -> document graph (build_doc_graph) -> recursive Leiden tree (build_tree_v2)
  -> routing summaries (summarize_tree_v2) -> cross-branch shortcuts
  (select_shortcuts) -> THIS FILE: an LLM agent that navigates the tree to
  collect the articles a question needs, revising its plan from KG evidence.

One question, one run:
  1. PLAN      one LLM call lists the question's information needs (1-3).
  2. NAVIGATE  from the root, one LLM call per step. The prompt shows the
               question, the needs checklist, opened articles, EVIDENCE NOTES,
               path, NOTEBOOK (previous moves + reasons), the current node with
               its children (or its articles at a leaf) and JUMP TARGETS
               (precomputed KG shortcuts of the chosen policy + evidence links).
               Actions: descend / jump / up / stop; at a leaf: open articles.
  3. REVIEW    after opening, the next call stays on the node with the opened
               articles' KG facts in view ([F1], [F2], ...: REQUIRES,
               HAS_LIMITATION, RESOLVED_BY, CAUSES, APPLIES_TO, LOCATED_IN,
               each with the other articles covering the same specific entity).
               The navigator may add a need (new_needs, must cite a fact) and
               jump to the leaf of a related article (evidence link).
  4. CONTROL   a stop is refused while a need is uncovered and budget remains
               (twice); redundant moves are refused and flagged EXHAUSTED
               (twice), then the run ends; a node seen a 4th time ends the run.
Budget B articles (default 5); cost measured in summaries read, articles
opened, LLM calls and tokens. Article bodies are never read: evidence notes
are pre-extracted KG facts = candidate leads, not verified answers.

Ablations: --policy none|typed|random_typed|strongest|random (precomputed
shortcuts); --no-evidence (no evidence notes/links, prompt and schema without
new_needs). Results: eval/results/nav_v4[noev]_<policy>_B<budget>__<tree>__<split>.json
Every LLM decision is cached (eval/cache/navigator_cache.jsonl) by full prompt:
re-runs are free and replay exactly; an API error leaves the question out of
the results file (rerun to fill it).

Usage
-----
    python src/retrieval/navigate_tree.py --dry-run                   # prompts, no API
    python src/retrieval/navigate_tree.py --policy none --limit 10    # pilot
    python src/retrieval/navigate_tree.py --policy none,typed         # main runs
    python src/retrieval/navigate_tree.py --policy none,typed --no-evidence   # 2x2 ablation
    python src/retrieval/navigate_tree.py --tree tfidf-kmeans --policy none  # partition ablation
    python src/retrieval/evaluate.py --strict

Env (.env): OPENAI_API_KEY, LLM_MODEL (gpt-6-luna), LLM_NAV_REASONING_EFFORT
(falls back to LLM_REASONING_EFFORT, default medium).
Needs: <tree>_summaries.json, kg_v5_*_canonical.jsonl, and for a shortcut
policy other than none: <tree>_shortcuts_<policy>.json (select_shortcuts.py).
"""
import argparse
import json
import os
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from retrieval.common import (CACHE_DIR, DATA_DIR, PROJECT_ROOT, SHORTCUT_POLICIES, TREES,  # noqa: E402
                              append_jsonl, load_corpus, load_questions, load_shortcuts, load_summaries,
                              num_tokens, quick_report, read_jsonl, save_results, sha256_text)

load_dotenv(PROJECT_ROOT / ".env")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-6-luna").strip()
REASONING_EFFORT = os.getenv("LLM_NAV_REASONING_EFFORT",
                             os.getenv("LLM_REASONING_EFFORT", "medium")).strip().lower()
IS_REASONING_MODEL = MODEL_NAME.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
CANON_PATH = DATA_DIR / "kg_v5_gpt-6-luna_medium_canonical.jsonl"

VARIANT = "v4"
SCHEMA_VERSION = f"navigator-{VARIANT}"
BUDGET = 5                     # articles the navigator may open
MAX_CALLS = 18                 # LLM calls per question incl. the plan and the review steps (v3 used 14)
MAX_VISITS = 3                 # a node seen a fourth time ends the run (loop guard)
MAX_STOP_REFUSALS = 2          # after two refused stops the navigator's stop is accepted
MAX_MOVE_REFUSALS = 2          # redundant moves are refused (and flagged EXHAUSTED) this many times, then the run ends
MAX_NEEDS = 4                  # plan needs + needs added from evidence
CHILD_QUESTIONS = 4
CHILD_TERMS = 10
SHORTCUT_SUMMARY_CHARS = 220
NOTEBOOK_STEPS = 10
DEFAULT_WORKERS = 4
MAX_OUTPUT_TOKENS = 2500
POLICIES = ("none",) + SHORTCUT_POLICIES
CACHE_PATH = CACHE_DIR / "navigator_cache.jsonl"

# evidence notes
DEP_PREDICATES = {"REQUIRES": 1.0, "RESOLVED_BY": 1.0, "CAUSES": 0.8, "HAS_LIMITATION": 0.8,
                  "APPLIES_TO": 0.8, "LOCATED_IN": 0.6}
GENERIC_NAMES = {"wix", "site", "wix site", "website", "page", "user", "visitor",
                 "wix account", "dashboard", "site dashboard"}
EVIDENCE_MAX_DF_RATIO = 0.05   # an entity in more than 5% of the articles is not a specific link
NOTES_PER_DOC = 6              # dependency facts shown per opened article (6x3 shows the partner gold article
                               # of 31/61 dev multi-gold pairs; 4x2 showed 22/61)
DOCS_PER_NOTE = 3              # related articles listed per fact
MAX_EVIDENCE_LINKS = 5         # leaves offered as evidence jump targets
MAX_FACTS_SHOWN = 12           # facts per prompt (most recently opened article first)
MAX_EVIDENCE_NEEDS = 1         # needs the navigator may add from evidence (most multi-gold questions have 2 gold articles)


# ------------------------------------------------------------
# schemas + prompts
# ------------------------------------------------------------

class NavPlan(BaseModel):
    needs: list[str]


class NewNeed(BaseModel):
    need: str
    fact: int            # index of the EVIDENCE NOTES fact ([F1], [F2], ...) this need comes from


class NavDecision(BaseModel):
    open_docs: list[int]
    covered_needs: list[int]
    new_needs: list[NewNeed]
    action: Literal["descend", "jump", "up", "stop"]
    target: int | None
    reason: str


class NavDecisionNoEvidence(BaseModel):
    open_docs: list[int]
    covered_needs: list[int]
    action: Literal["descend", "jump", "up", "stop"]
    target: int | None
    reason: str


PLAN_PROMPT = """
You prepare a retrieval plan for a question asked by a Wix website owner to
the Wix help center. List the distinct INFORMATION NEEDS a complete answer
must cover; each need is normally covered by one help-center article.

Most questions have one need. Add a second (rarely a third) need when:
- the question asks about two different things ("change X" and "does it
  affect Y"), or
- the task has a prerequisite or companion step that lives in its own
  article (connect a provider before changing its settings; create the plan
  before enabling its checkout option; the setting and the troubleshooting
  of that setting), or
- the question mixes "how do I" with "why does it fail / what are the limits".
Do not invent needs the question does not imply. Phrase each need as the
topic of the article that would cover it, 3-12 words, specific to the
product and task (e.g. "enabling guest checkout for Pricing Plans",
"creating a free Pricing Plan"). Return the list only.
"""

NAV_PROMPT = """
You navigate a tree of Wix help-center articles to find the article(s) that
answer a user's question. Internal nodes group related articles and carry a
routing summary; leaves list articles. JUMP TARGETS lead to a node in another
branch: precomputed knowledge-graph shortcuts, and EVIDENCE LINKS derived
from what you already opened.

You were given a plan: the INFORMATION NEEDS the answer must cover. Your goal
is to open an article for EVERY need - needs may sit in different branches -
within a budget of articles. Success is measured by whether the needed
articles are among the ones you opened, not by how few you opened: when
several articles could cover a need, open all plausible ones; if budget
remains at a leaf, also open the supporting article a support agent would
attach (the prerequisite, the related setting, the troubleshooting article).
Do not open articles that clearly do not fit. You never see article bodies;
judge from titles, one-line descriptions and key entities.

After you open an article, EVIDENCE NOTES list its knowledge-graph facts
([F1], [F2], ...): what it requires, what limits it, what causes or resolves
a problem, where a setting lives - each with other articles that cover the
same specific entity. These are candidate leads extracted automatically, not
verified answers. Use them to revise your plan: if a fact reveals something
the answer also needs (a prerequisite step, the fix of a cause, the companion
setting), add it with new_needs citing the fact, and follow the matching
evidence link. Do not add needs for facts that are merely context.

Each step shows the question, the needs and which are covered, what you
opened, the evidence notes, your path, your NOTEBOOK (previous moves and
reasons), the current node, its children (or its articles if it is a leaf)
and the jump targets available here. Return:
- open_docs: indices of articles to open now. Only when articles are listed,
  at most the remaining budget. Otherwise [].
- covered_needs: indices of the needs that the articles opened so far
  (including the ones you open now) cover.
- new_needs: at most ONE new information need discovered from the evidence
  notes, as {need: 3-12 words, fact: the [F#] index it comes from}; [] if
  none, if no evidence notes are shown, or if a need was already added.
- action: "descend" into a child (target = child index), "jump" to a jump
  target (target = its index), "up" to the parent (to reach the branch of an
  uncovered need), or "stop" (every need is covered and no other plausible
  article remains, or nothing in this tree fits the question).
- target: the child or jump-target index for descend/jump; null otherwise.
- reason: one short sentence.

Rules: pick the child whose summary or typical questions match the
question's product AND task, not just the product; while a need is uncovered
and budget remains, keep navigating - take the jump target that points to
that need's topic if there is one, otherwise go up to the ancestor whose
other children cover it; a child marked VISITED was already explored - the
notebook says what happened there - so do not go back into it unless you now
want a different article from it; never repeat a move the notebook shows led
nowhere.
"""

NAV_PROMPT_NO_EVIDENCE = """
You navigate a tree of Wix help-center articles to find the article(s) that
answer a user's question. Internal nodes group related articles and carry a
routing summary; leaves list articles. JUMP TARGETS lead to a node in another
branch (precomputed knowledge-graph shortcuts).

You were given a plan: the INFORMATION NEEDS the answer must cover. Your goal
is to open an article for EVERY need - needs may sit in different branches -
within a budget of articles. Success is measured by whether the needed
articles are among the ones you opened, not by how few you opened: when
several articles could cover a need, open all plausible ones; if budget
remains at a leaf, also open the supporting article a support agent would
attach (the prerequisite, the related setting, the troubleshooting article).
Do not open articles that clearly do not fit. You never see article bodies;
judge from titles, one-line descriptions and key entities.

Each step shows the question, the needs and which are covered, what you
opened, your path, your NOTEBOOK (previous moves and reasons), the current
node, its children (or its articles if it is a leaf) and the jump targets
available here. Return:
- open_docs: indices of articles to open now. Only when articles are listed,
  at most the remaining budget. Otherwise [].
- covered_needs: indices of the needs that the articles opened so far
  (including the ones you open now) cover.
- action: "descend" into a child (target = child index), "jump" to a jump
  target (target = its index), "up" to the parent (to reach the branch of an
  uncovered need), or "stop" (every need is covered and no other plausible
  article remains, or nothing in this tree fits the question).
- target: the child or jump-target index for descend/jump; null otherwise.
- reason: one short sentence.

Rules: pick the child whose summary or typical questions match the
question's product AND task, not just the product; while a need is uncovered
and budget remains, keep navigating - take the jump target that points to
that need's topic if there is one, otherwise go up to the ancestor whose
other children cover it; a child marked VISITED was already explored - the
notebook says what happened there - so do not go back into it unless you now
want a different article from it; never repeat a move the notebook shows led
nowhere.
"""


# ------------------------------------------------------------
# helpers
# ------------------------------------------------------------

def first_sentence(text, limit):
    text = " ".join(text.split())
    cut = text.find(". ")
    return (text[:cut + 1] if 0 < cut < limit else text[:limit]).rstrip()


def shortcut_label(sc):
    entities, relations = [], []
    for sup in sc.get("support", [])[:3]:
        for ev in sup.get("evidence", [])[:2]:
            names = [ev["via"]] if ev.get("via") else ev.get("shared_entities", [])
            for e in names:
                if e not in entities:
                    entities.append(e)
            if ev.get("predicate"):
                relations.append(f"{ev['subject']} {ev['predicate']} {ev['object']}")
            for r in ev.get("relations", [])[:1]:
                relations.append(f"{r['subject']} {r['predicate']} {r['object']}")
    relations = list(dict.fromkeys(relations))[:2]
    label = "linked via: " + (", ".join(entities[:4]) if entities else "document-graph edge")
    if relations:
        label += " (" + "; ".join(relations) + ")"
    return label


class EvidenceIndex:
    """KG dependency facts per article and, per specific entity, the other articles that cover it."""

    def __init__(self, doc_ids):
        records = read_jsonl(CANON_PATH)
        if not records:
            raise FileNotFoundError(f"Missing canonical KG: {CANON_PATH} (run canonicalize_entities.py)")
        docset = set(doc_ids)
        self.relations = {}                     # doc -> dependency relations (ranked)
        central = defaultdict(Counter)          # entity -> {doc: #relations involving it}
        presence = defaultdict(set)             # entity -> docs where it appears at all (entity list or relation)
        df = Counter()
        types = defaultdict(Counter)
        for r in records:
            d = r["document_id"]
            if d not in docset:
                continue
            names = set()
            for e in r["entities"]:
                names.add(e["name"])
                types[e["name"]][e["type"]] += 1
            deps = []
            for x in r["relations"]:
                names.update((x["subject"], x["object"]))
                central[x["subject"]][d] += 1
                central[x["object"]][d] += 1
                if x.get("subject_type"):
                    types[x["subject"]][x["subject_type"]] += 1
                if x.get("object_type"):
                    types[x["object"]][x["object_type"]] += 1
                if x["predicate"] in DEP_PREDICATES:
                    deps.append(x)
            for n in names:
                df[n] += 1
                presence[n].add(d)
            deps.sort(key=lambda x: -(DEP_PREDICATES[x["predicate"]] * (1.0 if x.get("certainty") == "DEFINITE" else 0.7)))
            self.relations[d] = deps
        n = len(docset)
        self.specific = {e for e, c in df.items()
                         if 2 <= c <= EVIDENCE_MAX_DF_RATIO * n and e.lower() not in GENERIC_NAMES
                         and (types[e].most_common(1)[0][0] != "PRODUCT" if types[e] else True)}
        self.central = central
        self.presence = presence
        self.df = df

    def notes_for(self, doc_id, exclude):
        """[(relation, entity, [related doc ids])] for an opened article; related docs are other
        articles where the same specific entity appears, ranked by how central it is there."""
        out, listed, used_entities = [], set(), set()
        for x in self.relations.get(doc_id, []):
            picked_entity, picked = None, []
            # the more specific endpoint first; an entity already used for an earlier fact of this article is
            # skipped (its further leads are lower-ranked and noisier); only articles not already listed
            for e in sorted({x["subject"], x["object"]}, key=lambda n: self.df.get(n, 0)):
                if e not in self.specific or e in used_entities:
                    continue
                # every article where the entity appears, the ones where it is a relation endpoint first
                candidates = sorted(self.presence[e], key=lambda d: (-self.central[e].get(d, 0), d))
                for d in candidates:
                    if d != doc_id and d not in exclude and d not in listed and d not in picked:
                        picked.append(d)
                        picked_entity = picked_entity or e
                    if len(picked) >= DOCS_PER_NOTE:
                        break
                if picked:
                    break
            if picked:
                listed.update(picked)
                used_entities.add(picked_entity)
                out.append((x, picked_entity, picked))
            if len(out) >= NOTES_PER_DOC:
                break
        return out


class Navigator:
    def __init__(self, tree_method, policy, budget, max_calls, client=None, use_cache=True, evidence=True):
        data = load_summaries(tree_method)
        self.tree_method = tree_method
        self.tree = data["tree"]
        self.nodes = self.tree["by_id"]
        self.root = self.tree["root_id"]
        self.leaf_of = self.tree["doc_to_leaf"]
        self.summaries = data["summaries"]
        self.doc_rows = data["doc_rows"]
        self.docsets = {n: set(v["doc_ids"]) for n, v in self.nodes.items()}
        self.policy = policy
        self.shortcuts = load_shortcuts(tree_method, policy) if policy != "none" else {}
        self.evidence = EvidenceIndex(list(self.doc_rows)) if evidence else None
        self.budget = budget
        self.max_calls = max_calls
        self.client = client
        self.use_cache = use_cache
        self.cache = {}
        self.lock = threading.Lock()
        if use_cache:
            for r in read_jsonl(CACHE_PATH):
                self.cache[r["key"]] = r
        self.decision_cls = NavDecision if evidence else NavDecisionNoEvidence
        self.nav_prompt = NAV_PROMPT if evidence else NAV_PROMPT_NO_EVIDENCE
        self.decision_schema = json.dumps(self.decision_cls.model_json_schema(), sort_keys=True)
        self.plan_schema = json.dumps(NavPlan.model_json_schema(), sort_keys=True)

    # ---- views -------------------------------------------------------
    def tree_path(self, node_id):
        out = []
        while node_id is not None:
            out.append(node_id)
            node_id = self.nodes[node_id]["parent_id"]
        return out[::-1]

    def evidence_block(self, opened):
        """(lines, evidence links, facts) - links: [(leaf_id, [doc ids], why)] in priority order."""
        if not self.evidence or not opened:
            return [], [], []
        lines = ["EVIDENCE NOTES (knowledge-graph facts from the articles you opened; candidate leads, not verified answers):"]
        links = {}                                       # leaf -> {"docs": set, "why": str}
        order, facts = [], []
        for d in reversed(opened):                       # most recently opened article first
            if len(facts) >= MAX_FACTS_SHOWN:
                break
            notes = self.evidence.notes_for(d, exclude=set(opened))[:MAX_FACTS_SHOWN - len(facts)]
            if not notes:
                continue
            lines.append(f"  From '{self.doc_rows[d]['title']}':")
            for x, entity, related in notes:
                fact = f"{x['subject']} {x['predicate']} {x['object']}"
                if x.get("certainty") == "POSSIBLE":
                    fact += " (possible)"
                facts.append(fact)
                refs = []
                for rd in related:
                    leaf = self.leaf_of.get(rd)
                    if leaf is None:
                        continue
                    if leaf not in links:
                        links[leaf] = {"docs": [], "why": f"'{entity}': {fact}"}
                        order.append(leaf)
                    if rd not in links[leaf]["docs"]:
                        links[leaf]["docs"].append(rd)
                    refs.append(f"'{self.doc_rows[rd]['title']}'")
                lines.append(f"    [F{len(facts)}] {fact}" + (f"  (also covered in: {', '.join(refs)})" if refs else ""))
        if len(lines) == 1:
            return [], [], []
        lines.append("")
        return lines, [(leaf, links[leaf]["docs"], links[leaf]["why"]) for leaf in order[:MAX_EVIDENCE_LINKS]], facts

    def render(self, question, needs, covered, node_id, opened, path, visits, notebook=(), exhausted=frozenset()):
        s = self.summaries[node_id]
        node = self.nodes[node_id]
        lines = [f"QUESTION: {question}", ""]
        lines.append("INFORMATION NEEDS (your plan" + ("; needs added from evidence are marked +" if self.evidence else "") + "):")
        for i, need in enumerate(needs, 1):
            lines.append(f"  [{i}] {need}   - {'COVERED' if i in covered else 'not yet covered'}")
        lines.append("")
        if opened:
            lines.append(f"OPENED SO FAR ({len(opened)} of {self.budget}):")
            lines += [f"  - {self.doc_rows[d]['title']}" for d in opened]
        else:
            lines.append(f"OPENED SO FAR (0 of {self.budget}): none")
        lines.append(f"REMAINING BUDGET: {self.budget - len(opened)} article(s)")
        lines.append("")
        ev_lines, ev_links, facts = self.evidence_block(opened)
        lines += ev_lines
        lines.append("PATH: " + " > ".join(self.summaries[p]["title"] for p in path) + "  [current]")
        lines.append("")
        if notebook:
            lines.append("NOTEBOOK (your previous moves):")
            lines += [f"  {entry}" for entry in notebook[-NOTEBOOK_STEPS:]]
            lines.append("")
        lines.append(f"CURRENT NODE [depth {node['depth']}, {node['size']} articles]: {s['title']}")
        lines.append(s["summary"])
        lines.append("")
        children, docs = [], []
        if node["is_leaf"]:
            lines.append("ARTICLES IN THIS LEAF:")
            for i, d in enumerate(node["doc_ids"], 1):
                row = self.doc_rows[d]
                flag = "   [OPENED]" if d in opened else ""
                ents = ", ".join(row.get("key_entities", [])[:5])
                lines.append(f"[{i}] {row['title']} - {row['one_line']}" + (f" (entities: {ents})" if ents else "") + flag)
                docs.append(d)
        else:
            lines.append("CHILDREN:")
            for i, c in enumerate(node["children"], 1):
                cs = self.summaries[c]
                flags = []
                if visits.get(c):
                    n_in = sum(1 for d in opened if d in self.docsets[c])
                    flags.append(f"VISITED, {n_in} article(s) opened there" if n_in else "VISITED, nothing opened there")
                if (node_id, c) in exhausted:
                    flags.append("EXHAUSTED - this move was refused, do not choose it again")
                flag = f"   [{'; '.join(flags)}]" if flags else ""
                lines.append(f"[{i}] {cs['title']} ({cs['size']} articles){flag}")
                lines.append(f"    {cs['summary']}")
                if cs.get("question_types"):
                    lines.append("    typical questions: " + " | ".join(cs["question_types"][:CHILD_QUESTIONS]))
                if cs.get("key_terms"):
                    lines.append("    key terms: " + ", ".join(cs["key_terms"][:CHILD_TERMS]))
                children.append(c)
        # jump targets: precomputed node shortcuts of the policy, then evidence links
        targets = []                                     # [(node_id, kind)]
        tlines = []
        for sc in self.shortcuts.get(node_id, []):
            t = sc["target"]
            if t not in self.summaries:
                continue
            ts = self.summaries[t]
            flags = (["VISITED"] if visits.get(t) else []) + (["EXHAUSTED"] if (node_id, t) in exhausted else [])
            flag = f"   [{'; '.join(flags)}]" if flags else ""
            tlines.append(f"[{len(targets) + 1}] -> {ts['title']} (depth {ts['depth']}, {ts['size']} articles){flag}  - KG shortcut")
            tlines.append(f"    {first_sentence(ts['summary'], SHORTCUT_SUMMARY_CHARS)}")
            tlines.append(f"    {shortcut_label(sc)}")
            targets.append((t, "shortcut"))
        for leaf, rel_docs, why in ev_links:
            if leaf == node_id:
                continue
            ts = self.summaries[leaf]
            flags = (["VISITED"] if visits.get(leaf) else []) + (["EXHAUSTED"] if (node_id, leaf) in exhausted else [])
            flag = f"   [{'; '.join(flags)}]" if flags else ""
            titles = "; ".join(f"'{self.doc_rows[d]['title']}'" for d in rel_docs[:DOCS_PER_NOTE])
            tlines.append(f"[{len(targets) + 1}] -> {ts['title']} (depth {ts['depth']}, {ts['size']} articles){flag}  - evidence link")
            tlines.append(f"    holds {titles}")
            tlines.append(f"    because of {why}")
            targets.append((leaf, "evidence"))
        if tlines:
            lines.append("")
            lines.append("JUMP TARGETS (other branches you can jump to from here):")
            lines += tlines
        lines.append("")
        lines.append("Decide: open_docs, covered_needs, " + ("new_needs, " if self.evidence else "")
                     + "action, target, reason.")
        return "\n".join(lines), children, docs, targets, facts

    # ---- llm ---------------------------------------------------------
    def _call(self, system_prompt, user_prompt, schema_cls, schema_json):
        key = sha256_text("\x00".join([MODEL_NAME, REASONING_EFFORT, schema_json, system_prompt, user_prompt]))
        if self.use_cache and key in self.cache:
            r = self.cache[key]
            return schema_cls(**r["decision"]), r["usage"], True
        if self.client is None:
            raise RuntimeError("LLM client not initialised (dry run?)")
        request = {
            "model": MODEL_NAME,
            "messages": [{"role": "system", "content": system_prompt},
                         {"role": "user", "content": user_prompt}],
            "response_format": schema_cls,
        }
        if IS_REASONING_MODEL:
            request["reasoning_effort"] = REASONING_EFFORT
            request["max_completion_tokens"] = MAX_OUTPUT_TOKENS
        response = self.client.chat.completions.parse(**request)
        choice = response.choices[0]
        if choice.finish_reason == "length" or choice.message.parsed is None:
            raise RuntimeError("No structured output returned.")
        u = response.usage
        details = getattr(u, "completion_tokens_details", None)
        usage = {"input_tokens": u.prompt_tokens, "output_tokens": u.completion_tokens,
                 "reasoning_tokens": getattr(details, "reasoning_tokens", 0) or 0}
        parsed = choice.message.parsed
        record = {"key": key, "model": MODEL_NAME, "reasoning": REASONING_EFFORT,
                  "decision": parsed.model_dump(), "usage": usage}
        if self.use_cache:
            with self.lock:
                self.cache[key] = record
            append_jsonl(CACHE_PATH, record)
        return parsed, usage, False

    def plan(self, question):
        parsed, usage, cached = self._call(PLAN_PROMPT, f"QUESTION: {question}", NavPlan, self.plan_schema)
        needs = [n.strip() for n in parsed.needs if n.strip()][:3] or [question]
        return needs, usage, cached

    def decide(self, user_prompt):
        return self._call(self.nav_prompt, user_prompt, self.decision_cls, self.decision_schema)

    # ---- navigation --------------------------------------------------
    def navigate(self, q):
        usage_total = {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}   # logical cost
        api_usage = {"input_tokens": 0, "output_tokens": 0}                           # billed in this run
        api_calls = 0

        def account(usage, cached):
            nonlocal api_calls
            for k in usage_total:
                usage_total[k] += usage.get(k, 0)
            if not cached:
                api_calls += 1
                for k in api_usage:
                    api_usage[k] += usage.get(k, 0)

        needs, usage, cached = self.plan(q["question"])
        account(usage, cached)
        calls = 1
        n_planned = len(needs)
        covered = set()

        node, opened, path = self.root, [], [self.root]
        visits = {self.root: 1}
        summaries_read, doc_rows_read = {self.root}, set()
        trace, invalid, ups, stop_refusals, osc_refusals = [], 0, 0, 0, 0
        jumps = Counter()
        notebook = []
        last_up_from, opened_at_last_up = None, 0
        exhausted = set()          # (node, target) moves refused -> flagged EXHAUSTED in later prompts
        moves_made = set()         # (node, target, n_opened) forward moves already made
        stop_reason = "max_calls"
        while calls < self.max_calls:
            prompt, children, docs, targets, facts = self.render(q["question"], needs, covered, node, opened, path,
                                                                 visits, notebook, exhausted)
            summaries_read.update(children)
            summaries_read.update(t for t, _ in targets)
            doc_rows_read.update(docs)
            decision, usage, cached = self.decide(prompt)   # an API error propagates: the question is left
            calls += 1                                         # out of the results and rerun later (cache)
            account(usage, cached)
            step = {"node": node, "title": self.summaries[node]["title"], "depth": self.nodes[node]["depth"],
                    "action": decision.action, "target": decision.target, "reason": decision.reason,
                    "cached": cached, "opened_here": [], "opened_here_titles": [], "covered_needs": [],
                    "new_needs": []}
            if docs:
                for i in decision.open_docs:
                    if 1 <= i <= len(docs) and docs[i - 1] not in opened and len(opened) < self.budget:
                        opened.append(docs[i - 1])
                        step["opened_here"].append(docs[i - 1])
                        step["opened_here_titles"].append(self.doc_rows[docs[i - 1]]["title"])
            if opened:
                for i in decision.covered_needs:
                    if 1 <= i <= len(needs) and i not in covered:
                        covered.add(i)                       # the navigator's claim, not verified coverage
                        step["covered_needs"].append(i)
                        step.setdefault("coverage_claims", []).append(
                            {"need": needs[i - 1], "opened_at_claim": [self.doc_rows[d]["title"] for d in opened]})
                # needs discovered from evidence: accepted only when grounded in a fact shown in THIS prompt
                for nn in (getattr(decision, "new_needs", None) or [])[:2]:
                    text = nn.need.strip()
                    known = {n.removeprefix("+ ").lower() for n in needs}
                    if text and 1 <= nn.fact <= len(facts) and len(needs) < MAX_NEEDS \
                            and len(needs) - n_planned < MAX_EVIDENCE_NEEDS and text.lower() not in known:
                        needs.append("+ " + text)
                        step["new_needs"].append({"need": text, "fact": facts[nn.fact - 1]})
                    elif text:
                        step.setdefault("rejected_new_needs", []).append({"need": text, "fact": nn.fact})
            if len(opened) >= self.budget:
                trace.append(step)
                stop_reason = "budget"
                break
            # read-revise step: articles were just opened, so their evidence notes are not in this
            # prompt yet. Stay on the node and decide the next move with the evidence in view
            # (one extra call per opening step; bounded because `opened` only grows).
            if step["opened_here"] and self.evidence is not None:
                step["note"] = "opened articles -> review evidence before moving"
                step["action_before_review"] = decision.action
                step["action"] = "review"
                entry = (f"{len(notebook) + 1}. at '{self.summaries[node]['title']}': opened "
                         + "; ".join(step["opened_here_titles"]) + " | reviewing their evidence notes before the next move")
                notebook.append(entry)
                trace.append(step)
                continue
            action, target = decision.action, decision.target
            target_title, kind = None, None
            if action == "descend" and children and target and 1 <= target <= len(children):
                target_title = self.summaries[children[target - 1]]["title"]
            elif action == "jump" and targets and target and 1 <= target <= len(targets):
                target_title = self.summaries[targets[target - 1][0]]["title"]
                kind = targets[target - 1][1]
            entry = f"{len(notebook) + 1}. at '{self.summaries[node]['title']}': {action}"
            entry += f" -> '{target_title}'" + (f" ({kind} link)" if kind else "") if target_title else ""
            if step["opened_here_titles"]:
                entry += "; opened: " + "; ".join(step["opened_here_titles"])
            if step["new_needs"]:
                entry += "; added needs: " + "; ".join(n["need"] for n in step["new_needs"])
            entry += f" | {decision.reason.strip()}"
            uncovered = [i for i in range(1, len(needs) + 1) if i not in covered]
            if action == "stop" and uncovered and stop_refusals < MAX_STOP_REFUSALS \
                    and self.nodes[node]["parent_id"] is not None:
                stop_refusals += 1
                action = "up"
                step["note"] = f"stop refused (need {uncovered} uncovered, budget left) -> up"
                entry += f" [stop refused: need {uncovered} still uncovered - search elsewhere]"
            # redundant-move guard: re-entering the child just left with nothing new opened since, or
            # repeating a move already refused or already made from this node at the same progress.
            # Refused moves are flagged EXHAUSTED in the next prompt; after MAX_MOVE_REFUSALS the run ends.
            move_target = None
            if action == "descend" and children and target and 1 <= target <= len(children):
                move_target = children[target - 1]
            elif action == "jump" and targets and target and 1 <= target <= len(targets):
                move_target = targets[target - 1][0]
            if move_target is not None:
                redundant = ((action == "descend" and move_target == last_up_from and len(opened) == opened_at_last_up)
                             or (node, move_target) in exhausted
                             or (node, move_target, len(opened)) in moves_made)
                if redundant:
                    exhausted.add((node, move_target))
                    if osc_refusals < MAX_MOVE_REFUSALS:
                        osc_refusals += 1
                        step["note"] = "redundant move (route already tried with nothing new) -> refused"
                        step["effective_action"] = "refused"
                        entry += " [move refused: that route was already tried with nothing new - choose another child, go up, or stop]"
                        notebook.append(entry)
                        trace.append(step)
                        continue
                    step["note"] = "redundant move repeated -> stop"
                    step["effective_action"] = "stop"
                    notebook.append(entry)
                    trace.append(step)
                    stop_reason = "oscillation"
                    break
                moves_made.add((node, move_target, len(opened)))
            step["effective_action"] = action            # the move actually applied (stop may have become up)
            notebook.append(entry)
            next_node = None
            if action == "descend" and children and target and 1 <= target <= len(children):
                next_node = children[target - 1]
            elif action == "jump" and targets and target and 1 <= target <= len(targets):
                next_node, kind = targets[target - 1]
                jumps[kind] += 1
                step["jump_kind"] = kind
            elif action == "up":
                next_node = self.nodes[node]["parent_id"]
                ups += 1
                last_up_from, opened_at_last_up = node, len(opened)
                if next_node is None:
                    step["note"] = (step.get("note", "") + " up at root -> stop").strip()
                    trace.append(step)
                    stop_reason = "stop_at_root"
                    break
            elif action == "stop":
                trace.append(step)
                stop_reason = "stop" if not uncovered else "stop_uncovered"
                break
            if next_node is None:
                invalid += 1
                step["note"] = "invalid action -> up"
                next_node = self.nodes[node]["parent_id"]
                if next_node is None:
                    trace.append(step)
                    stop_reason = "invalid_at_root"
                    break
            trace.append(step)
            visits[next_node] = visits.get(next_node, 0) + 1
            if visits[next_node] > MAX_VISITS:
                stop_reason = "loop"
                break
            node = next_node
            path = self.tree_path(node)
            summaries_read.add(node)
        return {
            "qid": q["qid"], "question": q["question"], "gold": q["gold"], "n_gold": q["n_gold"],
            "opened": opened,
            "needs": needs, "n_planned_needs": n_planned, "covered_needs": sorted(covered),
            "summaries_read": len(summaries_read), "doc_rows_read": len(doc_rows_read),
            "summary_tokens_read": sum(num_tokens(self.summaries[n]["summary"]) for n in summaries_read),
            "llm_calls": calls, **usage_total,
            "api_calls_this_run": api_calls, "api_input_tokens_this_run": api_usage["input_tokens"],
            "api_output_tokens_this_run": api_usage["output_tokens"],
            "jumps": sum(jumps.values()), "jumps_by_kind": dict(jumps), "ups": ups,
            "invalid_actions": invalid, "stop_refusals": stop_refusals, "move_refusals": osc_refusals,
            "stop_reason": stop_reason, "trace": trace, "notebook": notebook,
        }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", choices=list(TREES), default="leiden")
    parser.add_argument("--policy", default="none", help="comma list of " + ",".join(POLICIES) + " or 'all'")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="first N questions (pilot)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-evidence", action="store_true", help="ablation: no evidence notes/links")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts for the first question, no API")
    args = parser.parse_args()

    if args.split == "heldout":
        print("WARNING: held-out split is for the final run only; do not tune on it.")
    policies = list(POLICIES) if args.policy == "all" else args.policy.split(",")
    for p in policies:
        if p not in POLICIES:
            parser.error(f"unknown policy {p}")
    load_corpus()
    questions = load_questions(args.split)
    if args.limit:
        questions = questions[:args.limit]
    variant = VARIANT + ("noev" if args.no_evidence else "")

    if args.dry_run:
        nav = Navigator(args.tree, policies[0], args.budget, args.max_calls, evidence=not args.no_evidence)
        q = questions[0]
        print("=" * 70 + "\n[PLAN call]\n" + "=" * 70)
        print(PLAN_PROMPT.strip() + f"\n\nQUESTION: {q['question']}")
        needs = ["(information need 1 - filled by the plan call)"]
        print("\n" + "=" * 70 + "\n[root prompt]\n" + "=" * 70)
        prompt, children, _, _, _ = nav.render(q["question"], needs, set(), nav.root, [], [nav.root], {nav.root: 1})
        print(prompt)
        print(f"\n[root prompt ~{num_tokens(prompt)} tokens | system prompt ~{num_tokens(nav.nav_prompt)} tokens]")
        # a leaf view after opening one of the question's gold articles, to show evidence notes
        if nav.evidence and q["gold"][0] in nav.leaf_of:
            g = q["gold"][0]
            leaf = nav.leaf_of[g]
            path = nav.tree_path(leaf)
            prompt, _, _, targets, _ = nav.render(q["question"], needs, {1}, leaf, [g], path,
                                                  {n: 1 for n in path}, ["1. ... (previous moves)"])
            print("\n" + "=" * 70 + f"\n[leaf view after opening the gold article '{nav.doc_rows[g]['title']}'; "
                  f"{len(targets)} jump targets]\n" + "=" * 70)
            print(prompt)
            print(f"\n[leaf prompt ~{num_tokens(prompt)} tokens]")
        print(f"\nmodel {MODEL_NAME} (reasoning {REASONING_EFFORT}) | budget {args.budget} | max calls {args.max_calls} "
              f"| questions {len(questions)} | policies {policies} | tree {args.tree} | variant {variant}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Missing OPENAI_API_KEY in .env")
    from openai import OpenAI
    client = OpenAI(timeout=180.0, max_retries=3)

    for policy in policies:
        nav = Navigator(args.tree, policy, args.budget, args.max_calls, client=client,
                        use_cache=not args.no_cache, evidence=not args.no_evidence)
        system = f"nav_{variant}_{policy}_B{args.budget}"
        print(f"\n========== NAVIGATOR {variant} policy={policy} tree={args.tree} budget={args.budget} "
              f"model={MODEL_NAME}/{REASONING_EFFORT} | {len(questions)} questions ==========")
        records = {}
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(nav.navigate, q): q for q in questions}
            for fut in as_completed(futures):
                q = futures[fut]
                try:
                    r = fut.result()
                except Exception as error:
                    print(f"  FAILED {q['qid']}: {type(error).__name__}: {error}")
                    continue
                records[q["qid"]] = r
                gold = set(r["gold"])
                got = len(gold & set(r["opened"]))
                cached = sum(1 for s in r["trace"] if s.get("cached"))
                jk = r["jumps_by_kind"]
                print(f"  {r['qid']} gold {got}/{len(gold)} | needs {r['n_planned_needs']}+{len(r['needs']) - r['n_planned_needs']} "
                      f"covered {len(r['covered_needs'])} | opened {len(r['opened'])} | calls {r['llm_calls']} (cached {cached}) "
                      f"| jumps sc{jk.get('shortcut', 0)} ev{jk.get('evidence', 0)} ups {r['ups']} refused {r['stop_refusals']} "
                      f"| {r['stop_reason']:14s} | {r['question'][:50]}")
        ordered = [records[q["qid"]] for q in questions if q["qid"] in records]
        if not ordered:
            print("no results")
            continue
        config = {"model": MODEL_NAME, "reasoning_effort": REASONING_EFFORT, "budget": args.budget,
                  "max_calls": args.max_calls, "policy": policy, "variant": variant, "evidence": not args.no_evidence,
                  "schema_version": SCHEMA_VERSION, "prompt_sha256": sha256_text(nav.nav_prompt + PLAN_PROMPT),
                  "limit": args.limit}
        path = save_results(system, args.tree, args.split, config, ordered)
        tokens = {k: sum(r[k] for r in ordered) for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
        billed = {"api_calls": sum(r["api_calls_this_run"] for r in ordered),
                  "input_tokens": sum(r["api_input_tokens_this_run"] for r in ordered),
                  "output_tokens": sum(r["api_output_tokens_this_run"] for r in ordered)}
        stops = Counter(r["stop_reason"] for r in ordered)
        planned = Counter(r["n_planned_needs"] for r in ordered)
        added = sum(len(r["needs"]) - r["n_planned_needs"] for r in ordered)
        jk = Counter()
        for r in ordered:
            jk.update(r["jumps_by_kind"])
        print(quick_report(system, ordered, budgets=(args.budget,)))
        print(f"  planned needs per question {dict(sorted(planned.items()))} | needs added from evidence {added} "
              f"| stop refusals {sum(r['stop_refusals'] for r in ordered)} | jumps {dict(jk)}")
        print(f"  logical cost {tokens} | billed this run {billed} | stop reasons {dict(stops)}")
        print(f"  saved {path}")
        if len(ordered) < len(questions):
            print(f"  {len(questions) - len(ordered)} questions failed; rerun (cached decisions are free)")


if __name__ == "__main__":
    main()