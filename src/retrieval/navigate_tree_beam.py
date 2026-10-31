# src/retrieval/navigate_tree_beam.py
"""
Graph-guided hierarchical retrieval - the online navigator (step 7), v4 + BEAM SEARCH.

This is navigate_tree.py (v4) with one addition, as in RAPTOR tree traversal:
the root-to-leaf descent is a beam search. Each level is ONE LLM call that
rates every candidate 0-100 jointly (the children of the frontier nodes, their
KG shortcut targets for a policy other than none, and the leaves kept from the
previous level); the best W candidates (score >= MIN_CANDIDATE_SCORE) form the
next frontier until the frontier holds only leaves. The W leaves are the beam's
branches, in rating order. The v4 agent then works through them, unchanged
inside a branch: it opens articles, reviews their KG evidence, follows
evidence links and shortcuts, goes up. Between branches (variant v4c):
  * a "stop" ends the current branch, not the search: while budget remains
    and an unvisited beam branch is left, the agent continues at the next one
    (the notebook tells it why it is there); the search ends when the budget
    is spent, the branches are exhausted, or the agent stops at the last one;
  * a stop while a need is uncovered is refused as in v4, but the agent is
    sent to the next beam branch first and goes up only when none is left;
  * the other branches are also jump targets ("beam leaf") in every prompt;
  * a loop or oscillation in one branch moves the agent to the next branch.
W=1 is v4 with a scored (instead of chosen) descent and no second branch; the
earlier v4b variant (branches as passive jump targets only) is identical to
v4c at W=1.

--router chooses who rates the candidates in the descent:
  llm     one joint LLM rating call per level (default)
  cosine  RAPTOR's tree traversal, unchanged: cosine similarity between the
          question and the node's routing text (title, summary, typical
          questions, key terms; text-embedding-3-small, the vectors of
          baselines.py), tree children only, top-W per level, ties by node id,
          no LLM call. The descent is identical to raptor_traversal_bW for every
          policy (KG shortcuts are not descent candidates; with --policy typed
          they are offered to the agent as jump targets only), so RAPTOR and
          this navigator differ only in the leaf stage: cosine ranking of the
          pooled articles vs the KG-guided agent
  hybrid  W-1 slots by the LLM rating, the last slot for the best cosine
          candidate not already selected (W=1: LLM only); no weights to tune
The agent stage is the v4c agent for every router, so routers are compared
with everything else fixed. Results: nav_v4c[_cos|_hyb][noev]_W<W>_<policy>_B<B>.

End-to-end method (offline steps are the earlier scripts):
  corpus -> KG extraction (extract_relations_v5) -> entity canonicalisation
  -> document graph (build_doc_graph) -> recursive Leiden tree (build_tree_v2)
  -> routing summaries (summarize_tree_v2) -> cross-branch shortcuts
  (select_shortcuts) -> THIS FILE: an LLM agent that navigates the tree to
  collect the articles a question needs, revising its plan from KG evidence.

One question, one run:
  1. PLAN      one LLM call lists the question's information needs (1-3).
  1b. BEAM     from the root, one LLM call per level rates the candidate nodes;
               the W best are kept; ends at W leaves (the rating prompt is the
               one of the pure beam navigator, so its cached decisions are reused).
  2. NAVIGATE  from the best beam leaf, then the next ones while budget remains,
               one LLM call per step. The prompt shows the
               question, the needs checklist, opened articles, EVIDENCE NOTES,
               path, NOTEBOOK (previous moves + reasons), the current node with
               its children (or its articles at a leaf) and JUMP TARGETS
               (precomputed KG shortcuts of the chosen policy, the other beam
               leaves, and evidence links).
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
new_needs). Results: eval/results/nav_v4c[noev]_W<W>_<policy>_B<budget>__<tree>__<split>.json
Every LLM decision is cached (eval/cache/navigator_cache.jsonl) by full prompt:
re-runs are free and replay exactly; an API error leaves the question out of
the results file (rerun to fill it).

Usage
-----
    python src/retrieval/navigate_tree_beam.py --dry-run --widths 2                   # prompts, no API
    python src/retrieval/navigate_tree_beam.py --widths 2 --policy typed --limit 10    # pilot
    python src/retrieval/navigate_tree_beam.py --widths 1,2,3 --policy none,typed      # width sweep x shortcut ablation
    python src/retrieval/navigate_tree_beam.py --widths 1,2,3 --policy none,typed --router cosine   # RAPTOR descent + v4 agent
    python src/retrieval/navigate_tree_beam.py --widths 2,3 --policy none,typed --router hybrid
    python src/retrieval/navigate_tree_beam.py --widths 2 --policy none,typed --no-evidence
    python src/retrieval/evaluate_v2.py --strict --ref dense --only "bm25,dense,raptor_*,nav_v4_*,nav_v4c_*"

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
from retrieval.common import (CACHE_DIR, DATA_DIR, PROJECT_ROOT, SHORTCUT_POLICIES, TREES, Embedder,  # noqa: E402
                              append_jsonl, load_corpus, load_questions, load_shortcuts, load_summaries,
                              node_text, normalize_rows, num_tokens, quick_report, read_jsonl, save_results,
                              sha256_text)

load_dotenv(PROJECT_ROOT / ".env")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-6-luna").strip()
REASONING_EFFORT = os.getenv("LLM_NAV_REASONING_EFFORT",
                             os.getenv("LLM_REASONING_EFFORT", "medium")).strip().lower()
IS_REASONING_MODEL = MODEL_NAME.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
CANON_PATH = DATA_DIR / "kg_v5_gpt-6-luna_medium_canonical.jsonl"

VARIANT = "v4c"
SCHEMA_VERSION = f"navigator-{VARIANT}"
BUDGET = 5                     # articles the navigator may open
WIDTHS = (1, 2, 3)             # default beam widths of the sweep
MAX_ROUNDS = 8                 # beam levels per descent (tree depth <= 6, plus shortcut hops)
MIN_CANDIDATE_SCORE = 10       # a candidate rated below this is not kept even if a beam slot is free
MAX_CALLS = 24                 # LLM calls per question incl. plan, beam levels (~4) and the agent's steps in all branches
BRANCH_END_REASONS = ("stop", "stop_uncovered", "stop_at_root", "invalid_at_root", "loop", "oscillation")
                               # how a branch can end; with budget and an unvisited beam branch left the search continues
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
ROUTERS = ("llm", "cosine", "hybrid")
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


class CandidateScore(BaseModel):   # beam descent (same schema as the pure beam navigator -> shared cache)
    index: int
    score: int                     # 0-100


class ScoreDecision(BaseModel):
    scores: list[CandidateScore]
    reason: str


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

SCORE_PROMPT = """
You route a question from a Wix website owner through a tree of Wix
help-center articles. Internal nodes group related articles and carry a
routing summary (title, summary, typical questions, key terms); leaves list
articles. The search keeps several branches open at once (beam search): you
rate the candidate nodes, the best ones are expanded or kept for the next
round, the rest are dropped for good.

Rate EVERY candidate from 0 to 100: how likely is it that the candidate's
subtree holds an article the answer needs? Use the whole range:
  80-100  clearly covers the question's product AND task
  40-79   covers the product and a closely related task, or the task for a
          sibling product
  10-39   weakly related (the product only, or a distant task)
  0-9     unrelated
Judge product AND task, not the product alone. A question may need more than
one article (see INFORMATION NEEDS): a candidate that covers a different need
than the others - a prerequisite, a companion setting, the troubleshooting of
the same setting - is rated on its own merits, never lowered because another
candidate covers the main need. Candidates marked "KG shortcut" sit in another
branch and are linked through a knowledge-graph entity to a node of the
current path; candidates marked "kept" are leaves kept from the previous round;
rate both like any other candidate. Return a score for every index and one
short sentence of reasoning.
"""

NAV_PROMPT = """
You navigate a tree of Wix help-center articles to find the article(s) that
answer a user's question. Internal nodes group related articles and carry a
routing summary; leaves list articles. A beam search over the routing
summaries brought you to the leaf it rated best for the question; the other
leaves it kept are BEAM LEAVES. JUMP TARGETS lead to a node in another
branch: precomputed knowledge-graph shortcuts, the beam leaves, and EVIDENCE
LINKS derived from what you already opened.

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
that need's topic if there is one (a beam leaf or evidence link before going
up), otherwise go up to the ancestor whose other children cover it; a child
marked VISITED was already explored - the notebook says what happened there -
so do not go back into it unless you now want a different article from it;
never repeat a move the notebook shows led nowhere.
"""

NAV_PROMPT_NO_EVIDENCE = """
You navigate a tree of Wix help-center articles to find the article(s) that
answer a user's question. Internal nodes group related articles and carry a
routing summary; leaves list articles. A beam search over the routing
summaries brought you to the leaf it rated best for the question; the other
leaves it kept are BEAM LEAVES. JUMP TARGETS lead to a node in another
branch: precomputed knowledge-graph shortcuts and the beam leaves.

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
that need's topic if there is one (a beam leaf before going up), otherwise go
up to the ancestor whose other children cover it; a child marked VISITED was
already explored - the notebook says what happened there - so do not go back
into it unless you now want a different article from it; never repeat a move
the notebook shows led nowhere.
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
            for e in sorted({x["subject"], x["object"]}, key=lambda n: (self.df.get(n, 0), n)):   # name breaks ties
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
    def __init__(self, tree_method, policy, budget, max_calls, client=None, use_cache=True, evidence=True, width=1,
                 router="llm", embedder=None):
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
        self.score_schema = json.dumps(ScoreDecision.model_json_schema(), sort_keys=True)
        self.width = width
        self.router = router
        self.max_depth = max(v["depth"] for v in self.nodes.values())
        self.node_vec = None
        if router != "llm":                                  # embedding router: node routing texts, as in baselines
            self.embedder = embedder or Embedder()
            ids = sorted(self.summaries)
            vecs = normalize_rows(self.embedder.embed([node_text(self.summaries[n]) for n in ids], "(node summaries)"))
            self.node_vec = {n: vecs[i] for i, n in enumerate(ids)}

    def question_vec(self, question):
        return normalize_rows(self.embedder.embed([question], "(question)"))[0] if self.node_vec is not None else None

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

    def render(self, question, needs, covered, node_id, opened, path, visits, notebook=(), exhausted=frozenset(),
               beam_leaves=()):
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
        for leaf, score in beam_leaves:                   # the other leaves the beam search kept
            if leaf == node_id or any(t == leaf for t, _ in targets):   # already listed as a shortcut target
                continue
            ts = self.summaries[leaf]
            n_in = sum(1 for d in opened if d in self.docsets[leaf])
            flags = ([f"VISITED, {n_in} article(s) opened there" if n_in else "VISITED, nothing opened there"] if visits.get(leaf) else []) \
                + (["EXHAUSTED"] if (node_id, leaf) in exhausted else [])
            flag = f"   [{'; '.join(flags)}]" if flags else ""
            tlines.append(f"[{len(targets) + 1}] -> {ts['title']} (depth {ts['depth']}, {ts['size']} articles){flag}  - beam leaf, routing score {score}")
            tlines.append(f"    {first_sentence(ts['summary'], SHORTCUT_SUMMARY_CHARS)}")
            targets.append((leaf, "beam"))
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
                if key not in self.cache:
                    self.cache[key] = record
                    append_jsonl(CACHE_PATH, record)
        return parsed, usage, False

    def plan(self, question):
        parsed, usage, cached = self._call(PLAN_PROMPT, f"QUESTION: {question}", NavPlan, self.plan_schema)
        needs = [n.strip() for n in parsed.needs if n.strip()][:3] or [question]
        return needs, usage, cached

    def decide(self, user_prompt):
        return self._call(self.nav_prompt, user_prompt, self.decision_cls, self.decision_schema)

    # ---- beam descent (RAPTOR-style: rate every candidate per level, keep the best W) -------------
    def score_prompt(self, question, needs, covered, cands):
        """cands: [(node_id, kind, label)] - rendered exactly as in the pure beam navigator (shared cache)."""
        lines = [f"QUESTION: {question}", ""]
        lines.append("INFORMATION NEEDS (your plan" + ("; needs added from evidence are marked +" if self.evidence else "") + "):")
        for i, need in enumerate(needs, 1):
            lines.append(f"  [{i}] {need}   - {'COVERED' if i in covered else 'not yet covered'}")
        lines += ["", "CANDIDATES (rate every one):"]
        for i, (node, kind, label) in enumerate(cands, 1):
            s = self.summaries[node]
            what = "leaf" if self.nodes[node]["is_leaf"] else "node"
            lines.append(f"[{i}] {s['title']} ({what}, depth {s['depth']}, {s['size']} articles) - {label}")
            lines.append(f"    {s['summary']}")
            if s.get("question_types"):
                lines.append("    typical questions: " + " | ".join(s["question_types"][:CHILD_QUESTIONS]))
            if s.get("key_terms"):
                lines.append("    key terms: " + ", ".join(s["key_terms"][:CHILD_TERMS]))
        lines += ["", f"Return a score (0-100) for every index 1..{len(cands)}, and a short reason."]
        return "\n".join(lines)

    def beam_descend(self, question, needs, width, q_vec=None):
        """Returns ([(leaf, score)] best first, nodes whose summary was read, trace steps, [(usage, cached)]).
        Scores are 0-100: the LLM rating, or round(100 * cosine) for the cosine router."""
        kind_rank = {"retained": 0, "child": 1, "shortcut": 2}
        frontier = [(self.root, 100, "start")]
        scored, read, steps, usages = {self.root}, set(), [], []
        sc_candidates = sc_kept = 0
        descent_shortcuts = self.router != "cosine"         # cosine = RAPTOR's traversal: tree children only
        max_rounds = MAX_ROUNDS if self.router != "cosine" else max(MAX_ROUNDS, self.max_depth + 1)
        for rnd in range(1, max_rounds + 1):
            internal = [(n, s) for n, s, _ in frontier if not self.nodes[n]["is_leaf"]]
            if not internal:
                break
            retained = [(n, "retained", "kept from the previous round") for n, s, _ in frontier if self.nodes[n]["is_leaf"]]
            new, seen = [], {n for n, _, _ in retained}
            for n, _ in internal:
                title = self.summaries[n]["title"]
                for ch in self.nodes[n]["children"]:
                    if ch not in scored and ch not in seen:
                        seen.add(ch)
                        new.append((ch, "child", f"child of '{title}'"))
                for sc in (self.shortcuts.get(n, []) if descent_shortcuts else []):
                    t = sc["target"]
                    if t in self.summaries and t not in scored and t not in seen:
                        seen.add(t)
                        sc_candidates += 1
                        new.append((t, "shortcut", f"KG shortcut from '{title}' - {shortcut_label(sc)}"))
            if not new:
                frontier = [(n, s, k) for n, s, k in frontier if self.nodes[n]["is_leaf"]]
                break
            cands = retained + new
            read.update(n for n, _, _ in cands)
            cos = {n: float(q_vec @ self.node_vec[n]) for n, _, _ in cands} if q_vec is not None else {}
            if self.router == "cosine":                      # RAPTOR's router: no LLM call, nothing cached
                rated = [(n, max(0, int(round(100 * cos[n]))), k) for n, k, _ in cands]
                rated.sort(key=lambda x: (-cos[x[0]], x[0]))   # the tie-breaking of baselines.traverse
                kept = rated[:width]
                reason, cached = "cosine similarity between the question and the routing texts", None
            else:
                prompt = self.score_prompt(question, needs, set(), cands)
                decision, usage, cached = self._call(SCORE_PROMPT, prompt, ScoreDecision, self.score_schema)
                usages.append((usage, cached))
                given = {}
                for x in decision.scores:
                    if 1 <= x.index <= len(cands) and (x.index - 1) not in given:
                        given[x.index - 1] = max(0, min(100, x.score))
                rated = [(n, given.get(i, 0), k) for i, (n, k, _) in enumerate(cands)]
                rated.sort(key=lambda x: (-x[1], kind_rank[x[2]], x[0]))
                kept = [x for x in rated if x[1] >= MIN_CANDIDATE_SCORE][:width] or rated[:1]
                reason = decision.reason
                if self.router == "hybrid" and width > 1:    # W-1 LLM slots + the best cosine candidate not selected
                    llm_slots = [x for x in rated if x[1] >= MIN_CANDIDATE_SCORE][:width - 1] or rated[:1]
                    selected = {n for n, _, _ in llm_slots}
                    cosine_choice = next((x for x in sorted(rated, key=lambda x: (-cos[x[0]], x[0]))
                                          if x[0] not in selected), None)
                    kept = list(llm_slots) + ([cosine_choice] if cosine_choice is not None else [])
            scored.update(n for n, _, _ in new)
            sc_kept += sum(1 for _, _, k in kept if k == "shortcut")
            steps.append({"action": "beam", "round": rnd, "nodes": [n for n, _ in internal], "router": self.router,
                          "candidates": [{"node": n, "title": self.summaries[n]["title"], "kind": k, "score": s,
                                          **({"cos": round(cos[n], 4)} if cos else {})} for n, s, k in rated],
                          "kept": [n for n, _, _ in kept], "reason": reason, "cached": cached})
            frontier = kept
        leaves = [(n, s) for n, s, _ in frontier if self.nodes[n]["is_leaf"]]
        incomplete = any(not self.nodes[n]["is_leaf"] for n, _, _ in frontier)
        return leaves, read, steps, usages, {"candidates_scored": sc_candidates, "kept_in_beam": sc_kept,
                                             "descent_incomplete": incomplete}

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

        # beam descent: W leaves = the beam's branches, best first. The agent starts at the first one; the
        # others are jump targets in every prompt and are visited in turn while budget remains.
        beam_leaves, read, beam_steps, usages, beam_sc = self.beam_descend(q["question"], needs, self.width,
                                                                           self.question_vec(q["question"]))
        beam_incomplete = beam_sc.pop("descent_incomplete")   # round limit hit before all branches reached a leaf
        for usage, cached in usages:
            account(usage, cached)
        calls += len(usages)
        trace = list(beam_steps)
        if not beam_leaves:                                  # cannot happen with a non-empty tree; be safe
            beam_leaves = [(self.root, 0)]
        beam_score = dict(beam_leaves)
        node = beam_leaves[0][0]
        opened, path = [], self.tree_path(node)
        visits = {n: 1 for n in path}
        summaries_read, doc_rows_read = set(read) | set(path), set()
        invalid, ups, stop_refusals, osc_refusals, switches = 0, 0, 0, 0, 0
        jumps = Counter()
        notebook = [f"1. beam search (width {self.width}) rated the branches and reached " + ", ".join(
            f"'{self.summaries[n]['title']}' ({s})" for n, s in beam_leaves) + f" | starting at '{self.summaries[node]['title']}'"] \
            if len(beam_leaves) > 1 else []
        last_up_from, opened_at_last_up = None, 0
        exhausted = set()          # (node, target) moves refused -> flagged EXHAUSTED in later prompts
        moves_made = set()         # (node, target, n_opened) forward moves already made
        branch_ends = []           # how each finished beam branch ended

        def next_branch():
            """The best-rated beam leaf not visited yet, or None."""
            return next((n for n, _ in beam_leaves if not visits.get(n)), None)

        stop_reason = "max_calls"
        while True:                                          # one pass per beam branch
            while calls < self.max_calls:                    # the v4 agent loop
                prompt, children, docs, targets, facts = self.render(q["question"], needs, covered, node, opened, path,
                                                                     visits, notebook, exhausted, beam_leaves)
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
                # a stop with a need uncovered and budget left is refused (v4): the agent is sent to the next
                # unvisited beam branch, or up when none is left
                refused_to = None
                if action == "stop" and uncovered and stop_refusals < MAX_STOP_REFUSALS \
                        and self.nodes[node]["parent_id"] is not None:
                    stop_refusals += 1
                    refused_to = next_branch()
                    if refused_to is not None:
                        step["note"] = f"stop refused (need {uncovered} uncovered, budget left) -> next beam branch"
                        entry += (f" [stop refused: need {uncovered} still uncovered - continuing at beam leaf "
                                  f"'{self.summaries[refused_to]['title']}']")
                    else:
                        action = "up"
                        step["note"] = f"stop refused (need {uncovered} uncovered, budget left) -> up"
                        entry += f" [stop refused: need {uncovered} still uncovered - search elsewhere]"
                # redundant-move guard: re-entering the child just left with nothing new opened since, or
                # repeating a move already refused or already made from this node at the same progress.
                # Refused moves are flagged EXHAUSTED in the next prompt; after MAX_MOVE_REFUSALS the run ends.
                move_target = None
                if refused_to is not None:
                    move_target = refused_to
                elif action == "descend" and children and target and 1 <= target <= len(children):
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
                step["effective_action"] = "next_branch" if refused_to is not None else action
                notebook.append(entry)
                next_node = None
                if refused_to is not None:
                    next_node, kind = refused_to, "beam"
                    jumps[kind] += 1
                    switches += 1
                    step["jump_kind"] = kind
                    branch_ends.append({"node": node, "reason": "stop_refused"})
                elif action == "descend" and children and target and 1 <= target <= len(children):
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
            else:
                stop_reason = "max_calls"
            # the branch ended: with budget left and an unvisited beam branch, the search continues there
            nxt = next_branch() if stop_reason in BRANCH_END_REASONS and len(opened) < self.budget \
                and calls < self.max_calls else None
            if nxt is None:
                break
            branch_ends.append({"node": node, "reason": stop_reason})
            switches += 1
            notebook.append(f"{len(notebook) + 1}. finished with '{self.summaries[node]['title']}' ({stop_reason}); "
                            f"{self.budget - len(opened)} article(s) of budget left -> continuing at beam leaf "
                            f"'{self.summaries[nxt]['title']}' (routing score {beam_score.get(nxt)}): open the articles "
                            f"here that could be needed, stop if none fit")
            trace.append({"action": "next_branch", "from": node, "node": nxt, "title": self.summaries[nxt]["title"],
                          "branch_end": stop_reason, "opened_so_far": len(opened)})
            node, path = nxt, self.tree_path(nxt)
            visits[nxt] = visits.get(nxt, 0) + 1
            summaries_read.add(nxt)
            last_up_from, opened_at_last_up = None, 0
        return {
            "qid": q["qid"], "question": q["question"], "gold": q["gold"], "n_gold": q["n_gold"],
            "opened": opened,
            "needs": needs, "n_planned_needs": n_planned, "covered_needs": sorted(covered),
            "summaries_read": len(summaries_read), "doc_rows_read": len(doc_rows_read),
            "summary_tokens_read": sum(num_tokens(self.summaries[n]["summary"]) for n in summaries_read),
            "llm_calls": calls, **usage_total,
            "api_calls_this_run": api_calls, "api_input_tokens_this_run": api_usage["input_tokens"],
            "api_output_tokens_this_run": api_usage["output_tokens"],
            "width": self.width, "router": self.router, "beam_leaves": [n for n, _ in beam_leaves], "beam_rounds": len(beam_steps),
            "branches_visited": sum(1 for n, _ in beam_leaves if visits.get(n)), "branch_switches": switches,
            "branch_ends": branch_ends,
            "beam_incomplete": beam_incomplete,
            "shortcut_use": {**beam_sc, "jumps": jumps.get("shortcut", 0)},
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
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)), help="comma list of beam widths")
    parser.add_argument("--router", choices=ROUTERS, default="llm",
                        help="who rates the candidates in the descent: llm (default), cosine (RAPTOR's router), hybrid")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="first N questions (pilot)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-evidence", action="store_true", help="ablation: no evidence notes/links")
    parser.add_argument("--dry-run", action="store_true", help="print the prompts for the first question, no API")
    parser.add_argument("--check-raptor", action="store_true",
                        help="--router cosine: compare the descent's leaves with raptor_traversal_b<W> results, no LLM call")
    args = parser.parse_args()

    if args.split == "heldout":
        print("WARNING: held-out split is for the final run only; do not tune on it.")
    policies = list(POLICIES) if args.policy == "all" else args.policy.split(",")
    for p in policies:
        if p not in POLICIES:
            parser.error(f"unknown policy {p}")
    widths = [int(w) for w in args.widths.split(",")]
    if any(w < 1 for w in widths):
        parser.error("beam width must be >= 1")
    load_corpus()
    questions = load_questions(args.split)
    if args.limit:
        questions = questions[:args.limit]
    variant = {"llm": VARIANT, "cosine": VARIANT + "_cos", "hybrid": VARIANT + "_hyb"}[args.router] \
        + ("noev" if args.no_evidence else "")
    embedder = Embedder() if args.router != "llm" else None

    if args.check_raptor:                                    # the cosine descent must reach RAPTOR's leaves exactly
        if args.router != "cosine":
            parser.error("--check-raptor needs --router cosine")
        from retrieval.common import results_path
        embedder.embed([q["question"] for q in questions], "(questions)")
        for width in widths:
            path = results_path(f"raptor_traversal_b{width}", args.tree, args.split)
            if not path.exists():
                print(f"W={width}: missing {path} (run baselines.py --systems raptor_traversal --beams {width})")
                continue
            ref = {r["qid"]: r["trace"]["leaves_reached"] for r in json.load(path.open(encoding="utf-8"))["per_question"]}
            for policy in policies:
                nav = Navigator(args.tree, policy, args.budget, args.max_calls, evidence=False, width=width,
                                router="cosine", embedder=embedder)
                bad = [q["qid"] for q in questions if q["qid"] in ref and
                       [n for n, _ in nav.beam_descend(q["question"], [], width, nav.question_vec(q["question"]))[0]] != ref[q["qid"]]]
                print(f"W={width} policy={policy}: {sum(q['qid'] in ref for q in questions) - len(bad)} identical, "
                      f"{len(bad)} different from raptor_traversal_b{width}" + (f" {bad[:10]}" if bad else ""))
        return

    if args.dry_run:
        nav = Navigator(args.tree, policies[0], args.budget, args.max_calls, evidence=not args.no_evidence, width=widths[0],
                        router=args.router, embedder=embedder)
        q = questions[0]
        print("=" * 70 + "\n[PLAN call]\n" + "=" * 70)
        print(PLAN_PROMPT.strip() + f"\n\nQUESTION: {q['question']}")
        needs = ["(information need 1 - filled by the plan call)"]
        if args.router == "cosine":                       # no rating prompt: show where RAPTOR's descent ends
            leaves, _, steps, _, _ = nav.beam_descend(q["question"], needs, widths[0], nav.question_vec(q["question"]))
            print("\n" + "=" * 70 + f"\n[BEAM by cosine, width {widths[0]}: {len(steps)} levels, no LLM call]\n" + "=" * 70)
            for st in steps:
                print(f"  level {st['round']}: kept " + ", ".join(
                    f"'{c['title']}' ({c['score']})" for c in st["candidates"] if c["node"] in st["kept"]))
            gold_leaves = {nav.leaf_of[g] for g in q["gold"] if g in nav.leaf_of}
            print("  reached leaves: " + ", ".join(f"'{nav.summaries[n]['title']}' ({sc})" + (" [gold]" if n in gold_leaves else "")
                                                for n, sc in leaves))
        else:
            print("\n" + "=" * 70 + "\n[BEAM level 1: the root's children are rated]\n" + "=" * 70)
            cands = [(c, "child", f"child of '{nav.summaries[nav.root]['title']}'") for c in nav.nodes[nav.root]["children"]]
            prompt = nav.score_prompt(q["question"], needs, set(), cands)
            print(prompt)
            print(f"\n[~{num_tokens(prompt)} tokens | system prompt ~{num_tokens(SCORE_PROMPT)} tokens]")
        # the agent's first prompt at the gold leaf, with two other (sibling) leaves as beam leaves
        if q["gold"][0] in nav.leaf_of:
            g = q["gold"][0]
            leaf = nav.leaf_of[g]
            path = nav.tree_path(leaf)
            parent = nav.nodes[leaf]["parent_id"]
            others = [c for c in nav.nodes[parent]["children"] if c != leaf and nav.nodes[c]["is_leaf"]][:widths[0] - 1] if parent else []
            beam = [(leaf, 90)] + [(c, 60) for c in others]
            notebook = ["1. beam search (width %d) rated the branches and reached " % widths[0] + ", ".join(
                f"'{nav.summaries[n]['title']}' ({s})" for n, s in beam) + f" | starting at '{nav.summaries[leaf]['title']}'"] if len(beam) > 1 else []
            prompt, _, _, targets, _ = nav.render(q["question"], needs, set(), leaf, [], path, {n: 1 for n in path}, notebook, beam_leaves=beam)
            print("\n" + "=" * 70 + f"\n[agent prompt at the gold leaf '{nav.summaries[leaf]['title']}' reached by the beam; "
                  f"{len(targets)} jump targets]\n" + "=" * 70)
            print(prompt)
            print(f"\n[~{num_tokens(prompt)} tokens | system prompt ~{num_tokens(nav.nav_prompt)} tokens]")
            if nav.evidence:
                prompt, _, _, targets, _ = nav.render(q["question"], needs, {1}, leaf, [g], path, {n: 1 for n in path},
                                                      notebook + ["2. ... (opened the gold article)"], beam_leaves=beam)
                print("\n" + "=" * 70 + f"\n[review prompt after opening '{nav.doc_rows[g]['title']}'; {len(targets)} jump targets]\n" + "=" * 70)
                print(prompt)
                print(f"\n[~{num_tokens(prompt)} tokens]")
        print(f"\nmodel {MODEL_NAME} (reasoning {REASONING_EFFORT}) | budget {args.budget} | widths {widths} | max calls {args.max_calls} "
              f"| questions {len(questions)} | policies {policies} | tree {args.tree} | variant {variant}")
        return

    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Missing OPENAI_API_KEY in .env")
    from openai import OpenAI
    client = OpenAI(timeout=180.0, max_retries=3)
    if embedder is not None:                                 # one batch before the worker threads; free when cached
        embedder.embed([q["question"] for q in questions], "(questions)")

    for width, policy in [(w, p) for w in widths for p in policies]:
        nav = Navigator(args.tree, policy, args.budget, args.max_calls, client=client,
                        use_cache=not args.no_cache, evidence=not args.no_evidence, width=width,
                        router=args.router, embedder=embedder)
        system = f"nav_{variant}_W{width}_{policy}_B{args.budget}"
        print(f"\n========== NAVIGATOR {variant} router={args.router} W={width} policy={policy} tree={args.tree} budget={args.budget} "
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
                      f"| beam {r['beam_rounds']}lv/{len(r['beam_leaves'])}leaves visited {r['branches_visited']} sc kept {r['shortcut_use']['kept_in_beam']} "
                      f"| jumps sc{jk.get('shortcut', 0)} bm{jk.get('beam', 0)} ev{jk.get('evidence', 0)} ups {r['ups']} refused {r['stop_refusals']} "
                      f"| {r['stop_reason']:14s} | {r['question'][:50]}")
        ordered = [records[q["qid"]] for q in questions if q["qid"] in records]
        if not ordered:
            print("no results")
            continue
        config = {"model": MODEL_NAME, "reasoning_effort": REASONING_EFFORT, "budget": args.budget,
                  "max_calls": args.max_calls, "policy": policy, "variant": variant, "width": width, "router": args.router,
                  "descent_shortcuts": args.router != "cosine" and policy != "none",
                  "embedding_model": embedder.model if embedder else None,
                  "evidence": not args.no_evidence, "min_candidate_score": MIN_CANDIDATE_SCORE, "max_rounds": MAX_ROUNDS,
                  "schema_version": SCHEMA_VERSION, "prompt_sha256": sha256_text(nav.nav_prompt + PLAN_PROMPT + SCORE_PROMPT),
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
        n = len(ordered)
        print(quick_report(system, ordered, budgets=(args.budget,)))
        print(f"  beam levels/q {sum(r['beam_rounds'] for r in ordered) / n:.1f} | leaves/q {sum(len(r['beam_leaves']) for r in ordered) / n:.1f} "
              f"| branches visited/q {sum(r['branches_visited'] for r in ordered) / n:.2f} "
              f"| incomplete descents {sum(r['beam_incomplete'] for r in ordered)} "
              f"| questions continuing to a 2nd branch {sum(r['branch_switches'] > 0 for r in ordered)} "
              f"| shortcut targets rated {sum(r['shortcut_use']['candidates_scored'] for r in ordered)} kept in beam "
              f"{sum(r['shortcut_use']['kept_in_beam'] for r in ordered)} | opened/q {sum(len(r['opened']) for r in ordered) / n:.2f}")
        print(f"  planned needs per question {dict(sorted(planned.items()))} | needs added from evidence {added} "
              f"| stop refusals {sum(r['stop_refusals'] for r in ordered)} | jumps {dict(jk)}")
        print(f"  logical cost {tokens} | billed this run {billed} | stop reasons {dict(stops)}")
        print(f"  saved {path}")
        if len(ordered) < len(questions):
            print(f"  {len(questions) - len(ordered)} questions failed; rerun (cached decisions are free)")


if __name__ == "__main__":
    main()