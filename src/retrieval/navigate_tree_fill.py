# src/retrieval/navigate_tree_fill.py
"""
Graph-guided hierarchical retrieval - the BEAM navigator (variant v5).

Offline steps are unchanged (extract_relations_v5 -> canonicalize_entities ->
build_doc_graph -> build_tree_v2 -> summarize_tree_v2 -> select_shortcuts).
This file replaces the single-path agent of navigate_tree.py (v4: one walker
that descends, jumps, goes up) with a top-down BEAM SEARCH over the tree, so
several branches stay open at once. One question, one run:

  1. PLAN      one LLM call lists the question's information needs (1-3)
               (same prompt as v4, so the cached plans are reused).
  2. BEAM      frontier = {root}. Each round ONE LLM call rates every
               candidate 0-100 jointly: the children of the internal frontier
               nodes, their precomputed KG shortcut targets (policy != none),
               and the leaves kept from the previous round. The best W
               candidates (score >= MIN_CANDIDATE_SCORE) form the next
               frontier. Rounds end when the frontier holds only leaves.
  3. OPEN      one LLM call sees the articles of ALL reached leaves (title,
               one-line description, key entities) and picks the articles to
               open, each with a score and the needs it covers; picks are
               opened in score order within the article budget B.
  4. REVIEW    (evidence variant) one LLM call sees the opened articles' KG
               facts ([F1], [F2], ... each with the other articles covering the
               same specific entity) and FOLLOW-UP TARGETS: evidence links
               (leaves of related articles) and the KG shortcuts of the reached
               leaves. It may add ONE need grounded in a fact and follow up to
               W targets; followed nodes are expanded with the same beam
               procedure, their leaves go through step 3, and the review
               repeats (at most MAX_REVIEWS times) while budget remains.
               Without evidence (--no-evidence) the review offers shortcut
               targets only; with policy none and no evidence there is no
               review (pure beam + open).

Compared with v4 there is no up/jump/stop action, no notebook and no refusal
machinery: the beam keeps alternatives alive instead of backtracking, the
article budget is spent by a joint choice over W leaves instead of leaf by
leaf, and the KG shortcuts act in two places - as routing candidates during
the descent and as follow-up targets after the review. Width W=1 is a greedy
LLM descent; the raptor_traversal_bW baseline is the same beam with a cosine
scorer, so (W, scorer) is the only difference between the two.

Cost: summaries read = nodes whose routing summary was shown to the model,
article rows read, LLM calls, tokens (logical = all calls incl. cached;
billed = this run's API calls). Every decision is cached by its full prompt
(eval/cache/navigator_cache.jsonl); the prompts never mention W, so rounds
that coincide across widths (always the first) are free.

  5. FILL      (--fill) the OPEN call rates EVERY listed article instead of
               picking: ratings >= OPEN_SCORE are opened at once, the rest form
               a reserve; when the search ends with budget left, the budget is
               filled from the highest-rated reserve articles (no extra call).
               Without --fill the navigator opens only what it picked (3.2 of
               5 articles on dev) and the unused budget is lost, which the
               fixed-budget protocol (all-gold@B) never rewards.

Results: eval/results/beam[noev][fill]_W<W>_<policy>_B<B>__<tree>__<split>.json,
one file per (width, policy); the per-question record keeps the v4 fields and
adds width, rounds, reviews and shortcut/evidence/fill usage. --oracle replaces
the LLM by a gold-aware scorer (beamoracle_... files): the structural ceiling of
the beam protocol for a width and policy, and a free end-to-end plumbing test.

Usage
-----
    python src/retrieval/navigate_tree_v2.py --dry-run                        # prompts, no API
    python src/retrieval/navigate_tree_v2.py --oracle --widths 1,2,3 --policy none,typed   # ceilings, no API
    python src/retrieval/navigate_tree_v2.py --widths 1 --policy none --limit 10           # pilot
    python src/retrieval/navigate_tree_v2.py --widths 1,2,3 --policy none,typed            # width sweep x shortcut ablation
    python src/retrieval/navigate_tree_v2.py --widths 1,2,3 --policy none,typed --fill     # + budget fill
    python src/retrieval/navigate_tree_v2.py --widths 1,2,3 --policy none,typed --no-evidence
    python src/retrieval/evaluate_v2.py --strict --ref dense

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

VARIANT = "v5"
SCHEMA_VERSION = f"beam-navigator-{VARIANT}"
BUDGET = 5                     # articles the navigator may open
WIDTHS = (1, 2, 3)             # default beam widths of the sweep
MAX_CALLS = 24                 # hard cap on LLM calls per question (typical run: 6-11)
MAX_ROUNDS = 8                 # beam rounds per descent (tree depth <= 6, plus shortcut hops)
MAX_REVIEWS = 2                # evidence reviews per question
MIN_CANDIDATE_SCORE = 10       # a candidate below this score is not kept even if a beam slot is free
OPEN_SCORE = 60                # --fill: an article rated at least this is opened at once; lower ratings form the reserve
MAX_NEEDS = 4                  # plan needs + needs added from evidence
MAX_EVIDENCE_NEEDS = 1         # needs the navigator may add from evidence
MAX_REVIEW_SHORTCUTS = 6       # shortcut follow-up targets offered per review (after the evidence links)
CHILD_QUESTIONS = 4
CHILD_TERMS = 10
SHORTCUT_SUMMARY_CHARS = 220
DEFAULT_WORKERS = 4
MAX_OUTPUT_TOKENS = 4000
POLICIES = ("none",) + SHORTCUT_POLICIES
CACHE_PATH = CACHE_DIR / "navigator_cache.jsonl"
KIND_RANK = {"retained": 0, "start": 0, "child": 1, "shortcut": 2, "evidence": 3}

# evidence notes (identical to navigate_tree.py v4)
DEP_PREDICATES = {"REQUIRES": 1.0, "RESOLVED_BY": 1.0, "CAUSES": 0.8, "HAS_LIMITATION": 0.8,
                  "APPLIES_TO": 0.8, "LOCATED_IN": 0.6}
GENERIC_NAMES = {"wix", "site", "wix site", "website", "page", "user", "visitor",
                 "wix account", "dashboard", "site dashboard"}
EVIDENCE_MAX_DF_RATIO = 0.05
NOTES_PER_DOC = 6
DOCS_PER_NOTE = 3
MAX_EVIDENCE_LINKS = 5
MAX_FACTS_SHOWN = 12


# ------------------------------------------------------------
# schemas
# ------------------------------------------------------------

class NavPlan(BaseModel):          # unchanged from v4 -> cached plans are reused
    needs: list[str]


class CandidateScore(BaseModel):
    index: int
    score: int                     # 0-100


class ScoreDecision(BaseModel):
    scores: list[CandidateScore]
    reason: str


class ArticlePick(BaseModel):
    index: int
    score: int                     # 0-100
    covers_needs: list[int]


class LeafDecision(BaseModel):
    picks: list[ArticlePick]
    reason: str


class ArticleRating(BaseModel):   # --fill: every listed article is rated
    index: int
    score: int                     # 0-100
    covers_needs: list[int]


class LeafRatings(BaseModel):
    ratings: list[ArticleRating]
    reason: str


class NewNeed(BaseModel):
    need: str
    fact: int                      # index of the EVIDENCE NOTES fact ([F1], [F2], ...) this need comes from


class ReviewDecision(BaseModel):
    covered_needs: list[int]
    new_needs: list[NewNeed]
    follow: list[int]
    done: bool
    reason: str


class ReviewDecisionNoEvidence(BaseModel):
    covered_needs: list[int]
    follow: list[int]
    done: bool
    reason: str


# ------------------------------------------------------------
# prompts
# ------------------------------------------------------------

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

LEAF_PROMPT_HEAD = """
You choose which help-center articles to open for a question from a Wix
website owner. You see the articles of the leaves the search reached: title,
one-line description and key entities - never the bodies. Success is measured
by whether the articles the answer needs are among the opened ones, within an
article budget.

Open an article for EVERY information need. When several articles could
cover a need, pick all plausible ones; also pick the supporting article a
support agent would attach when it is listed (the prerequisite, the related
setting, the troubleshooting article). Do not pick articles that clearly do
not fit"""

LEAF_PROMPT_PICKS = {
    True: """, and do not fill the budget with weak candidates: after this step the
opened articles' knowledge-graph evidence is reviewed and budget left can be
spent on related branches. Rate each pick 0-100 (how likely the article is
needed); picks are opened in score order within the remaining budget. Return
the picks (index, score, the needs it covers) and one short sentence.
""",
    False: """; budget not spent now is lost. Rate each pick 0-100 (how likely the
article is needed); picks are opened in score order within the remaining
budget. Return the picks (index, score, the needs it covers) and one short
sentence.
"""}

LEAF_PROMPT_RATINGS = {
    True: """. Rate EVERY listed article 0-100: how likely is it that the answer
needs it? Articles rated 60 or higher are opened now, in score order within
the remaining budget; after this step the opened articles' knowledge-graph
evidence is reviewed and budget left can be spent on related branches; when
the search ends, budget still left is filled from the highest-rated articles
not yet opened, so the order below 60 matters too. Give each rating the needs
the article covers. Return a rating for every index and one short sentence.
""",
    False: """. Rate EVERY listed article 0-100: how likely is it that the answer
needs it? Articles rated 60 or higher are opened now, in score order within
the remaining budget; when the search ends, budget still left is filled from
the highest-rated articles not yet opened, so the order below 60 matters too.
Give each rating the needs the article covers. Return a rating for every
index and one short sentence.
"""}


def leaf_prompt_text(evidence, fill):
    return LEAF_PROMPT_HEAD + (LEAF_PROMPT_RATINGS if fill else LEAF_PROMPT_PICKS)[bool(evidence)]


REVIEW_PROMPT_EVIDENCE = """
You review what a search has opened for a question from a Wix website owner
and decide whether to open more. You see the information needs, the opened
articles, EVIDENCE NOTES - knowledge-graph facts extracted from the opened
articles ([F1], [F2], ...: what the task requires, what limits it, what causes
or resolves a problem, where a setting lives; each with the other articles
covering the same specific entity) - and FOLLOW-UP TARGETS: nodes in other
branches that can be explored next (evidence links derived from the notes,
and precomputed KG shortcuts of the reached leaves). Evidence notes are
candidate leads extracted automatically, not verified answers; you never see
article bodies.

Return:
- covered_needs: indices of the needs the opened articles cover.
- new_needs: at most ONE need discovered from the evidence - a prerequisite
  step, the fix of a cause, the companion setting the answer also needs - as
  {need: 3-12 words, fact: the [F#] it comes from}; [] if none or if a need
  was already added (marked +). Do not add needs for facts that are merely
  context.
- follow: indices of follow-up targets to explore now, best first, at most the
  number allowed. Budget left is free to spend: following a target that
  plausibly holds a needed article never hurts, skipping it may lose the
  article. Prefer targets for uncovered or new needs; while a need is
  uncovered you must follow a plausible target if there is one.
- done: true only when every need is covered and no target is plausible.
- reason: one short sentence.
"""

REVIEW_PROMPT_NO_EVIDENCE = """
You review what a search has opened for a question from a Wix website owner
and decide whether to open more. You see the information needs, the opened
articles and FOLLOW-UP TARGETS: nodes in other branches, reachable through
precomputed knowledge-graph shortcuts from the reached leaves, that can be
explored next. You never see article bodies.

Return:
- covered_needs: indices of the needs the opened articles cover.
- follow: indices of follow-up targets to explore now, best first, at most the
  number allowed. Budget left is free to spend: following a target that
  plausibly holds a needed article never hurts, skipping it may lose the
  article. Prefer targets for uncovered needs; while a need is uncovered you
  must follow a plausible target if there is one.
- done: true only when every need is covered and no target is plausible.
- reason: one short sentence.
"""


# ------------------------------------------------------------
# helpers (first_sentence, shortcut_label, EvidenceIndex: as in v4)
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
        self.relations = {}
        central = defaultdict(Counter)
        presence = defaultdict(set)
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
        out, listed, used_entities = [], set(), set()
        for x in self.relations.get(doc_id, []):
            picked_entity, picked = None, []
            for e in sorted({x["subject"], x["object"]}, key=lambda n: (self.df.get(n, 0), n)):   # name breaks ties: set order is per-process
                if e not in self.specific or e in used_entities:
                    continue
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


class Cand:
    """A candidate / frontier node of the beam."""
    __slots__ = ("node", "kind", "source", "label", "score", "via_shortcut", "via_evidence")

    def __init__(self, node, kind, source=None, label="", score=None, via_shortcut=False, via_evidence=False):
        self.node, self.kind, self.source, self.label, self.score = node, kind, source, label, score
        self.via_shortcut, self.via_evidence = via_shortcut, via_evidence


# ------------------------------------------------------------
# the beam navigator
# ------------------------------------------------------------

class BeamNavigator:
    def __init__(self, tree_method, policy, width, budget, max_calls, client=None, use_cache=True,
                 evidence=True, oracle=False, fill=False):
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
        self.width = width
        self.budget = budget
        self.max_calls = max_calls
        self.client = client
        self.use_cache = use_cache
        self.oracle = oracle
        self.fill = fill
        self.cache = {}
        self.lock = threading.Lock()
        if use_cache and not oracle:
            for r in read_jsonl(CACHE_PATH):
                self.cache[r["key"]] = r
        self.leaf_prompt_text = leaf_prompt_text(evidence, fill)
        self.leaf_cls = LeafRatings if fill else LeafDecision
        self.review_prompt_text = REVIEW_PROMPT_EVIDENCE if evidence else REVIEW_PROMPT_NO_EVIDENCE
        self.review_cls = ReviewDecision if evidence else ReviewDecisionNoEvidence
        self.schemas = {cls: json.dumps(cls.model_json_schema(), sort_keys=True)
                        for cls in (NavPlan, ScoreDecision, self.leaf_cls, self.review_cls)}

    # ---- views -------------------------------------------------------
    def tree_path(self, node_id):
        out = []
        while node_id is not None:
            out.append(node_id)
            node_id = self.nodes[node_id]["parent_id"]
        return out[::-1]

    def needs_lines(self, needs, covered):
        lines = ["INFORMATION NEEDS (your plan" + ("; needs added from evidence are marked +" if self.evidence else "") + "):"]
        for i, need in enumerate(needs, 1):
            lines.append(f"  [{i}] {need}   - {'COVERED' if i in covered else 'not yet covered'}")
        return lines

    def opened_lines(self, opened):
        if opened:
            return [f"OPENED SO FAR ({len(opened)} of {self.budget}):"] + [f"  - {self.doc_rows[d]['title']}" for d in opened] \
                + [f"REMAINING BUDGET: {self.budget - len(opened)} article(s)"]
        return [f"OPENED SO FAR (0 of {self.budget}): none", f"REMAINING BUDGET: {self.budget} article(s)"]

    def score_prompt(self, question, needs, covered, cands):
        lines = [f"QUESTION: {question}", ""] + self.needs_lines(needs, covered) + ["", "CANDIDATES (rate every one):"]
        for i, c in enumerate(cands, 1):
            s = self.summaries[c.node]
            what = "leaf" if self.nodes[c.node]["is_leaf"] else "node"
            lines.append(f"[{i}] {s['title']} ({what}, depth {s['depth']}, {s['size']} articles) - {c.label}")
            lines.append(f"    {s['summary']}")
            if s.get("question_types"):
                lines.append("    typical questions: " + " | ".join(s["question_types"][:CHILD_QUESTIONS]))
            if s.get("key_terms"):
                lines.append("    key terms: " + ", ".join(s["key_terms"][:CHILD_TERMS]))
        lines += ["", f"Return a score (0-100) for every index 1..{len(cands)}, and a short reason."]
        return "\n".join(lines)

    def leaf_prompt(self, question, needs, covered, opened, leaves):
        """(prompt, [(doc_id, leaf Cand)]) - the articles of all reached leaves, one numbered list."""
        remaining = self.budget - len(opened)
        lines = [f"QUESTION: {question}", ""] + self.needs_lines(needs, covered) + [""] + self.opened_lines(opened)
        lines += ["", f"ARTICLES (from the {len(leaves)} leaf/leaves the search reached; open at most {remaining} now):"]
        docs = []
        for c in leaves:
            s = self.summaries[c.node]
            via = " - reached through a KG shortcut" if c.via_shortcut else ""
            lines.append(f"Leaf '{s['title']}' ({s['size']} articles{via}):")
            for d in self.nodes[c.node]["doc_ids"]:
                row = self.doc_rows[d]
                docs.append((d, c))
                ents = ", ".join(row.get("key_entities", [])[:5])
                flag = "   [OPENED]" if d in opened else ""
                lines.append(f"[{len(docs)}] {row['title']} - {row['one_line']}" + (f" (entities: {ents})" if ents else "") + flag)
        if self.fill:
            lines += ["", f"Return a rating (index, score 0-100, covers_needs) for every index 1..{len(docs)} and a short reason."]
        else:
            lines += ["", "Return picks (index, score 0-100, covers_needs) and a short reason."]
        return "\n".join(lines), docs

    def evidence_block(self, opened):
        """(lines, [(leaf_id, [doc ids], why)], facts) - as in v4."""
        if not self.evidence or not opened:
            return [], [], []
        lines = ["EVIDENCE NOTES (knowledge-graph facts from the articles you opened; candidate leads, not verified answers):"]
        links, order, facts = {}, [], []
        for d in reversed(opened):
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

    def review_targets(self, st):
        """Follow-up targets: evidence links (leaves of related articles), then KG shortcuts of the reached leaves."""
        ev_lines, ev_links, facts = self.evidence_block(st["opened"])
        targets, seen = [], set()
        for leaf, rel_docs, why in ev_links:
            if leaf in st["followed"] or leaf in seen:
                continue
            seen.add(leaf)
            titles = "; ".join(f"'{self.doc_rows[d]['title']}'" for d in rel_docs[:DOCS_PER_NOTE])
            note = " (its articles were already listed)" if leaf in st["seen_leaves"] else ""
            targets.append(Cand(leaf, "evidence", None, f"evidence link{note}: holds {titles}; because of {why}",
                                via_evidence=True))
            st["ev_links_offered"] += 1
        if self.shortcuts:
            n_sc = 0
            for leaf in st["seen_leaves_order"]:
                for sc in self.shortcuts.get(leaf, []):
                    t = sc["target"]
                    if t not in self.summaries or t in seen or t in st["seen_leaves"] or t in st["followed"] \
                            or t in st["expanded"]:
                        continue
                    if n_sc >= MAX_REVIEW_SHORTCUTS:
                        break
                    seen.add(t)
                    n_sc += 1
                    rated = f"; rated {st['node_scores'][t]} during routing" if t in st["node_scores"] else ""
                    targets.append(Cand(t, "shortcut", leaf,
                                        f"KG shortcut from '{self.summaries[leaf]['title']}' - {shortcut_label(sc)}{rated}",
                                        via_shortcut=True))
            st["sc_review_offered"] += n_sc
        return ev_lines, facts, targets

    def review_prompt(self, question, needs, covered, opened, ev_lines, targets, max_follow):
        lines = [f"QUESTION: {question}", ""] + self.needs_lines(needs, covered) + [""] + self.opened_lines(opened) + [""]
        lines += ev_lines
        lines.append(f"FOLLOW-UP TARGETS (other branches you can explore next; follow at most {max_follow}):")
        for i, c in enumerate(targets, 1):
            s = self.summaries[c.node]
            what = "leaf" if self.nodes[c.node]["is_leaf"] else "node"
            lines.append(f"[{i}] -> {s['title']} ({what}, depth {s['depth']}, {s['size']} articles) - {c.label}")
            if c.kind == "shortcut":
                lines.append(f"    {first_sentence(s['summary'], SHORTCUT_SUMMARY_CHARS)}")
        lines += ["", "Decide: covered_needs, " + ("new_needs, " if self.evidence else "") + "follow, done, reason."]
        return "\n".join(lines)

    # ---- llm ---------------------------------------------------------
    def _call(self, system_prompt, user_prompt, schema_cls):
        schema_json = self.schemas[schema_cls]
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

    def decide(self, st, system_prompt, user_prompt, schema_cls, oracle_fn):
        """One decision: oracle, cache or API. Counts the call and its tokens in the question state."""
        if st["calls"] >= self.max_calls:
            raise RuntimeError(f"max calls ({self.max_calls}) reached")
        if self.oracle:
            decision, usage, cached = oracle_fn(), {"input_tokens": num_tokens(system_prompt + user_prompt),
                                                   "output_tokens": 0, "reasoning_tokens": 0}, True
        else:
            decision, usage, cached = self._call(system_prompt, user_prompt, schema_cls)
        st["calls"] += 1
        for k in st["usage"]:
            st["usage"][k] += usage.get(k, 0)
        if not cached:
            st["api_calls"] += 1
            st["api_usage"]["input_tokens"] += usage.get("input_tokens", 0)
            st["api_usage"]["output_tokens"] += usage.get("output_tokens", 0)
        return decision, cached

    # ---- stages ------------------------------------------------------
    def plan(self, st):
        q = st["q"]
        prompt = f"QUESTION: {q['question']}"
        decision, cached = self.decide(st, PLAN_PROMPT, prompt, NavPlan,
                                       lambda: NavPlan(needs=[f"gold article {i + 1}" for i in range(q["n_gold"])]))
        needs = [n.strip() for n in decision.needs if n.strip()][:3] or [q["question"]]
        st["needs"], st["n_planned"] = needs, len(needs)
        st["trace"].append({"action": "plan", "needs": list(needs), "cached": cached})

    def beam_descend(self, st, starts, width, stage):
        """Top-down beam from `starts` (Cand list). Returns the leaves of the final frontier."""
        frontier = list(starts)
        st["scored"].update(c.node for c in frontier)
        gold_left = set(st["q"]["gold"]) - set(st["opened"])
        for rnd in range(1, MAX_ROUNDS + 1):
            internal = [c for c in frontier if not self.nodes[c.node]["is_leaf"]]
            if not internal:
                break
            st["expanded"].update(c.node for c in internal)
            retained = [Cand(c.node, "retained", c.source, "kept from the previous round", c.score, c.via_shortcut, c.via_evidence)
                        for c in frontier if self.nodes[c.node]["is_leaf"]]
            new, seen = [], {c.node for c in retained}
            for c in internal:
                title = self.summaries[c.node]["title"]
                for ch in self.nodes[c.node]["children"]:
                    if ch in st["scored"] or ch in seen:
                        continue
                    seen.add(ch)
                    new.append(Cand(ch, "child", c.node, f"child of '{title}'",
                                    via_shortcut=c.via_shortcut, via_evidence=c.via_evidence))
                for sc in self.shortcuts.get(c.node, []):
                    t = sc["target"]
                    if t not in self.summaries or t in st["scored"] or t in seen:
                        continue
                    seen.add(t)
                    st["sc_candidates"] += 1
                    new.append(Cand(t, "shortcut", c.node, f"KG shortcut from '{title}' - {shortcut_label(sc)}",
                                    via_shortcut=True, via_evidence=c.via_evidence))
            if not new:                                  # nothing left to expand
                frontier = retained
                break
            cands = retained + new
            prompt = self.score_prompt(st["q"]["question"], st["needs"], st["covered"], cands)
            st["summaries_read"].update(c.node for c in cands)

            def oracle():
                return ScoreDecision(scores=[CandidateScore(index=i, score=100 if self.docsets[c.node] & gold_left else 0)
                                             for i, c in enumerate(cands, 1)], reason="oracle")
            decision, cached = self.decide(st, SCORE_PROMPT, prompt, ScoreDecision, oracle)
            given = {}
            for s in decision.scores:
                if 1 <= s.index <= len(cands) and (s.index - 1) not in given:
                    given[s.index - 1] = max(0, min(100, s.score))
            for i, c in enumerate(cands):
                c.score = given.get(i, 0)
                st["node_scores"][c.node] = c.score
            st["scored"].update(c.node for c in new)
            ranked = sorted(cands, key=lambda c: (-c.score, KIND_RANK[c.kind], c.node))
            kept = [c for c in ranked if c.score >= MIN_CANDIDATE_SCORE][:width] or ranked[:1]
            st["sc_kept"] += sum(1 for c in kept if c.kind == "shortcut")
            st["rounds"] += 1
            st["trace"].append({
                "action": "score", "stage": stage, "round": rnd, "width": width,
                "nodes": [c.node for c in internal],                 # nodes expanded in this round
                "candidates": [{"node": c.node, "title": self.summaries[c.node]["title"], "kind": c.kind,
                                "depth": self.nodes[c.node]["depth"], "score": c.score} for c in cands],
                "kept": [c.node for c in kept], "reason": decision.reason, "cached": cached})
            frontier = kept
        return [c for c in frontier if self.nodes[c.node]["is_leaf"]]

    def open_from_leaves(self, st, leaves, stage):
        """One joint pick over the articles of all reached leaves; opens the picks in score order."""
        remaining = self.budget - len(st["opened"])
        if not leaves or remaining <= 0:
            return
        q = st["q"]
        prompt, docs = self.leaf_prompt(q["question"], st["needs"], st["covered"], st["opened"], leaves)
        st["doc_rows_read"].update(d for d, _ in docs)
        st["summaries_read"].update(c.node for c in leaves)
        for c in leaves:
            if c.node not in st["seen_leaves"]:
                st["seen_leaves"].add(c.node)
                st["seen_leaves_order"].append(c.node)
        gold = set(q["gold"])

        all_needs = list(range(1, len(st["needs"]) + 1))

        def oracle():
            if self.fill:
                return LeafRatings(ratings=[ArticleRating(index=i, score=100 if d in gold else 0, covers_needs=all_needs if d in gold else [])
                                            for i, (d, _) in enumerate(docs, 1)], reason="oracle")
            return LeafDecision(picks=[ArticlePick(index=i, score=100, covers_needs=all_needs)
                                       for i, (d, _) in enumerate(docs, 1) if d in gold], reason="oracle")
        decision, cached = self.decide(st, self.leaf_prompt_text, prompt, self.leaf_cls, oracle)
        step = {"action": "open", "stage": stage, "nodes": [c.node for c in leaves],
                "titles": [self.summaries[c.node]["title"] for c in leaves], "n_listed": len(docs),
                "picks": [], "opened_here": [], "opened_here_titles": [], "covered_needs": [],
                "reason": decision.reason, "cached": cached}
        items = decision.ratings if self.fill else decision.picks
        picks = sorted((p for p in items if 1 <= p.index <= len(docs)), key=lambda p: (-p.score, p.index))
        # --fill: ratings below OPEN_SCORE go to the reserve, except that a stage never opens nothing while
        # budget remains (the best-rated article is opened so the evidence review has something to work with)
        force_first = self.fill and bool(picks) and not any(p.score >= OPEN_SCORE for p in picks)
        for k, p in enumerate(picks):
            d, leaf = docs[p.index - 1]
            entry = {"doc": d, "title": self.doc_rows[d]["title"], "score": max(0, min(100, p.score)),
                     "covers_needs": [i for i in p.covers_needs if 1 <= i <= len(st["needs"])], "leaf": leaf.node}
            if d in st["opened"] or any(x["doc"] == d for x in step["picks"]):
                entry["status"] = "duplicate"
            elif self.fill and entry["score"] < OPEN_SCORE and not (force_first and k == 0):
                entry["status"] = "reserve"
                if d not in st["pool"] or st["pool"][d][0] < entry["score"]:
                    st["pool"][d] = (entry["score"], leaf, entry["covers_needs"])
            elif len(st["opened"]) >= self.budget:
                entry["status"] = "over budget"
                if self.fill and (d not in st["pool"] or st["pool"][d][0] < entry["score"]):
                    st["pool"][d] = (entry["score"], leaf, entry["covers_needs"])
            else:
                entry["status"] = "opened"
                st["opened"].append(d)
                step["opened_here"].append(d)
                step["opened_here_titles"].append(entry["title"])
                if leaf.via_shortcut:
                    st["opened_via_shortcut"].append(d)
                if leaf.via_evidence:
                    st["opened_via_evidence"].append(d)
                for i in entry["covers_needs"]:
                    if i not in st["covered"]:
                        st["covered"].add(i)             # the navigator's claim, not verified coverage
                        step["covered_needs"].append(i)
            step["picks"].append(entry)
        st["trace"].append(step)

    def review(self, st, i):
        """Returns (targets to follow, done) or None when there is nothing to review."""
        ev_lines, facts, targets = self.review_targets(st)
        if not targets:
            return None
        q = st["q"]
        remaining = self.budget - len(st["opened"])
        max_follow = max(1, min(self.width, remaining))
        prompt = self.review_prompt(q["question"], st["needs"], st["covered"], st["opened"], ev_lines, targets, max_follow)
        st["summaries_read"].update(c.node for c in targets)
        gold_left = set(q["gold"]) - set(st["opened"])

        def oracle():
            follow = [j for j, c in enumerate(targets, 1) if self.docsets[c.node] & gold_left][:max_follow]
            kw = {"covered_needs": list(range(1, len(st["needs"]) + 1)) if not gold_left else [],
                  "follow": follow, "done": not gold_left, "reason": "oracle"}
            if self.evidence:
                kw["new_needs"] = []
            return self.review_cls(**kw)
        decision, cached = self.decide(st, self.review_prompt_text, prompt, self.review_cls, oracle)
        step = {"action": "review", "n": i, "covered_needs": [], "new_needs": [], "rejected_new_needs": [],
                "targets": [{"node": c.node, "title": self.summaries[c.node]["title"], "kind": c.kind} for c in targets],
                "follow": [], "done": decision.done, "reason": decision.reason, "cached": cached}
        for j in decision.covered_needs:
            if 1 <= j <= len(st["needs"]) and j not in st["covered"]:
                st["covered"].add(j)
                step["covered_needs"].append(j)
        for nn in (getattr(decision, "new_needs", None) or [])[:2]:
            text = nn.need.strip()
            known = {n.removeprefix("+ ").lower() for n in st["needs"]}
            if text and 1 <= nn.fact <= len(facts) and len(st["needs"]) < MAX_NEEDS \
                    and len(st["needs"]) - st["n_planned"] < MAX_EVIDENCE_NEEDS and text.lower() not in known:
                st["needs"].append("+ " + text)
                step["new_needs"].append({"need": text, "fact": facts[nn.fact - 1]})
            elif text:
                step["rejected_new_needs"].append({"need": text, "fact": nn.fact})
        follow = []
        for j in decision.follow:
            if 1 <= j <= len(targets) and targets[j - 1].node not in {c.node for c in follow}:
                follow.append(targets[j - 1])
            if len(follow) >= max_follow:
                break
        for c in follow:
            st["followed"].add(c.node)
            if c.kind == "shortcut":
                st["sc_followed"] += 1
            else:
                st["ev_followed"] += 1
        step["follow"] = [c.node for c in follow]
        uncovered = [j for j in range(1, len(st["needs"]) + 1) if j not in st["covered"]]
        if decision.done and uncovered and not follow:
            step["note"] = f"done claimed with need {uncovered} uncovered"
            st["done_uncovered"] = True
        st["reviews"] += 1
        st["trace"].append(step)
        return follow, decision.done

    # ---- one question --------------------------------------------------
    def navigate(self, q):
        st = {"q": q, "needs": [], "n_planned": 0, "covered": set(), "opened": [],
              "scored": set(), "expanded": set(), "seen_leaves": set(), "seen_leaves_order": [], "followed": set(),
              "node_scores": {},
              "summaries_read": set(), "doc_rows_read": set(), "calls": 0, "api_calls": 0,
              "usage": {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0},
              "api_usage": {"input_tokens": 0, "output_tokens": 0},
              "rounds": 0, "reviews": 0, "trace": [], "done_uncovered": False,
              "sc_candidates": 0, "sc_kept": 0, "sc_review_offered": 0, "sc_followed": 0,
              "ev_links_offered": 0, "ev_followed": 0, "opened_via_shortcut": [], "opened_via_evidence": [],
              "pool": {}, "filled": []}
        stop_reason = "max_calls"
        try:
            self.plan(st)
            leaves = self.beam_descend(st, [Cand(self.root, "start", None, "root", 100)], self.width, "descent")
            st["leaves_reached"] = [c.node for c in leaves]
            self.open_from_leaves(st, leaves, "descent")
            if len(st["opened"]) >= self.budget:
                stop_reason = "budget"
            elif not leaves:
                stop_reason = "no_leaves"
            elif self.evidence is None and not self.shortcuts:
                stop_reason = "no_review"                 # pure beam + open: nothing to follow
            else:
                stop_reason = "max_reviews"
                for i in range(1, MAX_REVIEWS + 1):
                    res = self.review(st, i)
                    if res is None:
                        stop_reason = "no_targets"
                        break
                    follow, done = res
                    if not follow:
                        stop_reason = "done" if done else "no_follow"
                        break
                    starts = [Cand(c.node, "start", c.source, c.label, 100, c.via_shortcut, c.via_evidence) for c in follow]
                    leaves = self.beam_descend(st, starts, len(starts), f"review{i}")
                    self.open_from_leaves(st, leaves, f"review{i}")
                    if len(st["opened"]) >= self.budget:
                        stop_reason = "budget"
                        break
        except RuntimeError as error:
            if "max calls" not in str(error):
                raise
            stop_reason = "max_calls"
        if self.fill and len(st["opened"]) < self.budget:
            # fill the budget left from the reserve: the highest-rated articles seen but not opened
            reserve = sorted(((sc, d, leaf, cov) for d, (sc, leaf, cov) in st["pool"].items() if d not in st["opened"]),
                             key=lambda x: (-x[0], self.doc_rows[x[1]]["title"]))
            step = {"action": "fill", "pool_size": len(reserve), "opened_here": [], "opened_here_titles": [], "scores": []}
            for sc, d, leaf, cov in reserve:
                if len(st["opened"]) >= self.budget:
                    break
                st["opened"].append(d)
                st["filled"].append(d)
                step["opened_here"].append(d)
                step["opened_here_titles"].append(self.doc_rows[d]["title"])
                step["scores"].append(sc)
                if leaf.via_shortcut:
                    st["opened_via_shortcut"].append(d)
                if leaf.via_evidence:
                    st["opened_via_evidence"].append(d)
            if step["opened_here"]:
                st["trace"].append(step)
        gold = set(q["gold"])
        return {
            "qid": q["qid"], "question": q["question"], "gold": q["gold"], "n_gold": q["n_gold"],
            "opened": list(st["opened"]),
            "needs": st["needs"], "n_planned_needs": st["n_planned"], "covered_needs": sorted(st["covered"]),
            "width": self.width, "policy": self.policy,
            "summaries_read": len(st["summaries_read"]), "doc_rows_read": len(st["doc_rows_read"]),
            "summary_tokens_read": sum(num_tokens(self.summaries[n]["summary"]) for n in st["summaries_read"]),
            "llm_calls": st["calls"], **st["usage"],
            "api_calls_this_run": st["api_calls"], "api_input_tokens_this_run": st["api_usage"]["input_tokens"],
            "api_output_tokens_this_run": st["api_usage"]["output_tokens"],
            "rounds": st["rounds"], "reviews": st["reviews"], "leaves_reached": st.get("leaves_reached", []),
            "done_uncovered": st["done_uncovered"],
            "shortcut_use": {"candidates_scored": st["sc_candidates"], "kept_in_beam": st["sc_kept"],
                             "offered_in_review": st["sc_review_offered"], "followed_in_review": st["sc_followed"],
                             "opened_via_shortcut": st["opened_via_shortcut"],
                             "gold_via_shortcut": bool(gold & set(st["opened_via_shortcut"]))},
            "evidence_use": {"links_offered": st["ev_links_offered"], "followed": st["ev_followed"],
                             "needs_added": len(st["needs"]) - st["n_planned"],
                             "opened_via_evidence": st["opened_via_evidence"],
                             "gold_via_evidence": bool(gold & set(st["opened_via_evidence"]))},
            "fill": {"enabled": self.fill, "pool_size": len(st["pool"]), "filled": st["filled"],
                     "gold_via_fill": bool(gold & set(st["filled"]))},
            "stop_reason": stop_reason, "trace": st["trace"],
        }


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def dry_run(nav, q):
    print("=" * 70 + "\n[PLAN call]\n" + "=" * 70)
    print(PLAN_PROMPT.strip() + f"\n\nQUESTION: {q['question']}")
    needs = ["(information need 1 - filled by the plan call)"]
    root_children = [Cand(c, "child", nav.root, f"child of '{nav.summaries[nav.root]['title']}'")
                     for c in nav.nodes[nav.root]["children"]]
    prompt = nav.score_prompt(q["question"], needs, set(), root_children)
    print("\n" + "=" * 70 + "\n[SCORE call, round 1: the root's children]\n" + "=" * 70)
    print(prompt)
    print(f"\n[~{num_tokens(prompt)} tokens | system prompt ~{num_tokens(SCORE_PROMPT)} tokens]")
    g = q["gold"][0]
    if g not in nav.leaf_of:
        return
    leaf = nav.leaf_of[g]
    parent = nav.nodes[leaf]["parent_id"]
    if parent is not None:
        cands = [Cand(c, "child", parent, f"child of '{nav.summaries[parent]['title']}'") for c in nav.nodes[parent]["children"]]
        for sc in nav.shortcuts.get(parent, []):
            if sc["target"] in nav.summaries:
                cands.append(Cand(sc["target"], "shortcut", parent,
                                  f"KG shortcut from '{nav.summaries[parent]['title']}' - {shortcut_label(sc)}", via_shortcut=True))
        prompt = nav.score_prompt(q["question"], needs, set(), cands)
        print("\n" + "=" * 70 + f"\n[SCORE call at depth {nav.nodes[parent]['depth']}: children of the gold leaf's parent"
              f" + its shortcut targets ({len(cands)} candidates)]\n" + "=" * 70)
        print(prompt)
        print(f"\n[~{num_tokens(prompt)} tokens]")
    leaves = [Cand(leaf, "child", parent, "", 90)]
    prompt, docs = nav.leaf_prompt(q["question"], needs, set(), [], leaves)
    print("\n" + "=" * 70 + f"\n[OPEN call: articles of the gold leaf ({len(docs)} rows)]\n" + "=" * 70)
    print(prompt)
    print(f"\n[~{num_tokens(prompt)} tokens | system prompt ~{num_tokens(nav.leaf_prompt_text)} tokens]")
    st = {"q": q, "opened": [g], "followed": set(), "expanded": set(), "seen_leaves": {leaf}, "seen_leaves_order": [leaf],
          "node_scores": {}, "ev_links_offered": 0, "sc_review_offered": 0}
    ev_lines, facts, targets = nav.review_targets(st)
    if targets:
        prompt = nav.review_prompt(q["question"], needs, {1}, [g], ev_lines, targets, max(1, min(nav.width, nav.budget - 1)))
        print("\n" + "=" * 70 + f"\n[REVIEW call after opening the gold article '{nav.doc_rows[g]['title']}'; "
              f"{len(targets)} follow-up targets]\n" + "=" * 70)
        print(prompt)
        print(f"\n[~{num_tokens(prompt)} tokens | system prompt ~{num_tokens(nav.review_prompt_text)} tokens]")
    else:
        print("\n[no review: no evidence links or shortcut targets from the gold leaf]")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tree", choices=list(TREES), default="leiden")
    parser.add_argument("--policy", default="none", help="comma list of " + ",".join(POLICIES) + " or 'all'")
    parser.add_argument("--widths", default=",".join(map(str, WIDTHS)), help="comma list of beam widths")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--max-calls", type=int, default=MAX_CALLS)
    parser.add_argument("--split", choices=["dev", "heldout"], default="dev")
    parser.add_argument("--limit", type=int, default=None, help="first N questions (pilot)")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--no-evidence", action="store_true", help="ablation: no evidence notes/links/new needs")
    parser.add_argument("--oracle", action="store_true", help="gold-aware scorer instead of the LLM (ceiling + plumbing test)")
    parser.add_argument("--fill", action="store_true",
                        help="rate every listed article; open ratings >= %d now and fill the budget left at the end "
                             "from the reserve (systems beam...fill_W<W>_<policy>_B<B>)" % OPEN_SCORE)
    parser.add_argument("--dry-run", action="store_true", help="print the prompts for the first question, no API")
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
    family = "beam" + ("oracle" if args.oracle else "") + ("noev" if args.no_evidence else "") + ("fill" if args.fill else "")
    variant = VARIANT + ("noev" if args.no_evidence else "") + ("oracle" if args.oracle else "") + ("fill" if args.fill else "")

    if args.dry_run:
        nav = BeamNavigator(args.tree, policies[0], widths[0], args.budget, args.max_calls, evidence=not args.no_evidence,
                            fill=args.fill)
        dry_run(nav, questions[0])
        print(f"\nmodel {MODEL_NAME} (reasoning {REASONING_EFFORT}) | budget {args.budget} | widths {widths} | max calls "
              f"{args.max_calls} | questions {len(questions)} | policies {policies} | tree {args.tree} | variant {variant}")
        return

    client = None
    if not args.oracle:
        if not os.getenv("OPENAI_API_KEY"):
            parser.error("Missing OPENAI_API_KEY in .env")
        from openai import OpenAI
        client = OpenAI(timeout=180.0, max_retries=3)

    for width in widths:
        for policy in policies:
            nav = BeamNavigator(args.tree, policy, width, args.budget, args.max_calls, client=client,
                                use_cache=not args.no_cache, evidence=not args.no_evidence, oracle=args.oracle,
                                fill=args.fill)
            system = f"{family}_W{width}_{policy}_B{args.budget}"
            print(f"\n========== BEAM {variant} W={width} policy={policy} tree={args.tree} budget={args.budget} "
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
                    su, eu = r["shortcut_use"], r["evidence_use"]
                    print(f"  {r['qid']} gold {got}/{len(gold)} | needs {r['n_planned_needs']}+{eu['needs_added']} "
                          f"covered {len(r['covered_needs'])} | opened {len(r['opened'])} | calls {r['llm_calls']} (cached {cached}) "
                          f"| rounds {r['rounds']} reviews {r['reviews']} | sc kept {su['kept_in_beam']} followed {su['followed_in_review']} "
                          f"ev followed {eu['followed']}" + (f" fill {len(r['fill']['filled'])}" if args.fill else "")
                          + f" | {r['stop_reason']:11s} | {r['question'][:50]}")
            ordered = [records[q["qid"]] for q in questions if q["qid"] in records]
            if not ordered:
                print("no results")
                continue
            config = {"model": MODEL_NAME, "reasoning_effort": REASONING_EFFORT, "budget": args.budget,
                      "max_calls": args.max_calls, "policy": policy, "variant": variant, "width": width,
                      "evidence": not args.no_evidence, "oracle": args.oracle, "fill": args.fill,
                      "open_score": OPEN_SCORE if args.fill else None, "schema_version": SCHEMA_VERSION,
                      "min_candidate_score": MIN_CANDIDATE_SCORE, "max_rounds": MAX_ROUNDS, "max_reviews": MAX_REVIEWS,
                      "prompt_sha256": sha256_text(PLAN_PROMPT + SCORE_PROMPT + nav.leaf_prompt_text + nav.review_prompt_text),
                      "limit": args.limit}
            path = save_results(system, args.tree, args.split, config, ordered)
            tokens = {k: sum(r[k] for r in ordered) for k in ("input_tokens", "output_tokens", "reasoning_tokens")}
            billed = {"api_calls": sum(r["api_calls_this_run"] for r in ordered),
                      "input_tokens": sum(r["api_input_tokens_this_run"] for r in ordered),
                      "output_tokens": sum(r["api_output_tokens_this_run"] for r in ordered)}
            stops = Counter(r["stop_reason"] for r in ordered)
            n = len(ordered)
            sc = {k: sum(r["shortcut_use"][k] for r in ordered) for k in ("candidates_scored", "kept_in_beam", "followed_in_review")}
            sc["gold_via_shortcut"] = sum(r["shortcut_use"]["gold_via_shortcut"] for r in ordered)
            ev = {k: sum(r["evidence_use"][k] for r in ordered) for k in ("links_offered", "followed", "needs_added")}
            ev["gold_via_evidence"] = sum(r["evidence_use"]["gold_via_evidence"] for r in ordered)
            print(quick_report(system, ordered, budgets=(args.budget,)))
            print(f"  rounds/q {sum(r['rounds'] for r in ordered) / n:.1f} | reviews/q {sum(r['reviews'] for r in ordered) / n:.1f} "
                  f"| leaves reached/q {sum(len(r['leaves_reached']) for r in ordered) / n:.1f} "
                  f"| done with a need uncovered {sum(r['done_uncovered'] for r in ordered)}")
            print(f"  shortcuts {sc} | evidence {ev}")
            if args.fill:
                print(f"  fill: articles filled {sum(len(r['fill']['filled']) for r in ordered)} "
                      f"| questions with a gold article opened by the fill {sum(r['fill']['gold_via_fill'] for r in ordered)} "
                      f"| opened/q {sum(len(r['opened']) for r in ordered) / n:.2f}")
            print(f"  logical cost {tokens} | billed this run {billed} | stop reasons {dict(stops)}")
            print(f"  saved {path}")
            if len(ordered) < len(questions):
                print(f"  {len(questions) - len(ordered)} questions failed; rerun (cached decisions are free)")


if __name__ == "__main__":
    main()
    