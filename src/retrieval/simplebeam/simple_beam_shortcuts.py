"""Generic optional-shortcut extension of the frozen one-descent beam.

The caller supplies an ordered node adjacency mapping; no graph or corpus is
loaded here. With policy='none', retrieval delegates directly to the original
portable core. With policy='typed', shortcut candidates and labels reproduce the
original research navigator's first descent, without its alternate-start waves.
"""
from collections.abc import Mapping
import copy
from dataclasses import dataclass
import json

from .simple_beam import (MAX_ROUNDS, MIN_CANDIDATE_SCORE, PLAN, SCORE,
                          TITLE_SELECT, NavPlan, ScoreDecision)
from .simple_beam import SimpleBeamRetriever as _SimpleBeamRetriever


SHORTCUT_POLICY = {
    "policies": ["none", "typed"],
    "candidate_order": "retained_then_each_internal_stored_children_then_stored_shortcuts",
    "duplicate_policy": "first_node_offering_wins",
    "revisit_ban": "every_previously_scored_node_including_pruned_candidates",
    "tie_order": ["retained", "child", "shortcut", "node_id"],
    "outgoing_edges_from": "current_internal_nodes_only",
    "unknown_target": "skip_like_legacy",
    "query_time_edge_reranking_or_filtering": False,
    "extra_waves": 0,
}


def fact_text(fact, quote_chars=180):
    """Exact original policy rendering, including fallible signed relations."""
    text = f"{fact['subject']} {fact['predicate']} {fact['object']}"
    text += f" [polarity={fact.get('polarity', 'UNSPECIFIED')}; certainty={fact.get('certainty', 'UNSPECIFIED')}]"
    if fact.get("condition"):
        text += " IF " + fact["condition"]
    if fact.get("evidence"):
        quote = " ".join(fact["evidence"].split())[:quote_chars]
        text += " | source: " + json.dumps(quote, ensure_ascii=False)
    return text


def shortcut_text(link):
    """Exact legacy support/evidence order and label character bounds."""
    facts, entities = [], []
    for support in link.get("support", [])[:3]:
        for evidence in support.get("evidence", [])[:2]:
            entities.extend([evidence["via"]] if evidence.get("via") else evidence.get("shared_entities", []))
            if evidence.get("predicate"):
                facts.append(fact_text(evidence, 110))
            for fact in evidence.get("relations", [])[:1]:
                facts.append(fact_text(fact, 110))
    label = "linked entities: " + ", ".join(dict.fromkeys(entities))[:180]
    if facts:
        label += "; " + "; ".join(list(dict.fromkeys(facts))[:2])
    return label


def _validate_fact(fact):
    if not isinstance(fact, Mapping) or not all(isinstance(fact.get(k), str) for k in ("subject", "predicate", "object")):
        raise ValueError("Shortcut relations require text subject, predicate and object")
    for key in ("polarity", "certainty", "condition", "evidence"):
        if fact.get(key) is not None and not isinstance(fact[key], str):
            raise ValueError("Shortcut relation metadata must be text or None")


def _ordered_adjacency(shortcuts):
    if not isinstance(shortcuts, Mapping):
        raise ValueError("shortcuts must be an ordered node-to-link-list mapping")
    result = copy.deepcopy(dict(shortcuts))
    for source, links in result.items():
        if not isinstance(source, str) or not isinstance(links, list):
            raise ValueError("Shortcut sources must be strings with ordered link lists")
        for link in links:
            if not isinstance(link, Mapping) or not isinstance(link.get("target"), str):
                raise ValueError("Each shortcut requires a text target node")
            supports = link.get("support", [])
            if not isinstance(supports, list):
                raise ValueError("Shortcut support must be an ordered list")
            for support in supports[:3]:
                if not isinstance(support, Mapping) or not isinstance(support.get("evidence", []), list):
                    raise ValueError("Shortcut evidence must be an ordered list")
                for evidence in support.get("evidence", [])[:2]:
                    if not isinstance(evidence, Mapping):
                        raise ValueError("Shortcut evidence entries must be mappings")
                    if evidence.get("via") is not None and not isinstance(evidence["via"], str):
                        raise ValueError("Shortcut via labels must be text")
                    entities = evidence.get("shared_entities", [])
                    if not isinstance(entities, list) or not all(isinstance(e, str) for e in entities):
                        raise ValueError("Shared shortcut entities must be text lists")
                    if evidence.get("predicate"):
                        _validate_fact(evidence)
                    relations = evidence.get("relations", [])
                    if not isinstance(relations, list):
                        raise ValueError("Shortcut relations must be an ordered list")
                    for fact in relations[:1]:
                        _validate_fact(fact)
    return result


@dataclass
class _Candidate:
    node: str
    kind: str
    score: int = 0
    label: str = ""
    source: str | None = None
    via_shortcut: bool = False


class SimpleBeamRetriever(_SimpleBeamRetriever):
    """Same core interface, plus ``policy`` and caller-owned ``shortcuts``.

    Adjacency list order is meaningful. Unknown target IDs are skipped, matching
    the legacy navigator; graph cycles and self-edges are suppressed by its
    ever-scored-node rule. No ancestor/subtree or relation-type filter is added.
    Policy none ignores adjacency and delegates to the frozen base verbatim.
    The existing PlanConditionedSelection wrapper can be passed as ``ask``.
    """

    def __init__(self, tree, summaries, doc_rows, root, ask,
                 *, width=8, budget=5, max_rounds=MAX_ROUNDS,
                 card_view="title", selector_prompt=TITLE_SELECT,
                 policy="none", shortcuts=None):
        if policy not in ("none", "typed"):
            raise ValueError("policy must be none or typed")
        self.policy = policy
        self.shortcuts = _ordered_adjacency({} if shortcuts is None else shortcuts) if policy == "typed" else {}
        super().__init__(tree, summaries, doc_rows, root, ask, width=width,
                         budget=budget, max_rounds=max_rounds,
                         card_view=card_view, selector_prompt=selector_prompt)

    def retrieve(self, question):
        if self.policy == "none":
            return super().retrieve(question)
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        trace = []
        plan = self._decision(PLAN, f"QUESTION: {question}", NavPlan, "plan")
        needs = [need.strip() for need in plan.needs if need.strip()][:3] or [question]
        trace.append(dict(action="plan", stage="plan", needs=list(needs), decision=plan.model_dump()))
        frontier = [_Candidate(self.root, "start", 100, "root")]
        scored = {self.root}; rounds = 0; shortcut_candidates = 0; shortcuts_kept = 0
        for round_number in range(1, self.max_rounds + 1):
            internal = [c for c in frontier if not self.tree[c.node]["is_leaf"]]
            if not internal:
                break
            retained = [_Candidate(c.node, "retained", c.score, "kept from the previous round", c.source, c.via_shortcut)
                        for c in frontier if self.tree[c.node]["is_leaf"]]
            new = []; seen = {c.node for c in retained}
            for parent in internal:
                title = self.summaries[parent.node]["title"]
                for child in self.tree[parent.node]["children"]:
                    if child in scored or child in seen:
                        continue
                    seen.add(child)
                    new.append(_Candidate(child, "child", label=f"child of '{title}'",
                                          source=parent.node, via_shortcut=parent.via_shortcut))
                for link in self.shortcuts.get(parent.node, []):
                    target = link["target"]
                    if target not in self.summaries or target in scored or target in seen:
                        continue
                    seen.add(target); shortcut_candidates += 1
                    new.append(_Candidate(target, "shortcut",
                        label=f"KG shortcut from '{title}' - {shortcut_text(link)}",
                        source=parent.node, via_shortcut=True))
            if not new:
                frontier = retained
                break
            candidates = retained + new
            stage = f"score_round_{round_number}"
            decision = self._decision(SCORE, self._score_user(question, needs, candidates), ScoreDecision, stage)
            given = {}
            for entry in decision.scores:
                if 1 <= entry.index <= len(candidates) and entry.index - 1 not in given:
                    given[entry.index - 1] = max(0, min(100, entry.score))
            for i, candidate in enumerate(candidates):
                candidate.score = given.get(i, 0)
            scored.update(c.node for c in new)
            ranked = sorted(candidates, key=lambda c: (-c.score, {"retained": 0, "child": 1, "shortcut": 2}[c.kind], c.node))
            kept = [c for c in ranked if c.score >= MIN_CANDIDATE_SCORE][:self.width] or ranked[:1]
            shortcuts_kept += sum(c.kind == "shortcut" for c in kept)
            rounds += 1
            trace.append(dict(action="score", stage=stage, round=round_number,
                nodes=[c.node for c in internal], candidates=[dict(node=c.node, kind=c.kind, score=c.score,
                    source=c.source, via_shortcut=c.via_shortcut, label=c.label) for c in candidates],
                kept=[c.node for c in kept], decision=decision.model_dump(), reason=decision.reason))
            frontier = kept
        leaves = [c.node for c in frontier if self.tree[c.node]["is_leaf"]]
        if not leaves:
            raise RuntimeError("Beam search reached no candidate leaves")
        documents = [doc for leaf in leaves for doc in self.tree[leaf]["doc_ids"]]
        if len(documents) != len(set(documents)):
            raise RuntimeError("Final leaves violate the document partition")
        prompt = self._selection_user(question, documents); count = min(self.budget, len(documents))
        for attempt in range(2):
            stage = "select" if attempt == 0 else "select_retry"
            decision = self._decision(self.selector_prompt, prompt, self.selection_cls, stage)
            picks = list(decision.picks)
            valid = len(picks) == count and len(set(picks)) == count and all(1 <= i <= len(documents) for i in picks)
            trace.append(dict(action="select", stage=stage, decision=decision.model_dump(), valid_indices=valid))
            if valid:
                break
            if attempt:
                raise RuntimeError("Invalid joint selection after validation retry")
            prompt += f"\nYour previous indices {picks} were invalid. Return exactly {count} distinct indices in range 1..{len(documents)}."
        opened = [documents[index - 1] for index in picks]
        trace.append(dict(action="open", stage="direct_selection", opened_here=list(opened)))
        via_leaves = [c.node for c in frontier if self.tree[c.node]["is_leaf"] and c.via_shortcut]
        via_documents = {doc for leaf in via_leaves for doc in self.tree[leaf]["doc_ids"]}
        return dict(opened=opened, candidate_ids=documents, trace=trace, needs=needs,
                    rounds=rounds, leaves_reached=leaves,
                    shortcut_use=dict(candidates_scored=shortcut_candidates, kept_in_beam=shortcuts_kept,
                                      leaves_via_shortcut=via_leaves,
                                      opened_via_shortcut=[doc for doc in opened if doc in via_documents]))
