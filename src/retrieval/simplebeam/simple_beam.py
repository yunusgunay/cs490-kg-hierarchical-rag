"""Portable one-descent LLM beam with one direct document selection.

The caller owns every model, cache, request limit, and transport. This module
accepts only a tree/index and question text; it has no corpus paths, query IDs,
gold labels, embeddings, KG lookups, provider client, or dataset taxonomy.
Planning/routing and the title selector preserve the original research-beam-r1
message strings and schema JSON for exact request-cache reuse where inputs match.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import copy
from typing import Callable

from pydantic import BaseModel, Field, create_model


PLAN = """Plan evidence retrieval for the user's question over a document collection.
List the distinct information needs and plausible interpretations that documents
would need to support. Preserve explicit constraints and scope. For an ambiguous
question retain its reasonable interpretations rather than silently assuming one.
Use between one and three concise needs, without inventing unrelated requirements.
This is a retrieval plan, not an answer. Return only the required structured plan.
"""

SCORE = """Route a question through a hierarchical index of documents. The node
descriptions are lossy summaries of their descendants, not exhaustive inventories.
Rate EVERY listed candidate from 0 to 100 for the chance it contains useful
evidence for the question, including complementary evidence for a different need.
Match the complete question and its constraints, not just repeated entity names.
Consider minority topics in broad or mixed groups. An exclusion in a summary is
not proof that no descendant can help. A linked node is a candidate, not an answer.
Use high scores for direct or complementary coverage, medium scores for plausible
interpretations, low scores for weak overlap, and zero for clearly unrelated scope.
Do not reduce an alternative's score solely because another candidate is good.
Return a score for each candidate index and one concise comparative reason.
"""

TITLE_SELECT = "Rank the candidate documents by relevance to the user's question. Return exactly the requested number of distinct indices, most relevant first, and a short reason. Use the question, document titles and document descriptions. Do not answer the question."

MAX_ROUNDS = 8
MIN_CANDIDATE_SCORE = 10


class NavPlan(BaseModel):
    needs: list[str]


class CandidateScore(BaseModel):
    index: int
    score: int


class ScoreDecision(BaseModel):
    scores: list[CandidateScore]
    reason: str


@dataclass
class _Candidate:
    node: str
    kind: str
    score: int = 0
    label: str = ""


class SimpleBeamRetriever:
    """Single globally pruned beam, followed by one direct ranked source set.

    ``tree`` maps node IDs to ``children``, ``is_leaf`` and ordered ``doc_ids``.
    ``summaries`` maps those IDs to title/summary and optional question_types,
    key_terms, coverage, depth and size. The root depth is zero; omitted depth
    and size are derived. ``doc_rows`` maps document IDs to title and optional
    key_entities; other fields are never exposed to the model.

    ``ask(system, user, schema, stage)`` returns the parsed Pydantic object or a
    mapping. Stages are plan, score_round_N, select and, only for invalid picked
    indices, select_retry. No second descent or additional selector is invoked.
    """
    def __init__(self, tree, summaries, doc_rows, root, ask: Callable,
                 *, width=8, budget=5, max_rounds=MAX_ROUNDS,
                 card_view="title", selector_prompt=TITLE_SELECT):
        for name, value in (("width", width), ("budget", budget), ("max_rounds", max_rounds)):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if card_view not in ("title", "terms"):
            raise ValueError("card_view must be title or terms")
        if not callable(ask) or not isinstance(selector_prompt, str) or not selector_prompt.strip():
            raise ValueError("A callable ask and nonempty selector prompt are required")
        if not isinstance(tree, Mapping) or not tree or root not in tree:
            raise ValueError("tree must map node IDs and contain root")
        if not isinstance(summaries, Mapping) or set(summaries) != set(tree):
            raise ValueError("Every tree node must have one summary")
        if not isinstance(doc_rows, Mapping) or not doc_rows:
            raise ValueError("doc_rows must map the document inventory")
        self.width, self.budget, self.max_rounds = width, budget, max_rounds
        self.card_view, self.selector_prompt = card_view, selector_prompt.strip()
        self.root, self.ask = root, ask
        self.tree = copy.deepcopy(dict(tree))
        depths = self._validate_tree(self.tree, root, set(doc_rows))
        self.summaries = {}
        for node, value in summaries.items():
            if not isinstance(value, Mapping) or not all(isinstance(value.get(k), str) for k in ("title", "summary")):
                raise ValueError("Each node requires a title and summary")
            item = dict(title=value["title"], summary=value["summary"],
                        depth=value.get("depth", depths[node]), size=value.get("size", len(self.tree[node]["doc_ids"])))
            if type(item["depth"]) is not int or item["depth"] != depths[node] or type(item["size"]) is not int or item["size"] != len(self.tree[node]["doc_ids"]):
                raise ValueError("Summary depth/size disagrees with tree")
            for key in ("question_types", "key_terms", "coverage"):
                entries = value.get(key, [])
                if not isinstance(entries, list) or not all(isinstance(text, str) for text in entries):
                    raise ValueError("Node metadata fields must be lists of strings")
                item[key] = list(entries)
            self.summaries[node] = item
        self.doc_rows = {}
        for doc, value in doc_rows.items():
            if not isinstance(value, Mapping) or not isinstance(value.get("title"), str):
                raise ValueError("Each document requires a title")
            terms = value.get("key_entities", [])
            if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
                raise ValueError("Document key_entities must be a list of strings")
            self.doc_rows[doc] = dict(title=value["title"], key_entities=list(terms))
        # These names and bounds match the original direct-selection cache schema.
        self.selection_cls = create_model("JointEvidenceSet",
            picks=(list[int], Field(min_length=1, max_length=budget)), reason=(str, ...))

    @staticmethod
    def _validate_tree(tree, root, documents):
        parents = {node: [] for node in tree}
        for node, row in tree.items():
            if not isinstance(node, str) or not isinstance(row, Mapping):
                raise ValueError("Node IDs must be strings and nodes mappings")
            children, docs = row.get("children"), row.get("doc_ids")
            if type(row.get("is_leaf")) is not bool or not isinstance(children, list) or not all(isinstance(child, str) for child in children) or len(set(children)) != len(children):
                raise ValueError("Each node needs unique children and boolean is_leaf")
            if not isinstance(docs, list) or not docs or not all(isinstance(doc, str) for doc in docs) or len(set(docs)) != len(docs):
                raise ValueError("Each node needs a nonempty unique document list")
            if row["is_leaf"] != (not children):
                raise ValueError("Leaves have no children; internal nodes have children")
            for child in children:
                if child not in tree:
                    raise ValueError("Unknown child node")
                parents[child].append(node)
        if parents[root] or any(len(values) != 1 for node, values in parents.items() if node != root):
            raise ValueError("Tree nodes require one parent, except root")
        depths, active = {}, set()
        def visit(node, depth):
            if node in active or node in depths:
                raise ValueError("Tree must be acyclic and have one parent")
            active.add(node); depths[node] = depth
            row = tree[node]
            if row["children"]:
                child_docs = []
                for child in row["children"]:
                    visit(child, depth + 1)
                    child_docs.extend(tree[child]["doc_ids"])
                if len(child_docs) != len(set(child_docs)) or set(child_docs) != set(row["doc_ids"]):
                    raise ValueError("Child documents must partition their parent")
            active.remove(node)
        visit(root, 0)
        if set(depths) != set(tree) or set(tree[root]["doc_ids"]) != documents:
            raise ValueError("Reachable leaves must partition the full document inventory")
        return depths

    def _decision(self, system, user, schema, stage):
        value = self.ask(system, user, schema, stage)
        if isinstance(value, BaseModel):
            value = value.model_dump()
        return schema.model_validate(value)

    @staticmethod
    def _needs_lines(needs):
        return ["INFORMATION NEEDS (your plan):"] + [
            f"  [{i}] {need}   - not yet covered" for i, need in enumerate(needs, 1)]

    def _score_user(self, question, needs, candidates):
        lines = [f"QUESTION: {question}", ""] + self._needs_lines(needs) + ["", "CANDIDATES (rate every one):"]
        for i, candidate in enumerate(candidates, 1):
            node = self.tree[candidate.node]; summary = self.summaries[candidate.node]
            what = "leaf" if node["is_leaf"] else "node"
            lines.append(f"[{i}] {summary['title']} ({what}, depth {summary['depth']}, {summary['size']} articles) - {candidate.label}")
            lines.append(f"    {summary['summary']}")
            if summary["question_types"]:
                lines.append("    typical questions: " + " | ".join(summary["question_types"][:4]))
            if summary["key_terms"]:
                lines.append("    key terms: " + ", ".join(summary["key_terms"][:10]))
        lines += ["", f"Return a score (0-100) for every index 1..{len(candidates)}, and a short reason."]
        prompt = "\n".join(lines)
        coverage = [f"Candidate [{i}]: " + " | ".join(self.summaries[c.node]["coverage"])
                    for i, c in enumerate(candidates, 1) if self.summaries[c.node]["coverage"]]
        if coverage:
            prompt += "\n\nADDITIONAL POSITIVE COVERAGE (same candidate numbering):\n" + "\n".join(coverage)
        return prompt

    def _selection_user(self, question, documents):
        label = "titles only" if self.card_view == "title" else "titles and key concepts"
        lines = [f"QUESTION: {question}", "", "", f"CANDIDATE DOCUMENTS ({len(documents)}; {label}):"]
        for i, doc in enumerate(documents, 1):
            text = f"[{i}] {self.doc_rows[doc]['title']}"
            if self.card_view == "terms":
                text += "\nKey concepts: " + ", ".join(self.doc_rows[doc]["key_entities"])
            lines.append(text)
        lines.append(f"Return exactly {min(self.budget, len(documents))} distinct indices in relevance order, covering the question jointly.")
        return "\n".join(lines)

    def retrieve(self, question):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        trace = []
        plan = self._decision(PLAN, f"QUESTION: {question}", NavPlan, "plan")
        needs = [need.strip() for need in plan.needs if need.strip()][:3] or [question]
        trace.append(dict(action="plan", stage="plan", needs=list(needs), decision=plan.model_dump()))
        frontier = [_Candidate(self.root, "start", 100, "root")]
        scored = {self.root}; rounds = 0
        for round_number in range(1, self.max_rounds + 1):
            internal = [c for c in frontier if not self.tree[c.node]["is_leaf"]]
            if not internal:
                break
            retained = [_Candidate(c.node, "retained", c.score, "kept from the previous round")
                        for c in frontier if self.tree[c.node]["is_leaf"]]
            new = []; seen = {c.node for c in retained}
            for parent in internal:
                title = self.summaries[parent.node]["title"]
                for child in self.tree[parent.node]["children"]:
                    if child in scored or child in seen:
                        continue
                    seen.add(child)
                    new.append(_Candidate(child, "child", label=f"child of '{title}'"))
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
            ranked = sorted(candidates, key=lambda c: (-c.score, 0 if c.kind == "retained" else 1, c.node))
            kept = [c for c in ranked if c.score >= MIN_CANDIDATE_SCORE][:self.width] or ranked[:1]
            rounds += 1
            trace.append(dict(action="score", stage=stage, round=round_number,
                nodes=[c.node for c in internal], candidates=[dict(node=c.node, kind=c.kind, score=c.score) for c in candidates],
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
        return dict(opened=opened, candidate_ids=documents, trace=trace, needs=needs,
                    rounds=rounds, leaves_reached=leaves)
