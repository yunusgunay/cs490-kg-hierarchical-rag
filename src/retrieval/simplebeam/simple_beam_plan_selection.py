"""Reuse one query's existing retrieval plan in its single final selection.

This callable wraps a caller-owned ask; it adds no model stage or source lookup.
Create one instance for each retrieval/question, never share it across queries.
"""
from pydantic import BaseModel

from .simple_beam import NavPlan


PROVISIONAL_NEEDS_HEADER = "PROVISIONAL RETRIEVAL NEEDS (from the earlier retrieval plan):"
PROVISIONAL_NEEDS_INSTRUCTION = (
    "The original question below is authoritative. These provisional needs may be "
    "mistaken or incomplete. Use them only where supported by the question; they "
    "are not new facts, constraints, or requirements."
)
PROVISIONAL_NEEDS_FOOTER = "END PROVISIONAL RETRIEVAL NEEDS"
PLAN_SELECTION_POLICY = {
    "state_scope": "one_question_per_wrapper_instance",
    "capture": "actual_validated_NavPlan_response",
    "normalization": "strip_nonempty_first_three_or_original_question",
    "clear_state_before_every_plan_attempt": True,
    "prefix_stages": ["select", "select_retry"],
    "original_question_authoritative": True,
    "return_provider_response_unchanged": True,
    "additional_model_stages": 0,
}


class PlanConditionedSelection:
    """``ask(system, user, schema, stage)`` with query-local plan conditioning.

    Plan/routing calls are forwarded unchanged. A failed or invalid plan leaves
    no usable previous plan. Selection requires a successful validated plan and
    prefixes its own incoming user string once, without mutating a retry buffer
    or feeding a previous selection's answer into a later call.
    """

    def __init__(self, ask, question):
        if not callable(ask):
            raise ValueError("ask must be callable")
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be nonempty text")
        self.ask, self.question = ask, question
        self.needs = None

    def __call__(self, system, user, schema, stage):
        if stage == "plan":
            self.needs = None
            response = self.ask(system, user, schema, stage)
            value = response.model_dump() if isinstance(response, BaseModel) else response
            plan = NavPlan.model_validate(value)
            normalized = [need.strip() for need in plan.needs if need.strip()][:3] or [self.question]
            self.needs = tuple(normalized)
            return response
        if stage in ("select", "select_retry"):
            if self.needs is None:
                raise RuntimeError("A successful validated plan is required before selection")
            prefix = "\n".join([
                PROVISIONAL_NEEDS_HEADER,
                PROVISIONAL_NEEDS_INSTRUCTION,
                *[f"  [{i}] {need}" for i, need in enumerate(self.needs, 1)],
                PROVISIONAL_NEEDS_FOOTER,
                "", "",
            ])
            return self.ask(system, prefix + user, schema, stage)
        return self.ask(system, user, schema, stage)
