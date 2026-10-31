# src/kg/extract_relations_v5.py
"""
KG extraction v5 — per-document, typed entities, compact predicates.

Changes relative to v4
----------------------
* Unit of extraction is the whole article (no 300-word chunks, no
  overlap). Very long articles are split at paragraph boundaries and
  later parts see the entities found in earlier parts.
* Entity layer: short canonical names (1-4 words) with a type.
  Relations reference entity names; specifics live in evidence /
  condition, never in the node name.
* 9 predicates + polarity/certainty flags replace 16 predicates.
* The product named in the title prefix ("Wix Stores: ...") is passed
  to the model and stored on the record for the graph builder.
* Concurrent requests, per-document checkpoint, resumable.
* `--report` prints the same connectivity diagnostics that exposed the
  v4 problem (entity length, singleton share, components), so pilot
  runs can be judged before spending the full budget.

Usage
-----
    python src/kg/extract_relations_v5.py --dry-run
    python src/kg/extract_relations_v5.py --sample 20       # pilot
    python src/kg/extract_relations_v5.py --report          # gates
    python src/kg/extract_relations_v5.py                   # full run
"""
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data/processed/subset/corpus.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/subset"

load_dotenv(PROJECT_ROOT / ".env")

MODEL_NAME = os.getenv("LLM_MODEL", "gpt-6-luna").strip()
REASONING_EFFORT = os.getenv(
    "LLM_REASONING_EFFORT", "medium"
).strip().lower()

IS_REASONING_MODEL = MODEL_NAME.startswith(
    ("gpt-5", "gpt-6", "o1", "o3", "o4")
)

SCHEMA_VERSION = "kg-extraction-v5"

# Articles longer than this are split at paragraph boundaries.
MAX_SEGMENT_CHARS = 14_000

MAX_COMPLETION_TOKENS = 8_000
MAX_ENTITY_WORDS = 4          # soft target given to the model
HARD_MAX_ENTITY_WORDS = 5     # relations above this are rejected
MAX_EVIDENCE_CHARS = 400

DEFAULT_WORKERS = 6

SAFE_MODEL_NAME = re.sub(r"[^a-zA-Z0-9_-]", "_", MODEL_NAME)
RUN_NAME = SAFE_MODEL_NAME + (
    f"_{REASONING_EFFORT}" if IS_REASONING_MODEL else ""
)

EXTRACTION_PATH = OUTPUT_DIR / f"kg_v5_{RUN_NAME}_extracted.jsonl"


# ============================================================
# 2. SCHEMA
# ============================================================

EntityType = Literal[
    "PRODUCT",    # Wix product, app or editor
    "FEATURE",    # capability or object inside a product
    "SETTING",    # a specific option, toggle or field
    "LOCATION",   # dashboard page, tab, panel or menu
    "PLAN",       # plan or subscription tier
    "TASK",       # user goal, verb + noun ("publish site")
    "ISSUE",      # problem, error, failure
    "EXTERNAL",   # non-Wix service (Google, PayPal, Meta Pixel)
    "CONCEPT",    # anything else (domain, sitemap, VAT)
]

Predicate = Literal[
    "PART_OF",
    "LOCATED_IN",
    "REQUIRES",
    "APPLIES_TO",
    "SUPPORTS",
    "HAS_LIMITATION",
    "CAUSES",
    "RESOLVED_BY",
    "RELATED_TO",
]

Polarity = Literal["POSITIVE", "NEGATIVE"]
Certainty = Literal["DEFINITE", "POSSIBLE"]


class Entity(BaseModel):
    name: str
    type: EntityType
    aliases: list[str]


class Relation(BaseModel):
    subject: str
    predicate: Predicate
    object: str
    polarity: Polarity
    certainty: Certainty
    condition: str | None
    evidence: str


class ExtractionResult(BaseModel):
    entities: list[Entity]
    relations: list[Relation]


# ============================================================
# 3. PROMPT
# ============================================================

EXTRACTION_PROMPT = """
You extract a compact, well-grounded knowledge graph from ONE Wix
help-center article. The graph is used to organize thousands of
articles and to connect related articles, so entity names must be
reusable across articles.

OUTPUT
- entities: the concepts this article is about.
- relations: typed links between those entities, each grounded in a
  verbatim quote from the article.

ENTITY RULES
1. An entity name is a short canonical noun phrase, 1-4 words, no
   leading article, singular where natural, Title Case only for
   proper names.
   Good: "Wix Stores", "flat rate shipping", "SEO Settings",
         "Google Search Console", "guest checkout", "Premium plan",
         "payout", "301 redirect", "Invoices tab".
   Bad:  "connecting a payment provider" (action clause),
         "the invoices tab in the site's dashboard" (too long),
         "a connected payment provider that supports recurring
         payments" (a clause, not a concept),
         "An upgraded site" (use "Premium plan").
2. Specifics (numbers, lists, conditions, UI steps) belong in the
   relation's evidence or condition, never in the entity name.
3. Use one name per concept for the whole article. For the main
   product use exactly the PRODUCT string given in the input; never
   create variants such as "the Editor" or "Wix's editor".
   List surface forms you merged under "aliases".
4. Types:
   PRODUCT   Wix product, app or editor (Wix Stores, Studio Editor)
   FEATURE   capability or object inside a product (custom text field)
   SETTING   a specific option, toggle or field (autoplay toggle)
   LOCATION  dashboard page, tab, panel, menu (Shipping settings)
   PLAN      plan or subscription tier (Business Premium plan)
   TASK      user goal as verb + noun (publish site, connect domain)
   ISSUE     problem, error, failure (failed payout, site not indexed)
   EXTERNAL  non-Wix service (Google Workspace, PayPal, Meta Pixel)
   CONCEPT   anything else (domain, sitemap, VAT, chargeback)
5. Aim for 4-15 entities. Do not list trivial UI words (button,
   click, page, icon) or pronouns as entities.

RELATION RULES
6. Predicates:
   PART_OF        feature/setting belongs to a product or larger feature
   LOCATED_IN     setting/feature/task is accessed in a LOCATION
   REQUIRES       mandatory prerequisite (plan, app, account state,
                  prior step); never for optional methods
   APPLIES_TO     a rule, limit, fee or behavior applies to a scope
                  (product, plan, region, element type)
   SUPPORTS       capability statement; polarity NEGATIVE = explicitly
                  unsupported / not possible
   HAS_LIMITATION explicit constraint (quantity caps, restrictions)
   CAUSES         X leads to Y; certainty POSSIBLE for "may/can cause"
   RESOLVED_BY    issue is fixed or worked around by a task or setting;
                  certainty POSSIBLE for workarounds / "try"
   RELATED_TO     only when a real link exists and nothing else fits
7. polarity: NEGATIVE for negated statements (not supported, cannot,
   does not apply). Otherwise POSITIVE.
8. certainty: POSSIBLE for may / might / can / sometimes / try.
   Otherwise DEFINITE.
9. condition: exact wording of a condition when one applies, e.g.
   "If your domain is connected to Wix via pointing". Otherwise null.
10. evidence: one exact contiguous quote from the article (under 300
    characters) that supports the relation. Copy it verbatim; do not
    fix typos or punctuation.
11. subject and object must be names from your entities list.
12. Prefer relations that connect this article to other articles:
    product membership, prerequisites, locations, limitations,
    cross-product dependencies. Skip step-by-step UI clicks and
    marketing sentences.
13. Aim for 5-20 relations depending on article length. Precision
    over quantity; do not pad.
14. For FEATURE REQUEST articles ("X Request: Y") the main relation is
    PRODUCT SUPPORTS Y with polarity NEGATIVE, plus any stated
    workaround as RESOLVED_BY with certainty POSSIBLE.
"""


# ============================================================
# 4. FILE UTILITIES
# ============================================================

def read_jsonl(path):
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from error
    return records


_WRITE_LOCK = threading.Lock()


def append_jsonl(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(line)
            file.flush()
            os.fsync(file.fileno())


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def content_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def fingerprint(configuration):
    serialized = json.dumps(
        configuration, sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ============================================================
# 5. TITLE PREFIX -> PRODUCT
# ============================================================

def title_product(title):
    """
    "Wix Stores: Setting Up ..."        -> "Wix Stores"
    "Wix Video Request: Editing ..."    -> "Wix Video"
    "Request: Something"                -> None
    "Managing Invoices Paid for Wix"    -> None
    """
    if not title or ":" not in title:
        return None
    prefix = title.split(":", 1)[0].strip()
    prefix = re.sub(r"\s+Request$", "", prefix).strip()
    if not prefix or prefix.casefold() == "request":
        return None
    if len(prefix.split()) > 5:
        return None
    return prefix


# ============================================================
# 6. DOCUMENT SEGMENTATION (long articles only)
# ============================================================

def split_document(content, max_chars=MAX_SEGMENT_CHARS):
    """
    Returns [(start_char, end_char)] covering the whole content.
    Splits at paragraph boundaries; no overlap.
    """
    if len(content) <= max_chars:
        return [(0, len(content))]

    boundaries = [m.end() for m in re.finditer(r"\n+", content)]
    segments = []
    start = 0
    while start < len(content):
        limit = start + max_chars
        if limit >= len(content):
            segments.append((start, len(content)))
            break
        candidates = [b for b in boundaries if start < b <= limit]
        end = candidates[-1] if candidates else limit
        segments.append((start, end))
        start = end
    return segments


# ============================================================
# 7. EVIDENCE MATCHING
# ============================================================

_CHAR_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ", "…": "...",
}


def normalize_with_offsets(text):
    """Drop whitespace/invisible chars, fold quotes and dashes."""
    normalized = []
    offsets = []
    for position, character in enumerate(text):
        character = _CHAR_MAP.get(character, character)
        if character.isspace() or character in ("﻿", "​"):
            continue
        for piece in character.casefold():
            normalized.append(piece)
            offsets.append(position)
    return "".join(normalized), offsets


def find_evidence(quote, source):
    quote = quote.strip()
    if not quote:
        return None
    start = source.find(quote)
    if start >= 0:
        return start, start + len(quote)
    normalized_source, offsets = normalize_with_offsets(source)
    normalized_quote, _ = normalize_with_offsets(quote)
    if not normalized_quote:
        return None
    position = normalized_source.find(normalized_quote)
    if position < 0:
        return None
    end = position + len(normalized_quote) - 1
    return offsets[position], offsets[end] + 1


# ============================================================
# 8. VALIDATION
# ============================================================

def normalize_name(name):
    name = re.sub(r"\s+", " ", name).strip().strip(" .,;:'\"")
    name = re.sub(r"^(the|a|an)\s+", "", name, flags=re.IGNORECASE)
    return name


def name_key(name):
    return normalize_name(name).casefold()


def validate_extraction(result, content, segment, known_entities):
    """
    result          : ExtractionResult from the model
    content         : full document text
    segment         : (start_char, end_char) of the text the model saw
    known_entities  : dict key -> entity dict accumulated so far

    Returns (entities_by_key, accepted_relations, rejected, warnings)
    """
    seg_start, seg_end = segment
    segment_text = content[seg_start:seg_end]

    entities = dict(known_entities)
    warnings = Counter()

    for entity in result.entities:
        name = normalize_name(entity.name)
        if not name:
            warnings["empty_entity_name"] += 1
            continue
        key = name.casefold()
        if len(name.split()) > HARD_MAX_ENTITY_WORDS:
            # A clause, not a concept. Relations that use it are
            # rejected below; keep it out of the entity layer too.
            warnings["entity_dropped_too_long"] += 1
            continue
        aliases = sorted({
            normalize_name(a) for a in entity.aliases
            if normalize_name(a) and name_key(a) != key
        })
        if key in entities:
            entities[key]["aliases"] = sorted(
                set(entities[key]["aliases"]) | set(aliases)
            )
        else:
            entities[key] = {
                "name": name,
                "type": entity.type,
                "aliases": aliases,
                "implicit": False,
            }
        if len(name.split()) > MAX_ENTITY_WORDS:
            warnings["entity_over_target_words"] += 1

    accepted = []
    rejected = []
    seen = set()

    for relation in result.relations:
        item = relation.model_dump()
        item["subject"] = normalize_name(item["subject"])
        item["object"] = normalize_name(item["object"])
        item["evidence"] = item["evidence"].strip()
        item["condition"] = (
            item["condition"].strip() if item.get("condition") else None
        )
        reason = None

        if not item["subject"] or not item["object"] or not item["evidence"]:
            reason = "missing_field"
        elif item["subject"].casefold() == item["object"].casefold():
            reason = "self_relation"
        elif max(
            len(item["subject"].split()), len(item["object"].split())
        ) > HARD_MAX_ENTITY_WORDS:
            reason = "entity_name_too_long"

        match = None
        if reason is None:
            match = find_evidence(item["evidence"], segment_text)
            if match is None:
                reason = "evidence_not_found"
            elif match[1] - match[0] > MAX_EVIDENCE_CHARS:
                reason = "evidence_too_long"

        if reason is None:
            signature = (
                item["subject"].casefold(),
                item["predicate"],
                item["object"].casefold(),
                item["polarity"],
            )
            if signature in seen:
                reason = "duplicate_relation"
            else:
                seen.add(signature)

        if reason is not None:
            rejected.append({"reason": reason, "relation": item})
            continue

        # Referential integrity: subject/object must be entities.
        for role in ("subject", "object"):
            key = item[role].casefold()
            if key not in entities:
                entities[key] = {
                    "name": item[role],
                    "type": "CONCEPT",
                    "aliases": [],
                    "implicit": True,
                }
                warnings["implicit_entity"] += 1
            item[f"{role}_type"] = entities[key]["type"]

        # Condition must appear somewhere in the document; otherwise
        # keep the relation but drop the condition.
        if item["condition"] and find_evidence(
            item["condition"], content
        ) is None:
            warnings["condition_not_found_dropped"] += 1
            item["condition"] = None

        start, end = match
        item["evidence"] = segment_text[start:end]
        item["evidence_start_char"] = seg_start + start
        item["evidence_end_char"] = seg_start + end
        accepted.append(item)

    return entities, accepted, rejected, warnings


# ============================================================
# 9. LLM CALL
# ============================================================

def call_llm(client, user_content):
    request = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "response_format": ExtractionResult,
    }
    if IS_REASONING_MODEL:
        request["reasoning_effort"] = REASONING_EFFORT
        request["max_completion_tokens"] = MAX_COMPLETION_TOKENS

    response = client.chat.completions.parse(**request)
    choice = response.choices[0]

    if choice.finish_reason == "length":
        raise RuntimeError("Completion token limit reached.")
    if choice.message.refusal:
        raise RuntimeError(f"Model refused: {choice.message.refusal}")
    if choice.message.parsed is None:
        raise RuntimeError("No structured extraction returned.")

    return choice.message.parsed, response.usage


def usage_dict(usage):
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0}
    details = getattr(usage, "completion_tokens_details", None)
    reasoning = (
        (getattr(details, "reasoning_tokens", None) or 0)
        if details is not None else 0
    )
    return {
        "input_tokens": usage.prompt_tokens,
        "output_tokens": usage.completion_tokens,
        "reasoning_tokens": reasoning,
    }


def build_user_content(document, product, segment, index, total, known):
    content = document["contents"]
    start, end = segment
    header = [
        f"TITLE: {document.get('title') or ''}",
        f"PRODUCT (from title): {product or '(none)'}",
        f"ARTICLE TYPE: {document.get('article_type') or 'article'}",
    ]
    if total > 1:
        header.append(f"ARTICLE PART {index + 1} OF {total}")
        if known:
            names = ", ".join(
                f"{e['name']} [{e['type']}]" for e in known.values()
            )
            header.append(
                "ENTITIES ALREADY EXTRACTED FROM EARLIER PARTS "
                f"(reuse these names): {names}"
            )
    return "\n".join(header) + "\n\nARTICLE:\n" + content[start:end]


# ============================================================
# 10. PER-DOCUMENT EXTRACTION
# ============================================================

def extract_document(client, document, run_fp):
    content = document.get("contents") or ""
    product = title_product(document.get("title") or "")
    segments = split_document(content)

    entities = {}
    relations = []
    rejected = []
    warnings = Counter()
    usage_total = Counter()
    raw_outputs = []

    for index, segment in enumerate(segments):
        user_content = build_user_content(
            document, product, segment, index, len(segments), entities
        )
        parsed, usage = call_llm(client, user_content)
        raw_outputs.append(parsed.model_dump())
        entities, accepted, bad, warn = validate_extraction(
            parsed, content, segment, entities
        )
        relations.extend(accepted)
        rejected.extend(bad)
        warnings.update(warn)
        usage_total.update(usage_dict(usage))

    for item in relations:
        item["document_id"] = document["id"]

    record = {
        "run_fingerprint": run_fp,
        "schema_version": SCHEMA_VERSION,
        "stage": "extraction",
        # Per-document checkpoint key: the corpus file may grow later
        # (subset v2, hard negatives); a saved result stays valid as
        # long as this document's text is unchanged.
        "content_sha256": content_sha256(content),
        "model": MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT if IS_REASONING_MODEL else None,
        "document_id": document["id"],
        "title": document.get("title"),
        "article_type": document.get("article_type"),
        "url": document.get("url"),
        "title_product": product,
        "n_chars": len(content),
        "n_segments": len(segments),
        "entities": sorted(entities.values(), key=lambda e: e["name"].casefold()),
        "relations": relations,
        "rejected": rejected,
        "warnings": dict(warnings),
        "raw_outputs": raw_outputs,
    }
    record.update(dict(usage_total))
    return record


# ============================================================
# 11. QUALITY REPORT (comparable to the v4 diagnosis)
# ============================================================

def quality_report(records, corpus_by_id=None):
    docs = {r["document_id"] for r in records}
    rels = [x for r in records for x in r["relations"]]
    ent_docs = defaultdict(set)
    ent_types = {}
    for r in records:
        for e in r["entities"]:
            ent_docs[e["name"].casefold()].add(r["document_id"])
            ent_types.setdefault(e["name"].casefold(), e["type"])

    names = list(ent_docs)
    words = [len(n.split()) for n in names] or [0]
    # Action clauses ("connecting a payment provider"), not noun
    # modifiers ("shipping region"): gerund-led AND 3+ words.
    gerund = sum(
        1 for n in names
        if re.match(r"^\w+ing\b", n) and len(n.split()) >= 3
    )
    singleton = sum(1 for n in names if len(ent_docs[n]) == 1)
    multi_doc = sum(1 for n in names if len(ent_docs[n]) > 1)

    def components(pairs, universe):
        parent = {u: u for u in universe}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in pairs:
            if a in parent and b in parent:
                parent[find(a)] = find(b)
        sizes = Counter(find(u) for u in universe)
        ordered = sorted(sizes.values(), reverse=True)
        return len(ordered), (ordered[0] if ordered else 0), sum(
            s == 1 for s in ordered
        )

    # Entity graph over relations.
    e_pairs = [(x["subject"].casefold(), x["object"].casefold()) for x in rels]
    e_comp, e_largest, _ = components(e_pairs, set(names) | {
        k for pair in e_pairs for k in pair
    })

    # Document graph: shared entities, excluding very frequent ones.
    n_docs = max(1, len(docs))
    generic_cut = max(3, int(0.25 * n_docs))
    d_pairs = []
    for name, ds in ent_docs.items():
        if 1 < len(ds) <= generic_cut:
            ds = sorted(ds)
            d_pairs.extend((ds[0], d) for d in ds[1:])
    d_comp, d_largest, d_isolated = components(d_pairs, docs)

    # Plus title-product edges.
    by_product = defaultdict(list)
    for r in records:
        if r.get("title_product"):
            by_product[r["title_product"].casefold()].append(r["document_id"])
    p_pairs = [(ds[0], d) for ds in by_product.values() for d in ds[1:]]
    dp_comp, dp_largest, dp_isolated = components(d_pairs + p_pairs, docs)

    reject_reasons = Counter(x["reason"] for r in records for x in r["rejected"])
    warn = Counter()
    for r in records:
        warn.update(r.get("warnings", {}))
    tokens = {
        k: sum(r.get(k, 0) for r in records)
        for k in ("input_tokens", "output_tokens", "reasoning_tokens")
    }

    print("\n========== KG v5 QUALITY REPORT ==========")
    print(f"documents: {len(docs)} | relations accepted: {len(rels)} "
          f"| rejected: {sum(reject_reasons.values())} {dict(reject_reasons)}")
    print(f"warnings: {dict(warn)}")
    print(f"tokens: {tokens}")
    print(f"relations/doc: mean {len(rels) / n_docs:.1f}")
    print(f"predicates: {Counter(x['predicate'] for x in rels).most_common()}")
    print(f"polarity NEGATIVE: {sum(x['polarity'] == 'NEGATIVE' for x in rels)} "
          f"| certainty POSSIBLE: {sum(x['certainty'] == 'POSSIBLE' for x in rels)} "
          f"| with condition: {sum(1 for x in rels if x['condition'])}")
    print("--- entity layer (gates: median words <=3, singletons <=40%, "
          "cross-doc share >=30%) ---")
    print(f"distinct entities: {len(names)} | median words: "
          f"{statistics.median(words):.0f} | >4 words: "
          f"{100 * sum(w > 4 for w in words) / max(1, len(words)):.0f}% "
          f"| gerund-led: {100 * gerund / max(1, len(names)):.0f}%")
    print(f"singleton entities (1 doc): {singleton} "
          f"({100 * singleton / max(1, len(names)):.0f}%) | entities in >1 doc: "
          f"{multi_doc} ({100 * multi_doc / max(1, len(names)):.0f}%)")
    print(f"entity types: {Counter(ent_types.values()).most_common()}")
    top = sorted(ent_docs.items(), key=lambda kv: -len(kv[1]))[:15]
    print("most shared entities: " + ", ".join(
        f"{n} ({len(d)})" for n, d in top
    ))
    print("--- connectivity (gates: isolated docs <10%, one component "
          ">80% of docs) ---")
    print(f"entity graph: components {e_comp}, largest {e_largest}")
    print(f"doc graph via shared entities (df<= {generic_cut}): components "
          f"{d_comp}, largest {d_largest}, isolated docs {d_isolated}")
    print(f"doc graph + title-product edges: components {dp_comp}, "
          f"largest {dp_largest}, isolated docs {dp_isolated}")

    # Gold-pair linkage if questions are available.
    questions_path = INPUT_PATH.parent / "questions.jsonl"
    if questions_path.exists():
        parent = {d: d for d in docs}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        for a, b in d_pairs + p_pairs:
            parent[find(a)] = find(b)
        linked = total = 0
        for q in read_jsonl(questions_path):
            ids = [i for i in q["article_ids"] if i in docs]
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    total += 1
                    linked += find(ids[i]) == find(ids[j])
        if total:
            print(f"multi-gold pairs in the same component: {linked}/{total}")


# ============================================================
# 12. SAMPLING
# ============================================================

def sample_documents(documents, sample_size, seed=42):
    """Pilot sample: proportional over article types, deterministic."""
    rng = random.Random(seed)
    by_type = defaultdict(list)
    for document in documents:
        by_type[document.get("article_type") or "article"].append(document)
    selected = []
    for article_type, docs in sorted(by_type.items()):
        quota = round(sample_size * len(docs) / len(documents))
        rng.shuffle(docs)
        selected.extend(docs[:quota])
    if len(selected) < sample_size:
        rest = [d for d in documents if d not in selected]
        rng.shuffle(rest)
        selected.extend(rest[: sample_size - len(selected)])
    return selected[:sample_size]


# ============================================================
# 13. MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of new documents to process")
    parser.add_argument("--sample", type=int, default=None,
                        help="Pilot: process a deterministic sample of N documents")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--report", action="store_true",
                        help="Only print the quality report for saved results")
    args = parser.parse_args()

    if not INPUT_PATH.exists():
        parser.error(f"Missing corpus: {INPUT_PATH}")

    documents = read_jsonl(INPUT_PATH)
    documents = [d for d in documents if (d.get("contents") or "").strip()]

    run_fp = fingerprint({
        "schema_version": SCHEMA_VERSION,
        "model": MODEL_NAME,
        "reasoning_effort": REASONING_EFFORT if IS_REASONING_MODEL else None,
        "schema": ExtractionResult.model_json_schema(),
        "prompt": EXTRACTION_PROMPT,
        "max_segment_chars": MAX_SEGMENT_CHARS,
    })

    # Saved results are keyed per document and validated against the
    # document's current text, so the corpus file may grow between runs.
    content_hashes = {d["id"]: content_sha256(d["contents"]) for d in documents}
    saved = {}
    stale = 0
    for record in read_jsonl(EXTRACTION_PATH):
        if record.get("run_fingerprint") != run_fp:
            raise ValueError(
                f"Configuration mismatch in {EXTRACTION_PATH} (prompt, schema "
                "or model changed). Rename or delete the file for a new experiment."
            )
        doc_id = record["document_id"]
        if content_hashes.get(doc_id) != record.get("content_sha256"):
            stale += 1          # document removed or its text changed
            continue
        saved[doc_id] = record

    if args.report:
        if not saved:
            parser.error("No saved results to report.")
        quality_report(list(saved.values()))
        return

    candidates = (
        sample_documents(documents, args.sample)
        if args.sample else documents
    )
    pending = [d for d in candidates if d["id"] not in saved]
    if args.limit is not None:
        pending = pending[: args.limit]

    print("\n========== WIXQA KG EXTRACTION V5 ==========")
    print(f"Documents in corpus: {len(documents)}")
    print(f"Already extracted: {len(saved)}" + (
        f" (ignored {stale} stale records)" if stale else ""
    ))
    print(f"Pending in this run: {len(pending)}")
    print(f"Model: {MODEL_NAME}" + (
        f" (reasoning {REASONING_EFFORT})" if IS_REASONING_MODEL else ""
    ))
    print(f"Output: {EXTRACTION_PATH}")
    long_docs = sum(len(split_document(d["contents"])) > 1 for d in pending)
    print(f"Documents needing segmentation: {long_docs}")

    if args.dry_run:
        print("Dry run. No API calls made.")
        return
    if not pending:
        print("Nothing to do.")
        quality_report(list(saved.values()))
        return
    if not os.getenv("OPENAI_API_KEY"):
        parser.error("Missing OPENAI_API_KEY in .env")

    client = OpenAI(timeout=180.0, max_retries=3)
    failures = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(extract_document, client, document, run_fp): document
            for document in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            document = futures[future]
            try:
                record = future.result()
            except Exception as error:  # keep going; rerun retries
                failures.append(document["id"])
                print(f"[{index}/{len(pending)}] FAILED {document['id'][:12]} "
                      f"{type(error).__name__}: {error}")
                continue
            append_jsonl(EXTRACTION_PATH, record)
            saved[record["document_id"]] = record
            print(f"[{index}/{len(pending)}] {len(record['relations']):2d} rel "
                  f"{len(record['entities']):2d} ent "
                  f"rej {len(record['rejected'])} | "
                  f"{(record['title'] or '')[:60]}")

    if failures:
        print(f"\n{len(failures)} documents failed; rerun to retry them.")
    quality_report(list(saved.values()))


if __name__ == "__main__":
    main()
