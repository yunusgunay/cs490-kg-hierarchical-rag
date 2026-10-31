# src/data/build_subset_v2.py
"""
WixQA subset v2 — gold-anchored corpus with hard negatives.

Why v2
------
v1 (256 docs) used 200 *random* distractors. Random Wix articles are
topically far from the gold articles, so flat BM25 already solved 46/50
questions (R@10 = 0.91) and the subset contained none of a gold
article's nearest neighbours. v2 keeps everything from v1 and adds the
documents that make the task realistic.

Corpus (deterministic)
----------------------
  1. gold articles of ALL 200 expert questions            (209 docs)
  2. for every dev gold article: its 3 nearest neighbours by TF-IDF
     cosine over title + first 3,000 chars, from the full corpus
  3. title twins: articles whose title after the "Product:" prefix is
     identical to a dev gold's (Wix Editor / Studio Editor pairs)
  4. the 200 random distractors of subset v1 (same seed, same draw),
     so v1 results stay comparable
Questions
---------
  dev      = all 52 multi-gold expert questions + the 37 single-gold
             questions already in subset v1                (89)
  held-out = the remaining 111 expert questions (single-gold); their
             gold articles are in the corpus, so they can be evaluated
             later as an untouched test set.

Outputs (data/processed/subset/)
  corpus.jsonl, questions.jsonl, questions_heldout.jsonl, metadata.json
"""
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS_PATH = PROJECT_ROOT / "data/raw/corpus.jsonl"
QUESTIONS_PATH = PROJECT_ROOT / "data/raw/expert.jsonl"
OUTPUT_DIR = PROJECT_ROOT / "data/processed/subset"

# --- subset v1 parameters (reproduced exactly) ---
V1_SEED = 42
V1_NUM_QUESTIONS = 50
V1_NUM_DISTRACTORS = 200

# --- v2 parameters ---
NEIGHBOURS_PER_GOLD = 3
NEIGHBOUR_TEXT_CHARS = 3000
DROP_FIELDS = ("html_content",)   # not used downstream; 50 MB -> 10 MB


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ------------------------------------------------------------
# subset v1, reproduced
# ------------------------------------------------------------

def reproduce_v1(questions, documents_by_id):
    rng = random.Random(V1_SEED)
    single_hop = [q for q in questions if len(q["article_ids"]) == 1]
    multi_hop = [q for q in questions if len(q["article_ids"]) > 1]
    multi_count = round(V1_NUM_QUESTIONS * len(multi_hop) / len(questions))
    single_count = V1_NUM_QUESTIONS - multi_count
    selected = rng.sample(single_hop, single_count) + rng.sample(multi_hop, multi_count)
    rng.shuffle(selected)
    gold_ids = {g for q in selected for g in q["article_ids"]}
    candidates = [d for d in documents_by_id if d not in gold_ids]
    distractors = set(rng.sample(candidates, V1_NUM_DISTRACTORS))
    return selected, gold_ids, distractors


# ------------------------------------------------------------
# TF-IDF nearest neighbours
# ------------------------------------------------------------

def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class TfidfIndex:
    def __init__(self, documents):
        self.ids = [d["id"] for d in documents]
        tokens = {
            d["id"]: tokenize(d["title"] + " " + d["contents"][:NEIGHBOUR_TEXT_CHARS])
            for d in documents
        }
        self.df = Counter()
        for t in tokens.values():
            self.df.update(set(t))
        self.n = len(documents)
        self.vectors = {i: self._vector(t) for i, t in tokens.items()}
        self.inverted = defaultdict(list)
        for i, vector in self.vectors.items():
            for word, weight in vector.items():
                if self.df[word] < 800:          # skip near-stopwords
                    self.inverted[word].append((i, weight))

    def _vector(self, tokens):
        counts = Counter(tokens)
        vector = {
            w: (1 + math.log(c)) * math.log(self.n / self.df[w])
            for w, c in counts.items()
        }
        norm = math.sqrt(sum(x * x for x in vector.values())) or 1.0
        return {w: x / norm for w, x in vector.items()}

    def neighbours(self, doc_id, k):
        scores = defaultdict(float)
        for word, weight in self.vectors[doc_id].items():
            for other, other_weight in self.inverted.get(word, ()):
                scores[other] += weight * other_weight
        scores.pop(doc_id, None)
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]


def title_suffix(title):
    return title.split(":", 1)[1].strip().casefold() if ":" in title else title.casefold()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------

def main():
    corpus = read_jsonl(CORPUS_PATH)
    questions = read_jsonl(QUESTIONS_PATH)
    documents_by_id = {d["id"]: d for d in corpus}

    v1_questions, v1_gold, v1_random = reproduce_v1(questions, documents_by_id)
    v1_texts = {q["question"] for q in v1_questions}

    multi = [q for q in questions if len(q["article_ids"]) > 1]
    dev_single = [q for q in questions if len(q["article_ids"]) == 1 and q["question"] in v1_texts]
    dev_questions = dev_single + multi
    dev_texts = {q["question"] for q in dev_questions}
    heldout_questions = [q for q in questions if q["question"] not in dev_texts]

    dev_gold = {g for q in dev_questions for g in q["article_ids"]}
    all_gold = {g for q in questions for g in q["article_ids"]}
    missing = all_gold - documents_by_id.keys()
    if missing:
        raise ValueError(f"{len(missing)} gold documents missing from corpus.")

    index = TfidfIndex(corpus)
    suffix_index = defaultdict(set)
    for d in corpus:
        suffix_index[title_suffix(d["title"])].add(d["id"])

    neighbour_ids, twin_ids = set(), set()
    neighbour_log = {}
    for g in sorted(dev_gold):
        nn = index.neighbours(g, NEIGHBOURS_PER_GOLD)
        neighbour_log[g] = [(j, round(s, 3)) for j, s in nn]
        neighbour_ids |= {j for j, _ in nn}
        twin_ids |= suffix_index[title_suffix(documents_by_id[g]["title"])] - {g}

    selected_ids = all_gold | neighbour_ids | twin_ids | v1_random
    selected_docs = [
        {k: v for k, v in d.items() if k not in DROP_FIELDS}
        for d in corpus if d["id"] in selected_ids          # corpus order
    ]

    def role(doc_id):
        if doc_id in dev_gold:
            return "dev_gold"
        if doc_id in all_gold:
            return "heldout_gold"
        if doc_id in neighbour_ids or doc_id in twin_ids:
            return "hard_negative"
        return "random_v1"

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_jsonl(OUTPUT_DIR / "corpus.jsonl", selected_docs)
    write_jsonl(OUTPUT_DIR / "questions.jsonl", dev_questions)
    write_jsonl(OUTPUT_DIR / "questions_heldout.jsonl", heldout_questions)

    metadata = {
        "subset_version": "v2",
        "rule": (
            "corpus = gold articles of all 200 expert questions + 3 TF-IDF "
            "nearest neighbours of every dev gold article + title twins of dev "
            "gold articles + the 200 random distractors of subset v1 (seed 42). "
            "dev questions = all multi-gold expert questions + the single-gold "
            "questions of subset v1; held-out = remaining expert questions."
        ),
        "v1_seed": V1_SEED,
        "neighbours_per_gold": NEIGHBOURS_PER_GOLD,
        "neighbour_text_chars": NEIGHBOUR_TEXT_CHARS,
        "total_documents": len(selected_docs),
        "document_roles": dict(Counter(role(d["id"]) for d in selected_docs)),
        "article_types": dict(Counter(d["article_type"] for d in selected_docs)),
        "dev_questions": len(dev_questions),
        "dev_single_gold": len(dev_single),
        "dev_multi_gold": len(multi),
        "dev_gold_documents": len(dev_gold),
        "heldout_questions": len(heldout_questions),
        "heldout_gold_documents": len(all_gold - dev_gold),
        "v1_questions_retained": len(v1_questions),
        "v1_random_distractors_retained": len(v1_random & selected_ids),
        "neighbour_log": neighbour_log,
    }
    with (OUTPUT_DIR / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print("========== WIXQA SUBSET V2 ==========")
    for key, value in metadata.items():
        if key != "neighbour_log":
            print(f"{key}: {value}")
    print(f"\nSaved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()