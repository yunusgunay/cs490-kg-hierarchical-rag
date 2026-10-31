
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "processed" / "full_dense_test"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 32
MAX_CHARS = 20_000

load_dotenv(ROOT / ".env")


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def download_wixqa():
    """Download the official corpus and expert-written questions."""
    print("Loading official WixQA files...")

    corpus_path = hf_hub_download(
        repo_id="Wix/WixQA",
        repo_type="dataset",
        filename="wix_kb_corpus/wix_kb_corpus.jsonl",
    )

    questions_path = hf_hub_download(
        repo_id="Wix/WixQA",
        repo_type="dataset",
        filename="wixqa_expertwritten/test.jsonl",
    )

    docs = list(read_jsonl(corpus_path))
    questions = list(read_jsonl(questions_path))

    return docs, questions


def document_text(doc):
    """
    Use article contents, truncated to 20k characters.
    This matches the full-article embedding approach.
    """
    return str(doc["contents"])[:MAX_CHARS]


def fingerprint(texts):
    h = hashlib.sha256()
    h.update(MODEL.encode("utf-8"))

    for text in texts:
        encoded = text.encode("utf-8")
        h.update(len(encoded).to_bytes(8, "big"))
        h.update(encoded)

    return h.hexdigest()[:20]


def embed_texts(client, texts, cache_name):
    """Embed texts with a resumable local cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    fp = fingerprint(texts)
    cache_path = CACHE_DIR / f"{cache_name}_{fp}.npz"

    vectors = []

    if cache_path.exists():
        with np.load(cache_path) as saved:
            vectors = list(saved["vectors"])

        print(
            f"{cache_name}: restored "
            f"{len(vectors)}/{len(texts)} embeddings"
        )

    while len(vectors) < len(texts):
        start = len(vectors)
        end = min(start + BATCH_SIZE, len(texts))

        batch = texts[start:end]

        response = client.embeddings.create(
            model=MODEL,
            input=batch,
        )

        ordered = sorted(response.data, key=lambda x: x.index)

        for item in ordered:
            vectors.append(np.asarray(item.embedding, dtype=np.float32))

        # Save after each successful batch for resume.
        np.savez_compressed(
            cache_path,
            vectors=np.stack(vectors),
        )

        print(f"{cache_name}: {len(vectors)}/{len(texts)}")

    return np.stack(vectors)


def normalize(matrix):
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


def evaluate(docs, questions, doc_vectors, question_vectors):
    ids = [str(doc["id"]) for doc in docs]
    id_to_index = {doc_id: i for i, doc_id in enumerate(ids)}

    assert len(id_to_index) == len(ids), "Duplicate document IDs."

    q_matrix = normalize(question_vectors)
    d_matrix = normalize(doc_vectors)

    similarities = q_matrix @ d_matrix.T

    metrics = {
        k: {
            "allgold": [],
            "recall": [],
            "mrr": [],
        }
        for k in (1, 3, 5, 10, 20)
    }

    details = []
    missing_gold = set()

    for qi, question in enumerate(questions):
        gold = {str(x) for x in question["article_ids"]}

        if not gold:
            raise ValueError(f"Question {qi} has no gold articles.")

        for article_id in gold:
            if article_id not in id_to_index:
                missing_gold.add(article_id)

        ranking = np.argsort(
            -similarities[qi],
            kind="stable",
        )[:20]

        ranked_ids = [ids[i] for i in ranking]
        n_gold = len(gold)

        record = {
            "question_index": qi,
            "question": question["question"],
            "gold": sorted(gold),
            "n_gold": n_gold,
            "top20": ranked_ids,
            "metrics": {},
        }

        for k in metrics:
            retrieved = ranked_ids[:k]
            found = gold.intersection(retrieved)

            allgold = float(found == gold)
            recall = len(found) / n_gold

            first_gold_rank = next(
                (
                    rank
                    for rank, article_id in enumerate(retrieved, 1)
                    if article_id in gold
                ),
                None,
            )

            mrr = (
                1.0 / first_gold_rank
                if first_gold_rank is not None
                else 0.0
            )

            metrics[k]["allgold"].append(allgold)
            metrics[k]["recall"].append(recall)
            metrics[k]["mrr"].append(mrr)

            record["metrics"][str(k)] = {
                "allgold": allgold,
                "recall": recall,
                "mrr": mrr,
            }

        details.append(record)

    if missing_gold:
        raise ValueError(
            f"{len(missing_gold)} gold article IDs missing "
            f"from the corpus. Examples: {list(missing_gold)[:5]}"
        )

    print("\n" + "=" * 72)
    print("FULL WIXQA DENSE RETRIEVAL")
    print("=" * 72)
    print(f"Embedding model: {MODEL}")
    print(f"Corpus documents: {len(docs)}")
    print(f"Questions: {len(questions)}")
    print(f"Document truncation: {MAX_CHARS} characters")

    print("\nOverall metrics")
    print(f"{'k':>4} {'AllGold@k':>12} {'Recall@k':>12} {'MRR@k':>12}")

    for k, values in metrics.items():
        a = np.mean(values["allgold"])
        r = np.mean(values["recall"])
        m = np.mean(values["mrr"])

        print(f"{k:>4} {a:>12.4f} {r:>12.4f} {m:>12.4f}")

    print("\nAllGold@5 by question type")

    for label, predicate in [
        ("Single-gold", lambda n: n == 1),
        ("Multi-gold", lambda n: n > 1),
    ]:
        rows = [
            record
            for record in details
            if predicate(record["n_gold"])
        ]

        if rows:
            successes = sum(
                record["metrics"]["5"]["allgold"]
                for record in rows
            )
            print(
                f"{label}: {int(successes)}/{len(rows)} "
                f"= {successes / len(rows):.4f}"
            )

    results_path = CACHE_DIR / "dense_full_results.json"
    summary_path = CACHE_DIR / "dense_full_summary.json"

    summary = {
        "model": MODEL,
        "documents": len(docs),
        "questions": len(questions),
        "metrics": {
            str(k): {
                name: float(np.mean(values[name]))
                for name in ("allgold", "recall", "mrr")
            }
            for k, values in metrics.items()
        },
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults: {results_path}")
    print(f"Summary: {summary_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use local JSONL paths instead of Hugging Face",
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--questions", type=Path)
    args = parser.parse_args()

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not found in .env")

    if args.skip_download:
        if not args.corpus or not args.questions:
            parser.error(
                "--skip-download requires --corpus and --questions"
            )
        docs = list(read_jsonl(args.corpus))
        questions = list(read_jsonl(args.questions))
    else:
        docs, questions = download_wixqa()

    assert len(docs) > 6000, (
        f"Expected full WixQA corpus, got {len(docs)} docs."
    )
    assert len(questions) == 200, (
        f"Expected 200 expert-written questions, "
        f"got {len(questions)}."
    )

    doc_ids = {str(d["id"]) for d in docs}
    assert len(doc_ids) == len(docs), "Duplicate article IDs."

    for q in questions:
        for gold_id in q["article_ids"]:
            if str(gold_id) not in doc_ids:
                raise ValueError(
                    f"Gold article {gold_id} absent from corpus."
                )

    print(f"Loaded {len(docs)} articles and {len(questions)} questions.")

    client = OpenAI()

    doc_texts = [document_text(d) for d in docs]
    question_texts = [q["question"] for q in questions]

    doc_vectors = embed_texts(client, doc_texts, "documents")
    question_vectors = embed_texts(
        client,
        question_texts,
        "questions",
    )

    evaluate(docs, questions, doc_vectors, question_vectors)


if __name__ == "__main__":
    main()
