# src/data/inspect_wixqa.py
from collections import Counter
from pathlib import Path
from statistics import median
import json

from datasets import load_dataset


DATASET_NAME = "Wix/WixQA"

CONFIGS = {
    "corpus": "wix_kb_corpus",
    "expert": "wixqa_expertwritten",
    "simulated": "wixqa_simulated",
    "synthetic": "wixqa_synthetic",
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_config(config_name):
    dataset = load_dataset(DATASET_NAME, config_name)

    print(f"\nConfig: {config_name}")
    print(f"Available splits: {list(dataset.keys())}")

    if len(dataset) != 1:
        raise ValueError(f"Expected one split, found: {list(dataset.keys())}")

    split_name = next(iter(dataset.keys()))
    return dataset[split_name]


def save_jsonl(dataset, output_path):
    with output_path.open("w", encoding="utf-8") as file:
        for record in dataset:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    datasets = {}

    print("========== WIXQA DATASET INSPECTION ==========")

    for name, config in CONFIGS.items():
        ds = load_config(config)
        datasets[name] = ds

        print(f"Records: {len(ds)}")
        print(f"Fields: {ds.column_names}")

        save_jsonl(ds, OUTPUT_DIR / f"{name}.jsonl")

    corpus = datasets["corpus"]

    document_ids = [doc["id"] for doc in corpus]
    corpus_id_set = set(document_ids)

    article_types = Counter(doc["article_type"] for doc in corpus)

    word_lengths = [
        len((doc["contents"] or "").split())
        for doc in corpus
    ]

    print("\n========== CORPUS STATISTICS ==========")
    print(f"Total documents: {len(corpus)}")
    print(f"Unique IDs: {len(corpus_id_set)}")
    print(f"Duplicate IDs: {len(corpus) - len(corpus_id_set)}")
    print(f"Article types: {dict(article_types)}")
    print(f"Empty documents: {sum(n == 0 for n in word_lengths)}")
    print(f"Min words: {min(word_lengths)}")
    print(f"Max words: {max(word_lengths)}")
    print(f"Average words: {sum(word_lengths) / len(word_lengths):.1f}")
    print(f"Median words: {median(word_lengths):.1f}")

    print("\n========== QUESTION STATISTICS ==========")

    for name in ["expert", "simulated", "synthetic"]:
        ds = datasets[name]

        missing_ids = set()
        gold_counts = []

        for record in ds:
            gold_ids = record["article_ids"] or []
            gold_counts.append(len(gold_ids))

            for doc_id in gold_ids:
                if doc_id not in corpus_id_set:
                    missing_ids.add(doc_id)

        print(f"\n{name.upper()}")
        print(f"Questions: {len(ds)}")
        print(f"Single-gold: {sum(n == 1 for n in gold_counts)}")
        print(f"Multi-gold: {sum(n > 1 for n in gold_counts)}")
        print(f"No-gold: {sum(n == 0 for n in gold_counts)}")
        print(f"Missing gold document IDs: {len(missing_ids)}")

    print("\n========== EXAMPLE DOCUMENT ==========")
    example_doc = corpus[0]

    print("ID:", example_doc["id"])
    print("Type:", example_doc["article_type"])
    print("URL:", example_doc["url"])
    print("Contents preview:", example_doc["contents"][:500])

    print("\n========== EXAMPLE EXPERT QUESTION ==========")
    example_question = datasets["expert"][0]

    print("Question:", example_question["question"])
    print("Gold article IDs:", example_question["article_ids"])
    print("Answer preview:", example_question["answer"][:500])

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
