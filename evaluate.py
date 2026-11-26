"""Measure retrieval quality (Recall@k, MRR) and optionally answer quality.

1. Copy eval/questions.example.json -> eval/questions.json and write your own questions.
2. python evaluate.py                # retrieval only (fast, no LLM needed)
3. python evaluate.py --with-llm     # also checks generated answers

Each question needs EITHER a short "keyword" that must appear in the correct chunk/answer
(robust when you change chunk size) OR the "file" + "page" where the answer lives.
"""
import argparse
import json

from src import config
from src.rag import answer, retriever


def is_hit(chunk, item):
    if "keyword" in item:
        return item["keyword"].lower() in chunk["text"].lower()
    return chunk["file"] == item["file"] and chunk["page"] == item["page"]


def eval_retrieval(items, k, rerank):
    hits, rr_sum = 0, 0.0
    for item in items:
        results = retriever.search(item["question"], k=k, rerank=rerank)
        rank = next((i for i, c in enumerate(results, 1) if is_hit(c, item)), None)
        if rank:
            hits += 1
            rr_sum += 1 / rank
    n = len(items)
    return hits / n, rr_sum / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval/questions.json")
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--with-llm", action="store_true")
    args = ap.parse_args()

    with open(args.file, encoding="utf-8") as f:
        items = json.load(f)
    print(f"{len(items)} questions, k={args.k}\n")

    print("| Setup | Recall@%d | MRR |" % args.k)
    print("|---|---|---|")
    for name, rerank in [("Dense search only", False), ("Dense + re-ranker", True)]:
        recall, mrr = eval_retrieval(items, args.k, rerank)
        print(f"| {name} | {recall:.1%} | {mrr:.3f} |")

    if args.with_llm:
        correct, idk, lat = 0, 0, 0.0
        answerable = [i for i in items if "keyword" in i]
        for item in answerable:
            res = answer(item["question"])
            lat += res["latency_s"]
            idk += not res["answered"]
            correct += item["keyword"].lower() in res["answer"].lower()
        n = max(len(answerable), 1)
        print(f"\nAnswer accuracy (keyword in answer): {correct / n:.1%}")
        print(f"'I don't know' rate: {idk / n:.1%}")
        print(f"Avg latency: {lat / n:.2f}s")


if __name__ == "__main__":
    main()
