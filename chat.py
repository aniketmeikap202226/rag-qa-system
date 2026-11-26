"""Terminal chat (no Streamlit needed).

    python chat.py                # full RAG (needs an LLM: Ollama / OpenAI / Anthropic)
    python chat.py --search-only  # just show retrieved chunks, no LLM needed
"""
import argparse

from src.rag import answer, retriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-only", action="store_true")
    args = ap.parse_args()

    history = []
    print("Ask questions about your documents. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break
        if not q:
            continue

        if args.search_only:
            for i, r in enumerate(retriever.search(q), 1):
                print(f"\n[{i}] {r['file']} p.{r['page']}  similarity={r['score']:.2f}"
                      f"  rerank={r.get('rerank_score', float('nan')):.2f}")
                print(r["text"])
            continue

        try:
            res = answer(q, history=history[-3:])
        except Exception as e:
            print("Error:", e)
            continue
        print("\nAssistant:", res["answer"])
        for i, s in enumerate(res["sources"], 1):
            print(f"  [{i}] {s['file']} p.{s['page']} (similarity {s['score']:.2f})")
        print(f"  ({res['latency_s']}s)")
        history.append({"user": q, "assistant": res["answer"]})


if __name__ == "__main__":
    main()
