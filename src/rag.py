"""The full RAG pipeline: retrieve -> (re-rank) -> prompt LLM -> answer + sources."""
import time

from src import config, llm
from src.retriever import Retriever

retriever = Retriever()

IDK = "I don't know based on the provided documents."

PROMPT = """You are a helpful assistant. Answer the question using ONLY the context below.
Rules:
- If the answer is not in the context, reply exactly: {idk}
- Be concise and accurate. Do not use outside knowledge.
- Cite the sources you used like [1], [2].

Conversation so far:
{history}

Context:
{context}

Question: {question}
Answer:"""


def _format_history(history):
    if not history:
        return "(none)"
    return "\n".join(f"User: {h['user']}\nAssistant: {h['assistant']}" for h in history)


def answer(question, history=None, rerank=True):
    """history: list of {"user": ..., "assistant": ...} (recent turns only)."""
    t0 = time.time()

    # For follow-ups ("what about contractors?") add the previous question to the search.
    search_query = f"{history[-1]['user']} {question}" if history else question
    results = retriever.search(search_query, rerank=rerank)

    # Confidence threshold: nothing close enough -> don't even call the LLM.
    if not results or max(r["score"] for r in results) < config.MIN_SCORE:
        return {"answer": IDK, "sources": [], "answered": False,
                "latency_s": round(time.time() - t0, 2)}

    context = "\n\n".join(
        f"[{i}] ({r['file']}, p.{r['page']}) {r['text']}" for i, r in enumerate(results, 1)
    )
    prompt = PROMPT.format(idk=IDK, history=_format_history(history),
                           context=context, question=question)
    text = llm.generate(prompt)

    return {
        "answer": text,
        "sources": results,
        "answered": not text.lower().startswith("i don't know"),
        "latency_s": round(time.time() - t0, 2),
    }
