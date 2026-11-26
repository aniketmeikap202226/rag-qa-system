"""Streamlit UI.  Run:  python -m streamlit run app.py"""
import os

import streamlit as st

# On hosted platforms, secrets arrive via st.secrets; copy them into the environment
# BEFORE src.config is imported (it reads settings from environment variables).
try:
    for _k in ("LLM_PROVIDER", "GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_FALLBACK_MODEL"):
        if _k in st.secrets and _k not in os.environ:
            os.environ[_k] = str(st.secrets[_k])
except Exception:
    pass

from src import config, ingest  # noqa: E402
from src.rag import answer, retriever  # noqa: E402

MAX_QUESTIONS = int(os.getenv("MAX_QUESTIONS_PER_SESSION", "15"))  # protects the API key

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Document Q&A (RAG)")
st.caption(f"LLM provider: {config.LLM_PROVIDER}")

if "history" not in st.session_state:
    st.session_state.history = []
if "n_questions" not in st.session_state:
    st.session_state.n_questions = 0

with st.sidebar:
    st.header("Documents")
    files = st.file_uploader("Upload PDF / TXT / MD", type=["pdf", "txt", "md"],
                             accept_multiple_files=True)
    if st.button("Build index") and files:
        os.makedirs(config.DOCS_FOLDER, exist_ok=True)
        for f in files:
            with open(os.path.join(config.DOCS_FOLDER, f.name), "wb") as out:
                out.write(f.getbuffer())
        with st.spinner("Reading, chunking and embedding..."):
            n = ingest.build_index()
            retriever.reload()
        st.success(f"Indexed {n} chunks")
    use_rerank = st.toggle("Cross-encoder re-ranking", value=True)
    if st.button("Clear chat"):
        st.session_state.history = []

# First run on a fresh server: build the index from the bundled documents.
if not os.path.exists(config.INDEX_FILE):
    has_docs = os.path.isdir(config.DOCS_FOLDER) and any(
        f.lower().endswith((".pdf", ".txt", ".md")) for f in os.listdir(config.DOCS_FOLDER)
    )
    if has_docs:
        with st.spinner("First start: building the search index (about a minute)..."):
            ingest.build_index()
            retriever.reload()
    else:
        st.info("Upload documents in the sidebar and click **Build index** to start.")
        st.stop()

# Replay conversation
for turn in st.session_state.history:
    with st.chat_message("user"):
        st.write(turn["user"])
    with st.chat_message("assistant"):
        st.write(turn["assistant"])

question = st.chat_input("Ask something about your documents...")
if question:
    if st.session_state.n_questions >= MAX_QUESTIONS:
        st.warning(f"Demo limit reached ({MAX_QUESTIONS} questions per session). "
                   "Refresh the page to start a new session.")
        st.stop()
    st.session_state.n_questions += 1

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                res = answer(question, history=st.session_state.history[-3:],
                             rerank=use_rerank)
            except Exception as e:
                st.error(str(e))
                st.stop()
        st.write(res["answer"])
        st.caption(f"Latency: {res['latency_s']}s")
        if res["sources"]:
            with st.expander("Sources"):
                for i, s in enumerate(res["sources"], 1):
                    st.markdown(f"**[{i}] {s['file']} - page {s['page']}** "
                                f"(similarity {s['score']:.2f})")
                    st.write(s["text"])
    st.session_state.history.append({"user": question, "assistant": res["answer"]})
