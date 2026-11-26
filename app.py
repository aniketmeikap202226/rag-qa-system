"""Streamlit UI.  Run:  streamlit run app.py"""
import os

import streamlit as st

from src import config, ingest
from src.rag import answer, retriever

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Document Q&A (RAG)")
st.caption(f"LLM provider: {config.LLM_PROVIDER}")

if "history" not in st.session_state:
    st.session_state.history = []

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

if not os.path.exists(config.INDEX_FILE):
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
