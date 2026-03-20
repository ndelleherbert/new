# ── IMPORTS ───────────────────────────────────────────────────────────────────
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb
from dotenv import load_dotenv
import os
import time
import json
import datetime
import streamlit as st

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Assistant", page_icon="🔍", layout="wide")

# ── LOAD ENV ──────────────────────────────────────────────────────────────────
load_dotenv()

# ── CACHED RESOURCES ──────────────────────────────────────────────────────────
@st.cache_resource
def load_llm(model: str, temperature: float):
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        st.error("❌ ANTHROPIC_API_KEY not found in .env")
        st.stop()
    return ChatAnthropic(model=model, temperature=temperature, anthropic_api_key=api_key)

@st.cache_resource
def load_embedder(model_name: str):
    return HuggingFaceEmbeddings(model_name=model_name)

@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path="./chroma_store")
    return client.get_or_create_collection(name="my_kb")

# ── HELPERS ───────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def build_prompt(context: str, question: str) -> str:
    template = PromptTemplate.from_template(
        "You are a helpful assistant.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )
    return template.format(context=context, question=question)

def clear_and_reload(collection, chunks: list[str], embedder):
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)
    if chunks:
        embeddings = embedder.embed_documents(chunks)
        collection.add(
            documents=chunks,
            ids=[f"chunk_{i}" for i in range(len(chunks))],
            embeddings=embeddings
        )

def log_query(query: str, answer: str, chunks: list[str], elapsed: float):
    entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "query": query,
        "answer": answer,
        "chunks_used": chunks,
        "response_time_s": round(elapsed, 2)
    }
    st.session_state.setdefault("query_log", []).append(entry)

# ── SESSION STATE DEFAULTS ────────────────────────────────────────────────────
DEFAULT_TEXT = (
    "This is a typical interview for a generative AI engineer including "
    "system design, NLP, LLMs, and deployment. Topics include transformer "
    "architecture, attention mechanisms, fine-tuning strategies, prompt "
    "engineering, retrieval-augmented generation, vector databases, model "
    "evaluation, MLOps pipelines, and responsible AI practices."
)
DEFAULTS = {
    "messages":  [],
    "query_log": [],
    "kb_text":   DEFAULT_TEXT,
    "settings": {
        "model":          "claude-sonnet-4-6",
        "temperature":    0.7,
        "chunk_size":     100,
        "n_results":      3,
        "embedder_model": "sentence-transformers/all-MiniLM-L6-v2"
    }
}
for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

settings   = st.session_state["settings"]
collection = load_collection()
embedder   = load_embedder(settings["embedder_model"])
llm        = load_llm(settings["model"], settings["temperature"])

# Seed default KB if empty
if collection.count() == 0:
    chunks = chunk_text(st.session_state["kb_text"], settings["chunk_size"])
    clear_and_reload(collection, chunks, embedder)

# ── SIDEBAR NAVIGATION ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 RAG Assistant")
    st.divider()
    page = st.radio(
        "Navigate",
        ["💬 Chat", "📄 Knowledge Base", "📊 Chat Analytics", "🧭 Vector Explorer", "⚙️ Settings"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"Model: `{settings['model']}`")
    st.caption(f"Chunks in store: `{collection.count()}`")
    st.caption(f"Queries logged: `{len(st.session_state['query_log'])}`")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
if page == "💬 Chat":
    st.title("💬 Chat")
    st.caption("Ask questions grounded in your knowledge base.")

    if st.button("🗑️ Clear history"):
        st.session_state["messages"] = []
        st.rerun()

    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "chunks" in msg:
                with st.expander("🔍 Retrieved context"):
                    for i, c in enumerate(msg["chunks"]):
                        st.markdown(f"**Chunk {i+1}:** {c}")

    query = st.chat_input("Ask a question about your document…")
    if query:
        st.session_state["messages"].append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    if collection.count() == 0:
                        st.warning("Knowledge base is empty. Add text in the Knowledge Base dashboard.")
                        st.stop()

                    t0         = time.time()
                    n          = min(settings["n_results"], collection.count())
                    q_emb      = embedder.embed_query(query)
                    results    = collection.query(query_embeddings=[q_emb], n_results=n)
                    top_chunks = results["documents"][0]
                    context    = "\n\n".join(top_chunks)
                    answer     = llm.invoke(build_prompt(context, query)).content
                    elapsed    = time.time() - t0

                    st.markdown(answer)
                    with st.expander("🔍 Retrieved context"):
                        for i, c in enumerate(top_chunks):
                            st.markdown(f"**Chunk {i+1}:** {c}")

                    st.session_state["messages"].append({
                        "role": "assistant", "content": answer, "chunks": top_chunks
                    })
                    log_query(query, answer, top_chunks, elapsed)

                except Exception as e:
                    st.error(f"Error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — KNOWLEDGE BASE MANAGER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📄 Knowledge Base":
    st.title("📄 Knowledge Base Manager")

    tab1, tab2 = st.tabs(["✏️ Edit Document", "📦 View Chunks"])

    with tab1:
        kb_text = st.text_area(
            "Document text",
            value=st.session_state["kb_text"],
            height=300,
            label_visibility="collapsed"
        )
        col1, col2, col3 = st.columns([1, 1, 3])
        with col1:
            if st.button("💾 Save & Rebuild"):
                if kb_text.strip():
                    chunks = chunk_text(kb_text, settings["chunk_size"])
                    clear_and_reload(collection, chunks, embedder)
                    st.session_state["kb_text"] = kb_text
                    st.success(f"✅ Rebuilt with {len(chunks)} chunk(s).")
                else:
                    st.warning("Document is empty.")
        with col2:
            if st.button("🗑️ Clear KB"):
                existing = collection.get()["ids"]
                if existing:
                    collection.delete(ids=existing)
                st.session_state["kb_text"] = ""
                st.success("Knowledge base cleared.")

        st.divider()
        words             = len(kb_text.split())
        estimated_chunks  = max(1, words // settings["chunk_size"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Words",             words)
        c2.metric("Estimated chunks",  estimated_chunks)
        c3.metric("Chunks in store",   collection.count())

    with tab2:
        if collection.count() == 0:
            st.info("No chunks stored yet. Add a document in the Edit tab.")
        else:
            stored = collection.get()
            search = st.text_input("🔎 Filter chunks", placeholder="Search within chunks…")
            for cid, doc in zip(stored["ids"], stored["documents"]):
                if search.lower() in doc.lower() or not search:
                    with st.expander(f"`{cid}`  —  {doc[:60]}…"):
                        st.write(doc)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — CHAT ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Chat Analytics":
    st.title("📊 Chat Analytics")

    log = st.session_state.get("query_log", [])

    if not log:
        st.info("No queries yet. Ask some questions in the Chat dashboard first.")
    else:
        import pandas as pd

        total      = len(log)
        avg_rt     = round(sum(e["response_time_s"] for e in log) / total, 2)
        avg_chunks = round(sum(len(e["chunks_used"]) for e in log) / total, 1)
        fastest    = min(log, key=lambda e: e["response_time_s"])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total queries",        total)
        c2.metric("Avg response time",    f"{avg_rt}s")
        c3.metric("Avg chunks retrieved", avg_chunks)
        c4.metric("Fastest response",     f"{fastest['response_time_s']}s")

        st.divider()
        st.subheader("Response time per query")
        df_rt = pd.DataFrame({
            "Query":             [f"Q{i+1}" for i in range(len(log))],
            "Response time (s)": [e["response_time_s"] for e in log]
        }).set_index("Query")
        st.bar_chart(df_rt)

        st.divider()
        st.subheader("Query log")
        for i, entry in enumerate(reversed(log)):
            with st.expander(f"Q{total - i}: {entry['query'][:70]}"):
                st.markdown(f"**Timestamp:** {entry['timestamp']}")
                st.markdown(f"**Response time:** {entry['response_time_s']}s")
                st.markdown(f"**Answer:** {entry['answer']}")
                st.markdown("**Chunks used:**")
                for j, c in enumerate(entry["chunks_used"]):
                    st.markdown(f"- Chunk {j+1}: {c[:80]}…")

        st.divider()
        if st.button("🗑️ Clear analytics log"):
            st.session_state["query_log"] = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — VECTOR EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧭 Vector Explorer":
    st.title("🧭 Embedding / Vector Explorer")

    if collection.count() == 0:
        st.info("No chunks stored yet. Add a document in the Knowledge Base dashboard.")
    else:
        stored     = collection.get(include=["documents", "embeddings"])
        docs       = stored["documents"]
        embeddings = stored["embeddings"]

        # Similarity search
        st.subheader("Similarity search")
        probe = st.text_input("Enter a query to rank chunks by similarity:", placeholder="e.g. system design")
        if probe:
            import numpy as np
            probe_vec = np.array(embedder.embed_query(probe))
            scores = []
            for doc, emb in zip(docs, embeddings):
                vec     = np.array(emb)
                sim     = float(np.dot(probe_vec, vec) / (np.linalg.norm(probe_vec) * np.linalg.norm(vec) + 1e-9))
                scores.append((sim, doc))
            scores.sort(reverse=True)
            for rank, (score, doc) in enumerate(scores[:5]):
                bar = "█" * int(score * 20)
                st.markdown(f"`{score:.3f}` {bar}")
                st.caption(doc[:120])
                st.divider()

        # PCA projection
        st.subheader("2D embedding projection (PCA)")
        try:
            import numpy as np
            import pandas as pd
            from sklearn.decomposition import PCA

            matrix = np.array(embeddings)
            if matrix.shape[0] >= 2:
                coords = PCA(n_components=2).fit_transform(matrix)
                df_pca = pd.DataFrame({
                    "x":     coords[:, 0],
                    "y":     coords[:, 1],
                    "chunk": [d[:40] + "…" for d in docs]
                })
                st.scatter_chart(df_pca, x="x", y="y")
            else:
                st.info("Need at least 2 chunks for PCA projection.")
        except ImportError:
            st.warning("Install scikit-learn for PCA: `pip install scikit-learn`")

        st.divider()
        st.subheader("Vector stats")
        import numpy as np
        matrix = np.array(embeddings)
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks",          matrix.shape[0])
        c2.metric("Embedding dim",   matrix.shape[1])
        c3.metric("Avg vector norm", round(float(np.mean(np.linalg.norm(matrix, axis=1))), 4))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — SETTINGS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "⚙️ Settings":
    st.title("⚙️ Settings")
    st.caption("Changes apply on the next query.")

    with st.form("settings_form"):
        st.subheader("LLM")
        models     = ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"]
        model      = st.selectbox("Model", models, index=models.index(settings["model"]))
        temperature = st.slider("Temperature", 0.0, 1.0, settings["temperature"], 0.05)

        st.subheader("Retrieval")
        n_results  = st.slider("Chunks to retrieve", 1, 10, settings["n_results"])
        chunk_size = st.slider("Chunk size (words)",  50, 500, settings["chunk_size"], 25)

        st.subheader("Embedder")
        emb_options = [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
            "sentence-transformers/paraphrase-MiniLM-L6-v2"
        ]
        embedder_model = st.selectbox("Embedding model", emb_options, index=0)

        if st.form_submit_button("💾 Save Settings"):
            st.session_state["settings"].update({
                "model":          model,
                "temperature":    temperature,
                "n_results":      n_results,
                "chunk_size":     chunk_size,
                "embedder_model": embedder_model
            })
            load_llm.clear()
            load_embedder.clear()
            st.success("✅ Settings saved. Resources will reload on next interaction.")
            st.rerun()

    st.divider()
    st.subheader("🔑 API key status")
    key = os.getenv("ANTHROPIC_API_KEY")
    if key:
        st.success(f"API key loaded · ends in `…{key[-4:]}`")
    else:
        st.error("No API key found. Add ANTHROPIC_API_KEY to your .env file.")

    st.divider()
    st.subheader("📤 Export")
    col1, col2 = st.columns(2)
    with col1:
        if st.session_state["query_log"]:
            st.download_button(
                "⬇️ Download query log (JSON)",
                data=json.dumps(st.session_state["query_log"], indent=2),
                file_name="query_log.json",
                mime="application/json"
            )
        else:
            st.caption("No query log to export yet.")
    with col2:
        if st.session_state["kb_text"]:
            st.download_button(
                "⬇️ Download knowledge base (.txt)",
                data=st.session_state["kb_text"],
                file_name="knowledge_base.txt",
                mime="text/plain"
            )