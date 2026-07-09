"""
ChromaDB Visualizer
--------------------
A local Streamlit app to explore a ChromaDB database (persisted on disk or
in-memory demo data): browse collections, inspect records (documents,
metadata, full vectors), search/filter, and visualize the embedding space
in 2D/3D via PCA or t-SNE.

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""

import json
import random

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="ChromaDB Visualizer", layout="wide")

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
if "client" not in st.session_state:
    st.session_state.client = None
if "client_signature" not in st.session_state:
    st.session_state.client_signature = None


# --------------------------------------------------------------------------
# Client helpers
# --------------------------------------------------------------------------
def get_persistent_client(path: str):
    import chromadb
    return chromadb.PersistentClient(path=path)


def get_ephemeral_client():
    import chromadb
    return chromadb.EphemeralClient()


def populate_demo_data(client, n_records: int = 60, dim: int = 384, seed: int = 42):
    """Fill an ephemeral client with a couple of demo collections using
    random (but clustered) vectors, so the UI can be explored without a
    real persisted DB or a downloaded embedding model."""
    rng = np.random.default_rng(seed)
    topics = ["cooking", "finance", "sports", "travel"]
    sample_docs = {
        "cooking": [
            "How to make a simple tomato pasta sauce",
            "Best way to knead sourdough bread dough",
            "Tips for grilling a medium-rare steak",
        ],
        "finance": [
            "Understanding compound interest on savings",
            "How index funds diversify risk",
            "Basics of reading a balance sheet",
        ],
        "sports": [
            "Rules of a tie-break in tennis",
            "How offside works in football",
            "Training plan for a first marathon",
        ],
        "travel": [
            "Best time of year to visit Kyoto",
            "Packing list for a two week backpacking trip",
            "How to find cheap flights using fare alerts",
        ],
    }

    if "demo_docs" in client.list_collections():
        client.delete_collection("demo_docs")
    coll = client.create_collection("demo_docs")

    ids, docs, metadatas, embeddings = [], [], [], []
    for i in range(n_records):
        topic = topics[i % len(topics)]
        center = rng.normal(loc=topics.index(topic) * 4.0, scale=1.0, size=dim)
        vec = center + rng.normal(scale=0.5, size=dim)
        doc_text = random.choice(sample_docs[topic]) + f" (variant {i})"
        ids.append(f"doc_{i}")
        docs.append(doc_text)
        metadatas.append({"topic": topic, "length": len(doc_text), "idx": i})
        embeddings.append(vec.tolist())

    coll.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    return coll


# --------------------------------------------------------------------------
# Sidebar: connection
# --------------------------------------------------------------------------
st.sidebar.title("🔌 Connection")

mode = st.sidebar.radio(
    "Data source",
    ["Persistent (local path)", "In-memory (demo data)"],
    help="Point at a folder created by chromadb.PersistentClient(path=...), "
         "or spin up in-memory demo data to explore the UI.",
)

connect_error = None

if mode == "Persistent (local path)":
    default_path = st.session_state.get("last_path", "./chroma_db")
    db_path = st.sidebar.text_input("Path to persist directory", value=default_path)
    connect_clicked = st.sidebar.button("Connect", type="primary")
    if connect_clicked or st.session_state.client_signature == ("persistent", db_path):
        try:
            st.session_state.client = get_persistent_client(db_path)
            st.session_state.client_signature = ("persistent", db_path)
            st.session_state.last_path = db_path
        except Exception as e:  # noqa: BLE001
            connect_error = str(e)
else:
    n_records = st.sidebar.slider("Demo record count", 10, 300, 60, step=10)
    if st.sidebar.button("Generate demo data", type="primary") or st.session_state.client_signature == "ephemeral":
        try:
            if st.session_state.client_signature != "ephemeral":
                st.session_state.client = get_ephemeral_client()
                st.session_state.client_signature = "ephemeral"
            populate_demo_data(st.session_state.client, n_records=n_records)
        except Exception as e:  # noqa: BLE001
            connect_error = str(e)

if connect_error:
    st.sidebar.error(f"Could not connect:\n\n{connect_error}")

client = st.session_state.client

st.title("🧭 ChromaDB Visualizer")

if client is None:
    st.info(
        "👈 Connect to a persisted ChromaDB directory, or generate in-memory "
        "demo data, using the sidebar to get started."
    )
    st.markdown(
        """
        **What this app shows you, once connected:**
        - All collections in the database, with record counts
        - A searchable/filterable table of records: id, document text, metadata, vector preview
        - Full embedding vectors for any record (raw values + dimensionality)
        - A 2D/3D projection (PCA or t-SNE) of the embedding space, colored by any metadata field
        - CSV/JSON export of whatever you're currently viewing
        """
    )
    st.stop()

# --------------------------------------------------------------------------
# Collection selection
# --------------------------------------------------------------------------
try:
    collection_names = client.list_collections()
    # Newer chromadb versions return plain names; older ones return Collection objects.
    collection_names = [c if isinstance(c, str) else c.name for c in collection_names]
except Exception as e:  # noqa: BLE001
    st.error(f"Failed to list collections: {e}")
    st.stop()

if not collection_names:
    st.warning("This database has no collections yet.")
    st.stop()

col_left, col_right = st.columns([2, 1])
with col_left:
    coll_name = st.selectbox("Collection", collection_names)
with col_right:
    coll = client.get_collection(coll_name)
    try:
        total_count = coll.count()
    except Exception:
        total_count = None
    st.metric("Records", total_count if total_count is not None else "—")

st.divider()

tab_table, tab_vectors, tab_json = st.tabs(
    ["📋 Records Table", "🌌 Vector Explorer", "🔎 Raw Record Inspector"]
)

# --------------------------------------------------------------------------
# Fetch data (cached per collection + limit)
# --------------------------------------------------------------------------
MAX_FETCH = 2000  # safety cap for very large collections


@st.cache_data(show_spinner="Fetching records from ChromaDB...")
def fetch_records(_client_sig, coll_name, limit):
    coll = client.get_collection(coll_name)
    result = coll.get(limit=limit, include=["documents", "metadatas", "embeddings"])
    return result


fetch_limit = min(total_count or MAX_FETCH, MAX_FETCH)
data = fetch_records(st.session_state.client_signature, coll_name, fetch_limit)

ids = data.get("ids", [])
docs = data.get("documents") or [None] * len(ids)
metas = data.get("metadatas") or [None] * len(ids)
embeddings = data.get("embeddings")
has_vectors = embeddings is not None and len(embeddings) > 0 and embeddings[0] is not None

if total_count and total_count > MAX_FETCH:
    st.warning(
        f"Collection has {total_count} records; showing the first {MAX_FETCH} "
        "for performance. Narrow with filters or raise MAX_FETCH in the script."
    )

# Build a flat dataframe
rows = []
for i, _id in enumerate(ids):
    row = {"id": _id}
    if docs[i] is not None:
        row["document"] = docs[i]
    meta = metas[i] or {}
    for k, v in meta.items():
        row[f"meta.{k}"] = v
    if has_vectors:
        vec = np.array(embeddings[i])
        row["_vector"] = vec
        row["vector_dim"] = vec.shape[0]
        preview = ", ".join(f"{x:.3f}" for x in vec[:5])
        row["vector_preview"] = f"[{preview}, ...]" if vec.shape[0] > 5 else f"[{preview}]"
    rows.append(row)

df = pd.DataFrame(rows)

# --------------------------------------------------------------------------
# Tab 1: Records table with search/filter
# --------------------------------------------------------------------------
with tab_table:
    fcol1, fcol2 = st.columns([2, 2])
    with fcol1:
        text_query = st.text_input("Search in document text", "")
    with fcol2:
        meta_cols = [c for c in df.columns if c.startswith("meta.")]
        meta_filter_col = st.selectbox("Filter by metadata field", ["(none)"] + meta_cols)

    filtered = df.copy()
    if text_query and "document" in filtered.columns:
        filtered = filtered[filtered["document"].astype(str).str.contains(text_query, case=False, na=False)]

    if meta_filter_col != "(none)":
        options = sorted(df[meta_filter_col].dropna().unique().tolist(), key=str)
        chosen = st.multiselect(f"Values for {meta_filter_col}", options, default=options)
        filtered = filtered[filtered[meta_filter_col].isin(chosen)]

    st.caption(f"Showing {len(filtered)} of {len(df)} fetched records")

    display_cols = [c for c in filtered.columns if c not in ("_vector",)]
    st.dataframe(filtered[display_cols], use_container_width=True, height=450)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            "⬇️ Download filtered as CSV",
            filtered[display_cols].to_csv(index=False).encode("utf-8"),
            file_name=f"{coll_name}_records.csv",
            mime="text/csv",
        )
    with dl_col2:
        export_records = filtered.drop(columns=["_vector"], errors="ignore").to_dict(orient="records")
        st.download_button(
            "⬇️ Download filtered as JSON",
            json.dumps(export_records, indent=2, default=str).encode("utf-8"),
            file_name=f"{coll_name}_records.json",
            mime="application/json",
        )

# --------------------------------------------------------------------------
# Tab 2: Vector explorer (dimensionality reduction)
# --------------------------------------------------------------------------
with tab_vectors:
    if not has_vectors:
        st.info(
            "This collection's `get()` did not return embeddings (some ChromaDB "
            "setups omit them by default, or the collection has no vectors)."
        )
    else:
        vcol1, vcol2, vcol3 = st.columns(3)
        with vcol1:
            method = st.selectbox("Reduction method", ["PCA", "t-SNE"])
        with vcol2:
            n_dims = st.radio("Dimensions", [2, 3], horizontal=True)
        with vcol3:
            color_by = st.selectbox("Color by", ["(none)"] + [c for c in df.columns if c.startswith("meta.")])

        vectors = np.stack(df["_vector"].values)
        st.caption(f"{vectors.shape[0]} vectors × {vectors.shape[1]} dimensions")

        if vectors.shape[0] < n_dims + 1:
            st.warning("Not enough records to project at this dimensionality.")
        else:
            with st.spinner(f"Running {method}..."):
                if method == "PCA":
                    from sklearn.decomposition import PCA
                    reducer = PCA(n_components=n_dims, random_state=42)
                    coords = reducer.fit_transform(vectors)
                    var_explained = reducer.explained_variance_ratio_.sum()
                    st.caption(f"Explained variance: {var_explained:.1%}")
                else:
                    from sklearn.manifold import TSNE
                    perplexity = min(30, max(2, vectors.shape[0] // 4))
                    reducer = TSNE(n_components=n_dims, random_state=42, perplexity=perplexity, init="pca")
                    coords = reducer.fit_transform(vectors)

            plot_df = df.copy()
            plot_df["x"] = coords[:, 0]
            plot_df["y"] = coords[:, 1]
            if n_dims == 3:
                plot_df["z"] = coords[:, 2]

            hover_cols = ["id"]
            if "document" in plot_df.columns:
                plot_df["document_preview"] = plot_df["document"].astype(str).str.slice(0, 80)
                hover_cols.append("document_preview")

            color_arg = None if color_by == "(none)" else color_by

            if n_dims == 2:
                fig = px.scatter(
                    plot_df, x="x", y="y", color=color_arg, hover_data=hover_cols,
                    title=f"{method} projection ({coll_name})",
                )
            else:
                fig = px.scatter_3d(
                    plot_df, x="x", y="y", z="z", color=color_arg, hover_data=hover_cols,
                    title=f"{method} projection ({coll_name})",
                )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# Tab 3: Raw record inspector (full vector + metadata)
# --------------------------------------------------------------------------
with tab_json:
    chosen_id = st.selectbox("Pick a record by id", ids)
    idx = ids.index(chosen_id)

    st.subheader("Document")
    st.write(docs[idx] if docs[idx] is not None else "_(no document text)_")

    st.subheader("Metadata")
    st.json(metas[idx] or {})

    st.subheader("Embedding vector")
    if has_vectors:
        vec = np.array(embeddings[idx])
        st.caption(f"Dimensionality: {vec.shape[0]}")
        st.dataframe(
            pd.DataFrame({"dimension": range(vec.shape[0]), "value": vec}),
            use_container_width=True,
            height=300,
        )
        st.download_button(
            "⬇️ Download this vector as JSON",
            json.dumps(vec.tolist()).encode("utf-8"),
            file_name=f"{coll_name}_{chosen_id}_vector.json",
            mime="application/json",
        )
    else:
        st.write("_(no embedding available for this record)_")