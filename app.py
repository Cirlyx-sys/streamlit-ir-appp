import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Load files (same folder as app.py)
# ----------------------------
@st.cache_data
def load_data():
    embeddings = np.load("embeddings.npy")  # shape: (num_docs, dim)
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = [line.strip() for line in f.readlines() if line.strip()]
    return embeddings, documents

embeddings, documents = load_data()

# Safety checks
if embeddings.shape[0] != len(documents):
    st.error(
        f"Mismatch! embeddings rows = {embeddings.shape[0]} but documents = {len(documents)}.\n"
        "Fix: documents.txt must have the same number of lines as embeddings.npy rows."
    )
    st.stop()

embedding_dim = embeddings.shape[1]

# ----------------------------
# Query embedding (demo placeholder)
# ----------------------------
def get_query_embedding(query: str, dim: int) -> np.ndarray:
    # Deterministic random embedding (same query -> same vector)
    seed = abs(hash(query)) % (2**32)
    rng = np.random.default_rng(seed)
    return rng.random(dim, dtype=np.float32)

# ----------------------------
# Retrieval
# ----------------------------
def retrieve_top_k(query_embedding: np.ndarray, doc_embeddings: np.ndarray, docs: list[str], k: int):
    sims = cosine_similarity(query_embedding.reshape(1, -1), doc_embeddings)[0]  # shape: (num_docs,)
    top_idx = np.argsort(sims)[::-1][:k]
    return [(docs[i], float(sims[i])) for i in top_idx]

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("IR App (Streamlit) — Top-K Retrieval")

st.write(f"Loaded **{len(documents)}** documents with embedding dim **{embedding_dim}**.")

query = st.text_input("Enter your query:")
top_k = st.slider("Top K results", min_value=1, max_value=min(10, len(documents)), value=5)

if st.button("Search"):
    if not query.strip():
        st.warning("Please type a query first.")
    else:
        q_emb = get_query_embedding(query, embedding_dim)
        results = retrieve_top_k(q_emb, embeddings, documents, top_k)

        st.subheader("Results")
        for rank, (doc, score) in enumerate(results, start=1):
            st.write(f"**{rank}.** {doc}")
            st.caption(f"Similarity: {score:.4f}")
