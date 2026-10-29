import json
import pickle
import re

import numpy as np
import faiss
import streamlit as st
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

JSON_PATH = "data/diseases_txt.json"
FAISS_PATH = "data/diseases.faiss"
BM25_PATH = "data/bm25.pkl"

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

CANDIDATE_K = 100
MIN_RELATIVE_SCORE = 0.25
BM25_WEIGHT = 0.45
DENSE_WEIGHT = 0.55
RRF_K = 60

st.set_page_config(
    page_title="Disease & Peptide Search",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Disease & Peptide Search")
st.caption(
    "Hybrid semantic + keyword search over disease descriptions and related peptides."
)


@st.cache_resource(show_spinner="Loading search index...")
def load_resources():
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        diseases = json.load(f)

    model = SentenceTransformer(MODEL_NAME)

    faiss_index = faiss.read_index(FAISS_PATH)

    with open(BM25_PATH, "rb") as f:
        bm25 = pickle.load(f)

    return diseases, model, faiss_index, bm25


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def tokenize(text: str):
    return re.findall(
        r"[a-zα-ωάέήίόύώϊϋΐΰ0-9]+",
        normalize_text(text)
    )


def bm25_search(query: str, top_k: int):
    tokens = tokenize(query)

    if not tokens:
        return []

    scores = BM25.get_scores(tokens)

    if scores.max() <= 0:
        return []

    normalized = scores / scores.max()
    ranked = np.argsort(normalized)[::-1][:top_k]

    return [
        (int(i), float(normalized[i]))
        for i in ranked
        if normalized[i] > 0
    ]


def dense_search(query: str, top_k: int):
    embedding = MODEL.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = FAISS_INDEX.search(embedding, top_k)

    return [
        (int(i), float(score))
        for score, i in zip(scores[0], indices[0])
        if score > 0
    ]


def weighted_rrf(ranked_lists, weights, k=RRF_K):
    fused = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(fused.items(), key=lambda x: x[1], reverse=True)


def hybrid_search(query: str, top_k: int):
    bm25_results = bm25_search(query, CANDIDATE_K)
    dense_results = dense_search(query, CANDIDATE_K)

    fused = weighted_rrf(
        [bm25_results, dense_results],
        [BM25_WEIGHT, DENSE_WEIGHT]
    )

    if not fused:
        return []

    max_score = fused[0][1]
    results = []

    for doc_id, score in fused:
        if score < max_score * MIN_RELATIVE_SCORE:
            continue

        disease = DISEASES[doc_id]

        results.append({
            "score": score,
            "disease": disease.get("disease", ""),
            "description": disease.get("description", ""),
            "peptides": disease.get("peptides", []),
            "source_file": disease.get("source_file", "")
        })

        if len(results) >= top_k:
            break

    return results


DISEASES, MODEL, FAISS_INDEX, BM25 = load_resources()

st.success(f"Loaded {len(DISEASES)} diseases.")

query = st.text_input(
    "Search symptom, disease or condition",
    placeholder="e.g. high triglycerides, low oxygen in blood, χαμηλός αιματοκρίτης"
)

top_k = st.slider(
    "Maximum results",
    min_value=5,
    max_value=50,
    value=15
)

if st.button("Search", type="primary") or query:
    if not query.strip():
        st.warning("Γράψε ένα query για αναζήτηση.")
        st.stop()

    with st.spinner("Searching..."):
        results = hybrid_search(query.strip(), top_k)

    if not results:
        st.info("Δεν βρέθηκαν σχετικά αποτελέσματα.")
        st.stop()

    st.write(f"### {len(results)} σχετικές παθήσεις")

    for i, result in enumerate(results, start=1):
        with st.expander(
            f"#{i} — {result['disease']}  |  score: {result['score']:.5f}"
        ):
            if result["peptides"]:
                st.markdown(
                    "**Related peptides:** " +
                    ", ".join(result["peptides"])
                )
            else:
                st.markdown("**Related peptides:** —")

            st.markdown("### Description")
            st.write(result["description"])
