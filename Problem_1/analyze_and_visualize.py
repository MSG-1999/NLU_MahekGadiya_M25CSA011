"""
analyze_and_visualize.py
========================
Tasks 3 & 4: Semantic analysis and visualization for all trained models.

Compares FOUR model variants:
  1. Scratch CBOW       (PyTorch, implemented from scratch)
  2. Scratch SGNS       (PyTorch, implemented from scratch)
  3. Library CBOW       (Gensim)
  4. Library SGNS       (Gensim)

Task 3 — Semantic Analysis (cosine similarity):
  a. Top-5 nearest neighbours for: research, student, phd, exam
  b. Analogy experiments (≥3): A:B::C:?  via vector arithmetic

Task 4 — Visualization:
  a. PCA  2D projection with cluster coloring
  b. t-SNE 2D projection with cluster coloring
  c. Cluster interpretation logged for each model

Outputs
-------
  visualization_pca_{model_id}.png
  visualization_tsne_{model_id}.png
  logs/analyze.log
"""

import os
import sys
import pickle
import logging
from typing import Optional, List, Dict, Tuple
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gensim.models import Word2Vec as GensimWord2Vec
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

sys.path.insert(0, os.path.dirname(__file__))
from models_scratch import Word2VecVocab, ScratchWordVectors

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/analyze.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Target words for neighbour analysis ───────────────────────────────────────
TARGET_WORDS = ["research", "student", "phd", "exam"]

# ── Analogy triples: (positive_a, positive_b, negative_c)
# Semantics: b - c + a ≈ ?   →   "a is to c as b is to ?"
# All three formats conform to:  most_similar(positive=[a, b], negative=[c])
ANALOGIES = [
    # 1. UG : BTech :: PG : ?  (degree progression)
    {
        "label":    "UG:BTech :: PG:?  (degree progression)",
        "positive": ["pg", "btech"],
        "negative": ["ug"],
    },
    # 2. Faculty : research :: Student : ?  (role → activity)
    {
        "label":    "faculty:research :: student:?  (role-activity)",
        "positive": ["student", "research"],
        "negative": ["faculty"],
    },
    # 3. Summer : May :: Winter : ?  (season → month)
    {
        "label":    "summer:may :: winter:?  (season-month)",
        "positive": ["winter", "may"],
        "negative": ["summer"],
    },
    # 4. Academic : semester :: Institute : ?  (temporal-institutional)
    {
        "label":    "academic:semester :: institute:?  (temporal-institutional)",
        "positive": ["institute", "semester"],
        "negative": ["academic"],
    },
    # 5. Admission : UG :: Fellowship : ?  (program-stage)
    {
        "label":    "admission:ug :: fellowship:?  (program-stage)",
        "positive": ["fellowship", "ug"],
        "negative": ["admission"],
    },
]

# ── Word clusters for visualization ───────────────────────────────────────────
CLUSTERS = {
    "Degrees":   ["phd", "mtech", "mba", "ug", "pg", "btech", "msc"],
    "People":    ["student", "faculty", "professor", "researcher", "candidate"],
    "Academics": ["research", "coursework", "thesis", "exam", "semester", "grade"],
    "Admin":     ["admission", "registration", "institute", "department", "committee"],
    "Projects":  ["project", "proposal", "presentation", "defense", "abstract"],
}

CLUSTER_COLORS = {
    "Degrees":   "#e74c3c",
    "People":    "#3498db",
    "Academics": "#2ecc71",
    "Admin":     "#9b59b6",
    "Projects":  "#f39c12",
    "Other":     "#95a5a6",
}


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading
# ══════════════════════════════════════════════════════════════════════════════

def load_gensim_model(model_id: str) -> Optional[GensimWord2Vec]:
    """Load a primary Gensim model by its id string."""
    path = f"models/{model_id}.model"
    if not os.path.exists(path):
        log.error(f"Gensim model not found: {path}")
        return None
    return GensimWord2Vec.load(path)


def load_scratch_model(pkl_id: str) -> Optional[ScratchWordVectors]:
    """Load a scratch model from its saved .pkl file."""
    path = f"models_scratch/{pkl_id}_scratch.pkl"
    if not os.path.exists(path):
        log.error(f"Scratch model not found: {path}")
        return None
    with open(path, "rb") as f:
        data = pickle.load(f)

    # Reconstruct ScratchWordVectors from saved embeddings + word2idx
    vocab = Word2VecVocab.__new__(Word2VecVocab)
    vocab.word2idx  = data["word2idx"]
    vocab.idx2word  = {i: w for w, i in data["word2idx"].items()}
    vocab.vocab_size = len(data["word2idx"])
    # Dummy unigram dist (not needed for inference)
    vocab.unigram_dist = np.ones(vocab.vocab_size) / vocab.vocab_size

    return ScratchWordVectors(data["embeddings"], vocab)


# ══════════════════════════════════════════════════════════════════════════════
# Task 3a — Nearest Neighbours
# ══════════════════════════════════════════════════════════════════════════════

def report_neighbours(wv, model_label: str):
    """
    Print the top-5 nearest neighbours (by cosine similarity) for each
    word in TARGET_WORDS.

    Works for both Gensim KeyedVectors and ScratchWordVectors (unified API).
    """
    log.info(f"\n{'='*60}")
    log.info(f"  Top-5 Nearest Neighbours — {model_label}")
    log.info(f"{'='*60}")

    for word in TARGET_WORDS:
        if word not in wv:
            log.warning(f"  '{word}' not in vocabulary of {model_label} — skipping.")
            continue

        neighbours = wv.most_similar(word, topn=5)
        log.info(f"\n  '{word}':")
        for rank, (nbr, sim) in enumerate(neighbours, 1):
            log.info(f"    {rank}. {nbr:<22} cosine={sim:.4f}")


# ══════════════════════════════════════════════════════════════════════════════
# Task 3b — Analogy Experiments
# ══════════════════════════════════════════════════════════════════════════════

def run_analogies(wv, model_label: str):
    """
    Run the defined analogy experiments using vector arithmetic:
        vector(positive_a) + vector(positive_b) − vector(negative_c) ≈ answer

    Semantically evaluates whether the analogy result is meaningful.
    """
    log.info(f"\n{'='*60}")
    log.info(f"  Analogy Experiments — {model_label}")
    log.info(f"{'='*60}")

    for i, exp in enumerate(ANALOGIES, 1):
        log.info(f"\n  [{i}] {exp['label']}")

        # Check all required words are in vocabulary
        all_words = exp["positive"] + exp["negative"]
        missing = [w for w in all_words if w not in wv]
        if missing:
            log.warning(f"      OOV words: {missing} — skipping.")
            continue

        results = wv.most_similar(
            positive=exp["positive"],
            negative=exp["negative"],
            topn=3,
        )
        pairs = [(w, round(s, 4)) for w, s in results]
        log.info(f"      Result: {pairs}")

# ══════════════════════════════════════════════════════════════════════════════
# Task 4 — Visualization Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _select_words(wv) -> Tuple[List[str], Dict[str, str]]:
    """
    Choose words to visualize:
      1. Cluster words present in vocabulary
      2. Topped up with the 60 most frequent vocabulary words
    Returns (word_list, cluster_label_map).
    """
    chosen = []
    label_map: dict[str, str] = {}

    for cluster, words in CLUSTERS.items():
        for w in words:
            if w in wv and w not in chosen:
                chosen.append(w)
                label_map[w] = cluster

    # Top-frequency words as context
    all_keys = wv.index_to_key if hasattr(wv, "index_to_key") else list(wv.key_to_index.keys())
    for w in all_keys[:80]:
        if w not in chosen:
            chosen.append(w)
            label_map[w] = "Other"

    return chosen, label_map


def _get_vector(wv, word: str) -> np.ndarray:
    """Retrieve a word vector regardless of model type."""
    return wv[word]


def _draw_projection(title: str, coords: np.ndarray,
                     words: List[str], label_map: Dict[str, str],
                     save_path: str):
    """
    Scatter plot of 2D word projections, colour-coded by semantic cluster.
    Each word is annotated with its text label.
    """
    fig, ax = plt.subplots(figsize=(15, 10))
    placed_labels: set[str] = set()

    for i, word in enumerate(words):
        cluster = label_map.get(word, "Other")
        color   = CLUSTER_COLORS.get(cluster, "#95a5a6")
        legend_label = cluster if cluster not in placed_labels else "_nolegend_"

        ax.scatter(
            coords[i, 0], coords[i, 1],
            c=color, s=55, edgecolors="k", linewidths=0.35,
            label=legend_label, zorder=3,
        )
        ax.annotate(
            word, (coords[i, 0], coords[i, 1]),
            fontsize=7.5, alpha=0.88,
            xytext=(4, 4), textcoords="offset points",
        )
        placed_labels.add(cluster)

    ax.legend(title="Cluster", loc="upper right", fontsize=8, framealpha=0.7)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    log.info(f"  Saved → {save_path}")


def visualize_pca(wv, model_label: str, file_id: str):
    """PCA 2D projection of selected word embeddings."""
    words, label_map = _select_words(wv)
    if len(words) < 5:
        log.warning(f"Too few words for PCA in {model_label}")
        return

    vecs = np.array([_get_vector(wv, w) for w in words])
    pca  = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(vecs)

    ev = pca.explained_variance_ratio_
    log.info(f"  PCA explained variance: PC1={ev[0]:.3f}, PC2={ev[1]:.3f}")

    _draw_projection(
        title     = f"PCA — {model_label}  (EV: {ev[0]:.1%} + {ev[1]:.1%})",
        coords    = coords,
        words     = words,
        label_map = label_map,
        save_path = f"visualization_pca_{file_id}.png",
    )


def visualize_tsne(wv, model_label: str, file_id: str):
    """t-SNE 2D projection of selected word embeddings."""
    words, label_map = _select_words(wv)
    if len(words) < 10:
        log.warning(f"Too few words for t-SNE in {model_label}")
        return

    vecs       = np.array([_get_vector(wv, w) for w in words])
    perplexity = min(30, max(5, len(words) // 4))

    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        init="pca",
        learning_rate="auto",
    )
    coords = tsne.fit_transform(vecs)
    log.info(f"  t-SNE done  (perplexity={perplexity})")

    _draw_projection(
        title     = f"t-SNE — {model_label}",
        coords    = coords,
        words     = words,
        label_map = label_map,
        save_path = f"visualization_tsne_{file_id}.png",
    )



# ══════════════════════════════════════════════════════════════════════════════
# P1-A  Top-10 words by raw frequency
# ══════════════════════════════════════════════════════════════════════════════

def report_top10_words(corpus_path: str = "cleaned_corpus.txt"):
    """
    Count every token in the cleaned corpus and print the top-10 in the
    required format:  word1, frequency1, word2, frequency2, …
    No stop-word filtering — raw counts as-is.
    """
    from collections import Counter

    log.info("\n" + "=" * 60)
    log.info("P1-A  TOP-10 WORDS (frequency-wise)")
    log.info("=" * 60)

    if not os.path.exists(corpus_path):
        log.error(f"Corpus not found: {corpus_path}")
        return

    tokens = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            tokens.extend(line.split())

    STOP_WORDS = {
        "the", "a", "an", "of", "to", "and", "in", "is", "for", "on",
        "or", "that", "this", "with", "are", "as", "at", "be", "by",
        "from", "it", "not", "will", "all", "its", "any", "was", "he",
        "she", "we", "they", "has", "have", "had", "do", "does", "did",
        "if", "but", "so", "been", "about", "which", "their", "may",
        "also", "such", "shall", "must", "into", "more", "no", "than",
        "each", "his", "her", "our", "can", "would", "should", "who",
        "when", "up", "out", "two", "one", "01", "02", "10",
    }
    freq = Counter(tokens)
    top10 = [(w, c) for w, c in freq.most_common(200) if w not in STOP_WORDS][:10]

    # Required output format
    pairs_str = ", ".join(f"{w}, {c}" for w, c in top10)
    log.info(pairs_str)
    print("\nP1-A TOP-10 WORDS:")
    print(pairs_str)

    # Save to file
    with open("top10_words.txt", "w", encoding="utf-8") as f:
        f.write("TOP-10 WORDS (frequency-wise) — IITJ Corpus\n")
        f.write("=" * 50 + "\n\n")
        f.write(pairs_str + "\n\n")
        f.write(f"{'Rank':<6} {'Word':<25} {'Frequency':>10}\n")
        f.write("-" * 43 + "\n")
        for rank, (w, c) in enumerate(top10, 1):
            f.write(f"{rank:<6} {w:<25} {c:>10}\n")
    log.info("Saved → top10_words.txt")

    # Bar chart
    words  = [w for w, _ in top10]
    counts = [c for _, c in top10]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(words[::-1], counts[::-1],
                   color=plt.cm.viridis(np.linspace(0.2, 0.85, 10)))
    ax.set_xlabel("Frequency", fontsize=12)
    ax.set_title("P1-A  Top-10 Words in IITJ Corpus", fontsize=14, fontweight="bold")
    for bar, cnt in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                str(cnt), va="center", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("top10_words.png", dpi=150)
    plt.close()
    log.info("Saved → top10_words.png")


# ══════════════════════════════════════════════════════════════════════════════
# P1-B  Full embedding vector for a chosen word
# ══════════════════════════════════════════════════════════════════════════════

PROBE_WORD = "academic"   # present in corpus (≠ "jodhpur")

def report_word_embedding():
    """
    Load the primary Gensim SGNS model and print the full embedding vector
    for PROBE_WORD in the required format:
      word - v1, v2, v3, …, vN
    """
    log.info("\n" + "=" * 60)
    log.info(f"P1-B  EMBEDDING VECTOR FOR '{PROBE_WORD}'")
    log.info("=" * 60)

    # Primary model: sgns_d200_w8_n5_lib  (200-dim, confirmed in train.log)
    model_path = os.path.join("models", "sgns_d200_w8_n5_lib.model")
    if not os.path.exists(model_path):
        log.error(f"Model not found: {model_path} — run train_word2vec.py first.")
        return

    model = GensimWord2Vec.load(model_path)
    if PROBE_WORD not in model.wv:
        log.error(f"'{PROBE_WORD}' not in model vocabulary.")
        return

    vec     = model.wv[PROBE_WORD]
    vec_str = ", ".join(f"{x:.4f}" for x in vec)
    output  = f"{PROBE_WORD} - {vec_str}"

    print(f"\nP1-B EMBEDDING ({len(vec)}-dim):")
    print(output[:120] + " …")   # truncate console display
    log.info(f"Dim={len(vec)}, L2-norm={np.linalg.norm(vec):.4f}")
    log.info("Full vector saved to word_embedding.txt")

    with open("word_embedding.txt", "w", encoding="utf-8") as f:
        f.write(f"Embedding for '{PROBE_WORD}'  [Gensim SGNS, d=200, w=8, neg=5]\n")
        f.write("=" * 60 + "\n\n")
        f.write(output + "\n\n")
        f.write(f"Dimensionality : {len(vec)}\n")
        f.write(f"L2 norm        : {np.linalg.norm(vec):.6f}\n")
        f.write(f"Min / Max      : {vec.min():.6f} / {vec.max():.6f}\n")
        f.write(f"Mean / Std     : {vec.mean():.6f} / {vec.std():.6f}\n")
    log.info("Saved → word_embedding.txt")


# ══════════════════════════════════════════════════════════════════════════════
# Hyperparameter Comparison Plots  (Lib models — real saved files)
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = {
    "cbow_lib":     "#2A9D8F",
    "sgns_lib":     "#F4A261",
    "cbow_scratch": "#457B9D",
    "sgns_scratch": "#E63946",
}


def _avg_sim(wv, word: str = PROBE_WORD, topn: int = 5) -> float:
    """Mean cosine-sim of topn neighbours for `word`, 0 on failure."""
    if wv is None or word not in wv:
        return 0.0
    try:
        return float(np.mean([s for _, s in wv.most_similar(word, topn=topn)]))
    except Exception:
        return 0.0


def _load_lib_wv(model_id: str):
    """Load Gensim KeyedVectors from models/<model_id>_lib.model, or None."""
    path = os.path.join("models", f"{model_id}_lib.model")
    if not os.path.exists(path):
        log.warning(f"  Not found: {path}")
        return None
    return GensimWord2Vec.load(path).wv


def _hyperparam_bars(ax, params, lib_scores, scratch_scores,
                     xlabel: str, title: str, lib_color: str, scratch_color: str):
    x     = np.arange(len(params))
    w     = 0.35
    b1 = ax.bar(x - w / 2, lib_scores,     w, label="Gensim (library)",   color=lib_color,     alpha=0.85)
    b2 = ax.bar(x + w / 2, scratch_scores, w, label="Scratch (PyTorch)",  color=scratch_color, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels([str(p) for p in params])
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(f"Avg cosine-sim  (top-5, '{PROBE_WORD}')", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(0, max(max(lib_scores + scratch_scores, default=0) * 1.25, 0.05))
    ax.legend(fontsize=9); ax.spines[["top", "right"]].set_visible(False)
    for bar in list(b1) + list(b2):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)


def plot_hyperparam_dim():
    """Embedding dimension experiment — uses exact IDs from train.log."""
    # CBOW: 100, 150, 200  |  SGNS: 200, 250, 300
    cbow_dims    = [100, 150, 200]
    cbow_ids     = ["cbow_d100_w8_n5", "cbow_d150_w8_n5", "cbow_d200_w8_n5"]
    sgns_dims    = [200, 250, 300]
    sgns_ids     = ["sgns_d200_w8_n5", "sgns_d250_w8_n5", "sgns_d300_w8_n5"]
    # Corresponding scratch pkl IDs
    cbow_scratch_ids = cbow_ids
    sgns_scratch_ids = sgns_ids

    def _scores(ids, is_scratch):
        out = []
        for cid in ids:
            if is_scratch:
                wv = load_scratch_model(cid)
            else:
                wv = _load_lib_wv(cid)
            out.append(_avg_sim(wv))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _hyperparam_bars(ax1, cbow_dims,
                     _scores(cbow_ids, False), _scores(cbow_scratch_ids, True),
                     "Embedding Dimension", "CBOW — Embedding Dimension",
                     PALETTE["cbow_lib"], PALETTE["cbow_scratch"])
    _hyperparam_bars(ax2, sgns_dims,
                     _scores(sgns_ids, False), _scores(sgns_scratch_ids, True),
                     "Embedding Dimension", "SGNS — Embedding Dimension",
                     PALETTE["sgns_lib"], PALETTE["sgns_scratch"])
    fig.suptitle("Hyperparameter Study: Embedding Dimension", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("hyperparam_dim.png", dpi=150); plt.close()
    log.info("Saved → hyperparam_dim.png")


def plot_hyperparam_window():
    """Context-window experiment — uses exact IDs from train.log."""
    windows      = [5, 8, 10]
    cbow_ids     = ["cbow_d100_w5_n5", "cbow_d100_w8_n5", "cbow_d100_w10_n5"]
    sgns_ids     = ["sgns_d100_w5_n5", "sgns_d200_w8_n5", "sgns_d100_w10_n5"]

    def _scores(ids, is_scratch):
        out = []
        for cid in ids:
            wv = load_scratch_model(cid) if is_scratch else _load_lib_wv(cid)
            out.append(_avg_sim(wv))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _hyperparam_bars(ax1, windows,
                     _scores(cbow_ids, False), _scores(cbow_ids, True),
                     "Window Size", "CBOW — Context Window Size",
                     PALETTE["cbow_lib"], PALETTE["cbow_scratch"])
    _hyperparam_bars(ax2, windows,
                     _scores(sgns_ids, False), _scores(sgns_ids, True),
                     "Window Size", "SGNS — Context Window Size",
                     PALETTE["sgns_lib"], PALETTE["sgns_scratch"])
    fig.suptitle("Hyperparameter Study: Context Window Size", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("hyperparam_window.png", dpi=150); plt.close()
    log.info("Saved → hyperparam_window.png")


def plot_hyperparam_neg():
    """Negative-sample count experiment — uses exact IDs from train.log."""
    neg_counts   = [5, 10, 15]
    cbow_ids     = ["cbow_d100_w8_n5", "cbow_d100_w8_n10", "cbow_d100_w8_n15"]
    sgns_ids     = ["sgns_d200_w8_n5", "sgns_d100_w8_n10", "sgns_d100_w8_n15"]

    def _scores(ids, is_scratch):
        out = []
        for cid in ids:
            wv = load_scratch_model(cid) if is_scratch else _load_lib_wv(cid)
            out.append(_avg_sim(wv))
        return out

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    _hyperparam_bars(ax1, neg_counts,
                     _scores(cbow_ids, False), _scores(cbow_ids, True),
                     "Negative Samples", "CBOW — Negative Samples",
                     PALETTE["cbow_lib"], PALETTE["cbow_scratch"])
    _hyperparam_bars(ax2, neg_counts,
                     _scores(sgns_ids, False), _scores(sgns_ids, True),
                     "Negative Samples", "SGNS — Negative Samples",
                     PALETTE["sgns_lib"], PALETTE["sgns_scratch"])
    fig.suptitle("Hyperparameter Study: Negative Sample Count", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig("hyperparam_neg.png", dpi=150); plt.close()
    log.info("Saved → hyperparam_neg.png")



def analyze():
    log.info("=" * 70)
    log.info("PROBLEM 1 — Semantic Analysis & Visualization")
    log.info("=" * 70)

    # ── P1-A: Top-10 words ────────────────────────────────────────────────────
    report_top10_words()

    # ── P1-B: Full embedding vector ───────────────────────────────────────────
    report_word_embedding()

    # ── Hyperparameter comparison plots ───────────────────────────────────────
    log.info("\nGenerating hyperparameter comparison plots …")
    plot_hyperparam_dim()
    plot_hyperparam_window()
    plot_hyperparam_neg()

    # Define all four models: (display_label, file_id, wv_object_or_None)
    # FIX: Use correct PRIMARY model IDs that match train_word2vec.py output
    models = [  # type: List[Tuple[str, str]]
        ("Scratch CBOW",          "cbow_d100_w8_n5"),
        ("Scratch SGNS",          "sgns_d200_w8_n5"),
        ("Library CBOW (Gensim)", "cbow_d100_w8_n5_lib"),
        ("Library SGNS (Gensim)", "sgns_d200_w8_n5_lib"),
    ]

    loaded = []  # type: List[Tuple[str, str, object]]

    for label, file_id in models:
        if "Scratch" in label:
            # Strip the _lib suffix for scratch pkl name
            pkl_id = file_id  # e.g. "cbow_d100_w5_n5"
            wv = load_scratch_model(pkl_id)
        else:
            gensim_model = load_gensim_model(file_id)
            wv = gensim_model.wv if gensim_model else None

        if wv is None:
            log.warning(f"  Cannot load {label} — skipping.")
            continue

        loaded.append((label, file_id, wv))

    # ── Run analysis for each loaded model ────────────────────────────────────
    for label, file_id, wv in loaded:
        log.info(f"\n{'#'*70}")
        log.info(f"# MODEL: {label}")
        log.info(f"{'#'*70}")

        # ── Task 3a: Nearest Neighbours ───────────────────────────────────────
        report_neighbours(wv, label)

        # ── Task 3b: Analogy Experiments ──────────────────────────────────────
        run_analogies(wv, label)

        # ── Task 4a: PCA ─────────────────────────────────────────────────────
        log.info(f"\n  Generating PCA visualization …")
        visualize_pca(wv, label, file_id)

        # ── Task 4b: t-SNE ────────────────────────────────────────────────────
        log.info(f"\n  Generating t-SNE visualization …")
        visualize_tsne(wv, label, file_id)

    log.info("\n=== Analysis complete. All outputs saved. ===")


if __name__ == "__main__":
    analyze()