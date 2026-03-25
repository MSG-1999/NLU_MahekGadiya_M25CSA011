import os
import re
import logging
from collections import Counter

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers / headless runs
import matplotlib.pyplot as plt

# FIX: Import wordcloud at module level so import errors are caught early
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/preprocess.log", mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

if not WORDCLOUD_AVAILABLE:
    log.warning("wordcloud package not installed — WordCloud generation will be skipped.")
    log.warning("Install it with: pip install wordcloud")

# ── Stop words to exclude from WordCloud ─────────────────────────────────────
# common words to ignore in wordcloud
STOP_WORDS = {
    "the", "a", "an", "of", "to", "and", "in", "is", "for", "on",
    "or", "that", "this", "with", "are", "as", "at", "be", "by",
    "from", "it", "not", "will", "all", "its", "any", "was", "he",
    "she", "we", "they", "has", "have", "had", "do", "does", "did",
    "if", "but", "so", "be", "been", "about", "which", "their",
    "may", "also", "such", "shall", "must", "into", "more", "no",
    "than", "each", "his", "her", "our", "can", "would", "should",
    "who", "when", "up", "out", "two", "one", "01", "02", "10",
}


def split_sentences(text: str) -> list:
    """
    Split a paragraph into approximate sentences using punctuation heuristics.
    Returns a list of raw sentence strings.
    """
    # Split on period/question/exclamation followed by whitespace
    sentences = re.split(r"(?<=[.?!])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize_sentence(sentence: str) -> list:
    """
    Extract lowercase alphabetic tokens of length >= 2 from a sentence.
    Discards numbers, punctuation, and single-character tokens.
    """
    tokens = re.findall(r"[a-z]{2,}", sentence.lower())
    return tokens


def make_wordcloud(word_freq: dict, out_path: str):
    """Generate and save a matplotlib-based word cloud."""
    if not WORDCLOUD_AVAILABLE:
        log.warning("Skipping WordCloud — wordcloud package not installed.")
        return
    try:
        wc = WordCloud(
            width=1200, height=600,
            background_color="white",
            max_words=120,
            colormap="viridis",
        ).generate_from_frequencies(word_freq)
        plt.figure(figsize=(14, 7))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        log.info(f"WordCloud saved → {out_path}")
    except Exception as e:
        log.warning(f"WordCloud generation failed: {e}")


def preprocess():
    raw_path = "raw_corpus.txt"
    
    # checking if raw corpus exists
    if not os.path.exists(raw_path):
        log.error(f"Raw corpus not found at '{raw_path}'. Run collect_data.py first.")
        return

    log.info("Loading raw corpus …")
    with open(raw_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # ── Step 1: Cleaning unwanted content ───────────────────
    # Remove our own inserted "# SOURCE: …" markers
    raw_text = re.sub(r"#\s*SOURCE:.*\n", "", raw_text)

    # Remove URLs, email addresses
    raw_text = re.sub(r"https?://\S+|www\.\S+", " ", raw_text)
    raw_text = re.sub(r"\S+@\S+\.\S+", " ", raw_text)

    # Remove standalone numbers and short codes
    raw_text = re.sub(r"\b\d+\b", " ", raw_text)

    # ── Step 2: Sentence splitting + tokenization ─────────────────────
    sentences = []
    for paragraph in raw_text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sent in split_sentences(paragraph):
            tokens = tokenize_sentence(sent)
            if len(tokens) >= 3:  # keep only sentences with 3+ useful tokens
                sentences.append(tokens)

    # ── Step 3: Statistics ────────────────────────────────────────────────────
    all_tokens = [tok for sent in sentences for tok in sent]
    vocab = set(all_tokens)
    word_freq = Counter(all_tokens)

    log.info("=" * 60)
    log.info("DATASET STATISTICS")
    log.info("=" * 60)
    log.info(f"  Total sentences (documents) : {len(sentences):,}")
    log.info(f"  Total tokens                : {len(all_tokens):,}")
    log.info(f"  Vocabulary size             : {len(vocab):,}")
    log.info(f"  Top-20 words:")
    count_printed = 0
    for word, count in word_freq.most_common():
        if word not in STOP_WORDS:
            log.info(f"    {word:<25} {count:>6}")
            count_printed += 1
            if count_printed == 20:
                break
    log.info("=" * 60)

    # ── Step 4: Write cleaned corpus ───────────────────────────────────────────────────
    out_path = "cleaned_corpus.txt"
    
    with open(out_path, "w", encoding="utf-8") as f:
        for sent in sentences:
            f.write(" ".join(sent) + "\n")
    log.info(f"Cleaned corpus written → {out_path}")

    # # removing stopwords before plotting ───────────────────────────────────────
    content_freq = {w: c for w, c in word_freq.items() if w not in STOP_WORDS and len(w) > 2}
    make_wordcloud(content_freq, "wordcloud.png")

# ── Run ─────
if __name__ == "__main__":
    preprocess()