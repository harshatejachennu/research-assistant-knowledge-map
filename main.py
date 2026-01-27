#!/usr/bin/env python3
"""
RAKM – Research Assistant & Knowledge Map

This module provides a command‑line interface to process PDF or plain‑text
documents, extract keywords, generate an extractive summary and build a
concept map.  An optional interactive mode allows users to ask questions
about the processed document.

The implementation includes:

* A simple tokenizer and stopword filter (NLTK data is not downloaded).
* A RAKE implementation for keyword extraction.
* A TextRank summarizer implemented from scratch using TF–IDF sentence
  vectors and PageRank.
* A concept map generator that draws a circular layout using Matplotlib.
* A semantic search mechanism based on TF–IDF similarity for Q&A.

Usage examples are provided in README.md.
"""

import argparse
import os
import re
import subprocess
from collections import defaultdict, Counter
from math import log
from typing import List, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs


##############################
# Utilities for text handling
##############################

def load_text(input_path: str) -> str:
    """Load and return text from a file.  Supports PDF and plain text.

    PDF files are converted to text using the `pdftotext` command.  If the
    conversion fails, an exception is raised.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file does not exist: {input_path}")
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pdf":
        # Use pdftotext to convert PDF into a temporary text file
        txt_path = input_path + ".txt"
        try:
            subprocess.run(["pdftotext", input_path, txt_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "Failed to convert PDF to text. Ensure `pdftotext` is installed and accessible."
            ) from e
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        # Clean up temporary file
        try:
            os.remove(txt_path)
        except OSError:
            pass
        return text
    elif ext in {".txt", ".md"}:
        with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    else:
        raise ValueError("Unsupported input format. Use .txt or .pdf files.")


def tokenize_sentences(text: str) -> List[str]:
    """Split text into sentences using a simple regular expression."""
    # Replace newlines with spaces, then split on punctuation followed by space
    cleaned = re.sub(r"\s+", " ", text)
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    # Filter out very short sentences
    return [s.strip() for s in sentences if len(s.strip().split()) > 2]


def tokenize_words(sentence: str) -> List[str]:
    """Split a sentence into lowercase words using a regex."""
    return [w for w in re.split(r"[^A-Za-z0-9]+", sentence.lower()) if w]


def get_stopwords() -> set:
    """Return a set of English stopwords.

    The list is derived from the NLTK English stopword set.  See the gist at
    https://gist.github.com/sebleier/554280 for the original source.
    """
    return {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
        "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her",
        "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
        "themselves", "what", "which", "who", "whom", "this", "that", "these", "those",
        "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about",
        "against", "between", "into", "through", "during", "before", "after", "above",
        "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under",
        "again", "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some",
        "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
        "very", "can", "will", "just", "don", "should", "now"
    }


###########################
# RAKE Keyword Extraction
###########################

def rake_keywords(
    sentences: List[str],
    stopwords: set,
    min_char_length: int = 3,
    max_words_length: int = 3,
    max_keywords: int = 20
) -> List[Tuple[str, float]]:
    """Extract keywords from the given sentences using the RAKE algorithm.

    :param sentences: List of sentences
    :param stopwords: Set of stopwords to ignore
    :param min_char_length: Minimum length of a word to be considered
    :param max_words_length: Maximum number of words in a candidate phrase
    :param max_keywords: Maximum number of top keywords to return
    :return: List of (keyword, score) tuples sorted by score descending
    """
    # Generate candidate phrases by grouping non‑stopwords
    phrase_list: List[List[str]] = []
    for sentence in sentences:
        words = tokenize_words(sentence)
        phrase: List[str] = []
        for word in words:
            if len(word) < min_char_length or word in stopwords:
                if phrase:
                    phrase_list.append(phrase)
                    phrase = []
            else:
                phrase.append(word)
        if phrase:
            phrase_list.append(phrase)

    # Calculate word frequency and degree
    word_freq: Dict[str, int] = defaultdict(int)
    word_degree: Dict[str, int] = defaultdict(int)
    for phrase in phrase_list:
        length = len(phrase)
        for word in phrase:
            word_freq[word] += 1
            # degree counts the number of words in the phrase (including itself)
            word_degree[word] += length

    # Compute word scores as degree/ frequency
    word_score: Dict[str, float] = {}
    for word in word_freq:
        word_score[word] = word_degree[word] / float(word_freq[word])

    # Compute candidate scores by summing word scores
    candidate_scores: Dict[str, float] = defaultdict(float)
    for phrase in phrase_list:
        if len(phrase) == 0 or len(phrase) > max_words_length:
            continue
        candidate = " ".join(phrase)
        score = sum(word_score[w] for w in phrase)
        candidate_scores[candidate] += score

    # Sort candidates by score
    sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_candidates[:max_keywords]


#######################################
# TextRank Summarization and TF–IDF
#######################################

def preprocess_sentences(sentences: List[str], stopwords: set) -> List[List[str]]:
    """Preprocess sentences by tokenizing and removing stopwords."""
    processed = []
    for sent in sentences:
        words = [w for w in tokenize_words(sent) if w not in stopwords]
        processed.append(words)
    return processed


def build_tfidf_matrix(processed_sentences: List[List[str]]) -> Tuple[np.ndarray, List[str]]:
    """Build a TF–IDF matrix for the processed sentences.

    Returns the matrix and the vocabulary list.
    """
    # Build vocabulary
    vocabulary = []
    vocab_index = {}
    for words in processed_sentences:
        for w in words:
            if w not in vocab_index:
                vocab_index[w] = len(vocabulary)
                vocabulary.append(w)
    # Compute IDF for each word
    N = len(processed_sentences)
    df = np.zeros(len(vocabulary))
    for words in processed_sentences:
        unique_words = set(words)
        for w in unique_words:
            df[vocab_index[w]] += 1
    idf = np.log((N) / (1 + df))
    # Compute TF–IDF vectors
    tfidf = np.zeros((N, len(vocabulary)))
    for i, words in enumerate(processed_sentences):
        term_counts = Counter(words)
        if len(words) == 0:
            continue
        for w, count in term_counts.items():
            j = vocab_index[w]
            # term frequency normalised by sentence length
            tfidf[i, j] = (count / len(words)) * idf[j]
    return tfidf, vocabulary


def sentence_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    num = np.dot(vec1, vec2)
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 0.0
    return num / denom


def textrank_summary(
    sentences: List[str],
    processed_sentences: List[List[str]],
    num_sentences: int = 5,
    damping: float = 0.85,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> Tuple[str, np.ndarray]:
    """Generate a summary using the TextRank algorithm.

    :param sentences: Original sentence list
    :param processed_sentences: Tokenized sentences without stopwords
    :param num_sentences: Number of sentences to include in summary
    :param damping: Damping factor for PageRank
    :param max_iter: Maximum number of iterations
    :param tol: Convergence tolerance
    :return: (summary string, sentence scores array)
    """
    n = len(sentences)
    if n == 0:
        return "", np.array([])
    # Build TF–IDF matrix and compute similarity matrix
    tfidf, _ = build_tfidf_matrix(processed_sentences)
    # Similarity matrix
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            s = sentence_similarity(tfidf[i], tfidf[j])
            sim[i, j] = sim[j, i] = s
    # Normalise rows
    row_sums = sim.sum(axis=1)
    # Avoid division by zero
    for i in range(n):
        if row_sums[i] == 0:
            row_sums[i] = 1.0
    # Iterative PageRank
    scores = np.ones(n) / n
    for iteration in range(max_iter):
        prev_scores = scores.copy()
        for i in range(n):
            # Summation of incoming similarity scores
            s = 0.0
            for j in range(n):
                if i == j:
                    continue
                if sim[j, i] == 0:
                    continue
                s += (sim[j, i] / row_sums[j]) * prev_scores[j]
            scores[i] = (1 - damping) / n + damping * s
        # Check convergence
        if np.abs(scores - prev_scores).sum() < tol:
            break
    # Select top sentences
    idx = np.argsort(scores)[::-1][:num_sentences]
    idx_sorted = sorted(idx)
    summary = " ".join([sentences[i] for i in idx_sorted])
    return summary, scores


#############################################
# Concept Map Generation
#############################################

def build_co_occurrence(
    keywords: List[str],
    processed_sentences: List[List[str]]
) -> Dict[Tuple[str, str], int]:
    """Compute co‑occurrence counts for each pair of keywords."""
    co_occurrence: Dict[Tuple[str, str], int] = defaultdict(int)
    # Convert list of keywords to set for faster membership tests
    keyword_set = set(keywords)
    for words in processed_sentences:
        present = [w for w in words if w in keyword_set]
        unique_present = set(present)
        # Increment co‑occurrence count for each pair
        for i, kw1 in enumerate(unique_present):
            for kw2 in list(unique_present)[i + 1:]:
                pair = tuple(sorted((kw1, kw2)))
                co_occurrence[pair] += 1
    return co_occurrence


def draw_concept_map(
    keywords: List[str],
    co_occurrence: Dict[Tuple[str, str], int],
    output_path: str
) -> None:
    """Draw and save a concept map showing keyword relationships."""
    if not keywords:
        return
    # Use the top 10 keywords for the map to avoid clutter
    top_keywords = keywords[:10]
    # Assign positions around a circle
    positions = {}
    n = len(top_keywords)
    for i, kw in enumerate(top_keywords):
        angle = 2 * np.pi * i / n
        positions[kw] = (np.cos(angle), np.sin(angle))
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6))
    # Draw nodes
    for kw, (x, y) in positions.items():
        ax.scatter(x, y, s=800, color="#8ecae6", zorder=2)
        ax.text(x, y, kw, fontsize=9, ha='center', va='center', zorder=3)
    # Draw edges
    for (kw1, kw2), weight in co_occurrence.items():
        if kw1 in positions and kw2 in positions:
            x1, y1 = positions[kw1]
            x2, y2 = positions[kw2]
            # Edge width scaled by weight
            lw = 0.5 + 0.5 * weight
            ax.plot([x1, x2], [y1, y2], color="#023047", linewidth=lw, zorder=1)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


#############################################
# Question Answering via Semantic Search
#############################################

def build_query_vector(
    query: str,
    vocabulary: List[str],
    idf: Dict[str, float]
) -> np.ndarray:
    """Convert a query into a TF–IDF vector using the provided vocabulary and IDF values."""
    words = [w for w in tokenize_words(query) if w]
    if not words:
        return np.zeros(len(vocabulary))
    counts = Counter(words)
    vec = np.zeros(len(vocabulary))
    for w, count in counts.items():
        if w in vocabulary:
            # term frequency normalised by query length
            vec[vocabulary.index(w)] = (count / len(words)) * idf.get(w, 0.0)
    return vec


def qa_interactive(
    sentences: List[str],
    processed_sentences: List[List[str]],
    tfidf_matrix: np.ndarray,
    vocabulary: List[str],
    idf_map: Dict[str, float]
) -> None:
    """Run an interactive question‑answering session on the processed document."""
    print("\nEnter your question (type 'exit' to quit):")
    while True:
        try:
            query = input("> ").strip()
        except EOFError:
            break
        if query.lower() in {"exit", "quit"}:
            break
        q_vec = build_query_vector(query, vocabulary, idf_map)
        # Compute cosine similarity to each sentence vector
        sims = []
        for i in range(len(sentences)):
            sims.append(sentence_similarity(q_vec, tfidf_matrix[i]))
        # Retrieve top 3 sentences
        top_indices = np.argsort(sims)[::-1][:3]
        answer = " ".join([sentences[i] for i in top_indices if sims[i] > 0])
        if not answer:
            print("I couldn't find an answer in the document.")
        else:
            print(answer)


#############################################
# Main execution
#############################################

def main():
    parser = argparse.ArgumentParser(description="Research Assistant & Knowledge Map (RAKM)")
    parser.add_argument("--input", required=True, help="Path to input .txt or .pdf file")
    parser.add_argument("--summary", help="Path to write the summary text")
    parser.add_argument("--map", help="Path to save the concept map image")
    parser.add_argument("--sentences", type=int, default=5, help="Number of sentences in summary")
    parser.add_argument("--qa", action="store_true", help="Enter interactive question‑answering mode")
    args = parser.parse_args()

    # Load and preprocess text
    text = load_text(args.input)
    sentences = tokenize_sentences(text)
    if not sentences:
        print("No valid sentences found in the document.")
        return
    stopwords = get_stopwords()
    processed_sentences = preprocess_sentences(sentences, stopwords)

    # Extract keywords
    keyword_scores = rake_keywords(sentences, stopwords)
    keywords = [kw for kw, score in keyword_scores]

    # Summarize
    summary, sentence_scores = textrank_summary(
        sentences,
        processed_sentences,
        num_sentences=args.sentences,
    )

    # Write or print summary
    if args.summary:
        with open(args.summary, "w", encoding="utf-8") as f:
            f.write(summary)
        print(f"Summary written to {args.summary}")
    else:
        print("\nSummary:\n")
        print(summary)

    # Build concept map
    if args.map:
        co_occurrence = build_co_occurrence(keywords, processed_sentences)
        draw_concept_map(keywords, co_occurrence, args.map)
        print(f"Concept map saved to {args.map}")

    # Build TF–IDF matrix and IDF map for Q&A
    tfidf_matrix, vocabulary = build_tfidf_matrix(processed_sentences)
    # Build IDF map for vocabulary
    # This replicates the IDF used in build_tfidf_matrix
    N = len(processed_sentences)
    idf_map = {}
    df_counts = Counter()
    for words in processed_sentences:
        for w in set(words):
            df_counts[w] += 1
    for w in vocabulary:
        df = df_counts.get(w, 0)
        idf_map[w] = log((N) / (1 + df))

    # Interactive QA
    if args.qa:
        qa_interactive(sentences, processed_sentences, tfidf_matrix, vocabulary, idf_map)


if __name__ == "__main__":
    main()