# Research Assistant & Knowledge Map (RAKM)

## Overview

RAKM is a lightweight research‑assistant tool written in Python that transforms a collection of academic documents into an accessible summary and a visual concept map.  It was designed as a portfolio‑quality project for college applications and demonstrates skills in natural language processing, algorithm implementation, information retrieval, and data visualization.  Unlike typical examples that rely on heavy machine‑learning libraries, RAKM implements key components such as keyword extraction and extractive summarization from scratch using only built‑in or commonly available Python libraries.

The application performs three core tasks:

1. **Keyword extraction** – RAKM implements the Rapid Automatic Keyword Extraction (RAKE) algorithm to identify important multi‑word phrases in the source text.  RAKE is a frequency‑based, unsupervised method that generates candidate phrases by grouping together non‑stopwords and then scores them by word frequency and degree (the number of co‑occurrences across phrases)【303286152743780†L35-L53】.  This approach allows RAKM to highlight the main topics of a document without any external training data.

2. **Extractive summarization** – To condense long documents into a digestible synopsis, RAKM uses a custom implementation of the TextRank algorithm.  TextRank treats each sentence as a node in a graph and connects nodes based on their similarity.  It then applies the PageRank algorithm to identify the most central sentences.  The top‑ranked sentences form the summary.  Extractive methods like TextRank preserve the original wording and work without training data【810671135289267†L27-L48】.

3. **Concept map generation** – Once key phrases have been extracted, RAKM computes co‑occurrence relationships between them and builds a concept map.  Nodes represent keywords and edges represent co‑occurrence counts.  The map is rendered using `matplotlib`, positioning nodes on a circle and drawing edges whose thickness corresponds to the strength of the relationship.  Although RAKM does not use the NetworkX library directly, the concept and visualization draw on graph‑analysis ideas: NetworkX is a Python library that creates and analyzes graph structures and provides basic visualization tools【405925609694343†L27-L34】.

Optionally, RAKM can answer user questions about the input text.  It embeds sentences and queries into a high‑dimensional vector space and uses cosine similarity to retrieve the most relevant sentences.  This retrieval‑based approach is inspired by the idea behind semantic search: embedding both the corpus and the query into the same vector space and then finding the nearest neighbours【985493886905673†L546-L550】.

## Features

* **PDF and plain‑text support** – RAKM accepts either plain‑text files or PDF documents.  PDF files are converted to text using the system utility `pdftotext` (available on most Linux systems).  If `pdftotext` is missing, the user can provide a `.txt` file instead.
* **Stopword handling** – A built‑in list of English stopwords (based on NLTK’s English stopword list) is bundled with the project, so the program works offline.
* **Configurable summarization** – Users can specify how many sentences should appear in the summary.  The TextRank implementation is generic and can accommodate different damping factors or convergence criteria if desired.
* **Concept map image output** – The generated concept map is saved as `concept_map.png` in the output directory.  Nodes are labelled and edges are scaled by co‑occurrence frequency.
* **Question answering** – An optional interactive mode allows users to ask questions about the document.  The tool performs a simple semantic search using TF–IDF similarity to return the most relevant sentences.

## Installation

RAKM is self‑contained and relies only on Python 3.11 and a few standard libraries (`numpy`, `scipy`, `matplotlib`).  It does not require internet access or external data downloads.

1. Ensure Python 3 is installed on your system.
2. If you plan to process PDF documents, make sure the `pdftotext` utility is available (it is included in Poppler; install via your package manager on Linux, e.g., `sudo apt‑get install poppler-utils`).
3. Clone or download the `research_assistant_project` directory.

## Usage

Run the tool from the command line:

```bash
python main.py --input path/to/document.pdf --summary summary.txt --map concept_map.png --sentences 5
```

Arguments:

* `--input` – Path to the input file.  Accepts `.pdf` or `.txt` files.
* `--summary` – Path to the output summary file.  If omitted, the summary is printed to stdout.
* `--map` – Path to save the concept map image.  If omitted, no image is generated.
* `--sentences` – Number of sentences to include in the summary (default 5).
* `--qa` – Activate interactive question‑answering mode after summarization.

Example:

```bash
python main.py --input samples/sample_article.txt --summary sample_summary.txt --map sample_concept_map.png --sentences 3 --qa
```

This command processes a sample article, writes a three‑sentence summary to `sample_summary.txt`, generates a concept map image called `sample_concept_map.png`, and then starts an interactive Q&A session on the processed document.

## Citation and Acknowledgements

RAKM’s keyword extraction is based on the Rapid Automatic Keyword Extraction (RAKE) algorithm, which generates key phrases by grouping contiguous non‑stopwords and scores them using word frequency and co‑occurrence【303286152743780†L35-L53】.  The summarization component implements the TextRank algorithm, an unsupervised extractive summarization technique that builds a graph of sentences and ranks them using PageRank【810671135289267†L27-L48】.  The idea of representing text for question answering via vector embeddings and retrieving similar sentences draws inspiration from the semantic search approach described in the Sentence Transformers documentation【985493886905673†L546-L550】.  The concept of graph visualization and analysis is motivated by the NetworkX library’s use for creating and analysing graph structures【405925609694343†L27-L34】.

This project was developed as part of a college‑application portfolio to showcase a combination of algorithm design, natural language processing, and data‑visualization skills.