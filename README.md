Research Assistant & Knowledge Map (RAKM)
Why I Built This

While studying for different classes, I kept running into the same problem: I could read long documents and notes, but it was hard to quickly see what the main ideas were and how everything connected. I wanted a tool that helped me learn faster, and I also wanted something meaningful to build for my college portfolio.

So I created RAKM — a small research assistant that summarizes documents, pulls out the important phrases, and turns them into a visual concept map. I built the core algorithms myself because I wanted to actually understand how these techniques work instead of relying on large black-box AI models.

This project represents the kind of learning I enjoy most: breaking things down, building them from scratch, and creating tools I can actually use.

Overview

RAKM is a lightweight Python tool that takes academic text (PDF or .txt) and produces:

A clean extractive summary

A set of keyword phrases

A generated concept map

An optional Q&A mode using semantic search

Unlike many modern NLP projects, RAKM does not rely on heavy machine-learning frameworks. The core pieces — keyword extraction, TextRank summarization, and TF–IDF retrieval — are implemented manually using only common Python libraries. The goal was to make something educational and transparent, while still being genuinely useful.

How It Works (Simple Explanation)

RAKM performs three main tasks:

1. Keyword Extraction (RAKE-style)

I implemented a version of the RAKE algorithm, which:

Splits text into candidate phrases

Removes stopwords

Scores phrases based on word frequency and co-occurrence

This highlights recurring multi-word ideas without requiring any training data.

2. Extractive Summarization (TextRank)

To summarize the document, each sentence is:

Converted into a vector

Compared to every other sentence

Represented as a graph

Ranked using a PageRank-style algorithm

The highest-ranked sentences become the summary. This preserves the original wording and works entirely without machine-learning models.

3. Concept Map Generation

After extracting phrases, RAKM looks at how often they appear together and builds a small graph:

Nodes = key phrases

Edges = how often phrases co-occur

Edge thickness = strength of relationship

The result is a visual map of how ideas relate to each other.

4. Optional Q&A (Semantic Search)

Using TF–IDF + cosine similarity, RAKM retrieves the most relevant sentences to any question the user asks. It’s a simple but surprisingly effective way to “query” a document.

Features

Works with PDF or plain text

Built-in stopword list

Configurable summary length

Auto-generated concept map image

Optional interactive Q&A mode

Completely offline (no external AI models)

Installation

This project requires Python 3 and a few standard libraries:

python -m pip install numpy scipy matplotlib


If you're using PDFs, install pdftotext (part of Poppler).

Place the project folder anywhere you like.

Usage
Basic use:
python3 main.py --input path/to/document.txt --summary summary.txt --map concept_map.png --sentences 5

With Q&A mode enabled:
python3 main.py --input samples/sample_article.txt --summary sample_summary.txt --map sample_concept_map.png --sentences 3 --qa

What I Learned

Building this project taught me a lot about:

How keyword extraction actually works

Why graph-based ranking (PageRank/TextRank) is effective

How TF–IDF encodes meaning numerically

How to build vector-similarity search from scratch

How to visualize idea connections in a document

How to structure a real-world Python project

Most importantly, I learned how to break down textbook algorithms and rebuild them myself into something I can use for school and studying.

Acknowledgments

RAKM’s design draws inspiration from several foundational NLP techniques:

RAKE for keyword extraction

TextRank for extractive summarization

TF–IDF and cosine similarity for document retrieval

Graph visualization concepts similar to tools like NetworkX

All implementations are written manually using basic Python tools for transparency and learning purposes.
