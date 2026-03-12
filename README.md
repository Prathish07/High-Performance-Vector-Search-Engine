# High-Performance Semantic Vector Search Engine

A **high-performance semantic search engine** implemented in **C++ and Python** that retrieves text using **vector similarity instead of keyword matching**.

The system generates embeddings using **SentenceTransformers**, builds an **Approximate Nearest Neighbor (ANN) graph index**, and performs **millisecond-level semantic retrieval**.

---

## Project Overview

Modern AI systems such as **RAG pipelines, recommendation systems, and vector databases** rely on fast similarity search over high-dimensional embeddings.

This project implements a **mini vector database engine** capable of:

* Generating sentence embeddings using transformer models
* Storing and indexing high-dimensional vectors
* Performing semantic search using cosine similarity
* Implementing ANN graph traversal for fast retrieval
* Evaluating performance using **latency and Recall@K**

---

## System Architecture

Text Dataset
↓
SentenceTransformer Embeddings
↓
Vector Storage (C++)
↓
ANN Graph Index
↓
efSearch Candidate Traversal
↓
Top-K Semantic Retrieval

---

## Features

* Sentence embedding generation using **SentenceTransformers**
* Custom **C++ vector indexing engine**
* Cosine similarity based semantic search
* **Approximate Nearest Neighbor (ANN)** graph indexing
* efSearch candidate exploration
* Performance benchmarking:

  * Search latency
  * Recall@K

---

## Tech Stack

* **C++**
* **Python**
* SentenceTransformers
* NumPy
* Approximate Nearest Neighbor Algorithms

---

## Example Output

Loaded vectors: 50000
Graph built with 5 neighbors per node

Search latency: 1.5 ms
Recall@5: 0.92

Top results:

US, EU Talk Aircraft Subsidies...
Clinton Breathing on Own After Surgery...
Insurers Object to New Provision in Medicare Law...

---

## Running the Project

### 1️⃣ Generate embeddings

Run the notebook:

```
embeddings/generate_embeddings.ipynb
```

This creates:

```
embeddings.txt
sentences.txt
```

---

### 2️⃣ Compile C++ Engine

```
g++ main.cpp vector_index.cpp -o search_engine
```

---

### 3️⃣ Run the Engine

```
./search_engine
```

---

## Performance

| Dataset Size | Latency |
| ------------ | ------- |
| 10K vectors  | ~2 ms   |
| 50K vectors  | ~5 ms   |

---

## Applications

* Retrieval-Augmented Generation (RAG)
* Semantic document search
* Recommendation systems
* Vector databases
* AI knowledge retrieval

---

## Future Improvements

* Hierarchical Navigable Small World (HNSW) indexing
* GPU acceleration
* Parallel ANN search using OpenMP
* Disk-based vector storage
* Hybrid keyword + vector search

---

## Author

Prathish A
AI/ML Engineer
