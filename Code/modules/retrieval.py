#!/usr/bin/env python3
"""
unified_rag.py

Single-file CLI to run one of: RAG | RAPTOR | GraphRAG
on a JSON dataset of the form:
{
  "squad": [
    {"Question": "...", "Context": "...", "Response": "..."},
    ...
  ],
  "trivia_qa": [ ... ]
}

Usage:
  python unified_rag.py --technique rag --input data.json --model-config model_config.json
  python unified_rag.py --technique raptor --input data.json --output out.json
  python unified_rag.py --technique graphrag --input data.json --chunk-size 120 --top-k 4

Model config JSON must include (example):
{
  "api_key": "YOUR_KEY",
  "model": "llama3-8b-8192",
  "temperature": 0.2,
  "max_tokens": 512,
  "top_p": 0.95,
  "stream": false,
  "stop": null,
  "type": "chat_completions"   // used to select endpoint from ACCESS_CONFIG in access-config file
}

The ACCESS_CONFIG mapping is read from --access-config (default: ./config.json).
Expected format:
{
  "ACCESS_CONFIG": {
    "chat_completions": "https://api.groq.com/openai/v1/chat/completions",
    "completions": "https://api.groq.com/openai/v1/completions"
  }
}
"""

from __future__ import annotations
import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests
import numpy as np
import networkx as nx

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

# ---- Embedding model (shared) ------------------------------------------------

_EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "paraphrase-MiniLM-L6-v2")
_SENTENCE_MODEL = None  # lazy init


def get_sentence_model():
    """Lazily initialize and cache the SentenceTransformer model."""
    global _SENTENCE_MODEL
    if _SENTENCE_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _SENTENCE_MODEL = SentenceTransformer(_EMBEDDING_MODEL_NAME)
    return _SENTENCE_MODEL


# ---- LLM call (from your Model_Config.py, cleaned up) ------------------------

def generate_response(model_config: Dict[str, Any], prompt: str, access_config_path: Path | None = None) -> str:
    """
    Generates a response using the Groq-style chat API based on provided model_config and prompt.

    model_config must include: api_key, model, temperature, max_tokens, top_p, stream, stop, type
    access_config_path should point to a JSON with ACCESS_CONFIG mapping (defaults to ./config.json).
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {model_config['api_key']}",
    }

    data = {
        "model": model_config["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": float(model_config["temperature"]),
        "max_tokens": int(model_config["max_tokens"]),
        "top_p": float(model_config["top_p"]),
        "stream": bool(model_config["stream"]),
        "stop": model_config.get("stop"),
    }

    # Load ACCESS_CONFIG map
    access_config_path = access_config_path or Path("config.json")
    try:
        with access_config_path.open("r", encoding="utf-8") as f:
            access_configurator = json.load(f)
        access_config = access_configurator["ACCESS_CONFIG"]
    except Exception as e:
        raise RuntimeError(f"Failed to read ACCESS_CONFIG from {access_config_path}: {e}") from e

    endpoint_key = model_config["type"]
    if endpoint_key not in access_config:
        raise ValueError(f"Model config 'type'='{endpoint_key}' not found in ACCESS_CONFIG keys: {list(access_config.keys())}")

    try:
        resp = requests.post(access_config[endpoint_key], headers=headers, json=data, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        return payload["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"LLM request failed: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Unexpected LLM response format: {e}") from e


# ---- Shared utilities --------------------------------------------------------

def chunk_documents(documents: List[str], chunk_size: int = 100) -> List[str]:
    """Split documents into smaller character-level chunks (kept same as your logic)."""
    chunks: List[str] = []
    for doc in documents:
        for i in range(0, len(doc), chunk_size):
            chunks.append(doc[i : i + chunk_size])
    return chunks


def generate_embeddings(texts: List[str]) -> np.ndarray:
    """Generate embeddings for each text using a shared SentenceTransformer model."""
    model = get_sentence_model()
    return np.asarray(model.encode(texts))


# ---- Strategy: RAG (baseline context stuffing) -------------------------------

class RAGTechnique:
    """Plain RAG that sends the original context with the question to the LLM."""

    @staticmethod
    def run(model_config: Dict[str, Any], data: Dict[str, List[Dict[str, Any]]], access_config_path: Path) -> Dict[str, Any]:
        for dataset_key in data:
            for entry in data[dataset_key]:
                question = entry["Question"]
                context = entry["Context"]
                prompt = f"User Question: {question}\n\nRelevant Excerpt(s):\n\n{context}"
                predicted = generate_response(model_config, prompt, access_config_path)
                entry["Predicted"] = predicted
        return data


# ---- Strategy: RAPTOR --------------------------------------------------------

class RAPTORTechnique:
    """
    RAPTOR-like flow as in your code:
    - chunk -> embed -> KMeans -> concat summaries -> 'tree' -> retrieve top-k summaries -> LLM
    The logic is preserved, with minor guards.
    """

    @staticmethod
    def _cluster_chunks(embeddings: np.ndarray, num_clusters: int = 5) -> np.ndarray:
        num_samples = len(embeddings)
        k = max(1, min(num_clusters, num_samples))
        km = KMeans(n_clusters=k, n_init="auto", random_state=42)
        km.fit(embeddings)
        return km.labels_

    @staticmethod
    def _summarize_chunks(chunks: List[str], labels: np.ndarray) -> Dict[int, str]:
        summaries: Dict[int, str] = {}
        for label in set(labels.tolist()):
            cluster_texts = [chunks[i] for i in range(len(chunks)) if labels[i] == label]
            summaries[label] = " ".join(cluster_texts)
        return summaries

    @staticmethod
    def _build_tree(summaries: Dict[int, str]) -> Dict[int, Dict[str, str]]:
        # Simple dict tree as per original logic
        return {i: {"summary": summary} for i, summary in summaries.items()}

    @staticmethod
    def _retrieve(tree: Dict[int, Dict[str, str]], query: str, top_k: int = 3) -> List[str]:
        model = get_sentence_model()
        query_embedding = model.encode([query])[0]
        sims: List[Tuple[int, float]] = []
        for node, data in tree.items():
            node_embedding = model.encode([data["summary"]])[0]
            sim = cosine_similarity([query_embedding], [node_embedding])[0][0]
            sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        return [tree[node]["summary"] for node, _ in sims[:top_k]]

    @classmethod
    def run(
        cls,
        model_config: Dict[str, Any],
        data: Dict[str, List[Dict[str, Any]]],
        access_config_path: Path,
        chunk_size: int = 100,
        num_clusters: int = 5,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        for dataset_key in data:
            for entry in data[dataset_key]:
                question = entry["Question"]
                context = entry["Context"]

                chunks = chunk_documents([context], chunk_size=chunk_size)
                if not chunks:
                    entry["Predicted"] = ""
                    continue

                embeddings = generate_embeddings(chunks)
                labels = cls._cluster_chunks(embeddings, num_clusters=num_clusters)
                summaries = cls._summarize_chunks(chunks, labels)
                tree = cls._build_tree(summaries)
                retrieved_info = cls._retrieve(tree, question, top_k=top_k)

                prompt = f"User Question: {question}\n\nRelevant Excerpt(s):\n\n{retrieved_info}"
                predicted = generate_response(model_config, prompt, access_config_path)
                entry["Predicted"] = predicted

        return data


# ---- Strategy: GraphRAG ------------------------------------------------------

class GraphRAGTechnique:
    """
    GraphRAG-like flow as in your code:
    - chunk -> embed -> build similarity graph (>0.5 edges) -> similarity to query -> pick top nodes
      -> refine by neighbor with strongest edge -> send refined contexts to LLM.
    Logic is kept intact, with small guards.
    """

    @staticmethod
    def _build_graph(chunks: List[str], embeddings: np.ndarray, edge_threshold: float = 0.5) -> nx.Graph:
        g = nx.Graph()
        for i in range(len(chunks)):
            g.add_node(i, text=chunks[i], embedding=embeddings[i])

        for i in range(len(chunks)):
            for j in range(i + 1, len(chunks)):
                sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
                if sim > edge_threshold:
                    g.add_edge(i, j, weight=sim)
        return g

    @staticmethod
    def _retrieve_from_graph(graph: nx.Graph, query_embedding: np.ndarray, top_k: int = 3) -> List[str]:
        sims: List[Tuple[int, float]] = []
        for node, data in graph.nodes(data=True):
            node_embedding = data["embedding"]
            sim = cosine_similarity([query_embedding], [node_embedding])[0][0]
            sims.append((node, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        top_nodes = [node for node, _ in sims[:top_k]]

        refined_contexts: List[str] = []
        for node in top_nodes:
            neighbors = list(graph.neighbors(node))
            if neighbors:
                edge_weights = [graph[node][nbr]["weight"] for nbr in neighbors]
                best_neighbor = neighbors[int(np.argmax(edge_weights))]
                refined_contexts.append(graph.nodes[best_neighbor]["text"])
            else:
                refined_contexts.append(graph.nodes[node]["text"])
        return refined_contexts

    @classmethod
    def run(
        cls,
        model_config: Dict[str, Any],
        data: Dict[str, List[Dict[str, Any]]],
        access_config_path: Path,
        chunk_size: int = 100,
        top_k: int = 3,
        edge_threshold: float = 0.5,
    ) -> Dict[str, Any]:
        model = get_sentence_model()
        for dataset_key in data:
            for entry in data[dataset_key]:
                question = entry["Question"]
                context = entry["Context"]

                chunks = chunk_documents([context], chunk_size=chunk_size)
                if not chunks:
                    entry["Predicted"] = ""
                    continue

                embeddings = generate_embeddings(chunks)
                graph = cls._build_graph(chunks, embeddings, edge_threshold=edge_threshold)

                query_embedding = model.encode([question])[0]
                retrieved_info = cls._retrieve_from_graph(graph, query_embedding, top_k=top_k)

                prompt = f"User Question: {question}\n\nRelevant Excerpt(s):\n\n{retrieved_info}"
                predicted = generate_response(model_config, prompt, access_config_path)
                entry["Predicted"] = predicted

        return data


# ---- CLI ---------------------------------------------------------------------

STRATEGY_REGISTRY = {
    "rag": RAGTechnique,
    "raptor": RAPTORTechnique,
    "graphrag": GraphRAGTechnique,
}


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj: Any) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run RAG / RAPTOR / GraphRAG over a dataset JSON.")
    p.add_argument("--technique", choices=STRATEGY_REGISTRY.keys(), required=True, help="Which strategy to run.")
    p.add_argument("--input", type=Path, required=True, help="Path to input dataset JSON.")
    p.add_argument("--output", type=Path, default=None, help="Where to write results JSON (default: <input>.out.json).")
    p.add_argument("--model-config", type=Path, required=True, help="Path to model config JSON (api_key, model, etc.).")
    p.add_argument("--access-config", type=Path, default=Path("config.json"), help="Path to ACCESS_CONFIG JSON (default: ./config.json).")

    # Common knobs (kept simple)
    p.add_argument("--chunk-size", type=int, default=100, help="Char-level chunk size for RAPTOR / GraphRAG.")
    p.add_argument("--top-k", type=int, default=3, help="Top-k retrieval for RAPTOR / GraphRAG.")
    p.add_argument("--num-clusters", type=int, default=5, help="RAPTOR: number of clusters (auto-capped by sample size).")
    p.add_argument("--edge-threshold", type=float, default=0.5, help="GraphRAG: edge similarity threshold.")
    p.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR).")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format="%(levelname)s: %(message)s")

    if args.technique not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown technique: {args.technique}")

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    if not args.model_config.exists():
        raise FileNotFoundError(f"Model config file not found: {args.model_config}")
    if not args.access_config.exists():
        raise FileNotFoundError(f"Access config file not found: {args.access_config}")

    data = load_json(args.input)
    model_config = load_json(args.model_config)

    Strategy = STRATEGY_REGISTRY[str(args.technique).lower()]

    logging.info(f"Running technique: {args.technique}")
    if args.technique == "rag":
        out = Strategy.run(model_config, data, access_config_path=args.access_config)
    elif args.technique == "raptor":
        out = Strategy.run(
            model_config,
            data,
            access_config_path=args.access_config,
            chunk_size=args.chunk_size,
            num_clusters=args.num_clusters,
            top_k=args.top_k,
        )
    elif args.technique == "graphrag":
        out = Strategy.run(
            model_config,
            data,
            access_config_path=args.access_config,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            edge_threshold=args.edge_threshold,
        )
    else:
        raise ValueError(f"Unhandled technique: {args.technique}")

    output_path = args.output or args.input.with_suffix(".out.json")
    save_json(output_path, out)
    logging.info(f"Wrote results to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
