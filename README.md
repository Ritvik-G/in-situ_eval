# In‑Situ Evaluation Pipeline

A Streamlit-powered pipeline to run **real-time evaluations** on datasets you upload and compare results against **custom baselines**. It supports multiple retrieval strategies (RAG, RAPTOR, GraphRAG) and a flexible evaluation suite. The UI lives in `interface.py`, which orchestrates the end‑to‑end flow via the core pipeline in `main.py`.

> **Entry point:** `interface.py` (run with Streamlit).  
> **Python:** 3.10+ (uses `|` type unions).

Code Link (GitHub): https://github.com/Ritvik-G/insitu-eval

---

## Features

- Upload JSON datasets and run the pipeline end‑to‑end from a web UI
- Subsample/curate datasets before inference
- Retrieval techniques: **RAG**, **RAPTOR**, **GraphRAG**
- Pluggable LLM call via a simple config (works with GROQ and OpenAI compatible endpoints)
- Metrics for exact match, overlap, and ROUGE-style similarities (Generic and RAG centric)
- Clear artifacts saved for each run (sampled data, predictions, enriched + metrics, summaries)

---

## Project Structure

```
Code/
  interface.py        # Streamlit app — main entry point
  main.py             # Pipeline orchestrator and CLI (subsample → retrieval → evals)
  modules/
    subsampling.py    # Sampling, dedupe, date windows, etc.
    retrieval.py      # RAG / RAPTOR / GraphRAG + LLM request helpers
    evals.py          # Quantitative metrics of two kinds - Generic and RAG centric (EM, F1, Jaccard, ROUGE-ish)
artifacts/            # Created on first run; holds outputs
```

> Note: Some helper/test utilities exist internally for development; they are not required to use the app.

---

## Installation

1) **Create a virtual environment (recommended)**

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

2) **Install dependencies**

```bash
pip install -r requirements.txt
```

> If you plan to use GPU-accelerated PyTorch, install the appropriate `torch` wheels from https://pytorch.org/get-started/locally/. The default `pip install torch` will install a CPU build on many platforms.

---

## Quick Start (UI)

From the repo root **either**:

```bash
streamlit run Code/interface.py
```

or change into the `Code/` folder and run:

```bash
cd Code
streamlit run interface.py
```

This launches a local app at a URL like `http://localhost:8501`. Use the sidebar to choose **mode**, **Subsampling method**, **retrieval technique (if in retrieval mode)**, **Evaluation suite**, and parameters; upload your dataset and (optionally) provide model/access configs (see below).

---

## Dataset Format

Input JSON should be either a list of records or a mapping of dataset name → list of records. Typical QnA-style record keys (for Retrieval Mode):

```json
{
  "my_dataset": [
    {
      "Question": "…",
      "Context": "…",
      "Response": "…"
    }
  ]
}
```

Typical QnA-style record keys (for Dataset Mode):

```json
{
  "my_dataset": [
    {
      "Question": "…",
      "Context": "…",
      "Response": "…",
      "Predicted": "…"
    }
  ]
}
```

The pipeline normalizes keys (case-insensitive) and maps to the JSON definitions as above.

---

## Configuration

The pipeline (main.py) reads a single YAML/JSON config (default `pipeline.config.yaml`). 

**NOTE: For executions through Interface.py, these configs generation is automatically handled through the components on the interface and need not be generated.**


Minimal example:

```yaml
# pipeline.config.yaml
mode: retrieval                # "retrieval" to generate predictions; "dataset" to directly run evaluations
input_dataset: ./data/sample.json
artifacts_dir: ./artifacts

# Subsampling
subsampling:
  max_items_per_dataset: 100     # cap items per dataset (optional)
  dedupe: true                   # deduplicate by Question/Context (optional)

# Retrieval / Inference
retrieval:
  technique: rag                 # one of: rag | raptor | graphrag
  model_config: ./model_config.json   # LLM model settings (see below)
  access_config: ./config.json        # endpoint map for API calls
  chunk_size: 100                # doc chunk size (chars)
  top_k: 3                       # number of chunks/nodes to retrieve
  num_clusters: 5                # RAPTOR: number of clusters
  edge_threshold: 0.4            # GraphRAG: graph edge threshold (0.0–1.0)

# Evaluation
evals:
  metrics: [exact_match, f1, jaccard, rouge1, rougeL]  # choose any subset
  k_values: [1, 3, 5]           # for @k variants where applicable

# Outputs (filenames are joined under artifacts_dir)
sampled_out: sampled.json
predictions_out: predictions.json
enriched_out: enriched_with_metrics.json
metrics_summary_out: metrics_summary.json
sampling_audit_out: sampling_audit.json
```

### Access & Model Config

`retrieval.py` expects two small JSON files so it can call your chosen LLM endpoint.

**1) `config.json`** — endpoint routing (OpenAI-compatible style):

```json
{
  "ACCESS_CONFIG": {
    "chat_completions": "https://api.groq.com/openai/v1/chat/completions",
    "completions": "https://api.groq.com/openai/v1/completions"
  }
}
```

**2) `model_config.json`** — model and request parameters:

```json
{
  "api_key": "YOUR_KEY",
  "model": "llama3-8b-8192",
  "temperature": 0.2,
  "max_tokens": 512,
  "top_p": 0.95,
  "stream": false,
  "stop": null,
  "type": "chat_completions"   // selects which endpoint key to use from ACCESS_CONFIG
}
```

> **Embedding model:** Set `EMBEDDING_MODEL` env var to override the SentenceTransformer used for retrieval. Default is `"paraphrase-MiniLM-L6-v2"`.

---

## Running via CLI (optional)

Instead of the UI, you can run the whole pipeline from the command line (refer to config generation from above):

```bash
python Code/main.py -c pipeline.config.yaml
# or, from inside Code/
python main.py -c ../pipeline.config.yaml
```

Logs and outputs are written under `artifacts_dir`:

- `sampled.json` — subsampled/cleaned dataset
- `predictions.json` — model outputs (when `mode: api`)
- `enriched_with_metrics.json` — original rows enriched with metric scores
- `metrics_summary.json` — aggregated scores by dataset/metric
- `sampling_audit.json` — details on which items were sampled / deduped

---

## Troubleshooting

- **Module not found when launching Streamlit from the repo root**  
  Prefer `streamlit run Code/interface.py` or `cd Code && streamlit run interface.py` so Python resolves sibling imports.

- **Torch / Transformers install issues**  
  Ensure your Python version is 3.10+ and upgrade `pip`. For GPUs, follow the PyTorch install matrix for your CUDA/Metal.

- **HTTP/401 from LLM endpoint**  
  Confirm `config.json` endpoint keys exist and `model_config.json.api_key` is valid. Some providers also require `Authorization: Bearer <key>`; the code sends an OpenAI‑style request with headers.

---

## Contributing

Feel free to open issues or PRs. For local dev, add type checking, linting, and tests as desired (e.g., `ruff`, `pytest`, `mypy`).

