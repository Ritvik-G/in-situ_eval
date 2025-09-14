# main.py
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Helpers: config & I/O
# ----------------------------

# ----------------------------
# Data loading & normalization (fast, schema-flexible)
# ----------------------------

# ----------------------------
# Data loading & normalization
# ----------------------------

from pathlib import Path
import json

# Canonical keys used downstream
KEY_SYNONYMS = {
    # question
    "question": "Question", "query": "Question", "prompt": "Question",
    # context
    "context": "Context", "passage": "Context", "ctx": "Context",
    # gold answer
    "response": "Response", "answer": "Response", "gold": "Response",
    "ground_truth": "Response", "groundtruth": "Response",
    # predicted
    "predicted": "Predicted", "prediction": "Predicted", "output": "Predicted",
    # RAG fields
    "retrievedcontexts": "RetrievedContexts", "contexts": "RetrievedContexts",
    "retrieveddocids": "RetrievedDocIds", "retrieved_docs": "RetrievedDocIds",
    "golddocid": "GoldDocId", "gold_doc_id": "GoldDocId",
}

def _coerce_key(k: str) -> str:
    lk = (k or "").strip().lower().replace(" ", "").replace("-", "_")
    return KEY_SYNONYMS.get(lk, k)

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                yield json.loads(s)

def _normalize_shape(obj, default_name: str = "uploaded"):
    """Accept list[dict] or {name: list[dict]} (also supports {"data": [...]}) and return {name: list}."""
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        return {default_name: obj}
    if isinstance(obj, dict):
        if "data" in obj and isinstance(obj["data"], list):
            return {default_name: obj["data"]}
        if all(isinstance(v, list) for v in obj.values()):
            return obj
        # single row dict → wrap
        if obj and all(isinstance(v, (str, int, float, list, dict, type(None))) for v in obj.values()):
            return {default_name: [obj]}
    raise ValueError("Input dataset must be either a list[dict] or {dataset_name: list[dict]}.")

def _normalize_row(row: dict, *, max_context_chars: int | None, keep_only: list[str] | None) -> dict:
    out = {}
    for k, v in row.items():
        out[_coerce_key(k)] = v

    # Trim huge Context
    if isinstance(out.get("Context"), str) and max_context_chars and len(out["Context"]) > max_context_chars:
        out["Context"] = out["Context"][:max_context_chars]

    # Ensure text fields are strings
    for tkey in ("Question", "Response", "Predicted"):
        if tkey in out and out[tkey] is None: out[tkey] = ""
        if tkey in out and not isinstance(out[tkey], str): out[tkey] = str(out[tkey])

    # RetrievedContexts → list[str]
    rc = out.get("RetrievedContexts")
    if rc is not None:
        if isinstance(rc, str):
            out["RetrievedContexts"] = [rc]
        elif isinstance(rc, list):
            out["RetrievedContexts"] = [str(x or "") for x in rc]
        else:
            out["RetrievedContexts"] = [str(rc)]

    if keep_only:
        out = {k: out.get(k) for k in keep_only if k in out}
    return out

def load_and_normalize_dataset(
    path_or_obj: str | Path | dict | list,
    *,
    dataset_name: str = "uploaded",
    max_items_per_dataset: int | None = None,
    max_context_chars: int | None = 8000,
    dedupe: bool = True,
    keep_only: list[str] | None = None,
):
    """Load .json or .jsonl or in-memory object, normalize shape/keys, trim, dedupe, cap rows."""
    # 1) Load
    if isinstance(path_or_obj, (str, Path)):
        p = Path(path_or_obj)
        if not p.exists():
            raise FileNotFoundError(f"Dataset not found: {p}")
        obj = list(_iter_jsonl(p)) if p.suffix.lower() == ".jsonl" else json.loads(p.read_text(encoding="utf-8"))
    else:
        obj = path_or_obj

    # 2) Normalize top-level shape
    ds_map = _normalize_shape(obj, default_name=dataset_name)

    # 3) Normalize rows
    out = {}
    for ds, rows in ds_map.items():
        cleaned, seen = [], set()
        for row in rows:
            if not isinstance(row, dict): continue
            r = _normalize_row(row, max_context_chars=max_context_chars, keep_only=keep_only)

            # Require at least some content (Question | Response | Predicted)
            q = (r.get("Question") or "").strip()
            if not (q or (r.get("Response") or "").strip() or (r.get("Predicted") or "").strip()):
                continue

            # Deduplicate by (Question, Context head)
            if dedupe:
                ctx_head = (r.get("Context") or "")[:256].lower()
                fp = (q.lower(), ctx_head)
                if fp in seen: continue
                seen.add(fp)

            cleaned.append(r)
            if max_items_per_dataset and len(cleaned) >= int(max_items_per_dataset):
                break
        out[ds] = cleaned
    return out


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    # YAML if available, else JSON
    try:
        import yaml  # type: ignore
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

def save_json(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)

def dynamic_import(module_path_options: List[str]) -> Any:
    """
    Try multiple file paths until one imports successfully.
    Returns the loaded module object.
    """
    for p in module_path_options:
        path = Path(p)
        if not path.exists():
            continue
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec and spec.loader:
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            return mod
    raise FileNotFoundError(f"Could not import any of: {module_path_options}")

# ----------------------------
# Pipeline steps
# ----------------------------

def run_subsampling(
    subsampling_mod,
    data: Dict[str, List[dict]],
    method: str,
    params: Dict[str, Any]
) -> Tuple[Dict[str, List[dict]], Dict[str, Any]]:
    """
    Apply subsampling per dataset split (e.g., 'squad', 'trivia_qa').
    subsampling.run_sampler can take a list[dict] directly — we pass rows per dataset.
    """
    sampled: Dict[str, List[dict]] = {}
    audit_all: Dict[str, Any] = {}
    for ds_name, rows in data.items():
        res = subsampling_mod.run_sampler(method, rows, params)
        sampled[ds_name] = res["items"]
        audit_all[ds_name] = res.get("audit", {})
        logging.info(f"[subsampling] {ds_name}: kept {len(sampled[ds_name])} rows")
    return sampled, audit_all

def run_retrieval(
    retrieval_mod,
    data: Dict[str, List[dict]],
    technique: str,
    model_config_path: str | Path,
    access_config_path: Optional[str | Path] = None,
    chunk_size: Optional[int] = None,
    top_k: Optional[int] = None,
    num_clusters: Optional[int] = None,
    edge_threshold: Optional[float] = None,
) -> Dict[str, List[dict]]:
    """
    Calls the selected strategy's .run(...) exactly as defined in your retrieval file.
    """
    STRATEGY_REGISTRY = retrieval_mod.STRATEGY_REGISTRY
    if technique not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown technique '{technique}'. Options: {list(STRATEGY_REGISTRY)}")

    Strategy = STRATEGY_REGISTRY[technique]
    model_cfg = load_json(model_config_path)
    access_path = Path(access_config_path) if access_config_path else Path("config.json")

    logging.info(f"[retrieval] technique={technique}")

    if technique == "rag":
        out = Strategy.run(model_cfg, data, access_config_path=access_path)
    elif technique == "raptor":
        out = Strategy.run(
            model_cfg, data, access_config_path=access_path,
            chunk_size=int(chunk_size or 100),
            num_clusters=int(num_clusters or 5),
            top_k=int(top_k or 3),
        )
    elif technique == "graphrag":
        out = Strategy.run(
            model_cfg, data, access_config_path=access_path,
            chunk_size=int(chunk_size or 100),
            top_k=int(top_k or 3),
            edge_threshold=float(edge_threshold or 0.5),
        )
    else:
        raise ValueError(f"Unhandled technique: {technique}")

    # Sanity: ensure Predicted exists (the strategies set it)
    for ds_name, rows in out.items():
        missing = sum(1 for r in rows if "Predicted" not in r)
        if missing:
            logging.warning(f"[retrieval] {ds_name}: {missing} rows missing 'Predicted'")
    return out

def aggregate_metrics(enriched: Dict[str, List[dict]]) -> Dict[str, Dict[str, float]]:
    """
    Average metrics across entries per dataset. (Non-existent metrics are ignored.)
    """
    agg: Dict[str, Dict[str, float]] = {}
    for ds_name, rows in enriched.items():
        totals: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for r in rows:
            m = r.get("metrics", {})
            for k, v in m.items():
                try:
                    v = float(v)
                except Exception:
                    continue
                totals[k] = totals.get(k, 0.0) + v
                counts[k] = counts.get(k, 0) + 1
        agg[ds_name] = {k: (totals[k] / counts[k]) for k in totals if counts[k] > 0}
    return agg

def run_evaluations(
    evals_mod,
    data_with_predictions: Dict[str, List[dict]],
    metrics: List[str],
    k_values: List[int],
) -> Tuple[Dict[str, List[dict]], Dict[str, Dict[str, float]]]:
    """
    evals.evals mutates/returns the data by attaching 'metrics' per entry.
    We also compute per-dataset averages for convenience.
    """
    logging.info(f"[evals] metrics={metrics or 'default groups'} k={k_values or [1,3,5,10]}")
    enriched = evals_mod.evals(data_with_predictions, metrics=metrics, k_values=k_values or [1,3,5,10])
    summary = aggregate_metrics(enriched)
    return enriched, summary

# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Pipeline: subsampling → RAG/RAPTOR/GraphRAG → evals")
    ap.add_argument("-c", "--config", default="pipeline.config.yaml", help="YAML/JSON config for the pipeline")
    ap.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, WARNING, ERROR)")
    return ap.parse_args()

# --- keep your existing imports & helpers above ---

def run_pipeline(config_path: str | Path, log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    cfg = load_config(config_path)

    # ----- Mode: "dataset" (no API calls) or "api" (run retrieval) -----
    mode = str(cfg.get("mode", "api")).lower()  # "api" | "dataset"

    # ----- Paths / outputs -----
    input_path = Path(cfg.get("input_dataset"))
    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")

    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    sampled_out_path   = artifacts_dir / cfg.get("sampled_out", "sampled.json")
    predictions_path   = artifacts_dir / cfg.get("predictions_out", "predictions.json")
    enriched_out_path  = artifacts_dir / cfg.get("enriched_out", "enriched_with_metrics.json")
    metrics_summary    = artifacts_dir / cfg.get("metrics_summary_out", "metrics_summary.json")
    sampling_audit_out = artifacts_dir / cfg.get("sampling_audit_out", "sampling_audit.json")

    # ----- Import only what we need for the chosen mode -----
    subsampling_mod = dynamic_import([cfg.get("subsampling_file", "subsampling.py")])
    evals_mod       = dynamic_import([cfg.get("evals_file", "evals_qna_rag.py"), "evals.py"])
    retrieval_mod   = None
    if mode != "dataset":
        retrieval_mod = dynamic_import([cfg.get("retrieval_file", "retrieval.py"), "unified_rag.py"])

    # ----- Load data (accept list[dict] or {name: list[dict]}) -----
    # ---------------- load dataset (robust) ----------------
    data = load_and_normalize_dataset(
        input_path,
        dataset_name=cfg.get("dataset_name", "uploaded"),
        max_items_per_dataset=cfg.get("max_items_per_dataset"),   # e.g., 1000
        max_context_chars=cfg.get("max_context_chars", 8000),     # trims giant contexts
        dedupe=cfg.get("dedupe", True),                            # drop near-duplicates
        keep_only=cfg.get("keep_only_keys"),                       # e.g. ["Question","Context","Response","Predicted"]
    )


    for k, v in data.items():
        if not isinstance(v, list) or (v and not isinstance(v[0], dict)):
            raise ValueError(f"Dataset '{k}' must be a list of objects (rows).")

    # ----- 1) Subsampling -----
    subs_cfg = cfg.get("subsampling", {})
    method = subs_cfg.get("method", "uniform_random")
    params = {k: v for k, v in subs_cfg.items() if k != "method"}

    if method in (None, "no_sampling"):
        sampled = data
        audit_all = {ds: {"note": "no_sampling"} for ds in data}
        logging.info("[subsampling] skipped (no_sampling)")
    else:
        sampled, audit_all = run_subsampling(subsampling_mod, data, method=method, params=params)
        save_json(sampling_audit_out, audit_all)
    save_json(sampled_out_path, sampled)

    # ----- 2) Retrieval (skip completely in Dataset Mode) -----
    if mode == "dataset":
        logging.info("[retrieval] skipped (mode=dataset)")
        preds = sampled  # use provided Predicted fields if present
    else:
        rag_cfg = cfg.get("retrieval", {})
        technique = rag_cfg.get("technique", "rag")
        model_config_path = rag_cfg.get("model_config")
        if not model_config_path:
            raise ValueError("retrieval.model_config is required in API mode (path to model config JSON).")
        access_config_path = rag_cfg.get("access_config", "config.json")

        preds = run_retrieval(
            retrieval_mod,
            sampled,
            technique=technique,
            model_config_path=model_config_path,
            access_config_path=access_config_path,
            chunk_size=rag_cfg.get("chunk_size"),
            top_k=rag_cfg.get("top_k"),
            num_clusters=rag_cfg.get("num_clusters"),
            edge_threshold=rag_cfg.get("edge_threshold"),
        )
    save_json(predictions_path, preds)

    # ----- 3) Evaluations -----
    eval_cfg = cfg.get("evals", {})
    metrics: List[str] = eval_cfg.get("metrics", [])
    k_values: List[int] = eval_cfg.get("k_values", [])

    if mode == "dataset":
        missing = sum(1 for _, rows in preds.items() for r in rows if "Predicted" not in r)
        if missing:
            logging.warning(f"[evals] {missing} rows have no 'Predicted' — prediction-based metrics may be zero/omitted.")

    enriched, summary = run_evaluations(evals_mod, preds, metrics=metrics, k_values=k_values)
    save_json(enriched_out_path, enriched)
    save_json(metrics_summary, summary)

    logging.info("Pipeline complete.")
    return {
        "artifacts_dir": str(artifacts_dir),
        "sampled": str(sampled_out_path),
        "predictions": str(predictions_path),
        "enriched": str(enriched_out_path),
        "metrics_summary": str(metrics_summary),
        "sampling_audit": str(sampling_audit_out),
    }

def main():
    args = parse_args()
    run_pipeline(args.config, args.log_level)

if __name__ == "__main__":
    main()
