# test2.py
# Streamlit UI for selecting QnA/RAG metrics (multi-checkboxes + "Select All")
# Uses evals_qna_rag.py (place in same directory).
import json
import io
from typing import Dict, List, Any

import streamlit as st

try:
    import evals as ev
except ImportError:
    st.error("Could not import evals_qna_rag.py. Make sure it's in the same directory.")
    st.stop()

st.set_page_config(page_title="QnA/RAG Evaluator", layout="wide")
st.title("QnA/RAG Evaluator — Non-LLM, Selectable Metrics")

# ---- metric menus (pull from module if available; else fallback) ----
GENERIC_GROUP = getattr(ev, "GENERIC_GROUP", [
    "em","f1","rougeL",
    "bleu","rouge1","rouge2","rougeLsum","meteor",
    "nli_entail","nli_contra","nli_neutral","cosine",
    "jaccard","tfidf_cos","levenshtein","seqmatch","jsd",
])
RAG_GROUP = getattr(ev, "RAG_GROUP", [
    "support_gold","support_pred","coverage_precision","coverage_recall","coverage_f1",
    "hit@1","hit@3","hit@5","hit@10","mrr","ndcg@1","ndcg@3","ndcg@5","ndcg@10","first_support_rank",
])

ALL_METRICS = GENERIC_GROUP + RAG_GROUP

# map CLI metric names -> keys emitted in entry["metrics"]
def _metric_key_in_output(name: str) -> str:
    name = name.strip()
    mapping = {
        "em": "EM",
        "f1": "F1",
        "rougeL": "ROUGE-L",
        "bleu": "bleu",
        "rouge1": "rouge1",
        "rouge2": "rouge2",
        "rougeLsum": "rougeLsum",
        "meteor": "meteor",
        "nli_entail": "nli_entail",
        "nli_contra": "nli_contra",
        "nli_neutral": "nli_neutral",
        "cosine": "cosine",
        "jaccard": "jaccard",
        "tfidf_cos": "tfidf_cos",
        "levenshtein": "levenshtein",
        "seqmatch": "seqmatch",
        "jsd": "jsd",
        "support_gold": "SupportGoldInContexts",
        "support_pred": "SupportPredInContexts",
        "coverage_precision": "CoveragePrecision",
        "coverage_recall": "CoverageRecall",
        "coverage_f1": "CoverageF1",
        "mrr": "MRR",
        "first_support_rank": "first_support_rank",
    }
    # hit@K / ndcg@K pass through as-is
    if name.lower().startswith("hit@") or name.lower().startswith("ndcg@"):
        return name.lower()
    return mapping.get(name, name)

# ---- sidebar controls ----
with st.sidebar:
    st.header("Metric Selection")
    select_all = st.checkbox("Select/Deselect All", value=False, help="When checked, all metrics are evaluated.")
    st.caption("Tip: you can still tick individual boxes; the master toggle overrides selection.")

    st.subheader("Generic")
    generic_selected = []
    if not select_all:
        cols = st.columns(2)
        for i, m in enumerate(GENERIC_GROUP):
            with cols[i % 2]:
                if st.checkbox(m, value=(m in ["em","f1","rougeL"]), key=f"g_{m}"):
                    generic_selected.append(m)

    st.subheader("RAG")
    rag_selected = []
    if not select_all:
        cols = st.columns(2)
        for i, m in enumerate(RAG_GROUP):
            with cols[i % 2]:
                default_on = m in ["support_gold","support_pred","hit@5","mrr"]
                if st.checkbox(m, value=default_on, key=f"r_{m}"):
                    rag_selected.append(m)

    st.subheader("K values for hit/ndcg")
    default_ks = [1, 3, 5, 10]
    k_vals_txt = st.text_input("K values (comma-separated)", value="1,3,5,10")
    try:
        k_values = sorted({int(x) for x in k_vals_txt.replace(" ", "").split(",") if x})
    except Exception:
        st.warning("Invalid K list. Falling back to 1,3,5,10.")
        k_values = default_ks

# Effective metrics to run
if select_all:
    chosen_metrics = ALL_METRICS
else:
    chosen_metrics = generic_selected + rag_selected
    if not chosen_metrics:
        st.info("Pick at least one metric in the sidebar, or toggle Select All.")
        chosen_metrics = []

# ---- main panel ----
st.markdown("### 1) Upload your `data.json`")
uploaded = st.file_uploader("Upload a JSON file with shape: { dataset_name: [ {Question, Context, Response, Predicted, ...}, ... ] }", type=["json"])

def _read_upload(fobj) -> Dict[str, List[Dict[str, Any]]]:
    return json.load(io.TextIOWrapper(fobj, encoding="utf-8"))

run_clicked = st.button("Run Evaluation", type="primary", disabled=(uploaded is None or len(chosen_metrics) == 0))

if run_clicked:
    try:
        data = _read_upload(uploaded)
        # Run evaluator (non-LLM)
        enriched = ev.evals(data, metrics=chosen_metrics, k_values=k_values)

        # Aggregate simple averages per selected metric (across all datasets)
        st.markdown("### 2) Results")
        def _collect_all_rows(d: Dict[str, List[dict]]) -> List[dict]:
            rows = []
            for ds, arr in d.items():
                for e in arr:
                    e2 = dict(e)  # shallow
                    e2["_dataset"] = ds
                    rows.append(e2)
            return rows

        rows = _collect_all_rows(enriched)

        # Build summary table
        import pandas as pd
        def _avg(values: List[float]) -> float:
            return sum(values) / len(values) if values else float("nan")

        # Per-dataset summary
        per_ds_records = []
        for ds, entries in enriched.items():
            rec = {"dataset": ds, "n": len(entries)}
            for m in chosen_metrics:
                key = _metric_key_in_output(m)
                vals = [float(e.get("metrics", {}).get(key)) for e in entries if isinstance(e.get("metrics", {}).get(key), (int, float))]
                rec[m] = _avg(vals)
            per_ds_records.append(rec)
        df_ds = pd.DataFrame(per_ds_records).set_index("dataset").sort_index()

        # Overall summary
        overall = {"dataset": "__ALL__", "n": sum(len(v) for v in enriched.values())}
        for m in chosen_metrics:
            key = _metric_key_in_output(m)
            vals = [float(e.get("metrics", {}).get(key)) for e in rows if isinstance(e.get("metrics", {}).get(key), (int, float))]
            overall[m] = _avg(vals)
        df_overall = pd.DataFrame([overall]).set_index("dataset")

        st.subheader("Summary (averages)")
        st.dataframe(pd.concat([df_overall, df_ds], axis=0))

        # Small preview of per-item metrics (first dataset)
        first_ds = next(iter(enriched.keys()))
        st.subheader(f"Per-item Preview — {first_ds}")
        preview_cols = ["id","Question","Predicted","Response"]
        metric_cols = [ _metric_key_in_output(m) for m in chosen_metrics ]
        records = []
        for e in enriched[first_ds][:25]:
            row = {c: e.get(c) for c in preview_cols if c in e}
            for mk, mv in (e.get("metrics") or {}).items():
                if mk in metric_cols:
                    row[mk] = mv
            records.append(row)
        if records:
            st.dataframe(pd.DataFrame(records))
        else:
            st.info("No preview rows available.")

        # Download enriched JSON
        st.subheader("Download")
        enriched_str = json.dumps(enriched, indent=2, ensure_ascii=False)
        st.download_button("Download enriched JSON", data=enriched_str, file_name="results_enriched.json", mime="application/json")

    except Exception as e:
        st.error(f"Error while running evaluation: {e}")
        st.exception(e)

# Footer
with st.expander("Expected JSON shape & notes", expanded=False):
    st.markdown("""
**Input shape** (same as your boilerplate):

```json
{
  "my_dataset": [
    {
      "id": "q1",
      "Question": "Who founded Acme?",
      "Context": "Acme was founded by...",
      "Response": "Wile E. Coyote",
      "Predicted": "Wile E. Coyote",
      "RetrievedContexts": ["...", "..."],       // optional, ranked
      "RetrievedDocIds": ["D1","D7","D9"],       // optional
      "GoldDocId": "D1"                          // optional
    }
  ]
}
Pick metrics in the sidebar (or toggle Select/Deselect All).

If you include hit@K/ndcg@K, set K values (comma-separated).

Outputs: averages per dataset + an overall row, plus a small per-item preview.
""")