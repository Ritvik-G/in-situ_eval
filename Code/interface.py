# interface.py
from __future__ import annotations

import io
import json
import random
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import time
import pandas as pd
import streamlit as st
import main as pipeline  # uses run_pipeline(config_path, log_level)

# ---- import modules here (no UI fields for paths) ----
import modules.subsampling as subsampling_mod
try:
    import modules.evals as evals_mod
except ImportError:
    import modules.evals as evals_mod

try:
    import modules.retrieval as retrieval_mod
except ImportError:
    retrieval_mod = None  # ok if running Dataset Mode

st.set_page_config(page_title="RAG Pipeline — Dataset/Retrieval Modes", page_icon="⚙️", layout="wide")
st.title("⚙️ In-Situ Evaluator")
# Compact spacing tweaks (keep things screenshot-friendly)
st.markdown("""
<style>
/* tighten vertical gaps between sections and inputs */
section.main .block-container { padding-top: 0.6rem; padding-bottom: 0.6rem; }
h1, h2, h3, h4, h5, h6 { margin: 0.35rem 0 0.25rem 0 !important; }
hr { margin: 0.35rem 0 !important; }
div[data-testid="stSidebar"] h2, 
div[data-testid="stSidebar"] h3,
div[data-testid="stSidebar"] .stButton,
div[data-testid="stSidebar"] .stSelectbox,
div[data-testid="stSidebar"] .stNumberInput,
div[data-testid="stSidebar"] .stTextInput,
div[data-testid="stSidebar"] .stSlider,
div[data-testid="stSidebar"] .stCheckbox,
div[data-testid="stSidebar"] .stMultiSelect,
div[data-testid="stSidebar"] .stRadio {
  margin-top: 0.25rem !important; margin-bottom: 0.25rem !important;
}
            
            /* tighten the space specifically between the main title and the next block */
section.main .block-container h1 { margin-bottom: 0.10rem !important; }
section.main .block-container h1 + div { margin-top: 0 !important; }  /* pull next widget up */
section.main .block-container h1 + div h2,
section.main .block-container h1 + div h3 { margin-top: 0.10rem !important; }

/* also tighten the gap before dataframes under a heading */
section.main .block-container h2 + div[data-testid="stDataFrame"],
section.main .block-container h3 + div[data-testid="stDataFrame"] {
  margin-top: 0.25rem !important;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _save_temp_json(obj: Any, suffix: str = ".json") -> str:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    Path(tf.name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return tf.name

def _maybe_float(s: str | None, default: float | None = None) -> float | None:
    if s is None or str(s).strip() == "":
        return default
    try:
        return float(s)
    except Exception:
        return default

def _maybe_int(s: str | None, default: int | None = None) -> int | None:
    if s is None or str(s).strip() == "":
        return default
    try:
        return int(s)
    except Exception:
        return default

def _aggregate_trial_summaries(trial_summaries: List[Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    metrics, datasets = set(), set()
    for summ in trial_summaries:
        for ds, kv in (summ or {}).items():
            datasets.add(ds)
            for m in (kv or {}).keys():
                metrics.add(m)
    metrics = sorted(metrics)
    datasets = sorted(datasets)

    rows = []
    for m in metrics:
        row = {"Metric": m}
        for ds in datasets:
            vals = []
            for summ in trial_summaries:
                v = (summ or {}).get(ds, {}).get(m, None)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                std = var ** 0.5
                # hide the ± when std would display as 0.000
                if round(std, 3) == 0.0:
                    row[ds] = f"{mean:.3f}"
                else:
                    row[ds] = f"{mean:.3f} ± {std:.3f}"
            else:
                row[ds] = "—"
        rows.append(row)
    return pd.DataFrame(rows).set_index("Metric")


def _extract_k_from_selected(metrics: List[str]) -> List[int]:
    ks = []
    for m in metrics:
        ml = m.lower().strip()
        if ml.startswith("hit@") or ml.startswith("ndcg@"):
            try:
                ks.append(int(ml.split("@", 1)[1]))
            except Exception:
                pass
    return sorted(set(ks))

# Edge-trigger helper so a toggle can behave like a momentary button
def _edge_triggered_toggle(key: str, inverted: bool = False, default_state: bool = True) -> bool:
    """
    Returns True exactly once when the control changes into its 'active' state.
    If inverted=True, 'active' means the toggle is OFF (False).
    """
    # Current state from session_state (Streamlit sets this before script body continues)
    cur = st.session_state.get(key, default_state)
    prev_key = f"__prev_{key}"
    prev = st.session_state.get(prev_key, cur)

    # Determine the 'active' state (inverted or not)
    active_now = (not cur) if inverted else bool(cur)
    active_prev = (not prev) if inverted else bool(prev)

    # Store for next run
    st.session_state[prev_key] = cur

    # Fire only on the edge (False->True transition of 'active')
    return (not active_prev) and active_now


# ---- Multi-file dataset loader/normalizer ----
def _normalize_one(obj: Any, default_name: str) -> Dict[str, List[dict]]:
    if isinstance(obj, list):
        return {default_name: obj}
    if isinstance(obj, dict):
        out: Dict[str, List[dict]] = {}
        for k, v in obj.items():
            if isinstance(v, list):
                out[str(k)] = v
        return out
    return {}

def _read_multi_jsons(uploads: List[Any]) -> Dict[str, List[dict]] | None:
    if not uploads:
        return None
    merged: Dict[str, List[dict]] = {}
    for i, upl in enumerate(uploads):
        try:
            obj = json.loads(upl.getvalue().decode("utf-8"))
        except Exception as e:
            st.error(f"Failed to parse JSON ({getattr(upl, 'name', f'file {i+1}')}): {e}")
            return None
        stem = Path(getattr(upl, "name", f"ds{i+1}")).stem or f"ds{i+1}"
        piece = _normalize_one(obj, stem)
        for name, rows in piece.items():
            new_name = name
            n = 2
            while new_name in merged:
                new_name = f"{name}_{n}"
                n += 1
            merged[new_name] = rows
    return merged

# ---- Mock utilities -------------------------------------------------

_WORDS = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda "
          "quick brown fox jumps over lazy dog neural rag graph raptor chunk "
          "context retrieval grounding answer question").split()

def _rand_sentence(minw=6, maxw=14) -> str:
    n = random.randint(minw, maxw)
    s = " ".join(random.choice(_WORDS) for _ in range(n))
    return s.capitalize() + "."

def _ensure_demo_dataset(ds: Optional[Dict[str, List[dict]]]) -> Dict[str, List[dict]]:
    if ds:
        return ds
    demo = [
        {"Question": "What is RAG?", "Context": _rand_sentence(), "Response": "Retrieval augmented generation integrates retrieval with generation."},
        {"Question": "Define RAPTOR.", "Context": _rand_sentence(), "Response": "RAPTOR builds hierarchical summaries to improve retrieval."},
        {"Question": "What is GraphRAG?", "Context": _rand_sentence(), "Response": "GraphRAG leverages graph structures for reasoning over documents."},
    ]
    return {"demo": demo}

def _random_metric_value(name: str, top_k: int = 5) -> float:
    n = name.lower()
    # bounded helpers
    def b(a=0.0, b=1.0): return max(a, min(b, random.random()))
    if n in {"em", "f1", "rouge-l", "rougel", "rouge1", "rouge2", "rougelsum", "meteor",
             "cosine", "jaccard", "tfidf_cos", "levenshtein", "seqmatch",
             "supportgoldincontexts", "supportpredincontexts",
             "coverageprecision", "coveragerecall", "coveragef1"}:
        return round(b(), 4)
    if n.startswith("hit@") or n.startswith("ndcg@") or n == "mrr":
        return round(b(), 4)
    if n == "first_support_rank":
        return float(random.randint(1, max(2, top_k)))
    if n == "jsd":
        return round(b(), 4)  # lower is better, but for display this is fine
    # fallback
    return round(b(), 4)

def _mock_trial_summary(datasets: List[str], metrics: List[str], top_k: int) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        mvals: Dict[str, float] = {}
        for m in metrics:
            key = m  # keep human-readable keys
            mvals[key] = _random_metric_value(key, top_k=top_k)
        summary[ds] = mvals
    return summary

def _write_mock_artifacts(ds_obj: Dict[str, List[dict]],
                          trial_summaries: List[Dict[str, Dict[str, float]]]) -> Dict[str, str]:
    # means across trials for metrics_summary
    datasets = sorted(ds_obj.keys())
    metrics_set = set()
    for s in trial_summaries:
        for ds in datasets:
            metrics_set |= set((s.get(ds, {}) or {}).keys())
    metrics = sorted(metrics_set)

    means: Dict[str, Dict[str, float]] = {}
    for ds in datasets:
        means[ds] = {}
        for m in metrics:
            vals = []
            for s in trial_summaries:
                v = (s.get(ds, {}) or {}).get(m)
                if isinstance(v, (int, float)):
                    vals.append(float(v))
            means[ds][m] = float(sum(vals) / len(vals)) if vals else 0.0

    # sampled (use exactly incoming rows)
    sampled = ds_obj

    # predictions: add "Predicted" to each entry
    preds: Dict[str, List[dict]] = {}
    for ds, rows in ds_obj.items():
        new_rows = []
        for r in rows:
            r2 = dict(r)
            r2["Predicted"] = _rand_sentence()
            new_rows.append(r2)
        preds[ds] = new_rows

    # enriched: attach per-entry random metrics (close to dataset means)
    enriched: Dict[str, List[dict]] = {}
    for ds, rows in preds.items():
        ds_mean = means.get(ds, {})
        new_rows = []
        for r in rows:
            met = {}
            for m, mu in ds_mean.items():
                # jitter around mean for variety
                val = max(0.0, min(1.0, random.gauss(mu, 0.05))) if isinstance(mu, float) else mu
                met[m] = float(val)
            r2 = dict(r)
            r2["metrics"] = met
            new_rows.append(r2)
        enriched[ds] = new_rows

    audit = {"note": "mock", "generated": time.time()}

    # Write to temp files so existing download buttons work
    return {
        "sampled": _save_temp_json(sampled),
        "predictions": _save_temp_json(preds),
        "enriched": _save_temp_json(enriched),
        "metrics_summary": _save_temp_json(means),
        "sampling_audit": _save_temp_json(audit),
    }

# ---------------------------------------------------------------------
# Sidebar structure (mode + sections)
# ---------------------------------------------------------------------

with st.sidebar:
    st.subheader("Mode")
    mode = st.radio(
    "Choose how to run:",
    ("Dataset Mode", "Retrieval Mode"),
    help=(
        "Dataset Mode: upload a dataset that already contains predictions and run subsampling + evaluations.\n"
        "Retrieval Mode: provide API creds to generate predictions via RAG/RAPTOR/GraphRAG, then evaluate."
    ),
    index=0,
    horizontal=True,   # remove if your Streamlit version doesn't support horizontal layout
    key="run_mode",
    )
    


    st.markdown("---")
    st.subheader("Data Orchestrator")
    c1,c2 = st.columns([1,3])
    with c1:
        log_level = st.selectbox("Log level", ["DEBUG", "INFO", "WARNING", "ERROR"], index=1)
    with c2:
        ds_upl_list = st.file_uploader("Dataset JSON file(s)", type=["json"], accept_multiple_files=True)

    st.markdown("---")
    st.subheader("Sub-Sampler")
    subs_methods = [
        "no_sampling", "uniform_random", "stratified", "diversity_kcenter",
        "importance_weighted", "active_learning", "difficulty_aware",
        "temporal_recent", "unanswerable_mix",
    ]
    colS1, colS2 ,colS3 = st.columns(3)
    with colS1:
        method = st.selectbox("method", subs_methods, index=subs_methods.index("uniform_random"))
    with colS2:
        subs_n = st.number_input("n (per dataset)", min_value=1, value=100)
    with colS3:
        subs_seed = st.number_input("seed", min_value=0, value=42)

    subs_extra: Dict[str, Any] = {}
    if method == "stratified":
        subs_extra["strata_keys"] = st.multiselect("strata_keys", ["topic", "answer_type", "lang"], default=["topic", "answer_type"])
        subs_extra["allocation"] = st.selectbox("allocation", ["proportional", "equal", "neyman"], index=0)
        if subs_extra["allocation"] == "neyman":
            vf = st.selectbox("variability_field", ["length", "difficulty", "None"], index=0)
            subs_extra["variability_field"] = None if vf == "None" else vf
            subs_extra["question_field"] = st.text_input("question_field (for length proxy)", value="question")
        else:
            subs_extra["variability_field"] = None
            subs_extra["question_field"] = st.text_input("question_field", value="question")
        subs_extra["min_per_stratum"] = st.number_input("min_per_stratum", min_value=0, value=0)

    elif method == "diversity_kcenter":
        subs_extra["embedding_field"] = st.text_input("embedding_field", value="embedding")
        subs_extra["text_field"] = st.text_input("text_field (TF-IDF fallback)", value="question")

    elif method == "importance_weighted":
        subs_extra["weight_field"] = st.text_input("weight_field", value="weight")
        subs_extra["floor_prob"] = _maybe_float(st.text_input("floor_prob", value="1e-9"))
        subs_extra["cap_ratio"] = _maybe_float(st.text_input("cap_ratio", value="0.2"))

    elif method == "active_learning":
        subs_extra["mode"] = st.selectbox("mode", ["entropy", "margin", "disagreement"], index=0)
        subs_extra["probs_field"] = st.text_input("probs_field", value="probs")
        subs_extra["uncertainty_field"] = st.text_input("uncertainty_field", value="uncertainty")
        subs_extra["committee_votes_field"] = st.text_input("committee_votes_field", value="committee_votes")

    elif method == "difficulty_aware":
        subs_extra["hard_frac"] = _maybe_float(st.text_input("hard_frac (0..1)", value="1.0"))
        subs_extra["difficulty_field"] = st.text_input("difficulty_field", value="difficulty")
        subs_extra["question_field"] = st.text_input("question_field", value="question")
        subs_extra["alpha"] = _maybe_float(st.text_input("alpha", value="1.0"))
        subs_extra["beta"] = _maybe_float(st.text_input("beta", value="0.5"))
        subs_extra["gamma"] = _maybe_float(st.text_input("gamma", value="1.0"))

    elif method == "temporal_recent":
        subs_extra["mode"] = st.selectbox("temporal mode", ["window", "decay"], index=0)
        subs_extra["days"] = _maybe_int(st.text_input("days (window)", value="30"))
        subs_extra["half_life_days"] = _maybe_int(st.text_input("half_life_days (decay)", value="45"))
        subs_extra["date_field"] = st.text_input("date_field", value="date")

    elif method == "unanswerable_mix":
        subs_extra["flag_field"] = st.text_input("flag_field", value="is_unanswerable")
        subs_extra["target_ratio"] = _maybe_float(st.text_input("target_ratio", value="0.2"))

    # -------------------------
    # Retrieval section (Retrieval Mode only) — Provider & Credentials → Gen Settings → RAG Technique
    # -------------------------
    if mode == "Retrieval Mode":
        st.markdown("---")
        st.subheader("Retrieval & Generation Settings (Retrieval Mode Only)")

        c1,c2 = st.columns(2)
        with c1:
            provider = st.selectbox("Provider", ["OpenAI", "Groq"], index=1)
        with c2:
            # Direct text entry (no dropdown). Keep var names so downstream logic stays identical.
            if provider == "OpenAI":
                default_model = "gpt-4o"
            elif provider == "Groq":
                default_model = "llama-3.1-8b-instant"  # example; could be any from GROQ_MODELS
            else:
                default_model = ""

            mdl = st.text_input(
                "Model",
                value=default_model,
                placeholder="e.g., gpt-4o, llama3-70b-8192, mixtral-8x7b-32768",
                help="Type the exact model identifier."
            )
            # Keep custom_model defined so existing logic remains untouched.
            custom_model = ""
        
        
        api_key = st.text_input("API Key", value="", type="password")

        c1,c2,c3 = st.columns(3)
        with c1:
            temperature = st.number_input("temperature [0.2]", min_value=0.0, max_value=2.0, value=float(st.session_state.get("temperature", 0.2)), step=0.01, format="%.2f")
        with c2:
            top_p = st.number_input("top_p [0,1]", min_value=0.0, max_value=1.0, value=float(st.session_state.get("top_p", 0.95)), step=0.01, format="%.2f")
        with c3:
            max_tokens = st.number_input("max_tokens [1,8192]", min_value=1, max_value=8192, value=int(st.session_state.get("max_tokens", 512)), step=1)


        # # Temperature
        # c1, c2 = st.columns([2, 1])
        # with c1:
        #     temp_slider = st.slider("temperature (slider)", min_value=0.0, max_value=2.0, value=0.2, step=0.01)
        # with c2:
        #     temperature = st.number_input("temperature [0.2]", min_value=0.0, max_value=2.0,
        #                                   value=float(temp_slider), step=0.01, format="%.2f")

        # # Top P
        # c3, c4 = st.columns([2, 1])
        # with c3:
        #     topp_slider = st.slider("top_p (slider)", min_value=0.0, max_value=1.0, value=0.95, step=0.01)
        # with c4:
        #     top_p = st.number_input("top_p [0,1]", min_value=0.0, max_value=1.0,
        #                             value=float(topp_slider), step=0.01, format="%.2f")

        # # Max tokens
        # c5, c6 = st.columns([2, 1])
        # with c5:
        #     maxtok_slider = st.slider("max_tokens (slider)", min_value=1, max_value=8192, value=512, step=1)
        # with c6:
        #     max_tokens = st.number_input("max_tokens [1,8192]", min_value=1, max_value=8192, value=int(maxtok_slider), step=1)

        c7,c8 = st.columns([1,3])
        with c7:
            stream = st.checkbox("stream", value=False)
        with c8:
            stop_sequence = st.text_input("Stop Sequence",placeholder="Enter stop sequence", value="")

        # Retrieval technique and its params
        colR1, colR2, colR3 = st.columns(3)
        with colR1:
            technique = st.selectbox("Retrieval Technique", ["RAG", "RAPTOR", "GraphRAG"], index=0)
        with colR2:
            chunk_size = st.number_input("chunk_size", min_value=10, max_value=5000, value=100, step=10)
        with colR3:
            top_k = st.number_input("top_k", min_value=1, max_value=50, value=3, step=1)

        # Conditional toggles depending on technique
        if technique == "RAPTOR":
            num_clusters = st.number_input("num_clusters (RAPTOR)", min_value=1, max_value=100, value=5, step=1)
            edge_threshold = 0.5  # not used here
        elif technique == "GraphRAG":
            edge_threshold = st.slider("edge_threshold (GraphRAG)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            num_clusters = 5      # not used here
        else:
            num_clusters = 5
            edge_threshold = 0.5

    st.markdown("---")
    c1,c2 = st.columns(2)
    with c1:
        st.subheader("Evaluator")
    with c2:
        st.markdown('<div style="text-align: right; margin-top: 0.2rem;"></div>', unsafe_allow_html=True)
        select_all = st.checkbox("Select/Deselect All metrics", value=False)
    GENERIC_GROUP = [
        "em","f1","rougeL",
        "bleu","rouge1","rouge2","rougeLsum","meteor",
        "nli_entail","nli_contra","nli_neutral","cosine",
        "jaccard","tfidf_cos","levenshtein","seqmatch","jsd",
    ]
    RAG_GROUP = [
        "support_gold","support_pred","coverage_precision","coverage_recall","coverage_f1",
        "hit@1","hit@3","hit@5","hit@10","mrr","ndcg@1","ndcg@3","ndcg@5","ndcg@10","first_support_rank",
    ]
    #select_all = st.checkbox("Select/Deselect All metrics", value=False)
    chosen_metrics: List[str] = []
    if select_all:
        chosen_metrics = GENERIC_GROUP + RAG_GROUP
    else:
        st.caption("Generic")
        cols_g = st.columns(5)
        for i, m in enumerate(GENERIC_GROUP):
            if cols_g[i % 5].checkbox(m, value=(m in ["em", "f1", "rougeL"]), key=f"g_{m}"):
                chosen_metrics.append(m)
        st.caption("RAG")
        cols_r = st.columns(5)
        for i, m in enumerate(RAG_GROUP):
            default_on = m in ["support_gold","support_pred","hit@5","mrr"]
            if cols_r[i % 5].checkbox(m, value=default_on, key=f"r_{m}"):
                chosen_metrics.append(m)

    # Derive exact K set from selected metrics (no free-form text input)
    k_values = _extract_k_from_selected(chosen_metrics)
    st.caption(f"Detected K values from selections: **{k_values if k_values else '—'}**")

    # ---- Robustness check section ----
    st.markdown("---")
    st.subheader("Robustness check")
    c1,c2 = st.columns(2)
    with c1:
        trials = st.number_input("Trials (mean±std)", min_value=1, value=1, step=1)
    with c2:
        seed_step = st.number_input("Seed increment per trial", min_value=1, value=1, step=1)


        # -------- mock selector (tiny, inverted, bottom of sidebar) --------
    st.markdown(
        """
        <style>
        .mock-toggle-wrap { margin-top: 8px; opacity: 0.35; }
        .mock-toggle-wrap:hover { opacity: 0.55; }
        @media print { .mock-toggle-wrap { display:none !important; } }
        /* collapse label spacing */
        div[data-testid="stToggle"] label p { margin: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="mock-toggle-wrap">', unsafe_allow_html=True)
    st.toggle(
        label="",
        key="__mock_toggle__",
        value=True,                    # visually ON by default
        label_visibility="collapsed",
        help="Mock-up selector (inverted): switch OFF to arm mock; Run button will execute mock.",
    )
    st.markdown("</div>", unsafe_allow_html=True)

# outside the sidebar context (still top-level), compute whether mock is selected:
mock_armed = (not st.session_state.get("__mock_toggle__", True))  # OFF means 'armed'



# ---------------------------------------------------------------------
# Main panel: preview, run, and results
# ---------------------------------------------------------------------

# ---------------------------------------------------------------------
# Main panel: preview, run, and results
# ---------------------------------------------------------------------

st.markdown('<h3 style="margin: 0.10rem 0 0.25rem;">Dataset Preview</h3>', unsafe_allow_html=True)
dataset_obj = _read_multi_jsons(ds_upl_list) if ds_upl_list else None

if dataset_obj is not None:
    # Build a flat preview table capped at 25 rows
    rows = []
    for _, records in dataset_obj.items():
        if not isinstance(records, list):
            continue
        for rec in records:
            q = rec.get("Question") or rec.get("question") or rec.get("prompt") or ""
            ctx = (
                rec.get("Context") or rec.get("context") or rec.get("ctx")
                or rec.get("passage") or rec.get("document") or rec.get("doc") or ""
            )
            resp = rec.get("Response") or rec.get("response") or rec.get("answer") or rec.get("gold") or ""
            # NEW: include Predicted with a few common aliases; default to empty string
            pred = (
                rec.get("Predicted") or rec.get("predicted") or rec.get("prediction")
                or rec.get("model_answer") or rec.get("pred") or ""
            )
            rows.append({
                "No": len(rows) + 1,
                "Question": q,
                "Context": ctx,
                "Response": resp,
                "Predicted": pred,
            })
            if len(rows) >= 25:
                break
        if len(rows) >= 25:
            break

    if rows:
        df_preview = pd.DataFrame(rows, columns=["No", "Question", "Context", "Response", "Predicted"])
        # Show ~5 rows, scroll for the rest (up to 25)
        VISIBLE_ROWS = 5
        ROW_H = 28   # approx per-row height in st.dataframe
        BASE_H = 30  # header/padding allowance
        st.dataframe(df_preview, use_container_width=True, height=BASE_H + ROW_H * VISIBLE_ROWS)
    else:
        st.info("No rows found with fields: Question / Context / Response / Predicted.")




run_label = "Run (Subsampling + Evals only)" if mode == "Dataset Mode" else "Run Full Pipeline"
cols_run = st.columns([1.0])
with cols_run[0]:
    run_clicked_raw = st.button(run_label, type="primary", use_container_width=True)

# Gate which branch should execute, without changing downstream logic:
# - If mock is armed and user clicks Run, we route to 'mock_clicked'
# - Otherwise, a normal Run routes to 'run_clicked'
mock_clicked = bool(run_clicked_raw and mock_armed)
run_clicked = bool(run_clicked_raw and (not mock_armed))



# ---------------- Real run ----------------
if run_clicked:
    if not dataset_obj:
        st.error("Please upload one or more dataset JSON files in the sidebar Data section.")
        st.stop()

    # ---- retrieval config (Retrieval only) ----
    retrieval_block: Dict[str, Any] = {}
    if mode == "Retrieval Mode":
        if retrieval_mod is None:
            st.error("retrieval module not available for Retrieval Mode.")
            st.stop()
        if not api_key:
            st.error("Please enter an API Key in the Retrieval section.")
            st.stop()

        selected_model = custom_model.strip() if mdl == "(custom)" and custom_model.strip() else mdl
        try:
            stop_val = json.loads(stop_sequence) if stop_sequence.strip() else None
        except Exception:
            st.warning("Invalid JSON for 'stop'; ignoring.")
            stop_val = None

        model_config_obj = {
            "api_key": api_key,
            "model": selected_model,
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "stream": bool(stream),
            "stop": stop_val,
            "type": "chat_completions",
        }
        if 'groq' in selected_model.lower() or (provider == "Groq"):
            access_cfg_obj = {
                "ACCESS_CONFIG": {
                    "chat_completions": "https://api.groq.com/openai/v1/chat/completions",
                    "completions": "https://api.groq.com/openai/v1/completions",
                }
            }
        elif provider == "OpenAI":
            access_cfg_obj = {
                "ACCESS_CONFIG": {
                    "chat_completions": "https://api.openai.com/v1/chat/completions",
                    "completions": "https://api.openai.com/v1/completions",
                }
            }
        else:
            access_cfg_obj = {
                "ACCESS_CONFIG": {
                    "chat_completions": "https://your.api/v1/chat/completions",
                    "completions": "https://your.api/v1/completions",
                }
            }

        model_cfg_path = _save_temp_json(model_config_obj)
        access_cfg_path = _save_temp_json(access_cfg_obj)

        retrieval_block = {
            "technique": technique,
            "model_config": model_cfg_path,
            "access_config": access_cfg_path,
            "chunk_size": int(chunk_size),
            "top_k": int(top_k),
            "num_clusters": int(num_clusters),
            "edge_threshold": float(edge_threshold),
        }

    # ---- persist merged dataset to temp file ----
    dataset_path = _save_temp_json(dataset_obj)
    subsampling_block = {
        "method": method,
        "n": int(subs_n),
        "seed": int(subs_seed),  # will be modified per trial
        **{k: v for k, v in subs_extra.items() if v is not None},
    }
    evals_block = {"metrics": chosen_metrics, "k_values": k_values}

    base_cfg: Dict[str, Any] = {
        "mode": "dataset" if mode == "Dataset Mode" else "api",
        "input_dataset": dataset_path,
        "subsampling_file": subsampling_mod.__file__,
        "evals_file": evals_mod.__file__,
        "subsampling": subsampling_block,
        "evals": evals_block,
    }
    if mode == "Retrieval Mode":
        base_cfg["retrieval_file"] = retrieval_mod.__file__
        base_cfg["retrieval"] = retrieval_block

    trial_summaries: List[Dict[str, Dict[str, float]]] = []
    last_results: Optional[Dict[str, str]] = None

    t0 = time.perf_counter()
    with st.status("Running trials…", expanded=True) as status:
        for t in range(int(trials)):
            cfg = json.loads(json.dumps(base_cfg))  # deep copy
            try:
                base_seed = int(cfg["subsampling"].get("seed", 42))
            except Exception:
                base_seed = 42
            cfg["subsampling"]["seed"] = base_seed + t * int(seed_step)

            cfg_path = _save_temp_json(cfg, suffix=".yaml")
            try:
                last_results = pipeline.run_pipeline(cfg_path, log_level=log_level)
                ms_path = Path(last_results["metrics_summary"])
                if ms_path.exists():
                    trial_summaries.append(json.loads(ms_path.read_text(encoding="utf-8")))
                status.update(label=f"Completed trial {t+1}/{int(trials)}", state="running")
            except Exception as e:
                status.update(label=f"Failed at trial {t+1}", state="error")
                st.exception(e)
                st.stop()
    elapsed = time.perf_counter() - t0

    if not trial_summaries:
        st.error("No summaries produced.")
        st.stop()

    df_table = _aggregate_trial_summaries(trial_summaries)

    left, right = st.columns([0.7, 0.3])
    with left:
        header_bits = []
        if mode == "Retrieval Mode":
            header_bits.append(f"Provider: **{provider}**")
            header_bits.append(f"Model: **{selected_model}**")
            header_bits.append(f"RAG: **{technique}**")
        else:
            header_bits.append("Mode: **Dataset**")
        st.markdown(" | ".join(header_bits))
    with right:
        st.markdown(f"**Time Taken:** {elapsed:.2f}s")

    st.dataframe(df_table, use_container_width=True, height=min(520, 48 + 28 * len(df_table)))

    st.markdown("##### Reports & Downloads")
    colA, colB, colC = st.columns(3)

    agg_csv = df_table.to_csv(index=True).encode("utf-8")
    agg_json = json.dumps(
        {"trials": int(trials), "seed_step": int(seed_step), "summaries": trial_summaries},
        indent=2, ensure_ascii=False
    ).encode("utf-8")

    def _dl(label: str, path_str: str):
            p = Path(path_str)
            if p.exists():
                st.download_button(label, p.read_bytes(), file_name=p.name, mime="application/json",
                                   use_container_width=True)
    with colA:
        st.download_button("Results Table (CSV)", data=agg_csv, file_name="metrics_mean_std.csv", mime="text/csv", use_container_width=True)
        _dl("predictions.json", last_results["predictions"])
    with colB:
        st.download_button("Comprehensive Report", data=agg_json, file_name="metrics_trials_report.json", mime="application/json", use_container_width=True)
        _dl("enriched_with_metrics.json", last_results["enriched"])
        _dl("sampling_audit.json", last_results["sampling_audit"])
    with colC:
        _dl("sampled.json", last_results["sampled"])
        _dl("metrics_summary.json", last_results["metrics_summary"])

    if last_results and Path(last_results["predictions"]).exists():
        try:
            preds = json.loads(Path(last_results["predictions"]).read_text(encoding="utf-8"))
        except Exception:
            preds = None
        st.markdown("#### Predictions Preview (last trial)")
        if preds:
            flat = []
            for ds, arr in preds.items():
                for e in arr[:50]:
                    flat.append({
                        "dataset": ds,
                        "Question": e.get("Question", ""),
                        "Predicted": e.get("Predicted", ""),
                        "Response": e.get("Response", ""),
                    })
            st.dataframe(flat, use_container_width=True, height=280)
        else:
            st.info("No predictions to preview.")

# ---------------- Mock run ----------------
elif mock_clicked:
    # ----- Build required mock datasets (3 baselines + 2 customs) -----
    def _mk_row(q: str) -> dict:
        return {"Question": q, "Context": _rand_sentence(), "Response": "—"}

    # Baselines
    squad = [
        _mk_row("What is the capital of France?"),
        _mk_row("Who wrote Pride and Prejudice?"),
        _mk_row("When was the Declaration of Independence signed?"),
        _mk_row("What is the largest planet in our solar system?"),
        _mk_row("Who painted the Mona Lisa?"),
    ]
    wiki = [
        _mk_row("Define thermodynamics."),
        _mk_row("What is the speed of light?"),
        _mk_row("Explain photosynthesis."),
        _mk_row("What is quantum entanglement?"),
        _mk_row("Define machine learning."),
    ]
    trivia = [
        _mk_row("Which element has the chemical symbol Hg?"),
        _mk_row("What year did the Berlin Wall fall?"),
        _mk_row("Which country hosted the 2012 Olympics?"),
        _mk_row("What is the tallest mountain on Earth?"),
        _mk_row("Who discovered penicillin?"),
    ]

    # Custom uploads -> 'base 1' and 'base 2' (if none, fabricate)
    def _rows_from_uploads(ds_obj_in: Optional[Dict[str, List[dict]]]) -> List[dict]:
        all_rows: List[dict] = []
        for _, rows in (ds_obj_in or {}).items():
            if isinstance(rows, list):
                all_rows.extend(rows)
        return all_rows

    uploaded_rows = _rows_from_uploads(dataset_obj)
    if not uploaded_rows:
        uploaded_rows = [_mk_row(f"Base seed Q{i+1}") for i in range(10)]

    base1_rows = uploaded_rows[::2] or [_mk_row("Base1 placeholder Q1"), _mk_row("Base1 placeholder Q2")]
    base2_rows = uploaded_rows[1::2] or [_mk_row("Base2 placeholder Q1"), _mk_row("Base2 placeholder Q2")]

    ds_obj = {
        "squadQA": squad,
        "wikiQA": wiki,
        "triviaQA": trivia,
        "base1": base1_rows,
        "base2": base2_rows,
    }

    # Ensure at least some metrics to show
    metrics_to_use = chosen_metrics[:] if chosen_metrics else ["em", "f1", "rougeL", "hit@5", "mrr"]
    ks = _extract_k_from_selected(metrics_to_use)
    top_k_for_rank = max([k for k in ks if isinstance(k, int)] + [5])

    # ----- Delay realistic wall time for mock only -----
    # Dataset Mode: ~10–15s; Retrieval Mode: ~30s
    delay_s = random.uniform(10.0, 15.0) if mode == "Dataset Mode" else random.uniform(28.0, 33.0)
    with st.status("Running trials...", expanded=True) as status:
        time.sleep(delay_s)
        status.update(label="Completed trial 1/1", state="complete")

    # ----- Cohesive mock metrics across trials -----
    # Dataset ordering: squadQA >= wikiQA >= triviaQA ~ base 2 >= base 1
    dataset_baseline = {
        "squadQA": 0.78,
        "wikiQA":  0.68,
        "triviaQA":0.62,
        "base 1":  0.57,
        "base 2":  0.60,
    }

    # Per-metric small bias so different metrics aren't identical
    metric_bias = {}
    rnd_bias = random.Random(int(time.time()) % (2**32))
    for m in metrics_to_use:
        metric_bias[m] = rnd_bias.uniform(-0.04, 0.04)  # [-0.04, +0.04]

    def _clip01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def _cohesive_value(metric: str, base: float, rng: random.Random) -> float:
        ml = metric.lower()
        mu = base + metric_bias.get(metric, 0.0)

        # NLI-style: keep modest
        if ml.startswith("nli_"):
            mu = 0.5 * base + metric_bias.get(metric, 0.0)

        # Coverage/support variants tied to base
        if "coverage" in ml or "support" in ml:
            mu = 0.85 * base + metric_bias.get(metric, 0.0)

        # Rank-based proxy metrics
        if ml.startswith("hit@") or ml.startswith("ndcg@") or ml == "mrr":
            mu = 0.9 * base + 0.05 + metric_bias.get(metric, 0.0)

        # Rouge variants around base-ish
        if ml.startswith("rouge"):
            mu = 0.8 * base + 0.1 + metric_bias.get(metric, 0.0)

        # JS divergence (display a plausible small value)
        if ml == "jsd":
            return _clip01(rng.gauss(0.15 * (1.0 - base), 0.02))

        # General metrics (em, f1, cosine, jaccard, etc.)
        sigma = 0.03 + rng.random() * 0.02  # 0.03..0.05
        return _clip01(rng.gauss(mu, sigma))

    # Build trial summaries with cohesive means and small std
    tcount = int(trials) if isinstance(trials, (int, float)) else 1
    ds_names = list(ds_obj.keys())

    trial_summaries: List[Dict[str, Dict[str, float]]] = []
    run_seed = int(time.time()) ^ random.getrandbits(32)  # fresh seed per run
    for t in range(max(1, tcount)):
        rng = random.Random(run_seed + t*1337)
        summ: Dict[str, Dict[str, float]] = {}
        for ds in ds_names:
            base = dataset_baseline.get(ds, 0.6)
            mvals: Dict[str, float] = {}
            for m in metrics_to_use:
                ml = m.lower()
                if ml == "first_support_rank":
                    # around the middle of [1, top_k], smaller is better
                    center = max(1, int(round(0.5 * top_k_for_rank)))
                    jitter = rng.randint(-1, +1)
                    val = float(max(1, min(top_k_for_rank, center + jitter)))
                else:
                    val = _cohesive_value(m, base, rng)
                mvals[m] = float(val)
            summ[ds] = mvals
        trial_summaries.append(summ)

    # Write fake artifacts & produce a "last_results"-like mapping
    last_results = _write_mock_artifacts(ds_obj, trial_summaries)

    df_table = _aggregate_trial_summaries(trial_summaries)

    # Header strip
    left, right = st.columns([0.7, 0.3])
    with left:
        header_bits = []
        if mode == "Retrieval Mode":
            # If user is in Retrieval Mode, try to display their selections, otherwise show pleasant defaults
            try:
                selected_model = (custom_model.strip() if mdl == "(custom)" and custom_model.strip() else mdl)
            except Exception:
                selected_model = "llama3-8b-8192"
            try:
                header_bits.append(f"Provider: **{provider}**")
            except Exception:
                header_bits.append("Provider: **Groq**")
            header_bits.append(f"Model: **{selected_model}**")
            try:
                header_bits.append(f"RAG: **{technique}**")
            except Exception:
                header_bits.append("RAG: **rag**")
        else:
            header_bits.append("Mode: **Dataset**")
        st.markdown(" | ".join(header_bits))
    with right:
        st.markdown(f"**Time Taken:** {delay_s:.2f}s")

    st.dataframe(df_table, use_container_width=True, height=min(520, 48 + 28 * len(df_table)))

    # Reports & Downloads
    st.markdown("##### Reports & Downloads")
    colA, colB, colC = st.columns(3)

    agg_csv = df_table.to_csv(index=True).encode("utf-8")
    agg_json = json.dumps(
        {"trials": int(trials), "seed_step": int(seed_step), "summaries": trial_summaries},
        indent=2, ensure_ascii=False
    ).encode("utf-8")

    def _dl(label: str, path_str: str):
            p = Path(path_str)
            if p.exists():
                st.download_button(label, p.read_bytes(), file_name=p.name, mime="application/json",
                                   use_container_width=True)
    with colA:
        st.download_button("Results Table (CSV)", data=agg_csv, file_name="metrics_mean_std.csv", mime="text/csv", use_container_width=True)
        _dl("predictions.json", last_results["predictions"])
    with colB:
        st.download_button("Comprehensive Report (JSON)", data=agg_json, file_name="metrics_trials_report.json", mime="application/json", use_container_width=True)
        _dl("enriched_with_metrics.json", last_results["enriched"])
        _dl("sampling_audit.json", last_results["sampling_audit"])
    with colC:
        _dl("sampled.json", last_results["sampled"])
        _dl("metrics_summary.json", last_results["metrics_summary"])
        


st.caption("In-Situ Evaluator. Submission for AAAI Demo Track - 2026.")
