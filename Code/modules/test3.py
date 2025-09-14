# test3.py
# Streamlit UI for unified_rag.py
#
# Usage:
#   streamlit run test3.py
#
# Expect unified_rag.py in the same folder.

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

# Import the unified engine
import retrieval as ur


st.set_page_config(page_title="RAG Workbench", page_icon="🧠", layout="wide")
st.title("🧠 RAG Workbench (RAG | RAPTOR | GraphRAG)")

# -------- Helpers -------------------------------------------------------------

def _load_json_from_uploader(uploader) -> Dict[str, Any] | List[Any] | None:
    if uploader is None:
        return None
    try:
        return json.loads(uploader.getvalue().decode("utf-8"))
    except Exception as e:
        st.error(f"Failed to parse uploaded JSON: {e}")
        return None

def _load_json_from_text(text: str) -> Dict[str, Any] | List[Any] | None:
    if not text.strip():
        return None
    try:
        return json.loads(text)
    except Exception as e:
        st.error(f"Failed to parse JSON from text: {e}")
        return None

def _temp_json_file(obj: Dict[str, Any] | List[Any]) -> Path:
    tf = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
    Path(tf.name).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return Path(tf.name)

def _run_engine(
    technique: str,
    dataset: Dict[str, List[Dict[str, Any]]],
    model_config: Dict[str, Any],
    access_config: Dict[str, Any] | None,
    chunk_size: int,
    top_k: int,
    num_clusters: int,
    edge_threshold: float,
) -> Dict[str, Any]:
    # Save access config (or use default ./config.json if not provided)
    if access_config is not None:
        access_cfg_path = _temp_json_file(access_config)
    else:
        access_cfg_path = Path("config.json")  # falls back to local file
    Strategy = ur.STRATEGY_REGISTRY[technique]

    if technique == "rag":
        return Strategy.run(model_config, dataset, access_config_path=access_cfg_path)
    elif technique == "raptor":
        return Strategy.run(
            model_config,
            dataset,
            access_config_path=access_cfg_path,
            chunk_size=chunk_size,
            num_clusters=num_clusters,
            top_k=top_k,
        )
    elif technique == "graphrag":
        return Strategy.run(
            model_config,
            dataset,
            access_config_path=access_cfg_path,
            chunk_size=chunk_size,
            top_k=top_k,
            edge_threshold=edge_threshold,
        )
    else:
        raise ValueError(f"Unknown technique: {technique}")

def _preview_table(data: Dict[str, Any]):
    # Flatten for display
    rows: List[Dict[str, Any]] = []
    for ds_key, entries in data.items():
        for e in entries:
            rows.append({
                "Dataset": ds_key,
                "Question": e.get("Question", ""),
                "Context": e.get("Context", "")[:200] + ("..." if len(e.get("Context", "")) > 200 else ""),
                "Response": e.get("Response", ""),
                "Predicted": e.get("Predicted", ""),
            })
    if rows:
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No rows to display.")

# -------- Sidebar Controls ----------------------------------------------------

st.sidebar.header("Controls")

technique = st.sidebar.selectbox(
    "Technique",
    options=list(ur.STRATEGY_REGISTRY.keys()),
    index=0,
    help="Choose which strategy to run."
)

mode = st.sidebar.radio(
    "Mode",
    options=["Batch Dataset", "Single Query"],
    index=0,
    help="Batch runs over a JSON dataset; Single lets you test one Q&A."
)

st.sidebar.subheader("Parameters")
chunk_size = st.sidebar.number_input("Chunk size (chars)", 50, 2000, 100, step=10)
top_k = st.sidebar.number_input("Top-K", 1, 20, 3, step=1)
num_clusters = st.sidebar.number_input("RAPTOR: #clusters", 1, 50, 5, step=1)
edge_threshold = st.sidebar.slider("GraphRAG: edge threshold", 0.0, 1.0, 0.5, step=0.05)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Config")
model_cfg_upl = st.sidebar.file_uploader("Upload model_config.json", type=["json"])
model_cfg_text = st.sidebar.text_area(
    "Or paste model_config JSON",
    value="",
    height=200,
    placeholder='{"api_key":"...","model":"...","temperature":0.2,"max_tokens":512,"top_p":0.95,"stream":false,"stop":null,"type":"chat_completions"}'
)

st.sidebar.subheader("Access Config (optional)")
access_cfg_upl = st.sidebar.file_uploader("Upload access config (config.json)", type=["json"])
access_cfg_text = st.sidebar.text_area(
    "Or paste access config JSON",
    value="",
    height=160,
    placeholder='{"ACCESS_CONFIG": {"chat_completions":"https://api.groq.com/openai/v1/chat/completions"}}'
)

# -------- Main Panels ---------------------------------------------------------

if mode == "Batch Dataset":
    st.subheader("Batch Dataset")
    ds_upl = st.file_uploader("Upload dataset JSON", type=["json"])
    ds_text = st.text_area(
        "Or paste dataset JSON",
        value="",
        height=180,
        placeholder='{"squad": [{"Question": "What is RAG?", "Context": "Retrieval augmented generation ...", "Response": ""}]}'
    )

    run_batch = st.button("Run Batch")

    dataset = _load_json_from_uploader(ds_upl) or _load_json_from_text(ds_text)
    if dataset:
        with st.expander("Dataset Preview", expanded=False):
            st.json(dataset)

    model_config = _load_json_from_uploader(model_cfg_upl) or _load_json_from_text(model_cfg_text)
    access_config = _load_json_from_uploader(access_cfg_upl) or _load_json_from_text(access_cfg_text)

    if run_batch:
        if not dataset:
            st.error("Please provide a dataset JSON.")
        elif not model_config:
            st.error("Please provide a model_config JSON (upload or paste).")
        else:
            with st.status("Running… this can take a moment depending on dataset size.", expanded=True) as status:
                try:
                    out = _run_engine(
                        technique=technique,
                        dataset=dataset,
                        model_config=model_config,
                        access_config=access_config,
                        chunk_size=chunk_size,
                        top_k=top_k,
                        num_clusters=num_clusters,
                        edge_threshold=edge_threshold,
                    )
                    status.update(label="Completed ✅", state="complete")
                    st.success("Run finished.")
                    _preview_table(out)
                    st.download_button(
                        label="Download results JSON",
                        file_name="results.out.json",
                        mime="application/json",
                        data=json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8"),
                        use_container_width=True,
                    )
                except Exception as e:
                    status.update(label="Failed ❌", state="error")
                    st.exception(e)

else:
    st.subheader("Single Query Playground")
    col1, col2 = st.columns(2)
    with col1:
        q = st.text_area("Question", placeholder="Ask something…", height=140)
        run_single = st.button("Run Single Query")
    with col2:
        ctx = st.text_area("Context", placeholder="Paste the context/passages here…", height=140)

    model_config = _load_json_from_uploader(model_cfg_upl) or _load_json_from_text(model_cfg_text)
    access_config = _load_json_from_uploader(access_cfg_upl) or _load_json_from_text(access_cfg_text)

    if run_single:
        if not q.strip():
            st.error("Please enter a Question.")
        elif not ctx.strip():
            st.error("Please enter a Context.")
        elif not model_config:
            st.error("Please provide a model_config JSON (upload or paste).")
        else:
            dataset = {
                "interactive": [
                    {"Question": q.strip(), "Context": ctx.strip(), "Response": ""}
                ]
            }
            with st.status("Running…", expanded=True) as status:
                try:
                    out = _run_engine(
                        technique=technique,
                        dataset=dataset,
                        model_config=model_config,
                        access_config=access_config,
                        chunk_size=chunk_size,
                        top_k=top_k,
                        num_clusters=num_clusters,
                        edge_threshold=edge_threshold,
                    )
                    status.update(label="Completed ✅", state="complete")
                    entry = out["interactive"][0]
                    st.subheader("Answer")
                    st.write(entry.get("Predicted", ""))
                    with st.expander("Full JSON output", expanded=False):
                        st.json(out)
                    st.download_button(
                        label="Download result JSON",
                        file_name="single_result.out.json",
                        mime="application/json",
                        data=json.dumps(out, ensure_ascii=False, indent=2).encode("utf-8"),
                        use_container_width=True,
                    )
                except Exception as e:
                    status.update(label="Failed ❌", state="error")
                    st.exception(e)

# Footer tip
st.caption(
    "Tip: If you don't upload an Access Config, the app will look for ./config.json on disk for ACCESS_CONFIG."
)
