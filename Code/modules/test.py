# app.py
import json, io
import streamlit as st
import subsampling as sampler  # your heavy lifter

st.title("QnA/RAG Sampler")

# 1) pick method
methods = list(sampler.list_methods().keys())
method = st.selectbox("Sampling method", methods, index=methods.index("uniform_random"))

# 2) common params
n = st.number_input("n", min_value=1, value=100)
seed = st.number_input("seed", min_value=0, value=42)
extra = {}

# --- stratified-specific controls (only change vs your original) ---
if method == "stratified":
    defaults = sampler.list_methods()["stratified"]["defaults"]
    # choose fields to stratify on
    extra["strata_keys"] = st.multiselect(
        "strata_keys",
        options=["topic", "answer_type", "lang"],
        default=defaults.get("strata_keys", ["topic","answer_type"]),
        help="Fields present in each item used to form strata"
    )

    # allocation rule
    extra["allocation"] = st.selectbox(
        "allocation",
        options=["proportional", "equal", "neyman"],
        index=["proportional", "equal", "neyman"].index(defaults.get("allocation", "proportional")),
        help="How many to take from each stratum"
    )

    # variability proxy (only for Neyman)
    if extra["allocation"] == "neyman":
        var_default = defaults.get("variability_field", "length") or "length"
        var_choice = st.selectbox(
            "variability_field (for Neyman)",
            options=["length", "difficulty", "None"],
            index=["length", "difficulty", "None"].index("None" if var_default is None else var_default),
            help="Proxy for per-stratum variability"
        )
        extra["variability_field"] = None if var_choice == "None" else var_choice

        # when using 'length' as proxy, tell sampler where the question text lives
        extra["question_field"] = st.text_input(
            "question_field (used if variability_field = length)",
            value=defaults.get("question_field", "question")
        )
    else:
        extra["variability_field"] = None
        # still allow specifying question_field if your data uses 'Question'
        extra["question_field"] = st.text_input(
            "question_field",
            value=defaults.get("question_field", "question"),
            help="Set to 'Question' if your key is capitalized"
        )

    # minimum per stratum (optional)
    extra["min_per_stratum"] = st.number_input(
        "min_per_stratum",
        min_value=0,
        value=int(defaults.get("min_per_stratum", 0)),
        help="Guarantee at least this many from each stratum when possible"
    )
# --- end stratified-specific ---

# 3) upload data (JSON list or JSONL)
file = st.file_uploader("Upload .json or .jsonl", type=["json","jsonl"])

def _read_upload(fobj):
    name = fobj.name.lower()
    if name.endswith(".jsonl"):
        return [json.loads(l) for l in io.TextIOWrapper(fobj, encoding="utf-8")]
    else:
        return json.load(io.TextIOWrapper(fobj, encoding="utf-8"))  # expect a list[dict]

if file and st.button("Run"):
    data = _read_upload(file)  # list[dict]
    params = {"n": int(n), "seed": int(seed), **extra}
    result = sampler.run_sampler(method, data, params)  # heavy lifter validates & runs
    st.write("Audit", result["audit"])
    st.json(result["items"])  # preview
