# evals.py  (drop-in replacement; API-compatible with your previous evals_qna_rag.py)
# Non-LLM QnA/RAG evaluator with selectable metrics via CLI.

from __future__ import annotations
import argparse, json, math, re, string
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter
import difflib
from functools import lru_cache

# ---------------- Config knobs (speed/robustness) ----------------
# If inputs are extremely long, Levenshtein and token LCS are O(n*m).
# To keep UI snappy, we cap and fall back to a fast proxy when exceeded.
MAX_LEVENSHTEIN_CHARS = 3000      # if max(len(a), len(b)) > cap → use seqmatch proxy
MAX_ROUGEL_TOKENS     = 2500      # if token lengths product > cap^2 → use seqmatch proxy

# ---------------- Light helpers ----------------
_ARTICLES = {"a", "an", "the"}
_PUNCT = str.maketrans("", "", string.punctuation)

def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)

@lru_cache(maxsize=100000)
def _normalize(s: Any, lower=True, rm_punct=True, rm_articles=True) -> str:
    s = _safe_str(s).strip()
    if lower: s = s.lower()
    if rm_punct: s = s.translate(_PUNCT)
    s = re.sub(r"\s+", " ", s).strip()
    if rm_articles:
        s = " ".join(t for t in s.split() if t not in _ARTICLES)
    return s

@lru_cache(maxsize=100000)
def _toks_cached(s: Any) -> Tuple[str, Tuple[str, ...]]:
    """Return (normalized_string, tuple_of_tokens) with caching to avoid rework."""
    n = _normalize(s)
    toks = tuple(n.split()) if n else tuple()
    return n, toks

def _toks(s: Any) -> List[str]:
    return list(_toks_cached(s)[1])

def _lcs_len_tokens(a: Tuple[str, ...], b: Tuple[str, ...]) -> int:
    # O(nm) DP on tokens; used only when sizes are reasonable.
    n, m = len(a), len(b)
    if n == 0 or m == 0: return 0
    dp = [0] * (m + 1)
    for i in range(1, n + 1):
        prev = 0
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            if ai == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = dp[j] if dp[j] >= dp[j - 1] else dp[j - 1]
            prev = tmp
    return dp[m]

# ---------------- Generic metrics (always local/fast) ----------------
def em(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0

def f1(pred: str, gold: str) -> float:
    _, pt = _toks_cached(pred)
    _, gt = _toks_cached(gold)
    if not pt and not gt: return 1.0
    if not pt or not gt:  return 0.0
    pc, gc = Counter(pt), Counter(gt)
    overlap = sum((pc & gc).values())
    if overlap == 0: return 0.0
    prec = overlap / sum(pc.values())
    rec  = overlap / sum(gc.values())
    return 2 * prec * rec / (prec + rec + 1e-12)

def rougeL_token(pred: str, gold: str, beta: float = 1.2) -> float:
    _, pt = _toks_cached(pred)
    _, gt = _toks_cached(gold)
    if not pt or not gt: return 0.0

    # guard: for very long inputs, use SequenceMatcher as a fast proxy
    if len(pt) * len(gt) > MAX_ROUGEL_TOKENS * MAX_ROUGEL_TOKENS:
        a, b = " ".join(pt), " ".join(gt)
        ratio = difflib.SequenceMatcher(None, a, b).ratio()
        return ratio  # proxy; fast and bounded

    lcs = _lcs_len_tokens(pt, gt)
    prec = lcs / len(pt); rec = lcs / len(gt)
    if prec == 0 and rec == 0: return 0.0
    b2 = beta * beta
    return (1 + b2) * prec * rec / (rec + b2 * prec + 1e-12)

# ----- Textual metrics (non-LLM, no extra deps) -----
def jaccard(pred: str, gold: str) -> float:
    _, A = _toks_cached(pred); A = set(A)
    _, B = _toks_cached(gold); B = set(B)
    if not A and not B: return 1.0
    if not A or not B:  return 0.0
    return len(A & B) / len(A | B)

def _lev_distance(a: str, b: str) -> int:
    # classic DP (O(nm)) with rolling row
    n, m = len(a), len(b)
    if n == 0: return m
    if m == 0: return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        ai = a[i - 1]
        for j in range(1, m + 1):
            tmp = dp[j]
            cost = 0 if ai == b[j - 1] else 1
            # min(delete, insert, substitute)
            dj = dp[j] + 1
            ij = dp[j - 1] + 1
            sj = prev + cost
            dp[j] = dj if dj <= ij and dj <= sj else (ij if ij <= sj else sj)
            prev = tmp
    return dp[m]

def levenshtein_sim(pred: str, gold: str) -> float:
    a = _normalize(pred); b = _normalize(gold)
    if not a and not b: return 1.0
    # guard long strings
    if max(len(a), len(b)) > MAX_LEVENSHTEIN_CHARS:
        return difflib.SequenceMatcher(None, a, b).ratio()
    dist = _lev_distance(a, b)
    return 1.0 - dist / max(1, max(len(a), len(b)))

def seqmatch(pred: str, gold: str) -> float:
    a = _normalize(pred); b = _normalize(gold)
    return difflib.SequenceMatcher(None, a, b).ratio()

def tfidf_cos(pred: str, gold: str) -> float:
    _, p_toks = _toks_cached(pred)
    _, g_toks = _toks_cached(gold)
    vocab = sorted(set(p_toks) | set(g_toks))
    if not vocab: return 1.0
    p_set, g_set = set(p_toks), set(g_toks)
    def tfidf_vec(toks):
        tf = Counter(toks)
        N = 2
        df = {t: (1 if t in p_set else 0) + (1 if t in g_set else 0) for t in vocab}
        idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in vocab}
        L = max(1, len(toks))
        return [(tf[t] / L) * idf[t] for t in vocab]
    va, vb = tfidf_vec(p_toks), tfidf_vec(g_toks)
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va)); nb = math.sqrt(sum(y * y for y in vb))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)

def jsd(pred: str, gold: str) -> float:
    pa = Counter(_toks(pred)); pb = Counter(_toks(gold))
    vocab = set(pa) | set(pb)
    if not vocab: return 0.0
    sa, sb = sum(pa.values()), sum(pb.values())
    if sa == 0 and sb == 0: return 0.0
    if sa == 0 or sb == 0:  return 1.0
    Pa = [pa[t] / sa for t in vocab]
    Pb = [pb[t] / sb for t in vocab]
    M = [(a + b) / 2 for a, b in zip(Pa, Pb)]
    def _kl(P, Q):
        eps = 1e-12
        return sum(p * math.log((p + eps) / (q + eps)) for p, q in zip(P, Q))
    return 0.5 * _kl(Pa, M) + 0.5 * _kl(Pb, M)

# ---------------- Optional deps (lazy) ----------------
_bleu = _rouge = _meteor = None
_nli_tok = _nli_model = None
_sbert = None

def _ensure_hf_metrics(wanted: set):
    """Lazy-load HuggingFace evaluate metrics if requested; tolerant if not installed."""
    global _bleu, _rouge, _meteor
    try:
        import evaluate  # type: ignore
    except Exception:
        return
    if _bleu is None and "bleu" in wanted: _bleu = evaluate.load("bleu")
    if _rouge is None and {"rouge1","rouge2","rougeLsum"} & wanted: _rouge = evaluate.load("rouge")
    if _meteor is None and "meteor" in wanted: _meteor = evaluate.load("meteor")

def _ensure_nli(wanted: set):
    global _nli_tok, _nli_model
    if _nli_model is None and {"nli_entail","nli_contra","nli_neutral"} & wanted:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
            _nli_tok = AutoTokenizer.from_pretrained("roberta-large-mnli")
            _nli_model = AutoModelForSequenceClassification.from_pretrained("roberta-large-mnli")
        except Exception:
            _nli_tok = _nli_model = None

def _ensure_sbert(wanted: set):
    global _sbert
    if _sbert is None and "cosine" in wanted:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            _sbert = SentenceTransformer("all-mpnet-base-v2")
        except Exception:
            _sbert = None

def nli_triplet(ref: str, pred: str) -> Dict[str, float]:
    import torch  # type: ignore
    inputs = _nli_tok(ref, pred, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = _nli_model(**inputs).logits
    probs = logits.softmax(dim=-1)[0]
    return {"nli_contra": probs[0].item(), "nli_neutral": probs[1].item(), "nli_entail": probs[2].item()}

def cosine_sim(ref: str, pred: str) -> float:
    from sentence_transformers import util  # type: ignore
    a = _sbert.encode(ref, convert_to_tensor=True, truncate=True).unsqueeze(0)
    b = _sbert.encode(pred, convert_to_tensor=True, truncate=True).unsqueeze(0)
    return float(util.pytorch_cos_sim(a, b)[0][0])

# ---------------- RAG helpers/metrics ----------------
def _contexts(entry: dict) -> List[str]:
    if isinstance(entry.get("RetrievedContexts"), list):
        return [str(x or "") for x in entry["RetrievedContexts"]]
    ctx = entry.get("Context", "")
    return [str(ctx)] if ctx else []

def _norm_if(s: str, norm: bool) -> str:
    return _normalize(s) if norm else (s or "")

def _contexts_tokens_and_set(ctxs: List[str], norm=True) -> Tuple[List[str], set]:
    # Join once for coverage stats; build token set once
    C = _norm_if(" ".join(ctxs), norm)
    Cset = set(C.split()) if C else set()
    return ([C], Cset)

def support_contains(needle: str, hay: str, norm=True) -> float:
    n = _norm_if(needle, norm); h = _norm_if(hay, norm)
    return 1.0 if n and (n in h) else 0.0

def coverage_stats(pred: str, gold: str, ctxs: List[str], norm=True) -> Dict[str, float]:
    _, Cset = _contexts_tokens_and_set(ctxs, norm)
    pt = _toks(pred); gt = _toks(gold)
    def frac_in(tokens):
        if not tokens or not Cset: return 0.0
        return sum(1 for t in tokens if t in Cset) / len(tokens)
    p = frac_in(pt); r = frac_in(gt)
    f = 0.0 if (p == 0 and r == 0) else 2 * p * r / (p + r)
    return {"coverage_precision": p, "coverage_recall": r, "coverage_f1": f}

def first_support_rank_by_substring(ans: str, ctxs: List[str], norm=True) -> Optional[int]:
    a = _norm_if(ans, norm)
    if not a: return None
    for i, c in enumerate(ctxs):
        if a in _norm_if(c, norm): return i
    return None

def retrieval_from_docids(entry: dict, ks: List[int]) -> Dict[str, float]:
    docids = entry.get("RetrievedDocIds")
    gold = entry.get("GoldDocId")
    out: Dict[str, float] = {}
    if not isinstance(docids, list) or gold is None:
        return out
    rank = docids.index(gold) if gold in docids else None
    for k in ks:
        out[f"hit@{k}"] = 1.0 if (rank is not None and rank < k) else 0.0
    out["mrr"] = 1.0 / (rank + 1) if rank is not None else 0.0
    for k in ks:
        denom = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(docids))))
        dcg = 1.0 / math.log2(rank + 2) if (rank is not None and rank < k) else 0.0
        out[f"ndcg@{k}"] = dcg / (denom if denom > 0 else 1.0)
    return out

def retrieval_from_substring(gold_text: str, ctxs: List[str], ks: List[int], norm=True) -> Dict[str, float]:
    out: Dict[str, float] = {}
    rank = first_support_rank_by_substring(gold_text, ctxs, norm=norm)
    for k in ks:
        out[f"hit@{k}"] = 1.0 if (rank is not None and rank < k) else 0.0
    out["mrr"] = 1.0 / (rank + 1) if rank is not None else 0.0
    for k in ks:
        denom = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ctxs))))
        dcg = 1.0 / math.log2(rank + 2) if (rank is not None and rank < k) else 0.0
        out[f"ndcg@{k}"] = dcg / (denom if denom > 0 else 1.0)
    if rank is not None:
        out["first_support_rank"] = rank
    return out

# ---------------- Scoring one entry ----------------
GENERIC_ALL = {
    "em","f1","rougeL",
    "bleu","rouge1","rouge2","rougeLsum","meteor",
    "nli_entail","nli_contra","nli_neutral","cosine",
    "jaccard","tfidf_cos","levenshtein","seqmatch","jsd"
}
RAG_ALL = {
    "support_gold","support_pred",
    "coverage_precision","coverage_recall","coverage_f1",
    "mrr","first_support_rank",
    # plus pattern metrics: hit@K, ndcg@K
}

def score_entry(entry: dict, wanted: set, ks: List[int], norm_substring=True) -> Dict[str, float]:
    ref  = _safe_str(entry.get("Response", ""))
    pred = _safe_str(entry.get("Predicted", ""))
    ctxs = _contexts(entry)

    out: Dict[str, float] = {}

    # --- Generic (light) ---
    if "em" in wanted:     out["EM"] = em(pred, ref)
    if "f1" in wanted:     out["F1"] = f1(pred, ref)
    if "rougeL" in wanted: out["ROUGE-L"] = rougeL_token(pred, ref)

    # HF evaluate families — robust to empty text and missing deps
    if {"bleu","rouge1","rouge2","rougeLsum","meteor"} & wanted:
        _ensure_hf_metrics(wanted)
        pred_has = bool(_toks(pred))
        ref_has  = bool(_toks(ref))

        # BLEU
        if "bleu" in wanted:
            if pred_has and ref_has and _bleu is not None:
                try:
                    out["bleu"] = float(_bleu.compute(predictions=[pred], references=[[ref]])["bleu"])
                except Exception:
                    out["bleu"] = 0.0
            else:
                out["bleu"] = 1.0 if (not pred_has and not ref_has) else 0.0

        # ROUGE (HF)
        if {"rouge1","rouge2","rougeLsum"} & wanted:
            r = {}
            if pred_has and ref_has and _rouge is not None:
                try:
                    r = _rouge.compute(predictions=[pred], references=[ref])
                except Exception:
                    r = {}
            if "rouge1"    in wanted: out["rouge1"]    = float(r.get("rouge1", 0.0))
            if "rouge2"    in wanted: out["rouge2"]    = float(r.get("rouge2", 0.0))
            if "rougeLsum" in wanted: out["rougeLsum"] = float(r.get("rougeLsum", 0.0))

        # METEOR
        if "meteor" in wanted:
            if pred_has and ref_has and _meteor is not None:
                try:
                    out["meteor"] = float(_meteor.compute(predictions=[pred], references=[ref])["meteor"])
                except Exception:
                    out["meteor"] = 0.0
            else:
                out["meteor"] = 1.0 if (not pred_has and not ref_has) else 0.0

    # NLI & SBERT cosine (optional deps → safe fallbacks)
    if {"nli_entail","nli_contra","nli_neutral"} & wanted:
        _ensure_nli(wanted)
        if _nli_model is not None and _nli_tok is not None and (_normalize(ref) or _normalize(pred)):
            try:
                trip = nli_triplet(ref, pred)
            except Exception:
                trip = {"nli_contra": 0.0, "nli_neutral": 0.0, "nli_entail": 0.0}
        else:
            trip = {"nli_contra": 0.0, "nli_neutral": 0.0, "nli_entail": 0.0}
        for k in ("nli_contra","nli_neutral","nli_entail"):
            if k in wanted: out[k] = float(trip[k])

    if "cosine" in wanted:
        _ensure_sbert(wanted)
        if _sbert is not None and (_normalize(ref) or _normalize(pred)):
            try:
                out["cosine"] = float(cosine_sim(ref, pred))
            except Exception:
                out["cosine"] = 0.0
        else:
            out["cosine"] = 0.0

    # Textual metrics (local)
    if "jaccard" in wanted:      out["jaccard"] = jaccard(pred, ref)
    if "tfidf_cos" in wanted:    out["tfidf_cos"] = tfidf_cos(pred, ref)
    if "levenshtein" in wanted:  out["levenshtein"] = levenshtein_sim(pred, ref)
    if "seqmatch" in wanted:     out["seqmatch"] = seqmatch(pred, ref)
    if "jsd" in wanted:          out["jsd"] = jsd(pred, ref)

    # --- RAG ---
    if "support_gold" in wanted:
        out["SupportGoldInContexts"] = max((support_contains(ref, c, norm_substring) for c in ctxs), default=0.0)
    if "support_pred" in wanted:
        out["SupportPredInContexts"] = max((support_contains(pred, c, norm_substring) for c in ctxs), default=0.0)
    if {"coverage_precision","coverage_recall","coverage_f1"} & wanted:
        cov = coverage_stats(pred, ref, ctxs, norm_substring)
        if "coverage_precision" in wanted: out["CoveragePrecision"] = cov["coverage_precision"]
        if "coverage_recall"    in wanted: out["CoverageRecall"]    = cov["coverage_recall"]
        if "coverage_f1"        in wanted: out["CoverageF1"]        = cov["coverage_f1"]

    want_any_retrieval = any(m.startswith("hit@") or m.startswith("ndcg@") or m in {"mrr","first_support_rank"} for m in wanted)
    if want_any_retrieval:
        if isinstance(entry.get("RetrievedDocIds"), list) and (entry.get("GoldDocId") is not None):
            ret = retrieval_from_docids(entry, ks)
        else:
            ret = retrieval_from_substring(ref, ctxs, ks, norm=norm_substring)
        for k, v in ret.items():
            key = k.lower()
            if (key in wanted) or any(key == f"hit@{K}" for K in ks if f"hit@{K}" in wanted) or any(key == f"ndcg@{K}" for K in ks if f"ndcg@{K}" in wanted):
                out["MRR" if key == "mrr" else k] = float(v)

    return out

# ---------------- Public API ----------------
def calculate_metrics(data: Dict[str, List[dict]], metrics: List[str], k_values: List[int]) -> Dict[str, List[dict]]:
    wanted = set(m.lower() for m in (metrics or []))
    ks = sorted(set(int(k) for k in (k_values or []) if int(k) > 0)) or [1, 3, 5, 10]
    for ds_name, rows in (data or {}).items():
        for entry in (rows or []):
            scores = score_entry(entry, wanted, ks)
            entry.setdefault("metrics", {}).update(scores)
    return data

def evals(data: Dict[str, List[dict]], metrics: List[str], k_values: List[int] = [1,3,5,10]):
    return calculate_metrics(data, metrics=metrics, k_values=k_values)

# ---------------- CLI ----------------
GENERIC_GROUP = [
    "em","f1","rougeL",
    "bleu","rouge1","rouge2","rougeLsum","meteor",
    "nli_entail","nli_contra","nli_neutral","cosine",
    "jaccard","tfidf_cos","levenshtein","seqmatch","jsd"
]
RAG_GROUP = [
    "support_gold","support_pred","coverage_precision","coverage_recall","coverage_f1",
    "hit@1","hit@3","hit@5","hit@10","mrr",
    "ndcg@1","ndcg@3","ndcg@5","ndcg@10","first_support_rank"
]

def _expand_groups(group: Optional[str], metrics: List[str]) -> List[str]:
    if metrics: return metrics
    if group == "generic": return GENERIC_GROUP
    if group == "rag":     return RAG_GROUP
    return GENERIC_GROUP + RAG_GROUP

def _extract_k_from_metrics(metrics: List[str]) -> List[int]:
    ks = []
    for m in metrics:
        m = m.lower()
        if m.startswith("hit@") or m.startswith("ndcg@"):
            try:
                ks.append(int(m.split("@", 1)[1]))
            except:  # noqa
                pass
    return sorted(set(ks))

def main():
    ap = argparse.ArgumentParser(description="Non-LLM QnA/RAG evaluator (metric-selectable)")
    ap.add_argument("--in", dest="input", required=True, help="Path to data.json (dict: dataset -> list of entries)")
    ap.add_argument("--metrics", nargs="*", default=[], help="Pick metrics, e.g., em f1 jaccard tfidf_cos hit@5 mrr")
    ap.add_argument("--group", choices=["generic","rag","all"], default=None, help="Shortcut to pick a metrics set")
    ap.add_argument("--k", nargs="*", type=int, default=[], help="K values for hit@K/ndcg@K (default 1 3 5 10)")
    ap.add_argument("--out", type=str, default=None, help="Write enriched JSON here")
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    metrics = _expand_groups(args.group, args.metrics)
    ks_from_names = _extract_k_from_metrics(metrics)
    k_values = args.k or ks_from_names or [1, 3, 5, 10]

    enriched = calculate_metrics(data, metrics=metrics, k_values=k_values)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)
        print(f"wrote {args.out}")
    else:
        preview = {ds: rows[:2] for ds, rows in enriched.items()}
        print(json.dumps({"metrics": metrics, "k_values": k_values, "preview": preview}, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
