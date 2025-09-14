# subsampling.py
# One-stop sampling module for QnA/RAG datasets.
from __future__ import annotations

import argparse, json, math, os, random, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Tuple, Iterable, Optional
import heapq

# -------------------------
# Loader / Saver
# -------------------------

def _load_json_any(path_or_list: Any) -> List[dict]:
    if isinstance(path_or_list, list):
        return list(path_or_list)
    if not isinstance(path_or_list, str):
        raise ValueError("input must be a path to .json/.jsonl or a list[dict]")
    if path_or_list.endswith(".jsonl"):
        items = []
        with open(path_or_list, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    elif path_or_list.endswith(".json"):
        with open(path_or_list, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            raise ValueError(".json must contain a list of objects")
    else:
        raise ValueError("Unsupported file type. Use .json or .jsonl")

def _save_jsonl(path: str, items: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------------------
# Utilities
# -------------------------

def _rng(seed: Optional[int]) -> random.Random:
    return random.Random(None if seed is None else int(seed))

def _tokenize(text: str) -> List[str]:
    # very lightweight tokenizer
    return [t for t in ''.join(ch.lower() if ch.isalnum() else ' ' for ch in (text or "")).split() if t]

def _ensure_list(x) -> list:
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _get(item: dict, key: str, default=None):
    # dotted access (e.g., "metadata.topic" or "retrieval.candidates")
    cur = item
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def _parse_date(val) -> Optional[datetime]:
    if val is None:
        return None
    if isinstance(val, (int, float)):
        # epoch seconds
        try:
            return datetime.fromtimestamp(val, tz=timezone.utc)
        except Exception:
            return None
    if isinstance(val, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                if fmt.endswith("Z"):
                    return datetime.strptime(val, fmt).replace(tzinfo=timezone.utc)
                return datetime.strptime(val, fmt)
            except Exception:
                pass
        try:
            return datetime.fromisoformat(val)
        except Exception:
            return None
    return None

def _idset(seq: Iterable[object]) -> set[int]:
    # identity set for unhashable dicts
    return {id(x) for x in seq}

# -------------------------
# Simple TF-IDF (no sklearn)
# -------------------------

def _build_tfidf_matrix(items: List[dict], text_field: str = "question") -> Tuple[List[Dict[str,float]], Dict[str,float]]:
    # Returns list of sparse vectors (dict term->tfidf) and idf dict
    docs = []
    for it in items:
        text = _get(it, text_field) if "." in text_field else it.get(text_field, "") or ""
        docs.append(_tokenize(text))
    N = len(docs)
    df = Counter()
    for toks in docs:
        df.update(set(toks))
    # smoothed idf
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1.0 for t in df}
    vectors = []
    for toks in docs:
        tf = Counter(toks)
        L = max(1, len(toks))
        vec = {t: (tf[t] / L) * idf.get(t, 0.0) for t in tf}
        vectors.append(vec)
    return vectors, idf

def _cosine_sparse(a: Dict[str,float], b: Dict[str,float]) -> float:
    if not a or not b: return 0.0
    # dot
    dot = 0.0
    if len(a) < len(b):
        for k,v in a.items():
            if k in b: dot += v * b[k]
    else:
        for k,v in b.items():
            if k in a: dot += v * a[k]
    # norms
    na = math.sqrt(sum(v*v for v in a.values()))
    nb = math.sqrt(sum(v*v for v in b.values()))
    if na == 0.0 or nb == 0.0: return 0.0
    return dot / (na * nb)

# Faster cosine for internal k-center where we precompute norms
def _cosine_sparse_pre(a: Dict[str,float], na: float, b: Dict[str,float], nb: float) -> float:
    if not a or not b or na == 0.0 or nb == 0.0: return 0.0
    dot = 0.0
    if len(a) < len(b):
        for k,v in a.items():
            if k in b: dot += v * b[k]
    else:
        for k,v in b.items():
            if k in a: dot += v * a[k]
    return dot / (na * nb) if na and nb else 0.0

# -------------------------
# Methods
# -------------------------

def _no_sampling(items: List[dict], p: dict, rng: random.Random):
    out = list(items)
    return out, {
        "note": "no_sampling (returned all items)",
        "requested_n": p.get("n"),
        "actual_n": len(out)
    }

def _uniform_random(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}
    # sample without shuffling all
    idxs = rng.sample(range(N), n)
    out = [items[i] for i in idxs]
    return out, {"requested_n": n, "actual_n": len(out)}

def _stratified(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}

    strata_keys: List[str] = p.get("strata_keys") or []
    allocation: str = p.get("allocation", "proportional")  # proportional|equal|neyman
    variability_field: Optional[str] = p.get("variability_field")  # e.g., "length" or "difficulty"
    min_per = int(p.get("min_per_stratum", 0))
    question_field: str = p.get("question_field", "question")

    if not strata_keys:
        return _uniform_random(items, p, rng)

    # derive strata
    buckets = defaultdict(list)
    for it in items:
        key_parts = []
        for k in strata_keys:
            key_parts.append(str(_get(it, k) if "." in k else it.get(k)))
        key = "||".join(key_parts)
        buckets[key].append(it)

    # compute allocations
    N_buckets = sum(len(v) for v in buckets.values())
    k = len(buckets)
    alloc: Dict[str, int] = {}
    if allocation == "equal":
        base = n // k if k else 0
        rem = n - base * k
        for i, (s, arr) in enumerate(buckets.items()):
            alloc[s] = min(len(arr), base + (1 if i < rem else 0))
    elif allocation == "neyman":
        # Neyman: prop to N_h * S_h (std dev of given field or proxy)
        weights: Dict[str, float] = {}
        for s, arr in buckets.items():
            vals = []
            for it in arr:
                if variability_field in (None, "length"):
                    qtxt = _get(it, question_field) if "." in question_field else it.get(question_field, "")
                    vals.append(len(_tokenize(qtxt)))
                else:
                    v = _get(it, variability_field) if "." in variability_field else it.get(variability_field)
                    if isinstance(v, (int,float)):
                        vals.append(float(v))
                    else:
                        qtxt = _get(it, question_field) if "." in question_field else it.get(question_field, "")
                        vals.append(len(_tokenize(qtxt)))
            if len(vals) <= 1:
                std = 1.0
            else:
                m = sum(vals)/len(vals)
                std = math.sqrt(sum((x-m)**2 for x in vals)/max(1,len(vals)-1)) or 1.0
            weights[s] = max(1, len(arr)) * std
        total_w = sum(weights.values()) or 1.0
        for s, arr in buckets.items():
            alloc[s] = min(len(arr), int(round(n * (weights[s]/total_w))))
    else:  # proportional
        for s, arr in buckets.items():
            alloc[s] = min(len(arr), int(round(n * (len(arr)/N_buckets)))) if N_buckets else 0

    # ensure min_per and adjust by largest remainder
    floors: Dict[str,int] = {}
    remainders = []
    for s, arr in buckets.items():
        want = alloc.get(s, 0)
        want = max(want, min_per if min_per>0 else 0)
        want = min(want, len(arr))
        base = int(math.floor(want))
        floors[s] = base
        frac = (alloc.get(s,0) - base)
        remainders.append((frac, s))
    total = sum(floors.values())
    remain = max(0, n - total)
    remainders.sort(reverse=True)
    i = 0
    while remain > 0 and remainders:
        s = remainders[i][1]
        if floors[s] < len(buckets[s]):
            floors[s] += 1
            remain -= 1
        i = (i + 1) % len(remainders)

    # sample within each stratum (prefer sample over full shuffle)
    out = []
    counts = {}
    for s, arr in buckets.items():
        m = floors.get(s, 0)
        if m <= 0:
            counts[s] = 0
            continue
        if m >= len(arr):
            pick = list(arr)
        else:
            pick = [arr[j] for j in _rng(rng.random()).sample(range(len(arr)), m)]  # re-seed for independence
        out.extend(pick)
        counts[s] = len(pick)

    # if short due to bucket scarcity, fill remainder uniformly (O(N))
    if len(out) < n:
        chosen_ids = _idset(out)
        remaining = [it for it in items if id(it) not in chosen_ids]
        R = n - len(out)
        if R >= len(remaining):
            fill = remaining
        else:
            fill = [remaining[j] for j in rng.sample(range(len(remaining)), R)]
        out.extend(fill)

    return out, {
        "requested_n": n,
        "actual_n": len(out),
        "counts_by_stratum": counts,
        "allocation": allocation,
        "variability_field": variability_field if allocation=="neyman" else None,
        "question_field": question_field
    }

def _get_embedding(item: dict, field: str = "embedding") -> Optional[List[float]]:
    v = _get(item, field) if "." in field else item.get(field)
    if isinstance(v, list) and all(isinstance(x, (int,float)) for x in v):
        return [float(x) for x in v]
    return None

def _diversity_kcenter(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}

    emb_field = p.get("embedding_field","embedding")
    text_field = p.get("text_field","question")

    # Vectors: prefer embeddings; else TF-IDF
    have_embedding = True
    for it in items:
        if _get_embedding(it, emb_field) is None:
            have_embedding = False
            break

    if have_embedding:
        vectors: List[Dict[str,float]] = []
        norms: List[float] = []
        for it in items:
            emb = _get_embedding(it, emb_field) or []
            vec = {str(i): float(v) for i, v in enumerate(emb) if v != 0.0}
            vectors.append(vec)
            norms.append(math.sqrt(sum(val*val for val in vec.values())) or 0.0)
        note = "used provided embeddings"
    else:
        vectors, _ = _build_tfidf_matrix(items, text_field=text_field)
        norms = [math.sqrt(sum(val*val for val in vec.values())) or 0.0 for vec in vectors]
        note = "used TF-IDF fallback"

    # k-center greedy (maximize minimum distance to current set), distance = 1 - cosine
    first_idx = rng.randrange(N)
    selected = [first_idx]
    selected_mask = [False]*N
    selected_mask[first_idx] = True

    # initialize min distances to first center
    min_dist = [1.0 - _cosine_sparse_pre(vectors[first_idx], norms[first_idx], v, norms[i]) for i, v in enumerate(vectors)]
    min_dist[first_idx] = -1.0

    while len(selected) < n:
        # farthest point from current set
        far_idx = max(range(N), key=lambda i: min_dist[i])
        selected.append(far_idx)
        selected_mask[far_idx] = True
        min_dist[far_idx] = -1.0

        # update min distances with the newly selected center
        vj, nj = vectors[far_idx], norms[far_idx]
        for i in range(N):
            if min_dist[i] < 0.0:  # already selected
                continue
            d = 1.0 - _cosine_sparse_pre(vj, nj, vectors[i], norms[i])
            if d < min_dist[i]:
                min_dist[i] = d

    out = [items[i] for i in selected]

    # diversity stat: avg pairwise cosine among selected (optional, small)
    if len(selected) > 1:
        cos_vals = []
        for ii in range(len(selected)):
            i = selected[ii]
            for jj in range(ii+1, len(selected)):
                j = selected[jj]
                cos_vals.append(_cosine_sparse_pre(vectors[i], norms[i], vectors[j], norms[j]))
        avg_cos = sum(cos_vals)/len(cos_vals)
    else:
        avg_cos = 1.0
    return out, {"requested_n": n, "actual_n": len(out), "note": note, "avg_pairwise_cosine": round(avg_cos,4)}

def _importance_weighted(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}

    wf = p.get("weight_field", "weight")
    floor_prob = float(p.get("floor_prob", 1e-9))
    cap_ratio = float(p.get("cap_ratio", 0.2))  # cap = mean_p * (1/cap_ratio)
    weights = []
    for it in items:
        w = _get(it, wf) if "." in wf else it.get(wf)
        try:
            w = float(w)
        except Exception:
            w = 0.0
        weights.append(max(0.0, w))
    total = sum(weights)
    if total <= 0:
        return _uniform_random(items, p, rng)

    mean_p = (1.0/len(items))
    probs = [(w/total) + floor_prob for w in weights]
    s = sum(probs); probs = [x/s for x in probs]
    max_allowed = mean_p * (1.0/cap_ratio if cap_ratio>0 else 1e9)
    probs = [min(x, max_allowed) for x in probs]
    s = sum(probs); probs = [x/s for x in probs]
    idxs = _weighted_sample_without_replacement(range(len(items)), probs, n, rng)
    out = [items[i] for i in idxs]
    return out, {"requested_n": n, "actual_n": len(out)}

def _weighted_sample_without_replacement(indices: Iterable[int], probs: List[float], k: int, rng: random.Random) -> List[int]:
    # Efraimidis–Spirakis algorithm (2006): keys = u^(1/w) (we adapt using probs)
    pairs = []
    for i, p in enumerate(probs):
        p = max(p, 1e-12)
        u = rng.random()
        key = u ** (1.0 / p)
        pairs.append((key, i))
    pairs.sort(reverse=True)
    return [i for _, i in pairs[:min(k, len(pairs))]]

def _active_learning(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}

    mode = p.get("mode","entropy")  # entropy|margin|disagreement
    prob_field = p.get("probs_field","probs")
    uncertainty_field = p.get("uncertainty_field","uncertainty")
    committee_field = p.get("committee_votes_field","committee_votes")

    def entropy(probs: List[float]) -> float:
        eps=1e-12
        return -sum(max(eps,pi)*math.log(max(eps,pi)) for pi in probs)

    scored: List[Tuple[float, dict]] = []
    for it in items:
        score = None
        if mode in ("entropy","margin"):
            probs = _get(it, prob_field) if "." in prob_field else it.get(prob_field)
            if isinstance(probs, list) and len(probs)>=2:
                ps = [float(max(1e-12, min(1-1e-12, p))) for p in probs]
                s = sum(ps); ps = [p/s for p in ps] if s > 0 else ps
                if mode=="entropy":
                    score = entropy(ps)
                else: # margin: small (p1 - p2) means hard → high score
                    top2 = sorted(ps, reverse=True)[:2]
                    margin = top2[0] - top2[1] if len(top2) >= 2 else 1.0
                    score = 1.0 - margin
        if score is None and mode=="disagreement":
            votes = _get(it, committee_field) if "." in committee_field else it.get(committee_field)
            if isinstance(votes, dict) and votes:
                counts = Counter(votes.values())
                m = counts.most_common(1)[0][1]
                score = 1.0 - (m / max(1, sum(counts.values())))
        if score is None:
            u = _get(it, uncertainty_field) if "." in uncertainty_field else it.get(uncertainty_field)
            if isinstance(u, (int,float)):
                score = float(u)
        if score is None:
            qtxt = it.get("question","") or ""
            score = len(_tokenize(qtxt))
        scored.append((float(score), it))

    # take top-n without sorting full list
    out = [it for _, it in heapq.nlargest(n, scored, key=lambda x: x[0])]
    return out, {"requested_n": n, "actual_n": len(out), "mode": mode}

def _difficulty_aware(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}

    difficulty_field = p.get("difficulty_field","difficulty")
    question_field = p.get("question_field","question")
    hard_frac = float(p.get("hard_frac",1.0))  # 0..1 fraction taken from hardest tail
    alpha = float(p.get("alpha", 1.0))  # weight for question length
    beta  = float(p.get("beta", 0.5))  # weight for entity count
    gamma = float(p.get("gamma", 1.0))  # weight to penalize lexical overlap

    def lexical_overlap(a: str, b: str) -> float:
        A, B = set(_tokenize(a)), set(_tokenize(b))
        if not A or not B: return 0.0
        return len(A & B) / len(A | B)

    scored: List[Tuple[float, dict]] = []
    for it in items:
        d = _get(it, difficulty_field) if "." in difficulty_field else it.get(difficulty_field)
        if isinstance(d, (int,float)):
            score = float(d)
        else:
            q = _get(it, question_field) if "." in question_field else it.get(question_field, "") or ""
            ans = ""
            ref = it.get("answers") or it.get("reference") or {}
            if isinstance(ref, dict) and "text" in ref: ans = str(ref["text"])
            elif isinstance(ref, list) and ref: ans = str(ref[0])
            ctxs = _get(it, "contexts")
            ctx_text = " ".join(ctxs) if isinstance(ctxs, list) else ""
            ents = [t for t in q.split() if t[:1].isupper()]
            score = alpha*len(_tokenize(q)) + beta*len(ents) - gamma*max(lexical_overlap(q, ans), lexical_overlap(q, ctx_text))
        scored.append((float(score), it))

    if hard_frac < 1.0:
        k_hard = int(round(n * hard_frac))
        hard = [it for _, it in heapq.nlargest(k_hard, scored, key=lambda x: x[0])]
        remain = n - k_hard
        # build the rest pool (exclude selected by identity)
        selected_ids = _idset(hard)
        pool = [it for _, it in scored if id(it) not in selected_ids]
        rng.shuffle(pool)
        hard.extend(pool[:max(0, remain)])
        out = hard
    else:
        out = [it for _, it in heapq.nlargest(n, scored, key=lambda x: x[0])]

    return out, {"requested_n": n, "actual_n": len(out), "hard_frac": hard_frac, "question_field": question_field}

def _temporal_recent(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}

    mode = p.get("mode","window")  # window|decay
    days = int(p.get("days", 30))
    half_life = int(p.get("half_life_days", 45))
    date_field = p.get("date_field","date")
    now = datetime.now(timezone.utc)

    dated = []
    for it in items:
        dt = _get(it, date_field) if "." in date_field else it.get(date_field)
        dtp = _parse_date(dt)
        dated.append((dtp, it))

    if mode == "window":
        cutoff = now - timedelta(days=days)
        pool = [it for dt,it in dated if dt and dt >= cutoff]
        if len(pool) < n:
            older = [it for dt,it in dated if dt and dt < cutoff] + [it for dt,it in dated if dt is None]
            rng.shuffle(older)
            pool.extend(older[:(n - len(pool))])
        rng.shuffle(pool)
        out = pool[:min(n, len(pool))]
        return out, {"requested_n": n, "actual_n": len(out), "mode": mode, "window_days": days}
    else:
        # exponential decay
        weights = []
        for dt, it in dated:
            if dt is None:
                w = 0.1
            else:
                age_days = max(0.0, (now - dt).total_seconds()/86400.0)
                w = math.exp(-math.log(2)*age_days/max(1,half_life))
            weights.append(w)
        total = sum(weights) or 1.0
        probs = [w/total for w in weights]
        idxs = _weighted_sample_without_replacement(list(range(len(items))), probs, n, rng)
        out = [items[i] for i in idxs]
        return out, {"requested_n": n, "actual_n": len(out), "mode": mode, "half_life_days": half_life}

def _unanswerable_mix(items: List[dict], p: dict, rng: random.Random):
    n = int(p["n"])
    N = len(items)
    if n >= N:
        return list(items), {"requested_n": n, "actual_n": N}

    flag_field = p.get("flag_field","is_unanswerable")
    target_ratio = float(p.get("target_ratio", 0.2))
    pos = [it for it in items if bool(_get(it, flag_field) if "." in flag_field else it.get(flag_field))]
    neg = [it for it in items if it not in pos]
    want_pos = min(len(pos), int(round(n * target_ratio)))
    want_neg = min(len(neg), n - want_pos)
    rng.shuffle(pos); rng.shuffle(neg)
    out = pos[:want_pos] + neg[:want_neg]
    if len(out) < n:
        chosen_ids = _idset(out)
        pool = [x for x in items if id(x) not in chosen_ids]
        rng.shuffle(pool)
        out.extend(pool[:(n - len(out))])
    return out, {"requested_n": n, "actual_n": len(out), "target_ratio": target_ratio, "selected_unanswerable": want_pos}

# -------------------------
# Dispatcher & Defaults
# -------------------------

AVAILABLE_METHODS: Dict[str, Dict[str, Any]] = {
    "no_sampling": {
        "fn": _no_sampling,
        "defaults": {"n": None, "seed": 42}
    },
    "uniform_random": {
        "fn": _uniform_random,
        "defaults": {"n": 100, "seed": 42}
    },
    "stratified": {
        "fn": _stratified,
        "defaults": {
            "n": 100, "seed": 42,
            "strata_keys": ["topic","answer_type"],
            "allocation": "proportional",           # proportional|equal|neyman
            "variability_field": None,              # e.g., "length" or "difficulty" (for Neyman)
            "min_per_stratum": 0,
            "question_field": "question"
        }
    },
    "diversity_kcenter": {
        "fn": _diversity_kcenter,
        "defaults": {"n": 100, "seed": 42, "embedding_field": "embedding", "text_field": "question"}
    },
    "importance_weighted": {
        "fn": _importance_weighted,
        "defaults": {"n": 100, "seed": 42, "weight_field": "weight", "floor_prob": 1e-9, "cap_ratio": 0.2}
    },
    "active_learning": {
        "fn": _active_learning,
        "defaults": {"n": 100, "seed": 42, "mode": "entropy", "probs_field": "probs", "uncertainty_field": "uncertainty", "committee_votes_field": "committee_votes"}
    },
    "difficulty_aware": {
        "fn": _difficulty_aware,
        "defaults": {"n": 100, "seed": 42, "hard_frac": 1.0, "difficulty_field": "difficulty", "question_field": "question", "alpha": 1.0, "beta": 0.5, "gamma": 1.0}
    },
    "temporal_recent": {
        "fn": _temporal_recent,
        "defaults": {"n": 100, "seed": 42, "mode": "window", "days": 30, "half_life_days": 45, "date_field": "date"}
    },
    "unanswerable_mix": {
        "fn": _unanswerable_mix,
        "defaults": {"n": 100, "seed": 42, "flag_field": "is_unanswerable", "target_ratio": 0.2}
    },
}

def list_methods() -> Dict[str, Dict[str, Any]]:
    return {k: {"defaults": v["defaults"]} for k,v in AVAILABLE_METHODS.items()}

# Core entrypoint
def run_sampler(method: str, input_path_or_list: Any, params: Optional[Dict[str,Any]] = None) -> Dict[str,Any]:
    if method not in AVAILABLE_METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(AVAILABLE_METHODS)}")
    items = _load_json_any(input_path_or_list)
    params = params or {}
    # merge defaults
    merged = dict(AVAILABLE_METHODS[method]["defaults"])
    merged.update({k:v for k,v in params.items() if v is not None})
    seed = merged.get("seed", 42)
    rng = _rng(seed)

    # optionally shuffle once for fairness (not for no_sampling)
    if method not in ("no_sampling",):
        items = list(items)  # copy
        rng.shuffle(items)

    sampled, meta = AVAILABLE_METHODS[method]["fn"](items, merged, rng)

    audit = {
        "method": method,
        "params": merged,
        "seed": seed,
        "requested_n": merged.get("n"),
        "actual_n": len(sampled) if merged.get("n") is not None else len(items),
    }
    if isinstance(meta, dict):
        audit.update(meta)
    return {"items": sampled, "audit": audit}

# -------------------------
# Optional CLI wrapper
# -------------------------

def _main_cli():
    ap = argparse.ArgumentParser(description="QnA/RAG Sampler")
    ap.add_argument("--input", required=True, help="Path to .json or .jsonl")
    ap.add_argument("--method", required=True, choices=list(AVAILABLE_METHODS.keys()))
    ap.add_argument("--n", type=int, default=None, help="Rows to sample (if applicable)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--params", type=str, default=None, help="JSON string with extra params")
    ap.add_argument("--output", type=str, default=None, help="Write sampled JSONL to this path")
    ap.add_argument("--audit", type=str, default=None, help="Write audit JSON to this path")
    args = ap.parse_args()

    user_params = {}
    if args.params:
        try:
            user_params = json.loads(args.params)
        except Exception as e:
            print(f"--params must be JSON. Error: {e}", file=sys.stderr)
            sys.exit(2)
    if args.n is not None:
        user_params["n"] = args.n
    user_params["seed"] = args.seed

    res = run_sampler(args.method, args.input, user_params)
    items = res["items"]

    if args.output:
        _save_jsonl(args.output, items)
        print(f"Wrote {len(items)} items to {args.output}")
    else:
        for obj in items:
            sys.stdout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    if args.audit:
        with open(args.audit, "w", encoding="utf-8") as f:
            json.dump(res["audit"], f, indent=2, ensure_ascii=False)
        print(f"Wrote audit to {args.audit}")

if __name__ == "__main__":
    _main_cli()
