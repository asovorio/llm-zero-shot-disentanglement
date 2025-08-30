from __future__ import annotations
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from collections import defaultdict

# ----------------------------
# Helpers shared by metrics
# ----------------------------

def _clusters_from_labels(labels: List[int]) -> List[set]:
    buckets: Dict[int, set] = defaultdict(set)
    for i, l in enumerate(labels):
        buckets[l].add(i)
    return list(buckets.values())

def _pairwise_same_cluster(labels: List[int]) -> np.ndarray:
    n = len(labels)
    arr = np.zeros((n, n), dtype=bool)
    L = np.array(labels)
    for i in range(n):
        arr[i] = (L == L[i])
    return arr

# ----------------------------
# Shen-F (gold-weighted best-F1 per gold cluster)
# ----------------------------

def shen_f_score(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Shen et al. (2006) style: for each gold conversation, take the best F1
    against any predicted conversation; average weighted by gold cluster size.
    """
    assert len(true_labels) == len(pred_labels)
    n = len(true_labels)
    if n == 0:
        return 0.0

    gold = _clusters_from_labels(true_labels)
    pred = _clusters_from_labels(pred_labels)

    gold_sizes = [len(g) for g in gold]
    pred_sizes = [len(p) for p in pred]

    total = 0.0
    for g, gsz in zip(gold, gold_sizes):
        if gsz == 0:
            continue
        best_f1 = 0.0
        for p, psz in zip(pred, pred_sizes):
            inter = len(g & p)
            if inter == 0:
                continue
            prec = inter / psz
            rec  = inter / gsz
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
        total += (gsz / n) * best_f1
    return float(total)

# ----------------------------
# VI (scaled 1 − VI / log(n))
# ----------------------------

def vi_one_minus(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Kummerfeld et al. (2019) §4.3:
      Scaled Variation of Information using the bound VI(X;Y) <= log(n).
      We return 1 - VI/log(n) in [0,1], larger is better.
    """
    n = len(true_labels)
    if n <= 1:
        return 1.0  # degenerate but well-defined

    gold = _clusters_from_labels(true_labels)
    pred = _clusters_from_labels(pred_labels)

    # Probabilities
    n_f = float(n)
    p_g = np.array([len(g)/n_f for g in gold], dtype=float)
    p_p = np.array([len(p)/n_f for p in pred], dtype=float)

    # Joint distribution
    R = np.zeros((len(gold), len(pred)), dtype=float)
    pred_sets = [set(p) for p in pred]
    for i, g in enumerate(gold):
        gi = set(g)
        for j, pj in enumerate(pred_sets):
            R[i, j] = len(gi & pj) / n_f

    def H(probs: np.ndarray) -> float:
        probs = probs[probs > 0]
        return float(-np.sum(probs * np.log(probs)))

    HX = H(p_g)
    HY = H(p_p)

    # Mutual information
    MI = 0.0
    for i in range(R.shape[0]):
        for j in range(R.shape[1]):
            rij = R[i, j]
            if rij > 0:
                MI += rij * (np.log(rij) - np.log(p_g[i]) - np.log(p_p[j]))

    VI = HX + HY - 2.0 * MI
    scaled = 1.0 - (VI / np.log(n))
    # clamp numerical noise
    if scaled < 0.0: scaled = 0.0
    if scaled > 1.0: scaled = 1.0
    return float(scaled)

# ----------------------------
# 1-1 Overlap (optimal matching)
# ----------------------------

def _hungarian_min_cost(cost: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Minimal, dependency-free Hungarian algorithm for square cost matrices.
    Returns (row_ind, col_ind). O(n^3). Assumes cost is float and finite.
    """
    cost = cost.copy().astype(float)
    n = cost.shape[0]

    # 1) Row/col reductions
    cost -= cost.min(axis=1)[:, None]
    cost -= cost.min(axis=0)[None, :]

    nrows, ncols = cost.shape
    starred = np.zeros_like(cost, dtype=bool)
    primed = np.zeros_like(cost, dtype=bool)
    cov_r = np.zeros(nrows, dtype=bool)
    cov_c = np.zeros(ncols, dtype=bool)

    # 2) Star a set of independent zeros
    for i in range(nrows):
        for j in range(ncols):
            if cost[i, j] == 0 and not cov_r[i] and not cov_c[j]:
                starred[i, j] = True
                cov_r[i] = True
                cov_c[j] = True
    cov_r[:] = False
    cov_c[:] = False

    def cover_cols_with_starred():
        cov_c[:] = starred.any(axis=0)

    cover_cols_with_starred()

    def find_uncovered_zero():
        for i in range(nrows):
            if cov_r[i]: continue
            for j in range(ncols):
                if not cov_c[j] and cost[i, j] == 0 and not primed[i, j] and not starred[i, j]:
                    return i, j
        return None

    def star_in_row(i):
        js = np.where(starred[i])[0]
        return js[0] if js.size else None

    def star_in_col(j):
        is_ = np.where(starred[:, j])[0]
        return is_[0] if is_.size else None

    def prime_in_row(i):
        js = np.where(primed[i])[0]
        return js[0] if js.size else None

    while True:
        if cov_c.sum() >= n:
            break
        z = find_uncovered_zero()
        while z is None:
            # Adjust matrix
            m = cost[~cov_r][:, ~cov_c].min()
            cost[~cov_r, :] -= m
            cost[:, cov_c] += m
            z = find_uncovered_zero()
        i, j = z
        primed[i, j] = True
        sj = star_in_row(i)
        if sj is None:
            # Augmenting path
            path = [(i, j)]
            col = j
            row = star_in_col(col)
            while row is not None:
                path.append((row, col))
                col = prime_in_row(row)
                path.append((row, col))
                row = star_in_col(col)
            for (r, c) in path:
                starred[r, c] = not starred[r, c]
            primed[:, :] = False
            cov_r[:] = False
            cov_c[:] = False
            cover_cols_with_starred()
        else:
            cov_r[i] = True
            cov_c[sj] = False

    row_ind = np.arange(n, dtype=int)
    col_ind = np.zeros(n, dtype=int)
    used = set()
    for i in range(n):
        js = np.where(starred[i])[0]
        if js.size:
            col_ind[i] = js[0]
            used.add(js[0])
        else:
            # (should be rare; fill any free column)
            for j in range(n):
                if j not in used:
                    col_ind[i] = j
                    used.add(j)
                    break
    return row_ind, col_ind

def one_to_one_overlap(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Elsner & Charniak (2008) / Kummerfeld et al. (2019):
      Maximize total overlap between gold/pred conversations with a 1-1 pairing.
      We return (sum of matched overlaps) / n in [0,1], keeping system msgs.
    """
    n = len(true_labels)
    if n == 0:
        return 0.0
    gold = _clusters_from_labels(true_labels)
    pred = _clusters_from_labels(pred_labels)

    r, c = len(gold), len(pred)
    size = max(r, c)
    W = np.zeros((size, size), dtype=float)
    pred_sets = [set(p) for p in pred]
    for i, g in enumerate(gold):
        gi = set(g)
        for j, pj in enumerate(pred_sets):
            W[i, j] = float(len(gi & pj))

    # Maximize overlap -> minimize (max - overlap)
    maxv = W.max() if W.size else 0.0
    C = (maxv - W)
    row_ind, col_ind = _hungarian_min_cost(C)

    total = 0.0
    for i, j in zip(row_ind, col_ind):
        if i < r and j < c:
            total += W[i, j]
    return float(total / n)

# ----------------------------
# Exact Match (exclude singletons)
# ----------------------------

def exact_match_prf(true_labels: List[int], pred_labels: List[int]) -> Tuple[float, float, float]:
    """
    Exact match over conversations, excluding clusters of size 1 in both gold and pred.
    Precision: (# predicted clusters that exactly equal some gold cluster) / (# predicted >=2)
    Recall:    (# gold clusters that exactly equal some predicted cluster) / (# gold >=2)
    F1:        harmonic mean.
    """
    gold = _clusters_from_labels(true_labels)
    pred = _clusters_from_labels(pred_labels)

    gold_sets = {frozenset(g) for g in gold if len(g) >= 2}
    pred_sets = [frozenset(p) for p in pred if len(p) >= 2]

    tp = sum(1 for s in pred_sets if s in gold_sets)
    P = tp / len(pred_sets) if pred_sets else 0.0
    R = tp / len(gold_sets) if gold_sets else 0.0
    F = (2 * P * R / (P + R)) if (P + R) > 0 else 0.0
    return float(P), float(R), float(F)

# ----------------------------
# Public entry
# ----------------------------

def compute_metrics(true_labels: List[int], pred_labels: List[int], metrics: List[str]) -> Dict[str, float]:
    """
    Supported keys (case/alias tolerant):
      - "nmi"                : sklearn NMI (0..1)
      - "ari"                : sklearn ARI (−1..1, usually 0..1 here)
      - "shen_f"             : Shen-style weighted F (0..1)
      - "vi"                 : scaled (1 - VI/log n), 0..1, larger is better
      - "one_to_one" | "1-1" : one-to-one overlap, 0..1
      - "exact_p", "exact_r", "exact_f" : exact conversation match (exclude singletons), 0..1
    """
    # Normalize requested metric names once
    req = {m.lower() for m in metrics}
    out: Dict[str, float] = {}

    if "nmi" in req:
        # Be explicit about averaging method
        out["nmi"] = float(normalized_mutual_info_score(true_labels, pred_labels, average_method="arithmetic"))
    if "ari" in req:
        out["ari"] = float(adjusted_rand_score(true_labels, pred_labels))

    if any(m in req for m in ("shen_f", "shen-f", "shenf")):
        out["shen_f"] = float(shen_f_score(true_labels, pred_labels))

    if "vi" in req:
        out["vi"] = float(vi_one_minus(true_labels, pred_labels))

    if ("one_to_one" in req) or ("1-1" in req) or ("one-to-one" in req):
        out["one_to_one"] = float(one_to_one_overlap(true_labels, pred_labels))

    if any(m in req for m in ("exact_p", "exact_r", "exact_f")):
        P, R, F = exact_match_prf(true_labels, pred_labels)
        if "exact_p" in req: out["exact_p"] = float(P)
        if "exact_r" in req: out["exact_r"] = float(R)
        if "exact_f" in req: out["exact_f"] = float(F)

    return out
