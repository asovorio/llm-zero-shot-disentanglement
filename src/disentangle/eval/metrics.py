from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

def _pairwise_same_cluster(labels: List[int]) -> np.ndarray:
    n = len(labels)
    arr = np.zeros((n, n), dtype=bool)
    L = np.array(labels)
    for i in range(n):
        arr[i] = (L == L[i])
    return arr

def shen_f_score(true_labels: List[int], pred_labels: List[int]) -> float:
    """
    Pairwise same-cluster F1 (commonly referred to as Shen-F in disentanglement papers).
    """
    assert len(true_labels) == len(pred_labels)
    T = _pairwise_same_cluster(true_labels)
    P = _pairwise_same_cluster(pred_labels)
    # Only consider upper triangle (i<j), ignore diagonal
    iu = np.triu_indices(len(true_labels), k=1)
    tp = np.logical_and(T[iu], P[iu]).sum()
    fp = np.logical_and(~T[iu], P[iu]).sum()
    fn = np.logical_and(T[iu], ~P[iu]).sum()
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec  = tp / (tp + fn) if (tp + fn) else 0.0
    if (prec + rec) == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)

def compute_metrics(true_labels: List[int], pred_labels: List[int], metrics: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if "nmi" in metrics:
        out["nmi"] = float(normalized_mutual_info_score(true_labels, pred_labels))
    if "ari" in metrics:
        out["ari"] = float(adjusted_rand_score(true_labels, pred_labels))
    if "shen_f" in metrics or "shen-f" in metrics or "shenf" in metrics:
        out["shen_f"] = float(shen_f_score(true_labels, pred_labels))
    return out
