from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List
import numpy as np
from .metrics import compute_metrics
from ..utils.io import write_jsonl, save_json
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


@dataclass
class EvaluationReport:
    per_chunk: List[Dict[str, Any]]
    aggregate: Dict[str, float]


def evaluate_chunks(golds: List[List[int]], preds: List[List[int]], metrics: List[str]) -> EvaluationReport:
    per_chunk: List[Dict[str, Any]] = []
    mats: Dict[str, List[float]] = {m: [] for m in metrics}
    for g, p in zip(golds, preds):
        m = compute_metrics(g, p, metrics)
        per_chunk.append(m)
        for k, v in m.items():
            mats[k].append(v)
    agg = {k: float(np.mean(v)) if v else 0.0 for k, v in mats.items()}
    return EvaluationReport(per_chunk=per_chunk, aggregate=agg)


def save_report(path_jsonl: str, path_summary: str, report: EvaluationReport) -> None:
    write_jsonl(path_jsonl, report.per_chunk)
    save_json(path_summary, report.aggregate)
    logger.info("Saved per-chunk metrics to %s and summary to %s", path_jsonl, path_summary)
