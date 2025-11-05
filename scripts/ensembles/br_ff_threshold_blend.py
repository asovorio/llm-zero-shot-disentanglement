#!/usr/bin/env python

"""
This script runs an ensemble of the FF model and our BR system.
It takes a predictions.jsonl file of the FF model that has an "x" for those message's whose parent it was <75% sure on.
Then, it takes BR's prediction only on those messages' parents with an "x" and keeps that parent as a result.
The output is a new predictions.jsonl file of the resulting ensemble.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.disentangle.config import load_config
from src.disentangle.methods.best_response import _collapse_parents
from src.disentangle.utils.io import ensure_dir, read_jsonl, write_jsonl
from src.disentangle.utils.logging import setup_logger


logger = setup_logger(__name__)


def _sanitize_parents(parents: List[Any]) -> List[int]:
    n = len(parents)
    cleaned: List[int] = []
    for i, p in enumerate(parents):
        try:
            q = int(p)
        except Exception:
            q = i
        if q < 0 or q >= n:
            q = i
        cleaned.append(q)
    return cleaned


def _resolve_threshold_parent(value: Any, fallback: int, idx: int, n: int) -> int:
    if isinstance(value, str):
        token = value.strip().lower()
        if token == "x":
            return fallback
        try:
            value = int(token)
        except Exception:
            try:
                value = int(float(token))
            except Exception:
                return fallback
    elif isinstance(value, float):
        if value.is_integer():
            value = int(value)
        else:
            return fallback

    if isinstance(value, int):
        if 0 <= value < n:
            return value
        return idx

    return fallback


def blend_parents(
    rows_best: List[Dict[str, Any]],
    rows_threshold: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    by_chunk_best: Dict[str, Dict[str, Any]] = {r["chunk_id"]: r for r in rows_best}
    by_chunk_thr: Dict[str, Dict[str, Any]] = {r["chunk_id"]: r for r in rows_threshold}

    ordered_chunk_ids = [r["chunk_id"] for r in rows_best]

    missing = [cid for cid in ordered_chunk_ids if cid not in by_chunk_thr]
    if missing:
        raise ValueError(f"Missing chunks in threshold predictions: {', '.join(missing)}")

    out_rows: List[Dict[str, Any]] = []

    for cid in ordered_chunk_ids:
        row_best = by_chunk_best[cid]
        row_thr = by_chunk_thr[cid]

        parents_best = _sanitize_parents(row_best.get("parents", []))
        parents_thr_raw = row_thr.get("parents", [])

        n = len(parents_best)
        if len(parents_thr_raw) != n:
            raise ValueError(
                f"Length mismatch for chunk {cid}: best_response={n} threshold={len(parents_thr_raw)}"
            )

        blended: List[int] = []
        for i in range(n):
            blended.append(_resolve_threshold_parent(parents_thr_raw[i], parents_best[i], i, n))

        clusters = _collapse_parents(blended)
        out_rows.append({
            "chunk_id": cid,
            "clusters": clusters,
            "num_conversations": len(set(clusters)),
            "parents": blended,
        })

    ensure_dir(output_path.parent)
    write_jsonl(output_path, out_rows)
    logger.info("Wrote blended BR/threshold predictions to %s", output_path)
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Blend Best Response parents with threshold FF predictions.")
    ap.add_argument("--config", required=True, help="Path to ensemble YAML config")
    ap.add_argument("--best-response", dest="best_response", help="Override path to best_response predictions.jsonl")
    ap.add_argument("--threshold", help="Override path to threshold FF predictions.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with Path(args.config).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    ensemble_block = raw.get("ensemble", {})

    def _resolve_path(key: str, override: str | None, fallback: str) -> Path:
        val = override or ensemble_block.get(key) or fallback
        if not val:
            raise ValueError(f"Missing path for {key}; provide in config under 'ensemble.{key}' or via CLI")
        return Path(val)

    split = cfg.run.split
    default_best = str(Path("data/results_batch") / "best_response" / "predictions.jsonl")
    default_threshold = str(Path("data/results") / "jonathans_models" / split / f"predictions-threshold-ff-{split}.jsonl")

    best_path = _resolve_path("best_response_predictions", args.best_response, default_best)
    threshold_path = _resolve_path("threshold_ff_predictions", args.threshold, default_threshold)

    rows_best = read_jsonl(best_path)
    rows_threshold = read_jsonl(threshold_path)

    output_path = Path("data/results_ensemble") / "br_ff_075" / split / "predictions.jsonl"

    out = blend_parents(rows_best, rows_threshold, output_path)
    print(str(out))


if __name__ == "__main__":
    main()
