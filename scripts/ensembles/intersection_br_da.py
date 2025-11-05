#!/usr/bin/env python

"""
This script runs an ensemble of a DR and a BR run (originally using gpt-5-mini).
It keeps only those conversations both models agree on.
It outputs a new predictions.jsonl file with the resulting conversations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any, List, Set

import yaml

from src.disentangle.config import load_config
from src.disentangle.methods.best_response import _collapse_parents
from src.disentangle.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.disentangle.utils.logging import setup_logger


logger = setup_logger(__name__)


def _sanitize_parents(parents: List[Any]) -> List[int]:
    n = len(parents)
    out: List[int] = []
    for i, p in enumerate(parents):
        try:
            q = int(p)
        except Exception:
            q = i
        if q < 0 or q >= n:
            q = i
        out.append(q)
    return out


def _canonicalize_clusters(labels: List[int]) -> List[int]:
    mapping: Dict[int, int] = {}
    next_label = 0
    out: List[int] = []
    for lbl in labels:
        if lbl not in mapping:
            mapping[lbl] = next_label
            next_label += 1
        out.append(mapping[lbl])
    return out


def _clusters_from_row(row: Dict[str, Any]) -> List[int]:
    if isinstance(row.get("clusters"), list):
        clusters = [int(c) for c in row["clusters"]]
        return _canonicalize_clusters(clusters)
    if isinstance(row.get("parents"), list):
        parents = _sanitize_parents(row["parents"])
        clusters = _collapse_parents(parents)
        return _canonicalize_clusters(clusters)
    raise ValueError("Row missing both 'clusters' and 'parents'")


def _cluster_sets(clusters: List[int]) -> List[Set[int]]:
    buckets: Dict[int, Set[int]] = {}
    for idx, lbl in enumerate(clusters):
        buckets.setdefault(lbl, set()).add(idx)
    return [buckets[k] for k in sorted(buckets.keys(), key=lambda k: min(buckets[k]))]


def _index_by_chunk(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cid = r.get("chunk_id")
        if cid is None:
            raise ValueError("Row missing 'chunk_id'")
        out[cid] = r
    return out


def build_intersection(
    best_rows: List[Dict[str, Any]],
    da_rows: List[Dict[str, Any]],
    output_path: Path,
) -> Path:
    idx_best = _index_by_chunk(best_rows)
    idx_da = _index_by_chunk(da_rows)

    ordered_chunk_ids = [r["chunk_id"] for r in best_rows]

    missing = [cid for cid in ordered_chunk_ids if cid not in idx_da]
    if missing:
        msg = ", ".join(missing)
        raise ValueError(f"Missing chunks in direct-assignment predictions: {msg}")

    out_rows: List[Dict[str, Any]] = []

    for cid in ordered_chunk_ids:
        row_b = idx_best[cid]
        row_d = idx_da[cid]

        clusters_b = _clusters_from_row(row_b)
        clusters_d = _clusters_from_row(row_d)

        n = len(clusters_b)
        if len(clusters_d) != n:
            raise ValueError(
                f"Length mismatch for chunk {cid}: direct_assignment={len(clusters_d)} best_response={n}"
            )

        cluster_sets_b = [frozenset(c) for c in _cluster_sets(clusters_b)]
        cluster_sets_d = set(frozenset(c) for c in _cluster_sets(clusters_d))

        consensus_sets = set(cluster_sets_b) & cluster_sets_d

        parents_best = _sanitize_parents(row_b.get("parents", list(range(n))))

        labels = [-1] * n
        consensus_members: Set[int] = set()
        next_label = 0

        for cluster_set in cluster_sets_b:
            if cluster_set in consensus_sets:
                for idx in sorted(cluster_set):
                    labels[idx] = next_label
                    consensus_members.add(idx)
                next_label += 1

        for i in range(n):
            if labels[i] == -1:
                labels[i] = next_label
                next_label += 1

        parents_out = [0] * n
        for i in range(n):
            if i in consensus_members:
                p = parents_best[i]
                if not (0 <= p < n) or labels[p] != labels[i]:
                    p = i
                parents_out[i] = p
            else:
                parents_out[i] = i

        labels = _canonicalize_clusters(labels)
        out_rows.append({
            "chunk_id": cid,
            "clusters": labels,
            "num_conversations": len(set(labels)),
            "parents": parents_out,
        })

    ensure_dir(output_path.parent)
    write_jsonl(output_path, out_rows)
    logger.info("Wrote BR/DA intersection ensemble predictions to %s", output_path)
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Intersection between Best Response and Direct Assignment clusters.")
    ap.add_argument("--config", required=True, help="Path to ensemble YAML config")
    ap.add_argument("--best-response", dest="best_response", help="Path to best_response predictions.jsonl")
    ap.add_argument("--direct-assignment", dest="direct_assignment", help="Path to direct_assignment predictions.jsonl")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with Path(args.config).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    ensemble_block = raw.get("ensemble", {})

    def _resolve_path(key: str, override: str | None, default: str | None = None) -> Path:
        val = override or ensemble_block.get(key) or default
        if not val:
            raise ValueError(f"Missing path for {key}; specify in config under 'ensemble.{key}' or via CLI override")
        return Path(val)

    default_br = str(Path("data/results_batch") / "best_response" / "predictions.jsonl")
    default_da = str(Path("data/results") / "direct_assignment" / cfg.run.split / "predictions.jsonl")

    br_path = _resolve_path("best_response_predictions", args.best_response, default_br)
    da_path = _resolve_path("direct_assignment_predictions", args.direct_assignment, default_da)

    split = cfg.run.split
    output_path = Path(cfg.paths.results_dir) / split / "predictions.jsonl"

    rows_br = read_jsonl(br_path)
    rows_da = read_jsonl(da_path)

    out = build_intersection(
        best_rows=rows_br,
        da_rows=rows_da,
        output_path=output_path,
    )
    print(str(out))


if __name__ == "__main__":
    main()
