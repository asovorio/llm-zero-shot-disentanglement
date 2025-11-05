#!/usr/bin/env python

"""
This script runs an ensemble of several DA and BR predictions.jsonl files and keeps the intersection of their conversation clusterings
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Sequence, Set, Tuple

import yaml

from src.disentangle.config import load_config
from src.disentangle.methods.best_response import _collapse_parents
from src.disentangle.utils.io import ensure_dir, read_jsonl, write_jsonl
from src.disentangle.utils.logging import setup_logger


logger = setup_logger(__name__)


def _sanitize_parents(parents: Sequence[Any]) -> List[int]:
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


def _canonicalize_clusters(labels: List[int]) -> List[int]:
    mapping: Dict[int, int] = {}
    out: List[int] = []
    next_label = 0
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


def _cluster_sets(clusters: List[int]) -> List[FrozenSet[int]]:
    buckets: Dict[int, Set[int]] = {}
    for idx, lbl in enumerate(clusters):
        buckets.setdefault(lbl, set()).add(idx)
    return [frozenset(buckets[k]) for k in sorted(buckets.keys(), key=lambda k: min(buckets[k]))]


def _index_by_chunk(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cid = row.get("chunk_id")
        if cid is None:
            raise ValueError("Row missing 'chunk_id'")
        out[cid] = row
    return out


def _build_consensus_row(
    chunk_id: str,
    base_row: Dict[str, Any],
    cluster_sets_runs: List[Set[FrozenSet[int]]],
) -> Dict[str, Any]:
    clusters_base = _clusters_from_row(base_row)
    base_sets = _cluster_sets(clusters_base)
    n = len(clusters_base)

    consensus_sets = set.intersection(*cluster_sets_runs) if cluster_sets_runs else set()

    parents_base = _sanitize_parents(base_row.get("parents", list(range(n))))

    labels = [-1] * n
    consensus_members: Set[int] = set()
    next_label = 0

    for cluster_set in base_sets:
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
            p = parents_base[i]
            if not (0 <= p < n) or labels[p] != labels[i]:
                p = i
            parents_out[i] = p
        else:
            parents_out[i] = i

    labels = _canonicalize_clusters(labels)
    return {
        "chunk_id": chunk_id,
        "clusters": labels,
        "num_conversations": len(set(labels)),
        "parents": parents_out,
    }


def _load_runs(paths: Sequence[Path]) -> List[List[Dict[str, Any]]]:
    return [read_jsonl(path) for path in paths]


def _validate_paths(paths: Sequence[Path], label: str) -> None:
    if not paths:
        raise ValueError(f"No paths provided for {label} runs")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"{label} predictions not found: {p}")


def build_ensembles(
    br_paths: Sequence[Path],
    da_paths: Sequence[Path],
    split: str,
    output_root: Path,
) -> Tuple[Path, Path, Path]:
    _validate_paths(br_paths, "best_response")
    _validate_paths(da_paths, "direct_assignment")

    br_runs = _load_runs(br_paths)
    da_runs = _load_runs(da_paths)

    # Use first BR run as canonical ordering
    ordered_chunk_ids = [row["chunk_id"] for row in br_runs[0]]

    # Index runs for quick chunk access
    idx_br = [_index_by_chunk(rows) for rows in br_runs]
    idx_da = [_index_by_chunk(rows) for rows in da_runs]

    # sanity: ensure all runs cover same chunks
    for cid in ordered_chunk_ids:
        for j, idx in enumerate(idx_br, start=1):
            if cid not in idx:
                raise ValueError(f"Chunk {cid} missing in BR run {j}")
        for j, idx in enumerate(idx_da, start=1):
            if cid not in idx:
                raise ValueError(f"Chunk {cid} missing in DA run {j}")

    out_da6: List[Dict[str, Any]] = []
    out_br3: List[Dict[str, Any]] = []
    out_da3: List[Dict[str, Any]] = []

    for cid in ordered_chunk_ids:
        # Precompute cluster sets per run
        br_sets = []
        for idx in idx_br:
            clusters = _clusters_from_row(idx[cid])
            br_sets.append(set(_cluster_sets(clusters)))

        da_sets = []
        for idx in idx_da:
            clusters = _clusters_from_row(idx[cid])
            da_sets.append(set(_cluster_sets(clusters)))

        # Intersection across all 6 runs
        row_all = _build_consensus_row(
            cid,
            base_row=idx_br[0][cid],
            cluster_sets_runs=br_sets + da_sets,
        )
        out_da6.append(row_all)

        # Intersection across BR runs (3)
        row_br = _build_consensus_row(
            cid,
            base_row=idx_br[0][cid],
            cluster_sets_runs=br_sets,
        )
        out_br3.append(row_br)

        # Intersection across DA runs (3)
        row_da = _build_consensus_row(
            cid,
            base_row=idx_da[0][cid],
            cluster_sets_runs=da_sets,
        )
        out_da3.append(row_da)

    out_dir = ensure_dir(output_root / split)

    path_da6 = out_dir / "predictions-br-da-6.jsonl"
    path_br3 = out_dir / "predictions-br-3.jsonl"
    path_da3 = out_dir / "predictions-da-3.jsonl"

    write_jsonl(path_da6, out_da6)
    write_jsonl(path_br3, out_br3)
    write_jsonl(path_da3, out_da3)

    logger.info("Wrote combined intersection predictions to %s", out_dir)

    return path_da6, path_br3, path_da3


def main():
    ap = argparse.ArgumentParser(description="Intersect BR/DA runs and persist consensus predictions.")
    ap.add_argument("--config", required=True, help="Path to ensemble YAML config")
    ap.add_argument("--br-runs", nargs="+", help="Override list of Best Response prediction files")
    ap.add_argument("--da-runs", nargs="+", help="Override list of Direct Assignment prediction files")
    ap.add_argument("--output-root", help="Override output root directory (default data/results_ensemble/br-da-gpt-5)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    with Path(args.config).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    ensemble_block = raw.get("ensemble", {})

    br_paths = args.br_runs or ensemble_block.get("br_runs")
    da_paths = args.da_runs or ensemble_block.get("da_runs")
    if br_paths is None or da_paths is None:
        raise ValueError("Provide BR and DA run paths via config ensemble.br_runs / ensemble.da_runs or CLI overrides")

    br_paths = [Path(p) for p in br_paths]
    da_paths = [Path(p) for p in da_paths]

    output_root = Path(args.output_root) if args.output_root else Path("data/results_ensemble") / "br-da-gpt-5"

    path_da6, path_br3, path_da3 = build_ensembles(
        br_paths=br_paths,
        da_paths=da_paths,
        split=cfg.run.split,
        output_root=output_root,
    )

    print(path_da6)
    print(path_br3)
    print(path_da3)


if __name__ == "__main__":
    main()

