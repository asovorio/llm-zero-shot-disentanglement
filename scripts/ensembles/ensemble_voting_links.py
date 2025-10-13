#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List

import yaml

from src.disentangle.config import load_config
from src.disentangle.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.disentangle.utils.logging import setup_logger
from src.disentangle.methods.best_response import _collapse_parents


logger = setup_logger(__name__)


def _index_by_chunk(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        cid = r.get("chunk_id")
        if cid is None:
            raise ValueError("Row missing 'chunk_id'")
        out[cid] = r
    return out


def _vote_parent(p_a: int, p_b: int, p_c: int, prefer_a_on_tie: bool = True) -> int:
    """
    Majority vote across three integers. If all three differ or there is no strict
    majority, prefer the first (model A) when prefer_a_on_tie is True.
    """
    counts = Counter([p_a, p_b, p_c])
    [(top_parent, top_count)] = counts.most_common(1)
    if top_count >= 2:
        return top_parent
    # all different (1-1-1) -> tie-break
    return p_a if prefer_a_on_tie else top_parent


def build_ensemble(
    path_structure: Path,
    path_ff: Path,
    path_best_response: Path,
    split: str,
    output_path: Path,
) -> Path:
    # Read inputs
    logger.info("Reading predictions: structure=%s ff=%s best_response=%s",
                path_structure, path_ff, path_best_response)
    rows_struct = read_jsonl(path_structure)
    rows_ff = read_jsonl(path_ff)
    rows_br = read_jsonl(path_best_response)

    # Index by chunk_id and validate alignment
    idx_struct = _index_by_chunk(rows_struct)
    idx_ff = _index_by_chunk(rows_ff)
    idx_br = _index_by_chunk(rows_br)

    # Use best_response order as canonical
    ordered_chunk_ids = [r["chunk_id"] for r in rows_br]

    missing = []
    for cid in ordered_chunk_ids:
        if cid not in idx_struct:
            missing.append(("structure", cid))
        if cid not in idx_ff:
            missing.append(("ff", cid))
    if missing:
        msg = ", ".join([f"{m[0]}:{m[1]}" for m in missing])
        raise ValueError(f"Missing chunks in inputs: {msg}")

    # Vote per message parent
    out_rows: List[Dict[str, Any]] = []
    for cid in ordered_chunk_ids:
        r_s = idx_struct[cid]
        r_f = idx_ff[cid]
        r_b = idx_br[cid]

        parents_s: List[int] = r_s.get("parents", [])
        parents_f: List[int] = r_f.get("parents", [])
        parents_b: List[int] = r_b.get("parents", [])

        n = len(parents_b)
        if len(parents_s) != n or len(parents_f) != n:
            raise ValueError(
                f"Length mismatch for chunk {cid}: structure={len(parents_s)} ff={len(parents_f)} best_response={len(parents_b)}"
            )

        voted_parents: List[int] = []
        for i in range(n):
            p = _vote_parent(parents_b[i], parents_s[i], parents_f[i], prefer_a_on_tie=True)
            voted_parents.append(p)

        labels = _collapse_parents(voted_parents)
        out_rows.append({
            "chunk_id": cid,
            "clusters": labels,
            "num_conversations": len(set(labels)),
            "parents": voted_parents,
        })

    ensure_dir(output_path.parent)
    write_jsonl(output_path, out_rows)
    logger.info("Wrote ensemble predictions (split=%s) to %s", split, output_path)
    return output_path


def main():
    ap = argparse.ArgumentParser(description="Ensemble voting over parent links for Ubuntu IRC chunks.")
    ap.add_argument("--config", required=True, help="Path to ensemble YAML config")
    ap.add_argument("--structure", help="Override path to structure model predictions.jsonl")
    ap.add_argument("--ff", help="Override path to ff model predictions.jsonl")
    ap.add_argument("--best-response", dest="best_response", help="Override path to best_response predictions.jsonl")
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

    structure_path = _resolve_path("structure_predictions", args.structure)
    ff_path = _resolve_path("ff_predictions", args.ff)
    br_path = _resolve_path("best_response_predictions", args.best_response, default_br)

    split = cfg.run.split
    output_path = Path(cfg.paths.results_dir) / "voting_links" / split / "predictions.jsonl"

    out = build_ensemble(
        path_structure=structure_path,
        path_ff=ff_path,
        path_best_response=br_path,
        split=split,
        output_path=output_path,
    )
    print(str(out))


if __name__ == "__main__":
    main()

