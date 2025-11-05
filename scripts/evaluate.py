#!/usr/bin/env python

"""
This file evaluates a predictions.jsonl file with all the specified (in the YAML file) metrics.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json

from src.disentangle.config import load_config
from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.disentangle.eval.scorer import evaluate_chunks, save_report, EvaluationReport
from src.disentangle.utils.io import read_jsonl, ensure_dir
from src.disentangle.utils.logging import setup_logger

logger = setup_logger(__name__)


# helper functions for link metrics

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


def _link_sets(gold_parents: List[int], pred_parents: List[int]) -> Tuple[set, set]:
    gp = _sanitize_parents(gold_parents)
    pp = _sanitize_parents(pred_parents)
    n = min(len(gp), len(pp))
    gold_edges = {(i, gp[i]) for i in range(n) if gp[i] != i}  # exclude self-links
    pred_edges = {(i, pp[i]) for i in range(n) if pp[i] != i}  # exclude self-links
    return gold_edges, pred_edges


def _link_prf(gold_parents: List[int], pred_parents: List[int]) -> Tuple[float, float, float, int, int, int]:
    gold_edges, pred_edges = _link_sets(gold_parents, pred_parents)
    tp = len(gold_edges & pred_edges)
    p = (tp / len(pred_edges)) if pred_edges else 0.0
    r = (tp / len(gold_edges)) if gold_edges else 0.0
    f = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f), tp, len(pred_edges), len(gold_edges)


def _split_to_raw_stem(split: str) -> str:
    split = split.lower()
    if split in ("dev", "validation", "valid"):
        return "ubuntu_validation"
    elif split in ("train", "training"):
        return "ubuntu_train"
    elif split == "test":
        return "ubuntu_test"
    else:
        return "ubuntu_validation"


def _load_reply_to_map(raw_root: Path, split: str) -> Dict[int, Optional[int]]:
    """
    Load gold reply edges: message_id -> reply_to_id (or None) from raw HF export.
    We evaluate non-self link messages
    """
    stem = _split_to_raw_stem(split)
    path = raw_root / f"{stem}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Raw HF export not found: {path}")
    mp: Dict[int, Optional[int]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            mid = obj.get("id")
            if mid is None:
                continue
            try:
                mid = int(mid)
            except Exception:
                continue
            rt = obj.get("reply_to", None)
            if isinstance(rt, (int, float, str)):
                try:
                    rt = int(rt)
                except Exception:
                    rt = None
            elif rt is not None:
                rt = None
            mp[mid] = rt
    return mp


# Loader functions

def _load_gold_and_pred_ubuntu(cfg, predictions_path: Path):
    ds = UbuntuIrcDataset(
        data_root=Path(cfg.paths.processed_dir) / "ubuntu_irc",
        split=cfg.run.split,
        chunk_size=cfg.run.chunk_size,
        seed=cfg.run.seed,
    )
    chunks = ds.load_chunks()
    by_id: Dict[str, Any] = {c.chunk_id: c for c in chunks}

    pred_rows = read_jsonl(predictions_path)
    golds: List[List[int]] = []
    preds: List[List[int]] = []
    order_ids: List[str] = []

    for row in pred_rows:
        cid = row.get("chunk_id")
        ch = by_id.get(cid)
        if ch is None:
            logger.warning("Prediction for unknown chunk_id=%s (skipping)", cid)
            continue
        gold_labels = list(ch.gold) if getattr(ch, "gold", None) else None
        if not gold_labels:
            logger.warning("Gold labels missing for chunk_id=%s (skipping)", cid)
            continue

        if "clusters" in row and isinstance(row["clusters"], list):
            pred = row["clusters"]
        elif "parents" in row and isinstance(row["parents"], list):
            # Convert parents to clusters for the standard clustering metrics
            n = len(row["parents"])
            parents = _sanitize_parents(row["parents"])
            roots: Dict[int, int] = {}
            clusters = [0]*n
            next_cid = 0

            def root(i: int) -> int:
                seen = set()
                while parents[i] != i and i not in seen and 0 <= parents[i] < n:
                    seen.add(i)
                    i = parents[i]
                return i
            for i in range(n):
                r = root(i)
                if r not in roots:
                    roots[r] = next_cid; next_cid += 1
                clusters[i] = roots[r]
            pred = clusters
        else:
            logger.warning("Malformed prediction for chunk_id=%s (skipping)", cid)
            continue

        golds.append([int(x) for x in gold_labels])
        preds.append([int(x) for x in pred])
        order_ids.append(cid)

    return golds, preds, order_ids, by_id


# main function of the script

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", help="path to predictions.jsonl (defaults to results_dir/split/predictions.jsonl)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.predictions:
        predictions_path = Path(args.predictions)
    else:
        predictions_path = Path(cfg.paths.results_dir) / cfg.run.split / "predictions.jsonl"
        logger.info("--predictions not provided; defaulting to %s", predictions_path)

    dataset = cfg.run.dataset.lower()
    if dataset == "ubuntu_irc":
        golds, preds, order_ids, by_id = _load_gold_and_pred_ubuntu(cfg, predictions_path)
    else:
        raise SystemExit(f"Unknown dataset: {cfg.run.dataset}")

    # Standard clustering metrics (unchanged)
    report: EvaluationReport = evaluate_chunks(golds, preds, cfg.eval.metrics)

    # Link metrics
    pred_rows = read_jsonl(predictions_path)
    has_parents_any = any(isinstance(r.get("parents"), list) for r in pred_rows)

    if dataset == "ubuntu_irc" and has_parents_any:
        raw_root = Path(cfg.paths.data_dir) / "ubuntu_hf_export"
        mid_to_reply = _load_reply_to_map(raw_root, cfg.run.split)

        # chunk_id -> predicted parents
        pred_parents_by_id: Dict[str, List[int]] = {}
        for row in pred_rows:
            cid = row.get("chunk_id")
            if isinstance(row.get("parents"), list):
                pred_parents_by_id[cid] = _sanitize_parents(row["parents"])

        # MICRO accumulators across chunks
        tot_tp = 0
        tot_pred = 0
        tot_gold = 0

        for i, cid in enumerate(order_ids):
            pp = pred_parents_by_id.get(cid)
            if pp is None:
                continue  # DA chunk -> no link metrics

            ch = by_id[cid]
            ids = getattr(ch, "ids", None) or []

            # local gold parents (index into 0..n-1); self if gold parent not inside window
            parent_idx: List[int] = list(range(len(ids)))
            pos_by_id: Dict[int, int] = {}
            for j, mid in enumerate(ids):
                try:
                    pos_by_id[int(str(mid))] = j
                except Exception:
                    pos_by_id = {}
                    break

            if pos_by_id:
                for j, mid in enumerate(ids):
                    try:
                        m_id = int(str(mid))
                    except Exception:
                        continue
                    rt = mid_to_reply.get(m_id, None)
                    if isinstance(rt, int) and rt in pos_by_id:
                        parent_idx[j] = pos_by_id[rt]
                    else:
                        parent_idx[j] = j  # self if none/outside

            # per-chunk (macro) values for per_chunk.jsonl
            p, r, f, tp, pred_e, gold_e = _link_prf(parent_idx, pp)
            report.per_chunk[i]["link_p"] = p
            report.per_chunk[i]["link_r"] = r
            report.per_chunk[i]["link_f"] = f

            # accumulate MICRO counts for summary
            tot_tp += tp
            tot_pred += pred_e
            tot_gold += gold_e

        # MICRO summary (IBM-style dataset score). Self-link F1 omitted by request.
        if tot_pred > 0 and tot_gold > 0:
            micro_p = tot_tp / tot_pred
            micro_r = tot_tp / tot_gold
            micro_f = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
            report.aggregate["link_p"] = float(micro_p)
            report.aggregate["link_r"] = float(micro_r)
            report.aggregate["link_f"] = float(micro_f)

    # Save outputs EXACTLY as before
    results_root = Path(cfg.paths.results_dir)
    if results_root.name == "voting_links":
        out_dir = ensure_dir(results_root / "eval" / cfg.run.split)
    else:
        out_dir = ensure_dir(results_root / "eval" / cfg.run.dataset / cfg.run.split)
    save_report(out_dir / "per_chunk.jsonl", out_dir / "summary.json", report)


if __name__ == "__main__":
    main()
