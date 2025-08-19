#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any
from src.disentangle.config import load_config
from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.disentangle.datasets.movie_dialogue import MovieDialogueDataset
from src.disentangle.eval.scorer import evaluate_chunks, save_report
from src.disentangle.utils.io import read_jsonl, ensure_dir
from src.disentangle.utils.logging import setup_logger

logger = setup_logger(__name__)

def _load_gold_and_pred_ubuntu(cfg, predictions_path: Path):
    ds = UbuntuIrcDataset(
        data_root=Path(cfg.paths.processed_dir) / "ubuntu_irc",
        split=cfg.run.split,
        chunk_size=cfg.run.chunk_size,
        seed=cfg.run.seed,
    )
    chunks = ds.load_chunks()
    pred_rows = read_jsonl(predictions_path)
    golds, preds = [], []
    by_id = {c.chunk_id: c for c in chunks}
    for row in pred_rows:
        cid = row["chunk_id"]
        ch = by_id[cid]
        pred = row["clusters"] if "clusters" in row else parents_to_clusters(row["parents"])
        # If no gold labels available in loader, you need to map to gold provided by your processed export
        if ch.gold is None:
            raise SystemExit("Gold labels for Ubuntu IRC not found. Ensure your export includes conv_id per message.")
        golds.append(ch.gold)
        preds.append(pred)
    return golds, preds

def _load_gold_and_pred_movie(cfg, predictions_path: Path):
    ds = MovieDialogueDataset(
        data_root=(Path(cfg.paths.data_dir) / "movie_dialogue_src" / "repo" / "dataset"),
        split=cfg.run.split,
    )
    diags = ds.load_dialogues()
    pred_rows = read_jsonl(predictions_path)
    golds, preds = [], []
    by_id = {d.did: d for d in diags}
    for row in pred_rows:
        did = row["chunk_id"]  # we reuse 'chunk_id' field to store dialogue id here
        d = by_id[did]
        pred = row["clusters"]
        golds.append(d.gold)
        preds.append(pred)
    return golds, preds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True, help="path to predictions.jsonl")
    args = ap.parse_args()
    cfg = load_config(args.config)

    predictions_path = Path(args.predictions)
    if cfg.run.dataset.lower() == "ubuntu_irc":
        golds, preds = _load_gold_and_pred_ubuntu(cfg, predictions_path)
    elif cfg.run.dataset.lower() == "movie_dialogue":
        golds, preds = _load_gold_and_pred_movie(cfg, predictions_path)
    else:
        raise SystemExit(f"Unknown dataset: {cfg.run.dataset}")

    report = evaluate_chunks(golds, preds, cfg.eval.metrics)

    out_dir = ensure_dir(Path(cfg.paths.results_dir) / "eval" / cfg.run.dataset / cfg.run.split)
    save_report(out_dir / "per_chunk.jsonl", out_dir / "summary.json", report)

def parents_to_clusters(parents: List[int]) -> List[int]:
    n = len(parents)
    roots, cluster = {}, [-1] * n
    cid = 0
    def root(i):
        seen = set()
        while parents[i] != i:
            if i in seen:
                break
            seen.add(i)
            i = parents[i]
        return i
    for i in range(n):
        r = root(i)
        if r not in roots:
            roots[r] = cid
            cid += 1
        cluster[i] = roots[r]
    return cluster

if __name__ == "__main__":
    main()
