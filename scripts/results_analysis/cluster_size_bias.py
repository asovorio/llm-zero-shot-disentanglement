#!/usr/bin/env python3

"""
This script takes in a predictions.jsonl file and for each gold cluster size (>=2), reports the average size difference
The objective is to find if the predictions are over-merging or under-merging clusters
"""

from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import json, sys

# --- config (edit if needed) ---
PRED_PATH = Path("data/results/direct_assignment/dev/predictions-run2.jsonl")
CONFIG_YAML = "configs/batch.yaml"
EXPECTED_SIZE = 50


def load_cfg_and_dataset():
    try:
        from src.disentangle.config import load_config
        from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
    except Exception as e:
        sys.exit("Import error. Run from repo root. " + str(e))
    cfg = load_config(CONFIG_YAML)
    ds = UbuntuIrcDataset(
        data_root=Path(cfg.paths.processed_dir) / "ubuntu_irc",
        split=cfg.run.split,
        chunk_size=cfg.run.chunk_size,
        seed=cfg.run.seed,
    )
    if cfg.run.chunk_size != EXPECTED_SIZE:
        print(f"[warn] CONFIG chunk_size={cfg.run.chunk_size} != EXPECTED_SIZE={EXPECTED_SIZE}")
    return ds


def read_predictions(path: Path):
    if not path.exists():
        sys.exit(f"Missing predictions file: {path}")
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln:
            continue
        rows.append(json.loads(ln))
    return rows


def label_to_sets(labels):
    """Return list of sets of indices, one per cluster label."""
    mp = defaultdict(list)
    for i, c in enumerate(labels):
        mp[int(c)].append(i)
    return [set(ixs) for ixs in mp.values()]


def main():
    ds = load_cfg_and_dataset()
    chunks = {c.chunk_id: c for c in ds.load_chunks()}
    preds = read_predictions(PRED_PATH)

    # Accumulators: per gold size -> (sum_diff, count)
    sum_diff = Counter()
    count_by_size = Counter()

    chunks_used = 0
    skipped = 0

    for row in preds:
        cid = row.get("chunk_id")
        pred = list(map(int, row.get("clusters", [])))
        ch = chunks.get(cid)
        if ch is None:
            print(f"[warn] chunk {cid} not found; skipping")
            skipped += 1
            continue
        gold = getattr(ch, "gold", None)
        if gold is None:
            sys.exit("Gold labels missing on chunk. Ensure processed export includes conv_id and loader sets ch.gold.")
        if len(pred) != len(gold):
            print(f"[warn] {cid}: pred len {len(pred)} != gold len {len(gold)}; skipping")
            skipped += 1
            continue

        gold_sets = label_to_sets(gold)
        pred_sets = label_to_sets(pred)
        if not pred_sets:
            skipped += 1
            continue

        # For each gold cluster (size>=2), find predicted cluster with max overlap
        for gset in gold_sets:
            s = len(gset)
            if s < 2:
                continue  # ignore singletons as requested
            # best predicted cluster by overlap
            best = max(pred_sets, key=lambda p: len(p & gset))
            diff = len(best) - s
            sum_diff[s] += diff
            count_by_size[s] += 1

        chunks_used += 1

    if not count_by_size:
        sys.exit("No non-singleton gold clusters found across provided chunks.")

    # Print summary
    print(f"\nChunks processed: {chunks_used}  |  Chunks skipped: {skipped}")
    print("Size  AvgDiff  Count")
    for s in sorted(count_by_size):
        n = count_by_size[s]
        avg = sum_diff[s] / n if n else 0.0
        # Negative = predicted smaller (under-split); Positive = predicted larger (over-merge)
        print(f"{str(s):<5} {avg:+7.3f}  {n}")


if __name__ == "__main__":
    main()
