#!/usr/bin/env python3
# scripts/cluster_ace_histogram.py
# Prints, for each cluster size s>=2:  (# exact-matched gold clusters of size s) / (total gold clusters of size s) >>> %
from __future__ import annotations
from pathlib import Path
from collections import defaultdict, Counter
import json, sys

# --- config (edit if needed) ---
PRED_PATH     = Path("data/results/direct_assignment/dev/predictions.jsonl")
CONFIG_YAML   = "configs/default.yaml"
EXPECTED_SIZE = 50
# --------------------------------

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
        if not ln: continue
        rows.append(json.loads(ln))
    return rows

def index_sets(labels):
    """Return list of frozenset index sets (one per cluster label)."""
    mp = defaultdict(list)
    for i, c in enumerate(labels):
        mp[int(c)].append(i)
    return [frozenset(ixs) for ixs in mp.values()]

def bin_label(size: int) -> str:
    if 2 <= size <= 15: return str(size)
    if 16 <= size <= 20: return "16-20"
    return "21+"

def main():
    ds = load_cfg_and_dataset()
    chunks = {c.chunk_id: c for c in ds.load_chunks()}
    preds = read_predictions(PRED_PATH)

    total_by_bin = Counter()  # bin -> count of gold clusters (size>=2)
    aced_by_bin  = Counter()  # bin -> count of exactly matched gold clusters
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
            sys.exit("Gold labels missing on chunk. Ensure export includes conv_id and loader sets ch.gold.")
        if len(pred) != len(gold):
            print(f"[warn] {cid}: pred len {len(pred)} != gold len {len(gold)}; skipping")
            skipped += 1
            continue

        gold_sets = index_sets(gold)
        pred_sets = set(index_sets(pred))

        for gset in gold_sets:
            s = len(gset)
            if s < 2:  # exclude singletons
                continue
            b = bin_label(s)
            total_by_bin[b] += 1
            if gset in pred_sets:
                aced_by_bin[b] += 1

        chunks_used += 1

    if not total_by_bin:
        sys.exit("No non-singleton gold clusters found in the provided chunks.")

    # Print summary
    total_gold_ns = sum(total_by_bin.values())
    total_aced = sum(aced_by_bin.values())
    pct_total = (100.0 * total_aced / total_gold_ns) if total_gold_ns else 0.0

    print(f"\nChunks processed: {chunks_used}  |  Chunks skipped: {skipped}")
    print(f"Aced non-singleton clusters (overall): {total_aced} / {total_gold_ns}  >>> {pct_total:.2f}%\n")

    # Order: 2..15, then 16-20, then 21+
    order = [str(s) for s in range(2,16)] + ["16-20", "21+"]
    print("Size  Count")
    for b in order:
        t = total_by_bin.get(b, 0)
        if t == 0:
            continue  # skip bins that don't appear
        a = aced_by_bin.get(b, 0)
        pct = (100.0 * a / t) if t else 0.0
        print(f"{b:<5} {a} / {t}  >>> {pct:.2f}%")

if __name__ == "__main__":
    main()