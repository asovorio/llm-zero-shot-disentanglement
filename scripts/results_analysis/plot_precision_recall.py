#!/usr/bin/env python
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt

from scripts.evaluate import _sanitize_parents
from src.disentangle.config import load_config
from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.disentangle.eval.scorer import evaluate_chunks
from src.disentangle.utils.io import read_jsonl


def infer_chunk_size(pred_rows: List[Dict]) -> int:
    lengths = []
    for row in pred_rows:
        if isinstance(row.get("clusters"), list):
            lengths.append(len(row["clusters"]))
        elif isinstance(row.get("parents"), list):
            lengths.append(len(row["parents"]))
    if not lengths:
        raise ValueError("Could not infer chunk size (no clusters/parents found).")
    guess, _ = Counter(lengths).most_common(1)[0]
    return guess


def build_chunk_lookup(cfg, chunk_size: int, cache: Dict[int, Dict[str, any]]) -> Dict[str, any]:
    if chunk_size in cache:
        return cache[chunk_size]

    processed_dir = Path(cfg.paths.processed_dir)
    if not processed_dir.is_absolute():
        processed_dir = Path.cwd() / processed_dir

    ds = UbuntuIrcDataset(
        data_root=processed_dir / "ubuntu_irc",
        split=cfg.run.split,
        chunk_size=chunk_size,
        seed=cfg.run.seed,
    )
    chunks = ds.load_chunks()
    cache[chunk_size] = {chunk.chunk_id: chunk for chunk in chunks}
    return cache[chunk_size]


def clusters_from_row(row: Dict) -> List[int] | None:
    if isinstance(row.get("clusters"), list):
        return [int(x) for x in row["clusters"]]

    parents = row.get("parents")
    if isinstance(parents, list):
        parents = _sanitize_parents(parents)
        n = len(parents)
        roots: Dict[int, int] = {}
        clusters = [0] * n
        next_id = 0

        def root(i: int) -> int:
            seen = set()
            while parents[i] != i and i not in seen and 0 <= parents[i] < n:
                seen.add(i)
                i = parents[i]
            return i

        for i in range(n):
            r = root(i)
            if r not in roots:
                roots[r] = next_id
                next_id += 1
            clusters[i] = roots[r]
        return clusters
    return None


def compute_precision_recall(pred_path: Path, cfg, chunk_cache: Dict[int, Dict[str, any]]) -> Tuple[Dict, List[str]]:
    pred_rows = read_jsonl(pred_path)
    if not pred_rows:
        raise ValueError(f"{pred_path} is empty.")

    chunk_size = infer_chunk_size(pred_rows)
    chunk_lookup = build_chunk_lookup(cfg, chunk_size, chunk_cache)

    golds, preds = [], []
    issues: List[str] = []
    missing = []
    mismatched = []

    for row in pred_rows:
        cid = row.get("chunk_id")
        if not cid:
            issues.append("missing chunk_id")
            continue

        chunk = chunk_lookup.get(cid)
        if chunk is None:
            missing.append(cid)
            continue

        gold_labels = list(chunk.gold) if getattr(chunk, "gold", None) else None
        if not gold_labels:
            issues.append(f"{cid}: gold labels unavailable")
            continue

        pred_clusters = clusters_from_row(row)
        if pred_clusters is None:
            issues.append(f"{cid}: malformed prediction row")
            continue

        if len(pred_clusters) != len(gold_labels):
            mismatched.append(cid)
            continue

        golds.append(gold_labels)
        preds.append(pred_clusters)

    if missing:
        issues.append(f"{len(missing)} chunk_id(s) missing from gold lookup (e.g. {missing[:3]})")
    if mismatched:
        issues.append(f"{len(mismatched)} chunk_id(s) skipped due to length mismatch (e.g. {mismatched[:3]})")

    if not golds:
        raise ValueError(f"No aligned chunks found for {pred_path.name}")

    report = evaluate_chunks(golds, preds, cfg.eval.metrics)
    precision = report.aggregate.get("exact_p")
    recall = report.aggregate.get("exact_r")

    result = {
        "file": pred_path.name,
        "stem": pred_path.stem,
        "precision": float(precision) if precision is not None else float("nan"),
        "recall": float(recall) if recall is not None else float("nan"),
        "chunks_scored": len(golds),
        "chunk_size": chunk_size,
    }
    return result, issues


def make_plot(results: List[Dict], output_path: Path):
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    xs = [r["precision"] for r in results]
    ys = [r["recall"] for r in results]

    ax.scatter(xs, ys, s=70, c="#1f77b4", edgecolors="white", linewidth=0.7)
    for r in results:
        ax.annotate(
            r["stem"],
            (r["precision"], r["recall"]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
            ha="left",
            va="bottom",
        )

    ax.set_xlim(min(0.0, min(xs) - 0.02), 1.0)
    ax.set_ylim(min(0.0, min(ys) - 0.02), 1.0)
    ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
    ax.set_yticks([round(i * 0.1, 1) for i in range(11)])
    ax.set_xlabel("Conversation-level Precision")
    ax.set_ylabel("Conversation-level Recall")
    ax.set_title(f"Dev Predictions: Precision vs Recall ({len(results)} files)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot conversation-level precision vs recall.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config used for evaluation.")
    parser.add_argument("--pred-dir", default="data/all_predictions_dev", help="Directory containing prediction JSONL files.")
    parser.add_argument("--output", default="precision_recall_scatter.png", help="Path for the output PNG plot.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise SystemExit(f"Prediction directory not found: {pred_dir}")

    chunk_cache: Dict[int, Dict[str, any]] = {}
    results = []
    warnings: Dict[str, List[str]] = defaultdict(list)

    for pred_path in sorted(pred_dir.glob("*.jsonl")):
        try:
            result, issues = compute_precision_recall(pred_path, cfg, chunk_cache)
            results.append(result)
            if issues:
                warnings[result["file"]] = issues
        except Exception as exc:
            warnings[pred_path.name] = [f"error: {exc!s}"]

    if not results:
        raise SystemExit("No usable results found in the prediction directory.")

    results.sort(key=lambda r: r["precision"], reverse=True)
    make_plot(results, Path(args.output))

    print(json.dumps(results, indent=2))
    print(f"\nSaved scatter plot to {args.output}")

    if warnings:
        print("\nNotes:")
        for fname, msgs in warnings.items():
            for msg in msgs:
                print(f"  - {fname}: {msg}")


if __name__ == "__main__":
    main()