#!/usr/bin/env python
"""
This script computes conversation-level precision/recall for every predictions JSONL file in a
directory using the exact evaluation logic from scripts/evaluate.py, then emits a
Matplotlib scatter plot (precision on the x-axis, recall on the y-axis).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.disentangle.utils.io import read_jsonl


def _load_evaluate_module():
    scripts_dir = Path(__file__).resolve().parent.parent
    eval_path = scripts_dir / "evaluate.py"
    spec = importlib.util.spec_from_file_location("plot_eval_module", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load evaluate module from {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


eval_mod = _load_evaluate_module()


@contextmanager
def captured_evaluate(config_path: Path, predictions_path: Path):
    """
    Run scripts.evaluate.main exactly as shipped, but capture the EvaluationReport
    instead of writing per-chunk/summary files.
    """
    holder: Dict[str, any] = {}

    def _capture(path_jsonl, path_summary, report):
        holder["report"] = report

    def _noop_ensure_dir(path):
        return Path(path)

    original_save_report = eval_mod.save_report
    original_ensure_dir = eval_mod.ensure_dir

    eval_mod.save_report = _capture
    eval_mod.ensure_dir = _noop_ensure_dir

    argv_backup = sys.argv[:]
    sys.argv = [
        "evaluate.py",
        "--config",
        str(config_path),
        "--predictions",
        str(predictions_path),
    ]

    try:
        eval_mod.main()
        if "report" not in holder:
            raise RuntimeError("Evaluation finished without producing a report.")
        yield holder["report"]
    finally:
        sys.argv = argv_backup
        eval_mod.save_report = original_save_report
        eval_mod.ensure_dir = original_ensure_dir


def extract_precision_recall(report) -> Tuple[float, float]:
    precision = report.aggregate.get("exact_p")
    recall = report.aggregate.get("exact_r")
    if precision is None or recall is None:
        raise ValueError("exact_p or exact_r missing from aggregate metrics.")
    return float(precision), float(recall)


def _mean_from_rows(rows: List[Dict], key: str, path: Path) -> float:
    total = 0.0
    count = 0
    for row in rows:
        value = row.get(key, None)
        if value is None:
            continue
        try:
            total += float(value)
            count += 1
        except (TypeError, ValueError):
            continue
    if count == 0:
        raise ValueError(f"{path.name} missing usable values for {key}.")
    return total / count


def summary_from_per_chunk(path: Path) -> Tuple[float, float, int]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path.name} is empty.")
    precision = _mean_from_rows(rows, "exact_p", path)
    recall = _mean_from_rows(rows, "exact_r", path)
    return precision, recall, len(rows)


def make_plot(results: List[Dict], output_path: Path):
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]

    ax.scatter(recalls, precisions, s=80, c="#2b6cb0", edgecolors="white", linewidth=0.6, alpha=0.9)

    ax.set_xlim(min(0.0, min(recalls) - 0.02), 1.0)
    ax.set_ylim(min(0.0, min(precisions) - 0.02), 1.0)
    ax.set_xticks([round(i * 0.1, 1) for i in range(11)])
    ax.set_yticks([round(i * 0.1, 1) for i in range(11)])
    ax.set_xlabel("Conversation-level Recall", fontsize=12)
    ax.set_ylabel("Conversation-level Precision", fontsize=12)
    ax.set_title(f"Dev Predictions: Recall vs Precision ({len(results)} files)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, format="png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot conversation-level precision vs recall.")
    parser.add_argument("--config", default="configs/default.yaml", help="YAML config used for evaluation (predictions files).")
    parser.add_argument("--pred-dir", default="data/all_predictions_dev", help="Directory containing prediction/per-chunk JSONL files.")
    parser.add_argument("--output", default="precision_recall_scatter.png", help="Path for the output PNG plot.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        raise SystemExit(f"Config file not found: {config_path}")

    pred_dir = Path(args.pred_dir)
    if not pred_dir.is_dir():
        raise SystemExit(f"Prediction directory not found: {pred_dir}")

    results = []
    notes: Dict[str, List[str]] = {}

    for pred_path in sorted(pred_dir.glob("*.jsonl")):
        try:
            name = pred_path.name
            if name.startswith("predictions-"):
                with captured_evaluate(config_path, pred_path) as report:
                    precision, recall = extract_precision_recall(report)
                    if math.isnan(precision) or math.isnan(recall):
                        raise ValueError("precision/recall evaluated to NaN")
                    chunks = len(report.per_chunk)
            elif name.startswith("per-chunk") or name.startswith("per_chunk"):
                precision, recall, chunks = summary_from_per_chunk(pred_path)
                if math.isnan(precision) or math.isnan(recall):
                    raise ValueError("precision/recall evaluated to NaN")
            else:
                raise ValueError("Unknown file naming convention; expected predictions-* or per-chunk*.")

            results.append(
                {
                    "file": pred_path.name,
                    "stem": pred_path.stem,
                    "precision": precision,
                    "recall": recall,
                    "chunks_scored": chunks,
                }
            )
        except Exception as exc:
            notes[pred_path.name] = [f"error: {exc!s}"]

    if not results:
        raise SystemExit("No usable results found in the prediction directory.")

    results.sort(key=lambda r: r["precision"], reverse=True)
    make_plot(results, Path(args.output))

    print(json.dumps(results, indent=2))
    print(f"\nSaved scatter plot to {args.output}")

    if notes:
        print("\nNotes:")
        for fname, msgs in notes.items():
            for msg in msgs:
                print(f"  - {fname}: {msg}")


if __name__ == "__main__":
    main()
