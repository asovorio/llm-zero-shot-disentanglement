#!/usr/bin/env python
"""
Compute conversation-level precision/recall for each predictions/per-chunk JSONL file
using scripts/evaluate.py where appropriate, then plot recall (x) vs precision (y)
with point colors encoding the per-file cost.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import importlib.util
import os
import sys
import tempfile
from contextlib import contextmanager

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle

FILE_PATH = Path(__file__).resolve()
SCRIPTS_DIR = FILE_PATH.parent.parent
ROOT_DIR = SCRIPTS_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.disentangle.utils.io import read_jsonl


def _load_evaluate_module():
    eval_path = SCRIPTS_DIR / "evaluate.py"
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


def prepare_predictions_path(path: Path) -> Tuple[Path, float | None, Path | None]:
    with path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    cost = None
    trimmed_lines = lines
    if lines:
        try:
            last_obj = json.loads(lines[-1])
        except Exception:
            last_obj = None
        if isinstance(last_obj, dict) and "cost" in last_obj:
            cost = float(last_obj["cost"])
            trimmed_lines = lines[:-1]

    if trimmed_lines is lines:
        return path, cost, None

    fd, tmp_path = tempfile.mkstemp(suffix=path.suffix)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            tmp.writelines(trimmed_lines)
    except Exception:
        os.close(fd)
        os.remove(tmp_path)
        raise
    return Path(tmp_path), cost, Path(tmp_path)


def summary_from_per_chunk(path: Path) -> Tuple[float, float, int, float | None]:
    rows = read_jsonl(path)
    if not rows:
        raise ValueError(f"{path.name} is empty.")
    cost = None
    if isinstance(rows[-1], dict) and "cost" in rows[-1]:
        cost = float(rows[-1]["cost"])
        rows = rows[:-1]
    metric_rows = [row for row in rows if row.get("exact_p") is not None and row.get("exact_r") is not None]
    if not metric_rows:
        raise ValueError(f"{path.name} has no per-chunk metric entries.")
    precision = _mean_from_rows(metric_rows, "exact_p", path)
    recall = _mean_from_rows(metric_rows, "exact_r", path)
    return precision, recall, len(metric_rows), cost


def make_plot(results: List[Dict], output_path: Path):
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        plt.style.use("seaborn-whitegrid")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    recalls = [r["recall"] for r in results]
    precisions = [r["precision"] for r in results]
    costs = [r["cost"] for r in results]

    special_cost = 10.5
    idx_other = [i for i, c in enumerate(costs) if not math.isclose(c, special_cost, rel_tol=1e-9, abs_tol=1e-9)]
    idx_special = [i for i, c in enumerate(costs) if math.isclose(c, special_cost, rel_tol=1e-9, abs_tol=1e-9)]

    cmap = LinearSegmentedColormap.from_list("cost_map", ["#2ca02c", "#ffffff", "#d62728"])
    if idx_other:
        other_costs = [costs[i] for i in idx_other]
        min_cost = min(other_costs)
        max_cost = max(other_costs)
        if math.isclose(max_cost, min_cost):
            norm = Normalize(vmin=min_cost - 1e-6, vmax=max_cost + 1e-6)
        else:
            norm = Normalize(vmin=min_cost, vmax=max_cost)
        scatter_other = ax.scatter(
            [recalls[i] for i in idx_other],
            [precisions[i] for i in idx_other],
            s=80,
            c=[costs[i] for i in idx_other],
            cmap=cmap,
            norm=norm,
            edgecolors="white",
            linewidth=0.6,
            alpha=0.9,
        )
    else:
        scatter_other = None
        norm = Normalize(vmin=0, vmax=1)

    if idx_special:
        ax.scatter(
            [recalls[i] for i in idx_special],
            [precisions[i] for i in idx_special],
            s=80,
            c="#7b3294",
            edgecolors="white",
            linewidth=0.6,
            alpha=0.95,
        )

    ax.set_xlim(0.2, 0.7)
    ax.set_ylim(0.2, 0.8)
    ax.set_xticks([round(x, 1) for x in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
    ax.set_yticks([round(y, 1) for y in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]])
    ax.set_xlabel("Conversation-level Recall", fontsize=12)
    ax.set_ylabel("Conversation-level Precision", fontsize=12)
    ax.set_title(f"Recall vs Performance in Dev Split ({len(results)} models)")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if scatter_other is not None:
        cbar = fig.colorbar(scatter_other, ax=ax, pad=0.02)
        cbar.set_label("Cost", fontsize=11)
        cbar.ax.tick_params(labelsize=9)
        if idx_special:
            cbar.ax.add_patch(
                Rectangle((0.02, 1.005), 0.96, 0.03, transform=cbar.ax.transAxes, color="#9552b4", clip_on=False)
            )
            cbar.ax.text(
                1.02,
                1.02,
                "10.50",
                transform=cbar.ax.transAxes,
                ha="left",
                va="center",
                fontsize=9,
            )
    ax.grid(True, linestyle="--", alpha=0.35)

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
                eval_path, cost, temp_path = prepare_predictions_path(pred_path)
                try:
                    with captured_evaluate(config_path, eval_path) as report:
                        precision, recall = extract_precision_recall(report)
                        if math.isnan(precision) or math.isnan(recall):
                            raise ValueError("precision/recall evaluated to NaN")
                        chunks = len(report.per_chunk)
                        if cost is None:
                            cost = 0.0
                finally:
                    if temp_path is not None and temp_path.exists():
                        temp_path.unlink(missing_ok=True)
            elif name.startswith("per-chunk") or name.startswith("per_chunk"):
                precision, recall, chunks, cost = summary_from_per_chunk(pred_path)
                if math.isnan(precision) or math.isnan(recall):
                    raise ValueError("precision/recall evaluated to NaN")
                if cost is None:
                    cost = 0.0
            else:
                raise ValueError("Unknown file naming convention; expected predictions-* or per-chunk*.")

            results.append(
                {
                    "file": pred_path.name,
                    "stem": pred_path.stem,
                    "precision": precision,
                    "recall": recall,
                    "chunks_scored": chunks,
                    "cost": cost,
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
