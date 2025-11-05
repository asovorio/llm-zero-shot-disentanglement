#!/usr/bin/env python
"""
This file prepares Ubuntu IRC gold JSONL with conv_id per message for a split of data (train, test or dev)

Example usage from repo root:
  python scripts/prepare_data.py --input data/raw/ubuntu_hf_export/ubuntu_test.jsonl --split test --out data/processed/ubuntu_irc

Output:
  data/processed/ubuntu_irc/ubuntu_<split>.jsonl
  Resulting rows contain: id, author, text, timestamp, conv_id, is_system, is_context
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

SYSTEM_PAT = re.compile(r"\b(join|part|quit|nick|mode|topic)\b", re.I)


# -----------------------------
# I/O helpers
# -----------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _list_candidates(root: Path, split: str) -> List[Path]:
    """Search for plausible raw files under a repo; prefer shorter paths."""
    pats = [f"*{split}*.json", f"*{split}*.jsonl"]
    out: List[Path] = []
    for pat in pats:
        out.extend(root.rglob(pat))
    out.sort(key=lambda p: (len(p.parts), p.name))
    return out


def _read_json_or_jsonl(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    else:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return obj
        for k in ("messages", "data", "rows", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        if "conversations" in obj and isinstance(obj["conversations"], list):
            rows = []
            for conv in obj["conversations"]:
                msgs = conv.get("messages", [])
                if isinstance(msgs, list):
                    rows.extend(msgs)
            return rows
        raise ValueError(f"Don't know how to read structure in {path}")


# -----------------------------
# Field normalization
# -----------------------------
def _norm_author(row: Dict[str, Any]) -> str:
    for k in ("author", "user", "username", "speaker", "nick"):
        if k in row and row[k] is not None:
            return str(row[k])
    return "UNK"


def _norm_text(row: Dict[str, Any]) -> str:
    for k in ("text", "body", "message", "content"):
        if k in row and row[k] is not None:
            return str(row[k])
    return ""


def _norm_id(row: Dict[str, Any]) -> str:
    for k in ("id", "mid", "message_id"):
        if k in row and row[k] is not None:
            return str(row[k])
    a = _norm_author(row)
    t = _norm_text(row)
    ts = _norm_ts(row)
    return f"{hash((a,t,ts)) & 0xffffffff}"


def _norm_ts(row: Dict[str, Any]) -> float:
    for k in ("timestamp", "ts", "time", "t"):
        if k in row and row[k] is not None:
            try:
                return float(row[k])
            except Exception:
                pass
    return 0.0


def _norm_parent(row: Dict[str, Any]) -> Optional[str]:
    """Extract a single explicit parent id if present; else None."""
    for k in ("reply_to", "parent", "parent_id"):
        if k in row and row[k] not in (None, "", 0):
            return str(row[k])
    return None


def _norm_is_system(row: Dict[str, Any], text_fallback: str) -> bool:
    if "is_system" in row:
        try:
            return bool(row["is_system"])
        except Exception:
            pass
    return bool(SYSTEM_PAT.search(text_fallback)) if isinstance(text_fallback, str) else False


def _norm_is_context(row: Dict[str, Any]) -> bool:
    # Pass through from your HF export; default False if missing
    try:
        return bool(row.get("is_context", False))
    except Exception:
        return False


# -----------------------------
# Union-Find for gold threads
# -----------------------------
class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[rb] < self.rank[ra]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# -----------------------------
# Main conversion
# -----------------------------
def convert(
    input_path: Optional[Path],
    ubuntu_repo: Optional[Path],
    split: str,
    out_dir: Path
) -> Path:
    if input_path is None:
        if ubuntu_repo is None:
            raise SystemExit("--ubuntu_repo or --input required")
        cands = _list_candidates(ubuntu_repo, split)
        if not cands:
            raise SystemExit(f"No candidate files for split={split} under {ubuntu_repo}")
        input_path = cands[0]

    rows = _read_json_or_jsonl(input_path)

    # Normalize messages (pass through is_context and prefer provided is_system)
    msgs: List[Dict[str, Any]] = []
    for r in rows:
        text = _norm_text(r)
        m = {
            "id": _norm_id(r),
            "author": _norm_author(r),
            "text": text,
            "timestamp": _norm_ts(r),
            "parent": _norm_parent(r),         # may be None
            "is_system": _norm_is_system(r, text),
            "is_context": _norm_is_context(r),
        }
        msgs.append(m)

    # Build id->index map
    id2idx = {m["id"]: i for i, m in enumerate(msgs)}

    # Reconstruct gold conversation components via union-find on parent links
    dsu = DSU(len(msgs))
    for i, m in enumerate(msgs):
        p = m.get("parent")
        if p is None:
            continue
        j = id2idx.get(p)
        if j is not None:
            dsu.union(i, j)

    # Assign conv_id = first-seen root order (stable, deterministic)
    root2cid: Dict[int, int] = {}
    next_cid = 0
    conv_id: List[int] = [0] * len(msgs)
    order = sorted(range(len(msgs)), key=lambda i: (msgs[i]["timestamp"], i))
    for i in order:
        r = dsu.find(i)
        if r not in root2cid:
            root2cid[r] = next_cid
            next_cid += 1
        conv_id[i] = root2cid[r]

    # Persist JSONL
    _ensure_dir(out_dir)
    out_file = out_dir / f"ubuntu_{split}.jsonl"
    with out_file.open("w", encoding="utf-8") as f:
        for i, m in enumerate(msgs):
            row = {
                "id": m["id"],
                "author": m["author"],
                "text": m["text"],
                "timestamp": m["timestamp"],
                "conv_id": conv_id[i],
                "is_system": m["is_system"],
                "is_context": m["is_context"],   # <-- NEW
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(msgs)} messages to {out_file}")
    return out_file


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ubuntu_repo", type=Path, default=None,
                    help="(Optional) Path to cloned jkkummerfeld/irc-disentanglement repo")
    ap.add_argument("--input", type=Path, default=None,
                    help="Direct path to a split file (.json/.jsonl); e.g., data/raw/ubuntu_hf_export/ubuntu_test.jsonl")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out", type=Path, required=True, help="Processed output dir (e.g., data/processed/ubuntu_irc)")
    args = ap.parse_args()

    convert(args.input, args.ubuntu_repo, args.split, args.out)


if __name__ == "__main__":
    main()
