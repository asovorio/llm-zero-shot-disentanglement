#!/usr/bin/env python
"""
Prepare Ubuntu IRC gold JSONL with conv_id per message.

Usage (from repo root):
  python scripts/prepare_data.py \
    --ubuntu_repo data/raw/ubuntu_irc_src/repo \
    --split train \
    --out data/processed/ubuntu_irc

You can also pass an explicit file:
  python scripts/prepare_data.py --input /path/to/train.json --split train --out data/processed/ubuntu_irc

Output:
  data/processed/ubuntu_irc/ubuntu_train.jsonl   # rows with: id, author, text, timestamp, conv_id, is_system
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SYSTEM_PAT = re.compile(r"\b(join|part|quit|nick|mode|topic)\b", re.I)

# -----------------------------
# I/O helpers
# -----------------------------
def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def _list_candidates(root: Path, split: str) -> List[Path]:
    """
    Search for plausible raw files under the cloned ubuntu repo.
    We look for files with the split name in them and extensions .json/.jsonl.
    """
    pats = [f"*{split}*.json", f"*{split}*.jsonl"]
    out: List[Path] = []
    for pat in pats:
        out.extend(root.rglob(pat))
    # Heuristic: prefer shorter paths (likely top-level prepared files)
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
        # accept either a list of messages OR an object with a top-level list (e.g., {"messages":[...]} or {"data":[...]})
        if isinstance(obj, list):
            return obj
        for k in ("messages", "data", "rows", "items"):
            if k in obj and isinstance(obj[k], list):
                return obj[k]
        # Some repos store a list of "conversations", each with its own messages
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
    # Fallback: synthesize from hash of text+author+ts
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
    """
    Try to extract an explicit gold parent id, if present.
    If absent, return None (we'll rely on other link encodings if available).
    """
    for k in ("reply_to", "parent", "parent_id"):
        if k in row and row[k]:
            return str(row[k])
    # Some formats embed links as objects: {"links":[{"parent":123,"child":456},...]}
    return None

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

    # Normalize messages
    msgs: List[Dict[str, Any]] = []
    for r in rows:
        m = {
            "id": _norm_id(r),
            "author": _norm_author(r),
            "text": _norm_text(r),
            "timestamp": _norm_ts(r),
            "parent": _norm_parent(r),   # may be None
        }
        # quick system-heuristic
        txt = m["text"]
        m["is_system"] = bool(SYSTEM_PAT.search(txt)) if isinstance(txt, str) else False
        msgs.append(m)

    # Build id->index map
    id2idx = {m["id"]: i for i, m in enumerate(msgs)}

    # If there are no parent links in rows, try to discover link arrays at top-level (rare)
    # (We keep it minimal; users can extend if their copy stores links separately.)
    # Reconstruct gold conversation components via union-find on parent links
    dsu = DSU(len(msgs))
    for i, m in enumerate(msgs):
        p = m.get("parent")
        if p is None:
            continue
        if p in id2idx:
            dsu.union(i, id2idx[p])
        else:
            # unseen parent id -> ignore (could be outside the sampled window)
            pass

    # Assign conv_id = first-seen root order (stable, deterministic)
    root2cid: Dict[int, int] = {}
    next_cid = 0
    conv_id: List[int] = [0] * len(msgs)
    # sort messages deterministically for id assignment (by timestamp, then stable index)
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
                    help="Path to cloned jkkummerfeld/irc-disentanglement repo")
    ap.add_argument("--input", type=Path, default=None,
                    help="Optional direct path to a split file (.json/.jsonl)")
    ap.add_argument("--split", required=True, choices=["train", "dev", "test"])
    ap.add_argument("--out", type=Path, required=True, help="Processed output dir (e.g., data/processed/ubuntu_irc)")
    args = ap.parse_args()

    convert(args.input, args.ubuntu_repo, args.split, args.out)

if __name__ == "__main__":
    main()
