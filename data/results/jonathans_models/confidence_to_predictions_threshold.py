#!/usr/bin/env python3
"""
confidence_to_predictions_with_x.py

Convert a "confidence" TXT file (format with candidate ids and probabilities per message)
directly into a predictions.jsonl file matching the zero-shot system’s format, with one change:
if a message's top predicted parent probability is below a threshold (default 0.75), put the
letter "x" in that position of the "parents" array in the output (instead of an integer).

Usage:
  python confidence_to_predictions_with_x.py <input_confidence.txt> <output_predictions.jsonl> [--threshold 0.75]

Input ("confidence") expected line format (comments ok):
  <msg_id> <window_size> <cand1_id> <cand1_prob> <cand2_id> <cand2_prob> ... -
Where <msg_id> looks like: YYYY-MM-DD[_HH]:<msg_index>
Example:
  2004-11-15_03:1003 50 1002 0.9971 1003 0.0025 993 0.00005 ... -

Output ("predictions.jsonl") one JSON object per line:
  {
    "chunk_id": "YYYY-MM-DD_<start_index_zero_padded_6>",
    "clusters": [ ... 50 ints ... ],
    "num_conversations": <int>,
    "parents": [ ... 50 items, int or "x" ... ]
  }

Rules (match the previous pipeline's behavior):
- Group by day (YYYY-MM-DD), ignoring optional hour segment.
- Remap message ids so 1000->0, 1010->10, etc. Use absolute zero-based indices.
- Slice the day timeline into 50-sized windows [0..49], [50..99], etc.
- DROP any window that doesn't contain all 50 messages (strict completeness).
- Map each message's parent into within-window indices; if the top parent falls
  outside the window, fallback to self-parent (its own index).
- If the top parent probability < threshold, output "x" for that position in the
  "parents" array (string). For the purpose of forming clusters, treat "x" as a
  self-parent (conservative fallback that keeps DSU well-defined).

Note:
- Mixed types in "parents" (ints and "x") are intentional, per the user's request.
- For clustering computation, "x" is treated as self-parent.

"""
import sys
import re
import json
from collections import defaultdict
from typing import Dict, Tuple, Any, List

CONF_LINE_RE = re.compile(r"^\s*(?P<msg>[^ ]+)\s+(?P<win>\d+)\s+(?P<body>.*?)-\s*$")

DATE_PREFIX_RE = re.compile(
    r"""
    ^\s*(?P<date>\d{4}-\d{1,2}-\d{1,2})   # date like 2004-11-15 or 2005-6-27
    (?:_[0-2]?\d)?                        # optional _HH
    :(?P<idx>\d+)\s*$                     # :<msg_index>
    """,
    re.VERBOSE,
)

def normalize_date(date_str: str) -> str:
    y, m, d = date_str.split("-")
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

def parse_confidence(in_path: str) -> Dict[str, Dict[int, Tuple[int, float]]]:
    """
    Parse the confidence file.

    Returns:
      by_day: dict mapping day -> dict[msg_id(int) -> (best_parent_id(int), best_prob(float))]

    - msg_id and parent_id are the raw indices (>=1000), not yet zero-based.
    - Comments and blank lines are ignored.
    """
    by_day: Dict[str, Dict[int, Tuple[int, float]]] = defaultdict(dict)
    with open(in_path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            m = CONF_LINE_RE.match(raw)
            if not m:
                # tolerate non-matching lines
                continue
            msg_token = m.group("msg").strip()
            body = m.group("body").strip()
            dm = DATE_PREFIX_RE.match(msg_token)
            if not dm:
                # skip malformed id
                continue
            day = normalize_date(dm.group("date"))
            msg_id = int(dm.group("idx"))

            # Walk pairs in body: cand_id prob cand_id prob ... (until '-')
            parts = body.split()
            best_parent = None
            best_prob = float("-inf")
            i = 0
            while i + 1 < len(parts):
                cand_tok = parts[i]
                if cand_tok == "-":
                    break
                prob_tok = parts[i + 1]
                # Skip malformed pairs gracefully
                try:
                    cand_id = int(cand_tok)
                    prob = float(prob_tok)
                except ValueError:
                    i += 2
                    continue
                if prob > best_prob:
                    best_prob = prob
                    best_parent = cand_id
                i += 2

            # Fallbacks
            if best_parent is None:
                best_parent = msg_id  # self
                best_prob = 0.0

            by_day[day][msg_id] = (best_parent, best_prob)
    return by_day

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[self.p[x]]
        return x
    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            self.p[ra] = rb
        elif self.r[ra] > self.r[rb]:
            self.p[rb] = ra
        else:
            self.p[rb] = ra
            self.r[ra] += 1

def chunkify_and_write(by_day: Dict[str, Dict[int, Tuple[int, float]]], out_path: str, threshold: float = 0.75) -> None:
    """
    Convert per-day best parents into sliding 50-message windows and write predictions.jsonl.
    """
    def to_zero(x: int) -> int:
        return x - 1000

    with open(out_path, "w", encoding="utf-8") as out:
        for day in sorted(by_day.keys()):
            if not by_day[day]:
                continue
            # Sort message ids (raw space, >=1000)
            all_msgs = sorted(by_day[day].keys())
            zero_ids = [to_zero(m) for m in all_msgs]
            zmin, zmax = min(zero_ids), max(zero_ids)

            # Build mapping: zero_msg_id -> (zero_parent_id, best_prob)
            mapping: Dict[int, Tuple[int, float]] = {}
            for m in all_msgs:
                p_raw, prob = by_day[day][m]
                mapping[to_zero(m)] = (to_zero(p_raw), prob)

            # iterate full 50-sized windows aligned to multiples of 50
            start = zmin - (zmin % 50)
            while start + 50 <= zmax + 1:
                window_ids = list(range(start, start + 50))
                if all(z in mapping for z in window_ids):
                    # Build parents list (output) and a numeric list for DSU (treat 'x' as self)
                    out_parents: List[Any] = []
                    dsu_parents: List[int] = []

                    for i, z in enumerate(window_ids):
                        pz, prob = mapping[z]
                        # map to within-window index when possible, else fallback to self for DSU
                        if start <= pz < start + 50:
                            parent_idx = pz - start
                        else:
                            parent_idx = i  # self

                        # Output rule: if low confidence, put "x"; else integer parent_idx
                        if prob < threshold:
                            out_parents.append("x")
                            dsu_parents.append(i)  # treat as self in DSU
                        else:
                            out_parents.append(parent_idx)
                            dsu_parents.append(parent_idx)

                    # Build clusters from DSU parents
                    dsu = DSU(50)
                    for i, p in enumerate(dsu_parents):
                        dsu.union(i, p)

                    label_of_root = {}
                    clusters = []
                    next_label = 0
                    for i in range(50):
                        r = dsu.find(i)
                        if r not in label_of_root:
                            label_of_root[r] = next_label
                            next_label += 1
                        clusters.append(label_of_root[r])

                    num_conversations = next_label
                    chunk_id = f"{day}_{start:06d}"
                    obj = {
                        "chunk_id": chunk_id,
                        "clusters": clusters,
                        "num_conversations": num_conversations,
                        "parents": out_parents,
                    }
                    line = json.dumps(obj, separators=(', ', ': '))
                    out.write(line + "\n")
                start += 50

def main():
    import argparse
    ap = argparse.ArgumentParser(description="Convert confidence.txt directly to predictions.jsonl with 'x' for low-prob parents.")
    ap.add_argument("input_confidence", help="Path to ff-confidence-*.txt")
    ap.add_argument("output_jsonl", help="Path to write predictions.jsonl")
    ap.add_argument("--threshold", type=float, default=0.75, help="Probability threshold for 'x' (default: 0.75)")
    args = ap.parse_args()

    by_day = parse_confidence(args.input_confidence)
    chunkify_and_write(by_day, args.output_jsonl, threshold=args.threshold)
    print(f"Wrote: {args.output_jsonl}")

if __name__ == "__main__":
    main()
