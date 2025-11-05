#!/usr/bin/env python3
"""
top_preds_to_predictions.py

Convert a "top-preds" TXT file (format: `YYYY-MM-DD[_HH]:<msg_id> <parent_id> -`)
into a predictions.jsonl file matching the zero-shot system’s format:

Each JSON line:
  {
    "chunk_id": "YYYY-MM-DD_<start_index_zero_padded_6>",
    "clusters": [ ... 50 ints ... ],
    "num_conversations": <int>,
    "parents": [ ... 50 ints ... ]
  }

Rules:
- Messages are grouped per day (YYYY-MM-DD), ignoring any hour segment if present.
- Message IDs in the TXT start at 1000; remap so 1000→0, 1010→10, etc.
- Sort by message id ascending within each day.
- Use sliding 50-message chunks: [0..49], [50..99], etc.
- DROP the final chunk of a day if it contains < 50 messages.
- Parents are remapped to within-chunk indices; if a parent falls outside the chunk window,
  fallback to self-parent.
- Clusters are induced by parent links (union-find), labeled by first occurrence order.
- Output uses spaces after commas (pretty JSON arrays like the reference).
"""

import sys
import re
import json
from collections import defaultdict

LINE_RE = re.compile(
    r"""
    ^\s*
    (?P<date>\d{4}-\d{1,2}-\d{1,2})        # date like 2005-6-27 or 2004-11-15
    (?:_[0-2]?\d)?                         # optional _HH (e.g., _03 or _3)
    :                                      # colon before msg id
    (?P<msg>\d+)\s+(?P<parent>\d+)\s+-\s*$
    """,
    re.VERBOSE,
)

def normalize_date(date_str: str) -> str:
    y, m, d = date_str.split("-")
    return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"

def parse_top_preds(path: str):
    """
    Returns dict: { 'YYYY-MM-DD': {msg_id:int -> parent_id:int} }
    msg_id and parent_id are as in the file (>=1000), not yet zero-based.
    """
    by_day = defaultdict(dict)
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            m = LINE_RE.match(raw)
            if not m:
                # tolerate odd lines quietly
                continue
            day = normalize_date(m.group("date"))
            msg = int(m.group("msg"))
            parent = int(m.group("parent"))
            by_day[day][msg] = parent
    return by_day

class DSU:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0]*n
    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
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

def chunkify_and_write(by_day, out_path: str):
    with open(out_path, "w", encoding="utf-8") as out:
        for day in sorted(by_day.keys()):
            # sort message ids ascending
            all_msgs = sorted(by_day[day].keys())
            if not all_msgs:
                continue

            # Convert to zero-based ids (1000->0) and build ordered arrays
            # Map original id -> zero-based id
            def to_zero(x): return x - 1000
            zero_ids = [to_zero(m) for m in all_msgs]
            min_zero = zero_ids[0]
            # We assume day streams start at 1000; if not, we still use absolute zero-based ids.

            # Build an array of parents in zero-based space aligned to sorted msg order
            parents_zero = [to_zero(by_day[day][m]) for m in all_msgs]

            # We will slice by absolute zero-based ids into [0..49], [50..99], etc.
            # To do that, we need the sequence to actually be contiguous from 0 upward.
            # If the day's first message id > 1000, we still respect absolute windows:
            # chunk starts at k where k % 50 == 0. Any gaps mean we can’t fill a complete chunk.
            # Build a dict for quick lookup: zero_id -> parent_zero_id
            mapping = dict(zip(zero_ids, parents_zero))

            # Determine available zero-id range for this day
            zmin, zmax = min(zero_ids), max(zero_ids)

            # Iterate full 50-sized windows starting from the first multiple of 50 >= zmin
            start = zmin - (zmin % 50)
            while start + 50 <= zmax + 1:  # ensure [start, start+49] fully within possible ids
                window_ids = list(range(start, start + 50))
                # Only keep chunks where we have *all 50* messages present
                if all(z in mapping for z in window_ids):
                    # Map parents to within-window indices; if outside, self-parent
                    parents = []
                    for i, z in enumerate(window_ids):
                        pz = mapping[z]
                        if pz < start or pz >= start + 50:
                            parents.append(i)  # fallback to self
                        else:
                            parents.append(pz - start)

                    # Build clusters via DSU from parents links
                    dsu = DSU(50)
                    for i, p in enumerate(parents):
                        dsu.union(i, p)

                    # Assign cluster labels by first appearance of each root
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

                    # chunk_id like "YYYY-MM-DD_000000"
                    chunk_id = f"{day}_{start:06d}"

                    obj = {
                        "chunk_id": chunk_id,
                        "clusters": clusters,
                        "num_conversations": num_conversations,
                        "parents": parents,
                    }
                    # Ensure spaces after commas and after colons
                    line = json.dumps(obj, separators=(', ', ': '))
                    out.write(line + "\n")
                # advance to next 50-block
                start += 50

def main():
    if len(sys.argv) != 3:
        print("Usage: python top_preds_to_predictions.py <input_top_preds.txt> <output_predictions.jsonl>")
        sys.exit(1)
    in_path, out_path = sys.argv[1], sys.argv[2]
    by_day = parse_top_preds(in_path)
    chunkify_and_write(by_day, out_path)
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
