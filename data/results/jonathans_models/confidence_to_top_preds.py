#!/usr/bin/env python3
"""
convert_confidence_to_top_preds.py

Usage:
  python convert_confidence_to_top_preds.py /path/to/ff-confidence.txt /path/to/out-top-preds.txt

Input ("confidence") expected line format (comments ok):
  <msg_id> <window_size> <cand1_id> <cand1_prob> <cand2_id> <cand2_prob> ... -

Output ("top-preds") line format:
  <msg_id> <chosen_parent_id> -
"""
import sys
from pathlib import Path

def convert_confidence_to_top_preds(in_path: Path, out_path: Path) -> None:
    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue  # skip comments/blank lines
            parts = line.split()
            # Expected: <msg_id> <window_size> <cand1_id> <cand1_prob> ... '-'
            msg_id = parts[0]

            # Find argmax over (candidate_id, probability) pairs after the first two tokens
            best_parent = None
            best_prob = float("-inf")
            i = 2  # start after msg_id and window_size
            while i < len(parts):
                tok = parts[i]
                if tok == "-":
                    break
                if i + 1 >= len(parts):
                    # Malformed trailing token; stop gracefully
                    break
                cand_id = parts[i]
                try:
                    prob = float(parts[i + 1])
                except ValueError:
                    # If prob is malformed, skip this pair
                    i += 2
                    continue
                if prob > best_prob:
                    best_prob = prob
                    best_parent = cand_id
                i += 2

            # Safety fallback: if no candidate parsed, self-link
            if best_parent is None:
                # If msg_id is like 'file:idx', use the trailing idx; else, reuse msg_id
                best_parent = msg_id.split(":")[-1] if ":" in msg_id else msg_id

            fout.write(f"{msg_id} {best_parent} -\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_confidence_to_top_preds.py <input_confidence.txt> <output_top_preds.txt>")
        sys.exit(1)
    convert_confidence_to_top_preds(Path(sys.argv[1]), Path(sys.argv[2]))
    print(f"Done. Wrote: {sys.argv[2]}")
