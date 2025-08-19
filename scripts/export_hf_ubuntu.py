"""
Export Ubuntu IRC splits from Hugging Face to JSONL that our prepare_data.py can read.
Writes to: data/raw/ubuntu_hf_export/ubuntu_{train,validation,test}.jsonl
"""

from __future__ import annotations
from datasets import load_dataset
from pathlib import Path
import json
import os

OUT_DIR = Path("data/raw/ubuntu_hf_export")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (split_name on HF) -> (filename stem)
SPLITS = {
    "train": "ubuntu_train",
    "validation": "ubuntu_validation",  # you'll pass --split dev later
    "test": "ubuntu_test",
}

def _norm_parent(v):
    # prepare_data.py expects a single parent id or None under key 'reply_to'
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return v[0] if v else None
    # Some datasets use -1 or self-link for roots; treat as None
    try:
        if int(v) < 0:
            return None
    except Exception:
        pass
    return v

def export_split(hf_split: str, out_stem: str) -> Path:
    print(f"Loading split '{hf_split}' from jkkummerfeld/irc_disentangle (config='ubuntu')...")
    ds = load_dataset("jkkummerfeld/irc_disentangle", "ubuntu", split=hf_split)

    out_path = OUT_DIR / f"{out_stem}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for ex in ds:
            row = {
                # keep fields our prepare_data.py can normalize
                "id": ex.get("id"),
                "user": ex.get("user") or ex.get("author"),
                "text": ex.get("text") or ex.get("body"),
                "timestamp": ex.get("timestamp") or ex.get("time") or 0,
                "reply_to": _norm_parent(ex.get("reply_to") or ex.get("parent") or ex.get("parent_id")),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"✅ Wrote {out_path}")
    return out_path

def main():
    for hf_split, stem in SPLITS.items():
        export_split(hf_split, stem)

if __name__ == "__main__":
    main()