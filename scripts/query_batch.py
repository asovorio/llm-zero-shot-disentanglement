#!/usr/bin/env python

"""
This script queries OpenAI's Batch API to see the state of a batch (how many of the prompts are completed, etc)
"""

from __future__ import annotations
import argparse, time
from typing import Any, Dict
from openai import OpenAI


def _rc_to_dict(rc: Any) -> Dict[str, int]:
    """Convert OpenAI BatchRequestCounts (Pydantic) to a plain dict."""
    if rc is None:
        return {}
    if isinstance(rc, dict):
        return rc
    if hasattr(rc, "to_dict"):
        return rc.to_dict()
    out = {}
    for k in ("total", "completed", "failed", "processing"):
        if hasattr(rc, k):
            out[k] = getattr(rc, k)
    return out


def show(batch_id: str):
    client = OpenAI()
    b = client.batches.retrieve(batch_id)
    rc = _rc_to_dict(getattr(b, "request_counts", None))
    total = int(rc.get("total", 0))
    completed = int(rc.get("completed", 0))
    failed = int(rc.get("failed", 0))
    remaining = max(total - completed - failed, 0)

    print(f"Batch: {batch_id}")
    print(f"Status: {getattr(b, 'status', 'unknown')}")
    print(f"Counts: completed={completed} / total={total}  (failed={failed}, remaining={remaining})")

    ofid = getattr(b, "output_file_id", None)
    efid = getattr(b, "error_file_id", None)
    if ofid:
        print(f"Output file id: {ofid}")
    if efid:
        print(f"Error file id:  {efid}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="Batch ID (e.g., batch_...)")
    ap.add_argument("--watch", type=int, default=0, help="Poll every N seconds until terminal state")
    args = ap.parse_args()

    if args.watch <= 0:
        show(args.id)
        return

    # Polling mode
    while True:
        show(args.id)
        # terminal states per docs: completed / failed / expired / canceled
        client = OpenAI()
        b = client.batches.retrieve(args.id)
        if getattr(b, "status", "") in ("completed", "failed", "expired", "canceled"):
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
