#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.disentangle.config import load_config
from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.disentangle.api import OpenAIClient
from src.disentangle.methods import BestResponseRunner
from src.disentangle.prompting import PromptLoader
from src.disentangle.utils.io import write_jsonl, ensure_dir
from src.disentangle.utils.logging import setup_logger

logger = setup_logger(__name__)

def _pin_model_snapshot(name: str) -> str:
    """
    Paper parity: prefer exact snapshots. Adjust this function if your account
    uses different snapshot aliases.
    """
    if name in {"gpt-4o"}:
        pinned = "gpt-4o-2024-08-06"
        logger.info("Pinning model %s -> %s for paper parity.", name, pinned)
        return pinned
    elif name in {"gpt-4o-mini"}:
        pinned = "gpt-4o-mini-2024-07-18"
        logger.info("Pinning model %s -> %s for paper parity.", name, pinned)
        return pinned
    # If user provided a specific dated snapshot already, keep it.
    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for processing chunks (>=1). "
             "Parallelism is across chunks only; within-chunk order remains sequential."
    )
    args = ap.parse_args()
    cfg = load_config(args.config)

    # Prompts + API client
    prompts = PromptLoader(Path(cfg.paths.prompts_dir))

    # Enforce paper-parity decoding for base methods:
    model_name = _pin_model_snapshot(cfg.model.name)
    client = OpenAIClient(
        model=model_name,
        temperature=0,                   # force temperature = 0
        max_tokens=cfg.model.max_tokens,
        structured=cfg.model.structured_outputs  # force Structured Outputs (JSON)
    )

    # Dataset (Ubuntu IRC for this script)
    if cfg.run.dataset.lower() != "ubuntu_irc":
        raise SystemExit("Best-Response script currently wired for Ubuntu IRC chunks.")
    ds = UbuntuIrcDataset(
        data_root=Path(cfg.paths.processed_dir) / "ubuntu_irc",
        split=cfg.run.split,
        chunk_size=cfg.run.chunk_size,  # paper uses 50
        seed=cfg.run.seed,
    )
    chunks = ds.load_chunks()

    # Keep original order for deterministic output file
    order_index = {ch.chunk_id: i for i, ch in enumerate(chunks)}

    results_dir = ensure_dir(Path(cfg.paths.results_dir) / "best_response" / cfg.run.split)

    def _process_chunk(ch):
        # Build a runner per task; share the same client & prompts
        runner = BestResponseRunner(client=client, prompts=prompts, dataset=cfg.run.dataset)
        ids = [str(getattr(m, "mid", getattr(m, "id", None))) for m in ch.messages]
        if any(mid is None for mid in ids):
            raise RuntimeError(f"[{ch.chunk_id}] Missing message ID (.mid/.id) for one or more messages.")
        texts = [m.text for m in ch.messages]
        is_system = [
            bool(getattr(m, "is_system", False) or (getattr(m, "role", "") or "").lower() == "system")
            for m in ch.messages
        ]
        authors = getattr(ch, "authors", [""] * len(ids))
        out = runner.run_chunk(ch.chunk_id, ids, authors, texts, is_system=is_system)
        logger.info("Ran chunk %s", ch.chunk_id)
        return out

    rows = []
    if args.workers <= 1:
        # Single-threaded (original behavior)
        for ch in chunks:
            rows.append(_process_chunk(ch))
    else:
        logger.info("Running %d chunks with %d workers…", len(chunks), args.workers)
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_chunk, ch): ch.chunk_id for ch in chunks}
            for fut in as_completed(futs):
                out = fut.result()  # re-raise on exception
                rows.append(out)
        # Preserve original chunk order in the output file
        rows.sort(key=lambda r: order_index.get(r["chunk_id"], 10**9))

    out_path = results_dir / "predictions.jsonl"
    write_jsonl(out_path, rows)
    logger.info("Wrote predictions to %s", out_path)

if __name__ == "__main__":
    main()
