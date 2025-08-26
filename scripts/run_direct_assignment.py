#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path

from src.disentangle.config import load_config
from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
# from src.disentangle.datasets.movie_dialogue import MovieDialogueDataset
from src.disentangle.api import OpenAIClient
from src.disentangle.methods.direct_assignment import DirectAssignmentRunner
from src.disentangle.prompting import PromptLoader
from src.disentangle.utils.io import write_jsonl, ensure_dir
from src.disentangle.utils.logging import setup_logger

logger = setup_logger(__name__)

def _pin_model_snapshot(name: str) -> str:
    if name in {"gpt-4o"}:
        pinned = "gpt-4o-2024-08-06"
        logger.info("Pinning model %s -> %s for paper parity.", name, pinned)
        return pinned

    elif name in {"gpt-4o-mini"}:
        pinned = "gpt-4o-mini-2024-07-18"
        logger.info("Pinning model %s -> %s for paper parity.", name, pinned)
        return pinned

    return name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    prompts = PromptLoader(Path(cfg.paths.prompts_dir))

    # Enforce paper-parity decoding for base methods:
    model_name = _pin_model_snapshot(cfg.model.name)
    client = OpenAIClient(
        model=model_name,
        temperature=0,                   # <-- force temperature = 0
        max_tokens=cfg.model.max_tokens,
        structured=cfg.model.structured_outputs  # <-- force Structured Outputs (JSON)
    )

    dataset_name = cfg.run.dataset.lower()
    if dataset_name == "ubuntu_irc":
        ds = UbuntuIrcDataset(
            data_root=Path(cfg.paths.processed_dir) / "ubuntu_irc",
            split=cfg.run.split,
            chunk_size=cfg.run.chunk_size,
            seed=cfg.run.seed,
        )
        dataset_key = "ubuntu_irc"
    elif dataset_name == "movie_dialogue":
        # ds = MovieDialogueDataset(...); dataset_key = "movie_dialogue"
        raise SystemExit("Wire up MovieDialogueDataset here.")
    else:
        raise SystemExit(f"Unknown dataset: {cfg.run.dataset}")

    chunks = ds.load_chunks()

    runner = DirectAssignmentRunner(client=client, prompts=prompts, dataset=dataset_key)

    results_dir = ensure_dir(Path(cfg.paths.results_dir) / "direct_assignment" / cfg.run.split)
    rows = []
    for ch in chunks:
        ids = [str(getattr(m, "mid", getattr(m, "id", None))) for m in ch.messages]
        if any(mid is None for mid in ids):
            raise SystemExit(f"[{ch.chunk_id}] Missing message ID (.mid/.id).")
        texts = [m.text for m in ch.messages]
        is_system = [
            bool(getattr(m, "is_system", False) or (getattr(m, "role", "") or "").lower() == "system")
            for m in ch.messages
        ]
        authors = getattr(ch, "authors", [""] * len(ids))

        out = runner.run_chunk(ch.chunk_id, ids, authors, texts, is_system=is_system)
        logger.info("Ran chunk %s", ch.chunk_id)
        rows.append(out)

    out_path = results_dir / "predictions.jsonl"
    write_jsonl(out_path, rows)
    logger.info("Wrote predictions to %s", out_path)

if __name__ == "__main__":
    main()
