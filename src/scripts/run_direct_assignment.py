#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List
from src.llm_disentangle.config import load_config
from src.llm_disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.llm_disentangle.api import OpenAIClient
from src.llm_disentangle.methods import DirectAssignmentRunner
from src.llm_disentangle.prompting import PromptLoader
from src.llm_disentangle.utils.io import write_jsonl, ensure_dir
from src.llm_disentangle.utils.logging import setup_logger

logger = setup_logger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)

    prompts = PromptLoader(cfg.paths.prompts_dir)
    client = OpenAIClient(model=cfg.model.name, temperature=cfg.model.temperature,
                          max_tokens=cfg.model.max_tokens, structured=cfg.model.structured_outputs)
    runner = DirectAssignmentRunner(client=client, prompts=prompts)

    if cfg.run.dataset.lower() != "ubuntu_irc":
        raise SystemExit("Direct-Assignment script currently wired for Ubuntu IRC chunks.")
    ds = UbuntuIrcDataset(
        data_root=(Path(cfg.paths.data_dir) / "ubuntu_irc_src" / "repo"),
        split=cfg.run.split,
        chunk_size=cfg.run.chunk_size,
        seed=cfg.run.seed,
    )
    chunks = ds.load_chunks()

    results_dir = ensure_dir(Path(cfg.paths.results_dir) / "direct_assignment" / cfg.run.split)
    rows = []
    for ch in chunks:
        texts = [m.text for m in ch.messages]
        out = runner.run_chunk(ch.chunk_id, texts)
        rows.append(out)

    write_jsonl(results_dir / "predictions.jsonl", rows)
    logger.info("Wrote predictions to %s", results_dir / "predictions.jsonl")

if __name__ == "__main__":
    main()
