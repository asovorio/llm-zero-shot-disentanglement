#!/usr/bin/env python
from __future__ import annotations
import argparse
from pathlib import Path
from src.llm_disentangle.config import load_config
from src.llm_disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.llm_disentangle.api import OpenAIClient
from src.llm_disentangle.methods import SelfCriticRefiner
from src.llm_disentangle.prompting import PromptLoader
from src.llm_disentangle.utils.io import read_jsonl, write_jsonl, ensure_dir
from src.llm_disentangle.utils.logging import setup_logger

logger = setup_logger(__name__)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--predictions", required=True, help="predictions.jsonl from a prior run")
    ap.add_argument("--iters", type=int, default=1)
    args = ap.parse_args()
    cfg = load_config(args.config)

    prompts = PromptLoader(cfg.paths.prompts_dir)
    client = OpenAIClient(model=cfg.model.name, temperature=cfg.model.temperature,
                          max_tokens=cfg.model.max_tokens, structured=cfg.model.structured_outputs)
    refiner = SelfCriticRefiner(client=client, prompts=prompts)

    ds = UbuntuIrcDataset(
        data_root=(Path(cfg.paths.data_dir) / "ubuntu_irc_src" / "repo"),
        split=cfg.run.split,
        chunk_size=cfg.run.chunk_size,
        seed=cfg.run.seed,
    )
    chunks = {c.chunk_id: c for c in ds.load_chunks()}

    preds = read_jsonl(args.predictions)
    refined = []
    for item in preds:
        cid = item["chunk_id"]
        texts = [m.text for m in chunks[cid].messages]
        if "clusters" in item:
            base = item["clusters"]
        elif "parents" in item:
            base = parents_to_clusters(item["parents"])
        else:
            continue
        out = refiner.refine_chunk(cid, texts, base, max_iters=args.iters)
        refined.append(out)

    out_dir = ensure_dir(Path(cfg.paths.results_dir) / "self_critic" / cfg.run.split)
    write_jsonl(out_dir / "predictions.jsonl", refined)
    logger.info("Wrote refined predictions to %s", out_dir / "predictions.jsonl")

def parents_to_clusters(parents):
    # same helper as in BestResponse
    n = len(parents)
    roots, cluster = {}, [-1] * n
    cid = 0
    def root(i):
        seen = set()
        while parents[i] != i:
            if i in seen:
                break
            seen.add(i)
            i = parents[i]
        return i
    for i in range(n):
        r = root(i)
        if r not in roots:
            roots[r] = cid
            cid += 1
        cluster[i] = roots[r]
    return cluster

if __name__ == "__main__":
    main()
