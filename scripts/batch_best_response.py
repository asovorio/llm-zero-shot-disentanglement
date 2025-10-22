#!/usr/bin/env python
from __future__ import annotations
import argparse, json, hashlib, time
from pathlib import Path
from typing import Dict, Any, List

from openai import OpenAI

from src.disentangle.config import load_config
from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.disentangle.datasets.movie_dialogue import MovieDialogueDataset
from src.disentangle.prompting.loader import PromptLoader
from src.disentangle.prompting.schema import parse_json_object
from src.disentangle.utils.io import ensure_dir, write_jsonl
from src.disentangle.utils.logging import setup_logger

# Reuse existing helpers from your BR runner to guarantee identical behavior
from src.disentangle.methods.best_response import (
    _build_user_prompt_for_step, _collapse_parents, _coerce_parent_id
)

logger = setup_logger(__name__)

def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

def _make_custom_id(chunk_id: str, idx: int) -> str:
    base = f"br::{chunk_id}::{idx}"
    return f"{base}::{_h(base)}"

def _build_batch_line(custom_id: str, model: str, messages: List[Dict[str, Any]], structured: bool) -> str:
    body: Dict[str, Any] = {"model": model, "messages": messages}
    if structured:
        body["response_format"] = {"type": "json_object"}
    return json.dumps({
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": body
    }, ensure_ascii=False)

def _poll_until_done(client: OpenAI, batch_id: str, sleep_s: float = 15.0):
    while True:
        b = client.batches.retrieve(batch_id)
        logger.info("Batch %s status: %s", batch_id, b.status)
        if b.status in ("completed", "failed", "expired", "canceled"):
            return b
        time.sleep(sleep_s)

def _make_dataset(name: str, paths, run):
    name = name.lower()
    if name == "ubuntu_irc":
        return UbuntuIrcDataset(
            data_root=Path(paths.processed_dir) / "ubuntu_irc",
            split=run.split, chunk_size=run.chunk_size, seed=run.seed
        )
    if name == "movie_dialogue":
        return MovieDialogueDataset(
            data_root=Path(paths.processed_dir) / "movie_dialogue",
            split=run.split, chunk_size=run.chunk_size, seed=run.seed
        )
    raise ValueError(f"Unknown dataset: {name}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/batch.yaml")
    ap.add_argument("--no-wait", action="store_true", help="Submit and exit (collect later)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run, model, paths = cfg.run, cfg.model, cfg.paths

    # Read optional 'batch' block (not in dataclass)
    import yaml
    with open(args.config, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    batch_cfg = raw.get("batch", {})

    ds = _make_dataset(run.dataset, paths, run)
    chunks = ds.load_chunks()
    prompts = PromptLoader(Path(cfg.paths.prompts_dir))

    # System prompt by dataset (mirrors BR runner)
    if run.dataset.lower() == "ubuntu_irc":
        system_prompt = prompts.load("ubuntu_best_response.txt")
    else:
        system_prompt = prompts.load("movie_best_response.txt")

    model_name = model.name
    structured = bool(model.structured_outputs)

    in_path = Path(batch_cfg.get("input_path", "./batch/in/requests.jsonl"))
    out_path = Path(batch_cfg.get("output_path", "./batch/out/output.jsonl"))
    completion_window = batch_cfg.get("completion_window", "24h")

    # 1) Prepare JSONL (one request per message i)
    ensure_dir(in_path.parent)
    lines: List[str] = []
    for ch in chunks:
        n = len(ch.ids)
        display_ids = [str(j + 1) for j in range(n)]
        for i in range(n):
            user_prompt = _build_user_prompt_for_step(display_ids, ch.authors, ch.texts, i)
            cid = _make_custom_id(ch.chunk_id, i)
            messages = [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}]
            lines.append(_build_batch_line(cid, model_name, messages, structured))

    with in_path.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    logger.info("Prepared %d requests into %s", len(lines), in_path)

    # 2) Submit
    client = OpenAI()
    up = client.files.create(file=in_path.open("rb"), purpose="batch")
    job = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window=completion_window,
    )
    logger.info("Submitted batch_id=%s; status=%s", job.id, job.status)

    if args.no_wait:
        print(job.id)
        return

    # 3) Wait & download
    job = _poll_until_done(client, job.id)
    if job.status != "completed":
        raise RuntimeError(f"Batch ended with status={job.status}")
    file_id = getattr(job, "output_file_id", None)
    if not file_id:
        raise RuntimeError("No output_file_id on completed batch")

    ensure_dir(out_path.parent)
    content = client.files.content(file_id).text
    out_path.write_text(content, encoding="utf-8")
    logger.info("Downloaded batch output to %s", out_path)

    answers: Dict[str, str] = {}
    with out_path.open("r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            obj = json.loads(ln)
            try:
                cid = obj["custom_id"]
                txt = obj["response"]["body"]["choices"][0]["message"]["content"]
                answers[cid] = txt
            except Exception:
                pass

    rows = []
    for ch in chunks:
        n = len(ch.ids)
        display_ids = [str(j + 1) for j in range(n)]
        parents = [0] * n

        for i in range(n):
            pref = f"br::{ch.chunk_id}::{i}"
            found = None
            for k, v in answers.items():
                if k.startswith(pref):
                    found = v
                    break
            if found is None:
                raise RuntimeError(f"Missing BR answer for {pref}")

            obj = parse_json_object(found)
            tok = _coerce_parent_id(obj.get("response_to"))

            # Ubuntu rule: if current is system => self
            if cfg.run.dataset.lower() == "ubuntu_irc" and ch.is_system[i]:
                parents[i] = i
                continue

            # Mirror best_response.py logic exactly:
            parent_idx = i  # default self
            if tok is not None:
                if tok == display_ids[i]:
                    parent_idx = i
                else:
                    try:
                        k = int(tok)
                        if 1 <= k <= i:  # prior only (DISPLAY index)
                            parent_idx = k - 1
                        else:
                            parent_idx = i
                    except Exception:
                        parent_idx = i

            # Reroute if chosen parent is a system msg: climb to nearest non-system; else self
            if parent_idx != i and ch.is_system[parent_idx]:
                p = parent_idx
                seen = set()
                while p != parents[p] and ch.is_system[p] and p not in seen:
                    seen.add(p)
                    p = parents[p]
                if ch.is_system[p]:
                    parent_idx = i
                else:
                    parent_idx = p

            parents[i] = parent_idx

        labels = _collapse_parents(parents)
        rows.append({
            "chunk_id": ch.chunk_id,
            "clusters": labels,
            "num_conversations": len(set(labels)),
            "parents": parents,
        })

    # Stable order + write predictions
    order = {c.chunk_id: idx for idx, c in enumerate(chunks)}
    rows.sort(key=lambda r: order.get(r["chunk_id"], 10 ** 9))

    pred_path = Path(paths.results_dir) / "best_response" / cfg.run.split / "predictions.jsonl"
    ensure_dir(pred_path.parent)
    write_jsonl(pred_path, rows)
    logger.info("Wrote predictions to %s", pred_path)

    (Path(paths.results_dir) / "LAST_RUN.txt").write_text(
        f"config={args.config}\nmethod=best_response\nsplit={cfg.run.split}\npredictions={pred_path}\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
