#!/usr/bin/env python

"""
This file runs the DA system using the BatchAPI. It's behaviour is identical to that of "run_direct_assignment.py"
Since within a chunk, message m's prompt depends on the previous messages (0,...,m-1) prompt results because we build the clusters sequentially,
we will do 50 Batch API calls (50 message chunks), with each call having one message per chunk.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
from openai import OpenAI

from src.disentangle.config import load_config
from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
from src.disentangle.methods.direct_assignment import (
    _build_user_prompt_for_step,
    _coerce_int,
    _normalize_ids,
)
from src.disentangle.prompting.loader import PromptLoader
from src.disentangle.prompting.schema import parse_json_object
from src.disentangle.utils.io import ensure_dir, write_jsonl
from src.disentangle.utils.logging import setup_logger

logger = setup_logger(__name__)


def _pin_model_snapshot(name: str) -> str:
    if name in {"gpt-4o"}:
        pinned = "gpt-4o-2024-08-06"
        logger.info("Pinning model %s -> %s for paper parity.", name, pinned)
        return pinned
    if name in {"gpt-4o-mini"}:
        pinned = "gpt-4o-mini-2024-07-18"
        logger.info("Pinning model %s -> %s for paper parity.", name, pinned)
        return pinned
    return name


def _h(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]


def _make_custom_id(chunk_id: str, idx: int) -> str:
    base = f"da::{chunk_id}::{idx}"
    return f"{base}::{_h(base)}"


def _build_batch_line(custom_id: str, model: str, messages: List[Dict[str, Any]], structured: bool) -> str:
    body: Dict[str, Any] = {"model": model, "messages": messages}
    if structured:
        body["response_format"] = {"type": "json_object"}
    return json.dumps(
        {"custom_id": custom_id, "method": "POST", "url": "/v1/chat/completions", "body": body},
        ensure_ascii=False,
    )


def _poll_until_done(client: OpenAI, batch_id: str, sleep_s: float = 15.0):
    while True:
        job = client.batches.retrieve(batch_id)
        logger.info("Batch %s status: %s", batch_id, job.status)
        if job.status in {"completed", "failed", "expired", "canceled"}:
            return job
        time.sleep(sleep_s)


def _make_dataset(name: str, paths, run):
    name = name.lower()
    if name == "ubuntu_irc":
        return UbuntuIrcDataset(
            data_root=Path(paths.processed_dir) / "ubuntu_irc",
            split=run.split,
            chunk_size=run.chunk_size,
            seed=run.seed,
        )
    raise ValueError(f"batch_direct_assignment currently supports dataset=ubuntu_irc (got {name}).")


@dataclass
class ChunkState:
    chunk: Any
    ids: List[str]
    authors: List[str]
    texts: List[str]
    is_system: List[bool]
    display_ids: List[str]
    clusters: List[List[int]]
    labels: List[int]

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def size(self) -> int:
        return len(self.ids)


def _init_states(chunks: List[Any]) -> List[ChunkState]:
    states: List[ChunkState] = []
    for ch in chunks:
        ids = _normalize_ids(ch.ids)
        texts = list(ch.texts)
        authors = list(ch.authors or [""] * len(ids))
        if len(authors) != len(ids):
            authors = (authors + [""] * len(ids))[:len(ids)]
        authors = [(a or "").strip() for a in authors]
        is_system = list(getattr(ch, "is_system", [False] * len(ids)))
        display_ids = [str(i + 1) for i in range(len(ids))]
        state = ChunkState(
            chunk=ch,
            ids=ids,
            authors=authors,
            texts=texts,
            is_system=is_system,
            display_ids=display_ids,
            clusters=[],
            labels=[-1] * len(ids),
        )
        states.append(state)
    return states


def _step_path(base: Path, step: int) -> Path:
    stem = base.stem or "requests"
    suffix = base.suffix or ".jsonl"
    return base.with_name(f"{stem}_step_{step:02d}{suffix}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/batch.yaml")
    ap.add_argument("--poll-interval", type=float, default=15.0, help="Seconds between batch status polls.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.model.provider.lower() != "openai":
        raise SystemExit("batch_direct_assignment only supports OpenAI provider at the moment.")

    with open(args.config, "r", encoding="utf-8") as f:
        raw_cfg = yaml.safe_load(f)
    batch_cfg = raw_cfg.get("batch", {}) if isinstance(raw_cfg, dict) else {}

    run, model, paths = cfg.run, cfg.model, cfg.paths
    ds = _make_dataset(run.dataset, paths, run)
    chunks = ds.load_chunks()
    if not chunks:
        raise SystemExit("No chunks loaded; check dataset paths and split.")

    states = _init_states(chunks)
    max_steps = max(st.size for st in states)
    logger.info("Loaded %d chunks; chunk_size=%d; planning %d sequential batch steps.", len(states), run.chunk_size, max_steps)

    prompts = PromptLoader(Path(paths.prompts_dir))
    dataset_name = run.dataset.lower()
    if dataset_name == "ubuntu_irc":
        system_prompt = prompts.load("ubuntu_direct_assignment.txt")

    structured = bool(model.structured_outputs)
    model_name = _pin_model_snapshot(model.name)

    client = OpenAI()
    base_in_path = Path(batch_cfg.get("input_path", "./batch/in/requests.jsonl"))
    base_out_path = Path(batch_cfg.get("output_path", "./batch/out/output.jsonl"))
    completion_window = batch_cfg.get("completion_window", "24h")
    poll_interval = float(batch_cfg.get("poll_interval", args.poll_interval))

    ensure_dir(base_in_path.parent)
    ensure_dir(base_out_path.parent)

    chunk_lookup: Dict[str, ChunkState] = {st.chunk_id: st for st in states}

    for step in range(max_steps):
        lines: List[str] = []
        cid_to_state: Dict[str, Tuple[ChunkState, int]] = {}

        for st in states:
            if step >= st.size:
                continue
            user_prompt = _build_user_prompt_for_step(st.display_ids, st.authors, st.texts, st.clusters, step)
            cid = _make_custom_id(st.chunk_id, step)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            lines.append(_build_batch_line(cid, model_name, messages, structured))
            cid_to_state[cid] = (st, step)

        if not lines:
            logger.info("No active messages at step %d; stopping early.", step)
            break

        step_in_path = _step_path(base_in_path, step)
        with step_in_path.open("w", encoding="utf-8") as f:
            for ln in lines:
                f.write(ln + "\n")
        logger.info("Step %d: wrote %d requests to %s", step + 1, len(lines), step_in_path)

        with step_in_path.open("rb") as fh:
            upload = client.files.create(file=fh, purpose="batch")
        job = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )
        logger.info("Step %d: submitted batch_id=%s (status=%s)", step + 1, job.id, job.status)

        job = _poll_until_done(client, job.id, sleep_s=poll_interval)
        if job.status != "completed":
            raise RuntimeError(f"Batch {job.id} ended with status={job.status}")

        file_id = getattr(job, "output_file_id", None)
        if not file_id:
            raise RuntimeError(f"No output_file_id returned for batch {job.id}")

        step_out_path = _step_path(base_out_path, step)
        ensure_dir(step_out_path.parent)
        content = client.files.content(file_id).text
        step_out_path.write_text(content, encoding="utf-8")
        logger.info("Step %d: downloaded batch output to %s", step + 1, step_out_path)

        answers: Dict[str, str] = {}
        with step_out_path.open("r", encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                obj = json.loads(ln)
                try:
                    cid = obj["custom_id"]
                    txt = obj["response"]["body"]["choices"][0]["message"]["content"]
                    answers[cid] = txt
                except Exception as exc:
                    logger.warning("Step %d: malformed batch line skipped (%s): %s", step + 1, exc, ln.strip())

        missing = [cid for cid in cid_to_state.keys() if cid not in answers]
        if missing:
            raise RuntimeError(f"Missing {len(missing)} answers at step {step}: {missing[:3]} ...")

        for cid, (st, idx) in cid_to_state.items():
            raw = answers[cid]
            try:
                obj = parse_json_object(raw)
            except Exception as exc:
                logger.warning("JSON parse failed (chunk=%s idx=%d): %s raw=%r", st.chunk_id, idx, exc, raw)
                obj = {}

            k = _coerce_int(obj.get("conversation_id"))
            if dataset_name == "ubuntu_irc" and st.is_system[idx]:
                k = 0

            if k is None or k <= 0 or k > len(st.clusters):
                new_idx = len(st.clusters)
                st.clusters.append([idx])
                st.labels[idx] = new_idx
            else:
                target = k - 1
                st.labels[idx] = target
                st.clusters[target].append(idx)

        logger.info("Step %d: assignments updated for %d chunks.", step + 1, len(cid_to_state))

    for st in states:
        if any(label < 0 for label in st.labels):
            raise RuntimeError(f"Incomplete assignments for chunk {st.chunk_id}")

    rows: List[Dict[str, Any]] = []
    for st in states:
        rows.append(
            {
                "chunk_id": st.chunk_id,
                "clusters": st.labels,
                "num_conversations": len(st.clusters),
            }
        )

    order = {st.chunk_id: idx for idx, st in enumerate(states)}
    rows.sort(key=lambda r: order.get(r["chunk_id"], 10**9))

    pred_path = Path(paths.results_dir) / "direct_assignment" / run.split / "predictions.jsonl"
    ensure_dir(pred_path.parent)
    write_jsonl(pred_path, rows)
    logger.info("Wrote predictions to %s", pred_path)

    last_run = Path(paths.results_dir) / "LAST_RUN.txt"
    last_run.write_text(
        f"config={args.config}\nmethod=batch_direct_assignment\nsplit={run.split}\npredictions={pred_path}\n",
        encoding="utf-8",
    )
    logger.info("Updated %s", last_run)


if __name__ == "__main__":
    main()
