from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable, Tuple
import re
import random
from ..utils.io import list_files, read_jsonl
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

SYSTEM_PAT = re.compile(r"\b(join|part|quit|nick|mode|topic)\b", re.I)

@dataclass
class Message:
    mid: str
    author: str
    text: str
    ts: float | int | None
    is_system: bool = False

@dataclass
class Chunk:
    chunk_id: str
    messages: List[Message]
    # gold labels optional: a list aligned with messages (conversation ids)
    gold: List[int] | None = None

class UbuntuIrcDataset:
    """
    Loads Ubuntu IRC annotated data from the official repo layout
    and exposes 50-message chunks (evaluation protocol used in recent work).
    Expected raw data: cloned repo under data/raw/ubuntu_irc_src/repo/
    We accept JSONL inputs with fields: id, author, text, timestamp, [conv_id].
    """
    def __init__(self, data_root: str | Path, split: str, chunk_size: int = 50, seed: int = 0):
        self.root = Path(data_root)
        self.split = split
        self.chunk_size = chunk_size
        random.seed(seed)

    def _iter_jsonl(self) -> Iterable[Dict[str, Any]]:
        # Heuristic: pick any .jsonl in repo that looks like a partition for the split.
        # You can adjust this pattern to your local copy.
        candidates = list_files(self.root, suffix=".jsonl")
        if not candidates:
            logger.warning("No JSONL files found under %s; please convert raw repo to jsonl.", self.root)
        for path in candidates:
            if self.split in path.name.lower():
                for row in read_jsonl(path):
                    yield row

    def _to_msg(self, row: Dict[str, Any]) -> Message:
        text = row.get("text") or row.get("body") or ""
        author = row.get("author") or row.get("user") or "UNK"
        is_system = bool(row.get("is_system")) or bool(SYSTEM_PAT.search(text))
        return Message(
            mid=str(row.get("id", row.get("mid", ""))),
            author=str(author),
            text=str(text),
            ts=row.get("timestamp") or row.get("ts"),
            is_system=is_system,
        )

    def load_chunks(self) -> List[Chunk]:
        rows = list(self._iter_jsonl())
        if not rows:
            logger.error("Ubuntu IRC rows are empty. Ensure you exported the dataset to JSONL with the split name.")
            return []
        msgs = [self._to_msg(r) for r in rows]
        # stable order by timestamp then by original order
        msgs.sort(key=lambda m: (m.ts if m.ts is not None else 0))
        chunks: List[Chunk] = []
        for i in range(0, len(msgs), self.chunk_size):
            sub = msgs[i:i + self.chunk_size]
            if not sub:
                continue
            cid = f"ubuntu_{self.split}_{i//self.chunk_size:06d}"
            # gold ids if present
            gold = None
            if "conv_id" in rows[0]:
                gold = [int(r.get("conv_id")) for r in rows[i:i + self.chunk_size]]
            chunks.append(Chunk(chunk_id=cid, messages=sub, gold=gold))
        logger.info("Loaded %d chunks of size ~%d for split=%s", len(chunks), self.chunk_size, self.split)
        return chunks
