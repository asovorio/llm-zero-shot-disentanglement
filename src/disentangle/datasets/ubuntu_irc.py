from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
import json
from collections import defaultdict

# Reuse your project’s logger if you have it
try:
    from ..utils.logging import setup_logger
    logger = setup_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)


@dataclass
class Message:
    mid: str                   # unique message ID
    text: str
    session_id: str
    ts: int                    # sortable timestamp or order index
    is_system: bool = False
    gold: Optional[Any] = None # original gold convo/thread id if present
    role: Optional[str] = None


@dataclass
class Chunk:
    chunk_id: str
    messages: List[Message]
    # gold labels per message (length == chunk size), or None if unavailable
    gold: Optional[List[Any]] = None


class UbuntuIrcDataset:
    """
    Expected directory layout (same as before):
        data_root/
            ubuntu_train.jsonl or train.jsonl
            ubuntu_dev.jsonl   or dev.jsonl
            ubuntu_test.jsonl  or test.jsonl

    Each line = one message. We try several field names to be robust.
    """

    def __init__(self, data_root: Path, split: str, chunk_size: int = 50, seed: int = 123):
        self.data_root = Path(data_root)
        self.split = split
        self.chunk_size = int(chunk_size)
        self.seed = seed

    # ---------- helpers ----------
    @staticmethod
    def _open_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    @staticmethod
    def _coalesce(d: Dict[str, Any], *keys, default=None):
        for k in keys:
            if k in d and d[k] is not None:
                return d[k]
        return default

    def _resolve_split_path(self) -> Path:
        # Try a few common filenames
        candidates = [
            self.data_root / f"{self.split}.jsonl",
            self.data_root / f"ubuntu_{self.split}.jsonl",
            self.data_root / f"ubuntu_{self.split}.json",
        ]
        for p in candidates:
            if p.exists():
                return p
        raise FileNotFoundError(f"Could not find split file for '{self.split}' in {self.data_root}")

    # ---------- public API ----------
    def load_chunks(self) -> List[Chunk]:
        """
        Paper-faithful chunking:
          - group by session_id
          - sort within session (by timestamp or original index)
          - split into non-overlapping windows of EXACTLY self.chunk_size
          - keep system messages
          - drop last incomplete window
          - attach gold labels per chunk if available (list aligned to messages)
        """
        path = self._resolve_split_path()
        logger.info("Loading Ubuntu IRC messages from %s", path)

        # 1) Load and normalize messages
        sessions: Dict[str, List[Message]] = defaultdict(list)
        for idx, row in enumerate(self._open_jsonl(path)):
            session_id = self._coalesce(row, "session_id", "sid", "session", default="unknown")
            mid       = str(self._coalesce(row, "mid", "id", "msg_id", default=f"{session_id}:{idx}"))
            text      = self._coalesce(row, "text", "message", "content", default="")
            role      = str(self._coalesce(row, "role", default="")).lower()
            is_system = bool(self._coalesce(row, "is_system", default=(role == "system")))
            # Use provided timestamp, else fallback to file order
            ts        = int(self._coalesce(row, "timestamp", "ts", default=idx))
            # ✳️ Expanded list of acceptable gold label fields
            gold      = self._coalesce(
                row,
                "gold", "gold_cluster", "conversation_id", "thread_id",
                "conv_id", "conversation", "conv", "cluster", "cluster_id",
                "label", "label_id", "convId", "threadId"
            )

            msg = Message(
                mid=mid, text=text, session_id=str(session_id),
                ts=ts, is_system=is_system, gold=gold, role=role
            )
            sessions[str(session_id)].append(msg)

        if not sessions:
            logger.warning("No messages loaded for split=%s from %s", self.split, path)

        # 2) Sort within each session
        for s in sessions.values():
            s.sort(key=lambda m: (m.ts, m.mid))

        # 3) Emit EXACT chunk_size chunks per session (drop remainder)
        chunks: List[Chunk] = []
        for sess_id, msgs in sessions.items():
            n = len(msgs)
            if n < self.chunk_size:
                continue
            for start in range(0, n - self.chunk_size + 1, self.chunk_size):
                window = msgs[start:start + self.chunk_size]
                if len(window) != self.chunk_size:
                    continue  # exact size only

                # Collect gold labels for the window (if all present)
                window_gold = [m.gold for m in window]
                if all(g is not None for g in window_gold):
                    gold_list: Optional[List[Any]] = window_gold
                else:
                    gold_list = None
                    # Optional: log once per bad window (helps debugging mismatched field names)
                    if any(g is not None for g in window_gold):
                        logger.debug(
                            "Partial gold found in chunk %s_%06d (some messages missing gold labels).",
                            sess_id, start
                        )

                # stable, human-readable chunk_id
                chunk_id = f"{sess_id}_{start:06d}"
                chunks.append(Chunk(chunk_id=chunk_id, messages=window, gold=gold_list))

        logger.info(
            "Built %d chunks (size=%d) across %d sessions for split=%s",
            len(chunks), self.chunk_size, len(sessions), self.split
        )
        return chunks
