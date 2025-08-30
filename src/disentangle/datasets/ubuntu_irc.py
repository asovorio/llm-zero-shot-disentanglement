from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterable
import json
from collections import defaultdict
import re
from datetime import datetime, timezone
import math

# Reuse your project’s logger if you have it
try:
    from ..utils.logging import setup_logger
    logger = setup_logger(__name__)
except Exception:
    import logging
    logger = logging.getLogger(__name__)

PATTERNS = [
    re.compile(r"^\s*<\s*([A-Za-z0-9_\-\[\]\{\}|^`]+)\s*>\s+"),       # <nick> msg
    re.compile(r"^\s*\*\s*([A-Za-z0-9_\-\[\]\{\}|^`]+)\b"),          # * nick action
    re.compile(r"^\s*([A-Za-z0-9_\-\[\]\{\}|^`]+)\s*:\s+"),          # nick: msg
    re.compile(r"^\s*([A-Za-z0-9_\-\[\]\{\}|^`]+)\s*>\s+"),          # nick> msg
    re.compile(r"^\s*([A-Za-z0-9_\-\[\]\{\}|^`]+)\s*,\s+"),          # nick, msg
    re.compile(r"^\s*\[\s*([A-Za-z0-9_\-\[\]\{\}|^`]+)\s*\]\s+"),    # [nick] msg
    re.compile(r"^\s*\(\s*([A-Za-z0-9_\-\[\]\{\}|^`]+)\s*\)\s+"),    # (nick) msg
]


def _fallback_author_from_text(t: str) -> str:
    s = t or ""
    for rx in PATTERNS:
        m = rx.match(s)
        if m:
            return m.group(1)
    return ""

def ts_to_yyyymmdd(ts: Any) -> str:
    """
    Best-effort conversion of timestamp-like values to 'YYYY-MM-DD' (UTC).
    Accepts:
      - int/float seconds or milliseconds
      - numeric strings
      - ISO 8601 strings (e.g., '2010-05-01T12:34:56Z')
    Returns 'unknown' if parsing fails.
    """
    try:
        # numeric: int/float or numeric string
        if isinstance(ts, (int, float)):
            x = float(ts)
        elif isinstance(ts, str) and ts.strip():
            s = ts.strip()
            # numeric string?
            try:
                x = float(s)
            except Exception:
                x = None
        else:
            x = None

        if x is not None and math.isfinite(x):
            # Heuristic: >1e12 → ms; >1e10 → probably ms from int cast
            if x > 1e12:
                x = x / 1000.0
            dt = datetime.utcfromtimestamp(x)
            return dt.strftime("%Y-%m-%d")

        # ISO-8601 string?
        if isinstance(ts, str) and ts.strip():
            s = ts.strip().replace("Z", "+00:00")
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None:
                    # treat naive as UTC
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                pass

    except Exception:
        pass
    return "unknown"

@dataclass
class Message:
    mid: str                   # unique message ID
    text: str
    session_id: str
    ts: int                    # sortable timestamp or order index
    is_system: bool = False
    gold: Optional[Any] = None # original gold convo/thread id if present
    role: Optional[str] = None
    author: Optional[str] = None  # <-- NEW: speaker/nick if available


@dataclass
class Chunk:
    chunk_id: str
    messages: List[Message]
    ids: List[str]                 # <-- NEW: convenience lists for runners
    authors: List[str]             # <-- NEW
    texts: List[str]               # <-- NEW
    is_system: List[bool]          # <-- NEW
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
            session_id = self._coalesce(row, "session_id", "sid", "session")
            if not session_id:
                day = row.get("day") or row.get("date")
                if day:
                    session_id = str(day)
                else:
                    ts = self._coalesce(row, "timestamp", "ts")
                    session_id = ts_to_yyyymmdd(ts) if ts is not None else "unknown"
            session_id = str(session_id)
            mid       = str(self._coalesce(row, "mid", "id", "msg_id", default=f"{session_id}:{idx}"))
            text      = self._coalesce(row, "text", "message", "content", default="")
            role      = str(self._coalesce(row, "role", default="")).lower()
            is_system = bool(self._coalesce(row, "is_system", default=(role == "system")))
            # Use provided timestamp, else fallback to file order
            ts        = int(self._coalesce(row, "timestamp", "ts", default=idx))
            # NEW: author/speaker/nick (if available)
            author    = self._coalesce(row, "author", "speaker", "user", "nick", "username", default=None)

            # Accept only official gold fields
            gold = self._coalesce(row, "conv_id", "conversation_id", default=None)
            if gold is None:
                # hard-fail to avoid silently evaluating against non-official or missing gold
                raise ValueError(
                    "Missing official gold conversation id (conv_id/conversation_id). "
                    "Ensure you're using the official Ubuntu IRC dev set with gold labels."
                )

            msg = Message(
                mid=mid, text=text, session_id=str(session_id),
                ts=ts, is_system=is_system, gold=gold, role=role, author=author
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

                # Convenience lists for runners (ids/authors/texts/is_system)
                ids       = [m.mid for m in window]
                authors = []
                for m in window:
                    a = (m.author or "").strip()
                    if not a:
                        a = _fallback_author_from_text(m.text)
                    authors.append(a)
                texts     = [m.text for m in window]
                is_system = [m.is_system for m in window]

                # stable, human-readable chunk_id
                chunk_id = f"{sess_id}_{start:06d}"
                # NOTE: gold_list is expected to be non-None for canonical dev; assert if you want
                # assert gold_list is not None, "Chunk includes rows without gold."

                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    messages=window,
                    ids=ids,
                    authors=authors,
                    texts=texts,
                    is_system=is_system,
                    gold=gold_list
                ))

        logger.info(
            "Built %d chunks (size=%d) across %d sessions for split=%s",
            len(chunks), self.chunk_size, len(sessions), self.split
        )
        return chunks
