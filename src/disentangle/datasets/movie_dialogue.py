from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Iterable
import json
from ..utils.io import list_files
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class Message:
    mid: str
    author: str
    text: str

@dataclass
class Dialogue:
    did: str
    messages: List[Message]
    gold: List[int]  # conversation ids per message

class MovieDialogueDataset:
    """
    Loads the Movie Dialogue dataset (authors' repo).
    Expects JSON files under data/raw/movie_dialogue_src/repo/dataset/{train,dev,test}.json
    Each dialogue contains messages and gold conversation ids.
    """
    def __init__(self, data_root: str | Path, split: str):
        self.root = Path(data_root)
        self.split = split

    def _load_split_file(self) -> Path | None:
        candidates = [p for p in list_files(self.root, suffix=".json") if self.split in p.name.lower()]
        if not candidates:
            logger.error("Movie Dialogue %s split not found under %s", self.split, self.root)
            return None
        return candidates[0]

    def load_dialogues(self) -> List[Dialogue]:
        path = self._load_split_file()
        if path is None:
            return []
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        diags: List[Dialogue] = []
        for item in data:
            did = str(item.get("id") or item.get("dialog_id"))
            msgs = [Message(mid=str(m["id"]), author=str(m.get("speaker", "UNK")), text=str(m["text"]))
                    for m in item["messages"]]
            gold = [int(x) for x in item["conversation_ids"]]
            diags.append(Dialogue(did=did, messages=msgs, gold=gold))
        logger.info("Loaded %d dialogues for split=%s", len(diags), self.split)
        return diags
