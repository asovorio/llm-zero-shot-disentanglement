from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..api.openai_client import OpenAIClient
from ..prompting.loader import PromptLoader
from ..prompting.schema import parse_json_object
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

def _normalize_ids(ids: List[Any]) -> List[str]:
    # Force everything to str to avoid int/str mismatches downstream
    return [str(x) for x in ids]

def _build_user_prompt(ids: List[str], texts: List[str], upto_i: int) -> str:
    """
    Builds the BR user content with ALL prior messages (0..i-1) plus the next message (i).
    """
    lines: List[str] = []
    lines.append("Chat log (each line shows ID followed by the message text):")
    if upto_i > 0:
        for j in range(upto_i):
            lines.append(f"{ids[j]}: {texts[j]}")
    else:
        lines.append("[no prior messages]")
    lines.append("")
    lines.append("Next message:")
    lines.append(f"{ids[upto_i]}: {texts[upto_i]}")
    return "\n".join(lines)

def _parents_to_clusters(parents: List[int]) -> List[int]:
    """
    Convert a parent-pointer array (forest) into cluster labels 0..(K-1).
    Two messages share a cluster if they share the same root.
    """
    n = len(parents)
    # Path-compressed find
    def find(i: int) -> int:
        root = i
        seen = set()
        while parents[root] != root:
            if root in seen:  # break cycles defensively
                break
            seen.add(root)
            root = parents[root]
        # Path compression
        while parents[i] != root:
            nxt = parents[i]
            parents[i] = root
            i = nxt
        return root

    roots: Dict[int, int] = {}
    cluster_id = 0
    labels = [0] * n
    for i in range(n):
        r = find(i)
        if r not in roots:
            roots[r] = cluster_id
            cluster_id += 1
        labels[i] = roots[r]
    return labels

@dataclass
class BestResponseRunner:
    client: OpenAIClient
    prompts: PromptLoader
    dataset: str = "ubuntu_irc"  # used only for Ubuntu-specific fallback behavior

    def run_chunk(
        self,
        chunk_id: str,
        ids: List[Any],
        texts: List[str],
        is_system: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Runs BR exactly as in the paper:
          - For each message i, prompt with ALL prior messages 0..i-1 plus the next message i.
          - Ask the model for the parent ID (or itself if new conversation).
          - Build a directed parent array and then collapse to clusters.

        Notes on fidelity:
          * We DO NOT short-circuit system messages. We still call the model (the paper relies on the prompt rule).
          * We POST-VALIDATE the output so that:
              - If Ubuntu and the last message is system, parent = self (as the prompt instructs).
              - If the predicted ID is not in the prior set, parent = self.
              - If the predicted parent index >= i (not prior), parent = self.
        """
        ids = _normalize_ids(ids)
        n = len(ids)
        assert n == len(texts), "ids and texts length mismatch"
        if is_system is None:
            is_system = [False] * n
        assert len(is_system) == n, "is_system length mismatch"

        # Choose prompt by dataset
        if self.dataset == "ubuntu_irc":
            system_prompt = self.prompts.load("ubuntu_best_response.txt")
        else:
            system_prompt = self.prompts.load("movie_best_response.txt")

        # Map from ID -> index for quick checks
        id_to_index: Dict[str, int] = {ids[i]: i for i in range(n)}

        # Initialize each node as its own parent (singleton cluster)
        parents: List[int] = list(range(n))

        for i in range(n):
            user_prompt = _build_user_prompt(ids, texts, i)

            # Always call the model (no short-circuit), matching paper’s procedure
            raw = self.client.chat(system=system_prompt, user=user_prompt)
            try:
                obj = parse_json_object(raw)
            except Exception as e:
                logger.warning("JSON parse failed on chunk %s msg %s: %r; raw=%r", chunk_id, ids[i], e, raw)
                obj = {}

            # Extract response_to (after key normalization in parser)
            parent_id_val = obj.get("response_to", None)

            # Ubuntu rule: if last message is system, parent is itself
            if self.dataset == "ubuntu_irc" and is_system[i]:
                parent_id = ids[i]
            else:
                # Normalize parent_id to str
                parent_id = str(parent_id_val) if parent_id_val is not None else None

            # Validate: parent must exist among PRIOR messages only
            if parent_id is None or parent_id not in id_to_index or id_to_index[parent_id] >= i:
                parent_idx = i  # self
            else:
                parent_idx = id_to_index[parent_id]

            parents[i] = parent_idx

        labels = _parents_to_clusters(parents)

        return {
            "chunk_id": chunk_id,
            "clusters": labels,
            "num_conversations": max(labels) + 1 if labels else 0,
        }
