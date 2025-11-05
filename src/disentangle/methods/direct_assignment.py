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


def _coerce_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        s = value.strip()
        if s.lstrip("-").isdigit():
            try:
                return int(s)
            except Exception:
                return None
    return None


def _render_clusters(display_ids: List[str], authors: List[str], texts: List[str],
                     clusters: List[List[int]]) -> str:
    lines: List[str] = []
    if not clusters:
        lines.append("No existing conversations yet.")
        return "\n".join(lines)

    for j, members in enumerate(clusters, start=1):  # 1..K per paper/prompt
        lines.append(f"Conversation {j}:")
        if members:
            for m in members:
                # author may be missing -> show UNKNOWN to avoid blank
                auth = authors[m] if authors and authors[m] else "UNKNOWN"
                lines.append(f"{display_ids[m]} | {auth}: {texts[m]}")
        else:
            lines.append("[empty]")
        lines.append("")  # blank line between clusters
    return "\n".join(lines).rstrip()


def _build_user_prompt_for_step(
    display_ids: List[str],
    authors: List[str],
    texts: List[str],
    clusters: List[List[int]],
    i: int,
) -> str:
    """
    Show full current cluster state (all messages assigned so far) + the one to assign.
    Uses simple numeric DISPLAY IDs (1..n) in the prompt only.
    """
    lines: List[str] = []
    lines.append("Current conversation clusters:")
    lines.append(_render_clusters(display_ids, authors, texts, clusters))
    lines.append("")
    lines.append("Next message to assign:")
    auth_i = authors[i] if authors and authors[i] else "UNKNOWN"
    lines.append(f"{display_ids[i]} | {auth_i}: {texts[i]}")
    return "\n".join(lines)


@dataclass
class DirectAssignmentRunner:
    client: OpenAIClient
    prompts: PromptLoader
    dataset: str = "ubuntu_irc"  # used for Ubuntu-specific rule on system messages

    def run_chunk(
        self,
        chunk_id: str,
        ids_in: List[Any],
        authors_in: Optional[List[str]],
        texts: List[str],
        is_system: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Direct Assignment:

          For i = 0..n-1:
            - Show ALL clusters formed so far (1..K), each with all members as "ID: text".
            - Show the next message i as "Next message to assign: ID: text".
            - Model returns {"conversation_id": k} where k in {0, 1..K}; 0 means new conversation.
            - If Ubuntu + message i is a system message, the correct output is 0 (we still call the model,
              but after parsing we enforce k=0 for fidelity).

          We keep deterministic numbering: the first new conversation is 1, then 2, etc., in the order created.

          Also:
            - Use simple numeric DISPLAY IDs (1..n) in the prompt only (internal ids untouched).
        """
        ids = _normalize_ids(ids_in)
        n = len(ids)
        assert n == len(texts), "ids and texts length mismatch"
        assert n == len(authors_in), "ids and authors length mismatch"
        if is_system is None:
            is_system = [False] * n
        assert len(is_system) == n, "is_system length mismatch"
        # authors may be None; normalize to list[str] of length n
        authors: List[str] = (authors_in or [""] * n)

        if len(authors) != n:
            # be forgiving: pad/truncate to n
            authors = (authors + [""] * n)[:n]

        # Choose system prompt per dataset
        if self.dataset == "ubuntu_irc":
            system_prompt = self.prompts.load("ubuntu_direct_assignment.txt")

        # DISPLAY ids for the prompt only: 1..n
        display_ids: List[str] = [str(j + 1) for j in range(n)]

        # clusters: list of lists of message indices assigned so far
        clusters: List[List[int]] = []
        # labels: output labels per message (0-based cluster indices for evaluation)
        labels: List[int] = [-1] * n

        for i in range(n):
            user_prompt = _build_user_prompt_for_step(display_ids, authors, texts, clusters, i)

            # Always call the model (no short-circuit), as in the paper
            raw = self.client.chat(system=system_prompt, user=user_prompt)
            try:
                obj = parse_json_object(raw)  # robust key stripping handled here
            except Exception as e:
                logger.warning(
                    "JSON parse failed on chunk %s msg %s: %r; raw=%r",
                    chunk_id, ids[i], e, raw
                )
                obj = {}

            # Extract conversation_id and coerce to int
            k = _coerce_int(obj.get("conversation_id"))
            # Ubuntu rule: if the message itself is a system message, enforce k=0
            if self.dataset == "ubuntu_irc" and is_system[i]:
                k = 0

            # Fallbacks: k==None, k<=0, or k>len(clusters) => start a new conversation
            if k is None or k <= 0 or k > len(clusters):
                new_idx = len(clusters)  # 0-based
                clusters.append([i])
                labels[i] = new_idx
            else:
                # Assign to existing cluster (1..K -> 0..K-1)
                labels[i] = (k - 1)
                clusters[labels[i]].append(i)

        return {
            "chunk_id": chunk_id,
            "clusters": labels,               # 0-based labels for evaluation code
            "num_conversations": len(clusters)
        }
