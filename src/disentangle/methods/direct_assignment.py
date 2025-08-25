from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..api.openai_client import OpenAIClient
from ..prompting.loader import PromptLoader
from ..prompting.schema import parse_json_object
from ..utils.logging import setup_logger

logger = setup_logger(__name__)


def _render_clusters(ids: List[str], texts: List[str], clusters: List[List[int]]) -> str:
    """
    Show the full state of clusters (1..K) with actual contents (IDs + texts),
    as required by the paper for Direct Assignment.
    """
    if not clusters:
        return "(no conversations yet)"
    lines: List[str] = []
    for j, members in enumerate(clusters, start=1):
        lines.append(f"Conversation {j}:")
        if not members:
            lines.append("  (empty)")
        else:
            for i in members:
                lines.append(f"  {ids[i]}: {texts[i]}")
        lines.append("")  # blank line between clusters
    return "\n".join(lines).rstrip()


@dataclass
class DirectAssignmentRunner:
    client: OpenAIClient
    prompts: PromptLoader
    dataset: str = "ubuntu_irc"

    def run_chunk(
        self,
        chunk_id: str,
        ids: List[str],
        texts: List[str],
        is_system: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        assert len(ids) == len(texts), "ids and texts length mismatch"
        n = len(ids)
        if is_system is None:
            is_system = [False] * n

        ds_key = (self.dataset or "ubuntu_irc").lower()
        if ds_key.startswith("ubuntu"):
            system_prompt = self.prompts.load("ubuntu_direct_assignment.txt")
            ubuntu = True
        else:
            system_prompt = self.prompts.load("movie_direct_assignment.txt")
            ubuntu = False

        clusters: List[List[int]] = []
        labels: List[int] = []

        for i in range(n):
            clusters_block = _render_clusters(ids, texts, clusters)
            sys_tag = " [SYSTEM]" if (ubuntu and is_system[i]) else ""
            next_utt = f"{ids[i]}{sys_tag}: {texts[i]}"

            user_msg = (
                "Current conversations (1..K):\n"
                f"{clusters_block}\n\n"
                "Next utterance:\n"
                f"{next_utt}\n\n"
                "Return ONLY JSON of the form {\"conversation_id\": <int>}."
            )

            out = self.client.chat(system=system_prompt, user=user_msg)
            obj = parse_json_object(out)

            # Parse conversation_id
            cid_raw = obj.get("conversation_id", 0)
            try:
                cid = int(cid_raw)
            except Exception:
                logger.warning("Non-integer conversation_id=%r; defaulting to 0 (new).", cid_raw)
                cid = 0

            # Ubuntu rule: system messages => force new conversation (0)
            if ubuntu and is_system[i]:
                cid = 0

            if cid <= 0 or cid > len(clusters):
                clusters.append([i])
                labels.append(len(clusters) - 1)
            else:
                clusters[cid - 1].append(i)
                labels.append(cid - 1)

        return {
            "chunk_id": chunk_id,
            "clusters": labels,
            "num_conversations": len(clusters),
        }
