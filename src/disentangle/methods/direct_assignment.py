from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from ..api.openai_client import OpenAIClient
from ..prompting.loader import PromptLoader
from ..prompting.schema import parse_json_object
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class DirectAssignmentRunner:
    client: OpenAIClient
    prompts: PromptLoader

    def run_chunk(self, chunk_id: str, texts: List[str]) -> Dict[str, Any]:
        """
        Direct-Assignment: maintain cluster map; for each message, ask LLM for {assign_to, new}.
        If new=True => open new cluster with next id; else assign_to must be an existing id.
        Returns cluster assignments per message.
        """
        system = self.prompts.load("ubuntu_direct_assignment.txt")
        clusters: List[int] = []
        cluster_count = 0
        for i, text in enumerate(texts):
            state_lines = []
            for j in range(i):
                state_lines.append(f"[{j}] (c={clusters[j]}) {texts[j]}")
            state = "\n".join(state_lines) if state_lines else "(no messages yet)"

            user = (
                f"Current clusters (message -> c):\n{state}\n\n"
                f"Next message [{i}]: {text}\n"
                "Return JSON {\"assign_to\": <int>, \"new\": <bool>} "
                "where assign_to is an existing conversation id if new=false; "
                "if new=true, a new conversation will be created."
            )
            out = self.client.chat(system=system, user=user)
            obj = parse_json_object(out)
            new = bool(obj.get("new", False))
            if new or (i == 0):
                cid = cluster_count
                cluster_count += 1
            else:
                cid = int(obj.get("assign_to", max(clusters) if clusters else 0))
                if cid < 0 or cid >= cluster_count:
                    # fallback to most recent cluster
                    cid = max(clusters) if clusters else 0
            clusters.append(cid)

        return {"chunk_id": chunk_id, "clusters": clusters}
