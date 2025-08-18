from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from ..api.openai_client import OpenAIClient
from ..prompting.loader import PromptLoader
from ..prompting.schema import parse_json_object
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class BestResponseRunner:
    client: OpenAIClient
    prompts: PromptLoader

    def run_chunk(self, chunk_id: str, texts: List[str]) -> Dict[str, Any]:
        """
        Best-Response: for each message i, ask LLM to output parent_index in [0..i].
        Build clusters by linking to parent or to itself (root).
        Returns a mapping with 'parents' and 'clusters' (per message).
        """
        system = self.prompts.load("ubuntu_best_response.txt")
        parents: List[int] = []
        for i, text in enumerate(texts):
            history = "\n".join([f"[{j}] {t}" for j, t in enumerate(texts[:i+1])])
            user = (
                "Messages (0-based index):\n"
                f"{history}\n\n"
                f"Now choose the parent for message [{i}]. "
                "Return JSON {\"parent_index\": <int in [0..i]>}. "
                "Use i itself if it starts a new thread."
            )
            out = self.client.chat(system=system, user=user)
            obj = parse_json_object(out)
            p = int(obj.get("parent_index", i))
            p = min(max(p, 0), i)  # clamp
            parents.append(p)

        # Build clusters from parent pointers (forest of trees)
        clusters = self._parents_to_clusters(parents)
        return {"chunk_id": chunk_id, "parents": parents, "clusters": clusters}

    @staticmethod
    def _parents_to_clusters(parents: List[int]) -> List[int]:
        """
        Convert parent array to conversation ids by root discovery.
        Assign cluster id = order of first appearance of root.
        """
        n = len(parents)
        roots = {}
        cluster_id = 0
        cluster = [-1] * n

        def find_root(i: int) -> int:
            seen = set()
            while parents[i] != i:
                if i in seen:  # cycle guard
                    break
                seen.add(i)
                i = parents[i]
            return i

        for i in range(n):
            r = find_root(i)
            if r not in roots:
                roots[r] = cluster_id
                cluster_id += 1
            cluster[i] = roots[r]
        return cluster
