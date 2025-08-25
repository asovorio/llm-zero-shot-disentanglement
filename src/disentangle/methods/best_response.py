from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from ..api.openai_client import OpenAIClient
from ..prompting.loader import PromptLoader
from ..prompting.schema import parse_json_object
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class BestResponseRunner:
    client: OpenAIClient
    prompts: PromptLoader
    dataset: str = "ubuntu_irc"  # <- add a dataset hint so we can apply Ubuntu rule

    def run_chunk(
        self,
        chunk_id: str,
        ids: List[str],
        texts: List[str],
        is_system: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Best-Response (paper-faithful + Ubuntu system-msg enforcement):
        - Show chat log with message IDs.
        - For Ubuntu system messages, force parent=self (no model call needed).
        - Otherwise, ask model for {"response_to": <ID>} and map to index.
        """
        assert len(ids) == len(texts), "ids and texts length mismatch"
        n = len(ids)
        if is_system is None:
            is_system = [False] * n

        system_prompt = self.prompts.load("ubuntu_best_response.txt" if self.dataset.lower().startswith("ubuntu")
                                          else "movie_best_response.txt")

        parents: List[int] = []
        id_to_index: Dict[str, int] = {}

        for i, (mid, text) in enumerate(zip(ids, texts)):
            id_to_index[str(mid)] = i

            # Ubuntu rule: system messages must be self-rooted
            if self.dataset.lower().startswith("ubuntu") and is_system[i]:
                parents.append(i)
                continue

            # --- render ONLY prior history here ---
            if i == 0:
                history = "(no prior messages)"
            else:
                history = "\n".join(f"{ids[j]}: {texts[j]}" for j in range(i))

            # --- show current message separately ---
            current = f"{mid}: {text}"

            user = (
                "Chat log so far (ID: text):\n"
                f"{history}\n\n"
                "Current message:\n"
                f"{current}\n\n"
                "Choose the parent for the *current* message by returning JSON:\n"
                "{\"response_to\": <ID>}.\n"
                "If it starts a new conversation, return its own ID."
            )

            out = self.client.chat(system=system_prompt, user=user)
            obj = parse_json_object(out)

            parent_id = str(obj.get("response_to", mid))
            p = id_to_index.get(parent_id, i)
            if p > i:  # guard against future parents
                p = i
            parents.append(p)

        clusters = self._parents_to_clusters(parents)
        return {"chunk_id": chunk_id, "parents": parents, "clusters": clusters}

    @staticmethod
    def _parents_to_clusters(parents: List[int]) -> List[int]:
        n = len(parents)
        roots: Dict[int, int] = {}
        cluster = [-1] * n
        cluster_id = 0

        def find_root(i: int) -> int:
            seen = set()
            while parents[i] != i:
                if i in seen:
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
