from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
from ..api.openai_client import OpenAIClient
from ..prompting.loader import PromptLoader
from ..prompting.schema import parse_json_object
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class SelfCriticRefiner:
    client: OpenAIClient
    prompts: PromptLoader

    def refine_chunk(self, chunk_id: str, texts: List[str], clusters: List[int], max_iters: int = 1) -> Dict[str, Any]:
        """
        Simple Examine->Act loop over a single chunk.
        """
        sys_examine = self.prompts.load("ubuntu_self_critic_examine.txt")
        sys_action  = self.prompts.load("ubuntu_self_critic_action.txt")

        for _ in range(max_iters):
            # Examine
            state = "\n".join([f"[{i}] (c={c}) {t}" for i, (c, t) in enumerate(zip(clusters, texts))])
            out = self.client.chat(system=sys_examine, user=state)
            obj = parse_json_object(out)
            edits = obj.get("edits", [])

            if not edits:
                break

            # Act
            action_input = {
                "current": clusters,
                "edits": edits
            }
            out2 = self.client.chat(system=sys_action, user=json_dumps(action_input))
            obj2 = parse_json_object(out2)
            new_assignments = obj2.get("assignments")
            if isinstance(new_assignments, list) and len(new_assignments) == len(clusters):
                clusters = [int(x) for x in new_assignments]
            else:
                break

        return {"chunk_id": chunk_id, "clusters": clusters}

def json_dumps(o: Any) -> str:
    import json
    return json.dumps(o, ensure_ascii=False)
