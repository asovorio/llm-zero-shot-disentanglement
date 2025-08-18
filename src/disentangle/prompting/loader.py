from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PromptLoader:
    base_dir: Path

    def load(self, name: str) -> str:
        path = self.base_dir / name
        if not path.exists():
            # Minimal built-in fallbacks so you can run without files
            if "direct_assignment" in name:
                return (
                    "You are grouping chat messages into conversation threads.\n"
                    "Given the current state of clusters and the next message, "
                    "return a JSON object: {\"assign_to\": <int>, \"new\": <bool>}.\n"
                    "assign_to is an existing conversation id (0..K-1); if new=true, ignore assign_to and open a new conversation.\n"
                    "Only respond with valid JSON."
                )
            if "best_response" in name:
                return (
                    "You are linking each message to its parent message in the same chunk (or to itself if it starts a new thread).\n"
                    "Return JSON: {\"parent_index\": <int>} where parent_index is an index in [0..i] (0-based within chunk) "
                    "and may be equal to i to indicate 'self' (new root).\n"
                    "Only respond with valid JSON."
                )
            if "self_critic_examine" in name:
                return (
                    "You are reviewing a proposed clustering of messages into threads. "
                    "Return JSON with suggested actions per message: {\"edits\": [{\"i\": <int>, \"action\": \"assign|new|keep\", \"to\": <int|null>}, ...]}."
                )
            if "self_critic_action" in name:
                return (
                    "Apply the following edits to the current clustering. Return JSON with the updated cluster id per message: {\"assignments\": [int, ...]}."
                )
            return "Return a JSON object describing the requested action."
        return path.read_text(encoding="utf-8")
