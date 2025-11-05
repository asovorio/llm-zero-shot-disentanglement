from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PromptLoader:
    base_dir: Path

    def load(self, name: str) -> str:
        # Direct path first
        path = self.base_dir / name
        if not path.exists():
            # Map dataset-specific action filenames to the shared one
            if "self_critic_action" in name:
                alt = self.base_dir / "self_critic_action.txt"
                if alt.exists():
                    path = alt
                else:
                    raise FileNotFoundError(f"Prompt file not found: {alt}")
            else:
                raise FileNotFoundError(f"Prompt file not found: {path}")

        return path.read_text(encoding="utf-8")