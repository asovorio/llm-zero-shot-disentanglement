from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from ..api.openai_client import OpenAIClient
from ..prompting.loader import PromptLoader
from ..prompting.schema import parse_json_object
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

def _normalize_ids(ids: List[Any]) -> List[str]:
    return [str(x) for x in ids]

def _coerce_parent_id(value: Any) -> Optional[str]:
    """
    Coerce model's response_to value into a normalized string token.
    Accepts ints/floats/strings, trims whitespace/quotes; 2.0 -> "2".
    (We will then interpret this token as a 1-based display index.)
    """
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip().strip('"').strip("'")
        # Try inty float like "2.0"
        try:
            f = float(s)
            if f.is_integer():
                return str(int(f))
        except Exception:
            pass
        return s
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return None

def _render_history(display_ids: List[str], authors: List[str], texts: List[str], upto: int) -> str:
    """
    Render messages 0..upto-1 as 'DISPLAY_ID: author: text' lines.
    (Display IDs are 1..n; internal logic still uses original ids.)
    """
    lines: List[str] = []
    for j in range(upto):
        auth = authors[j] if authors and authors[j] else "UNKNOWN"
        lines.append(f"{display_ids[j]}: {auth}: {texts[j]}")
    return "\n".join(lines)

def _build_user_prompt_for_step(
    display_ids: List[str],
    authors: List[str],
    texts: List[str],
    i: int,
) -> str:
    # Matches the paper’s BR structure: all prior lines, then the next line.
    parts: List[str] = []
    parts.append("Chat log (each line shows ID followed by the message text):")
    if i == 0:
        parts.append("[no prior messages]")
    else:
        parts.append(_render_history(display_ids, authors, texts, i))
    parts.append("")
    parts.append("Next message:")
    auth_i = authors[i] if authors and authors[i] else "UNKNOWN"
    parts.append(f"{display_ids[i]}: {auth_i}: {texts[i]}")
    return "\n".join(parts)

def _collapse_parents(parents: List[int]) -> List[int]:
    """
    Turn a parent array (each i points to [0..i]) into 0-based cluster labels.
    """
    n = len(parents)
    root = list(range(n))

    def find(x: int) -> int:
        while root[x] != x:
            root[x] = root[root[x]]
            x = root[x]
        return x

    for i in range(n):
        p = parents[i]
        rp, ri = find(p), find(i)
        root[ri] = rp

    # Map roots to compact labels 0..K-1
    label_of_root: Dict[int, int] = {}
    labels = [0] * n
    next_label = 0
    for i in range(n):
        r = find(i)
        if r not in label_of_root:
            label_of_root[r] = next_label
            next_label += 1
        labels[i] = label_of_root[r]
    return labels

@dataclass
class BestResponseRunner:
    client: OpenAIClient
    prompts: PromptLoader
    dataset: str = "ubuntu_irc"  # Ubuntu rule: system => self

    def run_chunk(
        self,
        chunk_id: str,
        ids_in: List[Any],
        authors_in: Optional[List[str]],
        texts: List[str],
        is_system: Optional[List[bool]] = None,
    ) -> Dict[str, Any]:
        """
        Best-Response as in the paper:
          - For each i, show ALL prior messages (0..i-1) and the 'Next message' (i).
          - Model returns {"response_to": "<some ID>"}.
          - Ubuntu: if message i is system, parent is self (id_i).
          - Validation: if response_to is missing/not a prior id, default to self.
          - Collapse parent links into conversation clusters.

        Change in this version:
          - Use simple numeric DISPLAY IDs (1..n) in the prompt only.
          - Map model's numeric response back to an index: k in {1..i} -> parent = k-1; else self.
          - NEW: If the chosen parent is a system message, reroute to the nearest non-system ancestor; if none, use self.
        """
        ids = _normalize_ids(ids_in)
        n = len(ids)
        assert n == len(texts), "ids and texts length mismatch"
        authors: List[str] = (authors_in or [""] * n)
        if len(authors) != n:
            authors = (authors + [""] * n)[:n]
        if is_system is None:
            is_system = [False] * n
        assert len(is_system) == n, "is_system length mismatch"

        # System prompt per dataset
        if self.dataset.lower() == "ubuntu_irc":
            system_prompt = self.prompts.load("ubuntu_best_response.txt")
        else:
            system_prompt = self.prompts.load("movie_best_response.txt")

        # DISPLAY ids: 1..n (prompt only; internal logic keeps original ids)
        display_ids = [str(j + 1) for j in range(n)]

        parents: List[int] = [0] * n
        for i in range(n):
            # Build prompt with DISPLAY ids
            user_prompt = _build_user_prompt_for_step(display_ids, authors, texts, i)

            # Call model
            raw = self.client.chat(system=system_prompt, user=user_prompt)
            try:
                obj = parse_json_object(raw)  # strips whitespace from keys
            except Exception as e:
                logger.warning("JSON parse failed on %s #%s: %r; raw=%r", chunk_id, ids[i], e, raw)
                obj = {}

            # Extract and normalize response_to (token will typically be a number like "12")
            rt = obj.get("response_to")
            if rt is None:
                for k in list(obj.keys()):
                    if isinstance(k, str) and k.strip() == "response_to":
                        rt = obj[k]
                        break
            tok = _coerce_parent_id(rt)

            # Ubuntu rule: system (current) => self
            if self.dataset.lower() == "ubuntu_irc" and is_system[i]:
                parents[i] = i
                continue

            # Map numeric token to DISPLAY index (1..i), else self
            parent_idx = i  # default self
            if tok is not None:
                # If model returned the same display id as current, treat as self
                if tok == display_ids[i]:
                    parent_idx = i
                else:
                    try:
                        k = int(tok)
                        if 1 <= k <= i:   # prior only
                            parent_idx = k - 1
                        else:
                            parent_idx = i
                    except Exception:
                        parent_idx = i

            # --- Reroute if parent is system: follow its ancestor until a non-system, else self
            if parent_idx != i and is_system[parent_idx]:
                p = parent_idx
                seen = set()
                while p != parents[p] and is_system[p] and p not in seen:
                    seen.add(p)
                    p = parents[p]
                if is_system[p]:
                    parent_idx = i
                else:
                    parent_idx = p

            parents[i] = parent_idx

        labels = _collapse_parents(parents)
        return {
            "chunk_id": chunk_id,
            "clusters": labels,
            "num_conversations": len(set(labels)),
            "parents": parents,
        }
