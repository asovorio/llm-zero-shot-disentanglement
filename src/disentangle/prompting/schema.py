from __future__ import annotations
import json
from typing import Any, Dict


def parse_json_object(s: str) -> Dict[str, Any]:
    """
    Best-effort JSON parser: trims code fences and trailing text.
    """
    s = s.strip()
    # Strip markdown fences if present
    if s.startswith("```"):
        s = s.strip("`")
        # remove possible language tag
        s = s.split("\n", 1)[1] if "\n" in s else s
    # Find first balanced JSON object
    first = s.find("{")
    last = s.rfind("}")
    if first >= 0 and last > first:
        s = s[first:last+1]
    try:
        obj = json.loads(s)
    except Exception:
        # attempt to fix common trailing commas or single quotes
        s = s.replace("'", '"')
        s = s.replace(",}", "}")
        s = s.replace(",]", "]")
        obj = json.loads(s)

    if isinstance(obj, dict):
        obj = {(k.strip() if isinstance(k, str) else k): v for k, v in obj.items()}

    return obj
