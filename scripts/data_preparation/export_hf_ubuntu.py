"""
This script extracts the Ubuntu IRC dataset from HuggingFace and stores it in data/raw
"""

from __future__ import annotations
from datasets import load_dataset
from pathlib import Path
from datetime import datetime
import argparse, json, re
from collections import defaultdict, deque

DEFAULT_OUT_DIR = Path("data/raw/ubuntu_hf_export")

SPLIT_NAME_TO_STEM = {
    "train": "ubuntu_train",
    "validation": "ubuntu_validation",
    "test": "ubuntu_test",
}

# Typical raw line: "[18:42] <nick> message"
RAW_RE = re.compile(r'^\[(\d{2}):(\d{2})\]\s+<([^>]+)>\s*(.*)$')


def parse_raw_line(raw: str):
    """
    Parse '[HH:MM] <nick> text' lines.
    Returns: (hh, mm, user, text, is_system)
    """
    if not isinstance(raw, str):
        return None, None, "SYSTEM", "", True
    m = RAW_RE.match(raw)
    if m:
        hh, mm, user, text = m.groups()
        return int(hh), int(mm), user.strip(), text.strip(), False
    # System-ish line (joins, parts, topics, etc.)
    return None, None, "SYSTEM", raw.strip(), True


def to_timestamp(date_str: str, hh: int | None, mm: int | None) -> float:
    hh = 0 if hh is None else hh
    mm = 0 if mm is None else mm
    dt = datetime.strptime(f"{date_str} {hh:02d}:{mm:02d}", "%Y-%m-%d %H:%M")
    return float(dt.timestamp())


def coerce_ints(xs):
    """Coerce connections into plain Python ints. Handles numpy ints, strings, dicts with 'id'."""
    out = []
    for x in xs or []:
        try:
            out.append(int(x))
            continue
        except Exception:
            pass
        try:
            out.append(int(x.get("id")))
            continue
        except Exception:
            pass
    return out


def choose_reply_to(msg_id: int, connections):
    """
    Pick a single parent from 'connections' (no day constraints).
    Prefer the nearest EARLIER message id; otherwise the smallest id.
    """
    ints = coerce_ints(connections)
    if not ints:
        return None
    msg_id = int(msg_id)
    earlier = [c for c in ints if c < msg_id]
    return max(earlier) if earlier else min(ints)


def connected_components_undirected(nodes: list[int], edges: dict[int, set[int]]):
    """
    Simple BFS connected components on an undirected graph.
    Returns list of components (each is a set of ids).
    """
    nodes_set = set(nodes)
    seen, comps = set(), []
    for n in nodes:
        if n in seen:
            continue
        comp = set()
        q = deque([n])
        seen.add(n)
        while q:
            u = q.popleft()
            comp.add(u)
            for v in edges.get(u, ()):
                if v in nodes_set and v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def export_split(hf_split: str, out_path: Path, write_conv_id: bool):
    print(f"Loading split '{hf_split}' from jkkummerfeld/irc_disentangle (config='ubuntu')...")
    ds = load_dataset("jkkummerfeld/irc_disentangle", "ubuntu", split=hf_split)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # first pass to determine context ids per day (1000 lowest ids per day)
    ids_by_day: dict[str, list[int]] = defaultdict(list)
    for ex in ds:
        ids_by_day[ex["date"]].append(int(ex["id"]))
    context_ids: set[int] = set()
    for day, ids in ids_by_day.items():
        ids.sort()
        context_ids.update(ids[:1000])  # if a day has <1000, all of them are "context"

    # Second pass to build rows and per-day graphs
    n_rows = n_text = n_user = n_parent = n_system = n_ctx = 0

    per_day_nodes: dict[str, list[int]] = defaultdict(list)
    per_day_edges: dict[str, dict[int, set[int]]] = defaultdict(lambda: defaultdict(set))
    buffered_rows: list[tuple[str, int, dict]] = []

    for ex in ds:
        msg_id = int(ex.get("id"))
        raw = ex.get("raw")
        date_str = ex.get("date")
        connections = coerce_ints(ex.get("connections") or [])

        hh, mm, user, text, is_system = parse_raw_line(raw)
        ts = to_timestamp(date_str, hh, mm)

        # Prepare base row
        row = {
            "id": msg_id,
            "user": user,
            "text": text,
            "timestamp": ts,
            "reply_to": choose_reply_to(msg_id, connections),
            "is_system": bool(is_system),
            "is_context": (msg_id in context_ids),
        }

        # basic quality stats
        n_rows += 1
        if text: n_text += 1
        if user and user != "SYSTEM": n_user += 1
        if row["reply_to"] is not None: n_parent += 1
        if row["is_system"]: n_system += 1
        if row["is_context"]: n_ctx += 1

        buffered_rows.append((date_str, msg_id, row))

        if write_conv_id:
            per_day_nodes[date_str].append(msg_id)
            for c in connections:
                if c != msg_id:
                    per_day_edges[date_str][msg_id].add(c)
                    per_day_edges[date_str][c].add(msg_id)

    # Compute conv_id per day if requested (over ALL rows)
    if write_conv_id:
        conv_lookup: dict[int, str] = {}
        for day, nodes in per_day_nodes.items():
            edges = per_day_edges[day]
            comps = connected_components_undirected(nodes, edges)
            for i, comp in enumerate(sorted(comps, key=lambda s: (len(s), min(s)), reverse=True)):
                label = f"{day}#{i}"
                for mid in comp:
                    conv_lookup[mid] = label
        for _, _, row in buffered_rows:
            row["conv_id"] = conv_lookup.get(row["id"])

    # Write out sorted by (date, id) for stability
    buffered_rows.sort(key=lambda t: (t[0], t[1]))  # (date_str, id)
    with out_path.open("w", encoding="utf-8") as f:
        for _, _, row in buffered_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(
        f"Wrote {out_path}  "
        f"(rows={n_rows} | text={n_text} | non-system users={n_user} | with_reply_to={n_parent} | "
        f"system={n_system} | context_marked={n_ctx})"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                    help="Output directory inside the repo (default: data/raw/ubuntu_hf_export)")
    ap.add_argument("--write-conv-id", action="store_true",
                    help="Also compute per-day connected-component labels and write a 'conv_id' field")
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"],
                    choices=["train", "validation", "test"],
                    help="Which splits to export")
    args = ap.parse_args()

    for split in args.splits:
        stem = SPLIT_NAME_TO_STEM[split]
        out_path = args.out_dir / f"{stem}.jsonl"
        export_split(split, out_path, write_conv_id=args.write_conv_id)


if __name__ == "__main__":
    main()
