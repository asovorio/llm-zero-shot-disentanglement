#!/usr/bin/env python3
# scripts/pretty_print_chunk.py
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import shutil, textwrap, json, re, sys

# ---------- CONFIG (edit these) ----------
CHUNK_ID   = "2004-11-14_000100"   # <-- the chunk to inspect
CONFIG_YAML = "configs/default.yaml"   # <-- same config you used for predictions/eval
PRED_PATH  = Path("data/results/direct_assignment/dev/predictions.jsonl")  # predictions JSONL
# ----------------------------------------

# ANSI colors
ANSI_RESET = "\x1b[0m"
FG_COLORS = ["31","32","33","34","35","36","91","92","93","94","95","96"]  # r,g,y,b,m,c + brights
ansi_re = re.compile(r"\x1b\[[0-9;]*m")
def colorize(s, code): return f"\x1b[{code}m{s}{ANSI_RESET}"
def strip_ansi(s): return ansi_re.sub("", s)

def term_width(default=120):
    try:
        return shutil.get_terminal_size((default, 40)).columns
    except Exception:
        return default

def read_predictions_for_chunk(pred_path: Path, chunk_id: str, expected_len: int = 50):
    if not pred_path.exists():
        sys.exit(f"Missing predictions file: {pred_path}")
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("chunk_id") == chunk_id:
            clusters = list(map(int, row["clusters"]))
            if len(clusters) != expected_len:
                sys.exit(f"{chunk_id}: expected {expected_len} preds, got {len(clusters)}")
            return clusters
    sys.exit(f"Chunk {chunk_id} not found in {pred_path}")

# Robust getters (works whether items are dicts or small objects)
def get_field(obj, *names, default=None):
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        return default
    # object with attributes
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default

def main():
    # 1) Load config & dataset like evaluate.py
    try:
        from src.disentangle.config import load_config
        from src.disentangle.datasets.ubuntu_irc import UbuntuIrcDataset
    except Exception as e:
        sys.exit("Failed to import project modules. Run from repo root. "
                 f"Import error: {e}")

    cfg = load_config(CONFIG_YAML)
    ds = UbuntuIrcDataset(
        data_root=Path(cfg.paths.processed_dir) / "ubuntu_irc",
        split=cfg.run.split,
        chunk_size=cfg.run.chunk_size,
        seed=cfg.run.seed,
    )
    chunks = ds.load_chunks()
    by_id = {c.chunk_id: c for c in chunks}
    if CHUNK_ID not in by_id:
        sys.exit(f"Chunk {CHUNK_ID} not found in dataset (split={cfg.run.split}, chunk_size={cfg.run.chunk_size}).")

    ch = by_id[CHUNK_ID]

    # 2) Extract gold labels & the 50 messages from the chunk object
    gold = getattr(ch, "gold", None)
    if gold is None:
        sys.exit("Gold labels not found on chunk. Ensure your processed export includes conv_id and your dataset loader sets ch.gold.")

    # Messages: try common attribute names
    msgs = getattr(ch, "messages", None)
    if msgs is None:
        msgs = getattr(ch, "rows", None)
    if msgs is None:
        msgs = getattr(ch, "items", None)
    if msgs is None:
        sys.exit("Could not find messages on chunk (tried .messages/.rows/.items).")

    if len(msgs) != len(gold):
        sys.exit(f"Mismatch: chunk has {len(msgs)} messages but {len(gold)} gold labels.")

    # 3) Read predictions for this chunk
    preds = read_predictions_for_chunk(PRED_PATH, CHUNK_ID, expected_len=len(gold))

    # 4) Build local gold labels for coloring (compact 0..K-1)
    #    (Keep a map to true conv_id for the right-side display.)
    true_gold_ids = [int(get_field(m, "conv_id", default=g)) if isinstance(g, int) else int(g) for m, g in zip(msgs, gold)]
    seen = {}
    local_gold = []
    next_id = 0
    for gid in true_gold_ids:
        if gid not in seen:
            seen[gid] = next_id
            next_id += 1
        local_gold.append(seen[gid])

    # 5) Color palettes
    pred_labels = sorted(set(preds))
    pred_color = {p: FG_COLORS[i % len(FG_COLORS)] for i, p in enumerate(pred_labels)}
    gold_labels = sorted(set(local_gold))
    gold_color = {g: FG_COLORS[i % len(FG_COLORS)] for i, g in enumerate(gold_labels)}

    # Legends
    from collections import Counter
    pred_counts = Counter(preds)
    gold_counts = Counter(local_gold)

    print(colorize(f"\n=== Chunk {CHUNK_ID} (split={cfg.run.split}, chunk_size={cfg.run.chunk_size}) ===", "1"))
    print(colorize("Left: PRED clusters (colored) + message          | Right: GOLD label (colored)\n", "2"))

    pred_leg = " ".join(colorize(f"P{p:02}({pred_counts[p]})", pred_color[p]) for p in pred_labels)
    gold_leg = " ".join(colorize(f"G{gl:02}({gold_counts[gl]})", gold_color[gl]) for gl in sorted(gold_counts))
    # Map Gxx -> true conv_id (limit to keep line short)
    reverse_seen = {v:k for k,v in seen.items()}
    mapping_items = [f"G{gl:02}→{reverse_seen[gl]}" for gl,_ in gold_counts.most_common()]
    gold_map = ", ".join(mapping_items[:12]) + ("" if len(mapping_items) <= 12 else ", …")

    print(colorize("Pred legend:", "1"), pred_leg)
    print(colorize("Gold legend:", "1"), gold_leg)
    print(colorize("Gold map   :", "1"), gold_map)
    print()

    # Layout
    cols = term_width()
    LEFT_W  = max(60, min(90, cols - 28))
    RIGHT_W = cols - LEFT_W - 3

    for i, (m, p_label, g_local, gid_true) in enumerate(zip(msgs, preds, local_gold, true_gold_ids)):
        author = get_field(m, "author", "user", "username", "speaker", "nick", default="UNK")
        text   = get_field(m, "text", "body", "message", "content", default="") or ""
        text   = text.replace("\n", " ")
        pfx = colorize(f"[{i:02}] P{p_label:02} ", pred_color[p_label])
        base = f"<{author}> "
        prefix_len = len(strip_ansi(pfx)) + len(base)
        msg_width = max(10, LEFT_W - prefix_len)
        msg = textwrap.shorten(text, width=msg_width, placeholder="…")
        left_colored = pfx + base + msg

        right = colorize(f"G{g_local:02}", gold_color[g_local]) + f" #{gid_true}"
        pad = max(0, LEFT_W - len(strip_ansi(left_colored)))
        print(left_colored + " " * pad + " | " + right[:RIGHT_W])

    print()

if __name__ == "__main__":
    main()
