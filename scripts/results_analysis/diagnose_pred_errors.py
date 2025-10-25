#!/usr/bin/env python3
# scripts/pretty_print_chunk.py
from __future__ import annotations
from pathlib import Path
from collections import Counter, defaultdict
import shutil, textwrap, json, re, sys, os, colorsys, hashlib, random

# ---------- CONFIG (edit these) ----------
CHUNK_ID   = "2005-06-27_000150"   # <-- the chunk to inspect
CONFIG_YAML = "configs/batch.yaml"   # <-- same config you used for predictions/eval
PRED_PATH  = Path("data/results_batch/best_response/predictions-run1.jsonl")  # predictions JSONL
# ----------------------------------------

# ANSI helpers
ANSI_RESET = "\x1b[0m"
ansi_re = re.compile(r"\x1b\[[0-9;]*m")
def strip_ansi(s): return ansi_re.sub("", s)
def colorize(s, code): return f"\x1b[{code}m{s}{ANSI_RESET}"

def term_width(default=120):
    try:
        return shutil.get_terminal_size((default, 40)).columns
    except Exception:
        return default

# -------- Color system: Truecolor -> 256 -> basic fallback --------
def _supports_truecolor():
    ct = os.environ.get("COLORTERM", "").lower()
    return "truecolor" in ct or "24bit" in ct

def _supports_256():
    term = os.environ.get("TERM", "").lower()
    return "256color" in term

USE_TRUECOLOR = _supports_truecolor()
USE_256 = _supports_256() or not USE_TRUECOLOR  # prefer 256 if truecolor off

# Basic ANSI fallback (only if neither truecolor nor 256 is available)
BASIC_FG = ["31","32","33","34","35","36","91","92","93","94","95","96"]

def _hsv_to_rgb255(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, max(0,min(1,s)), max(0,min(1,v)))
    return int(round(r*255)), int(round(g*255)), int(round(b*255))

def _rgb_to_256(r, g, b):
    # map to 6x6x6 cube (16..231)
    ri = int(round(r / 255 * 5))
    gi = int(round(g / 255 * 5))
    bi = int(round(b / 255 * 5))
    return 16 + 36 * ri + 6 * gi + bi

def _code_from_rgb(r, g, b):
    if USE_TRUECOLOR:
        return f"38;2;{r};{g};{b}"
    if USE_256:
        return f"38;5;{_rgb_to_256(r,g,b)}"
    # fallback: approximate by hue ring
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    idx = int(round(h * (len(BASIC_FG)-1))) % len(BASIC_FG)
    return BASIC_FG[idx]

def _seed_int(s: str) -> int:
    return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF

def build_distinct_palette(n: int, seed: str):
    """
    Build n visually distinct, 'random-looking' colors (no exact repeats).
    Uses golden-ratio hue stepping + slight S/V jitter, shuffled per-seed.
    Returns (codes, hues) where codes[i] is an ANSI code string, hues[i] in [0,1).
    """
    rnd = random.Random(_seed_int(seed))
    base_h = rnd.random()
    phi = 0.61803398875  # golden ratio conjugate
    order = list(range(n))
    rnd.shuffle(order)   # randomize assignment order (not a straight rainbow)

    codes = [None] * n
    hues  = [None] * n
    for k, idx in enumerate(order):
        h = (base_h + k * phi) % 1.0
        # jitter saturation/value but keep them high for contrast
        s = 0.80 + 0.15 * rnd.random()   # 0.80..0.95
        v = 0.80 + 0.15 * rnd.random()   # 0.80..0.95
        r, g, b = _hsv_to_rgb255(h, s, v)
        codes[idx] = _code_from_rgb(r, g, b)
        hues[idx]  = h
    return codes, hues

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

    msgs = getattr(ch, "messages", None) or getattr(ch, "rows", None) or getattr(ch, "items", None)
    if msgs is None:
        sys.exit("Could not find messages on chunk (tried .messages/.rows/.items).")

    if len(msgs) != len(gold):
        sys.exit(f"Mismatch: chunk has {len(msgs)} messages but {len(gold)} gold labels.")

    # 3) Read predictions for this chunk
    preds = read_predictions_for_chunk(PRED_PATH, CHUNK_ID, expected_len=len(gold))

    # 4) Build local gold labels for coloring (compact 0..K-1)
    true_gold_ids = [int(get_field(m, "conv_id", default=g)) if isinstance(g, int) else int(g) for m, g in zip(msgs, gold)]
    seen = {}
    local_gold = []
    next_id = 0
    for gid in true_gold_ids:
        if gid not in seen:
            seen[gid] = next_id
            next_id += 1
        local_gold.append(seen[gid])

    # 5) Build palettes:
    #    - GOLD: unique colors for each gold cluster (random-looking but deterministic per CHUNK_ID)
    #    - PRED: only the most similar predicted cluster to each gold shares that gold color.
    #            Others use the next-most-similar gold's color.
    gold_labels = sorted(set(local_gold))
    pred_labels = sorted(set(preds))

    # GOLD palette
    gold_codes_list, gold_hues_list = build_distinct_palette(len(gold_labels), seed=f"{CHUNK_ID}:gold")
    gold_color = {g: gold_codes_list[i] for i, g in enumerate(gold_labels)}

    # Overlap between predicted and gold to measure similarity
    overlap = defaultdict(int)
    first_co = {}
    for i, (p, g) in enumerate(zip(preds, local_gold)):
        overlap[(p, g)] += 1
        if (p, g) not in first_co:
            first_co[(p, g)] = i

    # Helper: similarity key for sorting golds per predicted
    def simkey(p, g):
        # higher is better
        return (overlap.get((p, g), 0), -first_co.get((p, g), 10**9), -g)

    # For each GOLD, find its single "winner" predicted cluster (the one most similar to it)
    winners = {}
    for g in gold_labels:
        best_p = max(pred_labels, key=lambda p: simkey(p, g))
        winners[g] = best_p

    # Assign colors to PRED clusters:
    # - If a predicted cluster p is the winner for its top gold g_top, use gold_color[g_top].
    # - Otherwise, use the color of the next-most-similar gold (g2, g3, ...) for p.
    pred_color = {}
    for p in pred_labels:
        # golds sorted by p's similarity (best to worst)
        sorted_gs = sorted(gold_labels, key=lambda g: simkey(p, g), reverse=True)
        if not sorted_gs:
            # fallback (shouldn't happen): just cycle basic colors
            pred_color[p] = gold_codes_list[0] if gold_codes_list else "37"
            continue

        g_top = sorted_gs[0]
        if winners.get(g_top) == p:
            # p is the most similar to g_top -> share g_top's color
            pred_color[p] = gold_color[g_top]
        else:
            # find the next gold color to use
            chosen_g = None
            # Prefer the first gold (after g_top) where p is that gold's winner
            for g in sorted_gs[1:]:
                if winners.get(g) == p:
                    chosen_g = g
                    break
            # If p isn't a winner for any other gold, just take its next-most-similar gold
            if chosen_g is None:
                chosen_g = sorted_gs[1] if len(sorted_gs) > 1 else g_top
            pred_color[p] = gold_color[chosen_g]

    # Legends
    pred_counts = Counter(preds)
    gold_counts = Counter(local_gold)

    print(colorize(f"\n=== Chunk {CHUNK_ID} (split={cfg.run.split}, chunk_size={cfg.run.chunk_size}) ===", "1"))
    print(colorize("Left: PRED clusters (colored) + message          | Right: GOLD label (colored)\n", "2"))

    pred_leg = " ".join(colorize(f"P{p:02}({pred_counts[p]})", pred_color[p]) for p in pred_labels)
    gold_leg = " ".join(colorize(f"G{gl:02}({gold_counts[gl]})", gold_color[gl]) for gl in sorted(gold_counts))
    # Map Gxx -> true conv_id (limit to keep line short)
    reverse_seen = {v: k for k, v in seen.items()}
    mapping_items = [f"G{gl:02}→{reverse_seen[gl]}" for gl, _ in gold_counts.most_common()]
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
