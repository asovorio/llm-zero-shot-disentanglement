#!/usr/bin/env python
from pathlib import Path
import json, random, math
from datetime import datetime, timezone
from collections import defaultdict

SRC = Path("data/processed/ubuntu_irc/ubuntu_dev.jsonl")
DST = Path("data/subsets/ubuntu_irc/ubuntu_dev.jsonl")
N_DAYS = 5
TAKE_PER_DAY = 110
SEED = 123

def ts_to_yyyymmdd(ts):
    """Return UTC 'YYYY-MM-DD' from seconds/ms or ISO8601; 'unknown' on failure."""
    try:
        # numeric path
        if isinstance(ts, (int, float)):
            x = float(ts)
        elif isinstance(ts, str) and ts.strip():
            s = ts.strip().replace("Z", "+00:00")
            # try ISO first
            try:
                dt = datetime.fromisoformat(s)
                if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
                else: dt = dt.astimezone(timezone.utc)
                return dt.strftime("%Y-%m-%d")
            except Exception:
                # then numeric string
                x = float(s)
        else:
            return "unknown"
        if x > 1e12: x /= 1000.0  # ms -> s
        return datetime.utcfromtimestamp(x).strftime("%Y-%m-%d")
    except Exception:
        return "unknown"

# Load
lines = [s for s in SRC.read_text(encoding="utf-8").splitlines() if s.strip()]
rows = [json.loads(s) for s in lines]

# Group by date (prefer explicit day/date; else derive from timestamp/ts)
by_day = defaultdict(list)
for i, r in enumerate(rows):
    day = r.get("day") or r.get("date") or ts_to_yyyymmdd(r.get("timestamp") or r.get("ts"))
    if day != "unknown":
        r["_idx"] = i  # fallback for ordering
        by_day[str(day)].append(r)

candidates = []
for day, msgs in by_day.items():
    ann = [m for m in msgs if not m.get("is_context", False)]
    if len(ann) >= TAKE_PER_DAY:
        # PAPER-FAITHFUL: preserve original file order within the day
        ann.sort(key=lambda r: r["_idx"])
        candidates.append((day, ann))

if len(candidates) < N_DAYS:
    raise SystemExit(f"Need {N_DAYS} days with >= {TAKE_PER_DAY} annotated msgs; found {len(candidates)}.")

rng = random.Random(SEED)
picked = rng.sample(candidates, N_DAYS)

# Take first TAKE_PER_DAY per picked day
out = []
for day, ann in picked:
    out.extend(ann[:TAKE_PER_DAY])

DST.parent.mkdir(parents=True, exist_ok=True)
with DST.open("w", encoding="utf-8") as f:
    for r in out:
        r.pop("_idx", None)
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"✅ Picked days: {', '.join(day for day, _ in picked)}")
print(f"➡️ Wrote {len(out)} rows (2 × {TAKE_PER_DAY}) to {DST}")
