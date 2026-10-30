"""Write the detector's stoppages to a sheet a teammate checks against the tactical video.

    python tools/make_review_sheet.py

Writes data/review/stoppages_review.csv. It lists every predicted stoppage with its video
timestamps, which rule placed each edge, and the nearest Smart Tagging set piece, plus empty
columns to fill in while watching. It is a per-frame extract of the match, so it lives under data/
(git-ignored) and must never be pasted into a hosted LLM. Report back in words and totals, e.g.
"of 12 reacquire_kick restarts, 9 were about 1 s late".

This script prints counts only.
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402
from sonar import atd, deadball  # noqa: E402
from starter import load_smart_tagging  # noqa: E402

# Params
FPS = 25
OUT_PATH = os.path.join(DATA_DIR, "review", "stoppages_review.csv")
NEAR_S = 10
COLUMNS = ("id", "start_frame", "stop_frame", "start_video_time", "stop_video_time",
           "duration_s", "start_rule", "end_rule", "nearest_set_piece", "set_piece_team",
           "real_stoppage_yn", "start_offset_s", "restart_offset_s", "what_happened", "reviewer")


def video_time(frame):
    """hh:mm:ss.ff of the tactical video, ff in frames, to type into a player's seek box."""
    seconds, ff = divmod(int(frame), FPS)
    minutes, ss = divmod(seconds, 60)
    hh, mm = divmod(minutes, 60)
    return "{:02d}:{:02d}:{:02d}.{:02d}".format(hh, mm, ss, ff)


def nearest_set_piece(start, stop, setpieces):
    best, distance = None, None
    for e in setpieces:
        d = max(0, e["start_frame"] - stop, start - e["end_frame"])
        if d <= NEAR_S * FPS and (distance is None or d < distance):
            best, distance = e, d
    if best is None:
        return "", ""
    return best["tags"].get("Type", ""), best["tags"].get("Team", "")


if __name__ == "__main__":
    data = atd.load()
    events = load_smart_tagging.load()
    _, details = deadball.detect(data, events)
    setpieces = [e for e in events if deadball.code_of(e) == "set pieces"]

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        for i, s in enumerate(details["stoppages"], start=1):
            kind, team = nearest_set_piece(s["start"], s["stop"], setpieces)
            writer.writerow([i, s["start"], s["stop"], video_time(s["start"]),
                             video_time(s["stop"]), round((s["stop"] - s["start"]) / FPS, 1),
                             s["start_rule"], s["end_rule"], kind, team, "", "", "", "", ""])

    n = len(details["stoppages"])
    tagged = sum(1 for s in details["stoppages"]
                 if nearest_set_piece(s["start"], s["stop"], setpieces)[0])
    print("wrote {} stoppages ({} near a tagged set piece) to {}".format(
        n, tagged, os.path.relpath(OUT_PATH, ROOT)))
    print("data/ is git-ignored; share the file with teammates directly, never with an LLM.")
