"""Schema facts about the tactical ATD, printed as counts only.

The competition terms forbid passing match data, or any extract of it, to a hosted LLM. This
script is written so its output can be pasted anywhere: it prints field counts, row counts and
shares, never a coordinate, a speed or a row.

    python tools/probe_schema.py
"""

import collections
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
from games import require, half_video_blocks, GAME  # noqa: E402

FIELD_SEP, GROUP_SEP, VALUE_SEP = ":", ";", ","


def is_missing(value):
    return value.strip().lower() == "nan" or value.strip() == ""


def probe_raw(raw_path):
    parts_per_line = collections.Counter()
    groups_per_line = collections.Counter()
    fields_per_player = collections.Counter()
    fields_per_ball = collections.Counter()
    ball_missing_pattern = collections.Counter()   # which ball fields are NaN, as a bit pattern
    ball_outside = collections.Counter()           # x<0, x>1, y<0, y>1 among visible balls
    n_lines = 0
    graded = half_video_blocks(GAME)
    n_graded = n_graded_ball = 0

    with open(raw_path) as handle:
        for line in handle:
            n_lines += 1
            parts = line.rstrip("\n").split(FIELD_SEP)
            parts_per_line[len(parts)] += 1
            if len(parts) != 3:
                continue
            players = parts[1].split(GROUP_SEP)
            groups_per_line[len(players)] += 1
            for group in players[:3]:   # field count is uniform in practice; sample a few
                fields_per_player[len(group.split(VALUE_SEP))] += 1

            ball = parts[2].split(VALUE_SEP)
            fields_per_ball[len(ball)] += 1
            pattern = "".join("N" if is_missing(v) else "v" for v in ball)
            ball_missing_pattern[pattern] += 1

            frame = int(parts[0]) - 1
            in_graded = any(first <= frame <= last for _, first, last in graded)
            n_graded += in_graded
            if not is_missing(ball[0]):
                n_graded_ball += in_graded
                x, y = float(ball[0]), float(ball[1])
                ball_outside["x<0"] += x < 0
                ball_outside["x>1"] += x > 1
                ball_outside["y<0"] += y < 0
                ball_outside["y>1"] += y > 1
                ball_outside["visible"] += 1

    print("ATD raw: {} lines".format(n_lines))
    print("  top-level parts per line : {}".format(dict(parts_per_line)))
    print("  player groups per line   : {}".format(dict(groups_per_line)))
    print("  fields per player group  : {}".format(dict(fields_per_player)))
    print("  fields per ball group    : {}".format(dict(fields_per_ball)))
    print("  ball NaN pattern (v=value, N=NaN) : {}".format(dict(ball_missing_pattern)))
    print("  graded frames {}, of which ball visible {} ({:.1f}%)".format(
        n_graded, n_graded_ball, 100.0 * n_graded_ball / max(n_graded, 1)))
    visible = ball_outside.pop("visible", 0)
    print("  visible-ball frames outside the 0-1 box, of {}: {}".format(
        visible, dict(ball_outside)))


def probe_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    tags = collections.Counter(element.tag for element in root.iter())
    print("\nATD metadata XML: element counts")
    for tag, count in sorted(tags.items()):
        print("  {:<32} {}".format(tag, count))
    # Attribute names on the channel-like elements tell us what each value in a group means.
    for tag in ("Channel", "PlayerChannel", "Sensor", "DataFormatSpecification", "StringRegister",
                "BallChannel", "SplitRegister", "PlayerChannelRef", "BallChannelRef"):
        attrs = collections.Counter()
        for element in root.iter(tag):
            attrs.update(element.attrib.keys())
        if attrs:
            print("  attributes on <{}>: {}".format(tag, sorted(attrs)))
    ids = collections.Counter()
    for tag in ("Channel", "PlayerChannel", "BallChannel"):
        for element in root.iter(tag):
            ids[element.get("id", "")] += 1
    if ids:
        print("  channel ids (names of measured quantities, not values): {}".format(
            sorted(ids)))


if __name__ == "__main__":
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    probe_raw(raw_path)
    probe_meta(meta_path)
