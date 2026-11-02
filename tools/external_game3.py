"""Metrica Sample_Game_3 (EPTS tracking text + JSON events) for the other-game yardstick.

Same conventions as tools/external_eval.py: metres on a 105 x 68 pitch, players (n, P, 2), team 0
(Team A) / 1 (Team B), frame index = tracking frame - 1. Proxy truth uses the same event rules.
"""

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
D = os.path.join(ROOT, "data", "external", "metrica", "data", "Sample_Game_3")
LENGTH, WIDTH = 105.0, 68.0
RESTARTS = {"PASS", "SHOT", "SET PIECE", "RECOVERY", "CARRY"}
STOPS = {"BALL OUT", "FAULT RECEIVED"}


def _channels():
    xml = open(os.path.join(D, "Sample_Game_3_metadata.xml"), encoding="utf8").read()
    team_of = dict((p, t) for p, t in re.findall(r'<Player id="([^"]+)" teamId="([^"]+)"', xml))
    order = re.findall(r'<PlayerChannel channelId="x" id="[^"]+" playerId="([^"]+)"', xml)
    return [0 if team_of.get(p) == "FIFATMA" else 1 for p in order], \
        [team_of.get(p) for p in order]


def load():
    # Each tracking line holds 22 players: 11 of Team A then 11 of Team B (the metadata lists 35
    # channels, substitutes included, but the file does not follow that order). Checked against
    # the events: the player nearest the ball at 1,125 of 1,141 passes/shots is in the right group.
    team = [0] * 11 + [1] * 11
    raw_team = ["FIFATMA"] * 11 + ["FIFATMB"] * 11
    with open(os.path.join(D, "Sample_Game_3_tracking.txt")) as f:
        lines = f.read().splitlines()
    n = len(lines)
    players = np.full((n, len(team), 2), np.nan)
    ball = np.full((n, 2), np.nan)
    for i, line in enumerate(lines):
        _, pl, bl = line.split(":")
        vals = pl.split(";")
        for j, v in enumerate(vals[:len(team)]):
            x, y = v.split(",")
            players[i, j] = (float(x), float(y))
        bx, by = bl.split(",")[:2]
        ball[i] = (float(bx), float(by))
    players[:, :, 0] *= LENGTH
    players[:, :, 1] *= WIDTH
    ball[:, 0] *= LENGTH
    ball[:, 1] *= WIDTH
    return {"ball": ball, "players": players, "team": np.array(team, dtype=np.int8),
            "pitch": (LENGTH, WIDTH)}, raw_team


def _events():
    rows = json.load(open(os.path.join(D, "Sample_Game_3_events.json")))["data"]
    rows.sort(key=lambda e: e["start"]["frame"])
    return rows


def periods(n):
    """Per-frame period from the events: period 2 starts at its first event's frame."""
    out = np.ones(n, dtype=np.int8)
    firsts = [e["start"]["frame"] for e in _events() if e["period"] == 2]
    if firsts:
        out[min(firsts) - 1:] = 2
    return out


def truth(n):
    labels = np.full(n, "X", dtype="<U1")
    state, dead, start, period = None, False, None, None
    for e in _events():
        frame = max(0, min(n, e["start"]["frame"] - 1))
        if e["period"] != period:
            state, dead, start, period = None, False, None, e["period"]
        if start is not None and state is not None:
            labels[start:frame] = "D" if dead else state
        name = e["type"]["name"]
        if name in RESTARTS:
            state, dead = ("H" if e["team"]["id"] == "FIFATMA" else "A"), False
        elif name in STOPS:
            dead = True
        start = frame
    return labels
