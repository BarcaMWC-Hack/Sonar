"""Apply the pre-registered rule of tools/external_joint.py to a sweep CSV (any machine).

    python tools/joint_report.py sweep.csv
"""

import csv
import sys
from collections import defaultdict

import numpy as np

DEFAULT = (1.0, 1.0, 4.0, 8.0, 1.5, 3.0, 0.05)     # radius falloff slow fast side cost shape
KEYS = ("A_off", "B_off", "A_on", "B_on")


def main(path):
    table = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        key = tuple(float(r[k]) for k in ("radius", "falloff", "slow", "fast", "side", "cost",
                                          "shape"))
        table[key][int(r["game"])] = [float(r[k]) for k in KEYS]
    games = sorted({g for v in table.values() for g in v})
    settings = [s for s, v in table.items() if len(v) == len(games)]
    print("{} settings x {} games".format(len(settings), len(games)))
    if DEFAULT not in table:
        sys.exit("default setting missing from the sweep")

    def sel(s, gs):
        return np.mean([np.mean(table[s][g][:2]) for g in gs])

    gains = []
    for held in games:
        rest = [g for g in games if g != held]
        pick = max(sorted(settings), key=lambda s: (sel(s, rest), s == DEFAULT))
        d = np.array(table[pick][held]) - np.array(table[DEFAULT][held])
        gains.append(d)
        print("held out {}: picked {}  A/off {:+.4f} B/off {:+.4f} A/on {:+.4f} B/on {:+.4f}"
              .format(held, pick, *d))
    g = np.array(gains)
    off = g[:, :2].mean(axis=1)
    ok = (off.mean() >= 0.02 and off.min() >= 0 and g[:, 0].mean() >= 0.01
          and g[:, 1].mean() >= 0.01 and g[:, 2:].mean() >= 0)
    print("mean {:+.4f}  min fold {:+.4f}  A {:+.4f}  B {:+.4f}  on {:+.4f}".format(
        off.mean(), off.min(), g[:, 0].mean(), g[:, 1].mean(), g[:, 2:].mean()))
    print("ADOPT under the pre-registered rule:", "YES" if ok else "NO")
    best = max(sorted(settings), key=lambda s: sel(s, games))
    print("best on all games (descriptive, not held out):", best,
          "{:+.4f} vs default".format(sel(best, games) - sel(DEFAULT, games)))


if __name__ == "__main__":
    main(sys.argv[1])
