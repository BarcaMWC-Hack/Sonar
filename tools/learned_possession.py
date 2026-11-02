"""Learned possession decoder vs the rule model, leave-one-game-out, other games only.

    python tools/learned_possession.py [--smoke]        # meant for a Kaggle GPU/CPU session

Design (fixed in advance, see the review notes): a bidirectional GRU and a frame-wise logistic
regression on the same per-frame features, both trained on proxy A labels (event-derived
possession) from two Metrica games with the ball masked at 0 / 20 / 41% of frames, then scored on
the third game with the competition's spellF1 on proxies A and B. Both are decoded with the same
two-state Viterbi as the rule model. Dead periods come from the proxy (oracle) for every method.
Switch cost: 3.0 fixed in advance, and a cost tuned only on the two training games (inner split).
Everything is sim_: proxy truth, simulated gaps, oracle dead.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import external_eval as ee  # noqa: E402
import external_joint as ej  # noqa: E402
from sonar import deadball, possession, team_shape  # noqa: E402

GAMES = (1, 2, 3)
TRAIN_MASKS = ((0.0, 0), (0.2, 1), (0.41, 2))     # (fraction, seed offset)
TEST_MASKS = ((0.0, 0), (0.41, 3), (0.41, 4), (0.41, 5))
COST = 3.0
COST_GRID = (1.5, 3.0, 5.0, 8.0)
WINDOW = 500


def build(game, frac, seed):
    """Features and labels for one game with the ball masked to `frac` (seeded)."""
    data = ej.load(game)
    if frac:
        data["ball"] = ee.mask_ball(data["ball"], frac, seed=seed * 10 + game)
    n = len(data["ball"])
    tr_a, tr_b = ej.truth(game, n, False), ej.truth(game, n, True)
    dead = tr_a == "D"
    ej.blocks(game, n, tr_a)
    possession.SIDE_SCALE = 1.5
    ball_ev = possession.ball_evidence(data, radius=1.0, falloff=1.0, slow=4.0, fast=8.0)
    x = team_shape.features(data)
    alive = deadball.graded_mask(n) & ~dead
    shape = team_shape.log_odds(x, team_shape.fit(x, ball_ev, alive))
    d_home, d_away = possession.nearest_by_team(data)
    speed = deadball.instant_ball_speed(data["ball"][:, :2].astype(np.float64))
    present = ~np.isnan(data["ball"][:, 0])
    cols = [np.minimum(np.nan_to_num(d_home, posinf=30.0), 30.0) / 30.0,
            np.minimum(np.nan_to_num(d_away, posinf=30.0), 30.0) / 30.0,
            np.minimum(np.nan_to_num(speed, nan=0.0), 20.0) / 20.0,
            present.astype(np.float64), ball_ev, shape / 4.0,
            np.nan_to_num(data["ball"][:, 1], nan=34.0) / 68.0]
    cols += [np.nan_to_num(x[:, j]) / 10.0 for j in range(x.shape[1])]
    feats = np.stack(cols, axis=1).astype(np.float32)
    y = np.where(tr_a == "H", 1, np.where(tr_a == "A", 0, -1)).astype(np.int8)
    return {"X": feats, "y": y, "dead": dead, "A": tr_a, "B": tr_b,
            "rule": ball_ev + possession.SHAPE_WEIGHT * shape}


def decode(logit, dead, cost):
    out = np.where(dead, "D", "X").astype("<U1")
    for a, b in ee.stretches(~dead):
        out[a:b] = np.where(deadball.viterbi_two_state(logit[a:b].astype(np.float64), cost),
                            "H", "A")
    return out


def f1(v, logit, cost):
    lab = decode(logit, v["dead"], cost)
    return ee.score(v["A"], lab)[0], ee.score(v["B"], lab)[0]


def fit_lr(versions):
    from sklearn.linear_model import LogisticRegression
    xs = np.concatenate([v["X"][v["y"] >= 0] for v in versions])
    ys = np.concatenate([v["y"][v["y"] >= 0] for v in versions])
    model = LogisticRegression(max_iter=300, C=1.0).fit(xs, ys)
    return lambda x: model.decision_function(x)


def fit_gru(versions, seed, epochs, steps):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.backends.cudnn.deterministic = True
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    f = versions[0]["X"].shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.gru = nn.GRU(f, 64, 2, batch_first=True, bidirectional=True, dropout=0.1)
            self.out = nn.Linear(128, 1)

        def forward(self, x):
            return self.out(self.gru(x)[0]).squeeze(-1)

    net = Net().to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        for _ in range(steps):
            xb, yb = [], []
            while len(xb) < 16:
                v = versions[int(rng.integers(len(versions)))]
                s = int(rng.integers(0, len(v["X"]) - WINDOW))
                y = v["y"][s:s + WINDOW]
                if (y >= 0).mean() < 0.3:
                    continue
                xb.append(v["X"][s:s + WINDOW])
                yb.append(y)
            x = torch.tensor(np.stack(xb), device=dev)
            y = torch.tensor(np.stack(yb).astype(np.float32), device=dev)
            mask = (y >= 0).float()
            loss = (loss_fn(net(x), y.clamp(min=0)) * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    net.eval()

    def predict(x, chunk=2000, pad=250):
        out = np.zeros(len(x), dtype=np.float32)
        with torch.no_grad():
            for s in range(0, len(x), chunk):
                lo, hi = max(0, s - pad), min(len(x), s + chunk + pad)
                z = net(torch.tensor(x[lo:hi][None], device=dev))[0].cpu().numpy()
                out[s:min(len(x), s + chunk)] = z[s - lo:s - lo + min(chunk, len(x) - s)]
        return out

    return predict


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default="learned_results.csv")
    args = parser.parse_args()
    epochs, steps, seeds = (1, 5, 1) if args.smoke else (14, 60, 2)
    games = GAMES[:2] if args.smoke else GAMES

    train = {g: [build(g, fr, 100 + so) for fr, so in TRAIN_MASKS] for g in games}
    test = {g: [(fr, build(g, fr, sd)) for fr, sd in TEST_MASKS] for g in games}
    print("features built", flush=True)

    rows = []
    for held in games:
        others = [g for g in games if g != held]

        def inner_cost(kind):
            """Cost tuned only inside the training games: fit on one, validate on the other."""
            best, scores = None, {c: [] for c in COST_GRID}
            for a, b in ((others[0], others[-1]), (others[-1], others[0])):
                if a == b:
                    continue
                m = fit_lr(train[a]) if kind == "lr" else fit_gru(train[a], 0, epochs, steps)
                for c in COST_GRID:
                    scores[c].append(np.mean([f1(v, m(v["X"]), c)[0] for v in train[b][:1]]))
            return max(COST_GRID, key=lambda c: (np.mean(scores[c]) if scores[c] else 0, c == COST))

        models = {"lr": fit_lr([v for g in others for v in train[g]])}
        for s in range(seeds):
            models["gru{}".format(s)] = fit_gru([v for g in others for v in train[g]], s,
                                                 epochs, steps)
        cost_lr = inner_cost("lr") if not args.smoke else COST
        cost_gru = inner_cost("gru") if not args.smoke else COST
        print("held {} costs lr {} gru {}".format(held, cost_lr, cost_gru), flush=True)
        for frac, v in test[held]:
            seed_logits = np.mean([models["gru{}".format(s)](v["X"]) for s in range(seeds)], axis=0)
            variants = {"rule": (v["rule"], COST), "lr@3": (models["lr"](v["X"]), COST),
                        "lr@tuned": (models["lr"](v["X"]), cost_lr),
                        "gru@3": (seed_logits, COST), "gru@tuned": (seed_logits, cost_gru)}
            for name, (z, c) in variants.items():
                a, b = f1(v, z, c)
                rows.append((held, frac, name, c, a, b))
                print("held {} mask {:.2f} {:10s} cost {} A {:.4f} B {:.4f}".format(
                    held, frac, name, c, a, b), flush=True)
    with open(args.out, "w") as f:
        f.write("held,mask,variant,cost,A,B\n")
        for r in rows:
            f.write("{},{},{},{},{:.5f},{:.5f}\n".format(*r))

    print("\nHeld-out gain over the rule (mean over folds), by test condition")
    for cond, sel in (("clean", lambda m: m == 0.0), ("masked41", lambda m: m > 0)):
        for name in ("lr@3", "lr@tuned", "gru@3", "gru@tuned"):
            ga, gb, mins = [], [], []
            for held in games:
                def mean(var, col):
                    return np.mean([r[col] for r in rows
                                    if r[0] == held and sel(r[1]) and r[2] == var])
                da, db = mean(name, 4) - mean("rule", 4), mean(name, 5) - mean("rule", 5)
                ga.append(da)
                gb.append(db)
                mins.append((da + db) / 2)
            print("  {:8s} {:10s} A {:+.4f} B {:+.4f} mean {:+.4f} min fold {:+.4f}".format(
                cond, name, np.mean(ga), np.mean(gb), np.mean(mins), np.min(mins)))
    ok = []
    for name in ("gru@3", "gru@tuned"):
        for sel in (lambda m: m == 0.0, lambda m: m > 0):
            ga, gb, mins = [], [], []
            for held in games:
                def mean(var, col):
                    return np.mean([r[col] for r in rows if r[0] == held and sel(r[1])
                                    and r[2] == var])
                da, db = mean(name, 4) - mean("rule", 4), mean(name, 5) - mean("rule", 5)
                ga.append(da)
                gb.append(db)
                mins.append((da + db) / 2)
            ok.append(np.mean(mins) >= 0.03 and np.mean(ga) >= 0.02 and np.mean(gb) >= 0.02
                      and min(mins) >= 0)
    print("CANDIDATE FOR CONFIRMATION (pre-registered bar, both costs, clean and masked):",
          "YES" if all(ok) else "NO")


if __name__ == "__main__":
    main()
