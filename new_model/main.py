"""Build the new_model submission end to end: features -> train -> decode.

This is the single entry point. The three stages are also runnable on their own:

    features.py   raw ATD -> data/cache/<game>_atd.npz + <game>_features.npz
    model.py      features + weak labels + release dead detector -> data/cache/<game>_model.joblib
    decode.py     artifact + features -> new_model/submission.csv

Typical use:

    python new_model/main.py                     # full run, reusing nothing
    python new_model/main.py --rebuild           # also re-parse the raw ATD file
    python new_model/main.py --skip-features     # reuse the feature cache
    python new_model/main.py --skip-train        # reuse the saved artifact
    python new_model/main.py --score             # add the public 5-minute check

The public five minutes is a correctness check, not a validation score, and is never used to
select a feature, a threshold or a setting. See TECHNICAL.md in this directory.

Data layout: put the organiser-supplied tactical files in data/tactical, or set MTAH_DATA_DIR to
the directory that contains tactical/. Nothing under data/ is committed.
"""

import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import DATA_DIR, GAME  # noqa: E402

SUBMISSION = os.path.join(HERE, "submission.csv")
ARTIFACT = os.path.join(DATA_DIR, "cache", "{}_model.joblib".format(GAME))
FEATURES = os.path.join(DATA_DIR, "cache", "{}_features.npz".format(GAME))


def run_stage(module, args):
    """Run one stage as a module from the repository root, so its imports resolve."""
    command = [sys.executable, "-m", module] + list(args)
    print("\n$ {}".format(" ".join(command)))
    subprocess.run(command, cwd=ROOT, check=True)


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_rows(path):
    with open(path, "rb") as handle:
        return sum(1 for _ in handle) - 1  # header


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rebuild", action="store_true",
                        help="re-parse the raw ATD file instead of reusing its cache")
    parser.add_argument("--skip-features", action="store_true",
                        help="reuse the feature cache; fail if it is missing")
    parser.add_argument("--skip-train", action="store_true",
                        help="reuse the saved model artifact; fail if it is missing")
    parser.add_argument("--split", type=int, default=None,
                        help="first validation frame passed to model.py (default: 150000)")
    parser.add_argument("--no-importance", action="store_true",
                        help="skip permutation importance (faster; does not change the artifact)")
    parser.add_argument("--score", action="store_true",
                        help="run the public 5-minute correctness check after writing")
    parser.add_argument("--expect-md5", default=None,
                        help="fail unless the written CSV has this md5")
    args = parser.parse_args()

    if not args.skip_features:
        run_stage("new_model.features", ["--rebuild"] if args.rebuild else [])
    elif not os.path.isfile(FEATURES):
        raise SystemExit("--skip-features given but {} is missing".format(FEATURES))

    if not args.skip_train:
        stage_args = []
        if args.split is not None:
            stage_args += ["--split", str(args.split)]
        if args.no_importance:
            stage_args.append("--no-importance")
        run_stage("new_model.model", stage_args)
    elif not os.path.isfile(ARTIFACT):
        raise SystemExit("--skip-train given but {} is missing".format(ARTIFACT))

    run_stage("new_model.decode", ["--score"] if args.score else [])

    if not os.path.isfile(SUBMISSION):
        raise SystemExit("decode did not write {}".format(SUBMISSION))

    got = md5(SUBMISSION)
    print("\nsubmission: {}".format(os.path.relpath(SUBMISSION, ROOT)))
    print("rows:       {}".format(count_rows(SUBMISSION)))
    print("md5:        {}".format(got))
    print("sha256:     {}".format(sha256(SUBMISSION)))

    if args.expect_md5 and got != args.expect_md5:
        raise SystemExit("md5 mismatch: expected {}, got {}".format(args.expect_md5, got))


if __name__ == "__main__":
    main()
