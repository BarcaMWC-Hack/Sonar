"""Clean-environment reproduction check: the submission must regenerate byte for byte.

    python tools/repro_check.py --expect MD5_OF_THE_SUBMITTED_CSV

Builds a fresh virtual environment from requirements-sonar.txt (exact versions), then regenerates
the submission twice, each time from a cold data cache, and compares the MD5 of the two files. It
also runs the official validator on the result. Exits 0 only if both files are identical AND equal
to the recorded hash of the submitted file (--expect is required). Needs the match data in
data/tactical/. It deletes and rebuilds data/cache/ (git-ignored), writes its two scratch CSVs to a
temporary folder, and prints only hashes and the tools' own summary counts.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, "evaluation", "results")


def md5(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(*cmd):
    subprocess.run(list(cmd), cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--expect", required=True, help="MD5 of the submitted CSV")
    args = parser.parse_args()

    print("Python", sys.version.split()[0])
    venv = tempfile.mkdtemp(prefix="sonar_repro_")
    scratch = tempfile.mkdtemp(prefix="sonar_repro_out_")
    try:
        run(sys.executable, "-m", "venv", venv)
        py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin",
                          "python.exe" if os.name == "nt" else "python")
        run(py, "-m", "pip", "install", "-q", "-r", "requirements-sonar.txt")
        hashes, files = [], []
        for i in range(2):
            cache = os.path.join(ROOT, "data", "cache")
            shutil.rmtree(cache, ignore_errors=True)
            if os.path.exists(cache):
                sys.exit("could not clear " + cache)
            name = "repro_{}".format(i)
            run(py, "-m", "sonar.make_submission", "--name", name, "--out", scratch)
            path = os.path.join(scratch, name + ".csv")
            files.append(path)
            hashes.append(md5(path))
        run(py, os.path.join("evaluation", "validate_submission.py"), files[-1])
        print("md5 run 1: {}\nmd5 run 2: {}".format(*hashes))
        same = hashes[0] == hashes[1]
        match = hashes[0] == args.expect.lower()
        print("identical across clean runs:", "YES" if same else "NO")
        print("equals the submitted file's hash {}:".format(args.expect), "YES" if match else "NO")
        sys.exit(0 if same and match else 1)
    finally:
        shutil.rmtree(venv, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
