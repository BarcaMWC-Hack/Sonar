"""Clean-environment reproduction check: the submission must regenerate byte for byte.

    python tools/repro_check.py [--expect MD5]

Builds a fresh virtual environment from requirements-sonar.txt (exact versions), then regenerates
the submission twice, each time from a cold data cache, and compares the MD5 of the two files. It
also runs the official validator on the result. Exits 0 only if both files are identical (and equal
to --expect when given). Needs the match data in data/tactical/. Prints hashes only; nothing from
the data is printed or written outside evaluation/results/.
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
    parser.add_argument("--expect", default="", help="MD5 the submission must have")
    args = parser.parse_args()

    print("Python", sys.version.split()[0])
    venv = tempfile.mkdtemp(prefix="sonar_repro_")
    try:
        run(sys.executable, "-m", "venv", venv)
        py = os.path.join(venv, "Scripts" if os.name == "nt" else "bin",
                          "python.exe" if os.name == "nt" else "python")
        run(py, "-m", "pip", "install", "-q", "-r", "requirements-sonar.txt")
        hashes, files = [], []
        for i in range(2):
            shutil.rmtree(os.path.join(ROOT, "data", "cache"), ignore_errors=True)
            name = "repro_{}".format(i)
            run(py, "-m", "sonar.make_submission", "--name", name)
            path = os.path.join(RESULTS, name + ".csv")
            files.append(path)
            hashes.append(md5(path))
        run(py, os.path.join("evaluation", "validate_submission.py"), files[-1])
        print("md5 run 1: {}\nmd5 run 2: {}".format(*hashes))
        same = hashes[0] == hashes[1]
        match = (not args.expect) or hashes[0] == args.expect.lower()
        print("identical across clean runs:", "YES" if same else "NO")
        if args.expect:
            print("equals expected {}:".format(args.expect), "YES" if match else "NO")
        for path in files:
            os.remove(path)
        sys.exit(0 if same and match else 1)
    finally:
        shutil.rmtree(venv, ignore_errors=True)


if __name__ == "__main__":
    main()
