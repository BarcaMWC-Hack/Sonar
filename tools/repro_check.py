"""Verify two isolated, cold-cache builds against the recorded submission MD5."""

import argparse
import hashlib
from importlib import metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from games import require  # noqa: E402


def verify_environment():
    if sys.version_info[:3] != (3, 11, 9):
        sys.exit("Use Python 3.11.9 for the frozen release.")
    for package, version in (("numpy", "2.4.6"), ("pandas", "3.1.0"),
                             ("python-dateutil", "4.0.0")):
        if metadata.version(package) != version:
            sys.exit(f"Expected {package}=={version}.")
    if sys.platform == "win32" and metadata.version("tzdata") != "2026.6":
        sys.exit("Expected tzdata==2026.6 on Windows.")


def run(*cmd, env=None):
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect", required=True, help="MD5 of the exact file to upload")
    parser.add_argument("--existing-env", action="store_true",
                        help="reuse an environment with the exact pinned dependencies")
    args = parser.parse_args()
    sources = [Path(path) for path in require(("atd_raw", "atd_meta", "smart_tagging"))]
    with tempfile.TemporaryDirectory(prefix="sonar_repro_") as directory:
        scratch = Path(directory)
        py = sys.executable
        if args.existing_env:
            verify_environment()
        else:
            venv = scratch / "venv"
            run(py, "-m", "venv", venv)
            py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            run(py, "-m", "pip", "install", "-r", "requirements-sonar.txt")
            run(py, "-B", "-c", "from tools.repro_check import verify_environment; verify_environment()")

        hashes = []
        for index in (1, 2):
            data = scratch / f"input_{index}"
            tactical = data / "tactical"
            tactical.mkdir(parents=True)
            for source in sources:
                shutil.copyfile(source, tactical / source.name)
            env = {**os.environ, "MTAH_DATA_DIR": str(data), "OPENBLAS_NUM_THREADS": "1",
                   "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONHASHSEED": "0"}
            name = f"repro_{index}"
            run(py, "-B", "-m", "sonar.make_submission", "--name", name, "--out", scratch, env=env)
            csv = scratch / f"{name}.csv"
            with csv.open("rb") as handle:
                hashes.append(hashlib.file_digest(handle, "md5").hexdigest())
            run(py, "-B", "evaluation/validate_submission.py", csv)
            run(py, "-B", "tools/check_submission.py", csv)

        print("MD5 run 1:", hashes[0])
        print("MD5 run 2:", hashes[1])
        if hashes != [args.expect.lower()] * 2:
            sys.exit("FAILED: the two builds must both equal the expected upload hash.")
        print("PASS: both independent cold builds match the expected upload file.")


if __name__ == "__main__":
    main()
