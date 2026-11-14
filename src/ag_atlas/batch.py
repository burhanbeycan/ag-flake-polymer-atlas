"""Run one immutable atlas job per compute-array task."""

import argparse
import shutil
from pathlib import Path

from .qe import parse_output, run_job
from .workflow import _check_pseudos, _job_path, _lock, _read_manifest


def run_one(root, index, *, executable="pw.x", threads=1, timeout=None, mpi=None):
    root = Path(root).expanduser().resolve()
    manifest = _read_manifest(root)
    jobs = manifest["jobs"]
    if index < 0 or index >= len(jobs):
        raise ValueError(f"Job index must be between 0 and {len(jobs) - 1}.")
    job = jobs[index]
    folder = _job_path(root, job)
    _check_pseudos(manifest)
    if not shutil.which(executable):
        raise ValueError(f"Quantum ESPRESSO executable is unavailable: {executable}")
    launcher = (shutil.which(mpi),) if mpi else ()
    if mpi and launcher[0] is None:
        raise ValueError(f"MPI launcher is unavailable: {mpi}")
    with _lock(folder):
        result = parse_output(folder)
        if result.status != "converged":
            result = run_job(folder, executable, timeout, threads, launcher=launcher)
    return job["job_id"], result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--list", action="store_true")
    selection.add_argument("--index", type=int)
    parser.add_argument("--executable", default="pw.x")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--timeout", type=float)
    parser.add_argument("--mpi", choices=["srun"])
    args = parser.parse_args()
    try:
        if args.list:
            for job in _read_manifest(args.root)["jobs"]:
                print(job["job_id"])
            return 0
        name, result = run_one(args.root, args.index, executable=args.executable,
                               threads=args.threads, timeout=args.timeout, mpi=args.mpi)
        print(f"{name}: {result.status} {result.message}".rstrip())
        return int(result.status != "converged")
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
