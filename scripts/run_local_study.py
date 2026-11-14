#!/usr/bin/env python3
"""Run an existing study sequentially on this machine, retaining failures."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile

from ag_atlas.workflow import collect, run


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def load_cases(root):
    """Validate every target before any expensive job can start."""
    specs = json.loads((root / "cases.json").read_text(encoding="utf-8"))
    if not isinstance(specs, list) or not specs:
        raise ValueError("cases.json must contain a nonempty list.")
    cases, ids, paths = [], set(), set()
    for spec in specs:
        if not isinstance(spec, dict):
            raise ValueError("Every study case must be an object.")
        case_id, relative = spec.get("id"), spec.get("run")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError("Every case needs a nonempty id.")
        if not isinstance(relative, str) or not relative.strip():
            raise ValueError(f"Case {case_id} needs a run directory.")
        folder = (root / relative).resolve()
        if folder == root or not folder.is_relative_to(root):
            raise ValueError(f"Case {case_id} escapes its study directory.")
        if case_id in ids or folder in paths:
            raise ValueError(f"Duplicate study case: {case_id} ({relative}).")
        if not folder.is_dir():
            raise ValueError(f"Case directory is missing: {folder}")
        ids.add(case_id)
        paths.add(folder)
        cases.append({"id": case_id, "run": str(folder.relative_to(root)), "status": "pending"})
    return cases


def save_report(root, report):
    report["updated_at"] = timestamp()
    destination = root / "local-study-status.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


def execute_study(root, executable, threads, timeout):
    root = root.expanduser().resolve()
    cases = load_cases(root)
    lock = root / ".local-study-runner.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError(f"Study runner lock already exists: {lock}") from error
    with os.fdopen(descriptor, "w") as stream:
        stream.write(str(os.getpid()))
    report = {
        "status": "running", "started_at": timestamp(), "executable": executable,
        "threads": threads, "timeout_per_job_seconds": timeout,
        "mpi_tasks_environment": os.environ.get("ATLAS_MPI_TASKS"),
        "scope": "Existing immutable inputs; no settings or geometries changed.", "cases": cases,
    }
    if os.name == "posix":
        import resource

        report["virtual_memory_limit_bytes"] = resource.getrlimit(resource.RLIMIT_AS)[0]
    try:
        save_report(root, report)
        for case in cases:
            folder = root / case["run"]
            case.update(status="running", started_at=timestamp(), errors=[], completed_jobs=[])
            save_report(root, report)

            def progress(job_id, result):
                case["completed_jobs"].append({
                    "job_id": job_id, "status": result.status,
                    "energy_ev": result.energy_ev, "message": result.message,
                })
                save_report(root, report)
                print(f"{case['id']}/{job_id}: {result.status} {result.message}".rstrip(), flush=True)

            try:
                run(folder, executable=executable, threads=threads, timeout=timeout, on_result=progress)
            except Exception as error:
                case["errors"].append(f"run: {type(error).__name__}: {error}")
            # Collection is useful even after an execution error or incomplete jobs.
            try:
                tables = collect(folder)
                jobs, adsorption = tables["jobs"], tables["adsorption"]
                case["job_status_counts"] = dict(Counter(row["status"] for row in jobs))
                case["adsorption_status_counts"] = dict(Counter(row["status"] for row in adsorption))
                case["jobs"] = jobs
                case["passed"] = (
                    not case["errors"] and bool(jobs)
                    and all(row["status"] == "converged" for row in jobs)
                    and all(row["status"] == "converged" for row in adsorption)
                )
            except Exception as error:
                case["errors"].append(f"collect: {type(error).__name__}: {error}")
                case["passed"] = False
            case.update(status="complete" if case["passed"] else "failed", finished_at=timestamp())
            save_report(root, report)
            print(f"{case['id']}: {case['status']}", flush=True)
            for error in case["errors"]:
                print(f"  {error}", flush=True)
        passed = all(case.get("passed", False) for case in cases)
        report.update(status="complete" if passed else "failed", finished_at=timestamp())
        save_report(root, report)
        return 0 if passed else 1
    except KeyboardInterrupt:
        report.update(status="interrupted", finished_at=timestamp())
        save_report(root, report)
        return 130
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Planned study containing cases.json.")
    parser.add_argument("--executable", default="pw.x")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, help="Maximum seconds per QE job; default: no limit.")
    args = parser.parse_args()
    if args.threads < 1 or (args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0)):
        parser.error("threads and timeout must be positive (timeout must be finite).")
    executable = shutil.which(str(Path(args.executable).expanduser()))
    if executable is None:
        parser.error(f"Quantum ESPRESSO executable is unavailable: {args.executable}")
    executable = str(Path(executable).resolve())

    def cancel(signum, frame):
        raise KeyboardInterrupt

    # Also clean up MPI ranks when stopping a detached driver with SIGTERM.
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return execute_study(args.root, executable, args.threads, args.timeout)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
