#!/usr/bin/env python3
"""Run an existing study sequentially on this machine, retaining failures."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import tempfile

from ag_atlas.qe import parse_output
from ag_atlas.workflow import _check_pseudos, _job_path, _lock, _read_manifest, collect, run


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


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine_identity(executable, engine_binary):
    """Record explicit engine/wrapper bytes and only relevant runtime settings."""
    paths = {"wrapper": Path(executable).resolve(), "binary": Path(engine_binary).resolve()}
    return {
        **{name: {"path": str(path), "sha256": file_hash(path)} for name, path in paths.items()},
        "environment": {name: os.environ.get(name) for name in (
            "LD_LIBRARY_PATH", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
            "MKL_NUM_THREADS", "ATLAS_MPI_TASKS", "ATLAS_BLAS_THREADS",
        )},
    }


def job_identity(root, job):
    """Verify immutable files and authentic UPFs before identifying a reusable job."""
    manifest = _read_manifest(root)
    planned = {item["job_id"]: item for item in manifest["jobs"]}
    if planned.get(job["job_id"]) != job:
        raise ValueError(f"Planned job changed: {root}/{job['job_id']}")
    _check_pseudos(manifest)
    folder = _job_path(root, job)
    return {
        "job_id": job["job_id"], "role": job["role"],
        "input_sha256": file_hash(folder / "pw.in"),
        "start_geometry_sha256": file_hash(folder / "start.extxyz"),
        # Strict whole-map matching is intentionally conservative, even for
        # unused elements: no equivalence of different protocols is inferred.
        "pseudopotentials": manifest["protocol"]["pseudopotentials"],
    }


def identity_key(identity):
    return json.dumps(identity, sort_keys=True, separators=(",", ":"))


def atomic_write(path, data):
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(data)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def cache_fresh_success(cache, study_root, case, job, engine):
    """Only called for a job actually executed successfully in this invocation."""
    if engine_identity(engine["wrapper"]["path"], engine["binary"]["path"]) != engine:
        raise ValueError("Engine fingerprint changed before recording a reusable result.")
    root = study_root / case["run"]
    folder = root / "jobs" / job["job_id"]
    identity = job_identity(root, job)
    result = parse_output(folder)
    if result.status != "converged" or not result.qe_version:
        raise ValueError(f"Reusable source is not converged: {folder}")
    entry = {
        "source_case": case["id"], "source_run": case["run"], "source_job_id": job["job_id"],
        "source_protocol_id": _read_manifest(root)["protocol_id"],
        "identity": identity, "engine": engine, "qe_version": result.qe_version,
        "output_sha256": file_hash(folder / "pw.out"),
        "source_chain": [{"case": case["id"], "run": case["run"], "job_id": job["job_id"]}],
    }
    atomic_write(folder / "execution.json", (json.dumps({
        "executed_at": timestamp(), "origin": "executed_in_current_study_invocation", **entry,
    }, indent=2) + "\n").encode())
    cache[identity_key(identity)] = entry


def reuse_case_outputs(cache, study_root, case, engine):
    """Copy validated completed outputs into exact duplicate future cases."""
    root = study_root / case["run"]
    reused = []
    with _lock(root):
        manifest = _read_manifest(root)
        for job in manifest["jobs"]:
            destination = root / "jobs" / job["job_id"]
            # A successful existing result always wins, regardless of cache.
            if parse_output(destination).status == "converged":
                continue
            identity = job_identity(root, job)
            entry = cache.get(identity_key(identity))
            if entry is None:
                continue
            if entry["engine"] != engine:
                raise ValueError("Reusable source engine fingerprint differs.")
            source_root = study_root / entry["source_run"]
            if source_root == root or (source_root / ".atlas.lock").exists():
                raise RuntimeError(f"Reusable source is active: {source_root}")
            source = source_root / "jobs" / entry["source_job_id"]
            with _lock(source), _lock(destination):
                source_manifest = _read_manifest(source_root)
                if source_manifest["protocol_id"] != entry["source_protocol_id"]:
                    raise ValueError(f"Reusable source protocol changed: {source_root}")
                source_job = next(j for j in source_manifest["jobs"] if j["job_id"] == entry["source_job_id"])
                if job_identity(source_root, source_job) != identity:
                    raise ValueError(f"Reusable source identity changed: {source}")
                result = parse_output(source)
                if result.status != "converged" or result.qe_version != entry["qe_version"]:
                    raise ValueError(f"Reusable source convergence/version changed: {source}")
                output = (source / "pw.out").read_bytes()
                if hashlib.sha256(output).hexdigest() != entry["output_sha256"]:
                    raise ValueError(f"Reusable source output changed: {source}")
                if (destination / "pw.out").exists():
                    previous = (destination / "pw.out").read_bytes()
                    history = destination / "attempt-history"
                    history.mkdir(exist_ok=True)
                    archive = history / f"before-reuse-{hashlib.sha256(previous).hexdigest()}.out"
                    if not archive.exists():
                        atomic_write(archive, previous)
                record = {
                    "schema_version": 1, "reused_at": timestamp(), **entry,
                    "destination_case": case["id"], "destination_job_id": job["job_id"],
                    "destination_protocol_id": manifest["protocol_id"],
                }
                # Save provenance first: a crash can never leave a copied output
                # without its reuse record. A partial attempt is retryable.
                atomic_write(destination / "reuse.json", (json.dumps(record, indent=2) + "\n").encode())
                atomic_write(destination / "pw.out", output)
                if parse_output(destination).status != "converged":
                    raise ValueError(f"Copied result failed destination validation: {destination}")
                reused.append(job["job_id"])
    return reused


def execute_study(root, executable, threads, timeout, *, reuse_identical=False, engine_binary=None):
    root = root.expanduser().resolve()
    cases = load_cases(root)
    engine = engine_identity(executable, engine_binary) if reuse_identical else None
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
    cache = {}
    report["reuse_identical"] = reuse_identical
    report["engine_fingerprint"] = engine
    if os.name == "posix":
        import resource

        report["virtual_memory_limit_bytes"] = resource.getrlimit(resource.RLIMIT_AS)[0]
    try:
        save_report(root, report)
        for case in cases:
            folder = root / case["run"]
            case.update(status="running", started_at=timestamp(), errors=[], completed_jobs=[], reused_jobs=[])
            save_report(root, report)

            planned, already_converged = {}, set()

            def progress(job_id, result):
                case["completed_jobs"].append({
                    "job_id": job_id, "status": result.status,
                    "energy_ev": result.energy_ev, "message": result.message,
                })
                save_report(root, report)
                print(f"{case['id']}/{job_id}: {result.status} {result.message}".rstrip(), flush=True)
                if reuse_identical and job_id not in already_converged and result.status == "converged":
                    cache_fresh_success(cache, root, case, planned[job_id], engine)

            try:
                manifest = _read_manifest(folder)
                planned = {job["job_id"]: job for job in manifest["jobs"]}
                already_converged = {
                    key for key in planned if parse_output(folder / "jobs" / key).status == "converged"
                }
                if reuse_identical:
                    if engine_identity(executable, engine_binary) != engine:
                        raise ValueError("Engine fingerprint changed during the study.")
                    case["reused_jobs"] = reuse_case_outputs(cache, root, case, engine)
                    already_converged.update(case["reused_jobs"])
                    save_report(root, report)
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
    parser.add_argument("--reuse-identical", action="store_true", help="Reuse exact duplicates completed in this invocation, with provenance.")
    parser.add_argument("--engine-binary", type=Path, help="Actual QE binary, required for --reuse-identical (not its launcher wrapper).")
    args = parser.parse_args()
    if args.threads < 1 or (args.timeout is not None and (not math.isfinite(args.timeout) or args.timeout <= 0)):
        parser.error("threads and timeout must be positive (timeout must be finite).")
    executable = shutil.which(str(Path(args.executable).expanduser()))
    if executable is None:
        parser.error(f"Quantum ESPRESSO executable is unavailable: {args.executable}")
    executable = str(Path(executable).resolve())
    if args.reuse_identical and (args.engine_binary is None or not args.engine_binary.is_file()):
        parser.error("--reuse-identical requires --engine-binary pointing to the actual QE binary.")

    def cancel(signum, frame):
        raise KeyboardInterrupt

    # Also clean up MPI ranks when stopping a detached driver with SIGTERM.
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        return execute_study(args.root, executable, args.threads, args.timeout,
                             reuse_identical=args.reuse_identical, engine_binary=args.engine_binary)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
