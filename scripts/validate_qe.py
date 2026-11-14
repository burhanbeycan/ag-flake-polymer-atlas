#!/usr/bin/env python3
"""Run the five-job NH3 execution check with a real Quantum ESPRESSO engine."""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from ag_atlas.workflow import collect, plan, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="New output directory; existing runs are never overwritten.")
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--executable", default="pw.x")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=600, help="Seconds per QE job.")
    parser.add_argument("--relax", action="store_true", help="Also exercise ionic relaxation at smoke settings.")
    args = parser.parse_args()
    executable = shutil.which(args.executable)
    if executable is None:
        parser.error(f"Quantum ESPRESSO executable is unavailable: {args.executable}")
    # Resolve before run_job changes the process working directory.
    executable = str(Path(executable).resolve())
    config_path = Path(__file__).resolve().parents[1] / "examples" / "smoke.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["pseudo_dir"] = str(args.pseudo_dir.expanduser().resolve())
    if args.relax:
        for role in ("slab", "molecule", "complex"):
            config[f"{role}_calculation"] = "relax"
    started = time.monotonic()
    manifest = plan(config, args.root)

    def report(job_id, result):
        print(f"{job_id}: {result.status} {result.message}".rstrip(), flush=True)

    run(args.root, executable=executable, timeout=args.timeout, threads=args.threads, on_result=report)
    tables = collect(args.root)
    jobs = tables["jobs"]
    passed = (
        len(jobs) == 5
        and all(row["status"] == "converged" for row in jobs)
        and all(row["energy_ev"] is not None and math.isfinite(row["energy_ev"]) for row in jobs)
        and len({row["qe_version"] for row in jobs}) == 1
        and bool(jobs[0]["qe_version"])
        and len(tables["adsorption"]) == 2
        and all(row["status"] == "converged" for row in tables["adsorption"])
        and len(tables["comparison"]) == 1
        and tables["comparison"][0]["status"] == ("complete" if args.relax else "fixed_geometry")
        and (args.relax or not tables["comparison"][0]["preferred_facet"])
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=config_path.parents[1],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=config_path.parents[1],
        capture_output=True, text=True, check=True,
    ).stdout.strip())
    summary = {
        "passed": passed,
        "scope": f"{'relaxation' if args.relax else 'fixed-geometry'} execution check; not a publication or convergence study",
        "source_commit": revision,
        "source_dirty": dirty,
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "executable": executable,
        "threads": args.threads,
        "timeout_per_job_seconds": args.timeout,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "protocol_id": manifest["protocol_id"],
        "pseudopotentials": manifest["protocol"]["pseudopotentials"],
        "jobs": jobs,
        "adsorption": tables["adsorption"],
        "comparison": tables["comparison"],
    }
    destination = args.root / "validation.json"
    destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"{'PASS' if passed else 'FAIL'}: {destination}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
