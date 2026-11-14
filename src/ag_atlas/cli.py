"""Command-line entry points for the atlas."""

import argparse
import json
import sys
from pathlib import Path

from .design import catalog
from .workflow import collect, plan, run


def main():
    parser = argparse.ArgumentParser(prog="ag-atlas", description="Model adsorption on Ag(111) and Ag(100).")
    actions = parser.add_subparsers(dest="command", required=True)
    actions.add_parser("catalog", help="List the built-in molecular models.")
    actions.add_parser("gui", help="Start the local browser interface.")
    prepare = actions.add_parser("plan", help="Prepare a new directory of Quantum ESPRESSO jobs.")
    prepare.add_argument("config", type=Path)
    prepare.add_argument("root", type=Path)
    prepare.add_argument("--pseudo-dir", type=Path)
    execute = actions.add_parser("run", help="Run pending QE jobs and retry failed jobs.")
    execute.add_argument("root", type=Path)
    execute.add_argument("--executable", default="pw.x")
    execute.add_argument("--threads", default=1, type=int)
    execute.add_argument("--timeout", type=float, help="Limit a single QE job in seconds.")
    gather = actions.add_parser("collect", help="Collect QE outputs into CSV and relaxed structures.")
    gather.add_argument("root", type=Path)
    replay = actions.add_parser("benchmark", help="Replay pose selection on completed DFT pools.")
    replay.add_argument("root", type=Path)
    replay.add_argument("--budget", type=int, required=True)
    replay.add_argument("--trials", type=int, default=20)
    replay.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    try:
        if args.command == "catalog":
            for key, model in catalog().items():
                print(f"{key:12s}  {model.name}  {model.smiles}")
        elif args.command == "gui":
            from .gui import run_streamlit

            return run_streamlit(args=[])
        elif args.command == "plan":
            config = json.loads(args.config.read_text(encoding="utf-8"))
            if args.pseudo_dir:
                config["pseudo_dir"] = str(args.pseudo_dir.expanduser().resolve())
            else:
                specified = Path(config.get("pseudo_dir", ".")).expanduser()
                config["pseudo_dir"] = str((args.config.resolve().parent / specified).resolve())
            manifest = plan(config, args.root)
            print(f"{len(manifest['jobs'])} QE jobs prepared in {args.root} ({manifest['protocol_id']}).")
            for warning in manifest["warnings"]:
                print(warning, file=sys.stderr)
        elif args.command == "run":
            states = []

            def report(job_id, result):
                states.append(result.status)
                print(f"{job_id}: {result.status} {result.message}".rstrip(), flush=True)

            run(args.root, executable=args.executable, timeout=args.timeout, threads=args.threads, on_result=report)
            collect(args.root)
            return int(any(state != "converged" for state in states))
        elif args.command == "benchmark":
            from .benchmark import benchmark

            result = benchmark(args.root, args.budget, trials=args.trials, seed=args.seed)
            print(f"{result['pools']} complete pools, {result['rows']} rows: {result['path']}")
            for item in result["skipped_pools"]:
                print(f"Skipped {item['model_id']} Ag({item['facet']}): {item['reason']}", file=sys.stderr)
        else:
            tables = collect(args.root)
            print(f"Wrote {len(tables['adsorption'])} adsorption rows and {len(tables['comparison'])} surface comparisons.")
    except ImportError as error:
        print(f"{error}. Install the GUI with: pip install '.[gui]'", file=sys.stderr)
        return 2
    except (ValueError, TypeError, KeyError, OSError, RuntimeError) as error:
        print(f"{error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
