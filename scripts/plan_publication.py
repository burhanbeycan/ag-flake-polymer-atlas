#!/usr/bin/env python3
"""Prepare explicit, matched DFT checks for the Ag adsorption study."""

import argparse
import json
from pathlib import Path

from ag_atlas.workflow import plan

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

REFERENCE_VARIANTS = (
    ("cutoff-75", {"ecutwfc": 75.0, "ecutrho": 600.0}),
    ("kmesh-finer", {"kmesh": [6, 6, 1]}),
    ("vacuum-larger", {"vacuum": 20.0}),
    ("layers-thicker", {"layers": 6}),
    ("molecule-box-larger", {"molecule_box": 32.0}),
    ("smearing-smaller", {"degauss": 0.005}),
    ("d3-zero-damping", {"dftd3_version": 3}),
)
OLIGOMER_VARIANTS = (
    ("cutoff-75", {"ecutwfc": 75.0, "ecutrho": 600.0}),
    ("kmesh-finer", {"kmesh": [4, 4, 1]}),
    ("vacuum-larger", {"vacuum": 20.0}),
    ("layers-thicker", {"layers": 6}),
    ("molecule-box-larger", {"molecule_box": 32.0}),
    ("smearing-smaller", {"degauss": 0.005}),
    ("d3-zero-damping", {"dftd3_version": 3}),
)


def load_example(name, pseudo_dir):
    config = json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8"))
    return {**config, "pseudo_dir": str(pseudo_dir.resolve())}


def create_study(root, pseudo_dir, scope):
    """Emit inputs and a machine-readable index without starting DFT."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    variants = []
    if scope in {"reference", "all"}:
        for model, kmesh in (("nh3", [5, 5, 1]), ("benzene", [4, 4, 1])):
            base = {**load_example(model, pseudo_dir), "kmesh": kmesh}
            variants.append((model, "base", "reference", base, {}))
            variants.extend((model, label, "reference", base, change) for label, change in REFERENCE_VARIANTS)
            # Set a coarser mesh only for the enlarged cell to keep the reciprocal resolution comparable.
            variants.append((model, "lateral-larger", "reference", base, {
                "size": base["size"] + 1, "kmesh": [4, 4, 1] if model == "nh3" else [3, 3, 1],
            }))
    if scope in {"oligomer", "all"}:
        base = load_example("pilot", pseudo_dir)
        variants.append(("pilot", "base", "oligomer", base, {}))
        variants.append(("dp6", "base", "oligomer", load_example("dp6", pseudo_dir), {}))
        matched = {**base, "models": ["bpei-311", "bpei-221", "aaee", "aeae"]}
        variants.append(("matched", "base", "oligomer", matched, {}))
        variants.extend(("matched", label, "oligomer", matched, change) for label, change in OLIGOMER_VARIANTS)
        variants.append(("matched", "lateral-larger", "oligomer", matched, {"size": 7}))
        variants.append(("matched", "pose-pool-larger", "oligomer", matched, {"poses": 8}))
    cases = []
    for series, label, purpose, base, change in variants:
        case = f"{series}-{label}"
        manifest = plan({**base, **change}, root / case)
        cases.append({
            "id": case, "run": case, "series": series, "purpose": purpose,
            "baseline": label == "base", "vary": sorted(change),
        })
        (root / "cases.json").write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
        print(f"{case}: {len(manifest['jobs'])} jobs", flush=True)
        for warning in manifest["warnings"]:
            print(f"  {warning}", flush=True)
    return cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--pseudo-dir", type=Path, required=True)
    parser.add_argument("--scope", choices=["reference", "oligomer", "all"], default="all")
    args = parser.parse_args()
    create_study(args.root, args.pseudo_dir, args.scope)
