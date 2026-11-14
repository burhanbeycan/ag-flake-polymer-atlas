"""Build auditable publication tables from explicitly selected atlas runs."""

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from ase.io import read

from .workflow import PAIRS, _csv, _job_path, _json, _read_manifest, _sha

PURPOSES = {"smoke", "reference", "oligomer"}
BASE = "case_id series purpose protocol_id geometry_kind qe_version".split()
POSE_COLUMNS = BASE + (
    "model_id facet pose_id job_id status energy_ads_ev initial_site final_site "
    "contact_min_ang contact_pairs unique_donors carbon_height_ang geometry_path note"
).split()
FACET_COLUMNS = BASE + (
    "model_id facet status n_planned n_eligible best_job_id best_energy_ev "
    "sampled_energy_span_ev initial_site final_site contact_min_ang contact_pairs "
    "unique_donors carbon_height_ang numerical_settings"
).split()
SELECTIVITY_COLUMNS = BASE + (
    "model_id status energy_111_ev energy_100_ev delta_111_minus_100_ev best_job_111 best_job_100"
).split()
PAIR_COLUMNS = BASE + (
    "contrast model_a model_b status model_a_111_ev model_a_100_ev model_b_111_ev "
    "model_b_100_ev delta_a_ev delta_b_ev delta_a_minus_b_ev"
).split()
SENSITIVITY_COLUMNS = (
    "case_id baseline_id series purpose quantity model_or_contrast facet changed_settings "
    "status unit value baseline change absolute_change sign_changed"
).split()


def _csv_index(path, key):
    if not path.is_file():
        return {}
    result = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            item = row[key]
            if item in result:
                raise ValueError(f"Duplicate {key} {item!r} in {path}")
            result[item] = row
    return result


def _number(value):
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _case_specs(path):
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("The case specification must be a nonempty JSON array.")
    cases, seen, baselines = [], set(), set()
    for spec in rows:
        case_id, purpose = spec["id"], spec["purpose"]
        if not isinstance(case_id, str) or not case_id.strip() or case_id in seen:
            raise ValueError("Each case needs a nonempty, unique string ID.")
        if purpose not in PURPOSES:
            raise ValueError(f"Invalid purpose for {case_id}: {purpose}")
        series = spec["series"]
        vary = spec.get("vary", [])
        if not isinstance(series, str) or not series.strip() or not isinstance(vary, list) or any(
            not isinstance(field, str) or not field.strip() for field in vary
        ):
            raise ValueError(f"Invalid series or vary fields for {case_id}.")
        if spec.get("baseline", False):
            if (series, purpose) in baselines:
                raise ValueError(f"Multiple baselines for {purpose}/{series}.")
            baselines.add((series, purpose))
        seen.add(case_id)
        root = Path(spec["run"]).expanduser()
        cases.append({**spec, "run": (path.parent / root).resolve()})
    return cases


def _load_case(spec):
    root = spec["run"]
    manifest = _read_manifest(root)
    tables = [_csv_index(root / name, "job_id") for name in ("adsorption_results.csv", "jobs.csv")]
    planned = {job["job_id"]: job for job in manifest["jobs"]}
    if len(planned) != len(manifest["jobs"]):
        raise ValueError(f"Duplicate job IDs in {root / 'manifest.json'}")
    input_errors = {}
    for job_id, record in tables[1].items():
        if record.get("status") == "converged" and job_id in planned:
            try:
                _job_path(root, planned[job_id])
            except ValueError as error:
                input_errors[job_id] = str(error)
    return {
        **spec, "manifest": manifest, "adsorption": tables[0], "jobs": tables[1],
        "settings": manifest["protocol"]["settings"], "planned": planned, "input_errors": input_errors,
    }


def _benzene_height(root, row, slab_count):
    relative = row.get("geometry_path")
    if not relative:
        return None
    geometry = (root / relative).resolve()
    if not geometry.is_relative_to(root) or not geometry.is_file():
        return None
    atoms = read(geometry, format="extxyz")
    symbols = np.asarray(atoms.get_chemical_symbols())
    if len(atoms) <= slab_count or not np.all(symbols[:slab_count] == "Ag"):
        return None
    carbon = atoms.positions[slab_count:, 2][symbols[slab_count:] == "C"]
    if len(carbon) != 6:
        return None
    silver = atoms.positions[:slab_count, 2]
    top = silver[silver >= silver.max() - 1.25]
    return float(carbon.mean() - top.mean())


def _common(case):
    return {
        "case_id": case["id"], "series": case["series"], "purpose": case["purpose"],
        "protocol_id": case["manifest"]["protocol_id"],
        "geometry_kind": "fixed" if case["settings"].get("complex_calculation") == "scf" else "relaxed",
    }


def _validated_energy(case, job, result):
    if result.get("status") != "converged":
        return None, result.get("status", "missing_output"), result.get("message", "")
    pid = case["manifest"]["protocol_id"]
    jid = job["job_id"]
    complex_row = case["jobs"].get(jid, {})
    slab_id = result.get("slab_job_id")
    mol_id = result.get("molecule_job_id")
    slab_row = case["jobs"].get(slab_id, {})
    molecule_row = case["jobs"].get(mol_id, {})
    models = case["manifest"]["protocol"]["models"]
    if (
        result.get("protocol_id") != pid or result.get("model_id") != job["model_id"]
        or result.get("facet") != job["facet"] or result.get("pose_id") != str(job["pose_id"])
        or result.get("smiles") != models[job["model_id"]]["smiles"]
        or result.get("geometry_kind") != _common(case)["geometry_kind"]
        or slab_id != f"slab-{job['facet']}"
        or complex_row.get("role") != "complex" or slab_row.get("role") != "slab"
        or molecule_row.get("role") != "molecule" or molecule_row.get("model_id") != job["model_id"]
    ):
        return None, "incompatible_record", "The result and references must match the manifest."
    version = result.get("qe_version")
    planned = case["planned"]
    molecule_jobs = [
        candidate for candidate in planned.values()
        if candidate["role"] == "molecule" and candidate["model_id"] == job["model_id"]
    ]
    if slab_id not in planned or mol_id not in {candidate["job_id"] for candidate in molecule_jobs}:
        return None, "incompatible_reference", "Selected slab and molecule must be planned references."
    alternatives = [case["jobs"].get(candidate["job_id"], {}) for candidate in molecule_jobs]
    records = (complex_row, slab_row, *alternatives)
    if any(record.get("job_id") in case["input_errors"] for record in records):
        return None, "incompatible_input", "A converged calculation input is missing or has changed."
    if not version or any(
        r.get("status") != "converged" or r.get("protocol_id") != pid or r.get("qe_version") != version
        for r in records
    ):
        return None, "incompatible_reference", "Three converged energies with one QE version are required."
    gas_energies = [_number(record.get("energy_ev")) for record in alternatives]
    gas = _number(molecule_row.get("energy_ev"))
    if None in gas_energies or gas is None or gas != min(gas_energies):
        return None, "incompatible_reference", "Use the minimum of all planned relaxed molecule references."
    terms = [_number(record.get("energy_ev")) for record in (complex_row, slab_row, molecule_row)]
    energy = _number(result.get("energy_ads_ev"))
    if None in terms or energy is None or not math.isclose(terms[0] - terms[1] - terms[2], energy, rel_tol=0.0, abs_tol=1e-7):
        return None, "incompatible_energy", "Reported adsorption energy fails the three-term check."
    return energy, "eligible", ""


def _pose(case, job):
    result = case["adsorption"].get(job["job_id"], {})
    energy, status, note = _validated_energy(case, job, result)
    try:
        contacts = json.loads(result.get("contacts") or "[]") if energy is not None else []
        donors = {item["molecule_index"] for item in contacts}
    except (KeyError, TypeError, ValueError):
        contacts = donors = None
        note = "Could not parse donor contact metadata."
    height = None
    if energy is not None and job["model_id"] == "benzene":
        try:
            height = _benzene_height(case["run"], result, job["slab_atom_count"])
        except (OSError, ValueError, IndexError):
            note = "Could not read the geometry for the carbon-plane height."
    return {
        **_common(case), "qe_version": result.get("qe_version"), "model_id": job["model_id"],
        "facet": job["facet"], "pose_id": job["pose_id"], "job_id": job["job_id"],
        "status": status, "energy_ads_ev": energy, "initial_site": job.get("site"),
        "final_site": result.get("site_final") if energy is not None else None,
        "contact_min_ang": _number(result.get("contact_min_ang")) if energy is not None else None,
        "contact_pairs": len(contacts) if energy is not None and contacts is not None else None,
        "unique_donors": len(donors) if energy is not None and donors is not None else None,
        "carbon_height_ang": height, "geometry_path": result.get("geometry_path"), "note": note,
    }


def _facet(case, model, facet, poses):
    group = [row for row in poses if row["model_id"] == model and row["facet"] == facet]
    eligible = [row for row in group if row["status"] == "eligible"]
    versions = {row["qe_version"] for row in eligible}
    complete = len(eligible) == len(group) == case["settings"]["poses"] and len(versions) == 1
    best = min(eligible, key=lambda row: (row["energy_ads_ev"], row["pose_id"])) if complete else {}
    return {
        **_common(case), "qe_version": best.get("qe_version"), "model_id": model,
        "facet": facet, "status": "complete" if complete else "incomplete",
        "n_planned": len(group), "n_eligible": len(eligible),
        "best_job_id": best.get("job_id"), "best_energy_ev": best.get("energy_ads_ev"),
        "sampled_energy_span_ev": (
            max(row["energy_ads_ev"] for row in eligible) - best["energy_ads_ev"] if complete else None
        ),
        **{key: best.get(key) for key in (
            "initial_site", "final_site", "contact_min_ang", "contact_pairs", "unique_donors", "carbon_height_ang"
        )},
        "numerical_settings": _json({
            field: value for field, value in case["settings"].items()
            if field not in {"models", "facets"}
        }),
    }


def _selectivity(case, model, facets):
    common = _common(case)
    by_facet = {row["facet"]: row for row in facets if row["model_id"] == model}
    left, right = (by_facet.get(key, {}) for key in ("111", "100"))
    complete = (
        left.get("status") == right.get("status") == "complete"
        and left.get("qe_version") == right.get("qe_version")
    )
    e111, e100 = left.get("best_energy_ev"), right.get("best_energy_ev")
    return {
        **common, "model_id": model, "status": "complete" if complete else "incomplete",
        "qe_version": left.get("qe_version") if complete else None,
        "energy_111_ev": e111, "energy_100_ev": e100,
        "delta_111_minus_100_ev": e111 - e100 if complete else None,
        "best_job_111": left.get("best_job_id"), "best_job_100": right.get("best_job_id"),
    }


def _pair(case, contrast, a, b, facets):
    common = _common(case)
    by_key = {(row["model_id"], row["facet"]): row for row in facets}
    legs = [by_key.get((model, facet), {}) for model in (a, b) for facet in ("111", "100")]
    versions = {leg.get("qe_version") for leg in legs}
    complete = (
        common["geometry_kind"] == "relaxed" and len(versions) == 1
        and all(leg.get("status") == "complete" for leg in legs)
        and case["settings"].get("molecule_calculation", "relax") == "relax"
    )
    energies = [leg.get("best_energy_ev") for leg in legs]
    if complete:
        delta_a, delta_b = energies[0] - energies[1], energies[2] - energies[3]
    else:
        delta_a = delta_b = None
    return {
        **common, "qe_version": legs[0].get("qe_version") if complete else None,
        "contrast": contrast, "model_a": a, "model_b": b,
        "status": "complete" if complete else "incomplete",
        **dict(zip(["model_a_111_ev", "model_a_100_ev", "model_b_111_ev", "model_b_100_ev"], energies, strict=True)),
        "delta_a_ev": delta_a, "delta_b_ev": delta_b,
        "delta_a_minus_b_ev": delta_a - delta_b if complete else None,
    }


def _pool_signature(case, model, facet):
    keys = ("model_id", "facet", "pose_id", "conformer", "site", "anchor", "second_anchor", "orientation")
    return {
        tuple(job.get(k) for k in keys) for job in case["manifest"]["jobs"]
        if job["role"] == "complex" and job["model_id"] == model and job["facet"] == facet
    }


def _protocol_match(case, baseline):
    left, right = case["settings"], baseline["settings"]
    changed = sorted(key for key in left.keys() | right.keys() if left.get(key) != right.get(key))
    if not changed:
        return changed, "identical_settings"
    if set(changed) - set(case.get("vary", [])):
        return changed, "confounded_settings"
    a, b = case["manifest"]["protocol"], baseline["manifest"]["protocol"]
    if any(a[key] != b[key] for key in ("pseudopotentials", "models", "software")):
        return changed, "incompatible_provenance"
    return changed, "comparable"


def _start_digests(case, pool_keys):
    models, facets = map(set, zip(*pool_keys, strict=True))
    paths = {}
    for job in case["manifest"]["jobs"]:
        use = (
            job["role"] == "complex" and (job["model_id"], job["facet"]) in pool_keys
            or job["role"] == "molecule" and job["model_id"] in models
            or job["role"] == "slab" and job["facet"] in facets
        )
        if use:
            job_id = job["job_id"]
            file = case["run"] / "jobs" / job_id / "start.extxyz"
            if not file.is_file():
                return None
            paths[job_id] = _sha(file)
    return paths


def _sensitivity(case, baseline, key, quantity, facet, row, old, field, pool_keys, changed, status, *, unit="eV"):
    if status == "comparable":
        if row.get("status") != "complete" or old.get("status") != "complete":
            status = "incomplete"
        elif row["geometry_kind"] != old["geometry_kind"] or row["qe_version"] != old["qe_version"]:
            status = "incompatible_output"
        elif any(_pool_signature(case, *p) != _pool_signature(baseline, *p) for p in pool_keys):
            status = "different_pose_pool"
        elif not set(changed).intersection({"size", "layers", "lattice", "vacuum", "molecule_box"}):
            initial = _start_digests(case, pool_keys)
            before = _start_digests(baseline, pool_keys)
            if initial is None or before is None:
                status = "missing_start_geometry"
            elif initial != before:
                status = "different_start_geometry"
    new, previous = row.get(field), old.get(field)
    if status == "comparable" and (new is None or previous is None):
        status = "observable_unavailable"
    shift = new - previous if status == "comparable" else None
    return {
        "case_id": case["id"], "baseline_id": baseline["id"], "series": case["series"],
        "purpose": case["purpose"], "quantity": quantity, "model_or_contrast": row[key],
        "facet": facet, "changed_settings": ",".join(changed), "status": status,
        "unit": unit, "value": new, "baseline": previous, "change": shift,
        "absolute_change": abs(shift) if shift is not None else None,
        "sign_changed": (
            (new < 0 < previous or previous < 0 < new)
            if shift is not None and quantity in {"delta_E_facet", "delta_delta_E"} else None
        ),
    }


def _series_sensitivity(cases, facets, selectivity, pairs):
    baselines = {(case["series"], case["purpose"]): case for case in cases if case.get("baseline", False)}
    output = []
    for case in cases:
        baseline = baselines.get((case["series"], case["purpose"]))
        if baseline is None or baseline["id"] == case["id"]:
            continue
        changed, status = _protocol_match(case, baseline)
        old_facets = {(row["model_id"], row["facet"]): row for row in facets[baseline["id"]]}
        for row in facets[case["id"]]:
            key = (row["model_id"], row["facet"])
            output.append(_sensitivity(
                case, baseline, "model_id", "Eads", key[1], row, old_facets.get(key, {}),
                "best_energy_ev", [key], changed, status,
            ))
            if row["model_id"] in {"nh3", "benzene"}:
                field = "contact_min_ang" if row["model_id"] == "nh3" else "carbon_height_ang"
                output.append(_sensitivity(
                    case, baseline, "model_id", field, key[1], row, old_facets.get(key, {}),
                    field, [key], changed, status, unit="Å",
                ))
        old_selectivity = {row["model_id"]: row for row in selectivity[baseline["id"]]}
        for row in selectivity[case["id"]]:
            key = row["model_id"]
            output.append(_sensitivity(
                case, baseline, "model_id", "delta_E_facet", "", row, old_selectivity.get(key, {}),
                "delta_111_minus_100_ev", [(key, "111"), (key, "100")], changed, status,
            ))
        old_pairs = {row["contrast"]: row for row in pairs[baseline["id"]]}
        for row in pairs[case["id"]]:
            keys = [(model, facet) for model in (row["model_a"], row["model_b"]) for facet in ("111", "100")]
            output.append(_sensitivity(
                case, baseline, "contrast", "delta_delta_E", "", row, old_pairs.get(row["contrast"], {}),
                "delta_a_minus_b_ev", keys, changed, status,
            ))
    return output


def _plot_reference(rows, cases, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence = [row for row in rows if row["purpose"] == "reference" and row["status"] == "complete"]
    groups = defaultdict(list)
    for row in evidence:
        groups[row["series"], row["geometry_kind"], row["model_id"]].append(row)
    if not groups:
        return []
    count = len(groups)
    fig, axes = plt.subplots(count, 1, figsize=(7.0, max(3.0, 2.7 * count)), squeeze=False)
    order = {case["id"]: index for index, case in enumerate(cases)}
    for ax, ((series, kind, model), group) in zip(axes[:, 0], sorted(groups.items()), strict=True):
        labels = sorted({row["case_id"] for row in group}, key=order.__getitem__)
        lookup = {(row["case_id"], row["facet"]): row["best_energy_ev"] for row in group}
        for facet, color, marker, offset in (("111", "#0072B2", "o", -.07), ("100", "#D55E00", "s", .07)):
            y = [lookup.get((label, facet), np.nan) for label in labels]
            if not all(np.isnan(y)):
                ax.scatter(np.arange(len(labels)) + offset, y, marker=marker, color=color, label=f"Ag({facet})")
        ax.set_title(f"{model} · {series} · {kind}", loc="left", fontsize=10)
        ax.set_ylabel(r"$E_\mathrm{ads}$ (eV)")
        ax.set_xticks(range(len(labels)), labels, rotation=15, ha="right")
        ax.grid(axis="y", linewidth=.4, alpha=.4)
        ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    paths = [target / f"reference_energy.{suffix}" for suffix in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=400)
    plt.close(fig)
    return paths


def _plot_pairs(rows, target):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    evidence = [row for row in rows if row["status"] == "complete"]
    if not evidence:
        return []
    fig, ax = plt.subplots(figsize=(7., max(3., 1.5 + .45 * len(evidence))))
    for index, row in enumerate(evidence):
        left, right = row["delta_a_ev"], row["delta_b_ev"]
        ax.plot([left, right], [index, index], color="#9a9a9a", lw=1.0, zorder=1)
        ax.scatter(left, index, marker="o", color="#0072B2", zorder=2, label="Model A" if index == 0 else None)
        ax.scatter(right, index, marker="s", color="#D55E00", zorder=2, label="Model B" if index == 0 else None)
    labels = [f"{row['case_id']} · {row['model_a']} / {row['model_b']}" for row in evidence]
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel(r"$E_\mathrm{ads}(111)-E_\mathrm{ads}(100)$ (eV)")
    ax.axvline(0., color="#777777", lw=.7, ls="--")
    ax.legend(frameon=False, loc="best")
    ax.invert_yaxis()
    fig.tight_layout()
    paths = [target / f"paired_selectivity.{suffix}" for suffix in ("pdf", "png")]
    for path in paths:
        fig.savefig(path, dpi=400)
    plt.close(fig)
    return paths


def publication(cases_path, output, *, plot=False):
    """Collect tables and optionally plot complete reference-run points.

    Each case supplies an ID, run folder, series, purpose and optional baseline
    and vary list. Relative run folders resolve beside the JSON specification.
    """
    cases = [_load_case(spec) for spec in _case_specs(Path(cases_path).resolve())]
    poses, facets, selectivity, pairs = [], {}, {}, {}
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        models = case["manifest"]["protocol"]["models"]
        rows = [_pose(case, job) for job in case["manifest"]["jobs"] if job["role"] == "complex"]
        poses.extend(rows)
        facet_rows = [
            _facet(case, model, facet, rows) for model in models for facet in case["settings"]["facets"]
        ]
        facets[case["id"]] = facet_rows
        selectivity[case["id"]] = [
            _selectivity(case, model, facet_rows) for model in models
            if set(case["settings"]["facets"]) == {"111", "100"}
        ]
        pairs[case["id"]] = [
            _pair(case, contrast, a, b, facet_rows) for contrast, a, b in PAIRS
            if case["purpose"] == "oligomer" and a in models and b in models
        ]
    all_facets = [row for rows in facets.values() for row in rows]
    all_pairs = [row for rows in pairs.values() for row in rows]
    references = [row for row in all_facets if row["purpose"] == "reference"]
    all_selectivity = [row for rows in selectivity.values() for row in rows]
    sensitivity = _series_sensitivity(cases, facets, selectivity, pairs)
    data = {
        "pose_pool": (POSE_COLUMNS, poses),
        "facet_results": (FACET_COLUMNS, all_facets),
        "reference_convergence": (FACET_COLUMNS, references),
        "facet_selectivity": (SELECTIVITY_COLUMNS, all_selectivity),
        "paired_oligomers": (PAIR_COLUMNS, all_pairs),
        "numerical_sensitivity": (SENSITIVITY_COLUMNS, sensitivity),
    }
    for name, (columns, rows) in data.items():
        _csv(output / f"{name}.csv", columns, rows)
    plots = []
    if plot:
        plots = _plot_reference(references, cases, output) + _plot_pairs(all_pairs, output)
        for basename in ("reference_energy", "paired_selectivity"):
            for suffix in ("pdf", "png"):
                artifact = output / f"{basename}.{suffix}"
                if artifact not in plots:
                    artifact.unlink(missing_ok=True)
    return {**{name: len(rows) for name, (_, rows) in data.items()}, "figures": plots}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cases", type=Path, help="JSON case specification")
    parser.add_argument("output", type=Path, help="Publication analysis directory")
    parser.add_argument("--plot", action="store_true", help="Render reference PDF and PNG (requires Matplotlib)")
    args = parser.parse_args()
    try:
        summary = publication(args.cases, args.output, plot=args.plot)
    except (ValueError, OSError, KeyError, ImportError) as error:
        print(f"Publication analysis failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
