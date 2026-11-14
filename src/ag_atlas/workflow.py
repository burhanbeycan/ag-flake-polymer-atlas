"""Plan, execute and collect reproducible Ag adsorption calculations."""

import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path

import numpy as np
from ase.geometry import find_mic
from ase.io import write
from rdkit import Chem

from .design import Model, _surface_sites, build_slab, catalog, conformers, placements
from .qe import Result, parse_output, run_job, write_input


DEFAULTS = {
    "facets": ["111", "100"], "poses": 4, "conformers": 4,
    "size": 6, "layers": 4, "lattice": 4.09, "vacuum": 15.0,
    "molecule_box": 25.0, "seed": 42, "ecutwfc": 60.0, "ecutrho": 480.0,
    "kmesh": [3, 3, 1], "dispersion": "d3bj", "smearing": "mv",
    "degauss": 0.01, "contact_cutoff_ang": 3.2, "nosym": True,
}
SETTINGS = {
    *DEFAULTS, "models", "custom_models", "pseudo_dir", "pseudos",
    "conv_thr", "forc_conv_thr", "etot_conv_thr", "nstep", "electron_maxstep",
    "mixing_beta", "input_dft", "dftd3_version", "dftd3_threebody",
    "slab_calculation", "molecule_calculation", "complex_calculation",
    "slab_assume_isolated", "molecule_assume_isolated", "complex_assume_isolated",
}
ADS_COLUMNS = (
    "model_id facet pose_id status site_initial site_final anchor_initial anchor_role "
    "second_anchor orientation conformer energy_ads_ev energy_complex_ev energy_slab_ev "
    "energy_molecule_ev slab_job_id molecule_job_id contact_min_ang contacts contact_cutoff_ang "
    "slab_atom_count smiles geometry_path geometry_kind protocol_id job_id qe_version message".split()
)
COMPARISON_COLUMNS = (
    "model_id status energy_111_ev energy_100_ev delta_111_minus_100_ev "
    "preferred_facet best_job_111 best_job_100 qe_version protocol_id".split()
)
PAIRED_COLUMNS = (
    "contrast model_a model_b status delta_a_ev delta_b_ev "
    "delta_a_minus_b_ev qe_version protocol_id".split()
)
PAIRS = (
    ("bpei-branch", "bpei-311", "bpei-221"),
    ("etox-ei-dp4", "aaee", "aeae"),
    ("etox-ei-dp6", "aaaeee", "aeaeae"),
)
JOB_COLUMNS = (
    "job_id role model_id facet pose_id conformer status energy_ev geometry_path "
    "max_force_ev_per_angstrom qe_version message protocol_id".split()
)


def _json(data):
    return json.dumps(data, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_integer(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer.")


def _config(config):
    unknown = set(config) - SETTINGS
    if unknown:
        raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
    settings = {**DEFAULTS, **config}
    models = catalog()
    custom_models = settings.pop("custom_models", {})
    if not isinstance(custom_models, dict):
        raise ValueError("custom_models must map model IDs to SMILES strings.")
    for key, smiles in custom_models.items():
        if key in models or not key or not key.isascii() or not key.replace("-", "").isalnum():
            raise ValueError(f"Use a unique ASCII identifier for custom model {key!r}.")
        if not isinstance(smiles, str) or not smiles.strip():
            raise ValueError(f"Provide a SMILES string for {key}.")
        models[key] = Model(key, "Custom", smiles, "User-supplied neutral molecular model.")
    for field, allowed in (("models", models), ("facets", {"111", "100"})):
        values = settings.get(field)
        if not isinstance(values, list) or not values or any(not isinstance(x, str) or x not in allowed for x in values):
            raise ValueError(f"Select valid {field}: {', '.join(sorted(allowed))}.")
        if len(set(values)) != len(values):
            raise ValueError(f"{field} must contain unique entries.")
    for field in ("poses", "conformers", "size", "layers", "seed"):
        _positive_integer(settings[field], field)
    if settings["layers"] < 2:
        raise ValueError("Use at least two Ag layers.")
    if len(settings["kmesh"]) != 3 or settings["kmesh"][2] != 1:
        raise ValueError("Use a slab k mesh [kx, ky, 1].")
    for n in settings["kmesh"]:
        _positive_integer(n, "k point count")
    for field in ("lattice", "vacuum", "molecule_box", "ecutwfc", "ecutrho", "contact_cutoff_ang"):
        if not math.isfinite(settings[field]) or settings[field] <= 0:
            raise ValueError(f"{field} must be positive and finite.")
    if settings["ecutrho"] < settings["ecutwfc"]:
        raise ValueError("ecutrho must be at least ecutwfc.")
    selected = {key: models[key] for key in settings["models"]}
    if settings.get("slab_calculation", "relax") != settings.get("complex_calculation", "relax"):
        raise ValueError("Slab and complex must use the same calculation type.")
    if settings.get("complex_assume_isolated", settings.get("slab_assume_isolated")) != settings.get("slab_assume_isolated"):
        raise ValueError("Slab and complex must use the same isolation scheme.")
    required = {"Ag"}
    for model in selected.values():
        mol = Chem.MolFromSmiles(model.smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES for {model.name}.")
        required.update(atom.GetSymbol() for atom in Chem.AddHs(mol).GetAtoms())
    folder = Path(settings.get("pseudo_dir") or ".").expanduser().resolve()
    pseudos = {}
    for element in sorted(required):
        filename = settings.get("pseudos", {}).get(element)
        if not filename or Path(filename).name != filename:
            raise ValueError(f"Provide a UPF filename for {element}.")
        path = folder / filename
        if not path.is_file() or not path.stat().st_size:
            raise ValueError(f"UPF for {element} is missing or empty: {path}")
        pseudos[element] = {"filename": filename, "sha256": _sha(path)}
    settings["pseudo_dir"] = str(folder)
    settings["pseudos"] = {el: item["filename"] for el, item in pseudos.items()}
    return settings, selected, pseudos


def plan(config, root):
    """Write a new immutable calculation plan and its job inputs."""
    root = Path(root).expanduser().resolve()
    if root.exists():
        raise ValueError(f"Choose a new run folder: {root}")
    settings, selected, pseudos = _config(config)
    protocol = {
        "settings": {k: v for k, v in settings.items() if k not in {"pseudo_dir", "pseudos"}},
        "pseudopotentials": pseudos,
        "models": {key: vars(value) for key, value in selected.items()},
        "software": {name: version(name) for name in ("ase", "rdkit", "numpy")},
    }
    protocol_id = hashlib.sha256(_json(protocol).encode()).hexdigest()[:16]
    jobs, warnings = [], []
    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{root.name}-", dir=root.parent))
    try:
        def add(job_id, role, atoms, **fields):
            folder = stage / "jobs" / job_id
            folder.mkdir(parents=True)
            write(folder / "start.extxyz", atoms, format="extxyz", write_info=False)
            write_input(folder / "pw.in", atoms, settings, role)
            jobs.append({
                "job_id": job_id, "role": role, **fields,
                "input_sha256": _sha(folder / "pw.in"),
            })

        slabs = {}
        for facet in settings["facets"]:
            slab = build_slab(facet, settings["size"], settings["layers"], settings["lattice"], settings["vacuum"])
            slabs[facet] = slab
            add(f"slab-{facet}", "slab", slab, facet=facet)
        for key, model in selected.items():
            candidates = conformers(model.smiles, settings["conformers"], settings["seed"])
            donor_roles = candidates[0].info["donor_roles"]
            for index, candidate in enumerate(candidates):
                molecule = candidate.copy()
                if np.ptp(molecule.positions, axis=0).max() > settings["molecule_box"] - 6:
                    raise ValueError(f"Molecular box is too small for {key}; increase molecule_box.")
                molecule.set_cell([settings["molecule_box"]] * 3)
                molecule.set_pbc(True)
                molecule.center()
                add(f"mol-{key}-{index:02}", "molecule", molecule, model_id=key, conformer=index)
            for facet, slab in slabs.items():
                poses = placements(slab, facet, candidates, settings["poses"], settings["seed"])
                if len(poses) < settings["poses"]:
                    warnings.append(f"{key} Ag({facet}) has {len(poses)} of {settings['poses']} requested starting poses.")
                for index, (metadata, geometry) in enumerate(poses, start=1):
                    add(f"ads-{key}-{facet}-{index:02}", "complex", geometry, model_id=key,
                        facet=facet, pose_id=index, slab_atom_count=len(slab),
                        donor_roles=donor_roles, **metadata)
        manifest = {
            "schema_version": 1, "protocol_id": protocol_id, "protocol": protocol,
            "pseudo_dir": settings["pseudo_dir"], "jobs": jobs, "warnings": warnings,
        }
        (stage / "manifest.json").write_text(_json(manifest) + "\n", encoding="utf-8")
        if root.exists():
            raise ValueError(f"Run folder already exists: {root}")
        stage.rename(root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return manifest


def _read_manifest(root):
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported calculation manifest schema.")
    expected_id = hashlib.sha256(_json(manifest["protocol"]).encode()).hexdigest()[:16]
    if manifest["protocol_id"] != expected_id:
        raise ValueError("The saved calculation protocol was modified.")
    return manifest


@contextmanager
def _lock(root):
    path = root / ".atlas.lock"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise RuntimeError(f"Run folder is in use: {root}") from exc
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(str(os.getpid()))
        yield
    finally:
        path.unlink(missing_ok=True)


def _check_pseudos(manifest):
    for spec in manifest["protocol"]["pseudopotentials"].values():
        path = Path(manifest["pseudo_dir"]) / spec["filename"]
        if not path.is_file() or _sha(path) != spec["sha256"]:
            raise ValueError(f"Missing or changed pseudopotential: {path}")


def _job_path(root, job):
    key = job["job_id"]
    if not key.isascii() or not key.replace("-", "").isalnum():
        raise ValueError(f"Invalid job ID: {key!r}")
    folder = root / "jobs" / key
    input_path = folder / "pw.in"
    if not input_path.is_file() or _sha(input_path) != job["input_sha256"]:
        raise ValueError(f"Missing or changed input: {input_path}")
    return folder


def run(root, executable="pw.x", timeout=None, threads=1, on_result=None):
    """Run pending jobs and retry jobs with unusable output."""
    root = Path(root).expanduser().resolve()
    manifest = _read_manifest(root)
    _check_pseudos(manifest)
    if shutil.which(str(Path(executable).expanduser()) if "/" in executable else executable) is None:
        raise ValueError(f"Quantum ESPRESSO executable is unavailable: {executable}")
    with _lock(root):
        for job in manifest["jobs"]:
            folder = _job_path(root, job)
            with _lock(folder):
                result = parse_output(folder)
                if result.status != "converged":
                    result = run_job(folder, executable=executable, timeout=timeout, threads=threads)
            if on_result is not None:
                on_result(job["job_id"], result)


def _contacts(atoms, job, cutoff):
    n = job["slab_atom_count"]
    coords = atoms.positions
    surface = np.where(coords[:n, 2] > coords[:n, 2].max() - 1.25)[0]
    contacts, minimum, closest = [], None, None
    for raw_index, role in job["donor_roles"].items():
        index = int(raw_index)
        _, lengths = find_mic(coords[surface] - coords[n + index], atoms.cell, pbc=(True, True, False))
        for ag, distance in zip(surface, lengths, strict=True):
            value = float(distance)
            if minimum is None or value < minimum:
                minimum, closest = value, index
            if value <= cutoff:
                contacts.append({
                    "molecule_index": index, "ag_index": int(ag), "element": atoms[n + index].symbol,
                    "distance_ang": round(value, 6), "role": role,
                })
    contacts.sort(key=lambda contact: contact["distance_ang"])
    if not contacts:
        return minimum, contacts, ""
    # Ideal site label from the final donor projection; preserve exact atom indices above.
    sites = _surface_sites(job["ideal_slab"])
    xy = coords[n + closest, :2]
    targets = np.array([[*target, 0.] for _, target in sites])
    # ASE surface translations use the primitive cell, while PW outputs use the supercell.
    primitive = atoms.cell.copy()
    primitive[0] /= job["lateral_size"]
    primitive[1] /= job["lateral_size"]
    _, distance = find_mic(targets - np.array([*xy, 0.]), primitive, pbc=(True, True, False))
    return minimum, contacts, sites[int(np.argmin(distance))][0]


def _csv(path, columns, rows):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False) as tmp:
        try:
            writer = csv.DictWriter(tmp, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            tmp.close()
            os.replace(tmp.name, path)
        finally:
            Path(tmp.name).unlink(missing_ok=True)


def _compare(adsorption, models, protocol_id, expected_facets):
    rows = []
    for model in models:
        model_rows = [row for row in adsorption if row["model_id"] == model]
        best = {}
        for facet in ("111", "100"):
            eligible = [
                row for row in model_rows if row["facet"] == facet and row["status"] == "converged"
            ]
            if eligible:
                best[facet] = min(eligible, key=lambda row: row["energy_ads_ev"])
        both = len(best) == 2
        versions = {row["qe_version"] for row in best.values()}
        consistent = both and len(versions) == 1
        complete = consistent and all(row["status"] == "converged" for row in model_rows)
        complete = complete and all(
            sum(row["facet"] == facet for row in model_rows) >= expected_facets[facet]
            for facet in ("111", "100")
        )
        delta = best["111"]["energy_ads_ev"] - best["100"]["energy_ads_ev"] if consistent else None
        preference = ""
        relaxed = complete and all(row["geometry_kind"] == "relaxed" for row in best.values())
        if relaxed:
            preference = "111" if delta < 0 else "100" if delta > 0 else "tie"
        rows.append({
            "model_id": model,
            "status": "complete" if relaxed else "fixed_geometry" if complete else "provisional" if consistent else "incomplete",
            "energy_111_ev": best.get("111", {}).get("energy_ads_ev"),
            "energy_100_ev": best.get("100", {}).get("energy_ads_ev"),
            "delta_111_minus_100_ev": delta, "preferred_facet": preference,
            "best_job_111": best.get("111", {}).get("job_id"),
            "best_job_100": best.get("100", {}).get("job_id"), "protocol_id": protocol_id,
            "qe_version": next(iter(versions)) if len(versions) == 1 else "",
        })
    return rows


def _paired_effects(comparisons, protocol_id):
    by_model = {row["model_id"]: row for row in comparisons}
    results = []
    for name, model_a, model_b in PAIRS:
        if model_a not in by_model or model_b not in by_model:
            continue
        a, b = by_model[model_a], by_model[model_b]
        same_version = a["qe_version"] and a["qe_version"] == b["qe_version"]
        ready = bool(same_version and a["preferred_facet"] and b["preferred_facet"])
        delta_a, delta_b = a["delta_111_minus_100_ev"], b["delta_111_minus_100_ev"]
        results.append({
            "contrast": name, "model_a": model_a, "model_b": model_b,
            "status": "complete" if ready else "incomplete",
            "delta_a_ev": delta_a, "delta_b_ev": delta_b,
            "delta_a_minus_b_ev": delta_a - delta_b if ready else None,
            "qe_version": a["qe_version"] if same_version else "", "protocol_id": protocol_id,
        })
    return results


def collect(root):
    """Write per-job, adsorption and surface comparison tables from saved outputs."""
    root = Path(root).expanduser().resolve()
    manifest = _read_manifest(root)
    settings = manifest["protocol"]["settings"]
    protocol_id = manifest["protocol_id"]
    models = manifest["protocol"]["models"]
    cutoff = settings["contact_cutoff_ang"]
    ideal_slabs = {
        facet: build_slab(facet, settings["size"], settings["layers"], settings["lattice"], settings["vacuum"])
        for facet in settings["facets"]
    }
    outputs, job_rows, adsorption = {}, [], []
    with _lock(root):
        for job in manifest["jobs"]:
            job_id = job["job_id"]
            try:
                folder = _job_path(root, job)
                with _lock(folder):
                    result = parse_output(folder)
            except RuntimeError as error:
                result = Result("pending", message=str(error))
            except ValueError as error:
                result = Result("failed", message=str(error))
            outputs[job_id] = result
            geometry = ""
            if result.atoms is not None:
                name = "single-point" if settings.get(f"{job['role']}_calculation") == "scf" else "relaxed"
                geometry = f"jobs/{job_id}/{name}.extxyz"
                write(root / geometry, result.atoms, format="extxyz", write_info=False)
            row = {key: job.get(key) for key in ("job_id", "role", "model_id", "facet", "pose_id", "conformer")}
            job_rows.append({
                **row, "protocol_id": protocol_id, "status": result.status, "energy_ev": result.energy_ev,
                "geometry_path": geometry, "message": result.message, "qe_version": result.qe_version,
                "max_force_ev_per_angstrom": result.max_force_ev_per_angstrom,
            })
        for job, details in zip(manifest["jobs"], job_rows, strict=True):
            if job["role"] != "complex":
                continue
            key, facet, job_id = job["model_id"], job["facet"], job["job_id"]
            complex_result = outputs[job_id]
            slab_id = f"slab-{facet}"
            slab = outputs[slab_id]
            mols = {j["job_id"]: outputs[j["job_id"]] for j in manifest["jobs"]
                    if j["role"] == "molecule" and j["model_id"] == key}
            valid_mols = [(name, res) for name, res in mols.items() if res.status == "converged"]
            mol_id, mol = min(valid_mols, key=lambda item: item[1].energy_ev) if valid_mols else ("", Result("pending"))
            status, message = complex_result.status, complex_result.message
            energy, minimum, contacts, site_final = None, None, [], ""
            if status == "converged":
                minimum, contacts, site_final = _contacts(
                    complex_result.atoms, {**job, "ideal_slab": ideal_slabs[facet], "lateral_size": settings["size"]}, cutoff
                )
                if slab.status != "converged" or len(valid_mols) != len(mols):
                    status, message = "reference_incomplete", "Clean slab or molecular conformer reference is incomplete."
                elif len({complex_result.qe_version, slab.qe_version, *(res.qe_version for res in mols.values())}) != 1:
                    status, message = "reference_incompatible", "QE versions differ between complex and references."
                else:
                    energy = complex_result.energy_ev - slab.energy_ev - mol.energy_ev
            adsorption.append({
                "model_id": key, "facet": facet, "pose_id": job["pose_id"], "status": status,
                "site_initial": job["site"],
                "site_final": "" if settings.get("complex_calculation") == "scf" else site_final,
                "anchor_initial": job["anchor"],
                "anchor_role": job["anchor_role"], "second_anchor": job["second_anchor"],
                "orientation": job["orientation"], "conformer": job["conformer"], "energy_ads_ev": energy,
                "energy_complex_ev": complex_result.energy_ev, "energy_slab_ev": slab.energy_ev,
                "energy_molecule_ev": mol.energy_ev, "slab_job_id": slab_id, "molecule_job_id": mol_id,
                "contact_min_ang": minimum, "contacts": _json(contacts), "contact_cutoff_ang": cutoff,
                "slab_atom_count": job["slab_atom_count"], "smiles": models[key]["smiles"],
                "geometry_path": details["geometry_path"], "protocol_id": protocol_id, "job_id": job_id,
                "geometry_kind": "fixed" if settings.get("complex_calculation") == "scf" else "relaxed",
                "qe_version": complex_result.qe_version, "message": message,
            })
        comparison = _compare(adsorption, models, protocol_id, dict.fromkeys(("111", "100"), settings["poses"]))
        paired = _paired_effects(comparison, protocol_id)
        _csv(root / "paired_effects.csv", PAIRED_COLUMNS, paired)
        _csv(root / "jobs.csv", JOB_COLUMNS, job_rows)
        _csv(root / "adsorption_results.csv", ADS_COLUMNS, adsorption)
        _csv(root / "surface_comparison.csv", COMPARISON_COLUMNS, comparison)
    return {"jobs": job_rows, "adsorption": adsorption, "comparison": comparison, "paired": paired}
