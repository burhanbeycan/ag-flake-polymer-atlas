import csv
import json

import pytest
from ase.io import read

from ag_atlas import workflow
from ag_atlas.publication import publication
from ag_atlas.qe import Result


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plan(tmp_path, name, **settings):
    symbols = ("Ag", "N", "H", "C", "O")
    for symbol in symbols:
        (tmp_path / f"{symbol}.UPF").write_text("Test-only placeholder")
    run = tmp_path / name
    defaults = {
        "models": ["nh3"], "conformers": 1, "poses": 1, "size": 2, "layers": 2,
        "pseudo_dir": str(tmp_path), "pseudos": {element: f"{element}.UPF" for element in symbols},
    }
    workflow.plan({**defaults, **settings}, run)
    return run


def populate(monkeypatch, run, energies):
    def completed(folder):
        if folder.name not in energies:
            return Result("pending")
        return Result("converged", energies[folder.name], read(folder / "start.extxyz"), qe_version="test")

    monkeypatch.setattr(workflow, "parse_output", completed)
    workflow.collect(run)


def specification(tmp_path, specs):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(specs))
    return path


def case(run, *, purpose="reference", series="validation", **kwargs):
    return {"id": run.name, "run": run.name, "purpose": purpose, "series": series, **kwargs}


def test_sensitivity_and_energy_provenance(tmp_path, monkeypatch):
    reference = plan(tmp_path, "base")
    varied = plan(tmp_path, "cutoff", ecutwfc=70)
    changed = plan(tmp_path, "wrong", ecutwfc=70, vacuum=20)
    energies = {
        "slab-111": -100., "slab-100": -97., "mol-nh3-00": -10.,
        "ads-nh3-111-01": -111., "ads-nh3-100-01": -107.5,
    }
    populate(monkeypatch, reference, energies)
    populate(monkeypatch, varied, {**energies, "ads-nh3-111-01": -111.05})
    populate(monkeypatch, changed, energies)
    cases = specification(tmp_path, [
        case(reference, baseline=True), case(varied, vary=["ecutwfc"]), case(changed, vary=["ecutwfc"]),
    ])
    output = tmp_path / "tables"
    publication(cases, output)
    compare = rows(output / "numerical_sensitivity.csv")
    accepted = next(row for row in compare if row["case_id"] == "cutoff" and row["facet"] == "111" and row["quantity"] == "Eads")
    assert float(accepted["change"]) == pytest.approx(-0.05)
    assert {row["status"] for row in compare if row["case_id"] == "wrong"} == {"confounded_settings"}
    assert len(rows(output / "facet_selectivity.csv")) == 3

    table = varied / "adsorption_results.csv"
    adsorption = rows(table)
    adsorption[0]["energy_ads_ev"] = "-2.0"
    with table.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(adsorption[0]))
        writer.writeheader()
        writer.writerows(adsorption)
    publication(cases, output)
    invalid = next(row for row in rows(output / "pose_pool.csv") if row["case_id"] == "cutoff" and row["facet"] == "111")
    assert invalid["status"] == "incompatible_energy"
    assert invalid["energy_ads_ev"] == ""


def test_oligomer_effect_requires_every_leg_and_relaxed_geometry(tmp_path, monkeypatch):
    run = plan(tmp_path, "pairs", models=["bpei-311", "bpei-221"], size=6)
    energies = {
        "slab-111": -100., "slab-100": -97., "mol-bpei-311-00": -10., "mol-bpei-221-00": -15.,
        "ads-bpei-311-111-01": -111., "ads-bpei-311-100-01": -107.5,
        "ads-bpei-221-111-01": -116., "ads-bpei-221-100-01": -113.,
    }
    populate(monkeypatch, run, energies)
    config = specification(tmp_path, [case(run, purpose="oligomer")])
    output = tmp_path / "tables"
    publication(config, output)
    row, = rows(output / "paired_oligomers.csv")
    assert float(row["delta_a_minus_b_ev"]) == pytest.approx(-0.5)
    assert row["status"] == "complete"

    populate(monkeypatch, run, {key: val for key, val in energies.items() if key != "ads-bpei-311-111-01"})
    publication(config, output)
    row, = rows(output / "paired_oligomers.csv")
    assert row["delta_a_minus_b_ev"] == ""
    assert row["status"] == "incomplete"


def test_benzene_height_and_smoke_scope(tmp_path, monkeypatch):
    run = plan(tmp_path, "benzene", models=["benzene"], facets=["111"], size=3)
    populate(monkeypatch, run, {"slab-111": -100., "mol-benzene-00": -15., "ads-benzene-111-01": -116.})
    specs = specification(tmp_path, [case(run, purpose="smoke")])
    output = tmp_path / "tables"
    summary = publication(specs, output)
    row, = rows(output / "facet_results.csv")
    assert float(row["carbon_height_ang"]) > 0
    assert row["purpose"] == "smoke"
    assert summary["reference_convergence"] == 0
