import csv

import pytest
from ase.io import read

from ag_atlas import workflow
from ag_atlas.qe import Result


@pytest.fixture
def reference(tmp_path):
    pseudos = {element: f"{element}.UPF" for element in ("Ag", "N", "H")}
    for filename in pseudos.values():
        (tmp_path / filename).write_text("Test input writer fixture")
    root = tmp_path / "run"
    config = {
        "models": ["nh3"], "conformers": 1, "poses": 1, "size": 2, "layers": 2,
        "pseudo_dir": str(tmp_path), "pseudos": pseudos,
    }
    return root, config


def test_plan_and_collect_preserve_references(reference, monkeypatch):
    root, config = reference
    manifest = workflow.plan(config, root)
    assert len(manifest["jobs"]) == 5
    assert workflow.collect(root)["comparison"][0]["status"] == "incomplete"

    energies = {"slab-111": -100.0, "slab-100": -97.0, "mol-nh3-00": -10.0,
                "ads-nh3-111-01": -111.0, "ads-nh3-100-01": -107.5}

    def completed(folder):
        return Result("converged", energies[folder.name], read(folder / "start.extxyz"), qe_version="test")

    monkeypatch.setattr(workflow, "parse_output", completed)
    result = workflow.collect(root)
    assert result["comparison"][0]["delta_111_minus_100_ev"] == pytest.approx(-0.5)
    assert result["comparison"][0]["preferred_facet"] == "111"
    with (root / "adsorption_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert all(row["molecule_job_id"] == "mol-nh3-00" for row in rows)
    assert all((root / row["geometry_path"]).is_file() for row in rows)

    (root / "jobs" / "slab-111" / "pw.in").write_text("modified")
    result = workflow.collect(root)
    assert result["adsorption"][0]["status"] == "reference_incomplete"
    assert result["adsorption"][0]["energy_ads_ev"] is None
    assert result["comparison"][0]["preferred_facet"] == ""


def test_refuse_inconsistent_slab_calculation(reference):
    root, config = reference
    with pytest.raises(ValueError, match="same calculation type"):
        workflow.plan({**config, "complex_calculation": "scf"}, root)
    assert not root.exists()


def test_paired_effect_requires_complete_facet_results():
    pair = [
        {"model_id": "aaee", "delta_111_minus_100_ev": -0.2, "preferred_facet": "111", "qe_version": "v"},
        {"model_id": "aeae", "delta_111_minus_100_ev": 0.1, "preferred_facet": "100", "qe_version": "v"},
    ]
    assert workflow._paired_effects(pair, "protocol")[0]["delta_a_minus_b_ev"] == pytest.approx(-0.3)
    pair[1]["preferred_facet"] = ""
    assert workflow._paired_effects(pair, "protocol")[0]["delta_a_minus_b_ev"] is None
