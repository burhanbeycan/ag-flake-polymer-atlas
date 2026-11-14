import csv
import json
from pathlib import Path

import pytest
from ase import Atoms
from ase.io import write

pytest.importorskip("plotly")
pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest

from ag_atlas import gui


def test_local_results_render_and_filter(monkeypatch, tmp_path):
    atoms = Atoms("AgNH3", positions=[(0, 0, 0), (0, 0, 2.3), (0, 0, 3.3), (1, 0, 2.5), (0, 1, 2.5)])
    write(tmp_path / "relaxed.extxyz", atoms)
    row = {
        "model_id": "nh3",
        "facet": "111",
        "pose_id": "pose-001",
        "status": "complete",
        "energy_ads_ev": "-0.32",
        "geometry_path": "relaxed.extxyz",
        "slab_atom_count": "1",
        "contact_cutoff_ang": "3.2",
        "contacts": json.dumps([{"molecule_index": 0, "ag_index": 0, "element": "N", "distance_ang": 2.3}]),
    }
    with (tmp_path / gui.ADSORPTION_CSV).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(row))
        writer.writeheader()
        writer.writerows([row, {**row, "facet": "100", "energy_ads_ev": "-0.40"}])
    comparison = {
        "model_id": "nh3",
        "status": "complete",
        "energy_111_ev": "-0.32",
        "energy_100_ev": "-0.40",
        "delta_111_minus_100_ev": "0.08",
        "preferred_facet": "100",
    }
    with (tmp_path / gui.COMPARISON_CSV).open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(comparison))
        writer.writeheader()
        writer.writerow(comparison)
    monkeypatch.setenv("AG_ATLAS_RUN_DIR", str(tmp_path))
    app = AppTest.from_file(gui.__file__).run()
    assert not app.exception
    assert len(app.get("plotly_chart")) == 2
    assert len(app.get("download_button")) == 3
    assert len(app.dataframe[0].value) == 2
    facet_filter = next(widget for widget in app.multiselect if widget.label == "Surface")
    facet_filter.set_value(["111"]).run()
    assert not app.exception
    assert app.dataframe[0].value["Ag surface"].tolist() == ["111"]


def test_geometry_cannot_escape_run_directory(tmp_path):
    with pytest.raises(ValueError, match="inside the run directory"):
        gui.local_geometry_path(tmp_path, "../other.extxyz")
    assert gui.local_geometry_path(tmp_path, "job/relaxed.extxyz") == Path(tmp_path / "job/relaxed.extxyz")


def test_planner_accepts_custom_smiles_and_writes_qe_input(tmp_path):
    pseudo_dir = tmp_path / "pseudos"
    pseudo_dir.mkdir()
    for element in ("Ag", "C", "H", "N"):
        (pseudo_dir / f"{element}.UPF").write_text(f"Input-writing fixture for {element}\n")
    run_dir = tmp_path / "new-run"
    app = AppTest.from_file(gui.__file__).run()

    def text_input(label):
        return next(widget for widget in app.text_input if widget.label == label)

    text_input("Model identifier").set_value("ethylamine-control")
    text_input("Molecular SMILES").set_value("CCN")
    next(widget for widget in app.multiselect if widget.label == "Ag surfaces").set_value(["111"])
    app.number_input[0].set_value(1)
    app.run()
    assert not app.exception
    text_input("New run folder").set_value(str(run_dir))
    text_input("Pseudopotential folder").set_value(str(pseudo_dir))
    for element in ("Ag", "C", "H", "N"):
        text_input(f"{element} UPF filename").set_value(f"{element}.UPF")
    next(widget for widget in app.button if widget.label == "Prepare calculation files").click().run()
    assert not app.exception
    assert not app.error
    assert len(app.success) == 1
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["protocol"]["models"]["ethylamine-control"]["smiles"] == "CCN"
    assert manifest["protocol"]["models"]["nh3"]["smiles"] == "N"
    assert (run_dir / "jobs" / "ads-ethylamine-control-111-01" / "pw.in").is_file()


def test_local_runner_starts_atlas_without_shell(monkeypatch, tmp_path):
    state = {}
    calls = []

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return object()

    monkeypatch.setattr(gui.shutil, "which", lambda name: "/test/pw.x")
    monkeypatch.setattr(gui.subprocess, "Popen", popen)
    monkeypatch.setattr(gui.st, "session_state", state)
    monkeypatch.setattr(gui.st, "button", lambda label, **kw: True)
    monkeypatch.setattr(gui.st, "rerun", lambda: None)
    monkeypatch.setattr(gui.st, "expander", lambda label: __import__("contextlib").nullcontext())
    monkeypatch.setattr(gui.st, "code", lambda text: None)
    gui.run_local(tmp_path)
    command, kwargs = calls[0]
    assert command[1:] == ["-m", "ag_atlas", "run", str(tmp_path), "--executable", "/test/pw.x"]
    assert "shell" not in kwargs
    assert str(tmp_path) in state["atlas_processes"]
