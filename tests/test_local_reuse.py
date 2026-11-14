"""Check exact-input reuse without substituting fixtures for real DFT validation."""

import importlib.util
import json
from pathlib import Path

import pytest

from ag_atlas.qe import parse_output
from ag_atlas.workflow import plan


spec = importlib.util.spec_from_file_location(
    "local_study_runner", Path(__file__).parents[1] / "scripts" / "run_local_study.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)

OUTPUT = """Program PWSCF v.7.4.1 starts
convergence has been achieved in 8 iterations
!    total energy = -150.0 Ry
JOB DONE.
"""


@pytest.fixture
def study(tmp_path):
    pseudos = tmp_path / "pseudos"
    pseudos.mkdir()
    for element in ("Ag", "N", "H"):
        (pseudos / f"{element}.UPF").write_text("INPUT-WRITER TEST FIXTURE ONLY\n")
    config = {
        "models": ["nh3"], "conformers": 1, "poses": 1, "size": 2, "layers": 2,
        "pseudo_dir": str(pseudos), "pseudos": {el: f"{el}.UPF" for el in ("Ag", "N", "H")},
        "slab_calculation": "scf", "molecule_calculation": "scf", "complex_calculation": "scf",
    }
    root = tmp_path / "study"
    cases = [{"id": name, "run": name} for name in ("source", "target")]
    manifests = {case["id"]: plan(config, root / case["run"]) for case in cases}
    (root / "cases.json").write_text(json.dumps(cases))
    wrapper, binary = tmp_path / "wrapper", tmp_path / "binary"
    wrapper.write_text("WRAPPER FIXTURE")
    binary.write_bytes(b"ENGINE FINGERPRINT FIXTURE")
    engine = runner.engine_identity(wrapper, binary)
    return root, cases, manifests, engine, pseudos, wrapper, binary


def prepared_cache(study):
    root, cases, manifests, engine, *_ = study
    job = manifests["source"]["jobs"][0]
    folder = root / "source" / "jobs" / job["job_id"]
    (folder / "pw.out").write_text(OUTPUT)
    cache = {}
    runner.cache_fresh_success(cache, root, cases[0], job, engine)
    return cache, job, folder


def test_reuse_records_origin_and_preserves_failed_attempt(study):
    root, cases, _, engine, *_ = study
    cache, job, source = prepared_cache(study)
    target = root / "target" / "jobs" / job["job_id"]
    (target / "pw.out").write_text("PREVIOUS FAILED ATTEMPT\n")
    reused = runner.reuse_case_outputs(cache, root, cases[1], engine)
    assert reused == [job["job_id"]]
    assert (source / "pw.out").read_bytes() == (target / "pw.out").read_bytes()
    assert parse_output(target).status == "converged"
    record = json.loads((target / "reuse.json").read_text())
    assert record["identity"]["input_sha256"] == job["input_sha256"]
    assert record["engine"] == engine
    assert record["source_case"] == "source" and record["destination_case"] == "target"
    assert record["source_chain"] == [{"case": "source", "run": "source", "job_id": job["job_id"]}]
    assert list((target / "attempt-history").glob("*.out"))[0].read_text() == "PREVIOUS FAILED ATTEMPT\n"
    assert not (target / ".atlas.lock").exists()


@pytest.mark.parametrize("changed", ["source-input", "source-geometry", "source-output", "pseudo"])
def test_reuse_rejects_changed_source_evidence(study, changed):
    root, cases, _, engine, pseudos, *_ = study
    cache, job, source = prepared_cache(study)
    if changed == "source-input":
        path = source / "pw.in"
    elif changed == "source-geometry":
        path = source / "start.extxyz"
    elif changed == "source-output":
        path = source / "pw.out"
    else:
        path = pseudos / "Ag.UPF"
    path.write_bytes(path.read_bytes() + b"\nCHANGED\n")
    with pytest.raises(ValueError):
        runner.reuse_case_outputs(cache, root, cases[1], engine)
    assert not (root / "target" / "jobs" / job["job_id"] / "pw.out").exists()


def test_reuse_rejects_changed_destination_and_engine(study):
    root, cases, _, engine, *_ = study
    cache, job, _ = prepared_cache(study)
    target = root / "target" / "jobs" / job["job_id"]
    with pytest.raises(ValueError, match="engine fingerprint"):
        runner.reuse_case_outputs(cache, root, cases[1], {**engine, "binary": {"sha256": "changed"}})
    (target / "pw.in").write_text("CHANGED INPUT")
    with pytest.raises(ValueError, match="changed input"):
        runner.reuse_case_outputs(cache, root, cases[1], engine)
    assert not (target / "pw.out").exists()


def test_reuse_never_replaces_usable_or_locked_output(study):
    root, cases, _, engine, *_ = study
    cache, job, _ = prepared_cache(study)
    target = root / "target" / "jobs" / job["job_id"]
    existing = OUTPUT.replace("-150.0", "-149.0")
    (target / "pw.out").write_text(existing)
    assert runner.reuse_case_outputs(cache, root, cases[1], engine) == []
    assert (target / "pw.out").read_text() == existing
    (target / "pw.out").unlink()
    (target / ".atlas.lock").write_text("ANOTHER ACTIVE JOB")
    with pytest.raises(RuntimeError, match="in use"):
        runner.reuse_case_outputs(cache, root, cases[1], engine)
    assert not (target / "pw.out").exists()


def test_study_does_not_seed_preexisting_converged_results(study, monkeypatch):
    root, _, manifests, _, _, wrapper, binary = study
    for job in manifests["source"]["jobs"]:
        (root / "source" / "jobs" / job["job_id"] / "pw.out").write_text(OUTPUT)
    executed = []

    def fake_run(folder, *, executable, threads, timeout, on_result):
        manifest = manifests[folder.name]
        for job in manifest["jobs"]:
            job_folder = folder / "jobs" / job["job_id"]
            if parse_output(job_folder).status != "converged":
                executed.append((folder.name, job["job_id"]))
                (job_folder / "pw.out").write_text(OUTPUT)
            on_result(job["job_id"], parse_output(job_folder))

    monkeypatch.setattr(runner, "run", fake_run)
    result = runner.execute_study(root, str(wrapper), 1, None,
                                  reuse_identical=True, engine_binary=binary)
    assert result == 0
    assert executed == [("target", job["job_id"]) for job in manifests["target"]["jobs"]]
    assert not list((root / "target").rglob("reuse.json"))
