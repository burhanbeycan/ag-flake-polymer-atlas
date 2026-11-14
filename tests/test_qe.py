"""Test QE's input and log contract without an installed pw.x."""

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from ase import Atoms
from ase.constraints import FixAtoms

from ag_atlas.qe import parse_output, run_job, write_input

PARAMETERS = {
    "pseudos": {"Ag": "Ag.upf", "N": "N.upf", "H": "H.upf"},
    "pseudo_dir": ".",
    "ecutwfc": 55,
    "ecutrho": 440,
    "kmesh": [3, 3, 1],
}
HEADER = """     Program PWSCF v.7.4.1 starts
     convergence has been achieved in   8 iterations
!    total energy              =    -1.500000000D+02 Ry

     Forces acting on atoms (cartesian axes, Ry/au):

     atom    1 type  1   force = 0.10000000  0.00000000  0.00000000
     atom    2 type  2   force = 0.00000000  0.00000000  0.00010000

     Total force = 0.100000     Total SCF correction = 0.000000
"""
GEOMETRY = """     bfgs converged in 1 scf cycles and 0 bfgs steps
     End of BFGS Geometry Optimization
Begin final coordinates

ATOMIC_POSITIONS (angstrom)
Ag 0.000000 0.000000 0.000000 0 0 0
N  0.000000 0.000000 2.250000

End final coordinates
"""
DONE = "     JOB DONE.\n"


def prepare_complex(folder: Path) -> None:
    atoms = Atoms("AgN", positions=[[0, 0, 0], [0, 0, 2]], cell=[8, 8, 20], pbc=True)
    atoms.set_constraint(FixAtoms(indices=[0]))
    write_input(folder / "pw.in", atoms, PARAMETERS, "complex")


def test_molecular_gamma_and_metallic_input(tmp_path: Path) -> None:
    prepare_complex(tmp_path)
    metal_input = (tmp_path / "pw.in").read_text()
    assert "K_POINTS automatic\n3 3 1  0 0 0" in metal_input
    assert "'smearing'" in metal_input
    assert "dftd3_version" in metal_input
    assert "nosym" in metal_input and ".true." in metal_input.partition("nosym")[2].splitlines()[0]
    assert "0 0 0" in metal_input.partition("ATOMIC_POSITIONS angstrom")[2]

    molecule = Atoms("NH3", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], cell=[15] * 3)
    write_input(tmp_path / "mol.in", molecule, PARAMETERS, "molecule")
    mol_input = (tmp_path / "mol.in").read_text()
    assert "K_POINTS gamma" in mol_input
    assert "'fixed'" in mol_input


def test_parses_relaxed_coordinates_and_energy(tmp_path: Path) -> None:
    prepare_complex(tmp_path)
    (tmp_path / "pw.out").write_text(HEADER + GEOMETRY + DONE)
    result = parse_output(tmp_path)
    assert result.status == "converged", result.message
    assert result.qe_version == "7.4.1"
    assert result.atoms.positions[1, 2] == pytest.approx(2.25)
    assert result.energy_ev == pytest.approx(-2040.8539)
    assert result.max_force_ev_per_angstrom == pytest.approx(0.00257110, abs=1e-8)
    assert result.atoms.constraints[0].get_indices().tolist() == [0]


@pytest.mark.parametrize(
    ("content", "status"),
    [(HEADER + DONE, "unconverged"), (HEADER + GEOMETRY, "failed"),
     (HEADER + "convergence NOT achieved after 10 iterations\n" + GEOMETRY + DONE, "unconverged"),
     (HEADER + GEOMETRY + DONE + HEADER, "failed")],
)
def test_gates_invalid_runs(tmp_path: Path, content: str, status: str) -> None:
    prepare_complex(tmp_path)
    (tmp_path / "pw.out").write_text(content)
    result = parse_output(tmp_path)
    assert result.status == status
    assert result.energy_ev is None


def test_runner_keeps_log_and_exit_status(tmp_path: Path) -> None:
    prepare_complex(tmp_path)
    executable = tmp_path / "fake_pw"
    executable.write_text("#!/bin/sh\nprintf '%s\\n' 'failed deliberately'\nexit 8\n")
    executable.chmod(0o755)
    result = run_job(tmp_path, str(executable), timeout=3)
    assert result.status == "failed"
    assert result.message == parse_output(tmp_path).message == "pw.x exited with status 8."
    assert "failed deliberately" in (tmp_path / "pw.out").read_text()


def test_timeout_is_retained_on_collect(tmp_path: Path) -> None:
    prepare_complex(tmp_path)
    executable = tmp_path / "fake_pw"
    executable.write_text("#!/bin/sh\nexec sleep 2\n")
    executable.chmod(0o755)
    result = run_job(tmp_path, str(executable), timeout=0.1)
    assert result.status == parse_output(tmp_path).status == "timeout"


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups and file locks")
def test_timeout_stops_launcher_children(tmp_path: Path) -> None:
    import fcntl

    prepare_complex(tmp_path)
    # A file lock is released even if an orphan becomes a zombie. Checking a PID
    # with kill(pid, 0) alone would incorrectly count such a child as still alive.
    child = tmp_path / "rank.py"
    child.write_text(
        "import fcntl, os, signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        "with open('rank.lock', 'w') as handle:\n"
        "    fcntl.flock(handle, fcntl.LOCK_EX)\n"
        "    Path('rank.pid').write_text(str(os.getpid()))\n"
        "    time.sleep(30)\n"
    )
    executable = tmp_path / "launcher.py"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, 'rank.py'])\n"
        "time.sleep(30)\n"
    )
    executable.chmod(0o755)
    try:
        result = run_job(tmp_path, str(executable), timeout=3)
        assert result.status == parse_output(tmp_path).status == "timeout"
        assert (tmp_path / "rank.pid").is_file(), "The child must start before timing out."
        deadline = time.monotonic() + 2
        with (tmp_path / "rank.lock").open() as handle:
            while True:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        pytest.fail("The timed-out launcher's child is still holding its lock.")
                    time.sleep(0.01)
    finally:
        # Keep a failing regression from leaking the intentionally stubborn child.
        if (tmp_path / "rank.pid").is_file():
            try:
                os.kill(int((tmp_path / "rank.pid").read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass


@pytest.mark.skipif(os.name != "posix", reason="POSIX process groups")
def test_interrupt_reaps_isolated_job(tmp_path: Path, monkeypatch) -> None:
    prepare_complex(tmp_path)
    executable = tmp_path / "fake_pw"
    executable.write_text("#!/bin/sh\nexec sleep 30\n")
    executable.chmod(0o755)
    original_wait = subprocess.Popen.wait
    interrupted = []

    def interrupt_once(process, *args, **kwargs):
        if not interrupted:
            interrupted.append(process)
            raise KeyboardInterrupt
        return original_wait(process, *args, **kwargs)

    monkeypatch.setattr(subprocess.Popen, "wait", interrupt_once)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_job(tmp_path, str(executable))
        assert interrupted[0].returncode is not None
        assert interrupted[0].returncode < 0
        with pytest.raises(ProcessLookupError):
            os.killpg(interrupted[0].pid, 0)
    finally:
        if interrupted and interrupted[0].returncode is None:
            interrupted[0].kill()
            original_wait(interrupted[0])
