"""Test QE's input and log contract without an installed pw.x."""

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
