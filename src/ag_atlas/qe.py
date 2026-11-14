"""Generate, run and validate Quantum ESPRESSO pw.x jobs."""

from __future__ import annotations

import math
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from ase import Atoms
from ase.io.espresso import (
    get_atomic_positions,
    label_to_symbol,
    read_espresso_in,
    write_espresso_in,
)
from ase.units import create_units

_QE_UNITS = create_units("2006")
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][+-]?\d+)?"
_ENERGY = re.compile(rf"^\s*!\s+total energy\s*=\s*({_NUMBER})\s+Ry", re.I | re.M)
_QE_VERSION = re.compile(r"Program\s+PWSCF\s+v[. ]*([^\s]+)", re.I)
_QE_ERROR = re.compile(r"^\s*(Error in routine[^\r\n]*)(?:\r?\n[ \t]*([^\r\n%]+))?", re.I | re.M)
_RUNNER_ERROR = re.compile(r"^Atlas runner \[(timeout|failed)\]: (.+)$", re.M)
_RUNNER_EXIT = re.compile(r"^Atlas runner exit status: (-?\d+)$", re.M)
_BFGS_CONVERGED = re.compile(r"\bbfgs\s+converged\s+in\b", re.I)
_SCF_CONVERGED = re.compile(r"\bconvergence\s+has\s+been\s+achieved\b", re.I)
_SCF_UNCONVERGED = re.compile(r"\bconvergence\s+(?:has\s+)?not\s+(?:been\s+)?achieved\b", re.I)
_FINAL_COORDINATES = re.compile(r"Begin final coordinates\s*\n(.*?)End final coordinates", re.I | re.S)
_LATTICE_PARAMETER = re.compile(rf"lattice parameter\s*\(alat\)\s*=\s*({_NUMBER})\s+a\.u\.", re.I)
_FORCE_BLOCK = re.compile(r"Forces acting on atoms[^\n]*\n(.*?)Total force\s*=", re.I | re.S)
_FORCE = re.compile(
    rf"^\s*atom\s+(\d+)\s+type\s+\d+\s+force\s*=\s*({_NUMBER})\s+({_NUMBER})\s+({_NUMBER})",
    re.I | re.M,
)
_ROLES = {"slab", "molecule", "complex"}


@dataclass(frozen=True)
class Result:
    status: str
    energy_ev: float | None = None
    atoms: Atoms | None = None
    message: str = ""
    qe_version: str | None = None
    max_force_ev_per_angstrom: float | None = None


def _fortran_float(value: str) -> float:
    return float(value.lower().replace("d", "e"))


def _system_settings(parameters: dict[str, Any], role: str) -> dict[str, Any]:
    molecular = role == "molecule"
    system: dict[str, Any] = {
        "ecutwfc": float(parameters["ecutwfc"]),
        "ecutrho": float(parameters["ecutrho"]),
        "input_dft": parameters.get("functional", parameters.get("input_dft", "PBE")),
        "nosym": bool(parameters.get("nosym", True)),
        "occupations": "fixed" if molecular else "smearing",
        "tot_charge": 0.0,
    }
    if not 0 < system["ecutwfc"] <= system["ecutrho"] or not math.isfinite(system["ecutrho"]):
        raise ValueError("Require finite cutoffs with 0 < ecutwfc <= ecutrho.")
    if not molecular:
        smearing = parameters.get("smearing", "mv")
        if isinstance(smearing, dict):
            system["smearing"] = smearing.get("name", "mv")
            system["degauss"] = float(smearing.get("degauss", parameters.get("degauss", 0.02)))
        else:
            system["smearing"] = smearing
            system["degauss"] = float(parameters.get("degauss", 0.02))
        if not math.isfinite(system["degauss"]) or system["degauss"] <= 0:
            raise ValueError("degauss must be a finite, positive energy in Ry.")
    dispersion = str(parameters.get("dispersion", "grimme-d3")).casefold()
    if dispersion in {"d3", "d3bj", "d3(bj)", "d3-bj", "grimme-d3", "grimme-d3-bj"}:
        system["vdw_corr"] = "grimme-d3"
        system["dftd3_version"] = int(parameters.get("dftd3_version", 4))
        system["dftd3_threebody"] = bool(parameters.get("dftd3_threebody", True))
    else:
        system["vdw_corr"] = dispersion
    isolation = parameters.get(f"{role}_assume_isolated")
    if role == "complex" and isolation is None:
        isolation = parameters.get("slab_assume_isolated")
    if isolation:
        system["assume_isolated"] = isolation
    return system


def write_input(path: Path, atoms: Atoms, parameters: dict[str, Any], role: str) -> None:
    """Write an SCF or constrained BFGS input using absolute pseudo_dir."""
    if role not in _ROLES:
        raise ValueError(f"Unknown QE job role: {role}")
    if not len(atoms) or atoms.cell.rank != 3 or not np.isfinite(atoms.positions).all():
        raise ValueError("QE requires finite atom positions and a three-vector periodic cell.")
    if not np.isfinite(atoms.cell.array).all() or atoms.cell.volume <= 0:
        raise ValueError("QE requires a finite cell with nonzero volume.")
    pseudo_dir = Path(parameters["pseudo_dir"]).expanduser().resolve()
    pseudos = parameters["pseudos"]
    missing = set(atoms.get_chemical_symbols()).difference(pseudos)
    if missing:
        raise ValueError(f"Missing pseudopotential mapping for {', '.join(sorted(missing))}.")
    molecular = role == "molecule"
    calculation = str(parameters.get(f"{role}_calculation", parameters.get("calculation", "relax")))
    if calculation not in {"relax", "scf"}:
        raise ValueError("Only relax and scf calculations are supported.")
    control = {
        "calculation": calculation,
        "prefix": "atlas",
        "pseudo_dir": str(pseudo_dir),
        "outdir": "./qe-scratch",
        "disk_io": "none",
        "tprnfor": True,
        "nstep": int(parameters.get("nstep", 150)),
        "etot_conv_thr": float(parameters.get("etot_conv_thr", 1e-4)),
        "forc_conv_thr": float(parameters.get("forc_conv_thr", 5e-4)),
    }
    if control["nstep"] < 1 or not all(
        math.isfinite(control[key]) and control[key] > 0
        for key in ("etot_conv_thr", "forc_conv_thr")
    ):
        raise ValueError("nstep and the ionic convergence thresholds must be positive.")
    electrons = {
        "conv_thr": float(parameters.get("conv_thr", 1e-8)),
        "mixing_beta": float(parameters.get("mixing_beta", 0.3)),
        "electron_maxstep": int(parameters.get("electron_maxstep", 200)),
        "scf_must_converge": True,
    }
    if (
        not math.isfinite(electrons["conv_thr"])
        or electrons["conv_thr"] <= 0
        or not 0 < electrons["mixing_beta"] <= 1
        or electrons["electron_maxstep"] < 1
    ):
        raise ValueError("SCF settings require positive thresholds and 0 < mixing_beta <= 1.")
    input_data = {
        "control": control,
        "system": _system_settings(parameters, role),
        "electrons": electrons,
    }
    if calculation == "relax":
        input_data["ions"] = {"ion_dynamics": "bfgs"}
    kmesh = None if molecular else tuple(parameters["kmesh"])
    if kmesh is not None and (
        len(kmesh) != 3 or any(not isinstance(value, int) or value <= 0 for value in kmesh)
        or kmesh[2] != 1
    ):
        raise ValueError("The slab kmesh must have form [nx, ny, 1] with positive integers.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        write_espresso_in(
            handle, atoms, input_data=input_data, pseudopotentials=pseudos, kpts=kmesh
        )


def _final_geometry(output: str, input_atoms: Atoms, calculation: str) -> Atoms:
    geometry = input_atoms.copy()
    if calculation == "scf":
        return geometry
    final_coordinates = list(_FINAL_COORDINATES.finditer(output))
    if not final_coordinates:
        raise ValueError("QE did not print a complete final-coordinate block.")
    lines = final_coordinates[-1][1].splitlines()
    lattice_parameters = _LATTICE_PARAMETER.findall(output)
    alat = _fortran_float(lattice_parameters[-1]) * _QE_UNITS.Bohr if lattice_parameters else None
    positions = get_atomic_positions(lines, len(input_atoms), cell=input_atoms.cell, alat=alat)
    if positions is None:
        raise ValueError("QE final-coordinate block contains no atomic positions.")
    symbols = [label_to_symbol(line[0]) for line in positions]
    if symbols != input_atoms.get_chemical_symbols():
        raise ValueError("QE final coordinates and input have different atom ordering.")
    geometry.set_positions([line[1] for line in positions], apply_constraint=False)
    if not np.isfinite(geometry.positions).all():
        raise ValueError("QE returned non-finite final coordinates.")
    return geometry


def _max_movable_force(output: str, geometry: Atoms) -> float | None:
    blocks = list(_FORCE_BLOCK.finditer(output))
    if not blocks:
        return None
    entries = _FORCE.findall(blocks[-1][1])
    # Higher QE verbosity can add component-specific forces after the total atomic forces.
    entries = entries[:len(geometry)]
    if [int(entry[0]) for entry in entries] != list(range(1, len(geometry) + 1)):
        return None
    forces = np.array([[_fortran_float(value) for value in entry[1:]] for entry in entries])
    if not np.isfinite(forces).all():
        return None
    forces *= _QE_UNITS.Ry / _QE_UNITS.Bohr
    for constraint in geometry.constraints:
        constraint.adjust_forces(geometry, forces)
    return float(np.linalg.norm(forces, axis=1).max())


def parse_output(folder: Path, returncode: int | None = None) -> Result:
    """Collect a completed pw.x result, enforcing electronic and ionic convergence."""
    folder = Path(folder)
    input_path, output_path = folder / "pw.in", folder / "pw.out"
    if not output_path.is_file():
        return Result("pending", message="pw.out has not been produced.")
    if not input_path.is_file():
        return Result("failed", message="pw.in is missing.")
    try:
        output = output_path.read_text(encoding="utf-8", errors="replace")
        input_text = input_path.read_text(encoding="utf-8")
    except OSError as exc:
        return Result("failed", message=str(exc))
    versions = _QE_VERSION.findall(output)
    version = versions[-1] if versions else None
    runner_error = _RUNNER_ERROR.search(output)
    if runner_error:
        return Result(runner_error[1], message=runner_error[2], qe_version=version)
    error = _QE_ERROR.search(output)
    if error:
        message = " ".join(line.strip() for line in error.groups() if line and line.strip())
        return Result("failed", message=message, qe_version=version)
    runner_exit = _RUNNER_EXIT.search(output)
    if returncode is None and runner_exit:
        returncode = int(runner_exit[1])
    if returncode not in (0, None):
        return Result("failed", message=f"pw.x exited with status {returncode}.", qe_version=version)
    if len(versions) != 1:
        return Result("failed", message="Expected one PWSCF invocation in pw.out.", qe_version=version)
    if not re.search(r"\bJOB\s+DONE\.", output):
        return Result("failed", message="QE did not finish normally.", qe_version=version)
    energies = _ENERGY.findall(output)
    if _SCF_UNCONVERGED.search(output) or not _SCF_CONVERGED.search(output) or not energies:
        return Result("unconverged", message="Electronic convergence is unconfirmed.", qe_version=version)
    calculation_match = re.search(r"""\bcalculation\s*=\s*['"](relax|scf)['"]""", input_text, re.I)
    if not calculation_match:
        return Result("failed", message="pw.in does not declare relax or scf.", qe_version=version)
    calculation = calculation_match[1].lower()
    if calculation == "relax" and not _BFGS_CONVERGED.search(output):
        return Result("unconverged", message="Ionic BFGS convergence is unconfirmed.", qe_version=version)
    try:
        input_atoms = read_espresso_in(input_path)
        geometry = _final_geometry(output, input_atoms, calculation)
        energy = _fortran_float(energies[-1]) * _QE_UNITS.Ry
        if not math.isfinite(energy):
            raise ValueError("QE energy is non-finite.")
    except (AssertionError, IndexError, KeyError, NotImplementedError, OSError, StopIteration, ValueError) as exc:
        return Result("failed", message=f"Cannot recover QE geometry and energy: {exc}", qe_version=version)
    return Result(
        "converged", energy, geometry, qe_version=version,
        max_force_ev_per_angstrom=_max_movable_force(output, geometry),
    )


def run_job(
    folder: Path, executable: str = "pw.x", timeout: float | None = None, threads: int = 1,
    *, launcher: tuple[str, ...] = (),
) -> Result:
    """Execute pw.x without a shell and keep stdout and stderr for audit."""
    folder = Path(folder).resolve()
    input_path, output_path = folder / "pw.in", folder / "pw.out"
    if not input_path.is_file():
        return Result("failed", message="pw.in is missing.")
    if not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    if timeout is not None and (not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("timeout must be a positive, finite number of seconds.")
    environment = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    command = [str(executable), "-in", "pw.in"]
    if "/" in str(executable):
        command[0] = str(Path(executable).expanduser().resolve())
    command = [*launcher, *command]
    try:
        with output_path.open("w", encoding="utf-8") as handle:
            try:
                run = subprocess.run(
                    command, cwd=folder, stdout=handle, stderr=subprocess.STDOUT,
                    env=environment, timeout=timeout, check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                if isinstance(exc, subprocess.TimeoutExpired):
                    status, message = "timeout", f"pw.x exceeded the {timeout:g} s timeout."
                else:
                    status, message = "failed", f"Could not start pw.x: {exc}"
                print(f"\nAtlas runner [{status}]: {message}", file=handle)
                return Result(status, message=message)
            if run.returncode:
                print(f"\nAtlas runner exit status: {run.returncode}", file=handle)
    except OSError as exc:
        return Result("failed", message=f"Could not write pw.out: {exc}")
    return parse_output(folder, returncode=run.returncode)
