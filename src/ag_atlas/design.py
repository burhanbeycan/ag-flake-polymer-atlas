"""Build explicit neutral oligomer models and periodic Ag adsorption starts."""

from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.build import fcc100, fcc111
from ase.constraints import FixAtoms
from rdkit import Chem
from rdkit.Chem import AllChem
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class Model:
    name: str
    family: str
    smiles: str
    description: str
    sequence: str = ""


def _sequence_smiles(sequence: str) -> str:
    repeats = {"A": "CCN(C(=O)CC)", "E": "CCN"}
    return "C" + "".join(repeats[unit] for unit in sequence) + "C"


def catalog() -> dict[str, Model]:
    """Return finite model structures with explicit backbone and end groups."""
    models = {
        "nh3": Model("Ammonia", "Reference", "N", "Neutral ammonia."),
        "benzene": Model("Benzene", "Reference", "c1ccccc1", "Aromatic reference."),
        "pvp": Model(
            "PVP proxy", "PVP", "CCN1CCCC1=O", "1-ethyl-2-pyrrolidone.",
        ),
        "peo": Model(
            "PEO proxy", "PEO", "COCCOC", "1,2-dimethoxyethane.",
        ),
        "lpei-n4": Model(
            "Linear PEI N4", "PEI", "NCCNCCNCCN", "Triethylenetetramine.",
        ),
        "bpei-n4": Model(
            "Branched PEI N4", "PEI", "N(CCN)(CCN)CCN", "Tris(2-aminoethyl)amine.",
        ),
        "lpei-n6": Model(
            "Linear PEI N6", "PEI", "NCCNCCNCCNCCNCCN", "Pentaethylenehexamine.",
        ),
        "bpei-311": Model(
            "Branched PEI N6 (3–1–1)", "PEI", "N(CCN)(CCN)CCNCCNCCN",
            "Central tertiary amine with three arms of 3, 1 and 1 ethylene bridges.",
        ),
        "bpei-221": Model(
            "Branched PEI N6 (2–2–1)", "PEI", "N(CCN)(CCNCCN)CCNCCN",
            "Central tertiary amine with three arms of 2, 2 and 1 ethylene bridges.",
        ),
        "pva": Model("PVA proxy", "PVA", "CC(C)O", "2-propanol."),
        "pvac": Model("PVAc proxy", "PVAc", "CC(C)OC(=O)C", "Isopropyl acetate."),
    }
    for sequence in ("AAAA", "AAEE", "AEAE", "EEEE", "AAAEEE", "AEAEAE"):
        models[sequence.lower()] = Model(
            f"EtOx–EI {sequence}", "EtOx–EI", _sequence_smiles(sequence),
            "Methyl-capped chain CH3–[CH2–CH2–N(R)]n–CH3. "
            "R is propionyl for A and hydrogen for E.", sequence,
        )
    return models


def build_slab(facet: str, size: int, layers: int, lattice: float, vacuum: float) -> Atoms:
    """Construct a clean fcc Ag surface and freeze its bottom half of layers."""
    builders = {"111": fcc111, "100": fcc100}
    if facet not in builders:
        raise ValueError("Facet must be '111' or '100'.")
    if size < 1 or layers < 2 or not np.isfinite([lattice, vacuum]).all():
        raise ValueError("Use at least one lateral repeat and two layers with finite cell settings.")
    if lattice <= 0 or vacuum <= 0:
        raise ValueError("Lattice constant and vacuum must be positive.")
    slab = builders[facet](
        "Ag", size=(size, size, layers), a=lattice, vacuum=vacuum,
    )
    slab.set_constraint(FixAtoms(mask=slab.get_tags() > (layers + 1) // 2))
    # Nested NumPy arrays are not serializable by ASE's extended XYZ writer.
    slab.info["adsorbate_info"]["cell"] = slab.info["adsorbate_info"]["cell"].tolist()
    return slab


def _validated_molecule(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Enter a valid molecular SMILES.")
    if len(Chem.GetMolFrags(mol)) != 1 or any(
        atom.GetFormalCharge() or atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()
    ):
        raise ValueError("Use a single neutral, closed-shell molecule without charged atoms.")
    return Chem.AddHs(mol)


def _carbonyl_carbon(atom: Chem.Atom) -> bool:
    return atom.GetSymbol() == "C" and any(
        bond.GetBondType() == Chem.BondType.DOUBLE
        and bond.GetOtherAtom(atom).GetSymbol() in {"O", "S"}
        for bond in atom.GetBonds()
    )


def _donors(mol: Chem.Mol) -> tuple[list[int], dict[int, str]]:
    donors: dict[str, list[int]] = {}
    roles: dict[int, str] = {}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol not in {"N", "O", "S", "P"}:
            continue
        # Amide nitrogen lone pairs are delocalized into adjacent carbonyl groups.
        if symbol == "N" and any(_carbonyl_carbon(neighbor) for neighbor in atom.GetNeighbors()):
            continue
        hydrogens = atom.GetTotalNumHs(includeNeighbors=True)
        if symbol == "N" and atom.GetIsAromatic() and hydrogens:
            continue
        index = atom.GetIdx()
        roles[index] = symbol
        if symbol == "N" and atom.GetIsAromatic():
            roles[index] = "aromatic_n"
        elif symbol == "N" and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in atom.GetBonds()):
            roles[index] = ("tertiary_amine", "secondary_amine", "primary_amine", "ammonia")[hydrogens]
        if symbol == "O":
            if any(bond.GetBondType() == Chem.BondType.DOUBLE for bond in atom.GetBonds()):
                roles[index] = "carbonyl_o"
            else:
                roles[index] = "hydroxyl_o" if hydrogens else "single_bonded_o"
        donors.setdefault(symbol, []).append(index)
    order = [
        indices[row] for row in range(max(map(len, donors.values()), default=0))
        for indices in donors.values() if row < len(indices)
    ]
    return order, roles


def _mapped_rmsd(first: np.ndarray, second: np.ndarray) -> float:
    first = first - first.mean(axis=0)
    second = second - second.mean(axis=0)
    left, _, right = np.linalg.svd(first.T @ second)
    if np.linalg.det(left @ right) < 0:
        left[:, -1] *= -1
    difference = first @ (left @ right) - second
    return float(np.sqrt(np.mean(np.sum(difference**2, axis=1))))


def conformers(smiles: str, count: int, seed: int) -> list[Atoms]:
    """Sample deterministic ETKDG starts and rank force-field-relaxed conformers."""
    if count < 1:
        raise ValueError("Request at least one conformer.")
    mol = _validated_molecule(smiles)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed % (2**31 - 1)
    params.numThreads = 1
    params.pruneRmsThresh = 0.1
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=max(12, min(128, 10 * count)), params=params))
    if not ids:
        raise ValueError("RDKit could not embed this molecular graph.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        method = "MMFF94s"
        relaxed = AllChem.MMFFOptimizeMoleculeConfs(
            mol, numThreads=1, maxIters=800, mmffVariant=method,
        )
    elif AllChem.UFFHasAllMoleculeParams(mol):
        method = "UFF"
        relaxed = AllChem.UFFOptimizeMoleculeConfs(mol, numThreads=1, maxIters=800)
    else:
        method = "ETKDGv3"
        relaxed = [(0, None)] * len(ids)

    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    heavy = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
    donors, donor_roles = _donors(mol)
    neighbors = {
        str(atom.GetIdx()): [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
        for atom in mol.GetAtoms()
    }
    aromatic_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetIsAromatic()]
    records = sorted(
        zip(ids, relaxed, strict=True),
        key=lambda item: (
            item[1][0] != 0 or (item[1][1] is not None and not np.isfinite(item[1][1])),
            item[1][1] if item[1][1] is not None else 0.0,
            item[0],
        ),
    )
    results: list[Atoms] = []
    for conformer_id, (status, energy) in records:
        if method != "ETKDGv3" and (status != 0 or not np.isfinite(energy)):
            continue
        positions = mol.GetConformer(conformer_id).GetPositions()
        if heavy and any(_mapped_rmsd(positions[heavy], old.positions[heavy]) < 0.25 for old in results):
            continue
        atoms = Atoms(symbols, positions=positions)
        atoms.info.update(
            donor_indices=donors,
            donor_roles=donor_roles,
            neighbors=neighbors,
            aromatic_indices=aromatic_atoms,
            conformer_method=method,
            conformer_energy_kcal_mol=energy,
            rdkit_conformer_id=conformer_id,
        )
        results.append(atoms)
        if len(results) == count:
            break
    if not results:
        raise ValueError("No converged molecular conformer was available.")
    return results


def _align_to_axis(positions: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if np.linalg.norm(source) < 1e-10:
        return positions
    rotation, _ = Rotation.align_vectors([target], [source / np.linalg.norm(source)])
    return rotation.apply(positions)


def _orient(
    molecule: Atoms, anchor: int | None, partner: int | None, mode: int, azimuth: float, tilt: float,
) -> np.ndarray:
    heavy = np.flatnonzero(molecule.numbers > 1)
    geometric_indices = molecule.info["aromatic_indices"] or heavy.tolist()
    if not geometric_indices:
        geometric_indices = list(range(len(molecule)))
    center = molecule.positions[geometric_indices].mean(axis=0)
    origin = molecule.positions[anchor] if anchor is not None else center
    positions = molecule.positions - origin
    if partner is not None:
        axis = positions[partner] / np.linalg.norm(positions[partner])
        center_from_pair = center - origin - positions[partner] / 2.
        perpendicular = center_from_pair - np.dot(center_from_pair, axis) * axis
        if np.linalg.norm(perpendicular) > 1e-10:
            vertical = perpendicular / np.linalg.norm(perpendicular)
            lateral = np.cross(vertical, axis)
            positions = positions @ np.stack([axis, lateral, vertical], axis=1)
        else:
            positions = _align_to_axis(positions, axis, np.array([1., 0., 0.]))
    elif anchor is not None and (mode == 0 or len(heavy) < 3):
        neighbor_vectors = positions[molecule.info["neighbors"][str(anchor)]]
        lengths = np.linalg.norm(neighbor_vectors, axis=1)
        donor_axis = np.sum(neighbor_vectors / lengths[:, None], axis=0)
        if np.linalg.norm(donor_axis) < 1e-10:
            donor_axis = center - origin
        positions = _align_to_axis(positions, donor_axis, np.array([0., 0., 1.]))
    elif len(geometric_indices) >= 3:
        plane = molecule.positions[geometric_indices] - center
        _, _, axes = np.linalg.svd(plane, full_matrices=False)
        normal = axes[-1]
        if np.dot(normal, center - origin) < 0:
            normal = -normal
        positions = _align_to_axis(positions, normal, np.array([0., 0., 1.]))
        if mode == 2:
            positions = Rotation.from_rotvec([tilt, 0., 0.]).apply(positions)
    elif anchor is None:
        positions = Rotation.from_rotvec([tilt, 0., 0.]).apply(positions)
    return Rotation.from_rotvec([0., 0., azimuth]).apply(positions)


def _lateral_clearance(positions: np.ndarray, cell: np.ndarray) -> float:
    offsets = [u * cell[0] + v * cell[1] for u in (-1, 0, 1) for v in (-1, 0, 1) if u or v]
    pair_distances = positions[:, None, :] - positions[None, :, :]
    return min(float(np.linalg.norm(pair_distances + shift, axis=2).min()) for shift in offsets)


def _surface_sites(slab: Atoms) -> list[tuple[str, np.ndarray]]:
    data = slab.info.get("adsorbate_info")
    if not data or not data.get("sites"):
        raise ValueError("Use an atlas Ag slab with labeled adsorption sites.")
    unit_cell = np.asarray(data["cell"])
    center = 0.5 * np.sum(np.asarray(slab.cell)[:2, :2], axis=0)
    shift = np.floor(np.linalg.solve(unit_cell.T, center))
    return [(name, (shift + fractional) @ unit_cell) for name, fractional in data["sites"].items()]


def _place(
    slab: Atoms, positions: np.ndarray, molecule: Atoms, anchor: int | None,
    target_xy: np.ndarray,
) -> tuple[Atoms, float, float] | None:
    cell = np.asarray(slab.cell)
    in_plane_coordinates = np.linalg.solve(cell[:2, :2].T, positions[:, :2].T)
    if np.any(np.ptp(in_plane_coordinates, axis=1) >= 1.):
        return None
    image_clearance = _lateral_clearance(positions, cell)
    if image_clearance < 2.0:
        return None
    top = float(slab.positions[:, 2].max())
    target_height = {"N": 2.40, "O": 2.45, "S": 2.55, "P": 2.55}
    anchor_symbol = molecule[anchor].symbol if anchor is not None else ""
    positions[:, :2] += target_xy
    positions[:, 2] += top + target_height.get(anchor_symbol, 3.10)
    closest_allowed = np.where(molecule.numbers == 1, 1.7, 2.10)
    # The top plane bound excludes slab collisions without favoring a particular lateral site.
    positions[:, 2] += max(0., float(np.max(top + closest_allowed - positions[:, 2])))
    slab_bottom = float(slab.positions[:, 2].min())
    gap = cell[2, 2] - (float(positions[:, 2].max()) - slab_bottom)
    if positions[:, 2].max() >= cell[2, 2] or gap < 6.0:
        return None
    combined = slab.copy()
    adsorbate = molecule.copy()
    adsorbate.positions = positions
    combined.extend(adsorbate)
    return combined, gap, image_clearance


def placements(
    slab: Atoms, facet: str, conformer_list: list[Atoms], count: int, seed: int,
) -> list[tuple[dict, Atoms]]:
    """Return site- and conformation-diverse slab+adsorbate starting geometries."""
    if facet not in {"111", "100"} or count < 1 or not conformer_list:
        raise ValueError("Choose a supported facet, at least one conformer and one placement.")
    sites = _surface_sites(slab)
    rng = np.random.default_rng(seed)
    results: list[tuple[dict, Atoms]] = []
    seen: set[bytes] = set()
    for slot in range(count + 2 * len(sites)):
        site_index = slot % len(sites)
        site, target = sites[site_index]
        for retry in range(15):
            conformer_index = (slot + retry) % len(conformer_list)
            molecule = conformer_list[conformer_index]
            donors = molecule.info["donor_indices"]
            donor_index = slot + slot // len(sites) + retry // len(conformer_list)
            anchor = donors[donor_index % len(donors)] if donors else None
            mode = (slot + retry) % 3 if len(molecule) > 4 else 0
            partner = None
            if mode == 1 and len(donors) > 1:
                partner = donors[(donor_index + 1 + slot // 3) % len(donors)]
                if partner == anchor:
                    partner = donors[(donor_index + 1) % len(donors)]
            positions = _orient(
                molecule, anchor, partner, mode, rng.uniform(0., 2 * np.pi), rng.uniform(0.5, 1.2),
            )
            placed = _place(slab, positions, molecule, anchor, target)
            if placed is None:
                continue
            combined, gap, image_clearance = placed
            key = np.round(combined.positions[len(slab):], decimals=4).tobytes()
            if key in seen:
                continue
            seen.add(key)
            results.append((
                {
                    "site": site,
                    "conformer": conformer_index,
                    "anchor": anchor,
                    "anchor_element": molecule[anchor].symbol if anchor is not None else "",
                    "anchor_role": molecule.info["donor_roles"].get(anchor, ""),
                    "second_anchor": partner,
                    "second_anchor_role": molecule.info["donor_roles"].get(partner, ""),
                    "orientation": (
                        "donor" if anchor is not None else "face",
                        "paired_donors" if partner is not None else "flat",
                        "tilted",
                    )[mode],
                    "vertical_image_gap_angstrom": round(gap, 6),
                    "lateral_image_clearance_angstrom": round(image_clearance, 6),
                },
                combined,
            ))
            break
        if len(results) == count:
            return results
    if not results:
        raise ValueError("No clash-free placement fits the slab. Increase the lateral cell or vacuum.")
    return results
