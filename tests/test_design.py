import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from ag_atlas.design import build_slab, catalog, conformers, placements


def test_matched_polymer_structures():
    models = catalog()
    bpei = [Chem.MolFromSmiles(models[key].smiles) for key in ("bpei-311", "bpei-221")]
    assert [CalcMolFormula(mol) for mol in bpei] == ["C10H28N6"] * 2
    for mol in bpei:
        amine_degrees = sorted(atom.GetDegree() for atom in mol.GetAtoms() if atom.GetSymbol() == "N")
        assert amine_degrees == [1, 1, 1, 2, 2, 3]
    assert len({Chem.MolToSmiles(mol) for mol in bpei}) == 2

    for left, right, expected_formula in (
        ("aaee", "aeae", "C16H34N4O2"),
        ("aaaeee", "aeaeae", "C23H48N6O3"),
    ):
        mols = [Chem.MolFromSmiles(models[key].smiles) for key in (left, right)]
        assert len({CalcMolFormula(mol) for mol in mols}) == 1
        assert len({Chem.MolToSmiles(mol) for mol in mols}) == 2
        assert CalcMolFormula(mols[0]) == expected_formula


@pytest.mark.parametrize(("facet", "expected_sites"), [
    ("111", {"ontop", "bridge", "fcc", "hcp"}),
    ("100", {"ontop", "bridge", "hollow"}),
])
def test_ammonia_placements_are_deterministic(facet, expected_sites):
    slab = build_slab(facet, 3, 4, 4.09, 12)
    ammonia = conformers("N", 2, 13)
    result = placements(slab, facet, ammonia, 4, 13)
    repeated = placements(slab, facet, ammonia, 4, 13)
    assert len(result) == 4
    assert {row["site"] for row, _ in result} == expected_sites
    assert len(slab.constraints[0].get_indices()) == 18
    for (row, atoms), (same_row, same_atoms) in zip(result, repeated, strict=True):
        assert row == same_row
        assert row["anchor"] == 0
        assert row["anchor_element"] == "N"
        assert row["lateral_image_clearance_angstrom"] > 2
        assert np.allclose(atoms.positions, same_atoms.positions)
        assert np.array_equal(atoms.constraints[0].get_indices(), slab.constraints[0].get_indices())
        assert np.allclose(atoms.positions[:len(slab)], slab.positions)
        assert atoms.positions[len(slab), 2] < atoms.positions[len(slab) + 1:, 2].min()


def test_amide_nitrogen_is_not_used_as_donor():
    atoms = conformers(catalog()["aaee"].smiles, 2, 13)[0]
    assert sorted(atoms[index].symbol for index in atoms.info["donor_indices"]) == ["N", "N", "O", "O"]
    assert sorted(atoms.info["donor_roles"].values()) == ["carbonyl_o", "carbonyl_o", "secondary_amine", "secondary_amine"]


@pytest.mark.parametrize("smiles", ["[NH4+]", "CC.O", "[CH3]", "invalid"])
def test_custom_smiles_requires_a_neutral_closed_shell_molecule(smiles):
    with pytest.raises(ValueError, match="SMILES|neutral"):
        conformers(smiles, 1, 0)


def test_paired_donor_placement_starts_with_matching_donor_heights():
    slab = build_slab("100", 5, 4, 4.09, 12)
    peo = conformers(catalog()["peo"].smiles, 2, 13)
    posed = placements(slab, "100", peo, 4, 13)
    row, atoms = next((row, atoms) for row, atoms in posed if row["orientation"] == "paired_donors")
    adsorbate = atoms[len(slab):]
    first = adsorbate[row["anchor"]]
    second = adsorbate[row["second_anchor"]]
    assert first.symbol == second.symbol == "O"
    assert first.position[2] == pytest.approx(second.position[2])
