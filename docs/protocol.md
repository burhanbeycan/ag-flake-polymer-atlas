# Computational protocol

## Scientific question and models

Test whether the spatial arrangement of the same donor inventory alters local binding or the difference between Ag(111) and Ag(100). The models are neutral, closed-shell finite molecules on ideal clean fcc silver terraces in periodic vacuum. Results describe low-coverage electronic adsorption within that model. Predicting adhesion, colloidal shape selection, or conductivity requires additional models for the relevant morphology, environment and application.

- Compare `bpei-311` with `bpei-221`. Both have formula C₁₀H₂₈N₆ and three primary, two secondary and one tertiary amine. They vary in the distribution of ethylene bridges among the three arms extending from the central tertiary N. Linear and branched models remain informative family comparisons when their different amine-class inventories are accounted for.
- Compare `aaee` with `aeae` using the same DP, end groups and 50% EI fraction. `A` denotes an N-propionyl ethyleneimine (EtOx-derived) unit and `E` an ethyleneimine unit with N–H. The methyl-capped models share the same terminal repeat types. The `aaaeee`/`aeaeae` pair tests the trend at DP6. These sequences are explicit constitutional models of partially hydrolyzed PEtOx chemistry. Their distribution in a particular polymer sample would require separate characterization. [de la Rosa et al.](https://doi.org/10.1039/C4PY00355A)
- Use PVP and PEO molecular proxies as comparison cases. Record the actual SMILES and model identities when comparing to published repeat-unit calculations. [Zhou et al.](https://doi.org/10.1021/jp403318h)

## Design and reference energies

DesignerAgent embeds candidate conformers, places them on an ideal surface, and records donor indices and initial sites. Vary orientation, contact topology and surface site when testing low-energy motifs. Top, bridge and facet-specific hollow starts can relax to a different final environment. Use the optimized geometry to interpret contact motif and record geometric criteria such as donor identity, distance threshold and periodic image convention. For larger molecules, ensure adequate lateral separation across periodic images. Report the lowest-energy eligible pose as a minimum **within the sampled pool** and retain other distinct low-energy poses.

For the same molecule M and facet F, use the compatible reference calculations:

```text
E_ads(F, M) = E(F + M) − E(F) − E(M)
ΔE_facet(M) = E_ads(111, M) − E_ads(100, M)
```

All terms use eV. A negative adsorption energy denotes favorable electronic adsorption relative to the calculated references. A negative facet difference favors Ag(111), and a positive difference favors Ag(100). Match the clean slab to each complex for its cell, facet, constraints, pseudopotentials and numerical settings. Use the corresponding gas-phase molecule energy and a converged free-molecule conformer search. For headline claims, include the sensitivity to the choice of free-molecule conformer and to the pose search. Converge the vacuum and lateral area as well as cutoff and k sampling so facet differences reflect local binding at comparably low coverages.

A selected energy decomposition can be done with additional single-point calculations on the molecular and slab fragments extracted at the optimized adsorption geometry. Use compatible cells, settings and isolated fragment conventions, and report how the deformation and interaction terms are constructed. The standard CSV reports the direct adsorption energy. The paired table reports the difference of facet energy differences in the displayed model order. A sampled energy ordering and its resolution under changes in numerical settings should be reported separately.

## Starting settings and validation

| Parameter | Starting value |
| --- | --- |
| XC functional and dispersion | PBE + D3(BJ), with QE `dftd3_version=4` |
| Silver slabs | `6 × 6` lateral repeats and four layers with the lower two fixed |
| Plane-wave / density cutoffs | `60 / 480 Ry`, subject to the chosen pseudopotentials and convergence tests |
| Surface Brillouin-zone sampling | `3 × 3 × 1` |
| Free-molecule reference | Gamma-point sampling in a separate periodic vacuum box |
| Geometry treatment | Fixed-cell ionic relaxation for pilot production calculations |

Record the exact QE version, numerical settings, pseudopotential release/filenames/hashes, seed, constraints, convergence outcome and corresponding reference job IDs. Establish production tolerances for adsorption and facet differences by increasing the cutoffs, k mesh, vacuum, lateral cell and slab thickness. Check smearing and the sensitivity to the dispersion treatment. A sign reversal or an effect comparable to convergence sensitivity calls for reporting an unresolved facet preference. Use [QE input documentation](https://www.quantum-espresso.org/Doc/INPUT_PW.html), the [SSSP library](https://sssp.materialscloud.org/) and [Hofmann et al.](https://doi.org/10.1039/D0CP06605B) for practical details.

Use three reference levels before expanding the atlas:

1. NH₃ on both facets to check the Ag–N site and order of binding energies against published DFT. Ag(001) is symmetry-equivalent to Ag(100) for fcc Ag. Expect method-dependent magnitudes. [Stachiotti](https://doi.org/10.1103/PhysRevB.79.115405)
2. Benzene on Ag(111) for an experimental reference with both adsorption height and binding energy. Compare observables and account for reference conditions and definitions. [Liu et al.](https://doi.org/10.1103/PhysRevLett.115.036104)
3. PVP and PEO proxy poses to check trends against documented model definitions. Consider that the literature reports environment-dependent selectivity for PVP. [Zhou et al.](https://doi.org/10.1021/jp403318h) and [Xu et al.](https://doi.org/10.1039/D2NR06762E)

A smoke run checks execution and parsing. Publication runs require converged settings, repeated starting conformers, appropriately matched reference energies and a versioned dataset. A run that completes successfully at reduced cell sizes, relaxed thresholds, or with single-point starting poses is suitable for software verification.

## Pilot and paper milestones

1. Run the NH₃ and benzene checks, then complete the first convergence study on representative N, O and aromatic motifs.
2. Prepare four starts per structure and facet for `bpei-311`, `bpei-221`, `aaee` and `aeae` (32 starting poses). Adding `pvp` and `peo` gives 48 starts. Expand the sampling for the headline comparisons where starting-pose spread, surface sensitivity or chemical interpretation warrants it. Include DP6 before making a claim about transfer across chain lengths.
3. If adding an AI-guided selector, use a shared pool and equal calculation budgets. Compare to seeded random and systematic selection. Measure the energy gap to the best fully evaluated pool result, recovery of relaxed contact motifs, recovery of the facet-preference sign, and actual CPU cost. Preserve job failures in that comparison. Existing LLM-guided pose search provides a method baseline. [Adsorb-Agent](https://doi.org/10.1039/D5DD00298B)
4. Release versioned inputs, outputs or archives, manifest, CSV tables, and figure-generation scripts together. Structure the paper results around reference validation, matched bPEI effects, matched EtOx–EI effects, and method sensitivity. Target *The Journal of Physical Chemistry C* first for a substantiated Ag interfacial physical chemistry result.
