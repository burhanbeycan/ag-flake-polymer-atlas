# Ag Flake–Polymer Atlas

Generate explicit oligomer structures and adsorption starting geometries on clean Ag(111) and Ag(100). Run them with Quantum ESPRESSO, inspect the relaxed contacts, and export reproducible adsorption and facet-comparison tables.

## Install

Use Python 3.10 or later. Install [Quantum ESPRESSO](https://www.quantum-espresso.org/) separately and make `pw.x` available on your `PATH`. Download and extract a consistent PBE pseudopotential set into a local folder. The example filenames are from [SSSP 1.3.0 PBE efficiency](https://archive.materialscloud.org/records/rcyfm-68h65). See the [scientific protocol](docs/protocol.md) before treating a run as research data.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[gui]'
```

## Calculate a reference

From the repository root, edit `examples/nh3.json` to point to your UPF directory and match your actual filenames. A relative `pseudo_dir` is resolved from the configuration file directory. You can also pass `--pseudo-dir /path/to/upf` to `plan`. The example uses NH₃ on both Ag facets.

```bash
ag-atlas plan examples/nh3.json runs/nh3
ag-atlas run runs/nh3 --executable pw.x --threads 2
ag-atlas collect runs/nh3
```

Planning builds the clean slabs, free molecule and distinct adsorption starts. AutoDFT runs QE jobs in individual directories. Collection evaluates each pose against its matching clean slab and free molecule. A failed or incomplete job remains visible in `jobs.csv`. The starting settings are for a pilot and need independent convergence checks for a publication.

| Export | Contents |
| --- | --- |
| `adsorption_results.csv` | One row per planned complex. Adsorption energy is reported when the calculation and references pass convergence and compatibility checks. |
| `surface_comparison.csv` | Lowest eligible energy sampled on each face and a completion status. Negative `E_ads(111) − E_ads(100)` favors Ag(111) when the result is numerically resolved. |
| `paired_effects.csv` | Difference between the two facet-selectivity values for each matched oligomer pair. |
| `jobs.csv` | Job-level calculation status and run diagnostics. |

Keep the manifest, input and output files, and geometry files with these tables to preserve provenance. The CSVs can be used independently for analysis. The full run folder is needed to inspect atomistic geometries.

Use `examples/smoke.json` for a reduced-cost, fixed-geometry execution check, `examples/benzene.json` for the aromatic reference, and `examples/pilot.json` and `examples/dp6.json` to prepare the matched polymer panels. The default runner uses a single QE job at a time. Larger plans can be dispatched as independent job folders on a compute cluster, with one `pw.out` per folder, then merged by `collect`. Choose a fresh output directory when changing settings or input structures. To add a custom structure, add a `custom_models` object such as `{"ethylenediamine": "NCCN"}` in the configuration and include its identifier in `models`. Supply a UPF file for every required element. The model generator accepts one neutral, closed-shell molecule per SMILES.

## Open the GUI

```bash
ag-atlas-gui
```

**Prepare calculations** lets you choose structures, surfaces and the number of starting poses, and enter your pseudopotential filenames. When `pw.x` is on your PATH, the GUI can launch AutoDFT locally and show its log. It also displays the equivalent terminal commands. **Explore results** opens a local run folder or downloaded CSV files and offers filtering and CSV export. When a run folder contains the corresponding geometry, the explorer can render the pose and its Ag–N/O contacts in 3D. The GUI runs locally.

## Research models

| Models | Purpose |
| --- | --- |
| `nh3`, `benzene` | Small-molecule references. |
| `bpei-311`, `bpei-221` | Formula- and amine-class-matched branched PEI models. |
| `lpei-n4`, `bpei-n4`, `lpei-n6` | PEI architecture comparisons with explicit finite structures. |
| `aaee`, `aeae`, `aaaeee`, `aeaeae` | Matched EtOx–EI sequence models. |
| `aaaa`, `eeee` | Homosequence references for the EtOx–EI series. |
| `pvp`, `peo`, `pva`, `pvac` | Small, explicitly defined proxies for common polymer chemistries. |

DesignerAgent uses reproducible structure and placement rules. The generated starts sample a finite pose pool. To benchmark adaptive GP selection on completed pools, compare it with systematic and seeded random selections at the same budget:

```bash
ag-atlas benchmark runs/pilot --budget 2 --trials 20
```

The optional benchmark exports `pose_benchmark.csv`, with regret against the best DFT energy in each fully evaluated pool. For a meaningful methods study, use a larger independently evaluated candidate pool than the first four-pose pilot. AutoDFT uses the external `pw.x` engine for energies and relaxation. See the [protocol](docs/protocol.md) for the model limits, convergence plan, and requirements for an AI-guided selection benchmark.

## Development

```bash
python -m pip install -e '.[gui,dev]'
python -m pytest
```

Software and documentation are available under the [MIT license](LICENSE). Pseudopotentials retain the terms of their respective distributions.

## Publication workflows

The [publication run guide](docs/publication_runs.md) covers reference
and oligomer sensitivity cases, local execution, SLURM arrays, and
provenance-checked publication tables.
