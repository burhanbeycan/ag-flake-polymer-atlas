# Reproduce the adsorption study

Install the atlas with `python -m pip install '.[publication]'` for plotting.
Install Quantum ESPRESSO and one consistent pseudopotential set.
The SSSP 1.3.0 PBE efficiency names in the example inputs are
listed at https://archive.materialscloud.org/records/rcyfm-68h65.
Use a shared absolute UPF directory on a compute cluster.

## Generate the study

```bash
python scripts/plan_publication.py runs/production --pseudo-dir /shared/sssp --scope all
```

A case directory holds immutable QE inputs and an input manifest. The study
index records the case role and which settings each case varies. Each setting
sensitivity is measured against a baseline within its series. The enlarged
reference cells use a roughly comparable reciprocal sampling density, so
their comparisons are labeled as a joint change of size and k mesh.
Dispersion damping is model sensitivity and should be interpreted separately
from numerical convergence. The listed settings are proposals for verification.

## Calculate and collect

Run cases sequentially locally using `ag-atlas run runs/production/nh3-base`,
or submit each case as a separate SLURM array on an MPI cluster.

```bash
export ATLAS_MPI_TASKS=8 ATLAS_MAX_ACTIVE=4 ATLAS_MEMORY=32G ATLAS_WALLTIME=12:00:00
scripts/submit_slurm.sh runs/production/nh3-base
```

The array allocates independent jobs with input hashes checked before launch.
Its dependent collection job runs after all array tasks finish, including
unsuccessful tasks. Set site-specific partition and account through SLURM
configuration or SBATCH environment variables. Use an MPI-enabled QE build
with the cluster's srun configuration. CPU, memory and time values are
initial resource requests. Benchmark a complex first and adjust them for the
larger oligomer cells. Avoid changing physical inputs inside a case.

Use the collected status to identify failures. A job that reached its limit
can be retried with more resources using the same immutable inputs. If a
different mixing setting, starting pose, or cutoff is needed, create a
new named case so the altered protocol remains traceable.

## Generate comparisons

```bash
python -m ag_atlas.publication runs/production/cases.json runs/publication_tables --plot
```

Collect each prepared case before running the analysis, including cases that
have not yet run, to write the status CSVs.

Keep failed and pending cases in the study index. Include completed model
comparisons alongside the sensitivity to numerical settings and pose-pool
size. Quantitative adsorption claims require relaxed geometries and compatible
reference energies. Retain raw inputs, outputs, pseudopotential provenance and
the exact source revision with every reported figure and table.
