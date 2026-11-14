# Real DFT on a local cloud machine

The standard Python development installation does not include a DFT engine or
pseudopotentials. On Debian 13 amd64, the tested rootless installer provides
Quantum ESPRESSO 6.7, SSSP 1.3.0 PBE efficiency, and optimized OpenBLAS:

```bash
bash scripts/setup_qe_debian.sh /workspace/tools
source .venv/bin/activate
export PATH="/workspace/tools/bin:$PATH"
```

It uses signed Debian package metadata, pins package versions and SHA256
checksums, and extracts packages outside the checkout. TLS and signature
verification remain enabled. It requires the host's Debian runtime libraries
(MPI, FFTW, gfortran); BLAS/LAPACK are supplied by the pinned OpenBLAS package.
Missing libraries cause installation to fail
with a diagnostic. The supplied cloud machine has these libraries. Other
operating systems should install Quantum ESPRESSO through their supported
package manager. The installer does not require root or modify system files.

The UPF directory is
`/workspace/tools/quantum-espresso/usr/share/espresso/pseudo`. It contains the
exact Ag/C/H/N/O filenames used in the examples. UPFs retain their original
distribution terms; do not add installed files or generated runs to Git.

## Check execution before production

Use a new output directory for every check:

```bash
python scripts/validate_qe.py runs/real-qe-smoke \
  --pseudo-dir /workspace/tools/quantum-espresso/usr/share/espresso/pseudo \
  --executable pw.x --threads 1 --timeout 600
python scripts/validate_qe.py runs/real-qe-relax \
  --pseudo-dir /workspace/tools/quantum-espresso/usr/share/espresso/pseudo \
  --executable pw.x --threads 1 --timeout 1800 --relax
```

Each check runs five real calculations: two clean Ag slabs, free NH3, and NH3
on both facets. Success requires all five to converge, compatible reference
energies, finite energies, and successful collection. The relaxation check
also exercises BFGS and final-coordinate parsing. Failures remain in `pw.out`
and `jobs.csv`; the command exits nonzero. `validation.json` records the outcome,
energies, engine version, input protocol and UPF hashes. Keep the complete run
folder to retain the raw evidence.

These are software execution checks at deliberately small cells and low
cutoffs (25/140 Ry, gamma sampling). They do **not** establish converged
adsorption energies or publication-quality facet preferences. In particular,
a successful relaxation at these settings is not a numerical convergence
study. Fixed-geometry collection is labeled `fixed_geometry` and supplies no
facet preference. See [the scientific protocol](protocol.md).

## Prepare and run the complete study

```bash
python scripts/plan_publication.py runs/production \
  --pseudo-dir /workspace/tools/quantum-espresso/usr/share/espresso/pseudo \
  --scope all
(
  # For the 4-CPU / 32-GiB cloud machine: at most 6 GiB virtual memory per rank.
  ulimit -v 6291456
  ATLAS_MPI_TASKS=4 python scripts/run_local_study.py runs/production \
    --executable pw-mpi.x --threads 1
)
```

`pw-mpi.x` launches four MPI ranks by default; set `ATLAS_MPI_TASKS` to a
positive integer to change the rank count. It preserves the supplied thread
setting. The Debian QE build used here relies on MPI for CPU parallelism;
raising `--threads` alone does not demonstrate use of more CPU cores. Allocate
ranks and threads within the machine's CPU and memory limits. Do not run
multiple full study drivers at once.
The installer sets one OpenBLAS thread per MPI rank by default;
`ATLAS_BLAS_THREADS` explicitly overrides this. The previous host reference
BLAS was much slower than this optimized implementation.
The example memory limit prevents a large rank from exhausting the entire
machine. Jobs exceeding it remain failed and require more memory; the driver
does not lower physical cutoffs or silently skip them. Change that limit only
within the total memory available for all ranks.

The driver processes cases sequentially, writes progress in the study
directory, collects completed and failed cases, and exits nonzero for
incomplete work. Re-running the same command skips converged jobs and retries
unusable outputs without changing the immutable inputs. `--timeout` optionally
limits each individual job in seconds. A timeout is a failed attempt, not a
completed calculation; retry it with suitable resources. Processes do not
survive environment publication or machine replacement. Restart the command
to resume from retained outputs.

If a machine was stopped abruptly, an old `.atlas.lock` or
`.local-study-runner.lock` file may remain. Check that no process is using the
corresponding run before removing that stale lock and restarting. Never remove
a lock for a live calculation.

With the tested RDKit version and seed, the complete plan has **30 cases and
803 jobs**. Many cases contain 144 or more Ag atoms, high cutoffs, dense
k-point sampling, multiple poses, and ionic relaxations. Preparing their
inputs does not execute them. A small cloud machine may need a long time or
more memory to finish this study; do not describe pending or failed jobs as
converged. Monitor `pw.out`, the driver's progress report, and the case CSVs.
The [publication guide](publication_runs.md) also supports SLURM arrays.

This is a measured limitation: a 260-atom production input reports **305.73
GiB** of dynamic memory and fails allocation under the 32-GiB machine's safety
limit. Faster BLAS and a different MPI layout do not remove this memory
requirement. The complete unchanged study cannot finish on this machine.
See the [resource check](validation/2026-11-14-resources.md).

To avoid repeating identical computations within one uninterrupted study run,
add these options to the local driver:

```bash
--reuse-identical --engine-binary /workspace/tools/quantum-espresso/usr/bin/pw.x
```

This is opt-in. Only newly executed, converged outputs can enter the cache.
Input bytes, starting geometry, pseudopotentials, engine identity and source
output hashes must match. Every copy gets `reuse.json` with its origin and
protocol IDs; prior failed outputs are archived. Existing completed outputs
are never replaced. Conservative matching can reduce this plan from 803
logical jobs to 562 actual executions if all converge in one invocation.
Restarting retains completed outputs but does not seed the cache from old
executions with an unverified runtime identity.

After the intended cases finish, generate provenance-checked tables:

```bash
python -m ag_atlas.publication runs/production/cases.json runs/publication_tables --plot
```

Publication eligibility still depends on completed matching references,
relaxed geometries and independent numerical convergence checks.
