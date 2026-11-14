# Production resource and performance check — 2026-11-14

The complete 803-job study has not finished. The four-CPU, 32-GiB cloud machine
has a confirmed memory limitation for the unchanged production inputs.

## Actual allocation failure

The following genuine QE attempt used the installed SSSP set, the original
immutable input and four MPI ranks:

- Case/job: `matched-layers-thicker/ads-bpei-221-111-01`.
- Protocol ID: `c1e75375ee8c76ae`.
- Input SHA256: `648d59c7698c78936b20ff9f6802615c12e3107f5bacf0a94a0787e87cfe6507`.
- 260 atoms, 2,521 bands, 5 k points, 60/480 Ry.
- QE estimate: **76.43 GiB per rank, 305.73 GiB total**.
- Result: failed, exit status 1; `allocate_wfc.f90` reported
  `Error allocating 5824034368 bytes: Cannot allocate memory`.

The execution inherited a 6-GiB virtual-memory cap per rank to protect the
32-GiB host. The 5.82-billion-byte allocation is only one array, not the job's
total requirement. Raising that cap on the same host would risk exhausting
the machine; it does not provide the reported total memory.

QE's memory estimate labels units as GB but uses 1024 cubed bytes. For this
input, one full complex wavefunction set is approximately 21.69 GiB. With
`disk_io='none'`, the buffered sets alone are approximately 130.14 GiB; the
default Davidson workspace requires additional sets. MPI pools and band
groups redistribute storage rather than removing the aggregate requirement.
See QE 6.7's [memory estimator](https://github.com/QEF/q-e/blob/qe-6.7.0/PW/src/memory_report.f90).

The input and failure log remain in the ignored study folder. The failure is
not a converged calculation and cannot support adsorption or facet claims.
A larger-memory machine is required to finish this unchanged study; the
largest jobs should be profiled there before selecting the final allocation.

## Removing avoidable work

The original QE process linked the host's reference BLAS/LAPACK. The rootless
installer now supplies Debian `libopenblas0-pthread=0.3.29+ds-3`:

- Package SHA256: `e3541ee2c956c64383054ba116fb6ef86bbaf40035d64e1795185029fdcf0807`.
- Representative complex matrix multiplication under the same load:
  9.49 seconds with reference BLAS, 1.19 seconds with OpenBLAS.
- Relative matrix-result difference: `2.95e-15`.
- Real NH3 smoke molecule: 3.15 seconds with one MPI rank and 1.64 seconds
  with four; both recovered **−322.35509561433787 eV**, matching the earlier
  reference-BLAS calculation at the parsed precision.

These are kernel and small-job measurements, not an extrapolated completion
time for the full study.

An exact-byte audit found 561 distinct inputs among 803 logical jobs.
Conservative reuse additionally requires identical complete UPF maps, which
leaves **562 executions**. The local driver records the original source and
both protocols for every reused output. It never treats a failed or pending
source as reusable and never changes the destination inputs or identities.
The software suite passed **41 tests**, including tamper rejection and
preservation of prior attempts.
