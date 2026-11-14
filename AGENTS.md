# Ag Flake–Polymer Atlas

- Preserve the neutral, finite oligomer and clean Ag(111)/Ag(100) scope of the reference protocol. Changes to scientific assumptions belong in `docs/protocol.md` with their provenance and an explicit configuration.
- Preserve matching clean slab and free oligomer references for adsorption energies. Calculate facet comparisons only from compatible, converged results. Keep the sign convention in the CLI, CSV and GUI consistent.
- Keep model IDs and atom indexing stable. A modified chemical structure needs a new identity. Generate poses reproducibly from the recorded seed and retain the initial and final geometries separately.
- Keep parser and workflow failures visible in the job records. Distinguish fixtures and smoke runs from calculations that meet the publication protocol.
- Do not add generated run directories, locally installed pseudopotentials or secrets to Git. Record exact pseudopotential filenames and hashes in calculation provenance.
- Run `python -m pytest` before changing scientific or user-facing behavior. Run the CLI planning path when changing configuration or adding structures.
