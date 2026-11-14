"""Local Streamlit interface for preparing calculations and exploring results."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from ase import Atoms
from ase.geometry import find_mic
from ase.io import read
from rdkit import Chem

from ag_atlas.design import catalog


ADSORPTION_CSV = "adsorption_results.csv"
COMPARISON_CSV = "surface_comparison.csv"
RESULT_COLUMNS = {"model_id", "facet", "pose_id", "status", "energy_ads_ev"}
COMPARISON_COLUMNS = {
    "model_id",
    "status",
    "energy_111_ev",
    "energy_100_ev",
    "delta_111_minus_100_ev",
    "preferred_facet",
}
ELEMENT_COLORS = {
    "Ag": "#8593a3",
    "C": "#33465c",
    "H": "#d5dde6",
    "N": "#2459ce",
    "O": "#cc4659",
}
RESULT_DISPLAY = {
    "model_id": "Oligomer",
    "facet": "Ag surface",
    "pose_id": "Pose",
    "status": "Status",
    "geometry_kind": "Geometry",
    "energy_ads_ev": "Adsorption energy (eV)",
    "site_initial": "Initial site",
    "site_final": "Nearest final donor site",
    "anchor_initial": "Initial anchor",
    "conformer": "Conformer",
    "contact_min_ang": "Closest Ag–N/O contact (Å)",
}
COMPARISON_DISPLAY = {
    "model_id": "Oligomer",
    "status": "Status",
    "energy_111_ev": "Ag(111) (eV)",
    "energy_100_ev": "Ag(100) (eV)",
    "delta_111_minus_100_ev": "ΔE(111−100) (eV)",
    "preferred_facet": "Lowest sampled facet",
}


def read_table(source: Path | BinaryIO, required: set[str]) -> pd.DataFrame:
    table = pd.read_csv(source, dtype=str, keep_default_na=False)
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"Missing CSV columns: {', '.join(sorted(missing))}")
    return table


def local_geometry_path(run_dir: Path, value: str) -> Path:
    root = run_dir.resolve()
    target = (root / value).resolve()
    if not target.is_relative_to(root) or target.suffix.lower() != ".extxyz":
        raise ValueError("Geometry must be an .extxyz file inside the run directory.")
    return target


def contact_pairs(row: pd.Series, atoms: Atoms) -> list[tuple[int, int, float]]:
    raw_contacts = row.get("contacts", "")
    if not raw_contacts:
        return []
    contacts = json.loads(raw_contacts)
    if contacts == []:
        return []
    slab_count = int(row["slab_atom_count"])
    if not isinstance(contacts, list) or not 0 < slab_count <= len(atoms):
        raise ValueError("Invalid contacts or slab atom count.")
    pairs = []
    symbols = atoms.get_chemical_symbols()
    for item in contacts:
        molecule_index = slab_count + int(item["molecule_index"])
        silver_index = int(item["ag_index"])
        distance = float(item["distance_ang"])
        if (
            not slab_count <= molecule_index < len(atoms)
            or not 0 <= silver_index < slab_count
            or symbols[molecule_index] not in {"N", "O"}
            or symbols[silver_index] != "Ag"
            or not np.isfinite(distance)
            or distance <= 0
        ):
            raise ValueError("A contact does not match the displayed geometry.")
        pairs.append((molecule_index, silver_index, distance))
    return pairs


def molecular_bonds(row: pd.Series, atoms: Atoms) -> list[tuple[int, int]]:
    model = catalog().get(row["model_id"])
    smiles = row.get("smiles") or (model.smiles if model else "")
    if not smiles or not row.get("slab_atom_count"):
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    mol = Chem.AddHs(mol)
    start = int(row["slab_atom_count"])
    if start < 0 or [atom.GetSymbol() for atom in mol.GetAtoms()] != atoms.get_chemical_symbols()[start:]:
        return []
    return [(start + bond.GetBeginAtomIdx(), start + bond.GetEndAtomIdx()) for bond in mol.GetBonds()]


def structure_figure(atoms: Atoms, pairs: list[tuple[int, int, float]], bonds: list[tuple[int, int]]) -> go.Figure:
    figure = go.Figure()
    symbols = np.asarray(atoms.get_chemical_symbols())
    positions = atoms.get_positions()
    if bonds:
        lines = []
        for first, second in bonds:
            vector, _ = find_mic(positions[second] - positions[first], atoms.cell, atoms.pbc)
            lines.extend([positions[first], positions[first] + vector, [None, None, None]])
        x, y, z = zip(*lines, strict=True)
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name="Covalent connectivity", showlegend=False,
            line={"color": "#526174", "width": 4}, hoverinfo="skip",
        ))
    for element in dict.fromkeys(symbols):
        indices = np.flatnonzero(symbols == element)
        coordinates = positions[indices]
        figure.add_trace(
            go.Scatter3d(
                x=coordinates[:, 0], y=coordinates[:, 1], z=coordinates[:, 2],
                mode="markers", name=element,
                text=[f"{element} · atom {index + 1}" for index in indices],
                hovertemplate="%{text}<extra></extra>",
                marker={
                    "color": ELEMENT_COLORS.get(element, "#7f5a9c"),
                    "size": {"Ag": 7, "H": 4}.get(element, 6),
                    "line": {"width": 0.6, "color": "#ffffff"},
                    "opacity": 0.92,
                },
            )
        )
    for index, (donor, silver, distance) in enumerate(pairs):
        vector, _ = find_mic(positions[silver] - positions[donor], atoms.cell, atoms.pbc)
        line = np.vstack([positions[donor], positions[donor] + vector])
        figure.add_trace(
            go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=line[:, 2],
                mode="lines+markers", name="Ag–N/O contact", legendgroup="contacts",
                showlegend=index == 0,
                hovertemplate=f"{symbols[donor]}–Ag · {distance:.3f} Å<extra></extra>",
                line={"color": "#b77725", "width": 5, "dash": "dot"},
                marker={"size": 3, "color": "#b77725"},
            )
        )
    figure.update_layout(
        margin={"l": 0, "r": 0, "t": 8, "b": 0},
        height=485,
        legend={"orientation": "h", "y": 1.02},
        scene={
            "aspectmode": "data",
            "xaxis_title": "x (Å)", "yaxis_title": "y (Å)", "zaxis_title": "z (Å)",
            "camera": {"eye": {"x": 1.35, "y": 1.45, "z": 0.9}},
        },
    )
    return figure


def energy_figure(table: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    for facet, key, color in (
        ("Ag(111)", "energy_111_ev", "#215b73"),
        ("Ag(100)", "energy_100_ev", "#b46b2f"),
    ):
        figure.add_trace(
            go.Bar(
                name=facet,
                x=table["model_id"],
                y=pd.to_numeric(table[key], errors="coerce"),
                marker_color=color,
                hovertemplate="%{x}<br>%{y:.3f} eV<extra>%{fullData.name}</extra>",
            )
        )
    figure.update_layout(
        height=350, barmode="group", margin={"l": 8, "r": 8, "t": 8, "b": 8},
        legend={"orientation": "h", "y": 1.10}, yaxis_title="Adsorption energy (eV)",
    )
    return figure


def display_table(table: pd.DataFrame, labels: dict[str, str], numbers: set[str]) -> None:
    display = table[[key for key in labels if key in table]].copy()
    for column in numbers.intersection(display.columns):
        display[column] = pd.to_numeric(display[column], errors="coerce")
    st.dataframe(display.rename(columns=labels), hide_index=True, width="stretch")


def download_csv(table: pd.DataFrame, name: str, label: str) -> None:
    st.download_button(label, table.to_csv(index=False), name, "text/csv", key=f"download-{name}")


def show_structure(table: pd.DataFrame, run_dir: Path | None) -> None:
    if run_dir is None or "geometry_path" not in table:
        return
    available = table.loc[table["geometry_path"].ne("")].reset_index(drop=True)
    if available.empty:
        return

    def pose_label(index: int) -> str:
        row = available.iloc[index]
        return f"{row['model_id']} · Ag({row['facet']}) · {row['pose_id']}"

    st.subheader("Atomic structure")
    index = st.selectbox("Pose to display", range(len(available)), format_func=pose_label)
    row = available.iloc[index]
    try:
        path = local_geometry_path(run_dir, row["geometry_path"])
        atoms = read(path, format="extxyz")
    except (OSError, ValueError, KeyError, IndexError) as error:
        st.error(f"Could not read the selected geometry. {error}")
        return
    try:
        pairs = contact_pairs(row, atoms)
    except (ValueError, KeyError, TypeError, OverflowError):
        st.warning("The contact records for this pose could not be matched to its geometry.")
        pairs = []
    try:
        bonds = molecular_bonds(row, atoms)
    except (ValueError, TypeError):
        bonds = []
    st.plotly_chart(structure_figure(atoms, pairs, bonds), width="stretch")
    cutoff = row.get("contact_cutoff_ang", "")
    if pairs and cutoff:
        st.caption(f"Dotted lines show recorded Ag–N/O contacts within {cutoff} Å, using the minimum image.")
        with st.expander("Contact distances"):
            st.dataframe(
                pd.DataFrame([
                    {"Donor atom": f"{atoms[donor].symbol} {donor + 1}", "Silver atom": f"Ag {silver + 1}", "Distance (Å)": distance}
                    for donor, silver, distance in pairs
                ]),
                hide_index=True, width="stretch",
            )
    if bonds:
        st.caption("Solid lines trace the initial molecular connectivity.")
    if row.get("protocol_id"):
        st.caption(f"Calculation protocol `{row['protocol_id']}`")
    geometry_kind = row.get("geometry_kind", "relaxed")
    if geometry_kind == "fixed":
        st.caption("Fixed starting coordinates used in a single-point calculation.")
    st.download_button("Download structure", path.read_bytes(), path.name, "chemical/x-xyz")


def browse_results() -> None:
    source_type = st.radio("Data source", ["Local run folder", "CSV files"], horizontal=True)
    run_dir = None
    if source_type == "Local run folder":
        if last_root := st.session_state.get("atlas_last_plan"):
            st.button(
                "Use last prepared run folder",
                on_click=st.session_state.update,
                kwargs={"atlas_run_dir": last_root},
            )
        if "atlas_run_dir" not in st.session_state:
            st.session_state["atlas_run_dir"] = os.environ.get("AG_ATLAS_RUN_DIR", "")
        value = st.text_input("Run folder", key="atlas_run_dir", placeholder="runs/nh3-reference")
        if not value.strip():
            st.caption("Choose a folder containing adsorption_results.csv. Select CSV files to open a downloaded atlas export.")
            return
        run_dir = Path(value.strip()).expanduser()
        result_source = run_dir / ADSORPTION_CSV
        comparison_source = run_dir / COMPARISON_CSV
        if not result_source.is_file():
            st.info("This folder has no adsorption results yet. Collect the run after Quantum ESPRESSO finishes.")
            return
        if not comparison_source.is_file():
            comparison_source = None
    else:
        result_source = st.file_uploader("Adsorption results CSV", type="csv", key="upload-results")
        comparison_source = st.file_uploader("Surface comparison CSV (optional)", type="csv", key="upload-comparison")
        if result_source is None:
            return
    try:
        table = read_table(result_source, RESULT_COLUMNS)
    except (OSError, ValueError, UnicodeError) as error:
        st.error(f"Could not open the adsorption results. {error}")
        return
    if table.empty:
        st.info("The results file contains no calculation rows.")
        return
    filters = st.columns(3)
    selected_models = filters[0].multiselect("Oligomer", sorted(table["model_id"].unique()), placeholder="All oligomers")
    selected_facets = filters[1].multiselect(
        "Surface", sorted(table["facet"].unique()),
        format_func=lambda facet: f"Ag({facet})", placeholder="Both surfaces",
    )
    selected_statuses = filters[2].multiselect("Status", sorted(table["status"].unique()), placeholder="All statuses")
    filtered = table
    for key, selected in (("model_id", selected_models), ("facet", selected_facets), ("status", selected_statuses)):
        if selected:
            filtered = filtered.loc[filtered[key].isin(selected)]
    st.subheader("Adsorption poses")
    if "geometry_kind" in table and table["geometry_kind"].eq("fixed").any():
        st.info("This table includes fixed-geometry single-point calculations. Use a convergence study for research comparisons.")
    if filtered.empty:
        st.info("No poses match these filters.")
    else:
        display_table(filtered, RESULT_DISPLAY, {"energy_ads_ev", "contact_min_ang"})
        download_csv(filtered, ADSORPTION_CSV, "Download displayed poses (CSV)")
    if comparison_source is not None:
        st.subheader("Surface preference")
        st.caption("ΔE(111−100) compares the lowest-energy collected pose on each surface. Negative values favor Ag(111).")
        try:
            comparison = read_table(comparison_source, COMPARISON_COLUMNS)
        except (OSError, ValueError, UnicodeError) as error:
            st.error(f"Could not open the surface comparison. {error}")
        else:
            if selected_models:
                comparison = comparison.loc[comparison["model_id"].isin(selected_models)]
            if comparison.empty:
                st.info("No surface comparisons match the selected oligomers.")
            else:
                energy_keys = {"energy_111_ev", "energy_100_ev", "delta_111_minus_100_ev"}
                paired = comparison.loc[comparison[list(energy_keys)].apply(pd.to_numeric, errors="coerce").notna().all(axis=1)]
                if not paired.empty:
                    st.plotly_chart(energy_figure(paired), width="stretch")
                display_table(comparison, COMPARISON_DISPLAY, energy_keys)
                download_csv(comparison, COMPARISON_CSV, "Download surface comparisons (CSV)")
    if run_dir is not None:
        matched = run_dir / "paired_effects.csv"
        if matched.is_file():
            try:
                contrasts = read_table(matched, {"contrast", "model_a", "model_b", "status", "delta_a_minus_b_ev"})
                if not contrasts.empty:
                    with st.expander("Matched oligomer pairs"):
                        st.dataframe(contrasts, hide_index=True, width="stretch")
                        download_csv(contrasts, matched.name, "Download matched comparisons (CSV)")
            except (OSError, ValueError, UnicodeError) as error:
                st.warning(f"Could not open the matched comparison. {error}")
    show_structure(filtered, run_dir)


def prepare_calculations() -> None:
    models = catalog()
    labels = {key: model.name for key, model in models.items()}
    default_model = next((key for key, model in models.items() if model.smiles == "N"), next(iter(models)))
    selected = st.multiselect(
        "Oligomers and reference molecules", list(models), default=[default_model],
        format_func=labels.__getitem__, key="plan-models",
    )
    with st.expander("Add a custom molecule"):
        slug_column, smiles_column = st.columns([1, 2])
        custom_slug = slug_column.text_input("Model identifier", placeholder="my-oligomer").strip()
        custom_smiles = smiles_column.text_input("Molecular SMILES", placeholder="NCCN").strip()
        st.caption("Use a unique ASCII identifier with letters, numbers or hyphens and a single neutral, closed-shell molecule.")
    custom_models = {}
    custom_elements = set()
    custom_error = ""
    if custom_slug or custom_smiles:
        if (
            custom_slug in models or not custom_slug.isascii()
            or not custom_slug.replace("-", "").isalnum()
        ):
            custom_error = "Enter a unique ASCII identifier for the custom molecule."
        else:
            molecule = Chem.MolFromSmiles(custom_smiles) if custom_smiles else None
            if molecule is None or not molecule.GetNumAtoms() or len(Chem.GetMolFrags(molecule)) != 1 or any(
                atom.GetFormalCharge() or atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()
            ):
                custom_error = "Enter a valid SMILES for a single neutral, closed-shell molecule."
            else:
                custom_models[custom_slug] = custom_smiles
                custom_elements.update(atom.GetSymbol() for atom in molecule.GetAtoms())
    if custom_error:
        st.warning(custom_error)
    controls = st.columns(2)
    facets = controls[0].multiselect(
        "Ag surfaces", ["111", "100"], default=["111", "100"],
        format_func=lambda facet: f"Ag({facet})", key="plan-facets",
    )
    poses = controls[1].number_input("Starting poses per model and surface", min_value=1, max_value=24, value=4, step=1)
    if selected:
        with st.expander("View molecular model details"):
            for key in selected:
                model = models[key]
                st.markdown(f"**{model.name}** · {model.family}")
                st.write(model.description)
                st.code(model.smiles, language=None)
    required = {"Ag", "H"} | custom_elements
    for key in selected:
        molecule = Chem.AddHs(Chem.MolFromSmiles(models[key].smiles))
        required.update(atom.GetSymbol() for atom in molecule.GetAtoms())
    with st.form("plan-run"):
        with st.expander("Advanced cell settings"):
            molecule_box = st.number_input(
                "Isolated molecule box (Å)", min_value=12.0, max_value=80.0,
                value=32.0 if {"aaaeee", "aeaeae"}.intersection(selected) else 25.0, step=1.0,
            )
        output = st.text_input("New run folder", value="runs/reference")
        pseudo_dir = st.text_input("Pseudopotential folder", placeholder="/path/to/UPF-files")
        st.caption("Enter the UPF filenames from one consistent pseudopotential set.")
        columns = st.columns(len(required))
        pseudos = {
            element: column.text_input(f"{element} UPF filename", placeholder=f"{element}.UPF", key=f"pseudo-{element}")
            for element, column in zip(sorted(required), columns, strict=True)
        }
        submitted = st.form_submit_button("Prepare calculation files", type="primary")
    if submitted:
        if custom_error:
            st.error("Correct the custom molecule fields before preparing the run.")
            return
        model_ids = [*selected, *custom_models]
        if not model_ids or not facets or not output.strip() or not pseudo_dir.strip() or not all(v.strip() for v in pseudos.values()):
            st.error("Choose at least one molecule and surface, and fill in the run folder and UPF filenames.")
            return
        config = {
            "models": model_ids, "custom_models": custom_models, "facets": facets, "poses": int(poses),
            "molecule_box": molecule_box,
            "pseudo_dir": str(Path(pseudo_dir.strip()).expanduser().resolve()),
            "pseudos": {key: value.strip() for key, value in pseudos.items()},
        }
        root = Path(output.strip()).expanduser().resolve()
        from ag_atlas.workflow import plan

        try:
            with st.spinner("Building surfaces and starting geometries…"):
                manifest = plan(config, root)
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            st.error(f"Could not prepare the run. {error}")
            return
        st.session_state["atlas_last_plan"] = str(root)
        st.session_state["atlas_last_plan_warnings"] = manifest["warnings"]
    if last_root := st.session_state.get("atlas_last_plan"):
        st.success(f"Calculation files prepared in {last_root}")
        for warning in st.session_state.get("atlas_last_plan_warnings", []):
            st.warning(warning)
        st.code(
            "\n".join(shlex.join(["ag-atlas", command, last_root]) for command in ("run", "collect")),
            language="bash",
        )
        run_local(Path(last_root))


def run_local(root: Path) -> None:
    executable = shutil.which("pw.x")
    if executable is None:
        st.caption("Install Quantum ESPRESSO and add pw.x to your PATH to run calculations.")
        return
    processes = st.session_state.setdefault("atlas_processes", {})
    process = processes.get(str(root))
    active = process is not None and process.poll() is None
    logfile = root / "atlas-run.log"
    if st.button("Run Quantum ESPRESSO", disabled=active or (root / ".atlas.lock").exists()):
        try:
            with logfile.open("a", encoding="utf-8") as log:
                processes[str(root)] = subprocess.Popen(
                    [sys.executable, "-m", "ag_atlas", "run", str(root), "--executable", executable],
                    stdout=log, stderr=subprocess.STDOUT,
                )
        except OSError as error:
            st.error(f"Could not start Quantum ESPRESSO. {error}")
        else:
            st.rerun()
    if active:
        st.info("Quantum ESPRESSO is running. Refresh to see the latest job status.")
        st.button("Refresh run status")
    elif process is not None:
        if process.returncode == 0:
            st.success("Calculations completed. Open the run folder in Explore results.")
        else:
            st.warning("One or more jobs did not complete. Review the job log and CSV.")
    if logfile.is_file():
        with st.expander("Calculation log"):
            st.code(logfile.read_text(encoding="utf-8", errors="replace")[-8000:])



def main() -> None:
    st.set_page_config(page_title="Ag Flake–Polymer Atlas", layout="wide")
    st.title("Ag Flake–Polymer Atlas")
    st.write("Explore oligomer adsorption on Ag(111) and Ag(100), or prepare a Quantum ESPRESSO run.")
    explore, prepare = st.tabs(["Explore results", "Prepare calculations"])
    with explore:
        browse_results()
    with prepare:
        prepare_calculations()


def run_streamlit(args=None) -> int:
    return subprocess.call(
        [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()), "--server.address=127.0.0.1", *(sys.argv[1:] if args is None else args)]
    )


if __name__ == "__main__":
    main()
