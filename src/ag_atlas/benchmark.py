"""Replay pose selection against fully evaluated, finite DFT candidate pools."""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.spatial.distance import cdist

from .workflow import _csv, _lock, _read_manifest

_COLUMNS = (
    "model_id facet geometry_kind protocol_id method trial seed pool_size evaluation_budget "
    "step job_id selected_energy_ev best_sampled_energy_ev running_best_energy_ev regret_ev"
).split()
_CATEGORICAL = ("site", "conformer", "anchor", "second_anchor", "anchor_role", "orientation")
_NUMERICAL = ("vertical_image_gap_angstrom", "lateral_image_clearance_angstrom")


def _kernel(jobs):
    features = []
    for key in _CATEGORICAL:
        values = [str(job.get(key, "")) for job in jobs]
        for value in sorted(set(values)):
            features.append(np.asarray([item == value for item in values], dtype=float))
    for key in _NUMERICAL:
        values = np.asarray([float(job.get(key, 0.)) for job in jobs])
        if np.ptp(values):
            features.append((values - values.mean()) / max(values.std(), 1.))
    distances = cdist(np.asarray(features).T, np.asarray(features).T, metric="sqeuclidean")
    # Each category mismatch contributes two units to squared feature distance.
    return np.exp(-distances / 6.)


def _adaptive_order(kernel, energies, budget):
    """Fixed-kernel GP with a lower confidence bound using observed labels only."""
    selected = [0]
    while len(selected) < budget:
        unseen = np.asarray(sorted(set(range(len(kernel))) - set(selected)))
        observed = energies[selected]
        mean = observed.mean()
        scale = max(observed.std(), .05)
        train_kernel = kernel[np.ix_(selected, selected)]
        fit = cho_factor(train_kernel + .02 * np.eye(len(selected)))
        cross = kernel[np.ix_(unseen, selected)]
        predicted = mean + scale * (cross @ cho_solve(fit, (observed - mean) / scale))
        variance = 1. - np.sum(cross.T * cho_solve(fit, cross.T), axis=0)
        acquisition = predicted - scale * np.sqrt(np.maximum(variance, 0.))
        selected.append(int(unseen[np.argmin(acquisition)]))
    return selected


def _available_labels(pool, labels, protocol_id, expected_poses):
    if len(pool) != expected_poses:
        return None, "fewer poses were prepared than requested"
    if not all(job["job_id"] in labels for job in pool):
        return None, "adsorption rows are missing"
    group = [labels[job["job_id"]] for job in pool]
    for job, row in zip(pool, group, strict=True):
        if (row["protocol_id"], row["model_id"], row["facet"], row["pose_id"]) != (
            protocol_id, job["model_id"], job["facet"], str(job["pose_id"]),
        ):
            return None, "adsorption rows disagree with the manifest"
    if any(row["status"] != "converged" for row in group):
        return None, "one or more adsorption calculations are incomplete"
    if len({row["geometry_kind"] for row in group}) != 1:
        return None, "adsorption rows mix incompatible geometry kinds"
    if len({row["qe_version"] for row in group}) != 1:
        return None, "adsorption rows mix Quantum ESPRESSO versions"
    try:
        energies = np.asarray([float(row["energy_ads_ev"]) for row in group])
    except (ValueError, TypeError):
        return None, "adsorption energies are missing or malformed"
    if not np.isfinite(energies).all():
        return None, "adsorption energies are non-finite"
    return energies, group[0]["geometry_kind"]


def benchmark(root, budget: int, trials: int = 20, seed: int = 42):
    """Export equal-budget selection trajectories and regret against each recorded pool.

    All methods begin at the same first recorded pose. Random selection repeats
    with independent seeded trials; deterministic methods have one trajectory.
    Energies for unselected candidates are only used for retrospective scoring.
    """
    if budget < 2 or trials < 1 or seed < 0:
        raise ValueError("Use a budget of at least two poses, positive trials and a nonnegative seed.")
    root = Path(root).expanduser().resolve()
    with _lock(root):
        manifest = _read_manifest(root)
        protocol_id = manifest["protocol_id"]
        expected = manifest["protocol"]["settings"]["poses"]
        labels = {}
        with (root / "adsorption_results.csv").open(encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                job_id = row["job_id"]
                if job_id in labels:
                    raise ValueError(f"Duplicate adsorption result for {job_id}.")
                labels[job_id] = row
        pools = defaultdict(list)
        for job in manifest["jobs"]:
            if job["role"] == "complex":
                pools[job["model_id"], job["facet"]].append(job)
        rows, skipped = [], []
        rng = np.random.default_rng(seed)
        for (model, facet), pool in sorted(pools.items()):
            pool.sort(key=lambda job: job["pose_id"])
            energies, details = _available_labels(pool, labels, protocol_id, expected)
            if energies is None or len(pool) <= budget:
                reason = details if energies is None else "pool has no unevaluated poses at the requested budget"
                skipped.append({"model_id": model, "facet": facet, "reason": reason})
                continue
            targets = float(energies.min())
            orders = [
                ("systematic", 1, range(budget)),
                ("adaptive_gp_lcb", 1, _adaptive_order(_kernel(pool), energies, budget)),
            ]
            orders.extend(
                ("random", index + 1, [0, *rng.permutation(np.arange(1, len(pool)))[:budget - 1]])
                for index in range(trials)
            )
            for method, trial, order in orders:
                running_best = float("inf")
                for step, index in enumerate(order, start=1):
                    energy = float(energies[index])
                    running_best = min(running_best, energy)
                    rows.append({
                        "model_id": model, "facet": facet, "geometry_kind": details,
                        "protocol_id": protocol_id, "method": method, "trial": trial, "seed": seed,
                        "pool_size": len(pool), "evaluation_budget": budget, "step": step,
                        "job_id": pool[index]["job_id"], "selected_energy_ev": energy,
                        "best_sampled_energy_ev": targets, "running_best_energy_ev": running_best,
                        "regret_ev": running_best - targets,
                    })
        path = root / "pose_benchmark.csv"
        _csv(path, _COLUMNS, rows)
    return {"path": path, "pools": len(pools) - len(skipped), "skipped_pools": skipped, "rows": len(rows)}
