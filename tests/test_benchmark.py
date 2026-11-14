import csv
import hashlib
from collections import defaultdict

import numpy as np

from ag_atlas.benchmark import _adaptive_order, benchmark
from ag_atlas.workflow import _json


def _fixture(root, partial=True):
    protocol = {"settings": {"poses": 5}}
    protocol_id = hashlib.sha256(_json(protocol).encode()).hexdigest()[:16]
    jobs, labels = [], []
    for facet in ("111", "100"):
        for pose in range(1, 6):
            job_id = f"ads-nh3-{facet}-{pose:02}"
            jobs.append({
                "role": "complex", "job_id": job_id, "model_id": "nh3", "facet": facet,
                "pose_id": pose, "site": ("ontop", "bridge", "fcc")[pose % 3],
                "anchor": 0, "anchor_role": "ammonia", "conformer": 0, "orientation": "donor",
            })
            labels.append({
                "job_id": job_id, "protocol_id": protocol_id, "model_id": "nh3",
                "facet": facet, "pose_id": pose, "geometry_kind": "relaxed", "qe_version": "7.2",
                "status": "pending" if partial and facet == "100" and pose == 5 else "converged",
                "energy_ads_ev": -.1 * pose,
            })
    (root / "manifest.json").write_text(_json({
        "schema_version": 1, "protocol": protocol, "protocol_id": protocol_id, "jobs": jobs,
    }), encoding="utf-8")
    with (root / "adsorption_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, labels[0])
        writer.writeheader()
        writer.writerows(labels)


def test_benchmark_equal_budget_and_only_complete_pools(tmp_path):
    _fixture(tmp_path)
    result = benchmark(tmp_path, 3, trials=4, seed=24)
    assert result["pools"] == 1
    assert result["skipped_pools"][0]["facet"] == "100"
    assert result["rows"] == 18
    with result["path"].open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_method = defaultdict(list)
    for row in rows:
        by_method[row["method"], row["trial"]].append(row)
    assert len(by_method) == 6
    for steps in by_method.values():
        assert len({row["job_id"] for row in steps}) == 3
        assert steps[0]["job_id"] == "ads-nh3-111-01"
        assert steps[0]["regret_ev"] == "0.4"
        regrets = [float(row["regret_ev"]) for row in steps]
        assert regrets == sorted(regrets, reverse=True)
    original = result["path"].read_bytes()
    benchmark(tmp_path, 3, trials=4, seed=24)
    assert result["path"].read_bytes() == original


def test_adaptive_does_not_read_unseen_energy_before_first_choice():
    kernel = np.asarray([[1., .2, .3], [.2, 1., .5], [.3, .5, 1.]])
    first = _adaptive_order(kernel, np.array([0., 100., -100.]), 2)
    second = _adaptive_order(kernel, np.array([0., -100., 100.]), 2)
    assert first == second == [0, 1]


def test_exhaustive_budget_writes_header_and_reports_skipped(tmp_path):
    _fixture(tmp_path, partial=False)
    result = benchmark(tmp_path, 5)
    assert result["pools"] == 0
    assert result["rows"] == 0
    assert len(result["skipped_pools"]) == 2
    assert result["path"].read_text().startswith("model_id,facet")
