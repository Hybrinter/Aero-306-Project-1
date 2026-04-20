"""End-to-end smoke test for the geometry optimization CLI."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "optimize_geometry.py"
BASE_YAML = REPO_ROOT / "config" / "problem_7.yaml"


@pytest.mark.slow
def test_cli_smoke_run_completes_and_beats_baseline(tmp_path: Path) -> None:
    """Run the CLI on smoke budget and confirm artifacts exist + best_design.yaml re-solves."""
    out_dir = tmp_path / "optimization_runs"
    run_id = "pytest_smoke"

    cmd = [
        sys.executable, str(SCRIPT),
        "--base", str(BASE_YAML),
        "--run-id", run_id,
        "--output-dir", str(out_dir),
        "--smoke",
        "--no-plot",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    run_dir = out_dir / run_id
    assert (run_dir / "best_design.yaml").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "ensemble_result.json").exists()

    from fea_solver.assembler import (
        assemble_global_force_vector, assemble_global_stiffness, build_dof_map,
    )
    from fea_solver.io_yaml import load_models_from_yaml
    from fea_solver.models import DOFType
    from fea_solver.solver import run_solve_pipeline

    model = load_models_from_yaml(run_dir / "best_design.yaml")[0]
    dof_map = build_dof_map(model)
    K = assemble_global_stiffness(model, dof_map)
    F = assemble_global_force_vector(model, dof_map)
    res = run_solve_pipeline(model, dof_map, K, F)
    tip_v = abs(float(res.displacements[dof_map.index(9, DOFType.V)]))
    assert tip_v > 0.0
    assert tip_v < 1.0  # mm
