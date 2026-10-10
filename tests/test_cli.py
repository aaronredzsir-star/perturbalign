"""Tests for perturbalign Command-Line Interface (P1).

Verify version, align, audit, cache etc. CLI          。
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import pandas as pd
import pytest

from perturbalign.cli import main


class TestCLI:
    """       cli.main Verify   row   。"""

    def test_cli_version(self, capsys):
        code = main(["version"])
        assert code == 0
        captured = capsys.readouterr()
        assert "perturbalign v" in captured.out
        assert "vocabulary SHA:" in captured.out

    def test_cli_align_csv(self, tmp_path):
        in_file = tmp_path / "test_input.csv"
        out_file = tmp_path / "test_output.csv"

        df = pd.DataFrame({
            "sample": ["A", "B", "C"],
            "perturbation": ["Trametinib 10nM", "DMSO", "sgTP53"],
        })
        df.to_csv(in_file, index=False)

        code = main([
            "align",
            str(in_file),
            "-o", str(out_file),
            "--offline",
        ])
        assert code == 0
        assert os.path.exists(out_file)

        res_df = pd.read_csv(out_file)
        assert len(res_df) == 3
        assert "perturbalign_status" in res_df.columns
        assert "perturbalign_modality" in res_df.columns
        assert res_df["perturbalign_parsed_dose"].iloc[0] == 10.0
        assert res_df["perturbalign_parsed_unit"].iloc[0] == "nM"

    def test_cli_align_with_standardize_and_explode(self, tmp_path):
        in_file = tmp_path / "test_combos.csv"
        out_file = tmp_path / "test_combos_out.csv"

        df = pd.DataFrame({
            "perturbation": ["DrugA 100nM", "DrugB 10nM + DrugC 50uM"],
        })
        df.to_csv(in_file, index=False)

        code = main([
            "align",
            str(in_file),
            "-o", str(out_file),
            "--offline",
            "--standardize-doses",
            "--explode-combos",
        ])
        assert code == 0
        res_df = pd.read_csv(out_file)
        # 1 row single agent + 1 row explode is 2 row = 3 row
        assert len(res_df) == 3
        assert "component_name" in res_df.columns

    def test_cli_audit(self, tmp_path, capsys):
        in_file = tmp_path / "test_audit.csv"
        df = pd.DataFrame({
            "perturbation": ["Trametinib 10nM", "DMSO", "sgTP53", "Hypoxia 1% O2"],
        })
        df.to_csv(in_file, index=False)

        code = main(["audit", str(in_file)])
        assert code == 0
        captured = capsys.readouterr()
        assert "perturbalign Audit Report" in captured.out
        assert "Modality Distribution" in captured.out
        assert "Status Breakdown" in captured.out

    def test_cli_cache_commands(self, tmp_path, capsys):
        # 1. cache stats
        code = main(["cache", "stats"])
        assert code == 0
        captured = capsys.readouterr()
        assert "Cache location" in captured.out
        assert "Cached entries" in captured.out

        # 2. cache export
        export_file = tmp_path / "backup_cache.sqlite"
        code_exp = main(["cache", "export", str(export_file)])
        assert code_exp == 0
        assert os.path.exists(export_file)

        # 3. cache import
        code_imp = main(["cache", "import", str(export_file)])
        assert code_imp == 0

    def test_cli_subprocesses(self, tmp_path):
        """Verify    to       row    `perturbalign.cli`   。"""
        src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{src_dir}{os.pathsep}{env.get('PYTHONPATH', '')}"
        res = subprocess.run(
            [sys.executable, "-m", "perturbalign.cli", "version"],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0
        assert "perturbalign v" in res.stdout

    def test_cli_mcp(self, monkeypatch):
        called = False

        def mock_run_stdio(self):
            nonlocal called
            called = True

        from perturbalign.mcp_server import PerturbAlignMCPServer

        monkeypatch.setattr(PerturbAlignMCPServer, "run_stdio", mock_run_stdio)
        code = main(["mcp"])
        assert code == 0
        assert called is True
