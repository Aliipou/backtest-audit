"""Tests for backup utilities."""
from __future__ import annotations

import gzip
import json

import pytest

from backtest_audit.auditor import AuditReport
from backtest_audit.backup import export_csv, export_json, rotate_backups, save_snapshot


@pytest.fixture
def report() -> AuditReport:
    return AuditReport(
        dsr_result={
            "dsr": 1.5, "pvalue": 0.05, "obs_sharpe": 0.8,
            "benchmark_sharpe": 0.2, "verdict": "PASS",
        },
        monte_carlo_result={
            "sharpe": 0.8, "percentile": 0.97, "pvalue": 0.03, "verdict": "PASS",
        },
    )


class TestExportJson:
    def test_creates_file(self, tmp_path, report):
        dest = export_json(report, tmp_path / "r.json")
        assert dest.exists()

    def test_valid_json(self, tmp_path, report):
        dest = export_json(report, tmp_path / "r.json")
        data = json.loads(dest.read_text())
        assert "overall_verdict" in data

    def test_compressed(self, tmp_path, report):
        dest = export_json(report, tmp_path / "r.json.gz", compress=True)
        with gzip.open(dest, "rt") as fh:
            data = json.load(fh)
        assert "overall_verdict" in data

    def test_creates_parent_dirs(self, tmp_path, report):
        dest = export_json(report, tmp_path / "sub" / "dir" / "r.json")
        assert dest.exists()


class TestExportCsv:
    def test_creates_file(self, tmp_path, report):
        dest = export_csv(report, tmp_path / "r.csv")
        assert dest.exists()

    def test_header_and_rows(self, tmp_path, report):
        dest = export_csv(report, tmp_path / "r.csv")
        lines = dest.read_text().splitlines()
        assert lines[0] == "test,metric,value,verdict"
        assert len(lines) > 1


class TestSaveSnapshot:
    def test_creates_snapshot(self, tmp_path, report):
        path = save_snapshot(report, strategy_id="my_strat", backup_dir=tmp_path, compress=False)
        assert path.exists()
        assert "my_strat" in path.name

    def test_compressed_snapshot(self, tmp_path, report):
        path = save_snapshot(report, backup_dir=tmp_path, compress=True)
        assert path.suffix == ".gz"


class TestRotateBackups:
    def test_keeps_last_n(self, tmp_path, report):
        for i in range(5):
            save_snapshot(report, strategy_id=f"s{i}", backup_dir=tmp_path, compress=False)
        deleted = rotate_backups(tmp_path, keep_last=3)
        assert len(list(tmp_path.iterdir())) == 3
        assert len(deleted) == 2

    def test_nonexistent_dir_ok(self, tmp_path):
        assert rotate_backups(tmp_path / "nope", keep_last=5) == []
