"""
Backup and export utilities for backtest-audit.

Helpers to:
  - Export AuditReport to JSON / CSV
  - Save timestamped audit snapshots to a local backup directory
  - Rotate old backups (keep last N)
"""
from __future__ import annotations

import csv
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .auditor import AuditReport

from .logging_config import get_logger

logger = get_logger("backup")

DEFAULT_BACKUP_DIR = Path.home() / ".backtest_audit" / "backups"
DEFAULT_KEEP_LAST = 30


def export_json(report: AuditReport, path: Path | str, compress: bool = False) -> Path:
    """Write *report* to a JSON file. Returns the resolved Path."""
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report.to_dict(), indent=2)

    if compress:
        with gzip.open(dest, "wt", encoding="utf-8") as fh:
            fh.write(payload)
    else:
        dest.write_text(payload, encoding="utf-8")

    logger.info("exported audit report", extra={"extra": {"path": str(dest)}})
    return dest


def export_csv(report: AuditReport, path: Path | str) -> Path:
    """Write a flat CSV of scalar audit metrics. Columns: test, metric, value, verdict."""
    dest = Path(path).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for test_name, result in {
        "dsr": report.dsr_result,
        "monte_carlo": report.monte_carlo_result,
        "pbo": report.pbo_result,
        "sensitivity": report.sensitivity_result,
    }.items():
        if not result:
            continue
        verdict = result.get("verdict", "")
        for k, v in result.items():
            if k == "verdict":
                continue
            rows.append({"test": test_name, "metric": k, "value": str(v), "verdict": verdict})

    with dest.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test", "metric", "value", "verdict"])
        writer.writeheader()
        writer.writerows(rows)

    logger.info("exported audit report to CSV", extra={"extra": {"path": str(dest)}})
    return dest


def save_snapshot(
    report: AuditReport,
    strategy_id: str = "unnamed",
    backup_dir: Path | str = DEFAULT_BACKUP_DIR,
    compress: bool = True,
) -> Path:
    """Save a timestamped JSON snapshot of *report* to *backup_dir*."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = ".json.gz" if compress else ".json"
    dest = Path(backup_dir) / f"{strategy_id}_{ts}{suffix}"
    return export_json(report, dest, compress=compress)


def rotate_backups(
    backup_dir: Path | str = DEFAULT_BACKUP_DIR,
    keep_last: int = DEFAULT_KEEP_LAST,
) -> list[Path]:
    """Delete old backups keeping only the *keep_last* most-recent. Returns deleted paths."""
    d = Path(backup_dir)
    if not d.exists():
        return []

    files = sorted(
        [f for f in d.iterdir() if f.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    to_delete = files[keep_last:]
    for f in to_delete:
        f.unlink()
        logger.info("rotated backup", extra={"extra": {"deleted": str(f)}})
    return to_delete
