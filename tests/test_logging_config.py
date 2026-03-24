"""Tests for logging_config module."""
from __future__ import annotations

import json
import logging

from backtest_audit.logging_config import configure_logging, get_logger


class TestConfigureLogging:
    def test_sets_level(self):
        configure_logging("WARNING")
        assert logging.getLogger().level == logging.WARNING
        configure_logging("INFO")

    def test_json_output(self, capsys):
        configure_logging("DEBUG")
        get_logger("test").info("hello world")
        line = capsys.readouterr().out.strip().splitlines()[0]
        data = json.loads(line)
        assert data["msg"] == "hello world"
        assert data["level"] == "INFO"
        assert "ts" in data


class TestGetLogger:
    def test_returns_logger(self):
        assert isinstance(get_logger("x"), logging.Logger)

    def test_name_prefixed(self):
        assert get_logger("mymod").name == "backtest_audit.mymod"
