"""
Unit tests for structured logging with JSON Lines format.
"""

import json
import logging
from pathlib import Path
import pytest
from src.observability.logger import JSONFormatter, get_trace_logger, write_trace


class TestJSONFormatter:
    """Test JSONFormatter class."""

    def test_format_basic_record(self):
        """Test formatting a basic log record."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "timestamp" in data
        assert data["level"] == "INFO"
        assert data["logger"] == "test"
        assert data["message"] == "Test message"

    def test_format_with_trace_fields(self):
        """Test formatting with trace-specific fields."""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="trace",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Trace event",
            args=(),
            exc_info=None,
        )
        record.trace_id = "trace-123"
        record.trace_type = "query"

        result = formatter.format(record)
        data = json.loads(result)

        assert data["trace_id"] == "trace-123"
        assert data["trace_type"] == "query"

    def test_format_with_exception(self):
        """Test formatting with exception info."""
        formatter = JSONFormatter()
        try:
            raise ValueError("Test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        result = formatter.format(record)
        data = json.loads(result)

        assert "exception" in data
        assert "ValueError: Test error" in data["exception"]


class TestGetTraceLogger:
    """Test get_trace_logger function."""

    def test_create_logger(self, tmp_path):
        """Test creating a trace logger."""
        logger = get_trace_logger("test_logger", str(tmp_path))

        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1
        assert isinstance(logger.handlers[0], logging.FileHandler)
        assert not logger.propagate

    def test_logger_creates_directory(self, tmp_path):
        """Test that logger creates log directory."""
        log_dir = tmp_path / "nested" / "logs"
        get_trace_logger("test", str(log_dir))

        assert log_dir.exists()
        assert (log_dir / "traces.jsonl").exists()

    def test_logger_reuse(self, tmp_path):
        """Test that calling get_trace_logger twice returns same logger."""
        logger1 = get_trace_logger("reuse_test", str(tmp_path))
        logger2 = get_trace_logger("reuse_test", str(tmp_path))

        assert logger1 is logger2
        assert len(logger1.handlers) == 1  # Should not add duplicate handlers

    def test_logger_writes_json(self, tmp_path):
        """Test that logger writes JSON formatted output."""
        logger = get_trace_logger("json_test", str(tmp_path))
        logger.info("Test message")

        trace_file = tmp_path / "traces.jsonl"
        assert trace_file.exists()

        with open(trace_file, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)

        assert data["message"] == "Test message"
        assert data["level"] == "INFO"


class TestWriteTrace:
    """Test write_trace function."""

    def test_write_single_trace(self, tmp_path):
        """Test writing a single trace."""
        trace_dict = {
            "trace_id": "trace-001",
            "trace_type": "query",
            "query": "test query",
            "start_time": "2024-01-01T00:00:00",
            "end_time": "2024-01-01T00:00:01",
            "total_elapsed_ms": 1000,
        }

        write_trace(trace_dict, str(tmp_path))

        trace_file = tmp_path / "traces.jsonl"
        assert trace_file.exists()

        with open(trace_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["trace_id"] == "trace-001"
        assert data["trace_type"] == "query"
        assert data["total_elapsed_ms"] == 1000

    def test_write_multiple_traces(self, tmp_path):
        """Test writing multiple traces."""
        traces = [
            {"trace_id": f"trace-{i}", "trace_type": "query"}
            for i in range(3)
        ]

        for trace in traces:
            write_trace(trace, str(tmp_path))

        trace_file = tmp_path / "traces.jsonl"
        with open(trace_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 3
        for i, line in enumerate(lines):
            data = json.loads(line)
            assert data["trace_id"] == f"trace-{i}"

    def test_write_creates_directory(self, tmp_path):
        """Test that write_trace creates log directory."""
        log_dir = tmp_path / "new_logs"
        trace_dict = {"trace_id": "test", "trace_type": "ingestion"}

        write_trace(trace_dict, str(log_dir))

        assert log_dir.exists()
        assert (log_dir / "traces.jsonl").exists()

    def test_write_appends_to_existing_file(self, tmp_path):
        """Test that write_trace appends to existing file."""
        trace1 = {"trace_id": "trace-1", "trace_type": "query"}
        trace2 = {"trace_id": "trace-2", "trace_type": "ingestion"}

        write_trace(trace1, str(tmp_path))
        write_trace(trace2, str(tmp_path))

        trace_file = tmp_path / "traces.jsonl"
        with open(trace_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 2
        assert json.loads(lines[0])["trace_id"] == "trace-1"
        assert json.loads(lines[1])["trace_id"] == "trace-2"

    def test_write_unicode_content(self, tmp_path):
        """Test writing trace with Unicode content."""
        trace_dict = {
            "trace_id": "trace-unicode",
            "trace_type": "query",
            "query": "测试查询 🔍",
            "metadata": {"author": "张三"},
        }

        write_trace(trace_dict, str(tmp_path))

        trace_file = tmp_path / "traces.jsonl"
        with open(trace_file, "r", encoding="utf-8") as f:
            line = f.readline()
            data = json.loads(line)

        assert data["query"] == "测试查询 🔍"
        assert data["metadata"]["author"] == "张三"

    def test_acceptance_criteria(self, tmp_path):
        """
        Acceptance test: Write a trace and verify file has one valid JSON line
        with trace_type field.
        """
        trace_dict = {
            "trace_id": "acceptance-test",
            "trace_type": "query",
            "query": "acceptance test query",
            "total_elapsed_ms": 500,
        }

        # Write trace
        write_trace(trace_dict, str(tmp_path))

        # Verify file exists and has one line
        trace_file = tmp_path / "traces.jsonl"
        assert trace_file.exists()

        with open(trace_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == 1

        # Verify line is valid JSON
        data = json.loads(lines[0])

        # Verify trace_type field exists
        assert "trace_type" in data
        assert data["trace_type"] == "query"
