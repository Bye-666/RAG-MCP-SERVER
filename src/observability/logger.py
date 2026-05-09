"""
Structured logging with JSON Lines format for trace persistence.

Provides JSON-formatted logging and trace persistence to logs/traces.jsonl.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs log records as JSON."""

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record as JSON string.

        Args:
            record: Log record to format

        Returns:
            JSON string representation of the log record
        """
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)

        # Add extra fields from record
        if hasattr(record, "trace_id"):
            log_data["trace_id"] = record.trace_id
        if hasattr(record, "trace_type"):
            log_data["trace_type"] = record.trace_type

        return json.dumps(log_data, ensure_ascii=False)


def get_trace_logger(name: str = "trace", log_dir: str = "logs") -> logging.Logger:
    """
    Get a logger configured for JSON Lines output.

    Args:
        name: Logger name
        log_dir: Directory for log files

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    # Create logs directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Add file handler with JSON formatter
    file_handler = logging.FileHandler(log_path / "traces.jsonl", encoding="utf-8")
    file_handler.setFormatter(JSONFormatter())
    logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def write_trace(trace_dict: Dict[str, Any], log_dir: str = "logs") -> None:
    """
    Write a trace dictionary to logs/traces.jsonl.

    Args:
        trace_dict: Trace dictionary from TraceContext.to_dict()
        log_dir: Directory for log files
    """
    # Ensure logs directory exists
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Append trace as JSON line
    trace_file = log_path / "traces.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        json.dump(trace_dict, f, ensure_ascii=False)
        f.write("\n")
