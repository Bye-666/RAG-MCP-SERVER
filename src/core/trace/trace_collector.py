"""
TraceCollector: Collects and persists trace data.

Responsible for collecting finished traces and triggering persistence.
"""

from typing import Optional
from .trace_context import TraceContext
from src.observability.logger import write_trace


class TraceCollector:
    """
    Collects trace data and triggers persistence to JSON Lines.
    """

    def __init__(self, log_dir: str = "logs"):
        """
        Initialize trace collector.

        Args:
            log_dir: Directory to write trace logs (default: "logs")
        """
        self.collected_traces = []
        self.log_dir = log_dir

    def collect(self, trace: TraceContext) -> None:
        """
        Collect a trace and persist to logs/traces.jsonl.

        Args:
            trace: TraceContext instance to collect
        """
        # Store trace data
        trace_dict = trace.to_dict()
        self.collected_traces.append(trace_dict)

        # Persist to JSON Lines file
        write_trace(trace_dict, self.log_dir)

    def get_traces(self):
        """
        Get all collected traces.

        Returns:
            List of trace dictionaries
        """
        return self.collected_traces

    def clear(self):
        """Clear all collected traces."""
        self.collected_traces.clear()
