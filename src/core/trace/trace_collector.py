"""
TraceCollector: Collects and persists trace data.

Responsible for collecting finished traces and triggering persistence.
"""

from typing import Optional
from .trace_context import TraceContext


class TraceCollector:
    """
    Collects trace data and triggers persistence.

    This is a simple collector that can be extended in F2 to write
    traces to JSON Lines files.
    """

    def __init__(self):
        """Initialize trace collector."""
        self.collected_traces = []

    def collect(self, trace: TraceContext) -> None:
        """
        Collect a trace for persistence.

        Args:
            trace: TraceContext instance to collect

        Note:
            In F2, this will trigger writing to logs/traces.jsonl
        """
        # Store trace data
        trace_dict = trace.to_dict()
        self.collected_traces.append(trace_dict)

        # TODO: In F2, write to JSON Lines file
        # For now, just collect in memory

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
