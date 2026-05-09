"""
TraceContext: Minimal implementation for tracking pipeline stages.

This is a placeholder implementation for Phase C. Will be enhanced in Phase F
with structured logging, persistence, and detailed metrics.
"""

import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class StageRecord:
    """Record of a single pipeline stage execution"""
    stage_name: str
    start_time: datetime
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> Optional[float]:
        """Calculate stage duration in milliseconds"""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000


class TraceContext:
    """
    Minimal trace context for tracking pipeline execution.

    Provides:
    - Unique trace_id generation
    - Stage recording with timing
    - Basic metadata storage

    Phase F will add:
    - JSON Lines persistence
    - Structured logging
    - Detailed metrics and waterfall visualization
    """

    def __init__(self, trace_id: Optional[str] = None, trace_type: str = "query"):
        """
        Initialize trace context.

        Args:
            trace_id: Optional trace ID. If not provided, generates a new UUID.
            trace_type: Type of trace - "query" or "ingestion"
        """
        self.trace_id = trace_id or str(uuid.uuid4())
        self.trace_type = trace_type
        self.stages: List[StageRecord] = []
        self.metadata: Dict[str, Any] = {}
        self.start_time = datetime.now()
        self.end_time: Optional[datetime] = None

    def record_stage(self, stage_name: str, metadata: Optional[Dict[str, Any]] = None) -> StageRecord:
        """
        Record the start of a pipeline stage.

        Args:
            stage_name: Name of the stage (e.g., "chunk_refiner", "dense_encoder")
            metadata: Optional metadata for this stage

        Returns:
            StageRecord object that can be used to mark completion
        """
        record = StageRecord(
            stage_name=stage_name,
            start_time=datetime.now(),
            metadata=metadata or {}
        )
        self.stages.append(record)
        return record

    def finish_stage(self, record: StageRecord, metadata: Optional[Dict[str, Any]] = None):
        """
        Mark a stage as finished.

        Args:
            record: The StageRecord returned by record_stage
            metadata: Optional additional metadata to merge
        """
        record.end_time = datetime.now()
        if metadata:
            record.metadata.update(metadata)

    def finish(self, metadata: Optional[Dict[str, Any]] = None):
        """
        Mark the entire trace as finished.

        Args:
            metadata: Optional metadata for the entire trace
        """
        self.end_time = datetime.now()
        if metadata:
            self.metadata.update(metadata)

    @property
    def total_duration_ms(self) -> Optional[float]:
        """Calculate total trace duration in milliseconds"""
        if self.end_time is None:
            return None
        delta = self.end_time - self.start_time
        return delta.total_seconds() * 1000

    def elapsed_ms(self, stage_name: Optional[str] = None) -> float:
        """
        Get elapsed time in milliseconds.

        Args:
            stage_name: Optional stage name. If provided, returns duration of that stage.
                       If None, returns total elapsed time.

        Returns:
            Elapsed time in milliseconds, or 0 if stage not found or not finished
        """
        if stage_name is None:
            # Return total elapsed time
            if self.end_time:
                delta = self.end_time - self.start_time
            else:
                delta = datetime.now() - self.start_time
            return delta.total_seconds() * 1000

        # Find stage and return its duration
        for stage in self.stages:
            if stage.stage_name == stage_name:
                if stage.duration_ms is not None:
                    return stage.duration_ms
                # Stage not finished yet, calculate current elapsed
                delta = datetime.now() - stage.start_time
                return delta.total_seconds() * 1000

        return 0.0  # Stage not found

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trace to dictionary for serialization.

        Returns:
            Dictionary representation of the trace with trace_type, timestamps, and stages
        """
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "started_at": self.start_time.isoformat(),
            "finished_at": self.end_time.isoformat() if self.end_time else None,
            "total_elapsed_ms": self.total_duration_ms,
            "metadata": self.metadata,
            "stages": [
                {
                    "stage_name": stage.stage_name,
                    "start_time": stage.start_time.isoformat(),
                    "end_time": stage.end_time.isoformat() if stage.end_time else None,
                    "duration_ms": stage.duration_ms,
                    "metadata": stage.metadata
                }
                for stage in self.stages
            ]
        }
