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

    def __init__(self, trace_id: Optional[str] = None):
        """
        Initialize trace context.

        Args:
            trace_id: Optional trace ID. If not provided, generates a new UUID.
        """
        self.trace_id = trace_id or str(uuid.uuid4())
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

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert trace to dictionary for serialization.

        Returns:
            Dictionary representation of the trace
        """
        return {
            "trace_id": self.trace_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_duration_ms": self.total_duration_ms,
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
