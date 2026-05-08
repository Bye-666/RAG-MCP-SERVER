import uuid
from typing import Any, Dict, List, Optional


class TraceContext:
    """Minimal trace context for tracking operation stages"""

    def __init__(self, trace_type: str = "query"):
        self.trace_id = str(uuid.uuid4())
        self.trace_type = trace_type
        self.stages: List[Dict[str, Any]] = []

    def record_stage(self, stage_name: str, **kwargs):
        self.stages.append({"name": stage_name, **kwargs})

    def finish(self):
        pass

    def elapsed_ms(self, stage_name=None):
        return 0.0

    def to_dict(self):
        return {
            "trace_id": self.trace_id,
            "trace_type": self.trace_type,
            "stages": self.stages,
        }
