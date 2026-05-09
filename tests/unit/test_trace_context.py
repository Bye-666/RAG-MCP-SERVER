"""Tests for TraceContext and TraceCollector."""

import pytest
import json
import time
from src.core.trace.trace_context import TraceContext, StageRecord
from src.core.trace.trace_collector import TraceCollector


class TestTraceContext:
    """Test TraceContext functionality."""

    def test_init_default_trace_type(self):
        """Test default trace type is 'query'."""
        trace = TraceContext()

        assert trace.trace_type == "query"
        assert trace.trace_id is not None
        assert len(trace.stages) == 0

    def test_init_with_trace_type(self):
        """Test initialization with custom trace type."""
        trace = TraceContext(trace_type="ingestion")

        assert trace.trace_type == "ingestion"

    def test_init_with_trace_id(self):
        """Test initialization with custom trace ID."""
        custom_id = "test-trace-123"
        trace = TraceContext(trace_id=custom_id)

        assert trace.trace_id == custom_id

    def test_record_stage(self):
        """Test recording a stage."""
        trace = TraceContext()
        stage = trace.record_stage("test_stage", {"key": "value"})

        assert isinstance(stage, StageRecord)
        assert stage.stage_name == "test_stage"
        assert stage.metadata["key"] == "value"
        assert len(trace.stages) == 1

    def test_finish_stage(self):
        """Test finishing a stage."""
        trace = TraceContext()
        stage = trace.record_stage("test_stage")

        time.sleep(0.01)  # Small delay
        trace.finish_stage(stage, {"result": "success"})

        assert stage.end_time is not None
        assert stage.duration_ms is not None
        assert stage.duration_ms > 0
        assert stage.metadata["result"] == "success"

    def test_finish_trace(self):
        """Test finishing entire trace."""
        trace = TraceContext()
        stage = trace.record_stage("stage1")
        trace.finish_stage(stage)

        time.sleep(0.01)
        trace.finish({"status": "completed"})

        assert trace.end_time is not None
        assert trace.total_duration_ms is not None
        assert trace.total_duration_ms > 0
        assert trace.metadata["status"] == "completed"

    def test_elapsed_ms_total(self):
        """Test elapsed_ms without stage name returns total elapsed."""
        trace = TraceContext()
        time.sleep(0.01)

        elapsed = trace.elapsed_ms()

        assert elapsed > 0
        assert elapsed < 1000  # Should be less than 1 second

    def test_elapsed_ms_with_stage_name(self):
        """Test elapsed_ms with stage name returns stage duration."""
        trace = TraceContext()
        stage = trace.record_stage("test_stage")
        time.sleep(0.01)
        trace.finish_stage(stage)

        elapsed = trace.elapsed_ms("test_stage")

        assert elapsed > 0
        assert elapsed == stage.duration_ms

    def test_elapsed_ms_stage_not_found(self):
        """Test elapsed_ms returns 0 for non-existent stage."""
        trace = TraceContext()

        elapsed = trace.elapsed_ms("nonexistent_stage")

        assert elapsed == 0.0

    def test_elapsed_ms_unfinished_stage(self):
        """Test elapsed_ms for unfinished stage returns current elapsed."""
        trace = TraceContext()
        stage = trace.record_stage("running_stage")
        time.sleep(0.01)

        elapsed = trace.elapsed_ms("running_stage")

        assert elapsed > 0
        assert stage.end_time is None  # Stage not finished

    def test_to_dict_structure(self):
        """Test to_dict returns correct structure."""
        trace = TraceContext(trace_type="query")
        stage = trace.record_stage("stage1", {"input": "test"})
        time.sleep(0.01)  # Ensure measurable duration
        trace.finish_stage(stage, {"output": "result"})
        trace.finish()

        trace_dict = trace.to_dict()

        # Check required fields
        assert "trace_id" in trace_dict
        assert "trace_type" in trace_dict
        assert "started_at" in trace_dict
        assert "finished_at" in trace_dict
        assert "total_elapsed_ms" in trace_dict
        assert "stages" in trace_dict

        # Check values
        assert trace_dict["trace_type"] == "query"
        assert trace_dict["finished_at"] is not None
        assert trace_dict["total_elapsed_ms"] > 0
        assert len(trace_dict["stages"]) == 1

    def test_to_dict_json_serializable(self):
        """Test to_dict output is JSON serializable."""
        trace = TraceContext(trace_type="ingestion")
        stage = trace.record_stage("stage1")
        trace.finish_stage(stage)
        trace.finish()

        trace_dict = trace.to_dict()

        # Should not raise
        json_str = json.dumps(trace_dict)
        assert len(json_str) > 0

        # Should be able to parse back
        parsed = json.loads(json_str)
        assert parsed["trace_type"] == "ingestion"

    def test_to_dict_unfinished_trace(self):
        """Test to_dict with unfinished trace."""
        trace = TraceContext()
        stage = trace.record_stage("stage1")

        trace_dict = trace.to_dict()

        assert trace_dict["finished_at"] is None
        assert trace_dict["total_elapsed_ms"] is None
        assert trace_dict["stages"][0]["end_time"] is None

    def test_multiple_stages(self):
        """Test trace with multiple stages."""
        trace = TraceContext()

        stage1 = trace.record_stage("stage1")
        time.sleep(0.01)
        trace.finish_stage(stage1)

        stage2 = trace.record_stage("stage2")
        time.sleep(0.01)
        trace.finish_stage(stage2)

        trace.finish()

        assert len(trace.stages) == 2
        assert trace.elapsed_ms("stage1") > 0
        assert trace.elapsed_ms("stage2") > 0

        trace_dict = trace.to_dict()
        assert len(trace_dict["stages"]) == 2

    def test_stage_metadata_update(self):
        """Test updating stage metadata on finish."""
        trace = TraceContext()
        stage = trace.record_stage("stage1", {"initial": "data"})

        trace.finish_stage(stage, {"additional": "info"})

        assert stage.metadata["initial"] == "data"
        assert stage.metadata["additional"] == "info"

    def test_trace_metadata_update(self):
        """Test updating trace metadata on finish."""
        trace = TraceContext()
        trace.metadata["early"] = "value"

        trace.finish({"late": "value"})

        assert trace.metadata["early"] == "value"
        assert trace.metadata["late"] == "value"


class TestTraceCollector:
    """Test TraceCollector functionality."""

    def test_collect_trace(self):
        """Test collecting a trace."""
        collector = TraceCollector()
        trace = TraceContext(trace_type="query")
        trace.finish()

        collector.collect(trace)

        traces = collector.get_traces()
        assert len(traces) == 1
        assert traces[0]["trace_type"] == "query"

    def test_collect_multiple_traces(self):
        """Test collecting multiple traces."""
        collector = TraceCollector()

        trace1 = TraceContext(trace_type="query")
        trace1.finish()
        collector.collect(trace1)

        trace2 = TraceContext(trace_type="ingestion")
        trace2.finish()
        collector.collect(trace2)

        traces = collector.get_traces()
        assert len(traces) == 2
        assert traces[0]["trace_type"] == "query"
        assert traces[1]["trace_type"] == "ingestion"

    def test_get_traces(self):
        """Test getting collected traces."""
        collector = TraceCollector()
        trace = TraceContext()
        trace.finish()

        collector.collect(trace)
        traces = collector.get_traces()

        assert isinstance(traces, list)
        assert len(traces) == 1
        assert "trace_id" in traces[0]

    def test_clear_traces(self):
        """Test clearing collected traces."""
        collector = TraceCollector()
        trace = TraceContext()
        trace.finish()

        collector.collect(trace)
        assert len(collector.get_traces()) == 1

        collector.clear()
        assert len(collector.get_traces()) == 0

    def test_collect_stores_dict(self):
        """Test that collect stores trace as dictionary."""
        collector = TraceCollector()
        trace = TraceContext(trace_type="query")
        stage = trace.record_stage("test_stage")
        trace.finish_stage(stage)
        trace.finish()

        collector.collect(trace)
        traces = collector.get_traces()

        assert isinstance(traces[0], dict)
        assert traces[0]["trace_type"] == "query"
        assert "stages" in traces[0]
        assert len(traces[0]["stages"]) == 1

    def test_collected_trace_is_serializable(self):
        """Test collected traces are JSON serializable."""
        collector = TraceCollector()
        trace = TraceContext(trace_type="ingestion")
        trace.finish()

        collector.collect(trace)
        traces = collector.get_traces()

        # Should not raise
        json_str = json.dumps(traces)
        assert len(json_str) > 0


class TestStageRecord:
    """Test StageRecord functionality."""

    def test_duration_ms_unfinished(self):
        """Test duration_ms returns None for unfinished stage."""
        stage = StageRecord(stage_name="test", start_time=TraceContext().start_time)

        assert stage.duration_ms is None

    def test_duration_ms_finished(self):
        """Test duration_ms calculates correctly for finished stage."""
        trace = TraceContext()
        stage = trace.record_stage("test")
        time.sleep(0.01)
        trace.finish_stage(stage)

        assert stage.duration_ms is not None
        assert stage.duration_ms > 0
