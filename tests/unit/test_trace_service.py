"""
测试 TraceService 追踪服务层。
"""

import pytest
import json
import tempfile
from pathlib import Path

from src.observability.dashboard.services.trace_service import TraceService


@pytest.fixture
def temp_log_dir():
    """创建临时日志目录"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_traces():
    """示例追踪数据"""
    return [
        {
            "trace_id": "trace_001",
            "trace_type": "ingestion",
            "started_at": "2026-05-09T10:00:00Z",
            "finished_at": "2026-05-09T10:00:05Z",
            "total_elapsed_ms": 5000,
            "metadata": {"file": "test.pdf"},
            "stages": [
                {
                    "stage_name": "load",
                    "start_time": "2026-05-09T10:00:00Z",
                    "end_time": "2026-05-09T10:00:01Z",
                    "duration_ms": 1000,
                    "metadata": {}
                },
                {
                    "stage_name": "split",
                    "start_time": "2026-05-09T10:00:01Z",
                    "end_time": "2026-05-09T10:00:03Z",
                    "duration_ms": 2000,
                    "metadata": {"chunk_count": 10}
                }
            ]
        },
        {
            "trace_id": "trace_002",
            "trace_type": "query",
            "started_at": "2026-05-09T10:05:00Z",
            "finished_at": "2026-05-09T10:05:01Z",
            "total_elapsed_ms": 1000,
            "metadata": {"query": "test query"},
            "stages": []
        },
        {
            "trace_id": "trace_003",
            "trace_type": "ingestion",
            "started_at": "2026-05-09T10:10:00Z",
            "finished_at": "2026-05-09T10:10:03Z",
            "total_elapsed_ms": 3000,
            "metadata": {"file": "doc.pdf"},
            "stages": []
        }
    ]


@pytest.fixture
def trace_service_with_data(temp_log_dir, sample_traces):
    """创建带有测试数据的 TraceService"""
    # 写入测试数据
    traces_file = Path(temp_log_dir) / "traces.jsonl"
    with open(traces_file, "w", encoding="utf-8") as f:
        for trace in sample_traces:
            f.write(json.dumps(trace) + "\n")

    return TraceService(log_dir=temp_log_dir)


def test_list_traces_all(trace_service_with_data):
    """测试列出所有追踪"""
    traces = trace_service_with_data.list_traces()

    assert len(traces) == 3
    # 应该按时间倒序
    assert traces[0]["trace_id"] == "trace_003"
    assert traces[1]["trace_id"] == "trace_002"
    assert traces[2]["trace_id"] == "trace_001"


def test_list_traces_by_type(trace_service_with_data):
    """测试按类型过滤追踪"""
    ingestion_traces = trace_service_with_data.list_traces(trace_type="ingestion")
    query_traces = trace_service_with_data.list_traces(trace_type="query")

    assert len(ingestion_traces) == 2
    assert len(query_traces) == 1
    assert all(t["trace_type"] == "ingestion" for t in ingestion_traces)
    assert all(t["trace_type"] == "query" for t in query_traces)


def test_list_traces_with_limit(trace_service_with_data):
    """测试限制返回数量"""
    traces = trace_service_with_data.list_traces(limit=2)

    assert len(traces) == 2
    assert traces[0]["trace_id"] == "trace_003"
    assert traces[1]["trace_id"] == "trace_002"


def test_get_trace_by_id(trace_service_with_data):
    """测试根据 ID 获取追踪"""
    trace = trace_service_with_data.get_trace_by_id("trace_001")

    assert trace is not None
    assert trace["trace_id"] == "trace_001"
    assert trace["trace_type"] == "ingestion"
    assert len(trace["stages"]) == 2


def test_get_trace_by_id_not_found(trace_service_with_data):
    """测试获取不存在的追踪"""
    trace = trace_service_with_data.get_trace_by_id("nonexistent")

    assert trace is None


def test_get_ingestion_traces(trace_service_with_data):
    """测试获取 Ingestion 追踪"""
    traces = trace_service_with_data.get_ingestion_traces()

    assert len(traces) == 2
    assert all(t["trace_type"] == "ingestion" for t in traces)


def test_get_query_traces(trace_service_with_data):
    """测试获取 Query 追踪"""
    traces = trace_service_with_data.get_query_traces()

    assert len(traces) == 1
    assert traces[0]["trace_type"] == "query"


def test_search_traces_by_trace_id(trace_service_with_data):
    """测试按 trace_id 搜索"""
    results = trace_service_with_data.search_traces("trace_001")

    assert len(results) == 1
    assert results[0]["trace_id"] == "trace_001"


def test_search_traces_by_metadata(trace_service_with_data):
    """测试按 metadata 搜索"""
    results = trace_service_with_data.search_traces("test.pdf")

    assert len(results) == 1
    assert results[0]["metadata"]["file"] == "test.pdf"


def test_search_traces_no_keyword(trace_service_with_data):
    """测试无关键词搜索（返回所有）"""
    results = trace_service_with_data.search_traces("")

    assert len(results) == 3


def test_search_traces_with_type_filter(trace_service_with_data):
    """测试带类型过滤的搜索"""
    results = trace_service_with_data.search_traces("trace", trace_type="ingestion")

    assert len(results) == 2
    assert all(t["trace_type"] == "ingestion" for t in results)


def test_get_trace_stats(trace_service_with_data):
    """测试获取统计信息"""
    stats = trace_service_with_data.get_trace_stats()

    assert stats["total_traces"] == 3
    assert stats["ingestion_traces"] == 2
    assert stats["query_traces"] == 1


def test_empty_traces_file(temp_log_dir):
    """测试空追踪文件"""
    trace_service = TraceService(log_dir=temp_log_dir)
    traces = trace_service.list_traces()

    assert traces == []


def test_invalid_json_lines(temp_log_dir):
    """测试包含无效 JSON 的文件"""
    traces_file = Path(temp_log_dir) / "traces.jsonl"
    with open(traces_file, "w", encoding="utf-8") as f:
        f.write('{"valid": "json"}\n')
        f.write('invalid json line\n')
        f.write('{"another": "valid"}\n')

    trace_service = TraceService(log_dir=temp_log_dir)
    traces = trace_service.list_traces()

    # 应该跳过无效行，只返回有效的
    assert len(traces) == 2
