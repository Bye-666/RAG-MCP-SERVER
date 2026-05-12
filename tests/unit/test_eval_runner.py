"""
EvalRunner 单元测试
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, MagicMock
from src.observability.evaluation.eval_runner import (
    EvalRunner,
    TestCase,
    TestResult,
    EvalReport
)


class TestTestCase:
    """测试 TestCase 数据类"""

    def test_create_test_case_minimal(self):
        """测试创建最小测试用例"""
        test_case = TestCase(
            query="test query",
            expected_chunk_ids=["id1", "id2"]
        )
        assert test_case.query == "test query"
        assert test_case.expected_chunk_ids == ["id1", "id2"]
        assert test_case.expected_sources is None
        assert test_case.metadata is None

    def test_create_test_case_full(self):
        """测试创建完整测试用例"""
        test_case = TestCase(
            query="test query",
            expected_chunk_ids=["id1", "id2"],
            expected_sources=["source1.pdf"],
            metadata={"category": "test"}
        )
        assert test_case.query == "test query"
        assert test_case.expected_chunk_ids == ["id1", "id2"]
        assert test_case.expected_sources == ["source1.pdf"]
        assert test_case.metadata == {"category": "test"}


class TestTestResult:
    """测试 TestResult 数据类"""

    def test_create_test_result_success(self):
        """测试创建成功的测试结果"""
        result = TestResult(
            query="test query",
            retrieved_ids=["id1", "id2"],
            expected_ids=["id1", "id2"],
            metrics={"precision": 1.0},
            success=True
        )
        assert result.query == "test query"
        assert result.retrieved_ids == ["id1", "id2"]
        assert result.expected_ids == ["id1", "id2"]
        assert result.metrics == {"precision": 1.0}
        assert result.success is True
        assert result.error is None

    def test_create_test_result_failure(self):
        """测试创建失败的测试结果"""
        result = TestResult(
            query="test query",
            retrieved_ids=[],
            expected_ids=["id1", "id2"],
            metrics={},
            success=False,
            error="Search failed"
        )
        assert result.success is False
        assert result.error == "Search failed"


class TestEvalReport:
    """测试 EvalReport 数据类"""

    def test_create_eval_report(self):
        """测试创建评估报告"""
        result = TestResult(
            query="test",
            retrieved_ids=["id1"],
            expected_ids=["id1"],
            metrics={"precision": 1.0},
            success=True
        )
        report = EvalReport(
            total_cases=1,
            successful_cases=1,
            failed_cases=0,
            aggregate_metrics={"precision": 1.0},
            test_results=[result]
        )
        assert report.total_cases == 1
        assert report.successful_cases == 1
        assert report.failed_cases == 0

    def test_to_dict(self):
        """测试转换为字典"""
        result = TestResult(
            query="test",
            retrieved_ids=["id1"],
            expected_ids=["id1"],
            metrics={"precision": 1.0},
            success=True
        )
        report = EvalReport(
            total_cases=1,
            successful_cases=1,
            failed_cases=0,
            aggregate_metrics={"precision": 1.0},
            test_results=[result]
        )
        report_dict = report.to_dict()
        assert isinstance(report_dict, dict)
        assert report_dict["total_cases"] == 1
        assert report_dict["successful_cases"] == 1

    def test_to_json(self):
        """测试转换为 JSON"""
        result = TestResult(
            query="test",
            retrieved_ids=["id1"],
            expected_ids=["id1"],
            metrics={"precision": 1.0},
            success=True
        )
        report = EvalReport(
            total_cases=1,
            successful_cases=1,
            failed_cases=0,
            aggregate_metrics={"precision": 1.0},
            test_results=[result]
        )
        json_str = report.to_json()
        assert isinstance(json_str, str)
        parsed = json.loads(json_str)
        assert parsed["total_cases"] == 1


class TestEvalRunnerInit:
    """测试 EvalRunner 初始化"""

    def test_init_minimal(self):
        """测试最小初始化"""
        settings = {}
        hybrid_search = Mock()
        evaluator = Mock()

        runner = EvalRunner(settings, hybrid_search, evaluator)

        assert runner.settings == settings
        assert runner.hybrid_search == hybrid_search
        assert runner.evaluator == evaluator
        assert runner.trace_context is None

    def test_init_with_trace(self):
        """测试带追踪上下文的初始化"""
        settings = {}
        hybrid_search = Mock()
        evaluator = Mock()
        trace_context = Mock()

        runner = EvalRunner(settings, hybrid_search, evaluator, trace_context)

        assert runner.trace_context == trace_context


class TestEvalRunnerLoadTestSet:
    """测试 EvalRunner 加载测试集"""

    def test_load_test_set_success(self, tmp_path):
        """测试成功加载测试集"""
        # 创建测试集文件
        test_set_data = {
            "test_cases": [
                {
                    "query": "test query 1",
                    "expected_chunk_ids": ["id1", "id2"]
                },
                {
                    "query": "test query 2",
                    "expected_chunk_ids": ["id3"],
                    "expected_sources": ["source.pdf"],
                    "metadata": {"category": "test"}
                }
            ]
        }
        test_set_path = tmp_path / "test_set.json"
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f)

        # 创建 runner
        runner = EvalRunner({}, Mock(), Mock())

        # 加载测试集
        test_cases = runner.load_test_set(str(test_set_path))

        assert len(test_cases) == 2
        assert test_cases[0].query == "test query 1"
        assert test_cases[0].expected_chunk_ids == ["id1", "id2"]
        assert test_cases[1].query == "test query 2"
        assert test_cases[1].expected_sources == ["source.pdf"]
        assert test_cases[1].metadata == {"category": "test"}

    def test_load_test_set_file_not_found(self):
        """测试文件不存在"""
        runner = EvalRunner({}, Mock(), Mock())

        with pytest.raises(FileNotFoundError, match="测试集文件不存在"):
            runner.load_test_set("nonexistent.json")

    def test_load_test_set_invalid_json(self, tmp_path):
        """测试无效的 JSON 格式"""
        test_set_path = tmp_path / "invalid.json"
        with open(test_set_path, 'w', encoding='utf-8') as f:
            f.write("invalid json {")

        runner = EvalRunner({}, Mock(), Mock())

        with pytest.raises(ValueError, match="JSON 格式错误"):
            runner.load_test_set(str(test_set_path))

    def test_load_test_set_missing_test_cases(self, tmp_path):
        """测试缺少 test_cases 字段"""
        test_set_path = tmp_path / "missing_field.json"
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump({"wrong_field": []}, f)

        runner = EvalRunner({}, Mock(), Mock())

        with pytest.raises(ValueError, match="缺少 'test_cases' 字段"):
            runner.load_test_set(str(test_set_path))


class TestEvalRunnerRun:
    """测试 EvalRunner 运行评估"""

    def test_run_single_test_success(self, tmp_path):
        """测试运行单个测试用例成功"""
        # 创建测试集
        test_set_data = {
            "test_cases": [
                {
                    "query": "test query",
                    "expected_chunk_ids": ["id1", "id2"]
                }
            ]
        }
        test_set_path = tmp_path / "test_set.json"
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f)

        # Mock HybridSearch
        mock_search_result = Mock()
        mock_search_result.id = "id1"
        hybrid_search = Mock()
        hybrid_search.search.return_value = [mock_search_result]

        # Mock Evaluator
        evaluator = Mock()
        evaluator.evaluate.return_value = {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5
        }

        # 创建 runner
        settings = {"retrieval": {"top_k": 5}}
        runner = EvalRunner(settings, hybrid_search, evaluator)

        # 运行评估
        report = runner.run(str(test_set_path))

        # 验证结果
        assert report.total_cases == 1
        assert report.successful_cases == 1
        assert report.failed_cases == 0
        assert len(report.test_results) == 1
        assert report.test_results[0].success is True
        assert report.test_results[0].query == "test query"
        assert report.test_results[0].retrieved_ids == ["id1"]

        # 验证调用
        hybrid_search.search.assert_called_once()
        evaluator.evaluate.assert_called_once()

    def test_run_multiple_tests(self, tmp_path):
        """测试运行多个测试用例"""
        # 创建测试集
        test_set_data = {
            "test_cases": [
                {"query": "query 1", "expected_chunk_ids": ["id1"]},
                {"query": "query 2", "expected_chunk_ids": ["id2"]},
                {"query": "query 3", "expected_chunk_ids": ["id3"]}
            ]
        }
        test_set_path = tmp_path / "test_set.json"
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f)

        # Mock
        mock_result = Mock()
        mock_result.id = "id1"
        hybrid_search = Mock()
        hybrid_search.search.return_value = [mock_result]

        evaluator = Mock()
        evaluator.evaluate.return_value = {"precision": 1.0}

        # 运行
        runner = EvalRunner({}, hybrid_search, evaluator)
        report = runner.run(str(test_set_path))

        # 验证
        assert report.total_cases == 3
        assert report.successful_cases == 3
        assert len(report.test_results) == 3
        assert hybrid_search.search.call_count == 3
        assert evaluator.evaluate.call_count == 3

    def test_run_with_custom_top_k(self, tmp_path):
        """测试使用自定义 top_k"""
        test_set_data = {
            "test_cases": [
                {"query": "test", "expected_chunk_ids": ["id1"]}
            ]
        }
        test_set_path = tmp_path / "test_set.json"
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f)

        mock_result = Mock()
        mock_result.id = "id1"
        hybrid_search = Mock()
        hybrid_search.search.return_value = [mock_result]

        evaluator = Mock()
        evaluator.evaluate.return_value = {}

        runner = EvalRunner({}, hybrid_search, evaluator)
        runner.run(str(test_set_path), top_k=10)

        # 验证 top_k 参数
        call_args = hybrid_search.search.call_args
        assert call_args[1]["top_k"] == 10

    def test_run_with_search_failure(self, tmp_path):
        """测试检索失败的情况"""
        test_set_data = {
            "test_cases": [
                {"query": "test", "expected_chunk_ids": ["id1"]}
            ]
        }
        test_set_path = tmp_path / "test_set.json"
        with open(test_set_path, 'w', encoding='utf-8') as f:
            json.dump(test_set_data, f)

        # Mock 检索失败
        hybrid_search = Mock()
        hybrid_search.search.side_effect = Exception("Search failed")

        evaluator = Mock()

        runner = EvalRunner({}, hybrid_search, evaluator)
        report = runner.run(str(test_set_path))

        # 验证失败被正确处理
        assert report.total_cases == 1
        assert report.successful_cases == 0
        assert report.failed_cases == 1
        assert report.test_results[0].success is False
        assert "Search failed" in report.test_results[0].error


class TestEvalRunnerAggregateMetrics:
    """测试聚合指标计算"""

    def test_compute_aggregate_metrics_empty(self):
        """测试空结果列表"""
        runner = EvalRunner({}, Mock(), Mock())
        metrics = runner._compute_aggregate_metrics([])
        assert metrics == {}

    def test_compute_aggregate_metrics_all_failed(self):
        """测试所有测试都失败"""
        results = [
            TestResult("q1", [], ["id1"], {}, False, "error"),
            TestResult("q2", [], ["id2"], {}, False, "error")
        ]
        runner = EvalRunner({}, Mock(), Mock())
        metrics = runner._compute_aggregate_metrics(results)
        assert metrics == {}

    def test_compute_aggregate_metrics_average(self):
        """测试计算平均值"""
        results = [
            TestResult("q1", ["id1"], ["id1"], {"precision": 1.0, "recall": 0.8}, True),
            TestResult("q2", ["id2"], ["id2"], {"precision": 0.6, "recall": 1.0}, True)
        ]
        runner = EvalRunner({}, Mock(), Mock())
        metrics = runner._compute_aggregate_metrics(results)

        assert "precision" in metrics
        assert "recall" in metrics
        assert metrics["precision"] == pytest.approx(0.8)  # (1.0 + 0.6) / 2
        assert metrics["recall"] == pytest.approx(0.9)  # (0.8 + 1.0) / 2

    def test_compute_aggregate_metrics_mrr(self):
        """测试 MRR 计算"""
        results = [
            # 第一个相关文档在位置 1
            TestResult("q1", ["id1", "id2"], ["id1"], {}, True),
            # 第一个相关文档在位置 2
            TestResult("q2", ["id3", "id4"], ["id4"], {}, True),
            # 没有相关文档
            TestResult("q3", ["id5", "id6"], ["id7"], {}, True)
        ]
        runner = EvalRunner({}, Mock(), Mock())
        metrics = runner._compute_aggregate_metrics(results)

        # MRR = (1/1 + 1/2 + 0) / 3 = 0.5
        assert "mrr" in metrics
        assert metrics["mrr"] == pytest.approx(0.5)

    def test_compute_aggregate_metrics_mixed_success(self):
        """测试混合成功和失败的情况"""
        results = [
            TestResult("q1", ["id1"], ["id1"], {"precision": 1.0}, True),
            TestResult("q2", [], [], {}, False, "error"),
            TestResult("q3", ["id2"], ["id2"], {"precision": 0.8}, True)
        ]
        runner = EvalRunner({}, Mock(), Mock())
        metrics = runner._compute_aggregate_metrics(results)

        # 只计算成功的测试
        assert metrics["precision"] == pytest.approx(0.9)  # (1.0 + 0.8) / 2
