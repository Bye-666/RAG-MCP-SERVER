"""
评估运行脚本

使用方法:
    python scripts/evaluate.py
    python scripts/evaluate.py --test-set tests/fixtures/golden_test_set.json
    python scripts/evaluate.py --evaluator ragas --top-k 10
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.settings import Settings
from src.retrieval.hybrid_search import HybridSearch
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner
from src.core.trace import TraceContext


def main():
    parser = argparse.ArgumentParser(description="运行 RAG 系统评估")
    parser.add_argument(
        "--test-set",
        type=str,
        default="tests/fixtures/golden_test_set.json",
        help="黄金测试集路径"
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="basic",
        choices=["basic", "ragas", "composite"],
        help="评估器类型"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="检索返回的文档数量（覆盖配置）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出报告文件路径（JSON 格式）"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("RAG 系统评估")
    print("=" * 60)
    print(f"测试集: {args.test_set}")
    print(f"评估器: {args.evaluator}")
    if args.top_k:
        print(f"Top-K: {args.top_k}")
    print()

    # 加载配置
    try:
        settings = Settings()
        print("✓ 配置加载成功")
    except Exception as e:
        print(f"✗ 配置加载失败: {str(e)}")
        return 1

    # 初始化 HybridSearch
    try:
        hybrid_search = HybridSearch(settings)
        print("✓ HybridSearch 初始化成功")
    except Exception as e:
        print(f"✗ HybridSearch 初始化失败: {str(e)}")
        return 1

    # 初始化评估器
    try:
        evaluator = EvaluatorFactory.create(args.evaluator)
        print(f"✓ 评估器 '{args.evaluator}' 初始化成功")
    except Exception as e:
        print(f"✗ 评估器初始化失败: {str(e)}")
        return 1

    # 创建追踪上下文（如果需要详细日志）
    trace_context = TraceContext() if args.verbose else None

    # 创建 EvalRunner
    try:
        eval_runner = EvalRunner(
            settings=settings.model_dump(),
            hybrid_search=hybrid_search,
            evaluator=evaluator,
            trace_context=trace_context
        )
        print("✓ EvalRunner 初始化成功")
    except Exception as e:
        print(f"✗ EvalRunner 初始化失败: {str(e)}")
        return 1

    print()
    print("开始运行评估...")
    print("-" * 60)

    # 运行评估
    try:
        report = eval_runner.run(args.test_set, top_k=args.top_k)
    except Exception as e:
        print(f"✗ 评估运行失败: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

    # 输出结果
    print()
    print("=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总测试用例数: {report.total_cases}")
    print(f"成功: {report.successful_cases}")
    print(f"失败: {report.failed_cases}")
    print()

    if report.aggregate_metrics:
        print("聚合指标:")
        print("-" * 60)
        for metric_name, value in sorted(report.aggregate_metrics.items()):
            print(f"  {metric_name:30s}: {value:.4f}")
        print()

    if args.verbose and report.test_results:
        print("详细结果:")
        print("-" * 60)
        for idx, result in enumerate(report.test_results, start=1):
            print(f"\n测试用例 {idx}: {result.query}")
            if result.success:
                print(f"  状态: ✓ 成功")
                print(f"  检索到的文档: {len(result.retrieved_ids)} 个")
                print(f"  期望的文档: {len(result.expected_ids)} 个")
                if result.metrics:
                    print("  指标:")
                    for metric_name, value in sorted(result.metrics.items()):
                        print(f"    {metric_name:28s}: {value:.4f}")
            else:
                print(f"  状态: ✗ 失败")
                print(f"  错误: {result.error}")

    # 保存报告到文件
    if args.output:
        try:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report.to_json())
            print(f"\n✓ 报告已保存到: {args.output}")
        except Exception as e:
            print(f"\n✗ 保存报告失败: {str(e)}")
            return 1

    print()
    print("=" * 60)

    # 返回状态码（如果有失败的测试用例则返回 1）
    return 1 if report.failed_cases > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
