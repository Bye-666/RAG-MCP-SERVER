"""
评估面板页面

功能：
- 选择评估后端（Ragas、Custom、Composite）
- 选择黄金测试集
- 运行评估并展示结果
- 显示聚合指标和详细结果
"""

import streamlit as st
from pathlib import Path
import json
from datetime import datetime

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch
from src.libs.evaluator.evaluator_factory import EvaluatorFactory
from src.observability.evaluation.eval_runner import EvalRunner
from src.observability.evaluation.composite_evaluator import create_composite_evaluator


def render():
    """渲染评估面板页面"""
    st.title("📊 评估面板")
    st.markdown("运行评估并查看 RAG 系统性能指标")
    st.markdown("---")

    # 加载配置
    try:
        settings = load_settings()
    except Exception as e:
        st.error(f"❌ 配置加载失败: {str(e)}")
        return

    # 侧边栏：评估配置
    st.sidebar.header("⚙️ 评估配置")

    # 选择评估后端
    evaluator_type = st.sidebar.selectbox(
        "评估器类型",
        ["basic", "ragas", "composite"],
        help="选择评估器类型"
    )

    # 如果选择 composite，显示后端选择
    backends = []
    if evaluator_type == "composite":
        st.sidebar.markdown("**组合评估器后端：**")
        use_ragas = st.sidebar.checkbox("Ragas", value=True)
        use_custom = st.sidebar.checkbox("Custom", value=True)

        if use_ragas:
            backends.append("ragas")
        if use_custom:
            backends.append("custom")

    # 选择测试集
    test_set_options = _get_available_test_sets()
    selected_test_set = st.sidebar.selectbox(
        "黄金测试集",
        test_set_options,
        help="选择要使用的黄金测试集"
    )

    # Top-K 配置
    top_k = st.sidebar.number_input(
        "Top-K",
        min_value=1,
        max_value=50,
        value=5,
        help="检索返回的文档数量"
    )

    # 运行按钮
    run_evaluation = st.sidebar.button("🚀 运行评估", type="primary", use_container_width=True)

    # 主内容区域
    if run_evaluation:
        _run_evaluation(
            settings,
            evaluator_type,
            backends,
            selected_test_set,
            top_k
        )
    else:
        _show_instructions()


def _get_available_test_sets():
    """获取可用的测试集列表"""
    fixtures_dir = Path("tests/fixtures")
    if not fixtures_dir.exists():
        return ["tests/fixtures/golden_test_set.json"]

    test_sets = []
    for file in fixtures_dir.glob("*.json"):
        if "test_set" in file.name or "golden" in file.name:
            test_sets.append(str(file))

    if not test_sets:
        test_sets = ["tests/fixtures/golden_test_set.json"]

    return test_sets


def _show_instructions():
    """显示使用说明"""
    st.info("👈 请在左侧配置评估参数，然后点击「运行评估」按钮")

    st.markdown("### 📖 使用说明")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**评估器类型：**")
        st.markdown("- **basic**: 基础指标（Hit Rate、Precision、Recall）")
        st.markdown("- **ragas**: Ragas 框架指标（Faithfulness、Answer Relevancy 等）")
        st.markdown("- **composite**: 组合多个评估器")

    with col2:
        st.markdown("**评估指标说明：**")
        st.markdown("- **Hit Rate**: 至少命中一个相关文档的比例")
        st.markdown("- **Precision**: 检索结果中相关文档的比例")
        st.markdown("- **Recall**: 相关文档中被检索到的比例")
        st.markdown("- **MRR**: 平均倒数排名")

    st.markdown("---")
    st.markdown("### 📝 黄金测试集格式")
    st.code("""
{
  "test_cases": [
    {
      "query": "如何配置 Azure OpenAI？",
      "expected_chunk_ids": ["chunk_001", "chunk_002"],
      "expected_sources": ["guide.pdf"]
    }
  ]
}
    """, language="json")


def _run_evaluation(settings, evaluator_type, backends, test_set_path, top_k):
    """运行评估"""
    st.markdown("### 🔄 正在运行评估...")

    progress_bar = st.progress(0)
    status_text = st.empty()

    try:
        # 1. 初始化 HybridSearch
        status_text.text("初始化 HybridSearch...")
        progress_bar.progress(20)

        hybrid_search = HybridSearch(settings)

        # 2. 初始化评估器
        status_text.text(f"初始化评估器 ({evaluator_type})...")
        progress_bar.progress(40)

        if evaluator_type == "composite":
            eval_settings = {
                'evaluation': {
                    'backends': backends if backends else ['custom']
                }
            }
            evaluator = create_composite_evaluator(eval_settings, EvaluatorFactory)
        else:
            evaluator = EvaluatorFactory.create({'evaluator': {'provider': evaluator_type}})

        # 3. 创建 EvalRunner
        status_text.text("创建 EvalRunner...")
        progress_bar.progress(60)

        eval_runner = EvalRunner(
            settings=settings.model_dump(),
            hybrid_search=hybrid_search,
            evaluator=evaluator
        )

        # 4. 运行评估
        status_text.text("运行评估...")
        progress_bar.progress(80)

        report = eval_runner.run(test_set_path, top_k=top_k)

        progress_bar.progress(100)
        status_text.text("✅ 评估完成！")

        # 5. 显示结果
        _display_results(report)

    except Exception as e:
        st.error(f"❌ 评估失败: {str(e)}")
        with st.expander("查看错误详情"):
            import traceback
            st.code(traceback.format_exc())


def _display_results(report):
    """显示评估结果"""
    st.markdown("---")
    st.markdown("### 📊 评估结果")

    # 概览指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总测试用例", report.total_cases)
    with col2:
        st.metric("成功", report.successful_cases, delta=None)
    with col3:
        st.metric("失败", report.failed_cases, delta=None)

    # 聚合指标
    if report.aggregate_metrics:
        st.markdown("### 📈 聚合指标")

        # 使用列显示指标
        metrics_items = list(report.aggregate_metrics.items())
        num_cols = min(4, len(metrics_items))
        cols = st.columns(num_cols)

        for idx, (metric_name, value) in enumerate(metrics_items):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.metric(
                    metric_name.replace("_", " ").title(),
                    f"{value:.4f}"
                )

    # 详细结果
    st.markdown("### 📋 详细结果")

    # 成功的测试
    successful_results = [r for r in report.test_results if r.success]
    if successful_results:
        with st.expander(f"✅ 成功的测试 ({len(successful_results)})", expanded=True):
            for idx, result in enumerate(successful_results, start=1):
                st.markdown(f"**{idx}. {result.query}**")

                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"检索到: {len(result.retrieved_ids)} 个文档")
                    st.write(f"期望: {len(result.expected_ids)} 个文档")
                with col2:
                    if result.metrics:
                        for metric_name, value in result.metrics.items():
                            st.write(f"{metric_name}: {value:.4f}")

                st.markdown("---")

    # 失败的测试
    failed_results = [r for r in report.test_results if not r.success]
    if failed_results:
        with st.expander(f"❌ 失败的测试 ({len(failed_results)})", expanded=False):
            for idx, result in enumerate(failed_results, start=1):
                st.markdown(f"**{idx}. {result.query}**")
                st.error(f"错误: {result.error}")
                st.markdown("---")

    # 下载报告
    st.markdown("### 💾 导出报告")
    report_json = report.to_json()
    st.download_button(
        label="📥 下载 JSON 报告",
        data=report_json,
        file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


if __name__ == "__main__":
    render()
