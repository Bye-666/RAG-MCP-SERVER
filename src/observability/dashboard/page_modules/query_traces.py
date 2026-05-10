"""
Query 追踪页面。

显示查询历史、Dense/Sparse 对比、Rerank 前后排名变化。
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

from src.observability.dashboard.services.trace_service import TraceService


def _init_trace_service():
    """初始化追踪服务"""
    if "trace_service" not in st.session_state:
        st.session_state.trace_service = TraceService()
    return st.session_state.trace_service


def _render_trace_list(trace_service: TraceService):
    """渲染查询追踪列表"""
    st.subheader("📋 查询历史")

    # 搜索框
    search_keyword = st.text_input("🔍 搜索查询", placeholder="输入查询关键词...")

    # 获取追踪记录
    if search_keyword:
        traces = trace_service.search_traces(search_keyword, trace_type="query")
    else:
        traces = trace_service.get_query_traces(limit=50)

    if not traces:
        st.info("📭 暂无查询追踪记录。请先执行查询操作。")
        return None

    # 显示统计
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总记录数", len(traces))
    with col2:
        avg_duration = sum(t.get("total_elapsed_ms", 0) for t in traces) / len(traces)
        st.metric("平均耗时", f"{avg_duration:.0f} ms")
    with col3:
        success_count = sum(1 for t in traces if t.get("finished_at"))
        st.metric("成功率", f"{success_count}/{len(traces)}")

    st.markdown("---")

    # 追踪列表
    selected_trace = None

    for idx, trace in enumerate(traces):
        trace_id = trace.get("trace_id", "unknown")
        started_at = trace.get("started_at", "")
        total_elapsed = trace.get("total_elapsed_ms", 0)
        metadata = trace.get("metadata", {})

        # 提取查询文本
        query_text = metadata.get("query", "未知查询")
        if len(query_text) > 50:
            query_text = query_text[:50] + "..."

        # 格式化时间
        try:
            dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = started_at[:19] if started_at else "未知"

        # 显示追踪项
        with st.expander(f"🔍 {time_str} - {query_text} ({total_elapsed:.0f}ms)", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Trace ID:** `{trace_id[:16]}...`")
                st.write(f"**查询:** {metadata.get('query', '未知')}")
                st.write(f"**总耗时:** {total_elapsed:.2f} ms")

            with col2:
                if metadata:
                    st.write("**参数:**")
                    if "top_k" in metadata:
                        st.caption(f"Top K: {metadata['top_k']}")
                    if "filters" in metadata:
                        st.caption(f"Filters: {metadata['filters']}")

            # 查看详情按钮
            if st.button("📊 查看详情", key=f"view_trace_{idx}"):
                selected_trace = trace_id

    return selected_trace


def _render_trace_detail(trace_service: TraceService, trace_id: str):
    """渲染查询追踪详情"""
    st.subheader("📊 查询追踪详情")

    # 返回按钮
    if st.button("⬅️ 返回列表"):
        if "selected_trace" in st.session_state:
            del st.session_state.selected_trace
        st.rerun()

    st.markdown("---")

    # 获取追踪详情
    trace = trace_service.get_trace_by_id(trace_id)

    if not trace:
        st.error(f"❌ 未找到追踪记录: {trace_id}")
        return

    # 显示基本信息
    metadata = trace.get("metadata", {})
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("查询", metadata.get("query", "未知")[:20] + "...")
    with col2:
        total_elapsed = trace.get("total_elapsed_ms", 0)
        st.metric("总耗时", f"{total_elapsed:.2f} ms")
    with col3:
        stage_count = len(trace.get("stages", []))
        st.metric("阶段数", stage_count)

    st.markdown("---")

    # Tabs: 耗时分析、检索对比、Rerank 分析
    tab1, tab2, tab3 = st.tabs(["⏱️ 耗时分析", "🔍 检索对比", "🎯 Rerank 分析"])

    with tab1:
        _render_stage_waterfall(trace)

    with tab2:
        _render_retrieval_comparison(trace)

    with tab3:
        _render_rerank_analysis(trace)


def _render_stage_waterfall(trace: Dict[str, Any]):
    """渲染阶段耗时瀑布图"""
    st.write("**阶段耗时分布**")

    stages = trace.get("stages", [])

    if not stages:
        st.info("无阶段数据")
        return

    # 准备数据
    stage_data = []
    for stage in stages:
        stage_name = stage.get("stage_name", "unknown")
        duration_ms = stage.get("duration_ms", 0)

        # 阶段名称映射
        stage_names = {
            "query_process": "🔤 查询处理",
            "dense_retrieval": "🔢 稠密检索",
            "sparse_retrieval": "📊 稀疏检索",
            "fusion": "🔀 结果融合",
            "rerank": "🎯 重排序",
            "response_build": "📝 响应构建"
        }
        display_name = stage_names.get(stage_name, stage_name)

        stage_data.append({
            "阶段": display_name,
            "耗时 (ms)": duration_ms
        })

    # 创建 DataFrame
    df = pd.DataFrame(stage_data)

    # 显示横向条形图
    st.bar_chart(df.set_index("阶段"))

    # 显示详细表格
    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_retrieval_comparison(trace: Dict[str, Any]):
    """渲染 Dense vs Sparse 检索对比"""
    st.write("**Dense vs Sparse 检索对比**")

    stages = trace.get("stages", [])

    # 查找 dense 和 sparse 阶段
    dense_stage = next((s for s in stages if s.get("stage_name") == "dense_retrieval"), None)
    sparse_stage = next((s for s in stages if s.get("stage_name") == "sparse_retrieval"), None)

    if not dense_stage and not sparse_stage:
        st.info("无检索阶段数据")
        return

    # 显示对比指标
    col1, col2 = st.columns(2)

    with col1:
        st.write("**🔢 Dense 检索**")
        if dense_stage:
            dense_meta = dense_stage.get("metadata", {})
            st.metric("耗时", f"{dense_stage.get('duration_ms', 0):.2f} ms")
            st.metric("结果数", dense_meta.get("result_count", 0))

            # 显示 Top 结果
            dense_results = dense_meta.get("top_results", [])
            if dense_results:
                st.write("**Top 结果:**")
                for idx, result in enumerate(dense_results[:5]):
                    score = result.get("score", 0)
                    chunk_id = result.get("id", "unknown")
                    st.caption(f"{idx+1}. `{chunk_id[:12]}...` (score: {score:.3f})")
        else:
            st.info("无 Dense 检索数据")

    with col2:
        st.write("**📊 Sparse 检索**")
        if sparse_stage:
            sparse_meta = sparse_stage.get("metadata", {})
            st.metric("耗时", f"{sparse_stage.get('duration_ms', 0):.2f} ms")
            st.metric("结果数", sparse_meta.get("result_count", 0))

            # 显示 Top 结果
            sparse_results = sparse_meta.get("top_results", [])
            if sparse_results:
                st.write("**Top 结果:**")
                for idx, result in enumerate(sparse_results[:5]):
                    score = result.get("score", 0)
                    chunk_id = result.get("id", "unknown")
                    st.caption(f"{idx+1}. `{chunk_id[:12]}...` (score: {score:.3f})")
        else:
            st.info("无 Sparse 检索数据")


def _render_rerank_analysis(trace: Dict[str, Any]):
    """渲染 Rerank 前后排名变化"""
    st.write("**Rerank 前后排名变化**")

    stages = trace.get("stages", [])

    # 查找 fusion 和 rerank 阶段
    fusion_stage = next((s for s in stages if s.get("stage_name") == "fusion"), None)
    rerank_stage = next((s for s in stages if s.get("stage_name") == "rerank"), None)

    if not fusion_stage or not rerank_stage:
        st.info("无 Rerank 数据（可能未启用 Reranker）")
        return

    # 获取 Rerank 前后的结果
    fusion_meta = fusion_stage.get("metadata", {})
    rerank_meta = rerank_stage.get("metadata", {})

    before_results = fusion_meta.get("fused_results", [])
    after_results = rerank_meta.get("reranked_results", [])

    if not before_results or not after_results:
        st.info("无 Rerank 结果数据")
        return

    # 显示 Rerank 指标
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rerank 耗时", f"{rerank_stage.get('duration_ms', 0):.2f} ms")
    with col2:
        st.metric("输入数量", len(before_results))
    with col3:
        st.metric("输出数量", len(after_results))

    st.markdown("---")

    # 显示排名变化表格
    st.write("**排名变化对比:**")

    # 构建排名映射
    before_rank = {r.get("id"): idx + 1 for idx, r in enumerate(before_results)}

    comparison_data = []
    for idx, result in enumerate(after_results[:10]):
        chunk_id = result.get("id", "unknown")
        after_rank = idx + 1
        before = before_rank.get(chunk_id, "-")

        # 计算排名变化
        if isinstance(before, int):
            change = before - after_rank
            if change > 0:
                change_str = f"↑ {change}"
            elif change < 0:
                change_str = f"↓ {abs(change)}"
            else:
                change_str = "="
        else:
            change_str = "新增"

        comparison_data.append({
            "Rerank 后": after_rank,
            "Rerank 前": before,
            "变化": change_str,
            "Chunk ID": chunk_id[:16] + "...",
            "Score": f"{result.get('score', 0):.3f}"
        })

    df = pd.DataFrame(comparison_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render():
    """渲染 Query 追踪页面"""
    st.title("🔍 Query 追踪")
    st.markdown("查看查询历史、检索对比和 Rerank 分析")
    st.markdown("---")

    # 初始化服务
    try:
        trace_service = _init_trace_service()
    except Exception as e:
        st.error(f"❌ 初始化服务失败: {e}")
        return

    # 检查是否有选中的追踪
    if "selected_trace" not in st.session_state:
        st.session_state.selected_trace = None

    # 渲染列表或详情
    if st.session_state.selected_trace:
        _render_trace_detail(trace_service, st.session_state.selected_trace)
    else:
        selected = _render_trace_list(trace_service)
        if selected:
            st.session_state.selected_trace = selected
            st.rerun()


if __name__ == "__main__":
    render()
