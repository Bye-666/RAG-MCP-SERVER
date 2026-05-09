"""
Ingestion 追踪页面。

显示摄取历史列表和阶段耗时瀑布图。
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Any

from src.observability.dashboard.services.trace_service import TraceService


def _init_trace_service():
    """初始化追踪服务"""
    if "trace_service" not in st.session_state:
        st.session_state.trace_service = TraceService()
    return st.session_state.trace_service


def _render_trace_list(trace_service: TraceService):
    """渲染追踪列表"""
    st.subheader("📋 摄取历史")

    # 获取追踪记录
    traces = trace_service.get_ingestion_traces(limit=50)

    if not traces:
        st.info("📭 暂无摄取追踪记录。请先执行文档摄取。")
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

        # 格式化时间
        try:
            dt = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = started_at[:19] if started_at else "未知"

        # 显示追踪项
        with st.expander(f"🔍 {time_str} - {total_elapsed:.0f}ms", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**Trace ID:** `{trace_id[:16]}...`")
                st.write(f"**开始时间:** {time_str}")
                st.write(f"**总耗时:** {total_elapsed:.2f} ms")

            with col2:
                if metadata:
                    st.write("**Metadata:**")
                    for key, value in list(metadata.items())[:3]:
                        st.caption(f"{key}: {value}")

            # 查看详情按钮
            if st.button("📊 查看详情", key=f"view_trace_{idx}"):
                selected_trace = trace_id

    return selected_trace


def _render_trace_detail(trace_service: TraceService, trace_id: str):
    """渲染追踪详情"""
    st.subheader("📊 追踪详情")

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
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Trace ID", f"{trace_id[:8]}...")
    with col2:
        total_elapsed = trace.get("total_elapsed_ms", 0)
        st.metric("总耗时", f"{total_elapsed:.2f} ms")
    with col3:
        stage_count = len(trace.get("stages", []))
        st.metric("阶段数", stage_count)

    st.markdown("---")

    # 显示阶段耗时瀑布图
    _render_stage_waterfall(trace)

    st.markdown("---")

    # 显示阶段详情表格
    _render_stage_table(trace)


def _render_stage_waterfall(trace: Dict[str, Any]):
    """渲染阶段耗时瀑布图"""
    st.write("**⏱️ 阶段耗时瀑布图**")

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
            "load": "📄 加载文档",
            "split": "✂️ 切分 Chunk",
            "transform": "🔄 转换增强",
            "encode": "🔢 向量编码",
            "upsert": "💾 存储数据",
            "integrity_check": "✅ 完整性检查"
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


def _render_stage_table(trace: Dict[str, Any]):
    """渲染阶段详情表格"""
    st.write("**📋 阶段详情**")

    stages = trace.get("stages", [])

    if not stages:
        st.info("无阶段数据")
        return

    # 准备表格数据
    table_data = []
    for idx, stage in enumerate(stages):
        stage_name = stage.get("stage_name", "unknown")
        duration_ms = stage.get("duration_ms", 0)
        start_time = stage.get("start_time", "")
        end_time = stage.get("end_time", "")
        metadata = stage.get("metadata", {})

        # 格式化时间
        try:
            start_dt = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
            start_str = start_dt.strftime("%H:%M:%S.%f")[:-3]
        except:
            start_str = start_time[:12] if start_time else "-"

        try:
            end_dt = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
            end_str = end_dt.strftime("%H:%M:%S.%f")[:-3]
        except:
            end_str = end_time[:12] if end_time else "-"

        # 提取关键 metadata
        metadata_str = ""
        if "chunk_count" in metadata:
            metadata_str += f"chunks: {metadata['chunk_count']}"
        if "doc_id" in metadata:
            metadata_str += f" | doc: {metadata['doc_id'][:8]}"

        table_data.append({
            "序号": idx + 1,
            "阶段": stage_name,
            "开始时间": start_str,
            "结束时间": end_str,
            "耗时 (ms)": f"{duration_ms:.2f}",
            "备注": metadata_str
        })

    # 显示表格
    df = pd.DataFrame(table_data)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render():
    """渲染 Ingestion 追踪页面"""
    st.title("📊 Ingestion 追踪")
    st.markdown("查看摄取历史和阶段耗时分析")
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
