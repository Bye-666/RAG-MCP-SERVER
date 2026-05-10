"""
追踪总览页面 - 统一的追踪入口
"""
import streamlit as st
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any


def load_traces() -> List[Dict[str, Any]]:
    """加载所有追踪记录"""
    trace_file = Path("logs/traces.jsonl")
    if not trace_file.exists():
        return []

    traces = []
    with open(trace_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    traces.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    # 按时间倒序排序
    traces.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    return traces


def format_timestamp(ts: str) -> str:
    """格式化时间戳"""
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except:
        return ts


def render():
    """渲染追踪总览页面"""
    st.title("🔍 追踪总览")
    st.markdown("查看所有追踪记录（摄取 + 查询）")
    st.markdown("---")

    # 加载追踪记录
    traces = load_traces()

    if not traces:
        st.info("📭 暂无追踪记录。执行摄取或查询操作后，追踪记录将显示在这里。")
        return

    # 统计信息
    col1, col2, col3 = st.columns(3)
    ingestion_count = sum(1 for t in traces if t.get('trace_type') == 'ingestion')
    query_count = sum(1 for t in traces if t.get('trace_type') == 'query')

    with col1:
        st.metric("总追踪数", len(traces))
    with col2:
        st.metric("摄取追踪", ingestion_count)
    with col3:
        st.metric("查询追踪", query_count)

    st.markdown("---")

    # 筛选器
    col1, col2 = st.columns([1, 3])
    with col1:
        trace_type_filter = st.selectbox(
            "追踪类型",
            ["全部", "摄取", "查询"],
            key="trace_type_filter"
        )

    with col2:
        search_query = st.text_input("🔍 搜索（trace_id 或关键词）", key="search_query")

    # 过滤追踪记录
    filtered_traces = traces
    if trace_type_filter == "摄取":
        filtered_traces = [t for t in filtered_traces if t.get('trace_type') == 'ingestion']
    elif trace_type_filter == "查询":
        filtered_traces = [t for t in filtered_traces if t.get('trace_type') == 'query']

    if search_query:
        search_lower = search_query.lower()
        filtered_traces = [
            t for t in filtered_traces
            if search_lower in t.get('trace_id', '').lower()
            or search_lower in str(t.get('metadata', {})).lower()
        ]

    st.write(f"**显示 {len(filtered_traces)} 条记录**")
    st.markdown("---")

    # 显示追踪列表
    for trace in filtered_traces:
        trace_type = trace.get('trace_type', 'unknown')
        trace_id = trace.get('trace_id', 'N/A')
        timestamp = format_timestamp(trace.get('timestamp', ''))

        # 根据类型显示不同的图标和颜色
        if trace_type == 'ingestion':
            icon = "📥"
            type_label = "摄取"
        elif trace_type == 'query':
            icon = "🔎"
            type_label = "查询"
        else:
            icon = "❓"
            type_label = trace_type

        with st.expander(f"{icon} {type_label} - {timestamp} - {trace_id[:16]}..."):
            col1, col2 = st.columns([1, 1])

            with col1:
                st.write("**基本信息**")
                st.write(f"- Trace ID: `{trace_id}`")
                st.write(f"- 类型: {type_label}")
                st.write(f"- 时间: {timestamp}")

            with col2:
                st.write("**元数据**")
                metadata = trace.get('metadata', {})
                if metadata:
                    for key, value in list(metadata.items())[:5]:
                        st.write(f"- {key}: {value}")
                else:
                    st.write("无元数据")

            # 显示阶段信息
            if 'stages' in trace:
                st.write("**阶段耗时**")
                stages = trace['stages']
                if isinstance(stages, dict):
                    for stage_name, stage_data in stages.items():
                        duration = stage_data.get('duration_ms', 0)
                        st.write(f"- {stage_name}: {duration:.2f} ms")
                elif isinstance(stages, list):
                    for stage in stages:
                        stage_name = stage.get('stage_name', 'unknown')
                        duration = stage.get('duration_ms', 0)
                        st.write(f"- {stage_name}: {duration:.2f} ms")

            # 提供跳转提示
            if trace_type == 'ingestion':
                st.info("💡 查看详细分析请前往 **📈 摄取追踪** 页面")
            elif trace_type == 'query':
                st.info("💡 查看详细分析请前往 **📉 查询追踪** 页面")


if __name__ == "__main__":
    render()
