"""
查询测试页面 - 交互式查询测试工具
"""
import streamlit as st
from pathlib import Path
import json
from datetime import datetime

from src.core.settings import load_settings
from src.core.query_engine.hybrid_search import HybridSearch


def render():
    """渲染查询测试页面"""
    st.title("🧪 查询测试")
    st.markdown("交互式查询测试工具 - 测试检索效果")

    # 添加刷新按钮
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🔄 刷新索引", help="重新加载索引数据，删除文档后需要点击此按钮"):
            if 'hybrid_search' in st.session_state:
                del st.session_state.hybrid_search
            st.rerun()

    st.markdown("---")

    # 初始化HybridSearch
    if 'hybrid_search' not in st.session_state:
        try:
            settings = load_settings()
            st.session_state.hybrid_search = HybridSearch(settings)
        except Exception as e:
            st.error(f"❌ 初始化失败: {str(e)}")
            st.info("请确保已配置向量存储和嵌入模型")
            return

    hybrid_search = st.session_state.hybrid_search

    # 查询配置
    st.subheader("⚙️ 查询配置")
    col1, col2, col3 = st.columns(3)

    with col1:
        top_k = st.number_input("检索数量 (top_k)", min_value=1, max_value=50, value=5)

    with col2:
        use_rerank = st.checkbox("启用重排序 (Rerank)", value=False, disabled=True)

    with col3:
        rerank_top_k = st.number_input(
            "重排序后数量",
            min_value=1,
            max_value=top_k,
            value=min(3, top_k),
            disabled=True
        )

    # 高级配置
    with st.expander("🔧 高级配置"):
        st.info("ℹ️ Dense/Sparse权重和重排序功能暂未在后端实现，当前仅支持基础混合检索")
        col1, col2 = st.columns(2)
        with col1:
            dense_weight = st.slider("Dense权重", 0.0, 1.0, 0.5, 0.1, disabled=True)
        with col2:
            sparse_weight = st.slider("Sparse权重", 0.0, 1.0, 0.5, 0.1, disabled=True)

        enable_trace = st.checkbox("启用追踪", value=True)

    st.markdown("---")

    # 查询输入
    st.subheader("🔍 输入查询")
    query_text = st.text_area(
        "查询文本",
        placeholder="输入您的查询...",
        height=100,
        key="query_input"
    )

    # 元数据过滤器（可选）
    with st.expander("🏷️ 元数据过滤器（可选）"):
        st.markdown("输入JSON格式的过滤条件，例如: `{\"source\": \"document.pdf\"}`")
        filter_text = st.text_area(
            "过滤器",
            placeholder='{"key": "value"}',
            height=80,
            key="filter_input"
        )

    # 执行查询按钮
    if st.button("🚀 执行查询", type="primary", disabled=not query_text.strip()):
        with st.spinner("正在检索..."):
            try:
                # 解析过滤器
                filters = None
                if filter_text.strip():
                    try:
                        filters = json.loads(filter_text)
                    except json.JSONDecodeError as e:
                        st.error(f"❌ 过滤器JSON格式错误: {str(e)}")
                        return

                # 执行查询
                # 注意：当前 HybridSearch 不支持 dense_weight, sparse_weight, rerank 参数
                # 这些参数在界面上显示但暂未实现
                trace_id = f"query_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if enable_trace else None

                # 创建 TraceContext（如果启用追踪）
                trace = None
                if enable_trace:
                    from src.core.trace import TraceContext
                    trace = TraceContext(trace_id=trace_id)

                results = hybrid_search.search(
                    query=query_text.strip(),
                    top_k=top_k,
                    filters=filters,
                    trace=trace
                )

                # 保存结果到session_state
                st.session_state.last_results = results
                st.session_state.last_query = query_text.strip()
                st.session_state.last_trace_id = trace_id

                st.success(f"✅ 检索完成！找到 {len(results)} 个结果")

            except Exception as e:
                st.error(f"❌ 查询失败: {str(e)}")
                import traceback
                with st.expander("查看错误详情"):
                    st.code(traceback.format_exc())
                return

    # 显示结果
    if 'last_results' in st.session_state and st.session_state.last_results:
        st.markdown("---")
        st.subheader("📊 检索结果")

        results = st.session_state.last_results
        query = st.session_state.last_query
        trace_id = st.session_state.last_trace_id

        # 结果统计
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("结果数量", len(results))
        with col2:
            avg_score = sum(r.score for r in results) / len(results) if results else 0
            st.metric("平均分数", f"{avg_score:.4f}")
        with col3:
            if trace_id:
                st.metric("Trace ID", trace_id[:16] + "...")

        st.markdown("---")

        # 显示每个结果
        for idx, result in enumerate(results, 1):
            with st.expander(f"📄 结果 #{idx} - 分数: {result.score:.4f}"):
                # 文档内容
                st.markdown("**文档内容**")
                st.text_area(
                    "内容",
                    value=result.text,
                    height=150,
                    key=f"result_content_{idx}",
                    disabled=True
                )

                # 元数据
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**元数据**")
                    if result.metadata:
                        for key, value in result.metadata.items():
                            st.write(f"- **{key}**: {value}")
                    else:
                        st.write("无元数据")

                with col2:
                    st.markdown("**检索信息**")
                    st.write(f"- **分数**: {result.score:.4f}")
                    st.write(f"- **块ID**: `{result.chunk_id}`")
                    if 'chunk_index' in result.metadata:
                        st.write(f"- **分块索引**: {result.metadata['chunk_index']}")

        # 追踪信息
        if trace_id:
            st.markdown("---")
            st.info("💡 查看详细追踪信息请前往 **📉 查询追踪** 页面")

    elif 'last_results' in st.session_state and not st.session_state.last_results:
        st.warning("⚠️ 未找到匹配的文档")

    # 使用说明
    with st.expander("ℹ️ 使用说明"):
        st.markdown("""
        ### 查询测试工具使用指南

        **基本步骤：**
        1. 配置查询参数（top_k、是否启用rerank等）
        2. 输入查询文本
        3. （可选）添加元数据过滤器
        4. 点击"执行查询"按钮
        5. 查看检索结果

        **参数说明：**
        - **top_k**: 初始检索的文档数量
        - **启用重排序**: 是否使用Reranker对结果重新排序
        - **重排序后数量**: 重排序后保留的文档数量
        - **Dense权重**: 密集向量检索的权重（0-1）
        - **Sparse权重**: 稀疏向量检索的权重（0-1）
        - **启用追踪**: 是否记录查询追踪信息

        **元数据过滤器：**
        - 使用JSON格式指定过滤条件
        - 例如: `{"source": "document.pdf", "page": 1}`
        - 只返回匹配所有条件的文档

        **注意事项：**
        - 确保已经摄取了文档数据
        - Dense权重 + Sparse权重 不一定要等于1.0
        - 重排序会提高精度但增加延迟
        """)


if __name__ == "__main__":
    render()
