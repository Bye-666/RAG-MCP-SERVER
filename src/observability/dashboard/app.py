"""
RAG-MCP-SERVER Dashboard

Multi-page Streamlit application for system monitoring and management.
"""

import streamlit as st

from src.observability.dashboard.pages import (
    overview,
    traces,
    data_browser,
    ingestion,
    ingestion_traces,
    query_traces,
    query_test,
    settings,
    evaluation_panel,
)


def main():
    """Main dashboard application"""

    # Page configuration
    st.set_page_config(
        page_title="RAG-MCP 管理平台",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Navigation
    pages = {
        "🏠 系统总览": overview,
        "📊 追踪总览": traces,
        "📚 数据浏览": data_browser,
        "📥 摄取管理": ingestion,
        "📈 摄取追踪": ingestion_traces,
        "🔍 查询测试": query_test,
        "📉 查询追踪": query_traces,
        "📊 评估面板": evaluation_panel,
        "⚙️ 系统设置": settings,
    }

    # Sidebar navigation
    st.sidebar.title("RAG-MCP 管理平台")
    st.sidebar.markdown("---")

    selection = st.sidebar.radio("导航", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.caption("v0.1.0 | Task G1")

    # Render selected page
    pages[selection].render()


if __name__ == "__main__":
    main()
