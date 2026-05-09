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
    query_test,
    settings,
)


def main():
    """Main dashboard application"""

    # Page configuration
    st.set_page_config(
        page_title="RAG-MCP Dashboard",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Navigation
    pages = {
        "🏠 Overview": overview,
        "📊 Traces": traces,
        "📚 Data Browser": data_browser,
        "📥 Ingestion": ingestion,
        "🔍 Query Test": query_test,
        "⚙️ Settings": settings,
    }

    # Sidebar navigation
    st.sidebar.title("RAG-MCP Dashboard")
    st.sidebar.markdown("---")

    selection = st.sidebar.radio("Navigation", list(pages.keys()))

    st.sidebar.markdown("---")
    st.sidebar.caption("v0.1.0 | Task G1")

    # Render selected page
    pages[selection].render()


if __name__ == "__main__":
    main()
