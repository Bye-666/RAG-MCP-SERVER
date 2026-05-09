"""
Ingestion Management page for Dashboard.

Upload and manage document ingestion.
"""

import streamlit as st


def render():
    """Render the ingestion management page"""
    st.title("📥 Ingestion Management")
    st.info("🚧 This page is under construction (Task G4)")
    st.markdown("""
    ### Planned Features:
    - Upload files for ingestion
    - Real-time progress tracking
    - View ingestion history
    - Batch ingestion support
    - Error handling and retry
    """)


if __name__ == "__main__":
    render()
