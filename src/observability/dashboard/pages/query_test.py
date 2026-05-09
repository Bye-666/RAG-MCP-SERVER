"""
Query Test page for Dashboard.

Test queries and view results.
"""

import streamlit as st


def render():
    """Render the query test page"""
    st.title("🔍 Query Test")
    st.info("🚧 This page is under construction (Task G5)")
    st.markdown("""
    ### Planned Features:
    - Interactive query input
    - View retrieved chunks and scores
    - Display multimodal results (text + images)
    - Adjust retrieval parameters
    - Compare different query strategies
    """)


if __name__ == "__main__":
    render()
