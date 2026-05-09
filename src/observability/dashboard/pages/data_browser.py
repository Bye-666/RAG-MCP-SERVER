"""
Data Browser page for Dashboard.

Browse documents, chunks, and images in the vector store.
"""

import streamlit as st


def render():
    """Render the data browser page"""
    st.title("📚 Data Browser")
    st.info("🚧 This page is under construction (Task G3)")
    st.markdown("""
    ### Planned Features:
    - Browse all documents in collections
    - View document metadata and chunks
    - Display embedded images
    - Search and filter documents
    - Delete or update documents
    """)


if __name__ == "__main__":
    render()
