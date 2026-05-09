"""
System Overview page for Dashboard.

Displays system configuration and component status.
"""

import streamlit as st
from pathlib import Path

from src.core.settings import load_settings
from src.observability.dashboard.services.config_service import ConfigService


def render():
    """Render the system overview page"""
    st.title("System Overview")
    st.markdown("---")

    # Load settings
    settings = load_settings()
    config_service = ConfigService(settings)

    # System Info
    st.header("📊 System Information")
    col1, col2 = st.columns(2)

    vs_config = config_service.get_vector_store_config()
    reranker_config = config_service.get_reranker_config()

    with col1:
        st.metric("Project", "RAG-MCP-SERVER")
        st.metric("Environment", "Development")
    with col2:
        st.metric("Vector Store", vs_config.get("provider", "N/A"))
        st.metric("Reranker", "Enabled" if reranker_config.get("enabled", False) else "Disabled")

    st.markdown("---")

    # Component Configurations
    st.header("⚙️ Component Configurations")

    # LLM Config
    with st.expander("🤖 LLM Configuration", expanded=True):
        llm_config = config_service.get_llm_config()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Provider:** {llm_config['provider']}")
            st.write(f"**Model:** {llm_config['model']}")
            st.write(f"**API Base:** {llm_config['api_base']}")
        with col2:
            st.write(f"**Temperature:** {llm_config['temperature']}")
            st.write(f"**Max Tokens:** {llm_config['max_tokens']}")

    # Embedding Config
    with st.expander("🔢 Embedding Configuration", expanded=True):
        emb_config = config_service.get_embedding_config()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Provider:** {emb_config['provider']}")
            st.write(f"**Model:** {emb_config['model']}")
        with col2:
            st.write(f"**API Base:** {emb_config['api_base']}")
            st.write(f"**Dimensions:** {emb_config['dimensions']}")

    # Splitter Config
    with st.expander("✂️ Splitter Configuration"):
        split_config = config_service.get_splitter_config()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**Type:** {split_config['type']}")
        with col2:
            st.write(f"**Chunk Size:** {split_config['chunk_size']}")
        with col3:
            st.write(f"**Overlap:** {split_config['chunk_overlap']}")

    # Vector Store Config
    with st.expander("💾 Vector Store Configuration"):
        vs_config = config_service.get_vector_store_config()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Provider:** {vs_config['provider']}")
            st.write(f"**Collection:** {vs_config['collection']}")
        with col2:
            st.write(f"**Persist Directory:** {vs_config['persist_directory']}")

    # Reranker Config
    with st.expander("🎯 Reranker Configuration"):
        rerank_config = config_service.get_reranker_config()
        if rerank_config['enabled']:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Provider:** {rerank_config['provider']}")
                st.write(f"**Model:** {rerank_config['model']}")
            with col2:
                st.write(f"**Top K:** {rerank_config['top_k']}")
        else:
            st.info("Reranker is disabled")

    # Retrieval Config
    with st.expander("🔍 Retrieval Configuration"):
        ret_config = config_service.get_retrieval_config()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Dense Top K:** {ret_config['dense_top_k']}")
        with col2:
            st.write(f"**Sparse Top K:** {ret_config['sparse_top_k']}")
        with col3:
            st.write(f"**Final Top K:** {ret_config['final_top_k']}")
        with col4:
            st.write(f"**RRF K:** {ret_config['rrf_k']}")

    st.markdown("---")
    st.caption("💡 Tip: Use the sidebar to navigate to other pages")


if __name__ == "__main__":
    render()
