"""
Dashboard 系统总览页面。

显示系统配置和组件状态。
"""

import streamlit as st
from pathlib import Path

from src.core.settings import load_settings
from src.observability.dashboard.services.config_service import ConfigService


def render():
    """渲染系统总览页面"""
    st.title("🏠 系统总览")
    st.markdown("---")

    # 加载设置
    settings = load_settings()
    config_service = ConfigService(settings)

    # 系统信息
    st.header("📊 系统信息")
    col1, col2 = st.columns(2)

    vs_config = config_service.get_vector_store_config()
    reranker_config = config_service.get_reranker_config()

    with col1:
        st.metric("项目", "RAG-MCP-SERVER")
        st.metric("环境", "开发环境")
    with col2:
        st.metric("向量数据库", vs_config.get("provider", "N/A"))
        st.metric("重排序", "已启用" if reranker_config.get("enabled", False) else "未启用")

    st.markdown("---")

    # 组件配置
    st.header("⚙️ 组件配置")

    # LLM 配置
    with st.expander("🤖 LLM 配置", expanded=True):
        llm_config = config_service.get_llm_config()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**提供商:** {llm_config['provider']}")
            st.write(f"**模型:** {llm_config['model']}")
            st.write(f"**API 地址:** {llm_config['api_base']}")
        with col2:
            st.write(f"**温度:** {llm_config['temperature']}")
            st.write(f"**最大 Token:** {llm_config['max_tokens']}")

    # Embedding Config
    with st.expander("🔢 Embedding 配置", expanded=True):
        emb_config = config_service.get_embedding_config()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**提供商:** {emb_config['provider']}")
            st.write(f"**模型:** {emb_config['model']}")
        with col2:
            st.write(f"**API 地址:** {emb_config['api_base']}")
            st.write(f"**维度:** {emb_config['dimensions']}")

    # Splitter Config
    with st.expander("✂️ 切分器配置"):
        split_config = config_service.get_splitter_config()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**类型:** {split_config['type']}")
        with col2:
            st.write(f"**块大小:** {split_config['chunk_size']}")
        with col3:
            st.write(f"**重叠:** {split_config['chunk_overlap']}")

    # Vector Store Config
    with st.expander("💾 向量数据库配置"):
        vs_config = config_service.get_vector_store_config()
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**提供商:** {vs_config['provider']}")
            st.write(f"**集合:** {vs_config['collection']}")
        with col2:
            st.write(f"**持久化目录:** {vs_config['persist_directory']}")

    # Reranker Config
    with st.expander("🎯 重排序配置"):
        rerank_config = config_service.get_reranker_config()
        if rerank_config['enabled']:
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**提供商:** {rerank_config['provider']}")
                st.write(f"**模型:** {rerank_config['model']}")
            with col2:
                st.write(f"**Top K:** {rerank_config['top_k']}")
        else:
            st.info("重排序未启用")

    # Retrieval Config
    with st.expander("🔍 检索配置"):
        ret_config = config_service.get_retrieval_config()
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.write(f"**Dense Top K:** {ret_config['dense_top_k']}")
        with col2:
            st.write(f"**Sparse Top K:** {ret_config['sparse_top_k']}")
        with col3:
            st.write(f"**最终 Top K:** {ret_config['final_top_k']}")
        with col4:
            st.write(f"**RRF K:** {ret_config['rrf_k']}")

    st.markdown("---")
    st.caption("💡 提示：使用侧边栏导航到其他页面")


if __name__ == "__main__":
    render()
