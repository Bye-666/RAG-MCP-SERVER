"""
系统设置页面 - 配置编辑器
"""
import streamlit as st
from pathlib import Path
import sys
import yaml
from typing import Dict, Any

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.config.settings import Settings


def load_config_file() -> Dict[str, Any]:
    """加载配置文件"""
    config_path = project_root / "config.yaml"
    if not config_path.exists():
        return {}

    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_config_file(config: Dict[str, Any]) -> bool:
    """保存配置文件"""
    try:
        config_path = project_root / "config.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        return True
    except Exception as e:
        st.error(f"保存失败: {str(e)}")
        return False


def render():
    """渲染系统设置页面"""
    st.title("⚙️ 系统设置")
    st.markdown("编辑系统配置（修改后需重启Dashboard生效）")
    st.markdown("---")

    # 加载当前配置
    if 'config' not in st.session_state:
        st.session_state.config = load_config_file()
        st.session_state.settings = Settings()

    config = st.session_state.config
    settings = st.session_state.settings

    # 创建标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🤖 LLM配置",
        "📊 Embedding配置",
        "🗄️ 向量存储",
        "🔍 检索配置",
        "📝 原始配置"
    ])

    # LLM配置
    with tab1:
        st.subheader("LLM 配置")

        llm_config = config.get('llm', {})

        col1, col2 = st.columns(2)
        with col1:
            llm_provider = st.text_input(
                "提供商",
                value=llm_config.get('provider', 'openai'),
                key="llm_provider"
            )
            llm_model = st.text_input(
                "模型",
                value=llm_config.get('model', 'gpt-3.5-turbo'),
                key="llm_model"
            )

        with col2:
            llm_api_base = st.text_input(
                "API地址",
                value=llm_config.get('api_base', ''),
                key="llm_api_base"
            )
            llm_api_key = st.text_input(
                "API密钥",
                value=llm_config.get('api_key', ''),
                type="password",
                key="llm_api_key"
            )

        col1, col2 = st.columns(2)
        with col1:
            llm_temperature = st.slider(
                "温度 (Temperature)",
                min_value=0.0,
                max_value=2.0,
                value=float(llm_config.get('temperature', 0.7)),
                step=0.1,
                key="llm_temperature"
            )

        with col2:
            llm_max_tokens = st.number_input(
                "最大Token数",
                min_value=1,
                max_value=32000,
                value=llm_config.get('max_tokens', 2000),
                key="llm_max_tokens"
            )

    # Embedding配置
    with tab2:
        st.subheader("Embedding 配置")

        embedding_config = config.get('embedding', {})

        col1, col2 = st.columns(2)
        with col1:
            emb_provider = st.text_input(
                "提供商",
                value=embedding_config.get('provider', 'openai'),
                key="emb_provider"
            )
            emb_model = st.text_input(
                "模型",
                value=embedding_config.get('model', 'text-embedding-ada-002'),
                key="emb_model"
            )

        with col2:
            emb_api_base = st.text_input(
                "API地址",
                value=embedding_config.get('api_base', ''),
                key="emb_api_base"
            )
            emb_api_key = st.text_input(
                "API密钥",
                value=embedding_config.get('api_key', ''),
                type="password",
                key="emb_api_key"
            )

        emb_dimensions = st.number_input(
            "向量维度",
            min_value=1,
            max_value=4096,
            value=embedding_config.get('dimensions', 1536),
            key="emb_dimensions"
        )

    # 向量存储配置
    with tab3:
        st.subheader("向量存储配置")

        vector_store_config = config.get('vector_store', {})

        col1, col2 = st.columns(2)
        with col1:
            vs_provider = st.selectbox(
                "提供商",
                options=['chroma', 'qdrant', 'milvus'],
                index=['chroma', 'qdrant', 'milvus'].index(vector_store_config.get('provider', 'chroma')),
                key="vs_provider"
            )
            vs_collection = st.text_input(
                "集合名称",
                value=vector_store_config.get('collection_name', 'rag_collection'),
                key="vs_collection"
            )

        with col2:
            vs_persist_dir = st.text_input(
                "持久化目录",
                value=vector_store_config.get('persist_directory', './data/chroma'),
                key="vs_persist_dir"
            )

    # 检索配置
    with tab4:
        st.subheader("检索配置")

        retrieval_config = config.get('retrieval', {})

        col1, col2, col3 = st.columns(3)
        with col1:
            dense_top_k = st.number_input(
                "Dense Top K",
                min_value=1,
                max_value=100,
                value=retrieval_config.get('dense_top_k', 10),
                key="dense_top_k"
            )

        with col2:
            sparse_top_k = st.number_input(
                "Sparse Top K",
                min_value=1,
                max_value=100,
                value=retrieval_config.get('sparse_top_k', 10),
                key="sparse_top_k"
            )

        with col3:
            final_top_k = st.number_input(
                "最终 Top K",
                min_value=1,
                max_value=50,
                value=retrieval_config.get('final_top_k', 5),
                key="final_top_k"
            )

        st.markdown("---")
        st.subheader("重排序配置")

        reranker_config = config.get('reranker', {})

        col1, col2 = st.columns(2)
        with col1:
            rerank_enabled = st.checkbox(
                "启用重排序",
                value=reranker_config.get('enabled', True),
                key="rerank_enabled"
            )
            rerank_provider = st.text_input(
                "提供商",
                value=reranker_config.get('provider', 'cohere'),
                disabled=not rerank_enabled,
                key="rerank_provider"
            )

        with col2:
            rerank_model = st.text_input(
                "模型",
                value=reranker_config.get('model', 'rerank-multilingual-v2.0'),
                disabled=not rerank_enabled,
                key="rerank_model"
            )
            rerank_top_k = st.number_input(
                "Rerank Top K",
                min_value=1,
                max_value=20,
                value=reranker_config.get('top_k', 3),
                disabled=not rerank_enabled,
                key="rerank_top_k"
            )

    # 原始配置（YAML编辑器）
    with tab5:
        st.subheader("原始配置文件")
        st.warning("⚠️ 直接编辑YAML可能导致配置错误，请谨慎操作")

        yaml_content = yaml.dump(config, default_flow_style=False, allow_unicode=True, sort_keys=False)
        edited_yaml = st.text_area(
            "config.yaml",
            value=yaml_content,
            height=400,
            key="raw_yaml"
        )

        if st.button("从YAML加载", key="load_yaml"):
            try:
                new_config = yaml.safe_load(edited_yaml)
                st.session_state.config = new_config
                st.success("✅ YAML配置已加载到编辑器")
                st.rerun()
            except yaml.YAMLError as e:
                st.error(f"❌ YAML格式错误: {str(e)}")

    # 保存按钮
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        if st.button("💾 保存配置", type="primary"):
            # 从表单收集配置
            new_config = {
                'llm': {
                    'provider': st.session_state.llm_provider,
                    'model': st.session_state.llm_model,
                    'api_base': st.session_state.llm_api_base,
                    'api_key': st.session_state.llm_api_key,
                    'temperature': st.session_state.llm_temperature,
                    'max_tokens': st.session_state.llm_max_tokens,
                },
                'embedding': {
                    'provider': st.session_state.emb_provider,
                    'model': st.session_state.emb_model,
                    'api_base': st.session_state.emb_api_base,
                    'api_key': st.session_state.emb_api_key,
                    'dimensions': st.session_state.emb_dimensions,
                },
                'vector_store': {
                    'provider': st.session_state.vs_provider,
                    'collection_name': st.session_state.vs_collection,
                    'persist_directory': st.session_state.vs_persist_dir,
                },
                'retrieval': {
                    'dense_top_k': st.session_state.dense_top_k,
                    'sparse_top_k': st.session_state.sparse_top_k,
                    'final_top_k': st.session_state.final_top_k,
                },
                'reranker': {
                    'enabled': st.session_state.rerank_enabled,
                    'provider': st.session_state.rerank_provider,
                    'model': st.session_state.rerank_model,
                    'top_k': st.session_state.rerank_top_k,
                }
            }

            # 保留其他配置项
            for key in config:
                if key not in new_config:
                    new_config[key] = config[key]

            if save_config_file(new_config):
                st.session_state.config = new_config
                st.success("✅ 配置已保存！请重启Dashboard使配置生效。")

    with col2:
        if st.button("🔄 重新加载"):
            st.session_state.config = load_config_file()
            st.success("✅ 配置已重新加载")
            st.rerun()

    with col3:
        st.info("💡 修改配置后需要重启Dashboard才能生效")


if __name__ == "__main__":
    render()
