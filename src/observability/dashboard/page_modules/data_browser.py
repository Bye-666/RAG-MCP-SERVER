"""
数据浏览器页面。

浏览文档列表、Chunk 详情、图片预览。
"""

import streamlit as st
from pathlib import Path
from datetime import datetime

from src.core.settings import load_settings
from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.ingestion.document_manager import DocumentManager
from src.observability.dashboard.services.data_service import DataService


def _init_services():
    """初始化服务实例"""
    if "data_service" not in st.session_state:
        settings = load_settings()

        # 初始化存储组件
        chroma_store = ChromaStore(
            collection_name=settings.vector_store.get("collection_name", "default"),
            persist_directory=settings.vector_store.get("persist_directory", "data/db/chroma")
        )

        image_storage = ImageStorage()

        bm25_indexer = BM25Indexer()

        file_integrity = SQLiteIntegrityChecker()

        # 初始化文档管理器
        document_manager = DocumentManager(
            chroma_store=chroma_store,
            bm25_indexer=bm25_indexer,
            image_storage=image_storage,
            file_integrity=file_integrity
        )

        # 初始化数据服务
        st.session_state.data_service = DataService(
            chroma_store=chroma_store,
            image_storage=image_storage,
            document_manager=document_manager
        )

    return st.session_state.data_service


def _render_document_list(data_service: DataService):
    """渲染文档列表视图"""
    st.subheader("📄 文档列表")

    # 搜索框
    col1, col2 = st.columns([3, 1])
    with col1:
        search_keyword = st.text_input("🔍 搜索文档", placeholder="输入文件名关键词...")
    with col2:
        st.write("")  # 占位
        st.write("")  # 占位
        if st.button("🔄 刷新", use_container_width=True):
            st.rerun()

    # 获取文档列表
    if search_keyword:
        documents = data_service.search_documents(search_keyword)
    else:
        documents = data_service.list_documents()

    if not documents:
        st.info("📭 暂无文档数据。请先通过 Ingestion 页面摄取文档。")
        return None

    # 显示统计信息
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("文档总数", len(documents))
    with col2:
        total_chunks = sum(doc.chunk_count for doc in documents)
        st.metric("Chunk 总数", total_chunks)
    with col3:
        total_images = sum(doc.image_count for doc in documents)
        st.metric("图片总数", total_images)

    st.markdown("---")

    # 文档列表表格
    st.write("**点击文档查看详情：**")

    # 使用 expander 展示每个文档
    selected_doc = None
    for idx, doc in enumerate(documents):
        file_name = Path(doc.source_path).name

        # 创建可展开的文档项
        with st.expander(f"📄 {file_name}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                st.write(f"**路径：** `{doc.source_path}`")
                st.write(f"**文件哈希：** `{doc.file_hash[:16]}...`")
                st.write(f"**状态：** {doc.status}")

            with col2:
                st.write(f"**Chunk 数：** {doc.chunk_count}")
                st.write(f"**图片数：** {doc.image_count}")
                st.write(f"**创建时间：** {doc.created_at[:19]}")

            # 查看详情按钮
            if st.button(f"🔍 查看详情", key=f"view_{idx}"):
                selected_doc = doc.source_path

    return selected_doc


def _render_document_detail(data_service: DataService, source_path: str):
    """渲染文档详情视图"""
    st.subheader(f"📖 文档详情")

    # 返回按钮
    if st.button("⬅️ 返回列表"):
        if "selected_document" in st.session_state:
            del st.session_state.selected_document
        st.rerun()

    st.markdown("---")

    # 获取文档详情
    detail = data_service.get_document_detail(source_path)

    if not detail:
        st.error(f"❌ 未找到文档：{source_path}")
        return

    # 显示文档基本信息
    st.write(f"**源路径：** `{detail.source_path}`")
    st.write(f"**文件哈希：** `{detail.file_hash}`")
    st.write(f"**Chunk 数量：** {len(detail.chunks)}")
    st.write(f"**图片数量：** {len(detail.images)}")

    st.markdown("---")

    # Tabs: Chunks 和 Images
    tab1, tab2 = st.tabs(["📝 Chunks", "🖼️ 图片"])

    with tab1:
        _render_chunks(detail.chunks)

    with tab2:
        _render_images(data_service, detail.images)


def _render_chunks(chunks):
    """渲染 Chunk 列表"""
    if not chunks:
        st.info("📭 该文档没有 Chunk 数据")
        return

    st.write(f"**共 {len(chunks)} 个 Chunk：**")

    for idx, chunk in enumerate(chunks):
        chunk_id = chunk.get("id", "unknown")
        text = chunk.get("text", "")
        metadata = chunk.get("metadata", {})

        with st.expander(f"Chunk #{idx + 1} - `{chunk_id}`", expanded=False):
            # 显示文本内容
            st.text_area(
                "内容",
                value=text,
                height=150,
                key=f"chunk_text_{idx}",
                disabled=True
            )

            # 显示 metadata
            if metadata:
                st.write("**Metadata：**")
                # 格式化显示 metadata
                for key, value in metadata.items():
                    if key == "image_ids" and isinstance(value, list):
                        st.write(f"- **{key}:** {', '.join(value) if value else '无'}")
                    else:
                        st.write(f"- **{key}:** {value}")


def _render_images(data_service: DataService, images):
    """渲染图片列表"""
    if not images:
        st.info("📭 该文档没有图片数据")
        return

    st.write(f"**共 {len(images)} 张图片：**")

    # 使用列布局展示图片
    cols_per_row = 3
    for i in range(0, len(images), cols_per_row):
        cols = st.columns(cols_per_row)

        for j, col in enumerate(cols):
            idx = i + j
            if idx >= len(images):
                break

            image_info = images[idx]
            image_id = image_info.get("image_id", "unknown")
            file_path = image_info.get("file_path", "")

            with col:
                st.write(f"**Image #{idx + 1}**")
                st.caption(f"ID: `{image_id}`")

                # 尝试显示图片
                if file_path and Path(file_path).exists():
                    try:
                        st.image(str(file_path), use_container_width=True)
                    except Exception as e:
                        st.error(f"无法加载图片：{e}")
                else:
                    st.warning("图片文件不存在")


def render():
    """渲染数据浏览器页面"""
    st.title("📚 数据浏览器")
    st.markdown("浏览已摄取的文档、Chunk 和图片")
    st.markdown("---")

    # 初始化服务
    try:
        data_service = _init_services()
    except Exception as e:
        st.error(f"❌ 初始化服务失败：{e}")
        return

    # 检查是否有选中的文档
    if "selected_document" not in st.session_state:
        st.session_state.selected_document = None

    # 渲染文档列表或详情
    if st.session_state.selected_document:
        _render_document_detail(data_service, st.session_state.selected_document)
    else:
        selected = _render_document_list(data_service)
        if selected:
            st.session_state.selected_document = selected
            st.rerun()


if __name__ == "__main__":
    render()
