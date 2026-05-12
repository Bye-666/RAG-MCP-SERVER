"""
Ingestion 管理页面。

文件上传、摄取触发、进度展示、文档删除。
"""

import streamlit as st
from pathlib import Path
from typing import Optional, Callable

from src.core.settings import load_settings
from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.ingestion.document_manager import DocumentManager
from src.observability.dashboard.services.data_service import DataService
from src.ingestion.pipeline import IngestionPipeline, PipelineConfig
from src.libs.loader.pdf_loader import PdfLoader
from src.ingestion.chunking.document_chunker import DocumentChunker
from src.ingestion.transform.chunk_refiner import ChunkRefiner
from src.ingestion.transform.metadata_enricher import MetadataEnricher
from src.ingestion.transform.image_captioner import ImageCaptioner
from src.ingestion.embedding.dense_encoder import DenseEncoder
from src.ingestion.embedding.sparse_encoder import SparseEncoder
from src.ingestion.embedding.batch_processor import BatchProcessor
from src.ingestion.storage.vector_upserter import VectorUpserter
from src.libs.llm.llm_factory import LLMFactory
from src.libs.embedding.embedding_factory import EmbeddingFactory
from src.libs.splitter.splitter_factory import SplitterFactory
from src.libs.vector_store.vector_store_factory import VectorStoreFactory


def _init_services():
    """初始化服务实例"""
    if "ingestion_services" not in st.session_state:
        settings = load_settings()

        # 初始化存储组件
        chroma_store = ChromaStore(
            collection_name=settings.vector_store.get("collection_name", "default"),
            persist_directory=settings.vector_store.get("persist_directory", "data/db/chroma")
        )

        image_storage = ImageStorage()
        bm25_indexer = BM25Indexer()

        # 加载现有的 BM25 索引（如果存在）
        try:
            bm25_indexer.load()
        except FileNotFoundError:
            pass  # 索引文件不存在，使用空索引

        file_integrity = SQLiteIntegrityChecker()

        # 初始化文档管理器
        document_manager = DocumentManager(
            chroma_store=chroma_store,
            bm25_indexer=bm25_indexer,
            image_storage=image_storage,
            file_integrity=file_integrity
        )

        # 初始化数据服务
        data_service = DataService(
            chroma_store=chroma_store,
            image_storage=image_storage,
            document_manager=document_manager
        )

        st.session_state.ingestion_services = {
            "settings": settings,
            "chroma_store": chroma_store,
            "image_storage": image_storage,
            "bm25_indexer": bm25_indexer,
            "file_integrity": file_integrity,
            "document_manager": document_manager,
            "data_service": data_service
        }

    return st.session_state.ingestion_services


def _create_pipeline(services: dict, on_progress: Optional[Callable] = None) -> IngestionPipeline:
    """创建 Ingestion Pipeline"""
    settings = services["settings"]

    # 创建组件（使用工厂类）
    # 将 Settings dataclass 转换为字典格式
    if hasattr(settings, '__dict__'):
        # Handle splitter's nested structure: settings.splitter = {'splitter': {...}}
        # We need to flatten it to {'splitter': {...}} for the factory
        splitter_config = settings.splitter
        if 'splitter' in splitter_config:
            # Already nested, use as-is
            splitter_dict = splitter_config
        else:
            # Not nested, wrap it
            splitter_dict = {'splitter': splitter_config}

        settings_dict = {
            'llm': settings.llm,
            'embedding': settings.embedding,
            'vector_store': settings.vector_store,
            'retrieval': settings.retrieval,
            'observability': settings.observability,
            'splitter': splitter_dict.get('splitter', splitter_config)  # Extract inner dict
        }
    else:
        settings_dict = settings

    llm = LLMFactory.create(settings_dict)
    embedding = EmbeddingFactory.create(settings_dict)
    splitter = SplitterFactory.create(settings_dict)
    vector_store = VectorStoreFactory.create(settings_dict)
    # vision_llm 暂时跳过，因为没有统一的 VisionLLMFactory
    vision_llm = None  # TODO: 实现 VisionLLMFactory

    # 创建 Pipeline 组件
    loader = PdfLoader()
    chunker = DocumentChunker(settings)

    transforms = [
        ChunkRefiner(settings, llm=llm),
        MetadataEnricher(settings, llm=llm),
        ImageCaptioner(settings, vision_llm=vision_llm)
    ]

    dense_encoder = DenseEncoder(embedding_model=embedding)
    sparse_encoder = SparseEncoder()

    batch_processor = BatchProcessor(
        dense_encoder=dense_encoder,
        sparse_encoder=sparse_encoder
    )

    vector_upserter = VectorUpserter(
        vector_store=vector_store
    )

    # 创建 Pipeline
    pipeline = IngestionPipeline(
        integrity_checker=services["file_integrity"],
        loader=loader,
        chunker=chunker,
        transforms=transforms,
        batch_processor=batch_processor,
        vector_upserter=vector_upserter,
        image_storage=services["image_storage"],
        on_progress=on_progress
    )

    return pipeline


def _render_file_upload(services: dict):
    """渲染文件上传区域"""
    st.subheader("📤 上传文档")

    # 文件上传器
    uploaded_file = st.file_uploader(
        "选择 PDF 文件",
        type=["pdf"],
        help="支持 PDF 格式文档"
    )

    # 集合选择（暂时使用默认集合）
    collection = st.text_input(
        "集合名称",
        value="default",
        help="文档将被存储到此集合中"
    )

    # 配置选项
    col1, col2 = st.columns(2)
    with col1:
        force_reprocess = st.checkbox(
            "强制重新处理",
            value=False,
            help="即使文档已处理过，也重新摄取"
        )
    with col2:
        enable_transforms = st.checkbox(
            "启用转换增强",
            value=True,
            help="启用 Chunk 精炼、元数据增强、图片描述等"
        )

    # 摄取按钮
    if st.button("🚀 开始摄取", type="primary", disabled=uploaded_file is None):
        if uploaded_file:
            _process_file(services, uploaded_file, collection, force_reprocess, enable_transforms)


def _process_file(
    services: dict,
    uploaded_file,
    collection: str,
    force_reprocess: bool,
    enable_transforms: bool
):
    """处理上传的文件"""
    # 从配置获取上传目录
    settings = services["settings"]
    upload_dir = Path(settings.storage.get("upload_directory", "./data/uploads"))
    upload_dir.mkdir(parents=True, exist_ok=True)

    # 保存文件到持久化目录，使用原始文件名
    original_filename = uploaded_file.name
    file_path = upload_dir / original_filename

    # 如果文件已存在，添加时间戳避免冲突
    if file_path.exists():
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = file_path.stem
        suffix = file_path.suffix
        file_path = upload_dir / f"{stem}_{timestamp}{suffix}"

    # 写入文件
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    try:
        # 显示进度区域
        st.markdown("---")
        st.subheader("📊 摄取进度")

        # 进度条容器
        progress_container = st.container()
        status_container = st.container()

        # 进度状态
        progress_state = {
            "current_stage": "",
            "current": 0,
            "total": 0
        }

        # 进度回调
        def on_progress(stage_name: str, current: int, total: int):
            progress_state["current_stage"] = stage_name
            progress_state["current"] = current
            progress_state["total"] = total

            # 更新进度显示
            with progress_container:
                stage_names = {
                    "integrity_check": "完整性检查",
                    "load": "加载文档",
                    "split": "切分 Chunk",
                    "transform": "转换增强",
                    "encode": "向量编码",
                    "upsert": "存储数据"
                }
                stage_display = stage_names.get(stage_name, stage_name)

                if total > 0:
                    progress = current / total
                    st.progress(progress, text=f"{stage_display}: {current}/{total}")
                else:
                    st.info(f"正在执行: {stage_display}")

        # 创建 Pipeline
        with status_container:
            with st.spinner("初始化 Pipeline..."):
                pipeline = _create_pipeline(services, on_progress=on_progress)

        # 配置
        config = PipelineConfig(
            force_reprocess=force_reprocess,
            enable_transforms=enable_transforms,
            collection=collection
        )

        # 创建追踪上下文
        from src.core.trace.trace_context import TraceContext
        from src.core.trace.trace_collector import TraceCollector

        trace = TraceContext(trace_type="ingestion")
        trace.metadata["file_path"] = str(file_path)
        trace.metadata["collection"] = collection
        trace.metadata["source"] = "dashboard"

        trace_collector = TraceCollector()

        # 执行摄取（使用持久化路径）
        with status_container:
            with st.spinner("正在摄取文档..."):
                result = pipeline.ingest_file(str(file_path), config=config, trace=trace)

        # 完成并收集追踪
        trace.finish()
        trace_collector.collect(trace)

        # 显示结果
        st.markdown("---")
        if result.get("skipped"):
            st.warning(f"⏭️ 文档已跳过（已处理过）")
            st.info(f"文件哈希: `{result['file_hash']}`")
        elif result.get("error"):
            st.error(f"❌ 摄取失败: {result['error']}")
            # 摄取失败时删除文件
            file_path.unlink(missing_ok=True)
        else:
            st.success(f"✅ 摄取成功！")
            st.info(f"📁 文件已保存至: `{file_path}`")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Chunk 数量", result.get("chunk_count", 0))
            with col2:
                st.metric("图片数量", result.get("image_count", 0))
            with col3:
                st.metric("文件哈希", f"{result['file_hash'][:8]}...")

    except Exception as e:
        st.error(f"❌ 处理失败: {str(e)}")
        # 处理失败时删除文件
        file_path.unlink(missing_ok=True)


def _render_document_list(services: dict):
    """渲染文档列表与删除功能"""
    st.subheader("📚 已摄取文档")

    data_service = services["data_service"]
    document_manager = services["document_manager"]

    # 获取文档列表
    documents = data_service.list_documents()

    if not documents:
        st.info("📭 暂无文档。请先上传文档进行摄取。")
        return

    # 显示统计
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

    # 文档列表
    for idx, doc in enumerate(documents):
        file_name = Path(doc.source_path).name

        col1, col2, col3 = st.columns([3, 1, 1])

        with col1:
            st.write(f"**{file_name}**")
            st.caption(f"Chunks: {doc.chunk_count} | Images: {doc.image_count} | {doc.created_at[:19]}")

        with col2:
            if st.button("🔍 查看", key=f"view_{idx}"):
                st.session_state.selected_document = doc.source_path
                st.info("💡 提示：请在侧边栏切换到「📚 数据浏览」页面查看文档详情")

        with col3:
            if st.button("🗑️ 删除", key=f"delete_{idx}", type="secondary"):
                st.session_state[f"confirm_delete_{idx}"] = True
                st.rerun()

        # 确认删除对话框
        if st.session_state.get(f"confirm_delete_{idx}", False):
            with st.container():
                st.warning(f"⚠️ 确认删除文档: {file_name}?")
                col_yes, col_no = st.columns(2)

                with col_yes:
                    if st.button("✅ 确认", key=f"confirm_yes_{idx}"):
                        # 执行删除
                        with st.spinner("正在删除..."):
                            result = document_manager.delete_document(doc.source_path)

                        if result.success:
                            st.success(f"✅ 已删除: {result.chunks_deleted} chunks, {result.images_deleted} images")
                            del st.session_state[f"confirm_delete_{idx}"]
                            st.rerun()
                        else:
                            st.error(f"❌ 删除失败: {result.error}")

                with col_no:
                    if st.button("❌ 取消", key=f"confirm_no_{idx}"):
                        del st.session_state[f"confirm_delete_{idx}"]
                        st.rerun()

        st.markdown("---")


def render():
    """渲染 Ingestion 管理页面"""
    st.title("📥 Ingestion 管理")
    st.markdown("上传文档、触发摄取、管理已有文档")
    st.markdown("---")

    # 初始化服务
    try:
        services = _init_services()
    except Exception as e:
        st.error(f"❌ 初始化服务失败: {e}")
        return

    # 渲染上传区域
    _render_file_upload(services)

    st.markdown("---")

    # 渲染文档列表
    _render_document_list(services)


if __name__ == "__main__":
    render()
