"""
清理临时文件路径的旧记录。

此脚本用于清理数据库中使用临时文件路径的旧记录。
这些记录是在修复文件上传逻辑之前创建的。
"""

import sys
import os
from pathlib import Path

# 设置 UTF-8 编码（Windows 兼容）
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# 添加项目根目录到 Python 路径，并切换工作目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)  # 切换到项目根目录

from src.core.settings import load_settings
from src.libs.vector_store.chroma_store import ChromaStore
from src.ingestion.storage.image_storage import ImageStorage
from src.ingestion.storage.bm25_indexer import BM25Indexer
from src.libs.loader.file_integrity import SQLiteIntegrityChecker
from src.ingestion.document_manager import DocumentManager


def is_temp_path(path: str) -> bool:
    """检查路径是否为临时文件路径"""
    temp_indicators = [
        "\\Temp\\",
        "/tmp/",
        "\\AppData\\Local\\Temp\\",
        "/var/tmp/",
    ]
    return any(indicator in path for indicator in temp_indicators)


def cleanup_temp_records():
    """清理所有临时路径的记录"""
    print("=" * 60)
    print("清理临时文件路径记录")
    print("=" * 60)
    print()

    # 初始化组件
    print("正在初始化存储组件...")
    settings = load_settings()

    chroma_store = ChromaStore(
        collection_name=settings.vector_store.get("collection", "default"),
        persist_directory=settings.vector_store.get("persist_directory", "data/db/chroma")
    )

    image_storage = ImageStorage()
    bm25_indexer = BM25Indexer()
    file_integrity = SQLiteIntegrityChecker()

    document_manager = DocumentManager(
        chroma_store=chroma_store,
        bm25_indexer=bm25_indexer,
        image_storage=image_storage,
        file_integrity=file_integrity
    )

    # 获取所有文档
    print("正在扫描文档记录...")
    documents = document_manager.list_documents()
    print(f"找到 {len(documents)} 个文档记录")
    print()

    # 筛选临时路径的文档
    temp_docs = [doc for doc in documents if is_temp_path(doc.source_path)]

    if not temp_docs:
        print("[OK] 没有找到临时路径的记录，无需清理。")
        return

    print(f"发现 {len(temp_docs)} 个临时路径记录：")
    print("-" * 60)
    for idx, doc in enumerate(temp_docs, 1):
        print(f"{idx}. {doc.source_path}")
        print(f"   - Chunks: {doc.chunk_count}, Images: {doc.image_count}")
        print(f"   - 创建时间: {doc.created_at[:19]}")
    print("-" * 60)
    print()

    # 确认删除
    response = input(f"是否删除这 {len(temp_docs)} 个记录？(y/N): ").strip().lower()
    if response != 'y':
        print("[取消] 已取消清理操作。")
        return

    print()
    print("开始清理...")
    print()

    # 删除记录
    success_count = 0
    failed_count = 0

    for idx, doc in enumerate(temp_docs, 1):
        print(f"[{idx}/{len(temp_docs)}] 正在删除: {Path(doc.source_path).name}")

        result = document_manager.delete_document(doc.source_path)

        if result.success:
            success_count += 1
            print(f"  [成功] Chunks: {result.chunks_deleted}, "
                  f"Images: {result.images_deleted}, "
                  f"Integrity: {'是' if result.integrity_record_deleted else '否'}")
        else:
            failed_count += 1
            print(f"  [失败] {result.error}")

        print()

    # 总结
    print("=" * 60)
    print("清理完成")
    print("=" * 60)
    print(f"[成功] 删除: {success_count} 个记录")
    if failed_count > 0:
        print(f"[失败] 删除失败: {failed_count} 个记录")
    print()


if __name__ == "__main__":
    try:
        cleanup_temp_records()
    except KeyboardInterrupt:
        print("\n\n[取消] 操作已取消。")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[错误] 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
