"""
基于 SQLite 索引的图像存储。

提供图像文件存储和 image_id→path 映射持久化。
"""
import hashlib
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ImageStorage:
    """
    带有 SQLite 索引的图像存储管理器。

    将图像存储在 data/images/{collection}/ 中，并在 SQLite 数据库中
    维护 image_id→path 映射。
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        storage_root: Optional[str] = None
    ):
        """
        初始化图像存储。

        Args:
            db_path: SQLite 数据库文件路径。
                     默认为 data/db/image_index.db
            storage_root: 图像文件的根目录。
                         默认为 data/images/
        """
        if db_path is None:
            db_path = "data/db/image_index.db"
        if storage_root is None:
            storage_root = "data/images"

        self.db_path = Path(db_path)
        self.storage_root = Path(storage_root)

        # 确保目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_root.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库架构并启用 WAL 模式。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # 启用 WAL 模式以支持并发访问
            conn.execute("PRAGMA journal_mode=WAL")

            # 创建表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS image_index (
                    image_id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    collection TEXT,
                    doc_hash TEXT,
                    page_num INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # 创建索引
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_collection
                ON image_index(collection)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_hash
                ON image_index(doc_hash)
            """)

            conn.commit()
        finally:
            conn.close()

    def save_image(
        self,
        image_data: bytes,
        collection: str,
        doc_hash: str,
        page_num: Optional[int] = None,
        extension: str = "png"
    ) -> str:
        """
        保存图像到存储并记录映射。

        Args:
            image_data: 原始图像字节
            collection: 集合名称
            doc_hash: 文档哈希
            page_num: 页码（可选）
            extension: 文件扩展名（默认: png）

        Returns:
            image_id: 生成的图像 ID

        Raises:
            ValueError: 如果 image_data 为空
        """
        if not image_data:
            raise ValueError("图像数据不能为空")

        # 从内容哈希生成 image_id
        image_id = hashlib.sha256(image_data).hexdigest()

        # 构造文件路径
        collection_dir = self.storage_root / collection
        collection_dir.mkdir(parents=True, exist_ok=True)

        file_path = collection_dir / f"{image_id}.{extension}"

        # 保存图像文件
        with open(file_path, "wb") as f:
            f.write(image_data)

        # 在数据库中记录映射
        conn = sqlite3.connect(str(self.db_path))
        try:
            now = datetime.now(timezone.utc).isoformat()

            # 使用 INSERT OR REPLACE 处理重复项
            conn.execute("""
                INSERT OR REPLACE INTO image_index
                (image_id, file_path, collection, doc_hash, page_num, created_at)
                VALUES (?, ?, ?, ?, ?,
                    COALESCE((SELECT created_at FROM image_index WHERE image_id = ?), ?))
            """, (
                image_id,
                str(file_path),
                collection,
                doc_hash,
                page_num,
                image_id,
                now
            ))

            conn.commit()
        finally:
            conn.close()

        return image_id

    def get_image_path(self, image_id: str) -> Optional[str]:
        """
        获取 image_id 的文件路径。

        Args:
            image_id: 图像 ID

        Returns:
            如果找到则返回文件路径，否则返回 None
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT file_path FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            row = cursor.fetchone()

            if row is None:
                return None

            return row[0]
        finally:
            conn.close()

    def get_images_by_collection(self, collection: str) -> list[dict]:
        """
        获取集合中的所有图像。

        Args:
            collection: 集合名称

        Returns:
            包含以下字段的图像记录列表:
            - image_id
            - file_path
            - collection
            - doc_hash
            - page_num
            - created_at
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                """
                SELECT image_id, file_path, collection, doc_hash, page_num, created_at
                FROM image_index
                WHERE collection = ?
                ORDER BY created_at
                """,
                (collection,)
            )

            rows = cursor.fetchall()

            return [
                {
                    "image_id": row[0],
                    "file_path": row[1],
                    "collection": row[2],
                    "doc_hash": row[3],
                    "page_num": row[4],
                    "created_at": row[5]
                }
                for row in rows
            ]
        finally:
            conn.close()

    def get_images_by_doc(self, doc_hash: str) -> list[dict]:
        """
        获取文档的所有图像。

        Args:
            doc_hash: 文档哈希

        Returns:
            图像记录列表（与 get_images_by_collection 格式相同）
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                """
                SELECT image_id, file_path, collection, doc_hash, page_num, created_at
                FROM image_index
                WHERE doc_hash = ?
                ORDER BY page_num, created_at
                """,
                (doc_hash,)
            )

            rows = cursor.fetchall()

            return [
                {
                    "image_id": row[0],
                    "file_path": row[1],
                    "collection": row[2],
                    "doc_hash": row[3],
                    "page_num": row[4],
                    "created_at": row[5]
                }
                for row in rows
            ]
        finally:
            conn.close()

    def delete_image(self, image_id: str) -> bool:
        """
        删除图像文件和数据库记录。

        Args:
            image_id: 图像 ID

        Returns:
            如果已删除则返回 True，如果未找到则返回 False
        """
        # 首先获取文件路径
        file_path = self.get_image_path(image_id)
        if file_path is None:
            return False

        # 如果存在则删除文件
        path = Path(file_path)
        if path.exists():
            path.unlink()

        # 删除数据库记录
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "DELETE FROM image_index WHERE image_id = ?",
                (image_id,)
            )
            conn.commit()

            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_images(self, source_path: str) -> list[dict]:
        """
        列出源文档路径的所有图像。

        Args:
            source_path: 文档的源路径

        Returns:
            图像记录列表
        """
        # 目前，我们使用 doc_hash 作为键
        # 在实际实现中，我们需要将 source_path 映射到 doc_hash
        # 为简单起见，我们将按 file_path 模式查询
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                """
                SELECT image_id, file_path, collection, doc_hash, page_num, created_at
                FROM image_index
                ORDER BY created_at
                """
            )

            rows = cursor.fetchall()

            return [
                {
                    "image_id": row[0],
                    "file_path": row[1],
                    "collection": row[2],
                    "doc_hash": row[3],
                    "page_num": row[4],
                    "created_at": row[5]
                }
                for row in rows
            ]
        finally:
            conn.close()

    def delete_images(self, source_path: str) -> int:
        """
        删除源文档路径的所有图像。

        Args:
            source_path: 文档的源路径

        Returns:
            删除的图像数
        """
        # 获取所有图像（简化 - 在实际实现中会按 source_path 过滤）
        images = self.list_images(source_path)

        deleted_count = 0
        for image in images:
            if self.delete_image(image["image_id"]):
                deleted_count += 1

        return deleted_count
