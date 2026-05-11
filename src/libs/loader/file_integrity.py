"""
用于跟踪摄取历史的文件完整性检查器。

提供基于 SHA256 的文件完整性检查和摄取状态跟踪。
"""
import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FileIntegrityChecker(ABC):
    """文件完整性检查的抽象接口。"""

    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """
        计算文件的 SHA256 哈希。

        参数:
            file_path: 文件路径

        返回:
            SHA256 哈希的十六进制字符串（64 个字符）

        异常:
            FileNotFoundError: 如果文件不存在
        """
        pass

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """
        根据文件哈希检查是否应跳过文件。

        参数:
            file_hash: 文件的 SHA256 哈希

        返回:
            如果文件之前已成功处理则为 True，否则为 False
        """
        pass

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        **metadata
    ) -> None:
        """
        将文件标记为已成功处理。

        参数:
            file_hash: 文件的 SHA256 哈希
            file_path: 文件路径
            **metadata: 其他元数据（chunk_count、total_tokens 等）
        """
        pass

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str
    ) -> None:
        """
        将文件标记为处理失败。

        参数:
            file_hash: 文件的 SHA256 哈希
            error_msg: 描述失败的错误消息
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """基于 SQLite 的文件完整性检查器。"""

    def __init__(self, db_path: Optional[str] = None):
        """
        初始化 SQLite 完整性检查器。

        参数:
            db_path: SQLite 数据库文件路径。
                     默认为 data/db/ingestion_history.db
        """
        if db_path is None:
            db_path = "data/db/ingestion_history.db"

        self.db_path = Path(db_path)

        # 确保父目录存在
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库架构并启用 WAL 模式。"""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # 启用 WAL 模式以支持并发写入
            conn.execute("PRAGMA journal_mode=WAL")

            # 创建表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT,
                    status TEXT NOT NULL,
                    error_msg TEXT,
                    metadata TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # 在 status 上创建索引以加快查询
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON ingestion_history(status)
            """)

            conn.commit()
        finally:
            conn.close()

    def compute_sha256(self, file_path: str) -> str:
        """
        计算文件的 SHA256 哈希。

        参数:
            file_path: 文件路径

        返回:
            SHA256 哈希的十六进制字符串（64 个字符）

        异常:
            FileNotFoundError: 如果文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件未找到: {file_path}")

        sha256_hash = hashlib.sha256()

        # 分块读取文件以处理大文件
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """
        根据文件哈希检查是否应跳过文件。

        参数:
            file_hash: 文件的 SHA256 哈希

        返回:
            如果文件之前已成功处理则为 True，否则为 False
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            row = cursor.fetchone()

            if row is None:
                return False

            status = row[0]
            return status == "success"
        finally:
            conn.close()

    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        **metadata
    ) -> None:
        """
        将文件标记为已成功处理。

        参数:
            file_hash: 文件的 SHA256 哈希
            file_path: 文件路径
            **metadata: 其他元数据（chunk_count、total_tokens 等）
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            now = datetime.now(timezone.utc).isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            # 使用 INSERT OR REPLACE 处理更新
            conn.execute("""
                INSERT OR REPLACE INTO ingestion_history
                (file_hash, file_path, status, error_msg, metadata, created_at, updated_at)
                VALUES (?, ?, 'success', NULL, ?,
                    COALESCE((SELECT created_at FROM ingestion_history WHERE file_hash = ?), ?),
                    ?)
            """, (file_hash, file_path, metadata_json, file_hash, now, now))

            conn.commit()
        finally:
            conn.close()

    def mark_failed(
        self,
        file_hash: str,
        error_msg: str
    ) -> None:
        """
        将文件标记为处理失败。

        参数:
            file_hash: 文件的 SHA256 哈希
            error_msg: 描述失败的错误消息
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            now = datetime.now(timezone.utc).isoformat()

            # 使用 INSERT OR REPLACE 处理更新
            conn.execute("""
                INSERT OR REPLACE INTO ingestion_history
                (file_hash, file_path, status, error_msg, metadata, created_at, updated_at)
                VALUES (?, NULL, 'failed', ?, NULL,
                    COALESCE((SELECT created_at FROM ingestion_history WHERE file_hash = ?), ?),
                    ?)
            """, (file_hash, error_msg, file_hash, now, now))

            conn.commit()
        finally:
            conn.close()

    def remove_record(self, file_hash: str) -> bool:
        """
        从摄取历史中删除记录。

        参数:
            file_hash: 文件的 SHA256 哈希

        返回:
            如果记录已删除则为 True，如果未找到则为 False
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            cursor = conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            conn.close()

    def list_processed(self, status: Optional[str] = None):
        """
        列出所有已处理的文件。

        参数:
            status: 可选的状态过滤器（'success' 或 'failed'）

        返回:
            包含 file_hash、file_path、status、metadata、created_at、updated_at 的字典列表
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            if status:
                cursor = conn.execute(
                    "SELECT file_hash, file_path, status, metadata, created_at, updated_at "
                    "FROM ingestion_history WHERE status = ? ORDER BY updated_at DESC",
                    (status,)
                )
            else:
                cursor = conn.execute(
                    "SELECT file_hash, file_path, status, metadata, created_at, updated_at "
                    "FROM ingestion_history ORDER BY updated_at DESC"
                )

            results = []
            for row in cursor.fetchall():
                metadata = json.loads(row[3]) if row[3] else {}
                results.append({
                    "file_hash": row[0],
                    "file_path": row[1],
                    "status": row[2],
                    "metadata": metadata,
                    "created_at": row[4],
                    "updated_at": row[5]
                })

            return results
        finally:
            conn.close()
