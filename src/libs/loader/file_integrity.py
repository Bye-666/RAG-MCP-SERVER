"""
File integrity checker for tracking ingestion history.

Provides SHA256-based file integrity checking and ingestion status tracking.
"""
import hashlib
import json
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class FileIntegrityChecker(ABC):
    """Abstract interface for file integrity checking."""

    @abstractmethod
    def compute_sha256(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash as hex string (64 characters)

        Raises:
            FileNotFoundError: If file does not exist
        """
        pass

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file should be skipped based on its hash.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            True if file was successfully processed before, False otherwise
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
        Mark a file as successfully processed.

        Args:
            file_hash: SHA256 hash of the file
            file_path: Path to the file
            **metadata: Additional metadata (chunk_count, total_tokens, etc.)
        """
        pass

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        error_msg: str
    ) -> None:
        """
        Mark a file as failed to process.

        Args:
            file_hash: SHA256 hash of the file
            error_msg: Error message describing the failure
        """
        pass


class SQLiteIntegrityChecker(FileIntegrityChecker):
    """SQLite-based file integrity checker."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize SQLite integrity checker.

        Args:
            db_path: Path to SQLite database file.
                     Defaults to data/db/ingestion_history.db
        """
        if db_path is None:
            db_path = "data/db/ingestion_history.db"

        self.db_path = Path(db_path)

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema and enable WAL mode."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Enable WAL mode for concurrent writes
            conn.execute("PRAGMA journal_mode=WAL")

            # Create table
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

            # Create index on status for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status
                ON ingestion_history(status)
            """)

            conn.commit()
        finally:
            conn.close()

    def compute_sha256(self, file_path: str) -> str:
        """
        Compute SHA256 hash of a file.

        Args:
            file_path: Path to the file

        Returns:
            SHA256 hash as hex string (64 characters)

        Raises:
            FileNotFoundError: If file does not exist
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256_hash = hashlib.sha256()

        # Read file in chunks to handle large files
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    def should_skip(self, file_hash: str) -> bool:
        """
        Check if a file should be skipped based on its hash.

        Args:
            file_hash: SHA256 hash of the file

        Returns:
            True if file was successfully processed before, False otherwise
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
        Mark a file as successfully processed.

        Args:
            file_hash: SHA256 hash of the file
            file_path: Path to the file
            **metadata: Additional metadata (chunk_count, total_tokens, etc.)
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            now = datetime.now(timezone.utc).isoformat()
            metadata_json = json.dumps(metadata) if metadata else None

            # Use INSERT OR REPLACE to handle updates
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
        Mark a file as failed to process.

        Args:
            file_hash: SHA256 hash of the file
            error_msg: Error message describing the failure
        """
        conn = sqlite3.connect(str(self.db_path))
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Use INSERT OR REPLACE to handle updates
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
