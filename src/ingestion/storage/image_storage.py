"""
Image storage with SQLite-based indexing.

Provides image file storage and image_id→path mapping persistence.
"""
import hashlib
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class ImageStorage:
    """
    Image storage manager with SQLite indexing.

    Stores images in data/images/{collection}/ and maintains
    image_id→path mappings in SQLite database.
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        storage_root: Optional[str] = None
    ):
        """
        Initialize image storage.

        Args:
            db_path: Path to SQLite database file.
                     Defaults to data/db/image_index.db
            storage_root: Root directory for image files.
                         Defaults to data/images/
        """
        if db_path is None:
            db_path = "data/db/image_index.db"
        if storage_root is None:
            storage_root = "data/images"

        self.db_path = Path(db_path)
        self.storage_root = Path(storage_root)

        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.storage_root.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema and enable WAL mode."""
        conn = sqlite3.connect(str(self.db_path))
        try:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")

            # Create table
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

            # Create indexes
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
        Save image to storage and record mapping.

        Args:
            image_data: Raw image bytes
            collection: Collection name
            doc_hash: Document hash
            page_num: Page number (optional)
            extension: File extension (default: png)

        Returns:
            image_id: Generated image ID

        Raises:
            ValueError: If image_data is empty
        """
        if not image_data:
            raise ValueError("Image data cannot be empty")

        # Generate image_id from content hash
        image_id = hashlib.sha256(image_data).hexdigest()

        # Construct file path
        collection_dir = self.storage_root / collection
        collection_dir.mkdir(parents=True, exist_ok=True)

        file_path = collection_dir / f"{image_id}.{extension}"

        # Save image file
        with open(file_path, "wb") as f:
            f.write(image_data)

        # Record mapping in database
        conn = sqlite3.connect(str(self.db_path))
        try:
            now = datetime.now(timezone.utc).isoformat()

            # Use INSERT OR REPLACE to handle duplicates
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
        Get file path for an image_id.

        Args:
            image_id: Image ID

        Returns:
            File path if found, None otherwise
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
        Get all images in a collection.

        Args:
            collection: Collection name

        Returns:
            List of image records with fields:
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
        Get all images for a document.

        Args:
            doc_hash: Document hash

        Returns:
            List of image records (same format as get_images_by_collection)
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
        Delete image file and database record.

        Args:
            image_id: Image ID

        Returns:
            True if deleted, False if not found
        """
        # Get file path first
        file_path = self.get_image_path(image_id)
        if file_path is None:
            return False

        # Delete file if exists
        path = Path(file_path)
        if path.exists():
            path.unlink()

        # Delete database record
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
