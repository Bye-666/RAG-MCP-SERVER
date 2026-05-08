"""
Unit tests for file integrity checker.
"""
import os
import tempfile
import pytest
from pathlib import Path

from src.libs.loader.file_integrity import (
    FileIntegrityChecker,
    SQLiteIntegrityChecker,
)


class TestSQLiteIntegrityChecker:
    """Test SQLite-based file integrity checker."""

    @pytest.fixture
    def temp_db(self):
        """Create a temporary database file."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        yield db_path
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)
        # Also cleanup WAL and SHM files
        for ext in ["-wal", "-shm"]:
            wal_path = db_path + ext
            if os.path.exists(wal_path):
                os.unlink(wal_path)

    @pytest.fixture
    def temp_file(self):
        """Create a temporary file with known content."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("Hello, World!")
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    def test_compute_sha256_consistency(self, temp_file):
        """Test that computing SHA256 for the same file returns consistent results."""
        checker = SQLiteIntegrityChecker()
        hash1 = checker.compute_sha256(temp_file)
        hash2 = checker.compute_sha256(temp_file)
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

    def test_compute_sha256_different_files(self, temp_db):
        """Test that different files produce different hashes."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)

        # Create two different files
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f1:
            f1.write("Content A")
            file1 = f1.name

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f2:
            f2.write("Content B")
            file2 = f2.name

        try:
            hash1 = checker.compute_sha256(file1)
            hash2 = checker.compute_sha256(file2)
            assert hash1 != hash2
        finally:
            os.unlink(file1)
            os.unlink(file2)

    def test_compute_sha256_nonexistent_file(self, temp_db):
        """Test that computing hash for nonexistent file raises error."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        with pytest.raises(FileNotFoundError):
            checker.compute_sha256("/nonexistent/file.txt")

    def test_should_skip_new_file(self, temp_db, temp_file):
        """Test that new file should not be skipped."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker.compute_sha256(temp_file)
        assert not checker.should_skip(file_hash)

    def test_should_skip_after_success(self, temp_db, temp_file):
        """Test that file should be skipped after marking success."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker.compute_sha256(temp_file)

        # Initially should not skip
        assert not checker.should_skip(file_hash)

        # Mark as success
        checker.mark_success(file_hash, temp_file)

        # Now should skip
        assert checker.should_skip(file_hash)

    def test_should_not_skip_after_failure(self, temp_db, temp_file):
        """Test that file should not be skipped after marking failure."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker.compute_sha256(temp_file)

        # Mark as failed
        checker.mark_failed(file_hash, "Test error")

        # Should not skip (allow retry)
        assert not checker.should_skip(file_hash)

    def test_mark_success_with_metadata(self, temp_db, temp_file):
        """Test marking success with additional metadata."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker.compute_sha256(temp_file)

        # Mark success with metadata
        checker.mark_success(
            file_hash,
            temp_file,
            chunk_count=10,
            total_tokens=1000
        )

        # Should skip
        assert checker.should_skip(file_hash)

    def test_database_file_creation(self, temp_db):
        """Test that database file is created correctly."""
        # Remove if exists
        if os.path.exists(temp_db):
            os.unlink(temp_db)

        # Create checker (should create database)
        checker = SQLiteIntegrityChecker(db_path=temp_db)

        # Database file should exist
        assert os.path.exists(temp_db)

    def test_default_database_path(self):
        """Test that default database path is used correctly."""
        checker = SQLiteIntegrityChecker()
        expected_path = Path("data/db/ingestion_history.db")

        # Check that database is created at expected location
        assert os.path.exists(expected_path)

        # Cleanup
        if os.path.exists(expected_path):
            os.unlink(expected_path)
            # Also cleanup WAL and SHM files
            for ext in ["-wal", "-shm"]:
                wal_path = str(expected_path) + ext
                if os.path.exists(wal_path):
                    os.unlink(wal_path)

    def test_concurrent_writes(self, temp_db, temp_file):
        """Test that multiple writes work correctly (WAL mode)."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker.compute_sha256(temp_file)

        # Perform multiple writes
        for i in range(5):
            checker.mark_success(file_hash, temp_file, iteration=i)

        # Should still skip
        assert checker.should_skip(file_hash)

    def test_mark_failed_with_error_message(self, temp_db, temp_file):
        """Test marking failure with error message."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker.compute_sha256(temp_file)

        error_msg = "Failed to parse PDF"
        checker.mark_failed(file_hash, error_msg)

        # Should not skip (allow retry)
        assert not checker.should_skip(file_hash)

    def test_update_existing_success(self, temp_db, temp_file):
        """Test updating an existing success record."""
        checker = SQLiteIntegrityChecker(db_path=temp_db)
        file_hash = checker.compute_sha256(temp_file)

        # Mark success first time
        checker.mark_success(file_hash, temp_file, chunk_count=10)

        # Mark success again with different metadata
        checker.mark_success(file_hash, temp_file, chunk_count=20)

        # Should still skip
        assert checker.should_skip(file_hash)

    def test_abstract_interface(self):
        """Test that FileIntegrityChecker is an abstract interface."""
        # Should not be able to instantiate abstract class
        with pytest.raises(TypeError):
            FileIntegrityChecker()

    def test_hash_format(self, temp_file):
        """Test that hash is in correct format (64 hex characters)."""
        checker = SQLiteIntegrityChecker()
        file_hash = checker.compute_sha256(temp_file)

        # Should be 64 characters
        assert len(file_hash) == 64

        # Should be all hex characters
        assert all(c in "0123456789abcdef" for c in file_hash)

    def test_empty_file_hash(self, temp_db):
        """Test computing hash for empty file."""
        # Create empty file
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            empty_file = f.name

        try:
            checker = SQLiteIntegrityChecker(db_path=temp_db)
            file_hash = checker.compute_sha256(empty_file)

            # Should still produce valid hash
            assert len(file_hash) == 64

            # Known SHA256 of empty file
            assert file_hash == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        finally:
            os.unlink(empty_file)
