"""
Unit tests for ImageStorage.
"""
import hashlib
import tempfile
from pathlib import Path

import pytest

from src.ingestion.storage.image_storage import ImageStorage


@pytest.fixture
def temp_storage():
    """Create temporary storage for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_image_index.db"
        storage_root = Path(tmpdir) / "images"

        storage = ImageStorage(
            db_path=str(db_path),
            storage_root=str(storage_root)
        )

        yield storage


def test_save_image_basic(temp_storage):
    """Test basic image saving."""
    image_data = b"fake image data"

    image_id = temp_storage.save_image(
        image_data=image_data,
        collection="test_collection",
        doc_hash="doc123",
        page_num=1
    )

    # Verify image_id is SHA256 hash
    expected_id = hashlib.sha256(image_data).hexdigest()
    assert image_id == expected_id

    # Verify file exists
    file_path = temp_storage.get_image_path(image_id)
    assert file_path is not None
    assert Path(file_path).exists()

    # Verify file content
    with open(file_path, "rb") as f:
        assert f.read() == image_data


def test_save_image_idempotent(temp_storage):
    """Test that saving the same image twice produces the same ID."""
    image_data = b"duplicate image"

    id1 = temp_storage.save_image(
        image_data=image_data,
        collection="test_collection",
        doc_hash="doc123"
    )

    id2 = temp_storage.save_image(
        image_data=image_data,
        collection="test_collection",
        doc_hash="doc123"
    )

    assert id1 == id2

    # Verify only one file exists
    file_path = temp_storage.get_image_path(id1)
    assert file_path is not None
    assert Path(file_path).exists()


def test_save_image_different_content(temp_storage):
    """Test that different images produce different IDs."""
    image1 = b"image one"
    image2 = b"image two"

    id1 = temp_storage.save_image(
        image_data=image1,
        collection="test_collection",
        doc_hash="doc123"
    )

    id2 = temp_storage.save_image(
        image_data=image2,
        collection="test_collection",
        doc_hash="doc123"
    )

    assert id1 != id2


def test_save_image_empty_data(temp_storage):
    """Test that empty image data raises ValueError."""
    with pytest.raises(ValueError, match="Image data cannot be empty"):
        temp_storage.save_image(
            image_data=b"",
            collection="test_collection",
            doc_hash="doc123"
        )


def test_save_image_custom_extension(temp_storage):
    """Test saving image with custom extension."""
    image_data = b"jpeg image"

    image_id = temp_storage.save_image(
        image_data=image_data,
        collection="test_collection",
        doc_hash="doc123",
        extension="jpg"
    )

    file_path = temp_storage.get_image_path(image_id)
    assert file_path.endswith(".jpg")


def test_get_image_path_not_found(temp_storage):
    """Test getting path for non-existent image."""
    result = temp_storage.get_image_path("nonexistent_id")
    assert result is None


def test_get_images_by_collection(temp_storage):
    """Test retrieving images by collection."""
    # Save multiple images
    id1 = temp_storage.save_image(
        image_data=b"image1",
        collection="collection_a",
        doc_hash="doc1",
        page_num=1
    )

    id2 = temp_storage.save_image(
        image_data=b"image2",
        collection="collection_a",
        doc_hash="doc2",
        page_num=2
    )

    id3 = temp_storage.save_image(
        image_data=b"image3",
        collection="collection_b",
        doc_hash="doc3"
    )

    # Query collection_a
    results = temp_storage.get_images_by_collection("collection_a")

    assert len(results) == 2
    assert {r["image_id"] for r in results} == {id1, id2}

    # Verify record structure
    for record in results:
        assert "image_id" in record
        assert "file_path" in record
        assert "collection" in record
        assert "doc_hash" in record
        assert "page_num" in record
        assert "created_at" in record
        assert record["collection"] == "collection_a"


def test_get_images_by_collection_empty(temp_storage):
    """Test querying empty collection."""
    results = temp_storage.get_images_by_collection("nonexistent_collection")
    assert results == []


def test_get_images_by_doc(temp_storage):
    """Test retrieving images by document hash."""
    # Save multiple images for same document
    id1 = temp_storage.save_image(
        image_data=b"page1",
        collection="test_collection",
        doc_hash="doc123",
        page_num=1
    )

    id2 = temp_storage.save_image(
        image_data=b"page2",
        collection="test_collection",
        doc_hash="doc123",
        page_num=2
    )

    id3 = temp_storage.save_image(
        image_data=b"other_doc",
        collection="test_collection",
        doc_hash="doc456"
    )

    # Query doc123
    results = temp_storage.get_images_by_doc("doc123")

    assert len(results) == 2
    assert {r["image_id"] for r in results} == {id1, id2}

    # Verify ordering by page_num
    assert results[0]["page_num"] == 1
    assert results[1]["page_num"] == 2


def test_get_images_by_doc_empty(temp_storage):
    """Test querying non-existent document."""
    results = temp_storage.get_images_by_doc("nonexistent_doc")
    assert results == []


def test_delete_image(temp_storage):
    """Test deleting an image."""
    image_data = b"to be deleted"

    image_id = temp_storage.save_image(
        image_data=image_data,
        collection="test_collection",
        doc_hash="doc123"
    )

    # Verify image exists
    file_path = temp_storage.get_image_path(image_id)
    assert file_path is not None
    assert Path(file_path).exists()

    # Delete image
    result = temp_storage.delete_image(image_id)
    assert result is True

    # Verify image is gone
    assert not Path(file_path).exists()
    assert temp_storage.get_image_path(image_id) is None


def test_delete_image_not_found(temp_storage):
    """Test deleting non-existent image."""
    result = temp_storage.delete_image("nonexistent_id")
    assert result is False


def test_save_image_without_page_num(temp_storage):
    """Test saving image without page number."""
    image_data = b"no page num"

    image_id = temp_storage.save_image(
        image_data=image_data,
        collection="test_collection",
        doc_hash="doc123"
    )

    # Verify record
    results = temp_storage.get_images_by_doc("doc123")
    assert len(results) == 1
    assert results[0]["page_num"] is None


def test_multiple_collections_isolation(temp_storage):
    """Test that collections are properly isolated."""
    # Save images to different collections
    id1 = temp_storage.save_image(
        image_data=b"collection1_image",
        collection="collection1",
        doc_hash="doc1"
    )

    id2 = temp_storage.save_image(
        image_data=b"collection2_image",
        collection="collection2",
        doc_hash="doc2"
    )

    # Verify isolation
    coll1_images = temp_storage.get_images_by_collection("collection1")
    coll2_images = temp_storage.get_images_by_collection("collection2")

    assert len(coll1_images) == 1
    assert len(coll2_images) == 1
    assert coll1_images[0]["image_id"] == id1
    assert coll2_images[0]["image_id"] == id2


def test_storage_directory_creation(temp_storage):
    """Test that storage directories are created automatically."""
    # Save image to new collection
    image_id = temp_storage.save_image(
        image_data=b"test",
        collection="new_collection",
        doc_hash="doc123"
    )

    # Verify collection directory exists
    collection_dir = temp_storage.storage_root / "new_collection"
    assert collection_dir.exists()
    assert collection_dir.is_dir()


def test_database_persistence(temp_storage):
    """Test that database records persist across instances."""
    image_data = b"persistent image"

    # Save image
    image_id = temp_storage.save_image(
        image_data=image_data,
        collection="test_collection",
        doc_hash="doc123"
    )

    # Create new storage instance with same paths
    new_storage = ImageStorage(
        db_path=str(temp_storage.db_path),
        storage_root=str(temp_storage.storage_root)
    )

    # Verify image is still accessible
    file_path = new_storage.get_image_path(image_id)
    assert file_path is not None
    assert Path(file_path).exists()

    results = new_storage.get_images_by_collection("test_collection")
    assert len(results) == 1
    assert results[0]["image_id"] == image_id
